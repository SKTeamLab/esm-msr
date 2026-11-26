# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Two-stage Fast Double Mutant Selection engine for ChimeraX-ESM-MSR.

Usable both as an importable library and as a standalone CLI script:
  Stage 1: Fast WT-pass single-mutant scan (`--mode singles --skip_reverse`)
  Select : Exact O(N log S) top-N max-sum double-mutant selection (scales smoothly to N=1,000,000)
  Stage 2: Dual-view WT+MT rescoring of the top-N doubles (`--input_csv ... --skip_additive --calculate_distances`)
  Merge  : Re-rank Stage-2 doubles by combined_pred, compute rank_change, and emit a unified CSV
           containing both Stage-1 singles and Stage-2 re-ranked doubles.
"""

from __future__ import annotations

import argparse
from array import array
import heapq
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_MUT_SINGLE_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)([A-Za-z])$")

PosKey = Tuple[int, str]

_UNIFIED_COL_ORDER = (
    "pdb_file",
    "code",
    "chain",
    "mut_type_renumbered",
    "mut_type_pdb",
    "wt_lora_pred",
    "mt_lora_pred",
    "combined_pred",
    "epi_pred",
    "combined_dddg_pred",
    "mt_lora_dddg_pred",
    "dist",
    "stage1_additive_pred",
    "additive_rank",
    "full_rank",
    "rank_change",
)


def _parse_single_mut_token(mut_str: str) -> Optional[Tuple[str, int, str, str]]:
    """Parse a single PDB mutation token into (wt, res_num, icode, mut)."""
    if not mut_str or not isinstance(mut_str, str):
        return None
    m = _MUT_SINGLE_RE.match(mut_str.strip())
    if not m:
        return None
    wt, num_s, ic, mt = m.groups()
    return wt.upper(), int(num_s), ic.strip().upper(), mt.upper()


def compute_pdb_heavy_atom_distances(
    pdb_file: str,
    chain: str,
) -> Dict[Tuple[PosKey, PosKey], float]:
    """Compute minimum heavy-atom pairwise distances (Angstroms) between residues in a PDB chain.

    Pure stdlib + numpy implementation (no external bioinformatics dependencies).
    Returns a dict keyed by canonical ordered (pos_key_a, pos_key_b) with pos_key_a < pos_key_b.
    """
    if not pdb_file or not os.path.isfile(pdb_file):
        return {}

    target_chain = (chain or "").strip()
    res_coords: Dict[PosKey, List[Tuple[float, float, float]]] = {}

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 54:
                continue
            ch = line[21].strip()
            if target_chain and ch != target_chain:
                continue
            altloc = line[16].strip()
            if altloc and altloc not in ("A", "1"):
                continue
            atom_name = line[12:16].strip()
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if elem == "H" or (not elem and atom_name.startswith("H")):
                continue
            try:
                res_num = int(line[22:26].strip())
                icode = line[26].strip().upper()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            res_coords.setdefault((res_num, icode), []).append((x, y, z))

    pos_keys = sorted(res_coords.keys())
    n_pos = len(pos_keys)
    if n_pos < 2:
        return {}

    coord_arrays = [np.asarray(res_coords[k], dtype=np.float32) for k in pos_keys]
    # Fast centroid + radius bounding filter to skip distant residue pairs quickly.
    centroids = np.stack([arr.mean(axis=0) for arr in coord_arrays], axis=0)
    radii = np.array(
        [float(np.linalg.norm(arr - c, axis=1).max()) if len(arr) > 1 else 0.0 for arr, c in zip(coord_arrays, centroids)],
        dtype=np.float32,
    )

    dists: Dict[Tuple[PosKey, PosKey], float] = {}
    for a in range(n_pos):
        arr_a = coord_arrays[a]
        ka = pos_keys[a]
        for b in range(a + 1, n_pos):
            arr_b = coord_arrays[b]
            diff = arr_a[:, None, :] - arr_b[None, :, :]
            min_d = float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))
            dists[(ka, pos_keys[b])] = min_d
    return dists


def compute_allowed_pos_pairs_within_distance(
    pdb_file: str,
    chain: str,
    distance_threshold: float,
) -> Optional[Set[Tuple[PosKey, PosKey]]]:
    """Return canonical (pos_a, pos_b) residue pairs with min heavy-atom distance <= distance_threshold."""
    if distance_threshold is None or distance_threshold <= 0:
        return None
    if not pdb_file or not os.path.isfile(pdb_file):
        return None

    target_chain = (chain or "").strip()
    res_coords: Dict[PosKey, List[Tuple[float, float, float]]] = {}

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  ") or len(line) < 54:
                continue
            ch = line[21].strip()
            if target_chain and ch != target_chain:
                continue
            altloc = line[16].strip()
            if altloc and altloc not in ("A", "1"):
                continue
            atom_name = line[12:16].strip()
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if elem == "H" or (not elem and atom_name.startswith("H")):
                continue
            try:
                res_num = int(line[22:26].strip())
                icode = line[26].strip().upper()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            res_coords.setdefault((res_num, icode), []).append((x, y, z))

    pos_keys = sorted(res_coords.keys())
    n_pos = len(pos_keys)
    if n_pos < 2:
        return set()

    coord_arrays = [np.asarray(res_coords[k], dtype=np.float32) for k in pos_keys]
    centroids = np.stack([arr.mean(axis=0) for arr in coord_arrays], axis=0)
    radii = np.array(
        [float(np.linalg.norm(arr - c, axis=1).max()) if len(arr) > 1 else 0.0 for arr, c in zip(coord_arrays, centroids)],
        dtype=np.float32,
    )
    # Vectorized centroid distance matrix for fast triangle-inequality pruning.
    cdiff = centroids[:, None, :] - centroids[None, :, :]
    cdist = np.sqrt(np.sum(cdiff * cdiff, axis=2))
    thresh = float(distance_threshold)
    thresh_sq = thresh * thresh

    allowed: Set[Tuple[PosKey, PosKey]] = set()
    for a in range(n_pos):
        arr_a = coord_arrays[a]
        ra = radii[a]
        ka = pos_keys[a]
        for b in range(a + 1, n_pos):
            if float(cdist[a, b]) - ra - radii[b] > thresh:
                continue
            arr_b = coord_arrays[b]
            diff = arr_a[:, None, :] - arr_b[None, :, :]
            if float(np.min(np.sum(diff * diff, axis=2))) <= thresh_sq:
                allowed.add((ka, pos_keys[b]))
    return allowed


def select_top_n_additive_doubles(
    singles_df: pd.DataFrame,
    top_n: int = 1000,
    score_col: str = "wt_lora_pred",
    exclude_wt_cys: bool = False,
    exclude_wt_pro: bool = False,
    exclude_mut_cys: bool = False,
    exclude_mut_pro: bool = False,
    min_seq_sep: int = 1,
    allowed_pos_pairs: Optional[Set[Tuple[PosKey, PosKey]]] = None,
) -> pd.DataFrame:
    """Select the exact top-N double mutants maximizing additive score_i + score_j.

    Uses a duplicate-free, visited-set-free monotone tree best-first max-heap over
    upper-triangular pairs (i, j) with 0 <= i < j < S:
      - Root: (0, 1)
      - Right child: (i, j + 1) when j + 1 < S
      - Down-diagonal child: (i + 1, i + 2) ONLY when j == i + 1 and i + 2 < S
    Every pair has a unique parent with sum >= child sum -> exact descending order,
    O(N log S) time, and O(S) heap memory (scales smoothly to top_n = 1,000,000).
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    has_renum_in = "mut_type_renumbered" in singles_df.columns
    base_cols = ["pdb_file", "code", "chain", "mut_type_pdb"]
    if has_renum_in:
        base_cols.append("mut_type_renumbered")
    base_cols.extend(["stage1_additive_pred", "additive_rank"])

    if singles_df is None or singles_df.empty:
        return pd.DataFrame(columns=base_cols)

    if "mut_type_pdb" not in singles_df.columns:
        raise ValueError("singles_df must contain a 'mut_type_pdb' column.")
    if score_col not in singles_df.columns:
        raise ValueError(f"singles_df must contain score column '{score_col}'.")

    ex_wt: Set[str] = set()
    if exclude_wt_cys:
        ex_wt.add("C")
    if exclude_wt_pro:
        ex_wt.add("P")
    ex_mt: Set[str] = set()
    if exclude_mut_cys:
        ex_mt.add("C")
    if exclude_mut_pro:
        ex_mt.add("P")

    score_vals = pd.to_numeric(singles_df[score_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    pdb_mut_col = singles_df["mut_type_pdb"].tolist()
    renum_mut_col = singles_df["mut_type_renumbered"].tolist() if has_renum_in else None

    pdb_file_val = str(singles_df["pdb_file"].iloc[0]) if "pdb_file" in singles_df.columns else ""
    code_val = str(singles_df["code"].iloc[0]) if "code" in singles_df.columns else "protein"
    chain_val = str(singles_df["chain"].iloc[0]) if "chain" in singles_df.columns else "A"

    # Parse, filter, and deduplicate singles (keeping highest score if a single repeats).
    dedup: Dict[str, Tuple[float, int, str, str, str, Optional[str]]] = {}
    all_have_renum = has_renum_in
    for idx, raw_mut in enumerate(pdb_mut_col):
        sc = float(score_vals[idx])
        if not np.isfinite(sc):
            continue
        if not isinstance(raw_mut, str):
            continue
        parsed = _parse_single_mut_token(raw_mut)
        if parsed is None:
            continue
        wt, res_num, icode, mt = parsed
        if wt in ex_wt or mt in ex_mt:
            continue
        canon_mut = f"{wt}{res_num}{icode}{mt}"
        renum_s: Optional[str] = None
        if renum_mut_col is not None:
            rv = renum_mut_col[idx]
            if pd.notna(rv) and str(rv).strip():
                renum_s = str(rv).strip()
            else:
                all_have_renum = False

        prev = dedup.get(canon_mut)
        if prev is None or sc > prev[0]:
            dedup[canon_mut] = (sc, res_num, icode, mt, canon_mut, renum_s)

    if len(dedup) < 2:
        out_cols = ["pdb_file", "code", "chain", "mut_type_pdb"]
        if all_have_renum:
            out_cols.append("mut_type_renumbered")
        out_cols.extend(["stage1_additive_pred", "additive_rank"])
        return pd.DataFrame(columns=out_cols)

    # Sort descending by score; deterministic tie-break on (res_num, icode, mut).
    items = sorted(dedup.values(), key=lambda t: (-t[0], t[1], t[2], t[3]))
    n_singles = len(items)

    scores = [t[0] for t in items]
    res_nums = [t[1] for t in items]
    pos_keys: List[PosKey] = [(t[1], t[2]) for t in items]
    mut_pdb = [t[4] for t in items]
    mut_renum = [t[5] for t in items] if all_have_renum else None

    # Precompute integer position rank for fast O(1) canonical ordering in the output builder.
    unique_pos_sorted = sorted(set(pos_keys))
    pos_to_id = {pk: idx for idx, pk in enumerate(unique_pos_sorted)}
    pos_ids = [pos_to_id[pk] for pk in pos_keys]

    # Precompute integer allowed_id_pairs set if distance filtering is active.
    allowed_id_pairs: Optional[Set[Tuple[int, int]]] = None
    if allowed_pos_pairs is not None:
        allowed_id_pairs = set()
        for ka, kb in allowed_pos_pairs:
            ia = pos_to_id.get(ka)
            ib = pos_to_id.get(kb)
            if ia is not None and ib is not None and ia != ib:
                allowed_id_pairs.add((ia, ib) if ia < ib else (ib, ia))

    # Tree-structured best-first max-heap over (i, j) with 0 <= i < j < n_singles.
    heap: List[Tuple[float, int, int]] = [(-(scores[0] + scores[1]), 0, 1)]
    heappop = heapq.heappop
    heappush = heapq.heappush
    heapreplace = heapq.heapreplace

    out_i = array("I")
    out_j = array("I")
    out_sum = array("d")
    append_i = out_i.append
    append_j = out_j.append
    append_sum = out_sum.append

    check_sep = min_seq_sep > 1
    sep_val = int(min_seq_sep)
    check_dist = allowed_id_pairs is not None
    limit = int(top_n)
    last_idx = n_singles - 1
    count = 0

    while heap and count < limit:
        neg_s, i, j = heap[0]
        next_j = j + 1
        spawn_diag = (j == i + 1) and (i + 2 < n_singles)

        if next_j < n_singles:
            heapreplace(heap, (-(scores[i] + scores[next_j]), i, next_j))
            if spawn_diag:
                ni = i + 1
                heappush(heap, (-(scores[ni] + scores[ni + 1]), ni, ni + 1))
        elif spawn_diag:
            ni = i + 1
            heapreplace(heap, (-(scores[ni] + scores[ni + 1]), ni, ni + 1))
        else:
            heappop(heap)

        pi = pos_ids[i]
        pj = pos_ids[j]
        if pi == pj:
            continue
        if check_sep and abs(res_nums[i] - res_nums[j]) < sep_val:
            continue
        if check_dist:
            pair_id = (pi, pj) if pi < pj else (pj, pi)
            if pair_id not in allowed_id_pairs:  # type: ignore[operator]
                continue

        if pi < pj:
            append_i(i)
            append_j(j)
        else:
            append_i(j)
            append_j(i)
        append_sum(-neg_s)
        count += 1

    # Build canonical double strings in tight list comprehensions (fast even at 1,000,000 rows).
    mp = mut_pdb
    double_pdb = [f"{mp[a]}:{mp[b]}" for a, b in zip(out_i, out_j)]

    data: Dict[str, object] = {
        "pdb_file": pdb_file_val,
        "code": code_val,
        "chain": chain_val,
        "mut_type_pdb": double_pdb,
    }
    if mut_renum is not None:
        mr = mut_renum
        data["mut_type_renumbered"] = [f"{mr[a]}:{mr[b]}" for a, b in zip(out_i, out_j)]

    data["stage1_additive_pred"] = np.frombuffer(out_sum, dtype=np.float64).copy()
    data["additive_rank"] = np.arange(1, count + 1, dtype=np.int64)

    return pd.DataFrame(data)


def merge_stage1_and_stage2(
    singles_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
    original_pdb_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge Stage-2 rescored doubles with Stage-1 additive ranks, re-rank, and unify with Stage-1 singles.

    Returns (unified_df, reranked_doubles_df).
    """
    sel_cols = ["mut_type_pdb", "stage1_additive_pred", "additive_rank"]
    sel_meta = selected_df[sel_cols].drop_duplicates(subset=["mut_type_pdb"])

    doubles = stage2_df.merge(sel_meta, on="mut_type_pdb", how="left")

    # Re-rank descending by combined_pred (fallback to wt_lora_pred if combined_pred unavailable).
    rank_col = "combined_pred" if (
        "combined_pred" in doubles.columns
        and pd.to_numeric(doubles["combined_pred"], errors="coerce").notna().any()
    ) else "wt_lora_pred"

    doubles["_sort_score"] = pd.to_numeric(doubles[rank_col], errors="coerce")
    doubles["_sort_add_rank"] = pd.to_numeric(doubles["additive_rank"], errors="coerce").fillna(10**12)
    doubles = doubles.sort_values(
        by=["_sort_score", "_sort_add_rank"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    ).drop(columns=["_sort_score", "_sort_add_rank"]).reset_index(drop=True)

    doubles["full_rank"] = np.arange(1, len(doubles) + 1, dtype=np.int64)
    add_rank_num = pd.to_numeric(doubles["additive_rank"], errors="coerce")
    doubles["rank_change"] = (add_rank_num - doubles["full_rank"]).astype("Int64")

    singles_out = singles_df.copy()
    if original_pdb_file:
        if "pdb_file" in singles_out.columns:
            singles_out["pdb_file"] = original_pdb_file
        if "pdb_file" in doubles.columns:
            doubles["pdb_file"] = original_pdb_file

    # Align columns cleanly (Stage-1 singles retain wt_lora_pred and NaN for Stage-2-only columns).
    unified = pd.concat([singles_out, doubles], ignore_index=True, sort=False)

    ordered = [c for c in _UNIFIED_COL_ORDER if c in unified.columns]
    extras = [c for c in unified.columns if c not in ordered]
    unified = unified[ordered + extras]

    return unified, doubles


def _run_inference_stage(cmd: Sequence[str], cwd: str, env: Dict[str, str], stage_label: str) -> None:
    """Run an inference.py subprocess while streaming stdout/stderr unbuffered."""
    print(f"{stage_label} Command: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{stage_label} failed with exit code {rc}.")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-stage Fast Double Mutant Selection runner for ESM-MSR."
    )
    p.add_argument("--repo_dir", type=str, required=True, help="Path to esm-msr repository root.")
    p.add_argument("--pdb_file", type=str, required=True, help="Path to input PDB structure.")
    p.add_argument("--code", type=str, default="protein", help="Protein code identifier.")
    p.add_argument("--chain", type=str, default="A", help="Target chain ID.")
    p.add_argument("--output_csv", type=str, required=True, help="Path to unified output CSV.")
    p.add_argument("--top_n", type=int, default=1000, help="Number of top additive doubles to rescore (1..1,000,000).")
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to LoRA checkpoint (.safetensors/.pt/.ckpt).")
    p.add_argument("--lora_config", type=str, default=None, help="Optional path to lora_config JSON.")
    p.add_argument("--hparams_path", type=str, default=None, help="Optional path to hparams.yaml.")
    p.add_argument("--base_model_loc", type=str, default=None, help="Optional local ESM3 weights directory.")
    p.add_argument("--device", type=str, default="cuda", help="Torch device (cuda, cuda:0, mps, cpu).")
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    p.add_argument("--sigma", type=float, default=1.0, help="LoRA alpha multiplier.")
    p.add_argument("--mask_strategy", type=str, default=None, choices=["marginal", "independent"])
    p.add_argument("--screen_residues", type=str, default=None, help="Comma-separated PDB residue indices to screen.")
    p.add_argument("--screen_residues_except", type=str, default=None, help="Comma-separated PDB residue indices to exclude.")
    p.add_argument("--skip_fix_noncanonical", action="store_true", help="Skip non-canonical residue fixing.")
    p.add_argument("--distance_threshold", type=float, default=None, help="Optional max heavy-atom distance (A) for pairing.")
    p.add_argument("--exclude_wt_cys", action="store_true", help="Exclude WT Cysteine positions from double pairing.")
    p.add_argument("--exclude_wt_pro", action="store_true", help="Exclude WT Proline positions from double pairing.")
    p.add_argument("--exclude_mut_cys", action="store_true", help="Exclude mutations to Cysteine from double pairing.")
    p.add_argument("--exclude_mut_pro", action="store_true", help="Exclude mutations to Proline from double pairing.")
    p.add_argument("--min_seq_sep", type=int, default=1, help="Minimum sequence index separation between paired residues.")
    p.add_argument("--keep_intermediates", action="store_true", help="Keep intermediate Stage-1/Stage-2 files alongside output_csv.")
    return p


def _print_top_summary(doubles_df: pd.DataFrame, max_rows: int = 10) -> None:
    if doubles_df.empty:
        print("[Summary] No Stage-2 double mutants produced.", flush=True)
        return
    print(f"\n[Summary] Top {min(max_rows, len(doubles_df))} Re-Ranked Double Mutants:", flush=True)
    header = (
        f"{'Rank':>4}  {'Mutation':<16}  {'Comb ΔΔG':>9}  {'Stg1 Add':>9}  "
        f"{'Epistasis':>9}  {'ΔRank':>6}  {'Dist(Å)':>7}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in doubles_df.head(max_rows).itertuples(index=False):
        r = getattr(row, "full_rank", "")
        mut = getattr(row, "mut_type_pdb", "")
        comb = getattr(row, "combined_pred", np.nan)
        s1 = getattr(row, "stage1_additive_pred", np.nan)
        epi = getattr(row, "epi_pred", np.nan)
        rc = getattr(row, "rank_change", None)
        dist = getattr(row, "dist", np.nan)
        comb_s = f"{comb:+9.3f}" if pd.notna(comb) else f"{'N/A':>9}"
        s1_s = f"{s1:+9.3f}" if pd.notna(s1) else f"{'N/A':>9}"
        epi_s = f"{epi:+9.3f}" if pd.notna(epi) else f"{'N/A':>9}"
        rc_s = f"{int(rc):+6d}" if pd.notna(rc) else f"{'N/A':>6}"
        dist_s = f"{dist:7.2f}" if pd.notna(dist) else f"{'N/A':>7}"
        print(f"{r:>4}  {mut:<16}  {comb_s}  {s1_s}  {epi_s}  {rc_s}  {dist_s}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.top_n < 1 or args.top_n > 1_000_000:
        parser.error("--top_n must be between 1 and 1,000,000.")
    if args.screen_residues and args.screen_residues_except:
        parser.error("--screen_residues and --screen_residues_except are mutually exclusive.")

    repo_dir = os.path.abspath(os.path.expanduser(args.repo_dir))
    inference_py = os.path.join(repo_dir, "src", "esm_msr", "inference.py")
    if not os.path.isfile(inference_py):
        raise FileNotFoundError(f"Could not locate inference.py at '{inference_py}'.")

    pdb_abs = os.path.abspath(os.path.expanduser(args.pdb_file))
    if not os.path.isfile(pdb_abs):
        raise FileNotFoundError(f"Input PDB file not found: '{pdb_abs}'.")

    output_csv = os.path.abspath(os.path.expanduser(args.output_csv))
    output_dir = os.path.dirname(output_csv) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Prepare writable working directory
    if args.keep_intermediates:
        stem = os.path.splitext(os.path.basename(output_csv))[0]
        work_dir = os.path.join(output_dir, f"{stem}_fast_doubles_work")
        os.makedirs(work_dir, exist_ok=True)
        cleanup_work = False
    else:
        work_dir = tempfile.mkdtemp(prefix="esm_msr_fast_doubles_")
        cleanup_work = True

    env = os.environ.copy()
    repo_src = os.path.join(repo_dir, "src")
    prev_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{prev_pypath}" if prev_pypath else repo_src
    env["PYTHONUNBUFFERED"] = "1"

    t_total = time.perf_counter()
    try:
        # Copy PDB into work_dir so *_inference_prep.pdb never pollutes the user's PDB folder.
        work_pdb = os.path.join(work_dir, os.path.basename(pdb_abs))
        shutil.copy2(pdb_abs, work_pdb)

        stage1_csv = os.path.join(work_dir, "stage1_singles.csv")
        stage2_input_csv = os.path.join(work_dir, "stage2_input_doubles.csv")
        stage2_csv = os.path.join(work_dir, "stage2_doubles.csv")

        def _common_model_args() -> List[str]:
            margs = [
                "--checkpoint_path", os.path.abspath(os.path.expanduser(args.checkpoint_path)),
                "--device", args.device,
                "--model_dtype", args.model_dtype,
                "--batch_size", str(args.batch_size),
                "--sigma", str(args.sigma),
            ]
            if args.lora_config:
                margs.extend(["--lora_config", os.path.abspath(os.path.expanduser(args.lora_config))])
            if args.hparams_path:
                margs.extend(["--hparams_path", os.path.abspath(os.path.expanduser(args.hparams_path))])
            if args.base_model_loc:
                margs.extend(["--base_model_loc", os.path.abspath(os.path.expanduser(args.base_model_loc))])
            if args.mask_strategy:
                margs.extend(["--mask_strategy", args.mask_strategy])
            return margs

        # -------- Stage 1: Fast WT-pass all-singles scan --------
        print(f"[Stage 1/2] Running fast WT-pass single-mutant scan on {os.path.basename(pdb_abs)} (chain {args.chain})...", flush=True)
        stage1_cmd = [
            sys.executable, "-u", inference_py,
            "--pdb_file", work_pdb,
            "--code", args.code,
            "--chain", args.chain,
            "--mode", "singles",
            "--skip_reverse",
            "--output_csv", stage1_csv,
        ] + _common_model_args()

        if args.screen_residues:
            stage1_cmd.extend(["--screen_residues", args.screen_residues])
        if args.screen_residues_except:
            stage1_cmd.extend(["--screen_residues_except", args.screen_residues_except])
        if args.skip_fix_noncanonical:
            stage1_cmd.append("--skip_fix_noncanonical")

        t1 = time.perf_counter()
        _run_inference_stage(stage1_cmd, cwd=work_dir, env=env, stage_label="[Stage 1/2]")
        if not os.path.isfile(stage1_csv):
            raise RuntimeError(f"Stage 1 completed but did not write expected CSV: '{stage1_csv}'.")
        print(f"[Stage 1/2] Completed in {time.perf_counter() - t1:.1f}s.", flush=True)

        # -------- Selection: Exact Top-N additive double mutants --------
        singles_df = pd.read_csv(stage1_csv)
        if singles_df.empty:
            raise RuntimeError("Stage 1 produced 0 single-mutant predictions.")

        stage1_pdb_used = str(singles_df["pdb_file"].iloc[0]) if "pdb_file" in singles_df.columns else work_pdb
        if not os.path.isfile(stage1_pdb_used):
            stage1_pdb_used = work_pdb

        allowed_pairs = None
        if args.distance_threshold is not None and args.distance_threshold > 0:
            allowed_pairs = compute_allowed_pos_pairs_within_distance(
                stage1_pdb_used, args.chain, args.distance_threshold
            )
            print(
                f"[Selection] Applied heavy-atom distance threshold <= {args.distance_threshold:.2f} A "
                f"({len(allowed_pairs) if allowed_pairs is not None else 0} eligible residue pairs).",
                flush=True,
            )

        t_sel = time.perf_counter()
        selected_df = select_top_n_additive_doubles(
            singles_df,
            top_n=args.top_n,
            score_col="wt_lora_pred",
            exclude_wt_cys=args.exclude_wt_cys,
            exclude_wt_pro=args.exclude_wt_pro,
            exclude_mut_cys=args.exclude_mut_cys,
            exclude_mut_pro=args.exclude_mut_pro,
            min_seq_sep=args.min_seq_sep,
            allowed_pos_pairs=allowed_pairs,
        )
        if selected_df.empty:
            raise RuntimeError("No valid double-mutant pairs remained after filtering.")

        n_singles = len(singles_df)
        n_possible_est = n_singles * (n_singles - 1) // 2
        s_min = float(selected_df["stage1_additive_pred"].min())
        s_max = float(selected_df["stage1_additive_pred"].max())
        print(
            f"[Selection] From {n_singles:,} singles (~{n_possible_est:,} raw pairs), selected top {len(selected_df):,} "
            f"additive doubles in {time.perf_counter() - t_sel:.2f}s "
            f"(Stage-1 additive ΔΔG range: [{s_min:+.3f}, {s_max:+.3f}]).",
            flush=True,
        )

        # Write Stage-2 input CSV pointing to the already-prepped Stage-1 PDB.
        s2_in_cols = ["pdb_file", "code", "chain", "mut_type_pdb"]
        if "mut_type_renumbered" in selected_df.columns:
            s2_in_cols.append("mut_type_renumbered")
        s2_in = selected_df[s2_in_cols].copy()
        s2_in["pdb_file"] = stage1_pdb_used
        s2_in.to_csv(stage2_input_csv, index=False)

        # -------- Stage 2: Dual-view WT+MT rescoring --------
        print(
            f"[Stage 2/2] Running dual-view (WT+MT) rescoring on {len(selected_df):,} selected double mutants...",
            flush=True,
        )
        stage2_cmd = [
            sys.executable, "-u", inference_py,
            "--input_csv", stage2_input_csv,
            "--output_csv", stage2_csv,
            "--skip_additive",
            "--calculate_distances",
            "--skip_fix_noncanonical",
        ] + _common_model_args()

        t2 = time.perf_counter()
        _run_inference_stage(stage2_cmd, cwd=work_dir, env=env, stage_label="[Stage 2/2]")
        if not os.path.isfile(stage2_csv):
            raise RuntimeError(f"Stage 2 completed but did not write expected CSV: '{stage2_csv}'.")
        print(f"[Stage 2/2] Completed in {time.perf_counter() - t2:.1f}s.", flush=True)

        # -------- Merge, Re-rank, and Save Unified CSV --------
        stage2_df = pd.read_csv(stage2_csv)
        unified_df, reranked_doubles = merge_stage1_and_stage2(
            singles_df=singles_df,
            selected_df=selected_df,
            stage2_df=stage2_df,
            original_pdb_file=pdb_abs,
        )
        unified_df.to_csv(output_csv, index=False)

        print(
            f"[Done] Wrote unified CSV ({len(singles_df):,} singles + {len(reranked_doubles):,} re-ranked doubles) "
            f"to '{output_csv}' in {time.perf_counter() - t_total:.1f}s total.",
            flush=True,
        )
        _print_top_summary(reranked_doubles, max_rows=10)
        return 0
    finally:
        if cleanup_work and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
