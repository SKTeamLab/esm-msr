#!/usr/bin/env python3
"""Prepare, collect, verify, and apply ColabFold AF2 structures for `rocklin_denovo` (`design_id_*`).

Workflow:
  1. `prepare`: Infer exact `SAGGSAGG` display-linker padding bounds for all 1,000 mutated
     `design_id_*` parent scaffolds in `data/mgnify/mgnify_all_mutations.csv.gz`,
     verify 100% coordinate consistency across all 209,818 variants, write single-sequence
     `.a3m` files partitioned into 3 length-sorted shards, emit `manifest.csv` and
     `af2_rocklin_denovo.sbatch` for node `gpu2`.
  2. `collect`: Verify all 1,000 ColabFold relaxed rank-1 PDBs (`Chain A`, `1..L_core`, exact
     CA sequence match, finite backbone heavy atoms, ProteinChain spot-check), copy them
     atomically into `<mgnify_dir>/structures_af2_denovo/<bucket>/<code_wt>.pdb`, and write
     `af2_metrics.csv` with per-scaffold pLDDT and pTM scores.
  3. `apply`: Strip symmetric linkers and shift `mut_type` coordinates to `1..L_core` across
     `mgnify_wildtypes.csv.gz`, `mgnify_all_mutations.csv.gz`, and `mgnify_all_collapsed.csv.gz`
     for the 1,000 folded scaffolds (leaving all `mgnify=True` rows untouched), recompute
     `annotate_additivity`, regenerate `mgnify_mapped.csv` and `mgnify_multis_only.csv`, and
     update `summary.json`.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess_mgnify  # noqa: E402

CONSTRUCT_LEN = 80
SAGG_FLANK = "SAGGSAGG" * 12
MUT_TOKEN_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
REQUIRED_BACKBONE_ATOMS = ("N", "CA", "C", "O")


def structure_rel_path_denovo(code_wt: str) -> str:
    """Return `<bucket>/<code_wt>.pdb` relative path under `structures_af2_denovo/`."""
    tail = code_wt.rsplit("_", 1)[1]
    bucket = int(tail) % 1000 if tail.isdigit() else 0
    return f"structures_af2_denovo/{bucket:03d}/{code_wt}.pdb"


def compose_trajectory_mutations(scaf: str, raw_name: str) -> List[Tuple[int, str, str]]:
    """Compose multi-step sequential edit tokens in `design_id_<id>_<tok1>_<tok2>...` order.

    Handles sequential same-site mutations (e.g., `K23G_G23H` -> `(23, 'K', 'H')`) and
    exact reversions (`K34N_N34K` -> omitted), returning sorted `(core_pos, orig_aa, mut_aa)`.
    """
    prefix = f"{scaf}_"
    if not raw_name.startswith(prefix):
        raise ValueError(f"Variant name {raw_name!r} does not start with scaffold prefix {prefix!r}")
    suffix = raw_name[len(prefix) :]
    if not suffix:
        raise ValueError(f"Variant name {raw_name!r} has empty mutation suffix")

    state: Dict[int, Tuple[str, str]] = {}
    for tok in suffix.split("_"):
        m = MUT_TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"Unrecognized mutation token {tok!r} in {raw_name!r}")
        fr, pos_s, to = m.groups()
        cp = int(pos_s)
        orig = state[cp][0] if cp in state else fr
        state[cp] = (orig, to)

    return [(cp, orig, cur) for cp, (orig, cur) in sorted(state.items()) if orig != cur]


def shift_mut_type(mut_type: str, delta: int) -> str:
    """Shift every residue index in a canonical `wt<pos>mt:...` string by `delta`."""
    parts = mut_type.split(":")
    shifted: List[str] = []
    for p in parts:
        m = MUT_TOKEN_RE.match(p)
        if not m:
            raise ValueError(f"Invalid mut_type token {p!r} in {mut_type!r}")
        fr, pos_s, to = m.groups()
        new_pos = int(pos_s) + delta
        if new_pos < 1:
            raise ValueError(f"Shifted position < 1 ({new_pos}) for token {p!r} with delta={delta}")
        shifted.append(f"{fr}{new_pos}{to}")
    return ":".join(shifted)


def infer_design_id_core_bounds(
    scaf: str,
    wt_seq: str,
    raw_names: Iterable[str],
    mut_types: Iterable[str],
) -> Tuple[int, int, int, str]:
    """Infer exact `(left_pad, right_pad, core_len, core_wt_seq)` for one `design_id_*` scaffold."""
    if len(wt_seq) != CONSTRUCT_LEN:
        raise ValueError(f"{scaf}: expected construct length {CONSTRUCT_LEN}, got {len(wt_seq)}")

    offsets: set[int] = set()
    max_core_pos = 0
    n_variants = 0

    for nm, mt in zip(raw_names, mut_types):
        n_variants += 1
        comp = compose_trajectory_mutations(scaf, nm)
        fms = str(mt).split(":")
        if not comp or len(comp) != len(fms):
            raise ValueError(f"{scaf}: composed mutation count mismatch on {nm!r} vs {mt!r}")

        for tok in nm[len(scaf) + 1 :].split("_"):
            cp_raw = int(tok[1:-1])
            if cp_raw > max_core_pos:
                max_core_pos = cp_raw

        for (cp, fr, to), fp in zip(comp, fms):
            m = MUT_TOKEN_RE.match(fp)
            if not m:
                raise ValueError(f"{scaf}: malformed full mut_type token {fp!r}")
            f_fr, f_pos_s, f_to = m.groups()
            if fr != f_fr or to != f_to:
                raise ValueError(f"{scaf}: AA mismatch between {nm!r} ({fr}{cp}{to}) and {mt!r} ({fp})")
            offsets.add(int(f_pos_s) - cp)

    if n_variants == 0:
        raise ValueError(f"{scaf}: zero variants provided")
    if len(offsets) != 1:
        raise ValueError(f"{scaf}: inconsistent core offsets {sorted(offsets)}")

    left_pad = next(iter(offsets))
    if left_pad < 0 or left_pad > 20:
        raise ValueError(f"{scaf}: unexpected left_pad={left_pad}")

    if left_pad >= 1:
        if wt_seq[:left_pad] != SAGG_FLANK[:left_pad]:
            raise ValueError(f"{scaf}: N-terminal prefix {wt_seq[:left_pad]!r} != {SAGG_FLANK[:left_pad]!r}")
        cands = [
            L
            for L in (CONSTRUCT_LEN - 2 * left_pad - 1, CONSTRUCT_LEN - 2 * left_pad)
            if max_core_pos <= L <= CONSTRUCT_LEN
            and wt_seq[left_pad + L :] == SAGG_FLANK[: CONSTRUCT_LEN - left_pad - L]
        ]
        if len(cands) != 1:
            raise ValueError(
                f"{scaf}: ambiguous/invalid core lengths {cands} for left_pad={left_pad}, max_core_pos={max_core_pos}"
            )
        core_len = cands[0]
    else:
        core_len = CONSTRUCT_LEN if (max_core_pos == CONSTRUCT_LEN or not wt_seq.endswith("S")) else (CONSTRUCT_LEN - 1)
        if core_len < max_core_pos:
            raise ValueError(f"{scaf}: inferred core_len={core_len} < max_core_pos={max_core_pos}")

    right_pad = CONSTRUCT_LEN - left_pad - core_len
    if right_pad > 0 and wt_seq[CONSTRUCT_LEN - right_pad :] != SAGG_FLANK[:right_pad]:
        raise ValueError(f"{scaf}: C-terminal suffix {wt_seq[CONSTRUCT_LEN - right_pad :]!r} != expected flank")

    core_wt_seq = wt_seq[left_pad : left_pad + core_len]
    return left_pad, right_pad, core_len, core_wt_seq


def partition_contiguous_balanced(weights: Sequence[float], n_shards: int) -> List[Tuple[int, int]]:
    """Partition `0..len(weights)` into `n_shards` contiguous ranges minimizing max shard weight."""
    n = len(weights)
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if n < n_shards:
        raise ValueError(f"Cannot partition {n} items into {n_shards} non-empty shards")
    if n_shards == 1:
        return [(0, n)]

    prefix = [0.0] * (n + 1)
    for i, w in enumerate(weights):
        prefix[i + 1] = prefix[i] + float(w)

    # Exact DP over split boundaries (k <= n_shards, n ~ 1000)
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(n_shards + 1)]
    prev_idx = [[-1] * (n + 1) for _ in range(n_shards + 1)]
    for i in range(1, n + 1):
        dp[1][i] = prefix[i]

    for k in range(2, n_shards + 1):
        # Monotonic split point optimization not needed for n=1000, k=3; simple loop takes <20ms
        for i in range(k, n + 1):
            best_val = inf
            best_j = -1
            for j in range(k - 1, i):
                cand = max(dp[k - 1][j], prefix[i] - prefix[j])
                if cand < best_val:
                    best_val = cand
                    best_j = j
                elif prefix[i] - prefix[j] >= best_val:
                    # As j decreases further, segment sum can only shrink, so scan downward? Actually we iterate j upward:
                    pass
            dp[k][i] = best_val
            prev_idx[k][i] = best_j

    bounds: List[Tuple[int, int]] = []
    cur = n
    for k in range(n_shards, 0, -1):
        start = 0 if k == 1 else prev_idx[k][cur]
        bounds.append((start, cur))
        cur = start
    bounds.reverse()
    return bounds


def _render_sbatch(remote_root: str) -> str:
    return f"""#!/bin/bash
#SBATCH -p sbatch
#SBATCH -w gpu2
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output={remote_root}/logs/%x-%j.out

module purge
set -o pipefail
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate nvcc
hostname
echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
nvidia-smi -i "${{CUDA_VISIBLE_DEVICES:-0}}" --query-gpu=name,memory.total,driver_version --format=csv || true
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
export PATH="/home/sreeves/software/localcolabfold/colabfold-conda/bin:$PATH"
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=4.0

R={remote_root}
IN="$R/inputs/shard_${{SHARD}}"
OUT="$R/predictions/shard_${{SHARD}}"
mkdir -p "$OUT"
date '+start %F %T'
set -e
colabfold_batch "$IN" "$OUT" \\
  --num-models 5 \\
  --num-recycle 3 \\
  --amber \\
  --num-relax 1 \\
  --use-gpu-relax \\
  --sort-queries-by length 2>&1 | grep -v "^$"
date '+end %F %T'
"""


def load_manifest(manifest_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load `af2_rocklin_denovo/manifest.csv` keyed by `code_wt`."""
    manifest: Dict[str, Dict[str, Any]] = {}
    with open(manifest_csv, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest[row["code_wt"]] = {
                "code_wt": row["code_wt"],
                "left_pad": int(row["left_pad"]),
                "right_pad": int(row["right_pad"]),
                "core_len": int(row["core_len"]),
                "shard": int(row["shard"]),
                "pdb_relpath": row["pdb_relpath"],
                "core_wt_seq": row["core_wt_seq"],
                "full_wt_seq": row["full_wt_seq"],
            }
    return manifest


def cmd_prepare(
    mgnify_dir: Path,
    remote_root: str = "/home/sreeves/esm-msr-devel/af2_rocklin_denovo",
    n_shards: int = 3,
) -> Dict[str, Any]:
    mgnify_dir = mgnify_dir.resolve()
    all_muts_csv = mgnify_dir / "mgnify_all_mutations.csv.gz"
    if not all_muts_csv.is_file():
        raise FileNotFoundError(f"Missing {all_muts_csv}")

    print(f"[prepare] Reading rocklin_denovo variants from {all_muts_csv} ...", flush=True)
    df = pd.read_csv(
        all_muts_csv,
        usecols=["name", "code_wt", "wt_seq", "aa_seq", "mut_type", "source"],
        low_memory=False,
    )
    rd = df[df["source"] == "rocklin_denovo"].copy()
    if rd.empty:
        raise RuntimeError(f"No source=='rocklin_denovo' rows found in {all_muts_csv}")

    wt_nuniq = rd.groupby("code_wt")["wt_seq"].nunique()
    if (wt_nuniq != 1).any():
        bad = wt_nuniq[wt_nuniq != 1].index.tolist()[:5]
        raise RuntimeError(f"Scaffolds with multiple wt_seq values: {bad}")

    records: List[Dict[str, Any]] = []
    flank_mut_errors = 0
    shifted_mut_errors = 0

    for scaf, sub in rd.groupby("code_wt", sort=False):
        wt_seq = str(sub["wt_seq"].iloc[0])
        raw_names = sub["name"].astype(str).tolist()
        mut_types = sub["mut_type"].astype(str).tolist()
        aa_seqs = sub["aa_seq"].astype(str).tolist()

        left_pad, right_pad, core_len, core_wt_seq = infer_design_id_core_bounds(
            str(scaf), wt_seq, raw_names, mut_types
        )

        # Verify 100% flank preservation and shifted core diff across all variants
        wt_left = wt_seq[:left_pad]
        wt_right = wt_seq[left_pad + core_len :]
        for ms, mt in zip(aa_seqs, mut_types):
            if ms[:left_pad] != wt_left or ms[left_pad + core_len :] != wt_right:
                flank_mut_errors += 1
            core_ms = ms[left_pad : left_pad + core_len]
            shifted = shift_mut_type(mt, -left_pad)
            diff_mt, _ = preprocess_mgnify.diff_to_mut_type(core_wt_seq, core_ms)
            if shifted != diff_mt:
                shifted_mut_errors += 1

        records.append({
            "code_wt": str(scaf),
            "left_pad": left_pad,
            "right_pad": right_pad,
            "core_len": core_len,
            "pdb_relpath": structure_rel_path_denovo(str(scaf)),
            "core_wt_seq": core_wt_seq,
            "full_wt_seq": wt_seq,
        })

    if flank_mut_errors or shifted_mut_errors:
        raise RuntimeError(
            f"Verification failed: flank_mut_errors={flank_mut_errors}, shifted_mut_errors={shifted_mut_errors}"
        )

    # Sort deterministically by (core_len, code_wt) and partition into contiguous length-balanced shards
    records.sort(key=lambda r: (r["core_len"], r["code_wt"]))
    weights = [r["core_len"] ** 1.5 for r in records]
    bounds = partition_contiguous_balanced(weights, n_shards)

    work_dir = mgnify_dir / "af2_rocklin_denovo"
    inputs_dir = work_dir / "inputs"
    if inputs_dir.exists():
        for child in inputs_dir.glob("shard_*"):
            if child.is_dir():
                shutil.rmtree(child)
    for k in range(n_shards):
        (inputs_dir / f"shard_{k}").mkdir(parents=True, exist_ok=True)

    shard_summaries: List[Dict[str, Any]] = []
    for k, (lo, hi) in enumerate(bounds):
        shard_recs = records[lo:hi]
        for rec in shard_recs:
            rec["shard"] = k
            a3m_path = inputs_dir / f"shard_{k}" / f"{rec['code_wt']}.a3m"
            a3m_path.write_text(f">101\n{rec['core_wt_seq']}\n", encoding="utf-8")
        shard_summaries.append({
            "shard": k,
            "n_scaffolds": len(shard_recs),
            "len_min": min(r["core_len"] for r in shard_recs),
            "len_max": max(r["core_len"] for r in shard_recs),
            "weight_sum": round(sum(weights[lo:hi]), 1),
        })

    manifest_csv = work_dir / "manifest.csv"
    tmp_manifest = manifest_csv.with_name(manifest_csv.name + ".tmp")
    cols = ["code_wt", "left_pad", "right_pad", "core_len", "shard", "pdb_relpath", "core_wt_seq", "full_wt_seq"]
    with open(tmp_manifest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in records:
            w.writerow({c: rec[c] for c in cols})
    tmp_manifest.replace(manifest_csv)

    sbatch_path = work_dir / "af2_rocklin_denovo.sbatch"
    sbatch_path.write_text(_render_sbatch(remote_root), encoding="utf-8")

    summary = {
        "n_scaffolds": len(records),
        "n_variants_verified": int(len(rd)),
        "flank_mut_errors": flank_mut_errors,
        "shifted_mut_errors": shifted_mut_errors,
        "core_len_min": min(r["core_len"] for r in records),
        "core_len_median": int(np.median([r["core_len"] for r in records])),
        "core_len_max": max(r["core_len"] for r in records),
        "shards": shard_summaries,
        "manifest_csv": str(manifest_csv),
        "sbatch_path": str(sbatch_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _verify_pdb_backbone_atoms(pdb_path: Path, expected_len: int) -> Optional[str]:
    """Verify every residue `1..expected_len` has finite `N, CA, C, O` heavy-atom coordinates."""
    found: Dict[int, set[str]] = collections.defaultdict(set)
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            atom = line[12:16].strip()
            if atom not in REQUIRED_BACKBONE_ATOMS:
                continue
            resid = int(line[22:26])
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                return f"{pdb_path.name}: unparseable coordinate on resid {resid} atom {atom}"
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return f"{pdb_path.name}: non-finite coordinate on resid {resid} atom {atom}"
            found[resid].add(atom)

    req_set = set(REQUIRED_BACKBONE_ATOMS)
    for r in range(1, expected_len + 1):
        if not req_set.issubset(found.get(r, set())):
            missing = sorted(req_set - found.get(r, set()))
            return f"{pdb_path.name}: resid {r} missing backbone atoms {missing}"
    return None


def _parse_ca_bfactors(pdb_path: Path) -> List[float]:
    bfacs: List[float] = []
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                bfacs.append(float(line[60:66]))
    return bfacs


def _spot_check_denovo_proteinchain(
    mgnify_dir: Path,
    manifest: Dict[str, Dict[str, Any]],
    n_spot: int = 20,
) -> Tuple[int, List[str]]:
    if n_spot <= 0:
        return 0, []
    import torch
    from esm.utils.structure.protein_chain import ProteinChain
    from esm_msr import inference, preprocess

    items = list(manifest.values())
    step = max(1, len(items) // min(n_spot, len(items)))
    sample = items[::step][: min(n_spot, len(items))]
    errors: List[str] = []
    checked = 0

    for rec in sample:
        pdb_abs = str((mgnify_dir / rec["pdb_relpath"]).resolve())
        pc = ProteinChain.from_pdb(pdb_abs, "A", is_predicted=True)
        coords, plddt, _ = pc.to_structure_encoder_inputs()
        if not np.isfinite(pc.atom37_positions[:, [0, 1, 2, 4], :]).all():
            errors.append(f"{rec['code_wt']}: non-finite ProteinChain backbone coordinates")
            continue
        pl_np = plddt.detach().cpu().numpy()
        if not np.isfinite(pl_np).all() or float(pl_np.min()) < 0.0 or float(pl_np.max()) > 100.0:
            errors.append(f"{rec['code_wt']}: pLDDT out of [0, 100] ({pl_np.min()}..{pl_np.max()})")
            continue
        dummy_struct = torch.zeros((1, len(pc.sequence) + 2), dtype=torch.long)
        corr_seq, _, _ = inference._handle_mutated_backbone(
            pc.sequence, coords, dummy_struct, backbone_mutation=None, assert_wt=True
        )
        if corr_seq != rec["core_wt_seq"]:
            errors.append(f"{rec['code_wt']}: ProteinChain sequence mismatch")
            continue
        res_map, aa_map = preprocess.get_pdb_to_seq_mapping(pdb_abs, "A", pc.sequence)
        if len(res_map) != rec["core_len"] or len(aa_map) != rec["core_len"]:
            errors.append(f"{rec['code_wt']}: get_pdb_to_seq_mapping length mismatch")
            continue
        checked += 1
    return checked, errors


def _extract_metrics_row(
    scaf: str,
    rec: Dict[str, Any],
    src_pdb: Optional[Path],
    j_path: Optional[Path],
    dst_pdb: Path,
) -> Dict[str, Any]:
    if src_pdb is not None and "_relaxed_rank_001_" in src_pdb.name:
        suffix_part = src_pdb.name.split("_relaxed_rank_001_", 1)[1]
        model_name = suffix_part[:-4] if suffix_part.endswith(".pdb") else suffix_part
    else:
        model_name = "alphafold2_ptm_rank_001"

    ptm = float("nan")
    if j_path is not None and j_path.is_file():
        meta = json.loads(j_path.read_text(encoding="utf-8"))
        plddt_vals = [float(x) for x in meta.get("plddt", [])]
        if not plddt_vals:
            plddt_vals = _parse_ca_bfactors(dst_pdb)
        if "ptm" in meta and meta["ptm"] is not None:
            ptm = float(meta["ptm"])
    else:
        plddt_vals = _parse_ca_bfactors(dst_pdb)

    return {
        "code_wt": scaf,
        "core_len": rec["core_len"],
        "shard": rec["shard"],
        "model_name": model_name,
        "mean_plddt": round(float(np.mean(plddt_vals)), 3),
        "min_plddt": round(float(np.min(plddt_vals)), 3),
        "ptm": round(ptm, 4) if math.isfinite(ptm) else "",
        "pdb_relpath": rec["pdb_relpath"],
    }


def _update_summary_denovo_metrics(
    mgnify_dir: Path,
    metrics_rows: List[Dict[str, Any]],
) -> None:
    summary_path = mgnify_dir / "summary.json"
    if not summary_path.is_file() or not metrics_rows:
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plddts = [float(r["mean_plddt"]) for r in metrics_rows]
    ptms = [float(r["ptm"]) for r in metrics_rows if r["ptm"] != ""]
    denovo_sec = summary.get("rocklin_denovo_af2", {})
    denovo_sec["n_structures_on_disk"] = len(metrics_rows)
    denovo_sec["mean_plddt_median"] = round(float(np.median(plddts)), 2)
    denovo_sec["mean_plddt_min"] = round(float(np.min(plddts)), 2)
    if ptms:
        denovo_sec["ptm_median"] = round(float(np.median(ptms)), 4)
    summary["rocklin_denovo_af2"] = denovo_sec
    tmp_sum = summary_path.with_name(summary_path.name + ".tmp")
    tmp_sum.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp_sum.replace(summary_path)


def cmd_collect(
    mgnify_dir: Path,
    predictions_dir: Optional[Path] = None,
    spot_check_pdbs: int = 20,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    mgnify_dir = mgnify_dir.resolve()
    work_dir = mgnify_dir / "af2_rocklin_denovo"
    manifest = load_manifest(work_dir / "manifest.csv")
    pred_root = (predictions_dir or (work_dir / "raw_predictions")).resolve()
    if not pred_root.is_dir():
        if not allow_partial:
            raise FileNotFoundError(f"Predictions directory not found: {pred_root}")
        pred_root.mkdir(parents=True, exist_ok=True)

    pdb_by_scaf: Dict[str, Path] = {}
    json_by_scaf: Dict[str, Path] = {}
    for p in sorted(pred_root.rglob("*.pdb")):
        name = p.name
        if "_relaxed_rank_001_" not in name:
            continue
        scaf = name.split("_relaxed_rank_001_", 1)[0]
        if scaf in manifest and scaf not in pdb_by_scaf:
            pdb_by_scaf[scaf] = p

    for j in sorted(pred_root.rglob("*.json")):
        name = j.name
        if "_scores_rank_001_" not in name:
            continue
        scaf = name.split("_scores_rank_001_", 1)[0]
        if scaf in manifest and scaf not in json_by_scaf:
            json_by_scaf[scaf] = j

    if not allow_partial:
        missing = sorted(
            scaf for scaf, rec in manifest.items()
            if scaf not in pdb_by_scaf and not (mgnify_dir / rec["pdb_relpath"]).is_file()
        )
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} relaxed rank-1 ColabFold PDBs in {pred_root}; first 5: {missing[:5]}"
            )

    metrics_rows: List[Dict[str, Any]] = []
    newly_installed: List[str] = []
    errors: List[str] = []

    for scaf, rec in manifest.items():
        dst_pdb = mgnify_dir / rec["pdb_relpath"]
        src_pdb = pdb_by_scaf.get(scaf)
        j_path = json_by_scaf.get(scaf)

        if (
            dst_pdb.is_file()
            and preprocess_mgnify._verify_one_pdb((scaf, str(dst_pdb), rec["core_wt_seq"])) is None
        ):
            # Already verified on disk: leave untouched so live inference readers never see torn writes
            metrics_rows.append(_extract_metrics_row(scaf, rec, src_pdb, j_path, dst_pdb))
            continue

        if src_pdb is None:
            continue

        dst_pdb.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst_pdb.with_name(f"{dst_pdb.name}.tmp.{os.getpid()}")
        shutil.copyfile(src_pdb, tmp_dst)

        err = preprocess_mgnify._verify_one_pdb((scaf, str(tmp_dst), rec["core_wt_seq"]))
        if err is None:
            err = _verify_pdb_backbone_atoms(tmp_dst, rec["core_len"])
        if err is not None:
            tmp_dst.unlink(missing_ok=True)
            if allow_partial:
                print(f"[collect] Skipping unverified partial file for {scaf}: {err}", flush=True)
                continue
            errors.append(err)
            continue

        os.replace(tmp_dst, dst_pdb)
        newly_installed.append(scaf)
        metrics_rows.append(_extract_metrics_row(scaf, rec, src_pdb, j_path, dst_pdb))

    if errors:
        raise RuntimeError(f"PDB verification failed on {len(errors)} structures: {errors[:5]}")

    if newly_installed and spot_check_pdbs > 0:
        sub_manifest = {s: manifest[s] for s in newly_installed}
        spot_checked, spot_errors = _spot_check_denovo_proteinchain(
            mgnify_dir, sub_manifest, n_spot=spot_check_pdbs
        )
        if spot_errors:
            raise RuntimeError(f"ProteinChain spot-check failed: {spot_errors[:5]}")
    else:
        spot_checked = 0

    metrics_csv = work_dir / "af2_metrics.csv"
    tmp_metrics = metrics_csv.with_name(metrics_csv.name + ".tmp")
    m_cols = ["code_wt", "core_len", "shard", "model_name", "mean_plddt", "min_plddt", "ptm", "pdb_relpath"]
    with open(tmp_metrics, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=m_cols)
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)
    tmp_metrics.replace(metrics_csv)
    _update_summary_denovo_metrics(mgnify_dir, metrics_rows)

    plddts = [float(r["mean_plddt"]) for r in metrics_rows]
    ptms = [float(r["ptm"]) for r in metrics_rows if r["ptm"] != ""]
    summary = {
        "n_target": len(manifest),
        "n_collected": len(metrics_rows),
        "n_newly_installed": len(newly_installed),
        "n_missing": len(manifest) - len(metrics_rows),
        "n_spot_checked": spot_checked,
        "mean_plddt_median": round(float(np.median(plddts)), 2) if plddts else None,
        "mean_plddt_p10": round(float(np.percentile(plddts, 10)), 2) if plddts else None,
        "mean_plddt_min": round(float(np.min(plddts)), 2) if plddts else None,
        "ptm_median": round(float(np.median(ptms)), 4) if ptms else None,
        "metrics_csv": str(metrics_csv),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _verify_all_manifest_pdbs(mgnify_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> None:
    errors: List[str] = []
    for scaf, rec in manifest.items():
        pdb_path = mgnify_dir / rec["pdb_relpath"]
        err = preprocess_mgnify._verify_one_pdb((scaf, str(pdb_path), rec["core_wt_seq"]))
        if err is not None:
            errors.append(err)
            if len(errors) >= 5:
                break
    if errors:
        raise RuntimeError(f"Verified denovo AF2 PDBs missing or invalid: {errors}")


def _update_wildtypes_csv_gz(mgnify_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> int:
    wt_path = mgnify_dir / "mgnify_wildtypes.csv.gz"
    tmp_path = wt_path.with_name(wt_path.name + ".tmp")
    updated = 0
    with (
        gzip.open(wt_path, "rt", encoding="utf-8", newline="") as fin,
        gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as fout,
    ):
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        idx = {col: i for i, col in enumerate(header)}
        for row in reader:
            src = row[idx["source"]]
            scaf = row[idx["code_wt"]]
            if src == "rocklin_denovo" and scaf in manifest:
                rec = manifest[scaf]
                cur_seq = row[idx["wt_seq"]]
                if cur_seq == rec["core_wt_seq"] and row[idx["pdb_file"]] == rec["pdb_relpath"]:
                    pass
                elif cur_seq == rec["full_wt_seq"]:
                    row[idx["wt_seq"]] = rec["core_wt_seq"]
                    row[idx["pdb_file"]] = rec["pdb_relpath"]
                    row[idx["structure_stem"]] = scaf
                    row[idx["backbone_mutation"]] = ""
                else:
                    raise RuntimeError(f"mgnify_wildtypes.csv.gz sequence mismatch for {scaf}")
                updated += 1
            writer.writerow(row)
    tmp_path.replace(wt_path)
    return updated


def _update_all_mutations_csv_gz(mgnify_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> int:
    muts_path = mgnify_dir / "mgnify_all_mutations.csv.gz"
    tmp_path = muts_path.with_name(muts_path.name + ".tmp")
    updated = 0
    with (
        gzip.open(muts_path, "rt", encoding="utf-8", newline="") as fin,
        gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as fout,
    ):
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        idx = {col: i for i, col in enumerate(header)}
        for row in reader:
            if row[idx["source"]] == "rocklin_denovo":
                scaf = row[idx["code_wt"]]
                if scaf not in manifest:
                    raise RuntimeError(f"Unexpected rocklin_denovo scaffold not in manifest: {scaf}")
                rec = manifest[scaf]
                lp, L = rec["left_pad"], rec["core_len"]
                ws, ms, mt = row[idx["wt_seq"]], row[idx["aa_seq"]], row[idx["mut_type"]]
                if ws == rec["core_wt_seq"] and row[idx["pdb_file"]] == rec["pdb_relpath"]:
                    chk_mt, _ = preprocess_mgnify.diff_to_mut_type(ws, ms)
                    if chk_mt != mt:
                        raise RuntimeError(f"Already-stripped row mismatch on {scaf}: {mt} vs {chk_mt}")
                elif ws == rec["full_wt_seq"]:
                    if ms[:lp] != ws[:lp] or ms[lp + L :] != ws[lp + L :]:
                        raise RuntimeError(f"Flank mutation found on {scaf} ({row[idx['name']]})")
                    core_ms = ms[lp : lp + L]
                    new_mt = shift_mut_type(mt, -lp)
                    chk_mt, _ = preprocess_mgnify.diff_to_mut_type(rec["core_wt_seq"], core_ms)
                    if new_mt != chk_mt:
                        raise RuntimeError(f"Core mut_type mismatch on {scaf}: {new_mt} vs {chk_mt}")
                    row[idx["wt_seq"]] = rec["core_wt_seq"]
                    row[idx["aa_seq"]] = core_ms
                    row[idx["mut_type"]] = new_mt
                    row[idx["pdb_file"]] = rec["pdb_relpath"]
                    row[idx["structure_stem"]] = scaf
                    row[idx["backbone_mutation"]] = ""
                else:
                    raise RuntimeError(f"mgnify_all_mutations.csv.gz wt_seq mismatch on {scaf}")
                updated += 1
            writer.writerow(row)
    tmp_path.replace(muts_path)
    return updated


def _sub_summary(df_sub: pd.DataFrame) -> Dict[str, Any]:
    multis_mask = df_sub["n_mut"] >= 2
    doubles_mask = df_sub["n_mut"] == 2
    gt2_mask = df_sub["n_mut"] >= 3
    has_s = df_sub["has_all_singles"].astype(bool)
    return {
        "total_rows": int(len(df_sub)),
        "unique_scaffolds": int(df_sub["code_wt"].nunique()),
        "n_mut_counts": {str(k): int(v) for k, v in df_sub["n_mut"].value_counts().sort_index().items()},
        "doubles_total": int(doubles_mask.sum()),
        "doubles_with_all_singles": int((doubles_mask & has_s).sum()),
        "doubles_without_all_singles": int((doubles_mask & ~has_s).sum()),
        "higher_order_gt2_total": int(gt2_mask.sum()),
        "higher_order_gt2_with_all_singles": int((gt2_mask & has_s).sum()),
        "higher_order_gt2_without_all_singles": int((gt2_mask & ~has_s).sum()),
        "multis_total": int(multis_mask.sum()),
        "multis_with_all_singles": int((multis_mask & has_s).sum()),
        "multis_without_all_singles": int((multis_mask & ~has_s).sum()),
        "in_range_1_5": int(df_sub["in_range_1_5"].sum()),
        "cens_counts": {str(k): int(v) for k, v in df_sub["cens"].value_counts().sort_index().items()},
        "qc_pass_both": int(df_sub["qc_pass_both"].sum()),
        "source_counts": {str(k): int(v) for k, v in df_sub["source"].value_counts().items()},
        "mgnify_flag_counts": {str(k): int(v) for k, v in df_sub["mgnify"].value_counts().items()},
    }


def cmd_apply(mgnify_dir: Path, allow_missing_pdbs: bool = False) -> Dict[str, Any]:
    mgnify_dir = mgnify_dir.resolve()
    work_dir = mgnify_dir / "af2_rocklin_denovo"
    manifest = load_manifest(work_dir / "manifest.csv")
    present_manifest = {
        scaf: rec for scaf, rec in manifest.items() if (mgnify_dir / rec["pdb_relpath"]).is_file()
    }
    if allow_missing_pdbs:
        _verify_all_manifest_pdbs(mgnify_dir, present_manifest)
        print(
            f"[apply] Pre-filling all {len(manifest):,} scaffold paths "
            f"({len(present_manifest):,}/{len(manifest):,} verified PDBs currently on disk)",
            flush=True,
        )
    else:
        _verify_all_manifest_pdbs(mgnify_dir, manifest)
        print(f"[apply] Verified {len(manifest):,} PDBs under {mgnify_dir / 'structures_af2_denovo'}", flush=True)
    wt_updated = _update_wildtypes_csv_gz(mgnify_dir, manifest)
    muts_updated = _update_all_mutations_csv_gz(mgnify_dir, manifest)
    print(f"  Updated mgnify_wildtypes.csv.gz ({wt_updated:,} rows) and mgnify_all_mutations.csv.gz ({muts_updated:,} rows)", flush=True)

    col_path = mgnify_dir / "mgnify_all_collapsed.csv.gz"
    df = pd.read_csv(
        col_path,
        dtype={
            "uid": str, "source": str, "code_wt": str, "code": str, "chain": str,
            "pdb_file": str, "structure_stem": str, "backbone_mutation": str,
            "mut_type": str, "mut_type_renumbered": str, "mut_type_pdb": str,
            "mut_info": str, "mut_info_seq_pos": str, "lib": str, "pairing_method": str,
            "cens": str, "mgnify_split": str, "wt_split": str, "wt_seq": str, "aa_seq": str,
        },
        keep_default_na=True,
        low_memory=False,
    )
    str_empty_cols = [
        "pdb_file", "structure_stem", "backbone_mutation", "mgnify_split", "wt_split", "lib", "pairing_method"
    ]
    for c in str_empty_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")

    rd_mask = df["source"] == "rocklin_denovo"
    rd_indices = df.index[rd_mask].tolist()
    if not rd_indices:
        raise RuntimeError("No source=='rocklin_denovo' rows found in mgnify_all_collapsed.csv.gz")

    for i in rd_indices:
        scaf = str(df.at[i, "code_wt"])
        if scaf not in manifest:
            raise RuntimeError(f"Unexpected rocklin_denovo scaffold in mgnify_all_collapsed: {scaf}")
        rec = manifest[scaf]
        lp, L = rec["left_pad"], rec["core_len"]
        ws = str(df.at[i, "wt_seq"])
        ms = str(df.at[i, "aa_seq"])
        mt = str(df.at[i, "mut_type"])
        code = str(df.at[i, "code"])

        if ws == rec["core_wt_seq"] and df.at[i, "pdb_file"] == rec["pdb_relpath"]:
            chk_mt, _ = preprocess_mgnify.diff_to_mut_type(ws, ms)
            if chk_mt != mt:
                raise RuntimeError(f"Collapsed already-stripped mismatch on {scaf}: {mt} vs {chk_mt}")
            continue
        if ws != rec["full_wt_seq"]:
            raise RuntimeError(f"Collapsed wt_seq mismatch on {scaf}")
        if ms[:lp] != ws[:lp] or ms[lp + L :] != ws[lp + L :]:
            raise RuntimeError(f"Collapsed flank mutation on {scaf} ({mt})")

        core_ms = ms[lp : lp + L]
        new_mt = shift_mut_type(mt, -lp)
        chk_mt, _ = preprocess_mgnify.diff_to_mut_type(rec["core_wt_seq"], core_ms)
        if new_mt != chk_mt:
            raise RuntimeError(f"Collapsed mut_type diff mismatch on {scaf}: {new_mt} vs {chk_mt}")

        df.at[i, "wt_seq"] = rec["core_wt_seq"]
        df.at[i, "aa_seq"] = core_ms
        df.at[i, "mut_type"] = new_mt
        df.at[i, "mut_type_renumbered"] = new_mt
        df.at[i, "mut_type_pdb"] = new_mt
        df.at[i, "mut_info"] = new_mt
        df.at[i, "mut_info_seq_pos"] = new_mt
        df.at[i, "uid"] = f"{code}_{new_mt}"
        df.at[i, "pdb_file"] = rec["pdb_relpath"]
        df.at[i, "structure_stem"] = scaf
        df.at[i, "backbone_mutation"] = ""

    df["_sort_mgn"] = np.where(df["mgnify"].astype(bool), 0, 1)
    df.sort_values(["_sort_mgn", "source", "code", "n_mut", "mut_type"], inplace=True, kind="mergesort")
    df.drop(columns=["_sort_mgn"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = preprocess_mgnify.annotate_additivity(df)

    # Sanity verification of rocklin_denovo epistasis retention
    rd_after = df[df["source"] == "rocklin_denovo"]
    rd_d_singles = int(((rd_after["n_mut"] == 2) & rd_after["has_all_singles"].astype(bool)).sum())
    rd_t_singles = int(((rd_after["n_mut"] == 3) & rd_after["has_all_singles"].astype(bool)).sum())
    print(f"  rocklin_denovo additivity preserved: doubles w/ singles={rd_d_singles}, triples w/ singles={rd_t_singles}", flush=True)

    tmp_col = col_path.with_name(col_path.name + ".tmp")
    df.to_csv(tmp_col, index=False, compression="gzip")
    tmp_col.replace(col_path)

    mapped_df = df[df["pdb_file"].astype(str) != ""].reset_index(drop=True)
    multis_df = mapped_df[mapped_df["n_mut"] >= 2].reset_index(drop=True)

    mapped_csv = mgnify_dir / "mgnify_mapped.csv"
    multis_csv = mgnify_dir / "mgnify_multis_only.csv"
    tmp_mapped = mapped_csv.with_name(mapped_csv.name + ".tmp")
    tmp_multis = multis_csv.with_name(multis_csv.name + ".tmp")
    mapped_df.to_csv(tmp_mapped, index=False)
    tmp_mapped.replace(mapped_csv)
    multis_df.to_csv(tmp_multis, index=False)
    tmp_multis.replace(multis_csv)

    # Update summary.json
    summary_path = mgnify_dir / "summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    metrics_csv = work_dir / "af2_metrics.csv"
    denovo_summary: Dict[str, Any] = {
        "n_scaffolds": len(manifest),
        "n_structures_on_disk": len(present_manifest),
        "n_variants": int(len(rd_after)),
        "core_len_min": min(r["core_len"] for r in manifest.values()),
        "core_len_median": int(np.median([r["core_len"] for r in manifest.values()])),
        "core_len_max": max(r["core_len"] for r in manifest.values()),
        "doubles_with_all_singles": rd_d_singles,
        "triples_with_all_singles": rd_t_singles,
    }
    if metrics_csv.is_file():
        mdf = pd.read_csv(metrics_csv)
        if not mdf.empty:
            denovo_summary["mean_plddt_median"] = round(float(mdf["mean_plddt"].median()), 2)
            denovo_summary["mean_plddt_min"] = round(float(mdf["mean_plddt"].min()), 2)
            if mdf["ptm"].notna().any():
                denovo_summary["ptm_median"] = round(float(mdf["ptm"].dropna().median()), 4)

    summary["mgnify_all_collapsed"] = _sub_summary(df)
    summary["mgnify_mapped"] = {
        **_sub_summary(mapped_df),
        "unique_structure_stems": int(mapped_df["structure_stem"].nunique()),
        "mutant_fallback_scaffolds": int(
            mapped_df.loc[mapped_df["backbone_mutation"] != "", "code_wt"].nunique()
        ),
        "mutant_fallback_rows": int((mapped_df["backbone_mutation"] != "").sum()),
        "mgnify_split_counts": {
            str(k): int(v) for k, v in mapped_df["mgnify_split"].replace("", np.nan).dropna().value_counts().sort_index().items()
        },
        "lib_counts": {str(k): int(v) for k, v in mapped_df["lib"].value_counts().items()},
    }
    summary["rocklin_denovo_af2"] = denovo_summary

    tmp_sum = summary_path.with_name(summary_path.name + ".tmp")
    tmp_sum.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp_sum.replace(summary_path)

    result = {
        "wildtypes_updated": wt_updated,
        "all_mutations_updated": muts_updated,
        "mgnify_mapped_rows": int(len(mapped_df)),
        "mgnify_multis_only_rows": int(len(multis_df)),
        "rocklin_denovo_af2": denovo_summary,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def cmd_sync(
    mgnify_dir: Path,
    remote_host: str = "subhak-gpu.uwaterloo.ca",
    remote_root: str = "/home/sreeves/esm-msr-devel/af2_rocklin_denovo",
    spot_check_pdbs: int = 5,
) -> Dict[str, Any]:
    """Rsync completed relaxed rank-1 PDBs + score JSONs from cluster and install verified files atomically."""
    mgnify_dir = mgnify_dir.resolve()
    raw_pred_dir = mgnify_dir / "af2_rocklin_denovo" / "raw_predictions"
    raw_pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "--include=*/",
        "--include=*_relaxed_rank_001_*.pdb",
        "--include=*_scores_rank_001_*.json",
        "--exclude=*",
        f"{remote_host}:{remote_root}/predictions/",
        f"{raw_pred_dir}/",
    ]
    subprocess.run(cmd, check=True)
    return cmd_collect(
        mgnify_dir,
        predictions_dir=raw_pred_dir,
        spot_check_pdbs=spot_check_pdbs,
        allow_partial=True,
    )


def main() -> None:
    default_dir = REPO_ROOT / "data" / "mgnify"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Infer core domains and write ColabFold inputs + SLURM script")
    p_prep.add_argument("--mgnify_dir", type=Path, default=default_dir)
    p_prep.add_argument("--remote_root", type=str, default="/home/sreeves/esm-msr-devel/af2_rocklin_denovo")
    p_prep.add_argument("--n_shards", type=int, default=3)

    p_col = sub.add_parser("collect", help="Verify ColabFold relaxed rank-1 PDBs and copy into structures_af2_denovo/")
    p_col.add_argument("--mgnify_dir", type=Path, default=default_dir)
    p_col.add_argument("--predictions_dir", type=Path, default=None)
    p_col.add_argument("--spot_check_pdbs", type=int, default=20)
    p_col.add_argument("--allow_partial", action="store_true")

    p_app = sub.add_parser("apply", help="Update the data/mgnify CSVs with core coords and verified AF2 PDB paths")
    p_app.add_argument("--mgnify_dir", type=Path, default=default_dir)
    p_app.add_argument("--allow_missing_pdbs", action="store_true")

    p_sync = sub.add_parser("sync", help="Rsync completed AF2 structures from cluster and install verified partial set")
    p_sync.add_argument("--mgnify_dir", type=Path, default=default_dir)
    p_sync.add_argument("--remote_host", type=str, default="subhak-gpu.uwaterloo.ca")
    p_sync.add_argument("--remote_root", type=str, default="/home/sreeves/esm-msr-devel/af2_rocklin_denovo")
    p_sync.add_argument("--spot_check_pdbs", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args.mgnify_dir, remote_root=args.remote_root, n_shards=args.n_shards)
    elif args.cmd == "collect":
        cmd_collect(
            args.mgnify_dir,
            predictions_dir=args.predictions_dir,
            spot_check_pdbs=args.spot_check_pdbs,
            allow_partial=args.allow_partial,
        )
    elif args.cmd == "apply":
        cmd_apply(args.mgnify_dir, allow_missing_pdbs=args.allow_missing_pdbs)
    elif args.cmd == "sync":
        cmd_sync(
            args.mgnify_dir,
            remote_host=args.remote_host,
            remote_root=args.remote_root,
            spot_check_pdbs=args.spot_check_pdbs,
        )


if __name__ == "__main__":
    main()
