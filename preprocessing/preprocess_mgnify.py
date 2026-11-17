#!/usr/bin/env python3
"""Preprocess the full MGnify Stability Dataset (Cho, Tsuboyama et al., 2026) for ESM-MSR.

Reads the two Zenodo tables under `<out_dir>/raw/`:
  - mgnify_training_index.csv
  - 230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv

Derives ALL engineered substitution variants (singles k=1, doubles k=2, higher-order k>=3)
with zero homology filtering and retaining ALL multi-mutants even when constituent single
mutants are absent (`has_all_singles == False`, `ddG_additive_ML` / `dddG_ML` = NaN).

Also emits companion tables for indels (`mgnify_indels.csv.gz`), all uncollapsed
substitutions (`mgnify_all_mutations.csv.gz`), and natural close-homolog WT-to-WT
substitution pairs (`mgnify_homolog_pairs.csv`), extracts required AlphaFold2 structures
to 1-indexed Chain-A `.pdb` files, and validates PDB sequences against WT sequences.

Usage:
    CUDA_VISIBLE_DEVICES="" python preprocessing/preprocess_mgnify.py --out_dir data/mgnify --download
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import download_mgnify  # noqa: E402

STANDARD_AAS: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")
DMS7_LINKER = "SAGGSAGGSA"

WT_NAME_RE = re.compile(r"^rocklin_batch2_(\d+)$")
BASE_NAME_RE = re.compile(r"^(rocklin_batch2_\d+)(?:_(.+))?$")
# Critical guard: dms7 _ins*/_del* rows were sliced back to linker-padded fixed length,
# so suffix matching MUST precede sequence-length checks.
INDEL_SUFFIX_RE = re.compile(r"(?:^|_)(?:ins|del)", re.IGNORECASE)

AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _to_float(val: Any) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        x = float(val)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _scaffold_sort_key(scaf: str) -> Tuple[int, str]:
    m = WT_NAME_RE.match(scaf)
    return (int(m.group(1)), scaf) if m else (10**12, scaf)


def trim_dms7_linkers(wt_seq: str, seq: str) -> str:
    """Strip exact 10-AA N/C `SAGGSAGGSA` linkers when dms7 retains padding around WT."""
    pad = len(DMS7_LINKER)
    if (
        len(seq) == len(wt_seq) + 2 * pad
        and seq.startswith(DMS7_LINKER)
        and seq.endswith(DMS7_LINKER)
    ):
        return seq[pad:-pad]
    return seq


def is_dms7_indel(suffix: str, wt_seq: str, mut_seq: str) -> bool:
    """Detect dms7 indels by suffix tag first, then length mismatch vs WT."""
    return bool(INDEL_SUFFIX_RE.search(suffix)) or len(mut_seq) != len(wt_seq)


def diff_to_mut_type(wt_seq: str, mut_seq: str) -> Tuple[str, int]:
    """Return canonical colon-separated `<wt><1pos><mt>` spec sorted by position and mutation count."""
    if len(wt_seq) != len(mut_seq):
        raise ValueError(f"Equal sequence lengths required for substitution diff: {len(wt_seq)} vs {len(mut_seq)}")
    diffs = [
        f"{w_aa}{pos}{m_aa}"
        for pos, (w_aa, m_aa) in enumerate(zip(wt_seq, mut_seq), start=1)
        if w_aa != m_aa
    ]
    return ":".join(diffs), len(diffs)


def compute_backbone_mutation(pdb_seq: str, wt_seq: str) -> str:
    """Return `<pdb_aa><1pos><wt_aa>` spec required by `inference._handle_mutated_backbone(..., assert_wt=True)`."""
    if len(pdb_seq) != len(wt_seq):
        raise ValueError(f"Structure sequence length {len(pdb_seq)} != WT length {len(wt_seq)}")
    diffs = [
        f"{p_aa}{pos}{w_aa}"
        for pos, (p_aa, w_aa) in enumerate(zip(pdb_seq, wt_seq), start=1)
        if p_aa != w_aa
    ]
    return ":".join(diffs)


def annotate_censoring_and_ddg(
    dG_mut: float,
    dG_wt: float,
    mut_ci: float = float("nan"),
    wt_ci: float = float("nan"),
) -> Dict[str, Any]:
    """Compute signed folding stability change (`ddG = dG_mut - dG_wt`), `[-1, 5]` censoring code, and QC flag.

    Convention matches `Tsuboyama2023` / ESM-MSR (`ddG > 0` stabilizing, `ddG < 0` destabilizing):
      - `cens = 0`: both WT and mutant dG lie in `[-1.0, 5.0]` kcal/mol (`in_range_1_5 = True`).
      - `cens = -1`: WT is in range, `dG_mut < -1.0`.
      - `cens = +1`: WT is in range, `dG_mut > 5.0`.
      - `cens = -2`: WT itself is outside `[-1.0, 5.0]`.
    """
    ddG = dG_mut - dG_wt
    wt_in = -1.0 <= dG_wt <= 5.0
    mut_in = -1.0 <= dG_mut <= 5.0
    in_range = bool(wt_in and mut_in)
    if wt_in and mut_in:
        cens = 0
    elif wt_in and dG_mut < -1.0:
        cens = -1
    elif wt_in and dG_mut > 5.0:
        cens = 1
    else:
        cens = -2
    qc_pass = bool(math.isfinite(wt_ci) and math.isfinite(mut_ci) and wt_ci < 1.0 and mut_ci < 1.0)
    return {
        "ddG": ddG,
        "ddG_ML": ddG,
        "in_range_1_5": in_range,
        "cens": cens,
        "qc_pass_both": qc_pass,
    }


def build_pigeonhole_index(
    wt_seqs: Dict[str, str],
    has_struct_fn,
) -> Tuple[Dict[str, Tuple[str, bool]], Dict[Tuple[int, int, str], List[Tuple[str, str, bool]]]]:
    """Build exact-WT lookup and 4-quarter pigeonhole index over WT sequences (`L >= 4`).

    For duplicate WT sequences, keeps the scaffold preferring structure availability, then scaffold ID.
    For multi-scaffold lengths, also keeps one representative per distinct `wt_seq` under the best key.
    """
    best_by_seq: Dict[str, Tuple[str, bool]] = {}
    for scaf, seq in wt_seqs.items():
        if not seq or any(ch not in STANDARD_AAS for ch in seq):
            continue
        has_st = bool(has_struct_fn(scaf, seq))
        prev = best_by_seq.get(seq)
        cand_rank = (0 if has_st else 1, _scaffold_sort_key(scaf))
        if prev is None or cand_rank < (0 if prev[1] else 1, _scaffold_sort_key(prev[0])):
            best_by_seq[seq] = (scaf, has_st)

    q_index: Dict[Tuple[int, int, str], List[Tuple[str, str, bool]]] = collections.defaultdict(list)
    for seq, (scaf, has_st) in best_by_seq.items():
        L = len(seq)
        if L < 4:
            continue
        q = L // 4
        quarters = (seq[0:q], seq[q : 2 * q], seq[2 * q : 3 * q], seq[3 * q :])
        entry = (scaf, seq, has_st)
        for p_i, sl in enumerate(quarters):
            q_index[(L, p_i, sl)].append(entry)

    return best_by_seq, q_index


def match_pigeonhole(
    seq: str,
    exact_wt_by_seq: Dict[str, Tuple[str, bool]],
    q_index: Dict[Tuple[int, int, str], List[Tuple[str, str, bool]]],
    max_hd: int = 3,
) -> Tuple[str, Optional[str], int]:
    """Match an unindexed equal-length sequence against WT library within Hamming distance `<= max_hd`.

    Returns `(status, matched_scaf, hd)` where status is one of:
      - `'wt_identical'` (`hd == 0`)
      - `'matched'` (`1 <= hd <= max_hd`)
      - `'unmatched'` (no match within `max_hd`)
    """
    if seq in exact_wt_by_seq:
        return "wt_identical", exact_wt_by_seq[seq][0], 0
    L = len(seq)
    if L < 4:
        return "unmatched", None, -1

    q = L // 4
    quarters = (seq[0:q], seq[q : 2 * q], seq[2 * q : 3 * q], seq[3 * q :])
    seen: Set[str] = set()
    best_rank: Optional[Tuple[int, int, Tuple[int, str]]] = None
    best_scaf: Optional[str] = None
    best_hd = -1

    for p_i, sl in enumerate(quarters):
        for scaf, wseq, has_st in q_index.get((L, p_i, sl), ()):
            if scaf in seen:
                continue
            seen.add(scaf)
            hd = 0
            for a, b in zip(wseq, seq):
                if a != b:
                    hd += 1
                    if hd > max_hd:
                        break
            if 1 <= hd <= max_hd:
                rank = (hd, 0 if has_st else 1, _scaffold_sort_key(scaf))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_scaf = scaf
                    best_hd = hd

    if best_scaf is None:
        return "unmatched", None, -1
    return "matched", best_scaf, best_hd


def annotate_additivity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute `ddG_additive_ML`, `dddG_ML`, and `has_all_singles` WITHOUT dropping any multi-mutants."""
    out = df.copy()
    singles = out.loc[out["n_mut"] == 1, ["code_wt", "mut_type", "ddG_ML"]]
    single_map: Dict[Tuple[str, str], float] = {
        (cw, mt): float(v)
        for cw, mt, v in zip(singles["code_wt"], singles["mut_type"], singles["ddG_ML"])
        if pd.notna(v)
    }

    has_all = np.zeros(len(out), dtype=bool)
    ddg_add = np.full(len(out), np.nan, dtype=np.float64)
    dddg = np.full(len(out), np.nan, dtype=np.float64)

    code_wts = out["code_wt"].tolist()
    mut_types = out["mut_type"].tolist()
    n_muts = out["n_mut"].tolist()
    ddg_vals = out["ddG_ML"].to_numpy(dtype=np.float64)

    for i, (cw, mt, k, ddg_v) in enumerate(zip(code_wts, mut_types, n_muts, ddg_vals)):
        if k < 2:
            continue
        parts = mt.split(":")
        tot = 0.0
        ok = True
        for sm in parts:
            sv = single_map.get((cw, sm))
            if sv is None:
                ok = False
                break
            tot += sv
        if ok:
            has_all[i] = True
            ddg_add[i] = tot
            dddg[i] = ddg_v - tot

    out["has_all_singles"] = has_all
    out["ddG_additive_ML"] = ddg_add
    out["dddG_ML"] = dddg
    return out


def _load_structure_index(
    index_csv: Path,
) -> Tuple[Dict[str, Tuple[str, str, str, str]], Dict[str, Tuple[str, float, str]]]:
    """Load `mgnify_training_index.csv`.

    Returns:
      - `best_by_pdb[pdb_name] = (chosen_stem, index_seq, chosen_split, wt_split)`
      - `index_wt_info[pdb_name] = (wt_seq, float(dG), wt_split)` for rows where `name == PDB_name`
    """
    best_rank: Dict[str, Tuple[int, str]] = {}
    best_by_pdb: Dict[str, Tuple[str, str, str, str]] = {}
    index_wt_info: Dict[str, Tuple[str, float, str]] = {}
    wt_split_map: Dict[str, str] = {}

    with open(index_csv, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            nm = row["name"]
            pdb_name = row["PDB_name"]
            seq = row["seq"]
            sp = row["split"]
            is_wt = nm == pdb_name
            if is_wt:
                wt_split_map[pdb_name] = sp
                index_wt_info[pdb_name] = (seq, _to_float(row["dG"]), sp)
            rank = (0 if is_wt else 1, nm)
            prev = best_rank.get(pdb_name)
            if prev is None or rank < prev:
                best_rank[pdb_name] = rank
                best_by_pdb[pdb_name] = (nm, seq, sp, sp)

    for pdb_name, (stem, seq, sp, _) in list(best_by_pdb.items()):
        best_by_pdb[pdb_name] = (stem, seq, sp, wt_split_map.get(pdb_name, sp))

    return best_by_pdb, index_wt_info


def _build_homolog_pairs(
    wt_struct_records: List[Tuple[str, str, str, str, str, str, float, float, str]],
    out_csv: Path,
    workers: int,
    max_hd: int = 5,
    max_cluster_expand: int = 400,
) -> Dict[str, int]:
    """Run MMseqs2 `easy-linclust` (`--min-seq-id 0.90 -c 0.95 --cov-mode 0`) on structure-backed WTs
    and extract all equal-length close-homolog WT-to-WT substitution pairs with `1 <= hd <= max_hd`.

    `wt_struct_records` items:
      `(scaf, wt_seq, struct_stem, pdb_file, bb_mut, mgnify_split, wt_split, dG_wt, wt_ci)`
    """
    by_scaf = {rec[0]: rec for rec in wt_struct_records}
    clusters: Dict[str, List[str]] = collections.defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="mgnify_mmseqs_") as tmpd:
        tmp_path = Path(tmpd)
        fasta = tmp_path / "wt.fasta"
        with open(fasta, "w", encoding="utf-8") as fh:
            for rec in wt_struct_records:
                fh.write(f">{rec[0]}\n{rec[1]}\n")

        clu_prefix = tmp_path / "clu"
        mmseqs_tmp = tmp_path / "tmp"
        mmseqs_tmp.mkdir()
        cmd = [
            "mmseqs",
            "easy-linclust",
            str(fasta),
            str(clu_prefix),
            str(mmseqs_tmp),
            "--min-seq-id",
            "0.90",
            "-c",
            "0.95",
            "--cov-mode",
            "0",
            "--threads",
            str(max(1, workers)),
            "-v",
            "1",
        ]
        subprocess.run(cmd, check=True)

        tsv = tmp_path / "clu_cluster.tsv"
        with open(tsv, "r", encoding="utf-8") as fh:
            for line in fh:
                rep, mem = line.rstrip("\n").split("\t", 1)
                clusters[rep].append(mem)

    n_clusters_ge2 = 0
    n_pairs = 0
    n_in_range = 0
    n_large_clusters_capped = 0
    hd_counts: collections.Counter = collections.Counter()
    needed_stems: Set[str] = set()

    header = [
        "uid", "code_wt", "code", "partner_code_wt", "chain", "pdb_file", "structure_stem",
        "backbone_mutation", "mut_type", "mut_type_renumbered", "mut_type_pdb",
        "mut_info", "mut_info_seq_pos", "n_mut", "dG_wt", "dG_ML", "ddG", "ddG_ML",
        "wt_CI", "mut_CI", "qc_pass_both", "in_range_1_5", "cens",
        "mgnify_split", "wt_split", "partner_split", "wt_seq", "aa_seq",
    ]
    tmp_out = out_csv.with_name(out_csv.name + ".tmp")
    with open(tmp_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)

        for rep, members in clusters.items():
            if len(members) < 2:
                continue
            n_clusters_ge2 += 1
            by_len: Dict[int, List[str]] = collections.defaultdict(list)
            for m in members:
                rec = by_scaf.get(m)
                if rec is not None:
                    by_len[len(rec[1])].append(m)

            for _L, scafs in by_len.items():
                if len(scafs) < 2:
                    continue
                scafs.sort(key=_scaffold_sort_key)
                if len(scafs) > max_cluster_expand:
                    n_large_clusters_capped += 1
                    pair_iter = ((rep, s) for s in scafs if s != rep and len(by_scaf[rep][1]) == _L)
                else:
                    pair_iter = (
                        (scafs[i], scafs[j])
                        for i in range(len(scafs))
                        for j in range(i + 1, len(scafs))
                    )

                for s_a, s_b in pair_iter:
                    ra, rb = by_scaf[s_a], by_scaf[s_b]
                    seq_a, seq_b = ra[1], rb[1]
                    hd = 0
                    for ca, cb in zip(seq_a, seq_b):
                        if ca != cb:
                            hd += 1
                            if hd > max_hd:
                                break
                    if hd == 0 or hd > max_hd:
                        continue

                    a_in = -1.0 <= ra[7] <= 5.0
                    b_in = -1.0 <= rb[7] <= 5.0
                    if b_in and not a_in:
                        ref_r, var_r = rb, ra
                    else:
                        ref_r, var_r = ra, rb

                    r_scaf, r_seq, r_stem, r_pdb, r_bb, r_sp, r_wsp, r_dg, r_ci = ref_r
                    v_scaf, v_seq, _vs, _vp, _vb, v_sp, _vw, v_dg, v_ci = var_r
                    if not (math.isfinite(r_dg) and math.isfinite(v_dg)):
                        continue

                    mut_type, n_mut = diff_to_mut_type(r_seq, v_seq)
                    ann = annotate_censoring_and_ddg(v_dg, r_dg, v_ci, r_ci)
                    code = r_scaf if not r_bb else f"{r_scaf}_{r_bb.replace(':', '_')}"
                    uid = f"{code}_to_{v_scaf}_{mut_type}"
                    w.writerow([
                        uid, r_scaf, code, v_scaf, "A", r_pdb, r_stem, r_bb,
                        mut_type, mut_type, mut_type, mut_type, mut_type, n_mut,
                        f"{r_dg:.6f}", f"{v_dg:.6f}", f"{ann['ddG']:.6f}", f"{ann['ddG_ML']:.6f}",
                        f"{r_ci:.6f}" if math.isfinite(r_ci) else "",
                        f"{v_ci:.6f}" if math.isfinite(v_ci) else "",
                        ann["qc_pass_both"], ann["in_range_1_5"], ann["cens"],
                        r_sp, r_wsp, v_sp, r_seq, v_seq,
                    ])
                    needed_stems.add(r_stem)
                    n_pairs += 1
                    hd_counts[str(n_mut)] += 1
                    if ann["in_range_1_5"]:
                        n_in_range += 1

    tmp_out.replace(out_csv)
    return {
        "n_clusters_ge2": n_clusters_ge2,
        "n_large_clusters_capped": n_large_clusters_capped,
        "n_pairs": n_pairs,
        "n_in_range": n_in_range,
        "hd_counts": dict(sorted(hd_counts.items())),
        "needed_stems": needed_stems,
    }


def parse_pdb_ca_seq_and_resids(pdb_path: str) -> Tuple[str, List[int], Set[str]]:
    """Fast CA-record scan verifying Chain ID, 1-based residue numbers, and 1-letter sequence."""
    seq_chars: List[str] = []
    resids: List[int] = []
    chains: Set[str] = set()
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                chains.add(line[21])
                resname = line[17:20].strip()
                seq_chars.append(AA3_TO_1.get(resname, "X"))
                resids.append(int(line[22:26]))
    return "".join(seq_chars), resids, chains


def _verify_one_pdb(task: Tuple[str, str, str]) -> Optional[str]:
    stem, pdb_path, expected_seq = task
    if not os.path.isfile(pdb_path):
        return f"{stem}: missing file {pdb_path}"
    try:
        pdb_seq, resids, chains = parse_pdb_ca_seq_and_resids(pdb_path)
    except Exception as exc:
        return f"{stem}: parse error ({exc})"
    if chains != {"A"}:
        return f"{stem}: unexpected chains {sorted(chains)}"
    expected_resids = list(range(1, len(expected_seq) + 1))
    if resids != expected_resids:
        return f"{stem}: residue numbering mismatch (len {len(resids)} vs {len(expected_resids)})"
    if pdb_seq != expected_seq:
        return f"{stem}: sequence mismatch ({pdb_seq} != {expected_seq})"
    return None


def _spot_check_with_esm_proteinchain(
    out_dir: Path,
    mapped_df: pd.DataFrame,
    n_spot: int,
) -> Dict[str, Any]:
    """Verify `ProteinChain.from_pdb(..., is_predicted=True)`, `_handle_mutated_backbone`,
    and `get_pdb_to_seq_mapping` on a stratified sample covering mutant-fallback and multi-mutant scaffolds.
    """
    import torch
    from esm.utils.structure.protein_chain import ProteinChain
    from esm_msr import inference, preprocess

    unique_scafs = mapped_df[
        ["code_wt", "pdb_file", "backbone_mutation", "wt_seq", "mut_type"]
    ].drop_duplicates(subset=["code_wt", "pdb_file"])

    bb_rows = unique_scafs[unique_scafs["backbone_mutation"].fillna("") != ""]
    multi_rows = mapped_df[mapped_df["n_mut"] >= 2][
        ["code_wt", "pdb_file", "backbone_mutation", "wt_seq", "mut_type"]
    ].drop_duplicates(subset=["code_wt", "pdb_file"])
    other_rows = unique_scafs[unique_scafs["backbone_mutation"].fillna("") == ""]

    parts = []
    if not bb_rows.empty:
        parts.append(bb_rows.sample(n=min(60, len(bb_rows)), random_state=42))
    if not multi_rows.empty:
        parts.append(multi_rows.sample(n=min(60, len(multi_rows)), random_state=42))
    rem = max(0, n_spot - sum(len(p) for p in parts))
    if rem > 0 and not other_rows.empty:
        parts.append(other_rows.sample(n=min(rem, len(other_rows)), random_state=42))

    sample_df = (
        pd.concat(parts, axis=0).drop_duplicates(subset=["code_wt", "pdb_file"])
        if parts else unique_scafs.head(0)
    )

    n_checked = 0
    n_bb_checked = 0
    errors: List[str] = []

    for row in sample_df.itertuples(index=False):
        pdb_abs = str((out_dir / row.pdb_file).resolve())
        pc = ProteinChain.from_pdb(pdb_abs, "A", is_predicted=True)
        coords, plddt, residue_index = pc.to_structure_encoder_inputs()
        # atom37 indices 0, 1, 2, 4 are N, CA, C, O (index 3 is CB, absent on Gly)
        if not np.isfinite(pc.atom37_positions[:, [0, 1, 2, 4], :]).all():
            errors.append(f"{row.code_wt}: non-finite backbone coordinates")
            continue
        pl_np = plddt.detach().cpu().numpy()
        if not np.isfinite(pl_np).all() or float(pl_np.min()) < 0.0 or float(pl_np.max()) > 100.0:
            errors.append(f"{row.code_wt}: pLDDT out of [0, 100] range ({pl_np.min()}..{pl_np.max()})")
            continue

        bb = row.backbone_mutation if pd.notna(row.backbone_mutation) and row.backbone_mutation else None
        dummy_struct = torch.zeros((1, len(pc.sequence) + 2), dtype=torch.long)
        corr_seq, _, _ = inference._handle_mutated_backbone(
            pc.sequence, coords, dummy_struct, backbone_mutation=bb, assert_wt=True
        )
        if corr_seq != row.wt_seq:
            errors.append(f"{row.code_wt}: corrected seq != wt_seq")
            continue

        res_map, aa_map = preprocess.get_pdb_to_seq_mapping(pdb_abs, "A", pc.sequence)
        if len(res_map) != len(pc.sequence) or len(aa_map) != len(pc.sequence):
            errors.append(f"{row.code_wt}: get_pdb_to_seq_mapping length mismatch")
            continue

        n_checked += 1
        if bb is not None:
            n_bb_checked += 1

    return {
        "n_spot_checked": n_checked,
        "n_backbone_fallback_checked": n_bb_checked,
        "errors": errors[:10],
        "n_errors": len(errors),
    }


def preprocess_mgnify(
    out_dir: Path,
    *,
    do_download: bool = False,
    all_wt_structures: bool = False,
    workers: int = 8,
    skip_homolog_pairs: bool = False,
    spot_check_pdbs: int = 200,
) -> Dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_csv = out_dir / "raw" / "mgnify_training_index.csv"
    concat_csv = out_dir / "raw" / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv"
    if do_download and (not index_csv.is_file() or not concat_csv.is_file()):
        download_mgnify.download_tables(out_dir)
    if not index_csv.is_file() or not concat_csv.is_file():
        raise FileNotFoundError(
            f"Raw tables missing under {out_dir / 'raw'}. Run preprocessing/download_mgnify.py --tables_only first."
        )

    print("[1/7] Loading AlphaFold2 structure index ...", flush=True)
    best_by_pdb, index_wt_info = _load_structure_index(index_csv)
    print(f"  Indexed scaffolds: {len(best_by_pdb):,} (with WT index row: {len(index_wt_info):,})", flush=True)

    print("[2/7] Scanning concat260429.csv for MGnify WT replicates ...", flush=True)
    wt_seq_by_scaf: Dict[str, str] = {}
    wt_train_name: Dict[str, str] = {}
    wt_concat_split: Dict[str, str] = {}
    wt_sums: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # dG, ci, t, c, n
    dms4_by_train_name: Dict[str, List[Tuple[str, float, float, float, float, str]]] = collections.defaultdict(list)
    scafs_needed_from_variants: Set[str] = set()
    raw_lib_counts: collections.Counter = collections.Counter()

    with open(concat_csv, "r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            lib = row["lib"]
            mgn = row["mgnify"]
            raw_lib_counts[f"{lib}_mgnify={mgn}"] += 1
            if mgn != "True":
                continue
            nm = row["name"]
            m = BASE_NAME_RE.match(nm)
            if m and m.group(2) is None:
                scaf = m.group(1)
                seq = row["aa_seq"]
                wt_seq_by_scaf[scaf] = seq
                tn = row["train_name"]
                if tn and WT_NAME_RE.match(tn):
                    wt_train_name[scaf] = tn
                    if lib == "dms4":
                        dms4_by_train_name[tn].append((
                            seq,
                            _to_float(row["deltaG"]),
                            _to_float(row["deltaG_95CI"]),
                            _to_float(row["deltaG_t"]),
                            _to_float(row["deltaG_c"]),
                            row["split"],
                        ))
                if row["split"] and scaf not in wt_concat_split:
                    wt_concat_split[scaf] = row["split"]
                dg = _to_float(row["deltaG"])
                if math.isfinite(dg):
                    acc = wt_sums[scaf]
                    acc[0] += dg
                    ci = _to_float(row["deltaG_95CI"])
                    if math.isfinite(ci):
                        acc[1] += ci
                    dgt = _to_float(row["deltaG_t"])
                    if math.isfinite(dgt):
                        acc[2] += dgt
                    dgc = _to_float(row["deltaG_c"])
                    if math.isfinite(dgc):
                        acc[3] += dgc
                    acc[4] += 1
            elif m and lib == "dms7":
                scafs_needed_from_variants.add(m.group(1))
            elif lib == "dms5" and row["train_name"]:
                m_tn = BASE_NAME_RE.match(row["train_name"])
                if m_tn:
                    scafs_needed_from_variants.add(m_tn.group(1))

    # Build final WT registry (dG_wt, wt_ci, dgt, dgc, n_wt_rep)
    wt_dG: Dict[str, float] = {}
    wt_CI: Dict[str, float] = {}
    wt_split_fallback_src: Dict[str, str] = {}
    for scaf, (sdg, sci, _st, _sc, cnt) in wt_sums.items():
        if cnt > 0:
            wt_dG[scaf] = sdg / cnt
            wt_CI[scaf] = sci / cnt

    rescued_wts: List[str] = []
    for scaf in sorted(scafs_needed_from_variants):
        if scaf in wt_seq_by_scaf and scaf in wt_dG:
            continue
        idx_info = index_wt_info.get(scaf)
        tn_rows = [r for r in dms4_by_train_name.get(scaf, ()) if idx_info is None or r[0] == idx_info[0]]
        if tn_rows and any(math.isfinite(r[1]) for r in tn_rows):
            valid_tn = [r for r in tn_rows if math.isfinite(r[1])]
            wt_seq_by_scaf[scaf] = valid_tn[0][0]
            wt_dG[scaf] = float(np.mean([r[1] for r in valid_tn]))
            cis = [r[2] for r in valid_tn if math.isfinite(r[2])]
            wt_CI[scaf] = float(np.mean(cis)) if cis else float("nan")
            wt_split_fallback_src[scaf] = valid_tn[0][5]
            rescued_wts.append(f"{scaf}:dms4_train_name")
        elif idx_info is not None and math.isfinite(idx_info[1]):
            wt_seq_by_scaf[scaf] = idx_info[0]
            wt_dG[scaf] = idx_info[1]
            wt_CI[scaf] = float("nan")
            wt_split_fallback_src[scaf] = idx_info[2]
            rescued_wts.append(f"{scaf}:training_index")

    print(
        f"  WT scaffolds: {len(wt_seq_by_scaf):,} with finite dG_wt: {len(wt_dG):,} "
        f"(rescued missing dms7/dms5 WTs: {rescued_wts})",
        flush=True,
    )

    def resolve_structure(scaf: str, wt_seq: str) -> Optional[Tuple[str, str, str, str, str, str]]:
        pdb_scaf: Optional[str] = None
        if scaf in best_by_pdb:
            pdb_scaf = scaf
        else:
            tn = wt_train_name.get(scaf)
            if tn and tn in best_by_pdb:
                pdb_scaf = tn
        if pdb_scaf is None:
            return None
        chosen_stem, index_seq, sp, wsp = best_by_pdb[pdb_scaf]
        if len(index_seq) != len(wt_seq):
            return None
        bb_mut = compute_backbone_mutation(index_seq, wt_seq)
        pdb_rel = download_mgnify.structure_rel_path(chosen_stem)
        return pdb_scaf, chosen_stem, pdb_rel, bb_mut, sp, wsp

    # Cache resolved structure metadata per scaffold
    struct_cache: Dict[str, Optional[Tuple[str, str, str, str, str, str]]] = {}
    for scaf, wseq in wt_seq_by_scaf.items():
        if scaf in wt_dG:
            struct_cache[scaf] = resolve_structure(scaf, wseq)

    def has_struct(scaf: str, _seq: str) -> bool:
        return struct_cache.get(scaf) is not None

    print("[3/7] Building 4-quarter pigeonhole index for unindexed dms5 rows ...", flush=True)
    exact_wt_by_seq, q_index = build_pigeonhole_index(
        {s: wt_seq_by_scaf[s] for s in wt_dG if s in wt_seq_by_scaf},
        has_struct,
    )

    print("[4/7] Streaming variants from concat260429.csv -> indels + all_mutations + collapsed buffer ...", flush=True)
    indels_path = out_dir / "mgnify_indels.csv.gz"
    all_muts_path = out_dir / "mgnify_all_mutations.csv.gz"
    indel_tmp = indels_path.with_name(indels_path.name + ".tmp")
    all_tmp = all_muts_path.with_name(all_muts_path.name + ".tmp")

    counts = collections.Counter()
    pigeonhole_counts = collections.Counter()

    # Replicate collapse state for structure-backed substitution rows:
    # (code_wt, mut_type) -> [sum_dg, sum_ci, sum_t, n_t, sum_c, n_c, n_rep, any_qc, set(libs), aa_seq, pairing_method]
    collapsed: Dict[Tuple[str, str], List[Any]] = {}
    uncollapsed_total = 0
    uncollapsed_no_struct = 0

    all_header = [
        "name", "lib", "pairing_method", "code_wt", "code", "chain", "pdb_file", "structure_stem",
        "backbone_mutation", "mut_type", "n_mut", "dG_wt", "dG_ML", "ddG", "ddG_ML",
        "wt_CI", "deltaG_95CI", "deltaG_t", "deltaG_c", "qc_pass_both", "in_range_1_5", "cens",
        "mgnify_split", "wt_split", "wt_seq", "aa_seq",
    ]
    indel_header = [
        "name", "lib", "code_wt", "code", "chain", "pdb_file", "structure_stem",
        "backbone_mutation", "indel_spec", "wt_len", "mut_len", "dG_wt", "dG_ML", "ddG", "ddG_ML",
        "wt_CI", "deltaG_95CI", "qc_pass_both", "in_range_1_5", "cens",
        "mgnify_split", "wt_split", "wt_seq", "aa_seq",
    ]
    indel_stems: Set[str] = set()

    with (
        open(concat_csv, "r", encoding="utf-8", errors="replace") as fh,
        gzip.open(all_tmp, "wt", encoding="utf-8", newline="") as f_all,
        gzip.open(indel_tmp, "wt", encoding="utf-8", newline="") as f_ind,
    ):
        w_all = csv.writer(f_all)
        w_ind = csv.writer(f_ind)
        w_all.writerow(all_header)
        w_ind.writerow(indel_header)

        for row in csv.DictReader(fh):
            if row["mgnify"] != "True":
                continue
            lib = row["lib"]
            if lib not in ("dms5", "dms7"):
                continue

            nm = row["name"]
            dg_mut = _to_float(row["deltaG"])
            if not math.isfinite(dg_mut):
                counts[f"{lib}_nonfinite_dG"] += 1
                continue

            ci_mut = _to_float(row["deltaG_95CI"])
            dgt_mut = _to_float(row["deltaG_t"])
            dgc_mut = _to_float(row["deltaG_c"])
            seq_raw = row["aa_seq"]

            scaf: Optional[str] = None
            pairing_method = ""
            wt_seq = ""
            mut_seq = seq_raw

            if lib == "dms7":
                m = BASE_NAME_RE.match(nm)
                if not m:
                    counts["dms7_unrecognized_name"] += 1
                    continue
                if m.group(2) is None:
                    continue  # exact WT already handled
                scaf = m.group(1)
                suffix = m.group(2)
                if scaf not in wt_seq_by_scaf or scaf not in wt_dG:
                    counts["dms7_no_wt"] += 1
                    continue
                wt_seq = wt_seq_by_scaf[scaf]
                mut_seq = trim_dms7_linkers(wt_seq, seq_raw)
                st_info = struct_cache.get(scaf)
                dg_wt = wt_dG[scaf]
                wt_ci = wt_CI.get(scaf, float("nan"))

                if is_dms7_indel(suffix, wt_seq, mut_seq):
                    ann = annotate_censoring_and_ddg(dg_mut, dg_wt, ci_mut, wt_ci)
                    if st_info is not None:
                        _, s_stem, pdb_rel, bb_mut, sp, wsp = st_info
                        code = scaf if not bb_mut else f"{scaf}_{bb_mut.replace(':', '_')}"
                        indel_stems.add(s_stem)
                    else:
                        s_stem, pdb_rel, bb_mut, sp, wsp, code = (
                            "", "", "", wt_concat_split.get(scaf, ""), wt_concat_split.get(scaf, ""), scaf
                        )
                    w_ind.writerow([
                        nm, lib, scaf, code, "A", pdb_rel, s_stem, bb_mut,
                        suffix, len(wt_seq), len(mut_seq),
                        f"{dg_wt:.6f}", f"{dg_mut:.6f}", f"{ann['ddG']:.6f}", f"{ann['ddG_ML']:.6f}",
                        f"{wt_ci:.6f}" if math.isfinite(wt_ci) else "",
                        f"{ci_mut:.6f}" if math.isfinite(ci_mut) else "",
                        ann["qc_pass_both"], ann["in_range_1_5"], ann["cens"],
                        sp, wsp, wt_seq, mut_seq,
                    ])
                    counts["dms7_indels"] += 1
                    continue
                pairing_method = "dms7_name"

            else:  # lib == "dms5"
                tn = row["train_name"]
                if tn:
                    m_tn = BASE_NAME_RE.match(tn)
                    if not m_tn:
                        counts["dms5_bad_train_name"] += 1
                        continue
                    scaf = m_tn.group(1)
                    if scaf not in wt_seq_by_scaf or scaf not in wt_dG:
                        counts["dms5_idx_no_wt"] += 1
                        continue
                    wt_seq = wt_seq_by_scaf[scaf]
                    if len(mut_seq) != len(wt_seq):
                        counts["dms5_idx_len_mismatch"] += 1
                        continue
                    pairing_method = "dms5_train_name"
                else:
                    status, matched_scaf, hd = match_pigeonhole(mut_seq, exact_wt_by_seq, q_index, max_hd=3)
                    if status == "wt_identical":
                        pigeonhole_counts["wt_identical"] += 1
                        counts["dms5_wt_identical"] += 1
                        continue
                    if status == "unmatched" or matched_scaf is None:
                        pigeonhole_counts["unmatched_hd_ge_4"] += 1
                        counts["dms5_unindexed_unmatched"] += 1
                        continue
                    scaf = matched_scaf
                    wt_seq = wt_seq_by_scaf[scaf]
                    pairing_method = f"pigeonhole_hd{hd}"
                    pigeonhole_counts[f"hd_{hd}"] += 1

            if any(ch not in STANDARD_AAS for ch in mut_seq) or any(ch not in STANDARD_AAS for ch in wt_seq):
                counts[f"{lib}_nonstandard_aa"] += 1
                continue

            mut_type, n_mut = diff_to_mut_type(wt_seq, mut_seq)
            if n_mut == 0:
                counts[f"{lib}_wt_identical"] += 1
                continue

            dg_wt = wt_dG[scaf]
            wt_ci = wt_CI.get(scaf, float("nan"))
            ann = annotate_censoring_and_ddg(dg_mut, dg_wt, ci_mut, wt_ci)

            st_info = struct_cache.get(scaf)
            if st_info is not None:
                _, s_stem, pdb_rel, bb_mut, sp, wsp = st_info
                code = scaf if not bb_mut else f"{scaf}_{bb_mut.replace(':', '_')}"
            else:
                s_stem, pdb_rel, bb_mut = "", "", ""
                sp = row["split"] or wt_concat_split.get(scaf, "")
                wsp = wt_concat_split.get(scaf, "")
                code = scaf
                uncollapsed_no_struct += 1

            uncollapsed_total += 1
            counts[f"{lib}_sub_n{n_mut if n_mut < 3 else '3plus'}"] += 1

            w_all.writerow([
                nm, lib, pairing_method, scaf, code, "A", pdb_rel, s_stem, bb_mut,
                mut_type, n_mut, f"{dg_wt:.6f}", f"{dg_mut:.6f}", f"{ann['ddG']:.6f}", f"{ann['ddG_ML']:.6f}",
                f"{wt_ci:.6f}" if math.isfinite(wt_ci) else "",
                f"{ci_mut:.6f}" if math.isfinite(ci_mut) else "",
                f"{dgt_mut:.6f}" if math.isfinite(dgt_mut) else "",
                f"{dgc_mut:.6f}" if math.isfinite(dgc_mut) else "",
                ann["qc_pass_both"], ann["in_range_1_5"], ann["cens"],
                sp, wsp, wt_seq, mut_seq,
            ])

            if st_info is None:
                continue

            key = (scaf, mut_type)
            acc = collapsed.get(key)
            if acc is None:
                collapsed[key] = [
                    dg_mut,
                    ci_mut if math.isfinite(ci_mut) else 0.0,
                    1 if math.isfinite(ci_mut) else 0,
                    dgt_mut if math.isfinite(dgt_mut) else 0.0,
                    1 if math.isfinite(dgt_mut) else 0,
                    dgc_mut if math.isfinite(dgc_mut) else 0.0,
                    1 if math.isfinite(dgc_mut) else 0,
                    1,
                    ann["qc_pass_both"],
                    {lib},
                    mut_seq,
                    pairing_method,
                    n_mut,
                ]
            else:
                acc[0] += dg_mut
                if math.isfinite(ci_mut):
                    acc[1] += ci_mut
                    acc[2] += 1
                if math.isfinite(dgt_mut):
                    acc[3] += dgt_mut
                    acc[4] += 1
                if math.isfinite(dgc_mut):
                    acc[5] += dgc_mut
                    acc[6] += 1
                acc[7] += 1
                acc[8] = acc[8] or ann["qc_pass_both"]
                acc[9].add(lib)

    all_tmp.replace(all_muts_path)
    indel_tmp.replace(indels_path)

    print(
        f"[5/7] Building collapsed dataset ({len(collapsed):,} unique structure-backed variants from "
        f"{uncollapsed_total:,} substitution rows; no-structure rows kept in all_mutations: {uncollapsed_no_struct:,}) ...",
        flush=True,
    )
    rows_out: List[Dict[str, Any]] = []
    mapped_stems: Set[str] = set()
    for (scaf, mut_type), acc in collapsed.items():
        n_rep = acc[7]
        dg_mut = acc[0] / n_rep
        ci_mut = (acc[1] / acc[2]) if acc[2] > 0 else float("nan")
        dgt_mut = (acc[3] / acc[4]) if acc[4] > 0 else float("nan")
        dgc_mut = (acc[5] / acc[6]) if acc[6] > 0 else float("nan")

        dg_wt = wt_dG[scaf]
        wt_ci = wt_CI.get(scaf, float("nan"))
        wt_seq = wt_seq_by_scaf[scaf]
        ann = annotate_censoring_and_ddg(dg_mut, dg_wt, ci_mut, wt_ci)

        _, s_stem, pdb_rel, bb_mut, sp, wsp = struct_cache[scaf]  # type: ignore[misc]
        mapped_stems.add(s_stem)
        code = scaf if not bb_mut else f"{scaf}_{bb_mut.replace(':', '_')}"
        uid = f"{code}_{mut_type}"
        rows_out.append({
            "uid": uid,
            "code_wt": scaf,
            "code": code,
            "chain": "A",
            "pdb_file": pdb_rel,
            "structure_stem": s_stem,
            "backbone_mutation": bb_mut,
            "mut_type": mut_type,
            "mut_type_renumbered": mut_type,
            "mut_type_pdb": mut_type,
            "mut_info": mut_type,
            "mut_info_seq_pos": mut_type,
            "n_mut": acc[12],
            "lib": ";".join(sorted(acc[9])),
            "pairing_method": acc[11],
            "n_replicates": n_rep,
            "dG_wt": round(dg_wt, 6),
            "dG_ML": round(dg_mut, 6),
            "ddG": round(ann["ddG"], 6),
            "ddG_ML": round(ann["ddG_ML"], 6),
            "wt_CI": round(wt_ci, 6) if math.isfinite(wt_ci) else np.nan,
            "deltaG_95CI": round(ci_mut, 6) if math.isfinite(ci_mut) else np.nan,
            "deltaG_t": round(dgt_mut, 6) if math.isfinite(dgt_mut) else np.nan,
            "deltaG_c": round(dgc_mut, 6) if math.isfinite(dgc_mut) else np.nan,
            "qc_pass_both": bool(acc[8]),
            "in_range_1_5": ann["in_range_1_5"],
            "cens": ann["cens"],
            "mgnify_split": sp,
            "wt_split": wsp,
            "wt_seq": wt_seq,
            "aa_seq": acc[10],
        })

    collapsed.clear()
    mapped_df = pd.DataFrame(rows_out)
    del rows_out
    mapped_df.sort_values(["code", "n_mut", "mut_type"], inplace=True, kind="mergesort")
    mapped_df.reset_index(drop=True, inplace=True)
    mapped_df = annotate_additivity(mapped_df)

    mapped_csv = out_dir / "mgnify_mapped.csv"
    multis_csv = out_dir / "mgnify_multis_only.csv"
    tmp_mapped = mapped_csv.with_name(mapped_csv.name + ".tmp")
    tmp_multis = multis_csv.with_name(multis_csv.name + ".tmp")
    mapped_df.to_csv(tmp_mapped, index=False)
    tmp_mapped.replace(mapped_csv)
    multis_df = mapped_df[mapped_df["n_mut"] >= 2]
    multis_df.to_csv(tmp_multis, index=False)
    tmp_multis.replace(multis_csv)
    print(f"  Wrote {mapped_csv} ({len(mapped_df):,} rows) and {multis_csv} ({len(multis_df):,} rows)", flush=True)

    homolog_stats: Dict[str, Any] = {"skipped": True, "n_pairs": 0, "needed_stems": set()}
    homolog_csv = out_dir / "mgnify_homolog_pairs.csv"
    if not skip_homolog_pairs:
        print("[6/7] Finding close-homolog WT-to-WT substitution pairs with MMseqs2 ...", flush=True)
        wt_struct_records = []
        for scaf, st_info in struct_cache.items():
            if st_info is None:
                continue
            _, s_stem, pdb_rel, bb_mut, sp, wsp = st_info
            wt_struct_records.append((
                scaf,
                wt_seq_by_scaf[scaf],
                s_stem,
                pdb_rel,
                bb_mut,
                sp,
                wsp,
                wt_dG[scaf],
                wt_CI.get(scaf, float("nan")),
            ))
        homolog_stats = _build_homolog_pairs(wt_struct_records, homolog_csv, workers=workers)
        print(
            f"  Wrote {homolog_csv} ({homolog_stats['n_pairs']:,} pairs across "
            f"{homolog_stats['n_clusters_ge2']:,} clusters >= 2)",
            flush=True,
        )

    needed_stems: Set[str] = mapped_stems | set(homolog_stats.get("needed_stems", set())) | indel_stems
    if all_wt_structures:
        needed_stems |= {vals[0] for vals in best_by_pdb.values()}

    stems_txt = out_dir / "needed_stems.txt"
    with open(stems_txt, "w", encoding="utf-8") as fh:
        for st in sorted(needed_stems):
            fh.write(f"{st}\n")

    dl_stats: Dict[str, Any] = {}
    verify_stats: Dict[str, Any] = {}
    if do_download:
        print(f"[7/7] Downloading/extracting {len(needed_stems):,} AF2 structures ...", flush=True)
        dl_stats = download_mgnify.download_structures(out_dir, needed_stems, workers=workers)
        if dl_stats.get("remaining_missing", 0) > 0:
            raise RuntimeError(
                f"{dl_stats['remaining_missing']} required structures missing after all shards: "
                f"{dl_stats.get('sample_missing')}"
            )

        # Build stem -> expected PDB sequence map across all indexed stems needed
        seq_by_stem: Dict[str, str] = {}
        for _, (c_stem, i_seq, _, _) in best_by_pdb.items():
            if c_stem in needed_stems:
                seq_by_stem[c_stem] = i_seq
        # Ensure all needed stems are verified against index sequences (even if any weren't best_by_pdb)
        if len(seq_by_stem) < len(needed_stems):
            with open(index_csv, "r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["name"] in needed_stems and row["name"] not in seq_by_stem:
                        seq_by_stem[row["name"]] = row["seq"]

        print(f"  Verifying CA chains, 1..L numbering, and sequences on all {len(needed_stems):,} PDBs ...", flush=True)
        v_tasks = [
            (st, str(out_dir / download_mgnify.structure_rel_path(st)), seq_by_stem[st])
            for st in sorted(needed_stems)
        ]
        pdb_errors: List[str] = []
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for err in pool.map(_verify_one_pdb, v_tasks, chunksize=256):
                if err is not None:
                    pdb_errors.append(err)
        if pdb_errors:
            raise RuntimeError(f"PDB verification failed on {len(pdb_errors)} files: {pdb_errors[:5]}")

        # Verify every variant row's mut_type matches wt_seq (spot check vector/fast pass)
        print("  Verifying backbone_mutation and mut_type reference residues across mapped scaffolds ...", flush=True)
        scaf_check_rows = mapped_df[
            ["code_wt", "structure_stem", "backbone_mutation", "wt_seq"]
        ].drop_duplicates()
        bb_mismatches = 0
        for r in scaf_check_rows.itertuples(index=False):
            iseq = list(seq_by_stem[r.structure_stem])
            if r.backbone_mutation:
                for tok in str(r.backbone_mutation).split(":"):
                    pos = int(tok[1:-1])
                    if iseq[pos - 1] != tok[0]:
                        bb_mismatches += 1
                    iseq[pos - 1] = tok[-1]
            if "".join(iseq) != r.wt_seq:
                bb_mismatches += 1
        if bb_mismatches:
            raise RuntimeError(f"{bb_mismatches} scaffold backbone_mutation -> wt_seq mismatches detected.")

        spot_stats = (
            _spot_check_with_esm_proteinchain(out_dir, mapped_df, spot_check_pdbs)
            if spot_check_pdbs > 0 else {"n_spot_checked": 0, "n_errors": 0}
        )
        if spot_stats.get("n_errors", 0) > 0:
            raise RuntimeError(f"ProteinChain spot-check failed: {spot_stats['errors']}")

        verify_stats = {
            "pdbs_sequence_verified": len(v_tasks),
            "pdb_verification_errors": 0,
            "scaffold_backbone_mismatches": bb_mismatches,
            **spot_stats,
        }

    # Assemble summary metrics cleanly from mapped_df
    n_mut_counts = {str(k): int(v) for k, v in mapped_df["n_mut"].value_counts().sort_index().items()}
    multis_mask = mapped_df["n_mut"] >= 2
    doubles_mask = mapped_df["n_mut"] == 2
    gt2_mask = mapped_df["n_mut"] >= 3
    has_s = mapped_df["has_all_singles"].astype(bool)

    summary = {
        "raw_lib_counts": dict(raw_lib_counts),
        "pigeonhole_dms5_unindexed": dict(pigeonhole_counts),
        "pipeline_counts": dict(counts),
        "rescued_wts": rescued_wts,
        "uncollapsed_substitution_rows": int(uncollapsed_total),
        "uncollapsed_no_structure_rows": int(uncollapsed_no_struct),
        "mgnify_mapped": {
            "total_rows": int(len(mapped_df)),
            "unique_scaffolds": int(mapped_df["code_wt"].nunique()),
            "unique_structure_stems": int(len(mapped_stems)),
            "mutant_fallback_scaffolds": int(
                mapped_df.loc[mapped_df["backbone_mutation"] != "", "code_wt"].nunique()
            ),
            "mutant_fallback_rows": int((mapped_df["backbone_mutation"] != "").sum()),
            "n_mut_counts": n_mut_counts,
            "doubles_total": int(doubles_mask.sum()),
            "doubles_with_all_singles": int((doubles_mask & has_s).sum()),
            "doubles_without_all_singles": int((doubles_mask & ~has_s).sum()),
            "higher_order_gt2_total": int(gt2_mask.sum()),
            "higher_order_gt2_with_all_singles": int((gt2_mask & has_s).sum()),
            "higher_order_gt2_without_all_singles": int((gt2_mask & ~has_s).sum()),
            "multis_total": int(multis_mask.sum()),
            "multis_with_all_singles": int((multis_mask & has_s).sum()),
            "multis_without_all_singles": int((multis_mask & ~has_s).sum()),
            "in_range_1_5": int(mapped_df["in_range_1_5"].sum()),
            "cens_counts": {str(k): int(v) for k, v in mapped_df["cens"].value_counts().sort_index().items()},
            "qc_pass_both": int(mapped_df["qc_pass_both"].sum()),
            "mgnify_split_counts": {
                str(k): int(v) for k, v in mapped_df["mgnify_split"].value_counts().sort_index().items()
            },
            "lib_counts": {str(k): int(v) for k, v in mapped_df["lib"].value_counts().items()},
        },
        "mgnify_indels": {"total_rows": int(counts["dms7_indels"])},
        "mgnify_homolog_pairs": {
            k: v for k, v in homolog_stats.items() if k != "needed_stems"
        },
        "needed_stems_total": len(needed_stems),
        "structure_download": dl_stats,
        "structure_verification": verify_stats,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)
    return summary


def main() -> None:
    default_out = REPO_ROOT / "data" / "mgnify"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=default_out, help="Target directory (default: data/mgnify)")
    ap.add_argument("--download", action="store_true", help="Download missing raw tables and extract needed AF2 PDBs")
    ap.add_argument("--all_wt_structures", action="store_true", help="Also include all 532,988 index WTs in needed_stems")
    ap.add_argument("--workers", type=int, default=8, help="Worker processes for PDB conversion/verification & MMseqs2")
    ap.add_argument("--skip_homolog_pairs", action="store_true", help="Skip MMseqs2 close-homolog WT-to-WT pairing")
    ap.add_argument("--spot_check_pdbs", type=int, default=200, help="Count of PDBs to spot-check via ProteinChain")
    args = ap.parse_args()

    preprocess_mgnify(
        args.out_dir,
        do_download=args.download,
        all_wt_structures=args.all_wt_structures,
        workers=args.workers,
        skip_homolog_pairs=args.skip_homolog_pairs,
        spot_check_pdbs=args.spot_check_pdbs,
    )


if __name__ == "__main__":
    main()
