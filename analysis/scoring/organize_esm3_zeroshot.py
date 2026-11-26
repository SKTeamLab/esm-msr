#!/usr/bin/env python3
"""Sync, organize, verify, and catalog all ESM3 zero-shot predictions.

Builds a unified, self-documenting catalog under `<MSR_ROOT>/esm3_zeroshot_catalog/` covering:
  1. esm3-small-open (ESM-MSR sigma=0, per_res_plddt=1.0): unmasked, independent, marginal
  2. esm3-small-2024-08 (Forge, independent masking)
  3. esm3-medium-2024-08 (Forge, independent masking)
  4. esm3-large-2024-03 (Forge, independent masking)

Across every dataset in the manuscript analysis:
  - External stability (9 parent sets covering all 12 benchmarks: s571, s783, s2648, s8754,
    s669 [->s461], ssym, q3421, k3822 [->k2369], ptmul [->ptmuld + k>=3])
  - Active DMS3 (DLG4_HUMAN_Faure_2021_binding_domain, GRB2_HUMAN_Faure_2021_binding_domain,
    MYO_HUMAN_Kung_2025_display)
  - Domainome 1.0 (522 domains)
  - MegaScale splits (splits_oct06_capped-{val,test}, mega_splits-{val,test})
  - ProteinGym (217 assays: WT additive + multi-mutant full/marginal)
  - MGnify + Rocklin de novo (976,608 rows across 181,367 scaffolds)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

MSR_ROOT = Path(os.environ.get("MSR_ROOT", "/home/sareeves/playground/esm-msr-devel")).resolve()
CATALOG_ROOT = MSR_ROOT / "esm3_zeroshot_catalog"
OLD_FORGE_ROOT = MSR_ROOT / "baselines" / "public" / "esm3_forge_zero_shot"
ZERO_SHOT_PGYM = MSR_ROOT / "zero_shot"
REMOTE_HOST = "sreeves@subhak-gpu.uwaterloo.ca"
REMOTE_REPO = "/home/sreeves/esm-msr-devel"

EXTERNAL_PARENTS = ["s571", "s783", "s2648", "s8754", "s669", "ssym", "q3421", "k3822", "ptmul"]
DMS3 = [
    "DLG4_HUMAN_Faure_2021_binding_domain",
    "GRB2_HUMAN_Faure_2021_binding_domain",
    "MYO_HUMAN_Kung_2025_display",
]
MEGASCALE_SPLITS = [
    "splits_oct06_capped-val",
    "splits_oct06_capped-test",
    "mega_splits-val",
    "mega_splits-test",
]
FORGE_MODELS = ["esm3-small-2024-08", "esm3-medium-2024-08", "esm3-large-2024-03"]

# Mapping from esm_msr_testing.py runner folder names to canonical dataset filenames
RUNNER_FOLDER_MAP = {
    "s571": ("external", "s571.csv"),
    "s783": ("external", "s783.csv"),
    "s2648": ("external", "s2648.csv"),
    "s8754": ("external", "s8754.csv"),
    "s669": ("external", "s669.csv"),
    "ssym": ("external", "ssym.csv"),
    "q3421": ("external", "q3421.csv"),
    "k3822": ("external", "k3822.csv"),
    "PTMUL": ("external", "ptmul.csv"),
    "ptmul": ("external", "ptmul.csv"),
    "DLG4_HUMAN_Faure_2021_binding_domain": ("dms", "DLG4_HUMAN_Faure_2021_binding_domain.csv"),
    "GRB2_HUMAN_Faure_2021_binding_domain": ("dms", "GRB2_HUMAN_Faure_2021_binding_domain.csv"),
    "MYO_HUMAN_Kung_2025_display": ("dms", "MYO_HUMAN_Kung_2025_display.csv"),
    "domainome": ("domainome", "domainome.csv"),
    "splits_oct06_capped-val": ("megascale", "splits_oct06_capped-val.csv"),
    "splits_oct06_capped-test": ("megascale", "splits_oct06_capped-test.csv"),
    "mega_splits-val": ("megascale", "mega_splits-val.csv"),
    "mega_splits-test": ("megascale", "mega_splits-test.csv"),
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return True
    shutil.copy2(src, dst)
    return True


def rsync_from_cluster(stage_root: Path) -> None:
    """Sync all completed Forge and ESM3-small-open outputs from subhak-gpu (excluding per-domain temp parts)."""
    stage_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (f"{REMOTE_HOST}:{REMOTE_REPO}/outputs/esm3_sigma0_plddt1/", str(stage_root / "esm3_sigma0_plddt1") + "/"),
        (f"{REMOTE_HOST}:{REMOTE_REPO}/outputs/esm3_forge_ind/", str(stage_root / "esm3_forge_ind") + "/"),
    ]
    for src, dst in jobs:
        Path(dst).mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync", "-az", "--info=stats1",
            "--exclude=per_domain/",
            "--exclude=per_pdb/",
            "--exclude=shards/",
            "--exclude=slurm_logs/",
            src, dst,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False)


def _find_single_csv(folder: Path) -> Optional[Path]:
    if not folder.is_dir():
        return None
    hits = sorted(p for p in folder.rglob("*_predictions.csv") if "_smoke" not in p.name)
    return hits[-1] if hits else None


def _is_single_only_pgym(csv_path: Path) -> bool:
    """Return True iff a ProteinGym CSV contains only single-site mutations (marginal == independent)."""
    try:
        df = pd.read_csv(csv_path, usecols=["mutant"])
        return not df["mutant"].astype(str).str.contains(":").any()
    except Exception:
        return False


def _merge_pgym_dir(
    dst_dir: Path,
    src_dirs: List[Path],
    copy_singles_fallback_dir: Optional[Path] = None,
) -> int:
    """Merge per-assay ProteinGym CSVs from one or more source directories (later overrides earlier).

    When ``copy_singles_fallback_dir`` is set, ONLY single-mutant-only assays (where marginal == independent
    identically) are copied from the fallback; multi-mutant assays must come from ``src_dirs``.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    if copy_singles_fallback_dir and copy_singles_fallback_dir.is_dir():
        for f in sorted(copy_singles_fallback_dir.glob("*.csv")):
            if f.name in ("summary.csv", "manifest.csv"):
                continue
            if _is_single_only_pgym(f):
                safe_copy(f, dst_dir / f.name)
    for sd in src_dirs:
        if not sd or not sd.is_dir():
            continue
        for f in sorted(sd.glob("*.csv")):
            if f.name in ("summary.csv", "manifest.csv"):
                continue
            safe_copy(f, dst_dir / f.name)
    assays = [f for f in sorted(dst_dir.glob("*.csv")) if f.name not in ("summary.csv", "manifest.csv")]
    return len(assays)


def organize_esm3_small_open(stage_root: Path, catalog_root: Path) -> None:
    raw = stage_root / "esm3_sigma0_plddt1"
    out_base = catalog_root / "esm3-small-open"
    if not raw.is_dir():
        return

    for strategy in ("unmasked", "independent", "marginal"):
        s_out = out_base / strategy
        # 1. Gamut (splits_oct06_capped + external + DMS + domainome)
        gamut_preds = raw / f"gamut_{strategy}" / "analysis" / "predictions"
        if gamut_preds.is_dir():
            for runner_dir, (cat, fname) in RUNNER_FOLDER_MAP.items():
                if runner_dir.startswith("mega_splits"):
                    continue
                hit = _find_single_csv(gamut_preds / runner_dir)
                if hit:
                    safe_copy(hit, s_out / cat / fname)
        # 2. mega_splits
        mega_preds = raw / f"mega_splits_{strategy}" / "analysis" / "predictions"
        if mega_preds.is_dir():
            for split_dir in ("mega_splits-val", "mega_splits-test"):
                hit = _find_single_csv(mega_preds / split_dir)
                if hit:
                    safe_copy(hit, s_out / "megascale" / f"{split_dir}.csv")
        # 3. MGnify
        mgn_src = raw / f"mgnify_{strategy}"
        if not mgn_src.is_dir() and strategy == "unmasked":
            mgn_src = raw / "mgnify"
        if mgn_src.is_dir():
            for fn in ("predictions.csv.gz", "stats.csv", "summary.json"):
                safe_copy(mgn_src / fn, s_out / "mgnify" / fn)

    # 4. ProteinGym across all 3 masking modes (merging long4 RTX6000 backfill!)
    long4 = raw / "pgym_long4"
    # independent WT (213 + 4 = 217)
    _merge_pgym_dir(
        out_base / "independent" / "proteingym_wt",
        [raw / "pgym_independent", long4 / "pgym_independent"],
    )
    # independent full multi
    _merge_pgym_dir(
        out_base / "independent" / "proteingym_full_multi",
        [raw / "pgym_independent_full_multi"],
    )
    # unmasked WT (213 + 4 = 217)
    _merge_pgym_dir(
        out_base / "unmasked" / "proteingym_wt",
        [raw / "pgym_unmasked_wt", raw / "pgym_unmasked", long4 / "pgym_unmasked"],
    )
    # unmasked multi (WT+MT)
    _merge_pgym_dir(
        out_base / "unmasked" / "proteingym_multi",
        [raw / "pgym_unmasked_multi"],
    )
    # marginal (148 single-only assays are mathematically identical to independent + 69 multi marginal + 4 long)
    marg_pgym_dir = out_base / "marginal" / "proteingym"
    # Remove any non-single fallback files not present in pgym_marginal_multi
    if marg_pgym_dir.is_dir():
        mm_dir = raw / "pgym_marginal_multi"
        mm_names = {p.name for p in mm_dir.glob("*.csv")} if mm_dir.is_dir() else set()
        for existing in list(marg_pgym_dir.glob("*.csv")):
            if existing.name not in mm_names and not _is_single_only_pgym(existing):
                existing.unlink()
    _merge_pgym_dir(
        marg_pgym_dir,
        [raw / "pgym_marginal_multi", long4 / "pgym_independent"],
        copy_singles_fallback_dir=(out_base / "independent" / "proteingym_wt"),
    )


def organize_forge_models(stage_root: Path, catalog_root: Path) -> None:
    for model in FORGE_MODELS:
        old_dir = OLD_FORGE_ROOT / model
        new_dir = stage_root / "esm3_forge_ind" / model
        dst = catalog_root / model

        # 1. External parent sets: existing 8 singles + NEW independent ptmul (fallback to old only if new absent)
        for ext in EXTERNAL_PARENTS:
            if ext == "ptmul":
                if not safe_copy(new_dir / "ptmul" / "ptmul.csv", dst / "external" / "ptmul.csv"):
                    pass  # Do NOT copy legacy random-path ptmul as independent!
            else:
                safe_copy(old_dir / f"{ext}.csv", dst / "external" / f"{ext}.csv")

        # 2. Active DMS3: prefer newly run independent CSV, else existing Forge CSV (doubles == independent)
        for dms in DMS3:
            if not safe_copy(new_dir / "dms" / f"{dms}.csv", dst / "dms" / f"{dms}.csv"):
                safe_copy(old_dir / f"{dms}.csv", dst / "dms" / f"{dms}.csv")

        # 3. Domainome
        if not safe_copy(new_dir / "domainome" / "domainome.csv", dst / "domainome" / "domainome.csv"):
            safe_copy(old_dir / "domainome.csv", dst / "domainome" / "domainome.csv")

        # 4. MegaScale splits
        for sp in MEGASCALE_SPLITS:
            safe_copy(new_dir / sp / f"{sp}.csv", dst / "megascale" / f"{sp}.csv")
        for hyp in ("tsuboyama_hyperopt_val.csv", "tsuboyama_hyperopt_test.csv"):
            safe_copy(old_dir / hyp, dst / "megascale" / hyp)

        # 5. ProteinGym
        _merge_pgym_dir(dst / "proteingym_wt", [new_dir / "pgym_wt_independent"])
        _merge_pgym_dir(dst / "proteingym_full_multi", [new_dir / "pgym_full_independent"])

        # 6. MGnify + Rocklin de novo
        safe_copy(new_dir / "mgnify_rocklin_denovo" / "predictions.csv.gz", dst / "mgnify" / "rocklin_denovo_predictions.csv.gz")
        safe_copy(new_dir / "mgnify_all" / "predictions.csv.gz", dst / "mgnify" / "mgnify_predictions.csv.gz")


def _has_valid_scores(csv_path: Path) -> bool:
    """Return True iff a ProteinGym CSV has at least one finite prediction (filters out HTTP-413 empty files)."""
    try:
        hdr = list(pd.read_csv(csv_path, nrows=0).columns)
        col = next((c for c in ("wt_lora_pred", "esm3_score", "combined_pred") if c in hdr), None)
        if not col:
            return False
        s = pd.read_csv(csv_path, usecols=[col])[col]
        return bool(s.notna().any())
    except Exception:
        return False


def _count_valid_pgym(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for f in d.glob("*.csv") if f.name not in ("summary.csv", "manifest.csv") and _has_valid_scores(f))


def _summarize_csv(path: Path) -> Tuple[int, int, int, float]:
    """Return (n_rows, n_ok, n_nan, spearman) for a single prediction CSV."""
    try:
        hdr = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return (0, 0, 0, float("nan"))
    pred_col = next((c for c in ("esm3_score", "combined_pred", "wt_lora_pred") if c in hdr), None)
    true_col = next((c for c in ("ddG_ML", "ddG", "dTm", "DMS_score", "scaled_fitness", "ddG_dir") if c in hdr), None)
    use = [c for c in (pred_col, true_col) if c]
    if not use:
        df = pd.read_csv(path, usecols=[hdr[0]])
        return (len(df), 0, len(df), float("nan"))
    df = pd.read_csv(path, usecols=use, low_memory=False)
    n = len(df)
    n_ok = int(df[pred_col].notna().sum()) if pred_col else 0
    n_nan = n - n_ok
    rho = float("nan")
    if pred_col and true_col:
        v = df[[pred_col, true_col]].dropna()
        if len(v) > 2 and v[pred_col].nunique() > 1 and v[true_col].nunique() > 1:
            rho = float(spearmanr(v[pred_col], v[true_col]).statistic)
    return (n, n_ok, n_nan, rho)


def audit_public_esm3_parity(pgym_wt_dir: Path) -> Dict[str, float]:
    """Compare our ESM3-small-open (sigma=0, plddt=1.0) independent WT ProteinGym against public ESM3."""
    if not pgym_wt_dir.is_dir() or not ZERO_SHOT_PGYM.is_dir():
        return {}
    sp_mod, pr_mod, d_rho = [], [], []
    for f in sorted(pgym_wt_dir.glob("*.csv")):
        if f.name in ("summary.csv", "manifest.csv"):
            continue
        pub_f = ZERO_SHOT_PGYM / f.name
        if not pub_f.is_file():
            continue
        ours = pd.read_csv(f, usecols=["mutant", "wt_lora_pred"])
        pub = pd.read_csv(pub_f, usecols=["mutant", "DMS_score", "ESM3"])
        m = pub.merge(ours, on="mutant", how="inner").dropna()
        if len(m) < 10:
            continue
        sp_mod.append(float(spearmanr(m["ESM3"], m["wt_lora_pred"]).statistic))
        pr_mod.append(float(pearsonr(m["ESM3"], m["wt_lora_pred"]).statistic))
        d_rho.append(
            float(spearmanr(m["wt_lora_pred"], m["DMS_score"]).statistic)
            - float(spearmanr(m["ESM3"], m["DMS_score"]).statistic)
        )
    if not sp_mod:
        return {}
    return {
        "n_assays_compared": len(sp_mod),
        "median_spearman_vs_public": float(np.median(sp_mod)),
        "mean_spearman_vs_public": float(np.mean(sp_mod)),
        "min_spearman_vs_public": float(np.min(sp_mod)),
        "median_pearson_vs_public": float(np.median(pr_mod)),
        "mean_delta_dms_rho": float(np.mean(d_rho)),
        "median_delta_dms_rho": float(np.median(d_rho)),
    }


def build_manifest_and_report(catalog_root: Path) -> pd.DataFrame:
    """Walk the catalog, compute MD5s/counts/correlations, write MANIFEST.tsv, COVERAGE_MATRIX.csv, README.md."""
    manifest_rows = []
    model_dirs = [
        ("esm3-small-open", "unmasked", catalog_root / "esm3-small-open" / "unmasked"),
        ("esm3-small-open", "independent", catalog_root / "esm3-small-open" / "independent"),
        ("esm3-small-open", "marginal", catalog_root / "esm3-small-open" / "marginal"),
        ("esm3-small-2024-08", "independent", catalog_root / "esm3-small-2024-08"),
        ("esm3-medium-2024-08", "independent", catalog_root / "esm3-medium-2024-08"),
        ("esm3-large-2024-03", "independent", catalog_root / "esm3-large-2024-03"),
    ]

    cov_rows = []
    for model, variant, root in model_dirs:
        ext_ok = sum(1 for e in EXTERNAL_PARENTS if (root / "external" / f"{e}.csv").is_file())
        dms_ok = sum(1 for d in DMS3 if (root / "dms" / f"{d}.csv").is_file())
        dom_ok = int((root / "domainome" / "domainome.csv").is_file())
        ms_ok = sum(1 for s in MEGASCALE_SPLITS if (root / "megascale" / f"{s}.csv").is_file())
        pg_wt_dir = root / ("proteingym" if variant == "marginal" else "proteingym_wt")
        pg_wt_n = _count_valid_pgym(pg_wt_dir)
        pg_mt_dir = root / ("proteingym_multi" if variant == "unmasked" else "proteingym_full_multi")
        pg_mt_n = _count_valid_pgym(pg_mt_dir)
        mgn_files = list((root / "mgnify").glob("*.csv.gz")) if (root / "mgnify").is_dir() else []

        cov_rows.append({
            "model": model,
            "masking": variant,
            "external_9": f"{ext_ok}/9",
            "dms_3": f"{dms_ok}/3",
            "domainome": "1/1" if dom_ok else "0/1",
            "megascale_4": f"{ms_ok}/4",
            "pgym_wt_217": f"{pg_wt_n}/217",
            "pgym_multi_69": f"{pg_mt_n}/69",
            "mgnify_files": len(mgn_files),
        })

        # Manifest entries for scalar benchmark CSVs (external, dms, domainome, megascale, mgnify)
        for cat in ("external", "dms", "domainome", "megascale", "mgnify"):
            cdir = root / cat
            if not cdir.is_dir():
                continue
            for f in sorted(cdir.iterdir()):
                if not f.is_file() or not (f.name.endswith(".csv") or f.name.endswith(".csv.gz")):
                    continue
                n, n_ok, n_nan, rho = _summarize_csv(f) if f.name.endswith(".csv") else (-1, -1, -1, float("nan"))
                manifest_rows.append({
                    "model": model,
                    "masking": variant,
                    "category": cat,
                    "dataset": f.stem.replace(".csv", ""),
                    "rows": n,
                    "n_ok": n_ok,
                    "n_nan": n_nan,
                    "spearman": round(rho, 5) if np.isfinite(rho) else "",
                    "bytes": f.stat().st_size,
                    "md5": md5_file(f),
                    "relpath": str(f.relative_to(catalog_root)),
                })

    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(catalog_root / "COVERAGE_MATRIX.csv", index=False)
    man_df = pd.DataFrame(manifest_rows)
    man_df.to_csv(catalog_root / "MANIFEST.tsv", sep="\t", index=False)

    parity = audit_public_esm3_parity(catalog_root / "esm3-small-open" / "independent" / "proteingym_wt")

    lines = [
        "# ESM3 Zero-Shot Prediction Catalog",
        "",
        "Unified, verified zero-shot predictions for all ESM3 models described in the manuscript analysis:",
        "- **`esm3-small-open`**: evaluated as ESM-MSR at σ = 0 with `per_res_plddt = 1.0` wherever structure coordinates are finite (`src/esm_msr/models.py`), across **all three masking variations** (`unmasked`, `independent`, `marginal`).",
        "- **`esm3-small-2024-08`**, **`esm3-medium-2024-08`**, **`esm3-large-2024-03`**: evaluated via Forge API using unified **independent masking** (`inference_scripts/esm3_zs_testing.py`).",
        "",
        "## Coverage Matrix",
        "",
        cov_df.to_markdown(index=False),
        "",
    ]
    if parity:
        lines.extend([
            "## Public ProteinGym `ESM3` Parity Audit (`esm3-small-open` σ = 0, `per_res_plddt = 1.0`, independent WT)",
            "",
            f"- Assays compared: **{parity['n_assays_compared']} / 217**",
            f"- Median Spearman vs. public `ESM3`: **{parity['median_spearman_vs_public']:.6f}** (mean `{parity['mean_spearman_vs_public']:.6f}`, min `{parity['min_spearman_vs_public']:.6f}`)",
            f"- Median Pearson vs. public `ESM3`: **{parity['median_pearson_vs_public']:.6f}**",
            f"- Mean Δρ vs. `DMS_score` (ours − public): **{parity['mean_delta_dms_rho']:+.6f}** (median `{parity['median_delta_dms_rho']:+.6f}`)",
            "",
        ])
    (catalog_root / "README.md").write_text("\n".join(lines))
    return cov_df


def main():
    parser = argparse.ArgumentParser(description="Sync, organize, and verify all ESM3 zero-shot predictions.")
    parser.add_argument("--no_sync", action="store_true", help="Skip rsync from subhak-gpu")
    parser.add_argument("--stage_root", type=str, default=str(MSR_ROOT / "esm3_zeroshot_stage"))
    parser.add_argument("--catalog_root", type=str, default=str(CATALOG_ROOT))
    args = parser.parse_args()

    stage_root = Path(args.stage_root).resolve()
    catalog_root = Path(args.catalog_root).resolve()
    catalog_root.mkdir(parents=True, exist_ok=True)

    if not args.no_sync:
        rsync_from_cluster(stage_root)
    organize_esm3_small_open(stage_root, catalog_root)
    organize_forge_models(stage_root, catalog_root)
    cov_df = build_manifest_and_report(catalog_root)
    print("\n=== ESM3 ZERO-SHOT COVERAGE MATRIX ===")
    print(cov_df.to_string(index=False))


if __name__ == "__main__":
    main()
