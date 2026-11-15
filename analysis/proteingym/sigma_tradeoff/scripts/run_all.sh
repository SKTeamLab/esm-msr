#!/usr/bin/env bash
# Re-run the whole sigma trade-off analysis. Picks up whatever per-assay CSVs exist in the
# sigma*_wt_only folders (re-run after the OOM-missing assays are added; no code change needed).
# Run from anywhere:  bash run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
export SIGMA_SCRATCH="${SIGMA_SCRATCH:-/tmp/sigma_scratch}"
mkdir -p "$SIGMA_SCRATCH"
cd "$HERE"
$PY 01_load.py "$SIGMA_SCRATCH/data.pkl"        # read all predictions (wt_lora_pred) -> pickle
$PY 02_metrics.py "$SIGMA_SCRATCH/data.pkl"     # per-assay rho, between/within-site, bootstrap
$PY 03_classify.py                              # curve classes + abundance-vs-other verdicts
$PY 04_fig_global.py "$SIGMA_SCRATCH/data.pkl"  # F01-F09
$PY 05_fig_multi.py "$SIGMA_SCRATCH/data.pkl"   # F10-F17, F20
$PY 06_dossiers.py "$SIGMA_SCRATCH/data.pkl"    # one figure per multi-assay protein
$PY 07_fig_substitutions.py "$SIGMA_SCRATCH/data.pkl"  # F18-F19
$PY 08_summary.py                               # summary tables
$PY 10_rank_stats.py "$SIGMA_SCRATCH/data.pkl"   # forward/inverse top-K selection stats
$PY 11_fig_rank.py "$SIGMA_SCRATCH/data.pkl"     # F21-F31
$PY 12_rank_extra.py "$SIGMA_SCRATCH/data.pkl"   # bottom-of-ranking and composition checks
$PY 09_report.py                               # fill REPORT.md tables from ../report_template.md
