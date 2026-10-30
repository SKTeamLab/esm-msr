#!/usr/bin/env bash
# Regenerate everything under analysis/results.  Needs the msr_venv-like environment (numpy, pandas, scipy, matplotlib) and the data
# root (MSR_ROOT, default ~/playground/esm-msr-devel).  CPU only.  The core bootstrap (01) is the slow step (~15 min, one core).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONPATH=.
$PY 00_validate_against_tables.py
$PY 01_core_rho.py
for s in 02_thermostability 03_tsuboyama 04_dms_domainome 05_proteingym 06_epistasis 07_rmse_calibration 08_screening 09_sequence_length 10_leakage_and_reference 11_publication_figures 11d_epistasis_levels_bootstrap 11b_publication_epistasis_levels 11c_publication_epistasis_scatters 12_pair_epistasis_heatmaps; do
  echo "== $s"; $PY $s.py
done
$PY 99_table_coverage.py
