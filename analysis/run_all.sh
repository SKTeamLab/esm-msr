#!/usr/bin/env bash
# Regenerate everything under analysis/results.  Needs the msr_venv-like environment (numpy, pandas, scipy, matplotlib) and the data
# root (MSR_ROOT, default ~/playground/esm-msr-devel).  CPU only.  The core bootstrap (01) is the slow step (~15 min, one core).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONPATH=.
$PY 00_validate_against_tables.py
$PY 01_core_rho.py
for s in 02_thermostability 03_tsuboyama 04_dms_domainome 05_proteingym 06_epistasis 07_rmse_calibration 08_screening 14_property_category_gain 15_tsuboyama_single_screening 09_sequence_length 10_leakage_and_reference 11_publication_figures 11d_epistasis_levels_bootstrap 11b_publication_epistasis_levels 11c_publication_epistasis_scatters 11f_epistasis_types 11g_link_figure 12_pair_epistasis_heatmaps 13_dddG_correlation_scatters; do
  echo "== $s"; $PY $s.py
done
# ProteinGym Activity (own folder).  pg_activity/01_wt_reference.py rebuilds data/wt_reference.csv from the raw DMS files (ProteinGym/substitutions_raw_DMS) and is run on demand.
echo "== pg_activity"; $PY pg_activity/02_activity_screening.py
echo "== proteingym/p0_sigma"; for s in 01_per_assay_rho 02_fig_multi_property 03_fig_property_summary; do $PY proteingym/p0_sigma/$s.py; done
echo "== 16_screening_overview"; $PY 16_screening_overview.py          # needs the curve tables of 08_screening and pg_activity/02
echo "== 17_gain_per_variant_fit"; $PY 17_gain_per_variant_fit.py          # the log-linear fit of the gain per variant
echo "== 98_final_ablation_table"; $PY 98_final_ablation_table.py          # the FINAL_* ablation tables: needs the testing queue's final_bench/ and final_pgym_*/
$PY 99_table_coverage.py
