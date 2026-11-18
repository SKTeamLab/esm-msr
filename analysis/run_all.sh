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
echo "== proteingym/p0_sigma"; for s in 01_per_assay_rho 04_leak_per_assay 02_fig_multi_property 03_fig_property_summary; do $PY proteingym/p0_sigma/$s.py; done
echo "== 16_screening_overview"; $PY 16_screening_overview.py          # needs the curve tables of 08_screening and pg_activity/02
echo "== 17_gain_per_variant_fit"; $PY 17_gain_per_variant_fit.py          # the log-linear fit of the gain per variant
echo "== tsuboyama/mega_splits"; bash tsuboyama/mega_splits/run_all.sh     # the Tsuboyama analyses on the mega_splits retrains (results/03_tsuboyama/mega_splits/)
echo "== combined_vs_wt"; for s in 01_benchmark_delta 02_pg_per_assay 03_unit_rho 04_fig_proteingym 05_fig_overview 06_fig_when 07_tables 08_all_families; do $PY combined_vs_wt/$s.py; done          # combined versus WT head by assay type, variant class and masking (p0, p1, pw0)
echo "== faure_epistasis"; for s in 01_proximity 02_snr 03_cell_ceiling 04_fig_ceiling 05_cell_mag_significance; do PYTHONPATH=.:../src $PY faure_epistasis/$s.py; done          # epistasis of the DLG4 / GRB2 abundance and binding datasets: proximity, signal-to-noise, cell_mag ceiling
echo "== ablation_vs_p0"; for s in 01_paired_vs_p0 02_pg_vs_p0 03_fig_heatmap 04_fig_thermostability 05_fig_forest 06_tables; do $PY ablation_vs_p0/$s.py; done          # every FINAL_* ablation against p0, paired, on every set (needs combined_vs_wt/02_pg_per_assay)
echo "== 18_mgnify"; $PY 18_mgnify.py          # MGnify: the Tsuboyama analyses on a set ESM-MSR never saw (p0 only), latent and Tobit-linked; needs final_mgnify/ and data/mgnify
echo "== 01_summary"; $PY 01_summary.py          # summary figure (pg_categories): needs 05_proteingym and 18_mgnify
echo "== 98_final_ablation_table"; $PY 98_final_ablation_table.py          # the FINAL_* ablation tables: needs the testing queue's final_bench/ and final_pgym_*/
$PY 99_table_coverage.py
