#!/usr/bin/env bash
# Every Tsuboyama analysis of analysis/ on the mega_splits retrains -> analysis/results/03_tsuboyama/mega_splits/<script's folder>/
# (MSR_TSU_SPLIT=mega re-points TSU, the loaders and RESULTS; see README.md here).  Needs the testing queue's FINAL_p0_mega predictions
# (final_bench/predictions/mega_splits-{val,test}/) and the cluster's mega_splits retrains collected to baselines/retrained/mega_splits/tracked/.
set -euo pipefail
cd "$(dirname "$0")/../.."          # analysis/
PY=${PYTHON:-python}
export PYTHONPATH=. MSR_TSU_SPLIT=mega
rm -rf results/03_tsuboyama/mega_splits
$PY tsuboyama/mega_splits/00_build_tasks.py
$PY -c "from msr_analysis.data import load_frame; [load_frame(d, refresh=True) for d in ('tsuboyama_mega_val', 'tsuboyama_mega_test')]"
$PY tsuboyama/mega_splits/01_rho_overview.py
for s in 06_epistasis 11d_epistasis_levels_bootstrap 11b_publication_epistasis_levels 11c_publication_epistasis_scatters 11f_epistasis_types \
         12_pair_epistasis_heatmaps 13_dddG_correlation_scatters 08_screening 15_tsuboyama_single_screening 17_gain_per_variant_fit; do
  echo "== $s"; $PY $s.py
done
