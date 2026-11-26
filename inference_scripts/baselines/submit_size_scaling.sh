#!/bin/bash
# Submit all 10 controlled protein-size scaling benchmark jobs on subhak-gpu (gpu4 fixed RTX A4500 20 GB slot).
# Usage:
#   OUT=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/runs \
#   PANEL=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/panel/panel.json \
#   bash inference_scripts/baselines/submit_size_scaling.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:?set OUT}
PANEL=${PANEL:?set PANEL}
NODE=${SLOT_NODE:-gpu4}
STRICT=${STRICT:-1}
TIME=${TIME:-02:00:00}

mkdir -p "$OUT"
MODELS=(
  esm_msr_wt
  esm_msr_dual
  me_st1
  me_st2
  thermompnn_single
  thermompnn_d
  spurs_single
  spurs_multi
  proteinmpnn
  prosst
)
for m in "${MODELS[@]}"; do
  SLOT_NODE="$NODE" STRICT="$STRICT" TIME="$TIME" \
  MODEL="$m" PANEL="$PANEL" OUT="$OUT" MAX_LEN="${MAX_LEN:-0}" \
    bash "$REPO/inference_scripts/sbatch_fixed_slot.sh" \
         "$REPO/inference_scripts/baselines/slurm_size_scaling.sh"
done
