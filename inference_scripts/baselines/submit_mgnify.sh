#!/bin/bash
# Queue the capped retrains (3 seeds each) of ThermoMPNN(-D) and SPURS(-multi, unfrozen adapter) on the MGnify scaffolds and
# Rocklin designs (mgnify_rows.py inputs in data/mgnify), one fixed-slot job per model and seed on gpu4. Run on the cluster from a
# checkout that has this directory. Results: $OUT_ROOT/<label>/mgnify/<source>/chunk_NNNN.csv.gz; a killed job resumes by resubmitting
# the same line (finished chunks are skipped).
#   ./submit_mgnify.sh            submit all six
#   ONLY=spurs_unfrozen_s1 ./submit_mgnify.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-$HOME/esm-msr-devel/outputs/tracked/mgnify}
export SPLIT=splits_oct06_capped MGNIFY_DIR=${MGNIFY_DIR:-$HOME/esm-msr-devel/data/mgnify} BATCH_MODE=fixed STRICT=0 TIME=${TIME:-24:00:00}
export EXTRA="--no_resource_tracking"           # 181k scaffolds: per-protein tracker rows are not wanted here
SPURS_LOGS=$HOME/spurs_retrain/SPURS/logs/oct06
OUT_TM=$HOME/esm-msr-devel/outputs
for s in 1 2 3; do
  label=spurs_unfrozen_s$s
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$label" ] || continue
  MODEL=spurs OUT=$OUT_ROOT/$label SPURS_SINGLE=$SPURS_LOGS/unf_seed$s/checkpoints/best.ckpt SPURS_MULTI=$SPURS_LOGS/unf_multi_seed$s/checkpoints/best.ckpt \
    "$HERE/../sbatch_fixed_slot.sh" "$HERE/slurm_inference.sh"
done
for s in 1 2 3; do
  label=thermompnn_s$s
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$label" ] || continue
  single=$(ls "$OUT_TM"/baselines_capped/thermompnn/single_seed$s/single_oct06_seed${s}_epoch=*.ckpt)
  epistatic=$(ls "$OUT_TM"/baselines_corrected/thermompnn/epistatic_seed$s/epistatic_oct06_seed${s}_epoch=*.ckpt)
  MODEL=thermompnn OUT=$OUT_ROOT/$label SINGLE=$single EPISTATIC=$epistatic "$HERE/../sbatch_fixed_slot.sh" "$HERE/slurm_inference.sh"
done
