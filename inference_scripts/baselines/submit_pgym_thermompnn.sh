#!/bin/bash
# Queue the ProteinGym evaluation of the capped ThermoMPNN + ThermoMPNN-D retrains (the composite: single network on every row, epistatic
# network on the doubles) for the given seeds, one fixed-slot job per seed on gpu4. Only seed 1 existed (outputs/pgym/thermompnn_corrected_seed1,
# 2026-10-28); seeds 2 and 3 make the ProteinGym ThermoMPNN-D column a triplicate like every other retrained model.
# Run on the cluster from a checkout that has this directory (the checkpoints are resolved as in submit_mgnify.sh):
#   SPLIT=<split> SINGLE_DIR=<dir> EPISTATIC_DIR=<dir> SEEDS="2 3" ./submit_pgym_thermompnn.sh      # e.g. SPLIT=splits_oct06_capped SINGLE_DIR=baselines_capped/thermompnn EPISTATIC_DIR=baselines_corrected/thermompnn
#   ... SEEDS=2 PGYM_DMS=A4_HUMAN_Seuma_2022 ./submit_pgym_thermompnn.sh                               # smoke: one assay
# Results: $OUT_ROOT/thermompnn_corrected_seed<k>/<DMS>.csv + summary.csv (resumable: resubmit the same line, finished assays are skipped).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-$HOME/esm-msr-devel/outputs/pgym}
OUT_TM=$HOME/esm-msr-devel/outputs
SINGLE_DIR=${SINGLE_DIR:?set SINGLE_DIR (the ThermoMPNN single-network runs, single_seed<k>/, under outputs/)}
EPISTATIC_DIR=${EPISTATIC_DIR:?set EPISTATIC_DIR (the ThermoMPNN-D epistatic runs, epistatic_seed<k>/, under outputs/)}
export SPLIT=${SPLIT:?set SPLIT (the Tsuboyama split the models were trained on; the job script requires it)} STRICT=${STRICT:-0} TIME=${TIME:-24:00:00} PGYM_ONLY=1 EXTRA="--no_resource_tracking"
for s in ${SEEDS:-2 3}; do
  single=$(ls "$OUT_TM/$SINGLE_DIR"/single_seed$s/single_oct06_seed${s}_epoch=*.ckpt)
  epistatic=$(ls "$OUT_TM/$EPISTATIC_DIR"/epistatic_seed$s/epistatic_oct06_seed${s}_epoch=*.ckpt)
  echo "seed $s: single=$(basename "$single") epistatic=$(basename "$epistatic")" >&2
  MODEL=thermompnn OUT=$OUT_ROOT/thermompnn_corrected_seed$s SINGLE=$single EPISTATIC=$epistatic ${PGYM_DMS:+PGYM_DMS=$PGYM_DMS} \
    "$HERE/../sbatch_fixed_slot.sh" "$HERE/slurm_inference.sh"
done
