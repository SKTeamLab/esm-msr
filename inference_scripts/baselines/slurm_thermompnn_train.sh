#!/bin/bash
#SBATCH --job-name=tmpnn_oct06
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# ThermoMPNN-D (upstream train_single.yaml / train_epistatic.yaml, unchanged) on the ESM-MSR-matched oct06 data.
# Usage: MODEL=single|epistatic SEED=1 [RUN=name] sbatch inference_scripts/baselines/slurm_thermompnn_train.sh [extra args]   (from the repo root)
#
# MUT_BACKBONES=include|exclude picks which prepared condition to train on: whether mutant-background libraries
# ('2K52_L42A', WT backbone + background-mutated sequence) are in train/val. The exclusion itself happens in
# prepare/prepare_oct06_data.py; here it only selects that run's data dir and output tree. Set MSR_BASELINE_DATA / BASE_OUT
# directly to override.
set -euo pipefail
REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}          # sbatch runs a copy of this script, so the repo cannot be found from $0
THERMOMPNN_DIR=${THERMOMPNN_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/ThermoMPNN-D}
hostname; nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU in allocation ($CUDA_VISIBLE_DEVICES)"; exit 1; }  # gpu2 idx 3 (D8:00.0) is unresponsive
unset PYTHONPATH
module load cuda/12.1
MODEL=${MODEL:?set MODEL}; SEED=${SEED:-1}; RUN=${RUN:-${MODEL}_seed${SEED}}
MUT_BACKBONES=${MUT_BACKBONES:-include}
case $MUT_BACKBONES in
  include) DEF=$REPO/outputs/baselines_capped;;     # the corrected baselines were trained on $REPO/outputs/baselines_capped_allbl: set MSR_BASELINE_DATA and BASE_OUT
  exclude) DEF=$REPO/outputs/baselines_nobb;;
  *) echo "bad MUT_BACKBONES=$MUT_BACKBONES"; exit 1;;
esac
export MSR_BASELINE_DATA=${MSR_BASELINE_DATA:-$DEF/data}
OUT=${BASE_OUT:-$DEF}/thermompnn/$RUN
mkdir -p $OUT
echo "mut_backbones=$MUT_BACKBONES data=$MSR_BASELINE_DATA out=$OUT"
"$THERMOMPNN_DIR/.venv/bin/python" -u "$REPO/inference_scripts/baselines/thermompnn_inference.py" train \
  --model $MODEL --seed $SEED --out $OUT --num_workers 10 "$@"
