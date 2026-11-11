#!/bin/bash
#SBATCH --job-name=tmpnn_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# Train ThermoMPNN (MODEL=single) or ThermoMPNN-D (MODEL=epistatic) on the default capped split.
# Usage (from repo root, locally with `bash` or on Slurm with `sbatch`):
#   MODEL=single|epistatic [SEED=1] [EPOCHS=100] [BASE_OUT=...] \
#     bash training_scripts/baselines/slurm_thermompnn_train.sh [extra args]
set -euo pipefail

REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ -z "${THIRD_PARTY_DIR:-}" ]; then
  if [ -d "$HOME/software/MutateEverything-oct06" ]; then
    THIRD_PARTY_DIR="$HOME/software"
  elif [ -d "$HOME/software/msr_baselines" ]; then
    THIRD_PARTY_DIR="$HOME/software/msr_baselines"
  else
    THIRD_PARTY_DIR="$HOME/software"
  fi
fi
export THIRD_PARTY_DIR
THERMOMPNN_DIR=${THERMOMPNN_DIR:-$THIRD_PARTY_DIR/ThermoMPNN-D}

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU ($CUDA_VISIBLE_DEVICES)"; exit 1; }
else
  nvidia-smi || { echo "unhealthy GPU"; exit 1; }
fi

unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [ -z "${THERMOMPNN_PY:-}" ]; then
  for cand in \
    "$THERMOMPNN_DIR/.venv/bin/python" \
    "$HOME/miniconda3/envs/ThermoMPNN-D/bin/python" \
    "$HOME/.conda/envs/ThermoMPNN-D/bin/python"; do
    if [ -x "$cand" ]; then
      THERMOMPNN_PY=$cand
      break
    fi
  done
fi
: "${THERMOMPNN_PY:?set THERMOMPNN_PY or create the ThermoMPNN-D environment (see training_scripts/baselines/requirements/README.md)}"
ENV_LIB="$(cd "$(dirname "$THERMOMPNN_PY")/.." && pwd)/lib"
if [ -d "$ENV_LIB" ]; then
  export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
fi

MODEL=${MODEL:?set MODEL (single or epistatic)}
SEED=${SEED:-1}
RUN=${RUN:-${MODEL}_seed${SEED}}
export MSR_BASELINE_DATA=${MSR_BASELINE_DATA:?set MSR_BASELINE_DATA (e.g. $REPO/outputs/baselines/oct06_capped_allbl/data)}
OUT=${BASE_OUT:-$(dirname "$MSR_BASELINE_DATA")}/thermompnn/$RUN
NUM_WORKERS=${NUM_WORKERS:-10}

mkdir -p "$OUT"
echo "data=$MSR_BASELINE_DATA out=$OUT py=$THERMOMPNN_PY"
"$THERMOMPNN_PY" -u "$REPO/inference_scripts/baselines/thermompnn_inference.py" train \
  --model "$MODEL" \
  --seed "$SEED" \
  --out "$OUT" \
  --num_workers "$NUM_WORKERS" \
  ${SPLIT:+--tsuboyama_split "$SPLIT"} \
  ${EPOCHS:+--epochs "$EPOCHS"} \
  "$@"
