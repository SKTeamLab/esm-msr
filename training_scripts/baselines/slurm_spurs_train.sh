#!/bin/bash
#SBATCH --job-name=spurs_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Train SPURS (MODEL=single) or SPURS-multi (MODEL=multi, initialized from matching seed's SPURS-single)
# on the default capped split (`oct06_capped_allbl`). Automatically prepares the patched SPURS training
# directory (`$SPURS_TRAIN_DIR`) on first run.
# Usage (from repo root, locally with `bash` or on Slurm with `sbatch`):
#   MODEL=single|multi [SEED=1] [EPOCHS=80] [SPURS_TRAIN_DIR=...] \
#     bash training_scripts/baselines/slurm_spurs_train.sh
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
SPURS_DIR=${SPURS_DIR:-$THIRD_PARTY_DIR/SPURS}
PROTEINMPNN_DIR=${PROTEINMPNN_DIR:-$THIRD_PARTY_DIR/ProteinMPNN}

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU ($CUDA_VISIBLE_DEVICES)"; exit 1; }
else
  nvidia-smi || { echo "unhealthy GPU"; exit 1; }
fi

module load cuda/12.1 2>/dev/null || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [ -z "${SPURS_PY:-}" ]; then
  for env_name in "${SPURS_ENV:-}" spurs_higher spurs; do
    [ -z "$env_name" ] && continue
    for root in "$HOME/miniconda3/envs" "$HOME/.conda/envs" "/usr/local/anaconda3/envs"; do
      if [ -x "$root/$env_name/bin/python" ]; then
        SPURS_PY="$root/$env_name/bin/python"
        break 2
      fi
    done
  done
fi
: "${SPURS_PY:?set SPURS_PY or create the spurs environment (see training_scripts/baselines/requirements/README.md)}"
ENV_PREFIX="$(cd "$(dirname "$SPURS_PY")/.." && pwd)"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

MSR_BASELINE_DATA=${MSR_BASELINE_DATA:?set MSR_BASELINE_DATA (e.g. $REPO/outputs/baselines/oct06_capped_allbl/data)}
SPURS_TRAIN_DIR=${SPURS_TRAIN_DIR:-$(dirname "$MSR_BASELINE_DATA")/spurs_train}

if [ ! -f "$SPURS_TRAIN_DIR/train.py" ]; then
  echo "Bootstrapping patched SPURS training tree at $SPURS_TRAIN_DIR ..."
  mkdir -p "$(dirname "$SPURS_TRAIN_DIR")"
  cp -r "$SPURS_DIR" "$SPURS_TRAIN_DIR"
  (cd "$SPURS_TRAIN_DIR" && patch -p1 < "$REPO/training_scripts/baselines/spurs/spurs_retrain.patch")
  mkdir -p "$SPURS_TRAIN_DIR/data/checkpoints/ThermoMPNN/vanilla_model_weights"
  cp "$PROTEINMPNN_DIR/vanilla_model_weights/v_48_020.pt" \
    "$SPURS_TRAIN_DIR/data/checkpoints/ThermoMPNN/vanilla_model_weights/"
fi

MODEL=${MODEL:-single}
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-80}
if [ -z "${SPLIT_NAME:-}" ]; then
  if [ -f "$MSR_BASELINE_DATA/variant.json" ]; then
    SPLIT_NAME=$("$SPURS_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$MSR_BASELINE_DATA/variant.json")
  else
    SPLIT_NAME=$(basename "$(dirname "$MSR_BASELINE_DATA")")
  fi
fi

# Guard against WSL2's ambient NAME=<hostname> environment variable when NAME was not explicitly set for SPURS
if [ "${NAME:-}" = "$(hostname)" ]; then
  unset NAME
fi

if [ -z "${OVERRIDES:-}" ]; then
  COMMON="datamodule._target_=items ++datamodule.data_dir=$MSR_BASELINE_DATA \
    ++datamodule.structure_json=$SPURS_TRAIN_DIR/parsed_structure_${SPLIT_NAME}.json \
    trainer.max_epochs=$EPOCHS \
    callbacks.model_checkpoint.save_last=false \
    model.encoder.pretrained_weights=data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt \
    model.legacy_esm_forward=true seed=$SEED train.seed=$SEED ${NUM_WORKERS:+datamodule.num_workers=$NUM_WORKERS}"
  if [ "$MODEL" = "multi" ]; then
    CONFIG_NAME=${CONFIG_NAME:-train_multi}
    RUN_NAME=${SPURS_RUN_NAME:-${NAME:-${SPLIT_NAME}/leg_multi_seed${SEED}}}
    SINGLE_CKPT=${SINGLE_CKPT:-$SPURS_TRAIN_DIR/logs/${SPLIT_NAME}/leg_seed${SEED}/checkpoints/best.ckpt}
    [ -f "$SINGLE_CKPT" ] || { echo "SPURS single checkpoint not found at $SINGLE_CKPT (train MODEL=single first)"; exit 1; }
    OVERRIDES="$COMMON model.init_from=$SINGLE_CKPT"
  elif [ "$MODEL" = "single" ]; then
    CONFIG_NAME=${CONFIG_NAME:-train}
    RUN_NAME=${SPURS_RUN_NAME:-${NAME:-${SPLIT_NAME}/leg_seed${SEED}}}
    OVERRIDES="$COMMON"
  else
    echo "Unknown MODEL=$MODEL (expected single or multi)" >&2
    exit 1
  fi
else
  RUN_NAME=${SPURS_RUN_NAME:-${NAME:?set SPURS_RUN_NAME or NAME}}
fi

cd "$SPURS_TRAIN_DIR"
export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONUNBUFFERED=1
echo "spurs_train_dir=$SPURS_TRAIN_DIR model=$MODEL name=$RUN_NAME py=$SPURS_PY"
# shellcheck disable=SC2086
"$SPURS_PY" train.py ${CONFIG_NAME:+--config-name "$CONFIG_NAME"} name="$RUN_NAME" $OVERRIDES "$@"
