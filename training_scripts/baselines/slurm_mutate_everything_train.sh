#!/bin/bash
#SBATCH --job-name=me_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Train MutateEverything (AlphaFold backbone) on the default capped split (`oct06_capped_allbl`):
#   Stage 1 (`STAGE=single`): single-mutation pretraining (20 epochs, --lambda_double 0. --lambda_single 1.)
#   Stage 2 (`STAGE=double`): double-mutation fine-tuning from Stage 1 (100 epochs, --lambda_double 1. --lambda_single 0.1)
# Usage (from repo root, locally with `bash` or on Slurm with `sbatch`):
#   [STAGE=both|single|double] [SEED=1] [SPLIT_NAME=oct06_capped_allbl] [OUT_ROOT=logs] \
#     bash training_scripts/baselines/slurm_mutate_everything_train.sh
set -eo pipefail

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
MUTATE_EVERYTHING_DIR=${MUTATE_EVERYTHING_DIR:-$THIRD_PARTY_DIR/MutateEverything-oct06}

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU ($CUDA_VISIBLE_DEVICES)"; exit 1; }
else
  nvidia-smi || { echo "unhealthy GPU"; exit 1; }
fi

module purge 2>/dev/null || true
unset PYTHONPATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Suppress per-step OpenFold/PyTorch UserWarnings (e.g. non-tuple sequence indexing) by default.
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}

if [ -z "${ME_ENV_BIN:-}" ]; then
  for cand in \
    "$HOME/miniconda3/envs/mutate_everything_cu128/bin" \
    "$HOME/.conda/envs/mutate_everything_cu128/bin" \
    "$HOME/miniconda3/envs/mutate_everything/bin" \
    "$HOME/.conda/envs/mutate_everything/bin"; do
    if [ -x "$cand/python" ]; then
      ME_ENV_BIN=$cand
      break
    fi
  done
fi
: "${ME_ENV_BIN:?set ME_ENV_BIN or create mutate_everything_cu128 (see training_scripts/baselines/requirements/README.md)}"
export PATH="$ME_ENV_BIN:$PATH"
ENV_LIB="$(cd "$ME_ENV_BIN/.." && pwd)/lib"
if [ -d "$ENV_LIB" ]; then
  export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${NPROC:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else
    NPROC=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  fi
fi

cd "$MUTATE_EVERYTHING_DIR"
SEED=${SEED:-1}
STAGE=${STAGE:-both}
S=${SPLIT_NAME:?set SPLIT_NAME (e.g. oct06_capped_allbl)}
D=data/$S/mutations
OUT_ROOT=${OUT_ROOT:-logs}
EP1=${STAGE1_EPOCHS:-${EPOCHS:-20}}
EP2=${STAGE2_EPOCHS:-${EPOCHS:-100}}
STAGE1_CKPT=${STAGE1_CKPT:-$OUT_ROOT/af_single_${S}_seed${SEED}/checkpoint-$((EP1 - 1)).pth}
EVAL=$D/${S}_val.csv,$D/${S}_test.csv
PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

echo "split_name=$S stage=$STAGE seed=$SEED nproc=$NPROC env=$ME_ENV_BIN out_root=$OUT_ROOT"

if [ "$STAGE" != double ]; then
  torchrun --nproc_per_node="$NPROC" --master_port="$PORT" main_train.py \
    --batch_size 1 \
    --finetune_backbone models/finetuning_ptm_2.pt \
    --data_path "$D/${S}_train.csv" \
    --eval_data_paths "$EVAL" \
    --dist_eval \
    --backbone af \
    --lambda_double 0. --lambda_single 1. \
    --eval_period 999 --epochs "$EP1" --warmup_epochs 2 --save_period "$EP1" \
    --seed "$SEED" \
    --disable_wandb \
    --output_dir "$OUT_ROOT/af_single_${S}_seed${SEED}"
fi

if [ "$STAGE" != single ]; then
  torchrun --nproc_per_node="$NPROC" --master_port="$PORT" main_train.py \
    --batch_size 1 \
    --finetune_backbone models/finetuning_ptm_2.pt \
    --data_path "$D/${S}_train.csv" \
    --eval_data_paths "$EVAL" \
    --dist_eval \
    --backbone af \
    --eval_period 1000 --epochs "$EP2" --warmup_epochs 10 --save_period "$EP2" \
    --finetune "$STAGE1_CKPT" \
    --seed "$SEED" \
    --disable_wandb \
    --output_dir "$OUT_ROOT/af_double_${S}_seed${SEED}"
fi
