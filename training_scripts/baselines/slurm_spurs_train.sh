#!/bin/bash
#SBATCH --job-name=spurs_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
# Train SPURS (or SPURS-multi) in the patched checkout made by README.md step 4.
# Usage: NAME=<run name> SPURS_TRAIN_DIR=<patched checkout> [CONFIG_NAME=train_multi] OVERRIDES="<hydra overrides>" sbatch slurm_spurs_train.sh
set -euo pipefail
hostname; nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" || { echo "unhealthy GPU"; exit 1; }
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
source "${CONDA_SH:-/usr/local/anaconda3/etc/profile.d/conda.sh}"
conda activate spurs_higher
cd "${SPURS_TRAIN_DIR:?set SPURS_TRAIN_DIR}"
export PROJECT_ROOT=$PWD HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python train.py ${CONFIG_NAME:+--config-name $CONFIG_NAME} name="${NAME:?set NAME}" $OVERRIDES
