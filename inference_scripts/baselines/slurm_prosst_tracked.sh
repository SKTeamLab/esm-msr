#!/bin/bash
# Fixed-slot RTX A4500 resource-tracked benchmark for ProSST-2048 (Megascale Test, External, Domainome, ProteinGym).
#SBATCH --job-name=prosst_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

OUT=${OUT:?set OUT}
REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SPLIT=${SPLIT:-splits_oct06_capped}
PGYM=${MSR_PGYM_INPUTS:-$HOME/esm-msr-devel/pgym_inputs}

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then nvidia-smi -i "$CUDA_VISIBLE_DEVICES"; else nvidia-smi; fi
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MSR_DATA_DIR="${MSR_DATA_DIR:-$HOME/esm-msr-devel/data}"
export MSR_STRUCTURES_DIR="${MSR_STRUCTURES_DIR:-$HOME/software/esm-msr/data/structures}"
export MSR_PGYM_INPUTS="$PGYM"

mkdir -p "$OUT"
"$HOME/prosst_env/bin/python" -u "$REPO/inference_scripts/baselines/prosst_tracked_benchmark.py" \
  --out "$OUT" --split "$SPLIT" --pgym_dir "$PGYM" --prosst_dir "$HOME/software/ProSST"
