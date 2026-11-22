#!/bin/bash
# Slurm job script to run ONE model of benchmark_size_scaling.py in the fixed A4500 slot.
# Submit via inference_scripts/sbatch_fixed_slot.sh:
#   MODEL=esm_msr_wt PANEL=.../panel.json OUT=.../size_scaling_a4500 TIME=02:00:00 \
#     inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_size_scaling.sh
#SBATCH --job-name=scale_bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

MODEL=${MODEL:?set MODEL}
PANEL=${PANEL:?set PANEL}
OUT=${OUT:?set OUT}
REPO=${MSR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SCRIPT=$REPO/inference_scripts/baselines/benchmark_size_scaling.py
MAX_LEN=${MAX_LEN:-0}

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
else
  nvidia-smi
fi
unset PYTHONPATH
module load cuda/12.1 2>/dev/null || true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ "$MODEL" != me_* ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

if [ -z "${CONDA_SH:-}" ]; then
  for _c in /usr/local/anaconda3/etc/profile.d/conda.sh "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$_c" ]; then CONDA_SH="$_c"; break; fi
  done
fi
activate() {
  set +u; source "$CONDA_SH"; conda activate "$1"; set -u
  if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  fi
}

case "$MODEL" in
  esm_msr_wt|esm_msr_dual|proteinmpnn)
    activate msr_venv
    PY=$(which python)
    ;;
  prosst)
    PY="$HOME/prosst_env/bin/python"
    ;;
  me_st1|me_st2)
    module purge 2>/dev/null || true
    activate mutate_everything
    PY=$(which python)
    ;;
  thermompnn_single|thermompnn_d)
    PY="$HOME/software/ThermoMPNN-D/.venv/bin/python"
    ENV_LIB="$(cd "$(dirname "$PY")/.." && pwd)/lib"
    [ -d "$ENV_LIB" ] && export LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH:-}"
    ;;
  spurs_single|spurs_multi)
    activate spurs_higher
    PY=$(which python)
    export SPURS_CKPT_DIR=${SPURS_CKPT_DIR:-$HOME/spurs_retrain/harness_spurs_dir/checkpoints}
    ;;
  *)
    echo "Unknown MODEL=$MODEL" >&2
    exit 1
    ;;
esac

echo "Running size-scaling benchmark: model=$MODEL py=$PY panel=$PANEL out=$OUT max_len=$MAX_LEN"
"$PY" -u "$SCRIPT" --model "$MODEL" --panel "$PANEL" --out "$OUT" --max_len "$MAX_LEN"
