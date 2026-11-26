#!/bin/bash
#SBATCH --job-name=scale_msr_cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/runs/slurm-%x-%j.out
set -euo pipefail

# CPU-only fixed slot (6 CPUs, 60 GB RAM, 0 GPUs)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export MSR_FIXED_CPUS=6
export MSR_FIXED_RAM_GB=60
export MSR_FIXED_GPUS=0
export MSR_STRICT_RESOURCES=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

STAGE=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/repo_stage
PANEL=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/panel/panel.json
OUT=/home/sreeves/esm-msr-devel/outputs/size_scaling_a4500/runs
PY=/home/sreeves/.conda/envs/msr_venv/bin/python

hostname
lscpu | grep -E "Model name|CPU\(s\):" || true
free -h

exec "$PY" "$STAGE/inference_scripts/baselines/benchmark_size_scaling.py" \
  --model esm_msr_wt_cpu \
  --panel "$PANEL" \
  --out "$OUT" \
  --double_cap_s 86400
