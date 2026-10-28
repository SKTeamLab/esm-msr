#!/bin/bash
# Submit an existing benchmark job script in THE fixed inference slot, so every model is measured on the same resources:
#   one GPU, SLOT_CPUS CPUs and SLOT_RAM_GB of host RAM on gpu4 (an RTX A4500 node: 4 GPUs, 24 CPUs, 250 GB,
#   so one quarter of the node: 1 GPU, 6 CPUs, 60 GB).
#
# Usage:   [VAR=value ...] inference_scripts/sbatch_fixed_slot.sh job_script.sh [script args]
#   The job script's own #SBATCH lines for node, GPUs, CPUs and memory are overridden by the command-line options
#   below (sbatch lets the command line win); its --time and --job-name stay (its --output too, unless OUT is set). Set the script's variables in
#   the environment (SEED=1 STAGE=double ... ), not with --export: sbatch splits --export on commas.
#     TIME=HH:MM:SS     override the script's --time
#     OUT=dir           where the job's results go; its Slurm log is written there too
#     STRICT=0          on by default: the job aborts at start if the slot is not verifiably in force (visible GPUs, CPU
#                       affinity mask, cgroup memory limit) and a process over the RAM limit raises at its next scope
#                       exit. gpu4's Slurm cgroups pass the check (smoke runs of 2026-10-28); resources_run.json records
#                       the check either way (limits.check).
#     SLOT_NODE / SLOT_CPUS / SLOT_RAM_GB   change the slot (a different slot is a different protocol: rerun everything)
#
# What it cannot do: stop other users' jobs sharing the node. Submit when gpu4 is otherwise idle; the tracker records the
# other jobs on the node at start and end (resources_run.json: neighbours_at_start / neighbours_at_end) and the node load
# per scope (loadavg1), so a contaminated run can be identified and repeated. This script warns about running neighbours.
#
# The per-model jobs that run every benchmark are baselines/slurm_inference.sh (MODEL=esm_msr|thermompnn|mutate_everything|
# spurs|proteinmpnn); MSR_REPO tells it where this repository is.
set -euo pipefail
NODE=${SLOT_NODE:-gpu4}
CPUS=${SLOT_CPUS:-6}
RAM_GB=${SLOT_RAM_GB:-60}
STRICT=${STRICT:-1}
[ $# -ge 1 ] || { sed -n '2,24p' "$0"; exit 1; }
SCRIPT=$1; shift
[ -f "$SCRIPT" ] || { echo "no such job script: $SCRIPT"; exit 1; }

busy=$(squeue -h -w "$NODE" -t R -o '%i %u %j %C %m' || true)
if [ -n "$busy" ]; then
  echo "WARNING: jobs already running on $NODE (the run will share the node with them):" >&2
  echo "$busy" | sed 's/^/  /' >&2
fi
echo "slot: node=$NODE gpus=1 cpus=$CPUS ram=${RAM_GB}G strict=$STRICT script=$SCRIPT" >&2

# the job log goes next to the results when OUT is set (the job script's own --output would land in the submit directory)
LOG=()
if [ -n "${OUT:-}" ]; then mkdir -p "$OUT"; LOG=(--output="$(cd "$OUT" && pwd)/slurm-%x-%j.out"); fi

exec sbatch ${LOG[@]+"${LOG[@]}"} --nodelist="$NODE" --gres=gpu:1 --cpus-per-task="$CPUS" --mem="${RAM_GB}G" ${TIME:+--time="$TIME"} \
  --export="ALL,MSR_REPO=$(cd "$(dirname "$0")/.." && pwd),MSR_FIXED_GPUS=1,MSR_FIXED_CPUS=$CPUS,MSR_FIXED_RAM_GB=$RAM_GB,MSR_STRICT_RESOURCES=$STRICT,OMP_NUM_THREADS=$CPUS,MKL_NUM_THREADS=$CPUS,OPENBLAS_NUM_THREADS=$CPUS" \
  "$SCRIPT" "$@"
