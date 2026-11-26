#!/bin/bash
# drive.sh then final_pass.sh as a small Slurm job: a login-node process was killed twice around 07:00 and left the run idle.  From $ROOT:
#   OUT=<abs run dir> START0=<first unsubmitted chunk> sbatch --export=ALL cluster_inference_scripts/rocklin/driver_job.sh
#SBATCH --job-name=ros-driver
#SBATCH --account=def-skal
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --output=logs/%x-%j.out
ROOT=${ROOT:-/lustre07/scratch/sareeves/esm-msr}; cd $ROOT
export OUT=${OUT:-$ROOT/rosetta_predictions_rocklin}
S=$ROOT/cluster_inference_scripts/rocklin; TAG=$(basename $OUT)
bash $S/drive.sh >> logs/${TAG}_drive_job.log 2>&1
bash $S/final_pass.sh >> logs/${TAG}_final_pass_job.log 2>&1
