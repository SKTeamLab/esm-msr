#!/bin/bash
#SBATCH --job-name=ros-relax-mg
#SBATCH --account=def-skal
#SBATCH --time=0:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out
# Step 1: cartesian FastRelax of each WT scaffold (one array task per line of tasks/structures.csv; $1 = index offset).  Flags as
# cluster_inference_scripts/capped/relax_capped.sh.   sbatch --array=0-999 relax.sh
ROOT=${ROOT:-/lustre07/scratch/sareeves/esm-msr}
OUT=${OUT:-$ROOT/rosetta_predictions_rocklin}
source $ROOT/env/setup_narval_rosetta.sh
hostname
IDX=$(( SLURM_ARRAY_TASK_ID + ${1:-0} ))
CODE=$(sed -n "$((IDX + 2))p" $OUT/tasks/structures.csv)
[ -n "$CODE" ] || { echo "ERROR: no structure for array index $IDX"; exit 1; }
WD=$OUT/relaxed/$CODE
mkdir -p $WD && cd $WD || exit 1
if [ -s ${CODE}_minimized_0001.pdb ]; then echo "$CODE already relaxed"; exit 0; fi
echo "Relaxing $CODE"
start=$(date +%s)
$EBROOTROSETTA/bin/relax.mpi.linuxiccrelease \
    -s $OUT/inputs/$CODE.pdb \
    -database $ROSETTA3_DB \
    -relax:constrain_relax_to_start_coords \
    -in:ignore_unrecognized_res \
    -in:missing_density_to_jump \
    -out:nstruct 1 \
    -relax:coord_constrain_sidechains \
    -relax:cartesian \
    -corrections:beta_nov16_cart \
    -packing:ex1 \
    -packing:ex2 \
    -relax:min_type lbfgs_armijo_nonmonotone \
    -packing:flip_HNQ \
    -packing:no_optH false \
    -out:suffix _minimized \
    -out:overwrite > relax.log 2>&1
rc=$?
tail -5 relax.log
[ $rc -eq 0 ] && [ -s ${CODE}_minimized_0001.pdb ] || { echo "ERROR: relax failed for $CODE (rc=$rc)"; exit 1; }
echo $(( $(date +%s) - start )) > relax_runtime.txt
echo "Relaxed $CODE in $(cat relax_runtime.txt) s"
