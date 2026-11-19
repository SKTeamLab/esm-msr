#!/bin/bash
#SBATCH --job-name=ros-ddg-mg
#SBATCH --account=def-skal
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out
# Step 3: cartesian_ddg for one chunk of tasks/chunks.csv ($1 = index offset added to the array index; a chunk with a DONE marker is skipped,
# so failed or timed-out chunks are simply resubmitted).  Flags as cluster_inference_scripts/capped/ddg_capped.sh.
ROOT=${ROOT:-/lustre07/scratch/sareeves/esm-msr}
OUT=${OUT:-$ROOT/rosetta_predictions_rocklin}
source $ROOT/env/setup_narval_rosetta.sh
hostname
IDX=$(( SLURM_ARRAY_TASK_ID + ${1:-0} ))
IFS=',' read -r CHUNK CODE BG NMUT REST <<< "$(sed -n "$((IDX + 2))p" $OUT/tasks/chunks.csv)"
[ -n "$CODE" ] || { echo "ERROR: no chunk for array index $IDX"; exit 1; }
NAME=$(printf 'chunk_%05d' $CHUNK)
PDB=$OUT/relaxed/$CODE/${CODE}_minimized_0001.pdb
[ -s $PDB ] || { echo "ERROR: input structure $PDB missing"; exit 1; }
WD=$OUT/ddg/$NAME
mkdir -p $WD && cd $WD || exit 1
if [ -f DONE ]; then echo "$NAME already done"; exit 0; fi
rm -f *.ddg
cp $OUT/tasks/mutfiles/$NAME.mut .
echo "$NAME: $CODE, $NMUT mutations, input $PDB"
start=$(date +%s)
$EBROOTROSETTA/bin/cartesian_ddg.mpi.linuxiccrelease \
    -in:file:s $PDB \
    -database $ROSETTA3_DB \
    -ddg:iterations 3 \
    -ddg:dump_pdbs false \
    -in:missing_density_to_jump \
    -ddg:mut_file $NAME.mut \
    -ddg:bbnbrs 1 \
    -corrections:beta_nov16_cart true \
    -packing:ex1 \
    -packing:ex2 \
    -score:fa_max_dis 9.0 \
    -ddg:optimize_proline true \
    -ddg:legacy true > ddg.log 2>&1
rc=$?
tail -3 ddg.log
[ $rc -eq 0 ] || { echo "ERROR: cartesian_ddg failed for $NAME (rc=$rc)"; exit 1; }
echo $(( $(date +%s) - start )) > runtime.txt
touch DONE
echo "$NAME done in $(cat runtime.txt) s"
