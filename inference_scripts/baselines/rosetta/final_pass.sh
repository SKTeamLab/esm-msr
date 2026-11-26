#!/bin/bash
# After the driver has submitted everything and the queue has drained: resubmit every chunk without a DONE marker (3 h timeouts, crashes) once, with a 6 h limit.
# OUT (absolute: ddg.sh changes directory before using it) picks the run.  Array indices must stay below the cluster's MaxArraySize (10000), so they go in blocks with ddg.sh's offset argument.
ROOT=${ROOT:-/lustre07/scratch/sareeves/esm-msr}; cd $ROOT
export OUT=${OUT:-$ROOT/rosetta_predictions_rocklin}
S=$(dirname "$0"); TAG=$(basename $OUT)
while pgrep -u $USER -f "rocklin/drive.sh" > /dev/null; do sleep 120; done
while [ "$(squeue -u $USER -h -r -n ros-ddg-mg | wc -l)" -gt 0 ]; do sleep 120; done
find $OUT/ddg -maxdepth 2 -name DONE | sed "s#.*/chunk_0*\([0-9]*\)/DONE#\1#; s#^\$#0#" | sort > logs/${TAG}_done_ids.txt
NC=$(( $(wc -l < $OUT/tasks/chunks.csv) - 1 ))
seq 0 $((NC-1)) | sort | comm -23 - logs/${TAG}_done_ids.txt | sort -n > logs/${TAG}_missing_final.txt
N=$(wc -l < logs/${TAG}_missing_final.txt); echo "$(date) $N chunks without DONE"
[ $N -gt 0 ] || exit 0
[ $N -le 900 ] || { echo "ERROR: $N chunks is too many for one pass (submit limit 1000): look at why they failed"; exit 1; }
for off in $(awk '{print int($1/10000)*10000}' logs/${TAG}_missing_final.txt | sort -un); do
  L=$(awk -v o=$off '$1>=o && $1<o+10000 {printf "%s%d", (n++?",":""), $1-o}' logs/${TAG}_missing_final.txt)
  J=$(sbatch --parsable --time=6:00:00 --array=$L $S/ddg.sh $off) || { echo "ERROR: sbatch failed for offset $off"; continue; }
  echo "$(date) final pass offset $off, 6h, job $J"
done
