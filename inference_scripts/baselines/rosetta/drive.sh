#!/bin/bash
# Keeps the narval queue under the 1000-job submit limit: relax every WT, then submit the ddG chunks in batches while fewer than $MAXQ of
# our jobs are queued/running.  Resumable (finished structures and chunks are skipped by their scripts).  Run from $ROOT under setsid nohup:
#   cd /lustre07/scratch/sareeves/esm-msr && setsid nohup bash cluster_inference_scripts/rocklin/drive.sh > logs/rocklin_drive.log 2>&1 &
ROOT=${ROOT:-/lustre07/scratch/sareeves/esm-msr}
OUT=${OUT:-$ROOT/rosetta_predictions_rocklin}
S=$(dirname "$0")
BATCH=${BATCH:-200}; MAXQ=${MAXQ:-900}
cd $ROOT
NS=$(( $(wc -l < $OUT/tasks/structures.csv) - 1 )); NC=$(( $(wc -l < $OUT/tasks/chunks.csv) - 1 ))
nq() { squeue -u $USER -h -r | wc -l; }
echo "$(date) $NS structures, $NC chunks"
if [ "$(ls $OUT/relaxed/*/*_minimized_0001.pdb 2>/dev/null | wc -l)" -lt "$NS" ]; then
  J=$(sbatch --parsable --array=0-$((NS-1)) $S/relax.sh 0); echo "$(date) relax array $J"
  while squeue -j $J -h | grep -q .; do sleep 60; done
fi
echo "$(date) relaxed: $(ls $OUT/relaxed/*/*_minimized_0001.pdb | wc -l)/$NS"
for (( START=0; START<NC; START+=BATCH )); do
  # skip a batch whose chunks are all done
  todo=0; for (( i=START; i<START+BATCH && i<NC; i++ )); do [ -f $OUT/ddg/chunk_$(printf %05d $i)/DONE ] || { todo=1; break; }; done
  [ $todo -eq 0 ] && continue
  while [ $(( $(nq) + BATCH )) -gt $MAXQ ]; do sleep 120; done
  N=$(( NC - START < BATCH ? NC - START : BATCH ))
  J=$(sbatch --parsable --array=0-$((N-1)) $S/ddg.sh $START); echo "$(date) ddg chunks $START..$((START+N-1)) job $J"
done
echo "$(date) all submitted"
