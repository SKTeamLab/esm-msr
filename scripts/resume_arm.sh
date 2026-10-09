#!/bin/bash
# Usage: scripts/resume_arm.sh NAME EPOCHS SEED COMET_KEY [training flags, the same as the original run...]
# Continues training_logs/NAME/0/last.ckpt up to EPOCHS epochs in total, logging into the same Comet experiment. The CSVLogger rewrites
# metrics.csv from the resume point, so the earlier rows are kept in metrics_backup_e<last epoch>.csv; the run log is kept the same way.
# Skips the start-up validation (the checkpoint's last epoch was validated already). The learning-rate plateaus restart their counters.
NAME=$1; EPOCHS=$2; SEED=$3; KEY=$4; shift 4
D=/home/sareeves/playground/esm-msr-devel
L=$D/training_logs/$NAME/0
[ -f $L/last.ckpt ] || { echo "$(date) $NAME RESUME failed: no $L/last.ckpt" >> $D/queue_history.txt; exit 1; }
last=$(ls $L/val_dump_e*.npz 2>/dev/null | sed 's/.*_e\([0-9]*\)\.npz/\1/' | sort -n | tail -1)
cp $L/metrics.csv $L/metrics_backup_e${last}.csv
cp $D/run_logs/$NAME.log $D/run_logs/${NAME}_to_e${last}.log
echo "$(date) $NAME RESUME from epoch $last to $EPOCHS epochs (Comet $KEY) $*" >> $D/queue_history.txt
export MSR_MEM_FRACTION=${MSR_MEM_FRACTION:-0.95}
"$(dirname "$(readlink -f "$0")")/run_arm.sh" $NAME $EPOCHS $SEED "$@" --ckpt_path $L/last.ckpt --comet_experiment_key $KEY --skip_val
tail -1 $D/run_logs/$NAME.log | grep -aq "DONE rc=0" && echo "$(date) $NAME END (resumed)" >> $D/queue_history.txt \
  || echo "$(date) $NAME END crashed (resume)" >> $D/queue_history.txt
