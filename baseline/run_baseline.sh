#!/bin/bash
# Trains the RELEASED code (unmodified objective and defaults: baseline/recipe_args.txt = the esm-msr-small `r1a4` hparams) on the capped split, with
# the new validation stats added by the transplant. Usage: baseline/run_baseline.sh RUN_NAME EPOCHS SEED [extra flags]   (log: run_logs/RUN_NAME.log)
W=$(dirname "$(dirname "$(readlink -f "$0")")")
D=/home/sareeves/playground/esm-msr-devel
cd $D
export PYTHONPATH=$W/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
NAME=$1; EPOCHS=$2; SEED=$3; shift 3
mkdir -p run_logs
COMET_KEY=$(sed -n 's/^ *api_key *= *//p' ~/.comet.config | head -1)      # the key shipped in the released hparams.yaml is not used
$PY $W/src/esm_msr/training.py \
  --experiment_name $NAME --version 0 \
  --raw_data_file /home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
  --af_model_folder /home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs \
  --split_file ${SPLIT:-$D/repo/data/splits_oct06_capped.pkl} \
  --cache_path $D/cache_main_capped --benchmark_data_path $D/repo/data/preprocessed \
  --checkpoint_path training_checkpoints --log_dir training_logs \
  --num_epochs $EPOCHS --seed $SEED --num_workers ${NUM_WORKERS:-2} \
  $(cat $W/baseline/recipe_args.txt) \
  --comet_project_name ${COMET_PROJECT:-esm-msr-agent-oct06-capped-baseline} --comet_api_key "$COMET_KEY" "$@" > run_logs/$NAME.log 2>&1
echo "DONE rc=$?" >> run_logs/$NAME.log
