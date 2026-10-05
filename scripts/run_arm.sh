#!/bin/bash
# Usage: scripts/run_arm.sh RUN_NAME EPOCHS SEED [extra training flags...]
# Canonical command (docs/epistasis_training_handoff.md §2) on THIS worktree's code and cache_v7.
WT=/home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/censored-margin-ranking-9b239c
cd /home/sareeves/playground/esm-msr-devel
export PYTHONPATH=$WT/src HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sareeves/miniconda3/envs/msr_venv/bin/python
NAME=$1; EPOCHS=$2; SEED=$3; shift 3
mkdir -p run_logs
# Comet: key read from ~/.comet.config at launch (never written into this file or the log)
COMET_KEY=$(sed -n 's/^ *api_key *= *//p' ~/.comet.config | head -1)
"$PY" $WT/src/esm_msr/training.py \
  --experiment_name $NAME --version 0 \
  --raw_data_file '/home/sareeves/software/esm-msr/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv' \
  --af_model_folder '/home/sareeves/software/esm-msr/data/tsuboyama/AlphaFold_model_PDBs' \
  --split_file '/home/sareeves/software/esm-msr/data/hyperopt_splits.pkl' \
  --cache_path cache_v7 \
  --benchmark_data_path $WT/data/preprocessed \
  --checkpoint_path training_checkpoints --log_dir training_logs \
  --num_epochs $EPOCHS --seed $SEED \
  --dataloading cycle --loader_strategy all \
  --subset_caps single=None cond=None native_cond=None \
  --min_additive_dG -1.0 --subfloor_rank_only \
  --lora_rank_wt 2  --lora_alpha_wt 4  --lora_dropout_wt 0.1 --target_mode_wt expanded \
  --lora_rank_mt 16 --lora_alpha_mt 16 --lora_dropout_mt 0.1 --target_mode_mt expanded \
  --incl_sequence_head_wt --incl_sequence_head_mt \
  --adapter_mode dual --lora_mode ensemble \
  --lambda_reg_wt 1.0 --lambda_rank_wt 1.0 \
  --lambda_reg_mt 1.0 --lambda_rank_mt 1.0 --flip_list_min 4 \
  --mt_single_anchor_weight 0.5 --cond_weight 0.5 --native_cond_weight 1.0 \
  --reg_loss mse --precision bf16-mixed \
  --batch_size 256 --micro_batch_size 64 --subset_size 16 \
  --learning_rate 2e-4 --lr_warmup_steps 500 \
  --shared_scale_init 0.3 --detach_ensemble_input \
  --num_workers 4 --log_every_n_steps 25 --save_top_k 3 \
  --monitor_metric val_rho_flip_avg --monitor_mode max \
  --comet_api_key "$COMET_KEY" "$@" > run_logs/$NAME.log 2>&1
echo "DONE rc=$?" >> run_logs/$NAME.log
