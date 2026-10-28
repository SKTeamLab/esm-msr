#!/bin/bash
#SBATCH --job-name=me_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Upstream MutateEverything (AF backbone), README recipe unchanged: stage 1 singles (20 ep), stage 2 doubles (100 ep)
# fine-tuned from stage 1, 3 GPUs, batch 1. Data: the ESM-MSR-matched tables of one split
# (prepare/prepare_baseline_data.py, then prepare/build_mutate_everything_data.py).
# Past runs of this recipe on gpu2: ~17 h for both stages on a 118-protein train set; this one has 192.
# Usage: SEED=1 SPLIT_NAME=<name> [STAGE=single|double|both] [STAGE1_SPLIT=<name>] sbatch training_scripts/baselines/slurm_mutate_everything_train.sh
# SPLIT_NAME: data/<name>/mutations of the MutateEverything checkout, as build_mutate_everything_data.py wrote it (the variant name
# paths.variant_name gives: the split's stem without `splits_`, plus `_allbl` when every measured double is kept, `_nobb` when
# mutant-background libraries are left out; those are choices made when preparing).
# STAGE1_SPLIT: take the stage-1 checkpoint from logs/af_single_<STAGE1_SPLIT>_seed<SEED> (default: SPLIT_NAME), e.g. a stage 2
# on <split>_allbl starts from the <split> stage 1, whose single mutations are the same rows
# (checkpoints and logs land in logs/ of the MutateEverything checkout, which is also where the inference script reads them)
set -eo pipefail
hostname; nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU in allocation ($CUDA_VISIBLE_DEVICES)"; exit 1; }  # gpu2 idx 3 (D8:00.0) is unresponsive
module purge
unset PYTHONPATH
export PATH=${ME_ENV_BIN:-$HOME/.conda/envs/mutate_everything/bin}:$PATH        # the conda env of envs/mutate_everything.yml
cd "${MUTATE_EVERYTHING_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/MutateEverything-oct06}"
SEED=${SEED:-1}; STAGE=${STAGE:-both}
S=${SPLIT_NAME:?set SPLIT_NAME (the data/<name> directory build_mutate_everything_data.py wrote)}
D=data/$S/mutations
echo "split_name=$S stage=$STAGE seed=$SEED"
EVAL=$D/${S}_val.csv,$D/${S}_test.csv
PORT=$((29500 + SLURM_JOB_ID % 1000))

if [ "$STAGE" != double ]; then
torchrun --nproc_per_node=3 --master_port=$PORT main_train.py \
    --batch_size 1 \
    --finetune_backbone models/finetuning_ptm_2.pt \
    --data_path $D/${S}_train.csv \
    --eval_data_paths $EVAL \
    --dist_eval \
    --backbone af \
    --lambda_double 0. --lambda_single 1. \
    --eval_period 999 --epochs 20 --warmup_epochs 2 --save_period 20 \
    --seed $SEED \
    --disable_wandb \
    --output_dir logs/af_single_${S}_seed$SEED
fi

if [ "$STAGE" != single ]; then
torchrun --nproc_per_node=3 --master_port=$PORT main_train.py \
    --batch_size 1 \
    --finetune_backbone models/finetuning_ptm_2.pt \
    --data_path $D/${S}_train.csv \
    --eval_data_paths $EVAL \
    --dist_eval \
    --backbone af \
    --eval_period 1000 --epochs 100 --warmup_epochs 10 --save_period 100 \
    --finetune logs/af_single_${STAGE1_SPLIT:-$S}_seed$SEED/checkpoint-19.pth \
    --seed $SEED \
    --disable_wandb \
    --output_dir logs/af_double_${S}_seed$SEED
fi
