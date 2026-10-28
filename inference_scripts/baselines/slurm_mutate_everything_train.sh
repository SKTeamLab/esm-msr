#!/bin/bash
#SBATCH --job-name=me_oct06
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX6000:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Upstream MutateEverything (AF backbone), README recipe unchanged: stage 1 singles (20 ep), stage 2 doubles (100 ep)
# fine-tuned from stage 1, 3 GPUs, batch 1. Data: ESM-MSR-matched oct06 split (prepare/build_mutate_everything_data.py).
# Past runs of this recipe on gpu2: ~17 h for both stages on a 118-protein train set; this one has 192.
# Usage: SEED=1 [STAGE=single|double|both] sbatch inference_scripts/baselines/slurm_mutate_everything_train.sh
# (checkpoints and logs land in logs/ of the MutateEverything checkout, which is also where the inference script reads them)
#
# MUT_BACKBONES=include|exclude picks which prepared data dir to train on: whether mutant-background libraries
# ('2K52_L42A', WT backbone + background-mutated sequence) are present in train/val. The exclusion happens in
# prepare/prepare_oct06_data.py; here it only resolves SPLIT_NAME, i.e. which data/<name>/mutations tree is read.
# SPLIT_NAME set explicitly wins, so existing runs are unaffected.
set -eo pipefail
hostname; nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || { echo "unhealthy GPU in allocation ($CUDA_VISIBLE_DEVICES)"; exit 1; }  # gpu2 idx 3 (D8:00.0) is unresponsive
module purge
unset PYTHONPATH
export PATH=${ME_ENV_BIN:-$HOME/.conda/envs/mutate_everything/bin}:$PATH        # the conda env of envs/mutate_everything.yml
cd "${MUTATE_EVERYTHING_DIR:-${THIRD_PARTY_DIR:-$HOME/software}/MutateEverything-oct06}"
SEED=${SEED:-1}; STAGE=${STAGE:-both}
MUT_BACKBONES=${MUT_BACKBONES:-include}
case $MUT_BACKBONES in
  include) DEF=oct06_capped;;
  exclude) DEF=oct06_nobb;;
  *) echo "bad MUT_BACKBONES=$MUT_BACKBONES"; exit 1;;
esac
S=${SPLIT_NAME:-$DEF}
D=data/$S/mutations
echo "mut_backbones=$MUT_BACKBONES split_name=$S stage=$STAGE seed=$SEED"
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
    --finetune logs/af_single_${S}_seed$SEED/checkpoint-19.pth \
    --seed $SEED \
    --disable_wandb \
    --output_dir logs/af_double_${S}_seed$SEED
fi
