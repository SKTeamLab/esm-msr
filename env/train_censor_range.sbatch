#!/bin/bash
#SBATCH --account=def-skal
#SBATCH --job-name=msr-censor-range
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# 10-epoch training run on a split (SPLIT=splits_<name>, a pickle under data/) with --censor_range; reuses the dataset cache of
# train_test.sbatch on the same split. Needs COMET_API_KEY in the environment.
# Compute nodes have no internet: Comet is written to disk (--comet_offline); upload from a login node afterwards with
#   COMET_API_KEY=... comet upload $R/logs/comet_offline/*.zip
cd /lustre10/scratch/sareeves/esm-msr-devel
hostname; nvidia-smi
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source venv_rorqual/bin/activate
export HF_HUB_OFFLINE=1 COMET_DISABLE_AUTO_LOGGING=1
SPLIT=${SPLIT:?set SPLIT, the split under data/ (e.g. SPLIT=splits_oct06_capped)}
TAG=${SPLIT#splits_}; TAG=${TAG//_/-}       # splits_oct06_capped -> oct06-capped: run directories and the Comet project
T=/scratch/sareeves/PSLMs/data/tsuboyama
R=/scratch/sareeves/esm-msr-devel-runs/$TAG-censor-range
mkdir -p $R
python src/esm_msr/training.py \
  --experiment_name ${TAG//-/_}_censor_range --version 1 --num_epochs 10 --censor_range \
  --raw_data_file $T/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
  --af_model_folder $T/AlphaFold_model_PDBs \
  --split_file data/$SPLIT.pkl \
  --benchmark_data_path data/preprocessed \
  --cache_path /scratch/sareeves/esm-msr-devel-runs/$TAG-test/cache --checkpoint_path $R/checkpoints --log_dir $R/logs \
  --num_workers 5 --micro_batch_size 64 \
  --comet_offline \
  --comet_api_key "$COMET_API_KEY" --comet_project_name esm-msr-agent-$TAG
