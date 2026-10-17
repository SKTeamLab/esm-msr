#!/bin/bash
#SBATCH --account=def-skal
#SBATCH --job-name=msr-oct06-test
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Test training run on the oct06 capped split. Needs COMET_API_KEY in the environment.
# The loss weights default to 0 (no gradient at all), so the ones from LoRA_models/esm-msr-small/hparams.yaml are set here.
# Compute nodes have no internet: Comet is written to disk (--comet_offline); upload from a login node afterwards with
#   COMET_API_KEY=... comet upload $R/logs/comet_offline/*.zip
cd /lustre10/scratch/sareeves/esm-msr-devel
hostname; nvidia-smi
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source venv_rorqual/bin/activate
export HF_HUB_OFFLINE=1 COMET_DISABLE_AUTO_LOGGING=1
T=/scratch/sareeves/PSLMs/data/tsuboyama
R=/scratch/sareeves/esm-msr-devel-runs/oct06-capped-test
mkdir -p $R
python src/esm_msr/training.py \
  --experiment_name oct06_capped_test --version 0 \
  --raw_data_file $T/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
  --af_model_folder $T/AlphaFold_model_PDBs \
  --split_file data/splits_oct06_capped.pkl \
  --benchmark_data_path data/preprocessed \
  --cache_path $R/cache --checkpoint_path $R/checkpoints --log_dir $R/logs \
  --num_workers 5 \
  --lambda_reg_wt 1.0 --lambda_rank_wt 1.0 --lambda_reg_mt 1.0 \
  --comet_offline \
  --comet_api_key "$COMET_API_KEY" --comet_project_name esm-msr-agent-oct06-capped
