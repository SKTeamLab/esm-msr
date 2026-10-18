#!/bin/bash
#SBATCH --account=def-skal
#SBATCH --job-name=msr-prep-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only: rebuild the dataset caches from freshly downloaded Tsuboyama data into a separate folder, for comparison.
cd /lustre10/scratch/sareeves/esm-msr-devel
hostname
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
source venv_rorqual/bin/activate
export HF_HUB_OFFLINE=1
P=/scratch/sareeves/esm-msr-devel-runs/pipeline-test
python env/preprocess_only.py \
  --experiment_name prep_test \
  --raw_data_file $P/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
  --af_model_folder $P/tsuboyama/AlphaFold_model_PDBs \
  --split_file data/splits_oct06_capped.pkl \
  --benchmark_data_path data/preprocessed \
  --cache_path $P/cache --num_workers 0
