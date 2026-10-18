#!/bin/bash
#SBATCH --account=def-skal
#SBATCH --job-name=msr-ext-test
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
cd /lustre10/scratch/sareeves/esm-msr-devel
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load openmm/8.5.2
module load python/3.11
source venv_prep/bin/activate
python env/external_structs_test.py --phase process --root /scratch/sareeves/esm-msr-devel-runs/pipeline-test/external
