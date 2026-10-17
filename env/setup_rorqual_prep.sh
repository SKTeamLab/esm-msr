#!/bin/bash
# venv for src/esm_msr/preprocess.py (external-dataset structure repair): adds pdbfixer, which needs the openmm module.
# Modules must be loaded BEFORE the venv is created so pip sees openmm. Login node.
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV=${ENV:-$REPO/venv_prep}
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load openmm/8.5.2
module load python/3.11
virtualenv --no-download "$ENV" && source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch esm biopython pandas numpy requests scipy pdbfixer tqdm
pip uninstall -y torchtext          # pins torch 2.3.1, which breaks the transformers esm pulls in
pip install --no-index -U torch
pip install --no-index -e "$REPO" --no-deps
