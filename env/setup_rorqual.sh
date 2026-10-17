#!/bin/bash
# Build the training venv on Rorqual (Alliance). Login node. Usage: bash env/setup_rorqual.sh  -> <repo>/venv_rorqual
# pdbfixer/openmm are NOT installed: only src/esm_msr/preprocess.py needs them (see env/setup_rorqual_prep.sh).
# The Alliance pandas/scipy wheels are newer than pyproject's pins (pandas<3, scipy<1.15); nothing here pins them.
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV=${ENV:-$REPO/venv_rorqual}
module purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11
virtualenv --no-download "$ENV" && source "$ENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch lightning peft comet_ml esm biopython scipy pandas numpy tqdm httpx pytest
# esm's wheel pulls torchtext, which pins torch 2.3.1; that torch is too old for the transformers it also pulls in
pip uninstall -y torchtext
pip install --no-index -U torch torchvision
pip install --no-index -e "$REPO" --no-deps
