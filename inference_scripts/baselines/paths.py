"""Where everything the baseline inference scripts need lives, in one place. Every location can be overridden by an
environment variable; the defaults are the layout `setup_third_party.sh` creates, relative to the home directory and to
this repository, so nothing here names a user or a machine.

  REPO_ROOT                  this repository (derived)
  THIRD_PARTY_DIR            ~/software          upstream clones at the pinned commits (third_party.lock)
    THERMOMPNN_DIR           $THIRD_PARTY_DIR/ThermoMPNN-D
    MUTATE_EVERYTHING_DIR    $THIRD_PARTY_DIR/MutateEverything-oct06   upstream MutateEverything, unmodified, plus data/ and weights
    SPURS_DIR                $THIRD_PARTY_DIR/SPURS
    PROTEINMPNN_DIR          $THIRD_PARTY_DIR/ProteinMPNN
  MSR_DATA_DIR               $REPO_ROOT/data     preprocessed benchmark tables (data/preprocessed/*.csv), splits, Tsuboyama tables
  MSR_TSUBOYAMA_DIR          $MSR_DATA_DIR/tsuboyama  (Tsuboyama2023_Dataset2_Dataset3_20230416.csv, AlphaFold_model_PDBs/)
  MSR_STRUCTURES_DIR         structures the preprocessed tables point at (their pdb_file column carries the prefix of the
                             machine that built them; `fix_pdb_path` rewrites it to this directory)
  MSR_DOMAINOME_TABLE        domainome_mapped_2026.csv
  MSR_ROSETTA_DIR            Tsuboyama FINAL_results (Rosetta models; ThermoMPNN-D's double augmentation, training only)
  MSR_BASELINE_DATA          the prepared item tables (prepare/prepare_oct06_data.py output) a baseline was trained on
  MSR_PGYM_INPUTS            pgym_inputs/ (preprocessing/pgym_preprocess.py)
  MSR_PGYM_STRUCTURES        ProteinGym_AF2_structures
"""
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()


def _env(name, default):
    return Path(os.environ.get(name) or default)


THIRD_PARTY_DIR = _env('THIRD_PARTY_DIR', HOME / 'software')
THERMOMPNN_DIR = _env('THERMOMPNN_DIR', THIRD_PARTY_DIR / 'ThermoMPNN-D')
MUTATE_EVERYTHING_DIR = _env('MUTATE_EVERYTHING_DIR', THIRD_PARTY_DIR / 'MutateEverything-oct06')
SPURS_DIR = _env('SPURS_DIR', THIRD_PARTY_DIR / 'SPURS')
PROTEINMPNN_DIR = _env('PROTEINMPNN_DIR', THIRD_PARTY_DIR / 'ProteinMPNN')

DATA_DIR = _env('MSR_DATA_DIR', REPO_ROOT / 'data')
PREPROCESSED = DATA_DIR / 'preprocessed'
TSUBOYAMA_DIR = _env('MSR_TSUBOYAMA_DIR', DATA_DIR / 'tsuboyama')
STRUCTURES_DIR = _env('MSR_STRUCTURES_DIR', THIRD_PARTY_DIR / 'esm-msr' / 'data' / 'structures')
DOMAINOME_TABLE = _env('MSR_DOMAINOME_TABLE', THIRD_PARTY_DIR / 'esm-msr' / 'data' / 'domainome1' / 'domainome_mapped_2026.csv')
ROSETTA_DIR = _env('MSR_ROSETTA_DIR', THIRD_PARTY_DIR / 'esm-msr' / 'data' / 'tsuboyama' / 'FINAL_results')
BASELINE_DATA = _env('MSR_BASELINE_DATA', REPO_ROOT / 'outputs' / 'baselines_capped_allbl' / 'data')
PGYM_INPUTS = _env('MSR_PGYM_INPUTS', REPO_ROOT / 'pgym_inputs')
PGYM_STRUCTURES = _env('MSR_PGYM_STRUCTURES', HOME / 'ProteinGym' / 'ProteinGym_AF2_structures')

# the structure prefix baked into the preprocessed tables: <anything>/esm-msr/data/structures
_BAKED = re.compile(r'^.*?/esm-msr/data/structures')


def fix_pdb_path(p):
    """A pdb_file from a preprocessed table pointed at this machine's structure directory."""
    return _BAKED.sub(str(STRUCTURES_DIR), str(p), count=1)


def fix_pdb_column(series):
    return series.astype(str).str.replace(_BAKED, str(STRUCTURES_DIR), regex=True)


def require(path, what):
    """A clear failure for a missing input instead of an error three calls deeper."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{what} not found at {path} (see inference_scripts/baselines/paths.py for the variable that moves it)')
    return Path(path)
