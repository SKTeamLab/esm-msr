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
  MSR_BASELINE_DATA          the prepared item tables (prepare/prepare_baseline_data.py output, one directory per split and
                             variant) a baseline is trained on and scored on Tsuboyama with; no default, the caller names it
  MSR_PGYM_INPUTS            pgym_inputs/ (preprocessing/pgym_preprocess.py)
  MSR_PGYM_STRUCTURES        ProteinGym_AF2_structures
"""
import json
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
BASELINE_DATA = os.environ.get('MSR_BASELINE_DATA') or None      # a prepared directory belongs to one split: no default
PGYM_INPUTS = _env('MSR_PGYM_INPUTS', REPO_ROOT / 'pgym_inputs')
PGYM_STRUCTURES = _env('MSR_PGYM_STRUCTURES', HOME / 'ProteinGym' / 'ProteinGym_AF2_structures')

# the structure prefix baked into the preprocessed tables: <anything>/esm-msr/data/structures
_BAKED = re.compile(r'^.*?/esm-msr/data/structures')


def fix_pdb_path(p):
    """A pdb_file from a preprocessed table pointed at this machine's structure directory."""
    return _BAKED.sub(str(STRUCTURES_DIR), str(p), count=1)


def fix_pdb_column(series):
    return series.astype(str).str.replace(_BAKED, str(STRUCTURES_DIR), regex=True)


def split_stem(split):
    """'data/splits_x.pkl' and 'splits_x' are the same split."""
    name = os.path.basename(str(split))
    return name[:-4] if name.endswith('.pkl') else name


def split_file(split):
    """The pickle of a split given as a path or as a name under data/."""
    return Path(split) if Path(split).is_file() else DATA_DIR / (split_stem(split) + '.pkl')


def variant_name(split, keep_all_doubles=False, mut_backbones='include'):
    """The name of a split's prepared baseline tables (prepare_baseline_data.py): the split's stem without a `splits_` prefix,
    `_allbl` when every measured double is kept, `_nobb` when mutant-background libraries are left out. It is also the name of
    MutateEverything's data/<name> directory (build_mutate_everything_data.py)."""
    stem = split_stem(split)
    stem = stem[len('splits_'):] if stem.startswith('splits_') else stem
    return stem + ('_allbl' if keep_all_doubles else '') + ('_nobb' if mut_backbones == 'exclude' else '')


def baseline_data(path=None):
    """The prepared item-table directory: `path`, else $MSR_BASELINE_DATA."""
    p = path or BASELINE_DATA
    if not p:
        raise SystemExit('no prepared baseline tables: pass --data_dir or set MSR_BASELINE_DATA to the output directory of '
                         'prepare/prepare_baseline_data.py')
    return str(require(p, 'the prepared baseline tables').resolve())      # absolute: ThermoMPNN's loaders chdir


def variant_info(data_dir):
    """variant.json of a prepared directory (name, split, flags), {} for tables prepared before it existed."""
    p = Path(data_dir) / 'variant.json'
    return json.load(open(p)) if p.exists() else {}


def check_split(data_dir, split):
    """Refuse to train on or score with tables prepared from another split than `split` (None: no check)."""
    have = variant_info(data_dir).get('split')
    if split and have and split_stem(have) != split_stem(split):
        raise SystemExit(f'{data_dir} holds the tables of split {split_stem(have)}, not {split_stem(split)}')


def require(path, what):
    """A clear failure for a missing input instead of an error three calls deeper."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{what} not found at {path} (see inference_scripts/baselines/paths.py for the variable that moves it)')
    return Path(path)
