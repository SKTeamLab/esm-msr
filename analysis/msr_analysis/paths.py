"""Canonical paths, environment overrides, dataset-to-folder tables, and atomic cache writers.

All repository paths resolve relative to ``REPO_ROOT`` (never hardcoding a clone directory name such as ``repo/``),
and external benchmark/prediction directories resolve under ``MSR_ROOT`` (default ``~/playground/esm-msr-devel``).
Every cache writer uses ``tempfile.NamedTemporaryFile`` + ``os.replace`` so gitignored symlinks in worktrees are
unlinked/replaced atomically and never written through.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ANALYSIS_ROOT.parent
SRC_DIR = REPO_ROOT / 'src'
DATA_DIR = Path(os.environ.get('MSR_DATA_DIR', REPO_ROOT / 'data')).expanduser()

CACHE = ANALYSIS_ROOT / 'cache'
REFERENCE = ANALYSIS_ROOT / 'reference'
RESULTS = ANALYSIS_ROOT / 'results'
RESULTS_DIR = RESULTS

MSR_ROOT = Path(os.environ.get('MSR_ROOT', '~/playground/esm-msr-devel')).expanduser()
ROOT = MSR_ROOT  # alias kept for compatibility

MSR = MSR_ROOT / 'msr_predictions'
FINAL_BENCH = Path(os.environ.get('MSR_FINAL_BENCH', MSR_ROOT / 'final_bench' / 'predictions')).expanduser()
P0 = FINAL_BENCH
FINAL_CTX = Path(os.environ.get('MSR_FINAL_CTX', MSR_ROOT / 'final_ctx')).expanduser()

FINAL_PGYM_WT = Path(os.environ.get('MSR_FINAL_PGYM_WT', MSR_ROOT / 'final_pgym_wt')).expanduser()
FINAL_PGYM_WT_S05 = Path(os.environ.get('MSR_FINAL_PGYM_WT_S05', MSR_ROOT / 'final_pgym_wt_sigma0.5')).expanduser()
FINAL_PGYM_COMB = Path(os.environ.get('MSR_FINAL_PGYM_COMB', MSR_ROOT / 'final_pgym_comb')).expanduser()
FINAL_PGYM_COMB_S05 = Path(os.environ.get('MSR_FINAL_PGYM_COMB_S05', MSR_ROOT / 'final_pgym_comb_sigma0.5')).expanduser()
FINAL_PGYM_XMASK = Path(os.environ.get('MSR_FINAL_PGYM_XMASK', MSR_ROOT / 'final_pgym_wt_xmask')).expanduser()
ESM3_S0_PGYM_COMB = Path(
    os.environ.get('MSR_ESM3_S0_PGYM_COMB', MSR_ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0')
).expanduser()

FINAL_MGNIFY = Path(os.environ.get('MSR_MGNIFY_RUNS', MSR_ROOT / 'final_mgnify')).expanduser()
FINAL_MGNIFY_S05 = Path(os.environ.get('MSR_MGNIFY_RUNS_S05', MSR_ROOT / 'final_mgnify_sigma0.5')).expanduser()
FINAL_MGNIFY_BY_SIGMA = {1.0: FINAL_MGNIFY, 0.5: FINAL_MGNIFY_S05}
MGNIFY_TABLE = Path(os.environ.get('MSR_MGNIFY_TABLE', DATA_DIR / 'mgnify' / 'mgnify_mapped.csv')).expanduser()

BASE = MSR_ROOT / 'baselines'
RETRAINED = BASE / 'retrained'
MEGA_RAW = RETRAINED / 'mega_splits' / 'tracked'
PUBLIC = BASE / 'public'
TASKS = PUBLIC / '_tasks'
PGYM = BASE / 'pgym_per_dms'
PGYM_PER_DMS_DIR = PGYM

PGYM_INPUTS = Path(os.environ.get('MSR_PGYM_INPUTS', MSR_ROOT / 'pgym_inputs')).expanduser()
PGYM_INPUTS_DIR = PGYM_INPUTS
PGYM_PUBLIC_ZIP = Path(
    os.environ.get('MSR_PGYM_PUBLIC_ZIP', Path.home() / 'stability-baselines' / 'pgym_public' / 'zero_shot_substitutions_scores.zip')
).expanduser()
PGYM_DIR = Path(os.environ.get('MSR_PROTEINGYM_DIR', MSR_ROOT / 'ProteinGym')).expanduser()
PGYM_REF_CSV = Path(os.environ.get('MSR_PGYM_REF_CSV', PGYM_DIR / 'DMS_substitutions.csv')).expanduser()
PGYM_AF2_DIR = PGYM_DIR / 'ProteinGym_AF2_structures'
DOMAINOME_AF2_DIR = DATA_DIR / 'domainome1' / 'alphafold_structures'

ROSETTA_EXT_DIR = Path(os.environ.get('MSR_ROSETTA_EXT', MSR_ROOT / 'rosetta_external_predictions')).expanduser()
RESOURCES_TRACKED = Path(os.environ.get('MSR_RESOURCES_TRACKED', MSR_ROOT / 'resources_tracked')).expanduser()
TRAINING_LOGS = Path(os.environ.get('MSR_TRAINING_LOGS', MSR_ROOT / 'training_logs')).expanduser()

# External feature / structure / split-provenance inputs with explicit environment overrides
TSUBOYAMA_PDB_DIR = Path(
    os.environ.get('MSR_TSUBOYAMA_PDB_DIR', Path.home() / 'software' / 'esm-msr' / 'data' / 'tsuboyama' / 'AlphaFold_model_PDBs')
).expanduser()
TSUBOYAMA_FEATS_CSV = Path(
    os.environ.get('MSR_TSUBOYAMA_FEATS_CSV', Path.home() / 'PSLMs' / 'data' / 'features' / 'tsuboyama_all_subs_corrected_feats.csv')
).expanduser()
SALVAGED_HOMOLOGY_CSV = Path(
    os.environ.get(
        'MSR_SALVAGED_HOMOLOGY_CSV',
        MSR_ROOT / 'analysis' / 'salvaged_2026-10-16' / 'structure-homology-splits-e5ca04' / 'data' / 'tsuboyama' / 'homology_work' / 'homology_pairs.csv',
    )
).expanduser()

# Single canonical dataset <-> runner-folder mapping across testing harnesses and collectors
RUNNER_FOLDER = {
    'tsuboyama_capped_val': 'splits_oct06_capped-val',
    'tsuboyama_capped_test': 'splits_oct06_capped-test',
    'tsuboyama_mega_val': 'mega_splits-val',
    'tsuboyama_mega_test': 'mega_splits-test',
    'tsuboyama_hyperopt_val': 'hyperopt_splits-val',
    'tsuboyama_hyperopt_test': 'hyperopt_splits-test',
    'ptmul': 'PTMUL',
}
FOLDER_TO_DATASET = {
    **{v: k for k, v in RUNNER_FOLDER.items()},
    'hyperopt_splits-validation': 'tsuboyama_hyperopt_val',
}
MEGA_OWN_FOLDER = {
    'tsuboyama_capped_val': 'mega_splits-val',
    'tsuboyama_capped_test': 'mega_splits-test',
}
PARENT_DATASET = {'s461': 's669', 'k2369': 'k3822', 'ptmuld': 'ptmul'}


def runner_folder(ds: str, mega_own: bool = False) -> str:
    """Return the testing-harness directory name corresponding to canonical dataset ``ds``."""
    if mega_own and ds in MEGA_OWN_FOLDER:
        return MEGA_OWN_FOLDER[ds]
    return RUNNER_FOLDER.get(ds, ds)


def dataset_from_folder(raw: str) -> str:
    """Map a testing-harness folder name (e.g. ``splits_oct06_capped-test``, ``PTMUL``) to its canonical dataset id."""
    return FOLDER_TO_DATASET.get(raw, raw)


def results_dir(name: str | None = None, split: str = 'capped', create: bool = True) -> Path:
    """Return the output directory under ``analysis/results`` for ``split`` ('capped' or 'mega')."""
    if split not in ('capped', 'mega'):
        raise ValueError(f"split must be 'capped' or 'mega', got {split!r}")
    base = (RESULTS / '03_tsuboyama' / 'mega_splits') if split == 'mega' else RESULTS
    out = base / name if name else base
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def source_signature(paths: Iterable[str | Path]) -> tuple[tuple[str, int, int], ...]:
    """Return sorted ``(resolved_name, st_mtime_ns, st_size)`` tuples for cache invalidation metadata."""
    sig = []
    for p in paths:
        pp = Path(p)
        if pp.exists():
            st = pp.stat()
            sig.append((str(pp), int(st.st_mtime_ns), int(st.st_size)))
    sig.sort()
    return tuple(sig)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` via temp-file + ``os.replace`` so symlinks are never written through."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f'.{dst.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        os.replace(tmp_name, dst)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return dst


def atomic_write_text(path: str | Path, text: str, encoding: str = 'utf-8') -> Path:
    """Atomically write UTF-8 text to ``path`` without mutating symlink targets."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_pickle_dump(obj, path: str | Path, protocol: int = 4) -> Path:
    """Atomically serialize ``obj`` to ``path`` via temp-file + ``os.replace``."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f'.{dst.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            pickle.dump(obj, fh, protocol=protocol)
        os.replace(tmp_name, dst)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return dst


def atomic_to_csv(df, path: str | Path, **kwargs) -> Path:
    """Atomically write a pandas DataFrame to CSV via temp-file + ``os.replace``."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f'.{dst.name}.', suffix='.tmp')
    os.close(fd)
    try:
        df.to_csv(tmp_name, **kwargs)
        os.replace(tmp_name, dst)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return dst
