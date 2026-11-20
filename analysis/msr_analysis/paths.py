"""Canonical paths, environment overrides, dataset-to-folder tables, and atomic cache writers.

All repository paths resolve relative to ``REPO_ROOT`` (never hardcoding a clone directory name such as ``repo/``),
and external benchmark/prediction directories resolve under ``MSR_ROOT`` (default ``~/playground/esm-msr-devel``).
Every cache writer uses ``tempfile.NamedTemporaryFile`` + ``os.replace`` so gitignored symlinks in worktrees are
unlinked/replaced atomically and never written through.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Iterable

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ANALYSIS_ROOT.parent
SRC_DIR = REPO_ROOT / 'src'
DATA_DIR = Path(os.environ.get('MSR_DATA_DIR', REPO_ROOT / 'data')).expanduser()

CACHE = ANALYSIS_ROOT / 'cache'
REFERENCE = ANALYSIS_ROOT / 'reference'
RESULTS = ANALYSIS_ROOT / 'results'
RESULTS_DIR = RESULTS

# Canonical intermediate cache & top-level result file paths
CORE_RHO_BOOTSTRAP_CSV = CACHE / 'core_rho_bootstrap.csv'
MANIFEST_OUTPUTS_CSV = RESULTS / 'manifest_outputs.csv'
TABLE_FIGURE_MAP_CSV = RESULTS / 'table_figure_map.csv'

# Canonical top-level and key nested directories under analysis/results/
FOR_REVIEWERS_DIR = RESULTS / '00_for_reviewers'
REVIEWER_REFERENCE_DIR = FOR_REVIEWERS_DIR / 'reference'
VALIDATION_VS_BENCHMARKS_CSV = REVIEWER_REFERENCE_DIR / 'validation_vs_benchmarks.csv'
MAIN_TEXT_FIGS_DIR = RESULTS / '00_main_text_figs'
SUPPLEMENTARY_FIGS_DIR = RESULTS / '00_supplementary_figs'
SPLIT_CONSTRUCTION_DIR = SUPPLEMENTARY_FIGS_DIR / 'split_construction'
MEGASCALE_DIR = RESULTS / '01_megascale'
MEGA_SPLITS_DIR = MEGASCALE_DIR / 'mega_splits'
THERMOSTABILITY_DIR = RESULTS / '02_thermostability'
PROTEINGYM_RESULTS_DIR = RESULTS / '03_proteingym'
DOMAINOME_RESULTS_DIR = RESULTS / '04_domainome'
MGNIFY_RESULTS_DIR = RESULTS / '05_mgnify'
SUMMARY_RESULTS_DIR = RESULTS / '06_summary'
SCREENING_OVERVIEW_DIR = RESULTS / '07_screening_overview'
SEQUENCE_LENGTH_DIR = RESULTS / '08_sequence_length'
ABLATION_RESULTS_DIR = RESULTS / '09_ablation'
INFERENCE_RESOURCES_DIR = RESULTS / '10_inference_resources'
MISC_RESULTS_DIR = RESULTS / '99_misc'
DMS3_RESULTS_DIR = MISC_RESULTS_DIR / 'dms3'

CANONICAL_RESULT_TOPS = (
    '00_for_reviewers',
    '00_main_text_figs',
    '00_supplementary_figs',
    '01_megascale',
    '02_thermostability',
    '03_proteingym',
    '04_domainome',
    '05_mgnify',
    '06_summary',
    '07_screening_overview',
    '08_sequence_length',
    '09_ablation',
    '10_inference_resources',
    '99_misc',
)

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


# Split-aware MegaScale routes (resolved under 01_megascale/ for 'capped' vs 01_megascale/mega_splits/ for 'mega')
_MEGA_SPLIT_ROUTES = MappingProxyType({
    'megascale': '',
    '01_megascale': '',
    'epistasis': 'epistasis',
    '01_megascale/epistasis': 'epistasis',
    'dG_additive_vs_pred_dddG': 'epistasis/dG_additive_vs_pred_dddG',
    'epistasis/dG_additive_vs_pred_dddG': 'epistasis/dG_additive_vs_pred_dddG',
    '01_megascale/epistasis/dG_additive_vs_pred_dddG': 'epistasis/dG_additive_vs_pred_dddG',
    'dddG_scatters': 'epistasis/dddG_scatters',
    'epistasis/dddG_scatters': 'epistasis/dddG_scatters',
    '01_megascale/epistasis/dddG_scatters': 'epistasis/dddG_scatters',
    'pair_heatmaps': 'epistasis/pair_heatmaps',
    'epistasis/pair_heatmaps': 'epistasis/pair_heatmaps',
    '01_megascale/epistasis/pair_heatmaps': 'epistasis/pair_heatmaps',
    'gain_fit': 'gain_fit',
    '01_megascale/gain_fit': 'gain_fit',
    'megascale_screening': 'screening',
    '01_megascale/screening': 'screening',
    'property_gain': 'screening/property_gain',
    'megascale_property_gain': 'screening/property_gain',
    '01_megascale/screening/property_gain': 'screening/property_gain',
})

# Fixed canonical routes under analysis/results/ (independent of split)
_FIXED_RESULT_ROUTES = MappingProxyType({
    'for_reviewers': '00_for_reviewers',
    '00_for_reviewers': '00_for_reviewers',
    'reviewer_reference': '00_for_reviewers/reference',
    'reference': '00_for_reviewers/reference',
    '00_for_reviewers/reference': '00_for_reviewers/reference',
    'main_text_figs': '00_main_text_figs',
    '00_main_text_figs': '00_main_text_figs',
    'supplementary_figs': '00_supplementary_figs',
    '00_supplementary_figs': '00_supplementary_figs',
    'split_construction': '00_supplementary_figs/split_construction',
    '00_supplementary_figs/split_construction': '00_supplementary_figs/split_construction',
    'mega_splits': '01_megascale/mega_splits',
    '01_megascale/mega_splits': '01_megascale/mega_splits',
    'mega_provenance': '01_megascale/mega_splits/provenance',
    'provenance': '01_megascale/mega_splits/provenance',
    '01_megascale/mega_splits/provenance': '01_megascale/mega_splits/provenance',
    'thermostability': '02_thermostability',
    '02_thermostability': '02_thermostability',
    'rmse': '02_thermostability/rmse',
    'thermostability_rmse': '02_thermostability/rmse',
    '02_thermostability/rmse': '02_thermostability/rmse',
    'thermostability_screening': '02_thermostability/screening',
    '02_thermostability/screening': '02_thermostability/screening',
    'proteingym': '03_proteingym',
    '03_proteingym': '03_proteingym',
    'proteingym_screening': '03_proteingym/screening',
    'pg_screening': '03_proteingym/screening',
    'pg_activity': '03_proteingym/screening',
    '03_proteingym/screening': '03_proteingym/screening',
    'domainome': '04_domainome',
    '04_domainome': '04_domainome',
    'domainome_screening': '04_domainome/screening',
    '04_domainome/screening': '04_domainome/screening',
    'mgnify': '05_mgnify',
    '05_mgnify': '05_mgnify',
    'mgnify_epistasis': '05_mgnify/new_data_epistasis',
    'new_data_epistasis': '05_mgnify/new_data_epistasis',
    '05_mgnify/new_data_epistasis': '05_mgnify/new_data_epistasis',
    'rocklin_designs': '05_mgnify/rocklin_designs',
    '05_mgnify/rocklin_designs': '05_mgnify/rocklin_designs',
    'summary': '06_summary',
    '06_summary': '06_summary',
    'screening_overview': '07_screening_overview',
    '07_screening_overview': '07_screening_overview',
    'sequence_length': '08_sequence_length',
    '08_sequence_length': '08_sequence_length',
    'ablation': '09_ablation',
    '09_ablation': '09_ablation',
    'inference_resources': '10_inference_resources',
    '10_inference_resources': '10_inference_resources',
    'misc': '99_misc',
    '99_misc': '99_misc',
    'dms3': '99_misc/dms3',
    '99_misc/dms3': '99_misc/dms3',
    'dual_view': '99_misc/dual_view',
    '99_misc/dual_view': '99_misc/dual_view',
    'esm_msr_vs_esm3dg': '99_misc/esm_msr_vs_esm3dg',
    '99_misc/esm_msr_vs_esm3dg': '99_misc/esm_msr_vs_esm3dg',
    'faure_epistasis': '99_misc/faure_epistasis',
    '99_misc/faure_epistasis': '99_misc/faure_epistasis',
    'rosetta_multi_vs_additive': '99_misc/rosetta_multi_vs_additive',
    '99_misc/rosetta_multi_vs_additive': '99_misc/rosetta_multi_vs_additive',
    'sigma_mechanisms': '99_misc/sigma_mechanisms',
    '99_misc/sigma_mechanisms': '99_misc/sigma_mechanisms',
})

RESULT_DIRS = MappingProxyType({
    **{k: f'01_megascale/{v}'.rstrip('/') for k, v in _MEGA_SPLIT_ROUTES.items()},
    **_FIXED_RESULT_ROUTES,
})

_RETIRED_RESULT_HINTS = MappingProxyType({
    '00_core': "retired; use CORE_RHO_BOOTSTRAP_CSV (analysis/cache/core_rho_bootstrap.csv) or 'reviewer_reference' (00_for_reviewers/reference)",
    '01_summary': "renamed to 'summary' ('06_summary')",
    '03_tsuboyama': "renamed to 'megascale' ('01_megascale')",
    '04_dms_domainome': "split into 'domainome' ('04_domainome') and 'dms3' ('99_misc/dms3')",
    '05_proteingym': "renamed to 'proteingym' ('03_proteingym')",
    '06_epistasis': "moved to 'epistasis' ('01_megascale/epistasis') and 'dms3' ('99_misc/dms3')",
    '07_rmse': "moved to 'rmse' ('02_thermostability/rmse')",
    '08_screening': "split across 'megascale_screening', 'thermostability_screening', 'proteingym_screening', and 'domainome_screening'",
    '09_sequence_length': "renamed to 'sequence_length' ('08_sequence_length')",
    '10_leakage_and_reference': "moved reference items to 'reviewer_reference' ('00_for_reviewers/reference'); leak_* & clean_means.* dropped",
    '12_pair_heatmaps': "moved to 'pair_heatmaps' ('01_megascale/epistasis/pair_heatmaps')",
    '16_screening_overview': "renamed to 'screening_overview' ('07_screening_overview')",
    '17_gain_fit': "moved to 'gain_fit' ('01_megascale/gain_fit')",
    '18_mgnify': "renamed to 'mgnify' ('05_mgnify')",
    '19_rosetta_multi_vs_additive': "moved to 'rosetta_multi_vs_additive' ('99_misc/rosetta_multi_vs_additive')",
    '22_esm_msr_vs_esm3dg': "moved to 'esm_msr_vs_esm3dg' ('99_misc/esm_msr_vs_esm3dg')",
    '23_pg_ours_vs_published': "absorbed into 'supplementary_figs' ('00_supplementary_figs')",
    '98_ablation': "renamed to 'ablation' ('09_ablation')",
    'studies': "moved to 'ablation' ('09_ablation') and 'misc' ('99_misc/*')",
})

_FIG_PREFIX_RE = re.compile(r'^fig(?:\d+[a-z]*)?_')


def strip_fig_prefix(name: str) -> str:
    """Strip a leading ``fig_`` or ``fig<N>[a-z]*_`` prefix from a filename or stem."""
    return _FIG_PREFIX_RE.sub('', name)


def results_dir(name: str | Path | None = None, split: str = 'capped', create: bool = True) -> Path:
    """Return the canonical output directory under ``analysis/results`` for ``split`` ('capped' or 'mega').

    Split-aware MegaScale keys (``'megascale'``, ``'epistasis'``, ``'pair_heatmaps'``, ``'gain_fit'``,
    ``'megascale_screening'``, ``'property_gain'``, ``'dG_additive_vs_pred_dddG'``, ``'dddG_scatters'``)
    resolve under ``01_megascale/`` when ``split == 'capped'`` and under ``01_megascale/mega_splits/``
    when ``split == 'mega'``. Retired legacy folder names raise ``ValueError`` so stale callers cannot
    recreate zombie directories.
    """
    if split not in ('capped', 'mega'):
        raise ValueError(f"split must be 'capped' or 'mega', got {split!r}")
    key = str(name).strip().strip('/') if name is not None else ''
    if not key or key == '.':
        out = MEGA_SPLITS_DIR if split == 'mega' else RESULTS
    elif key in _MEGA_SPLIT_ROUTES:
        mega_base = MEGA_SPLITS_DIR if split == 'mega' else MEGASCALE_DIR
        rel = _MEGA_SPLIT_ROUTES[key]
        out = (mega_base / rel) if rel else mega_base
    elif key in _FIXED_RESULT_ROUTES:
        out = RESULTS / _FIXED_RESULT_ROUTES[key]
    else:
        top = key.split('/', 1)[0]
        if top in _RETIRED_RESULT_HINTS:
            raise ValueError(f"Legacy results directory {key!r} is {_RETIRED_RESULT_HINTS[top]}")
        if top not in CANONICAL_RESULT_TOPS:
            raise ValueError(
                f"Unknown results directory {key!r}; expected a key in RESULT_DIRS or subpath under {CANONICAL_RESULT_TOPS}"
            )
        if top == '01_megascale' and not key.startswith('01_megascale/mega_splits') and split == 'mega':
            sub = key.split('/', 1)[1] if '/' in key else ''
            out = (MEGA_SPLITS_DIR / sub) if sub else MEGA_SPLITS_DIR
        else:
            out = RESULTS / key
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
