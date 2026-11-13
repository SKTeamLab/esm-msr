"""Unified seed-1 model selection and mutation-level bootstrap helpers for external benchmarks.

Used by ``02_thermostability.py``, ``04_dms_domainome.py``, and ``05_proteingym.py`` so every external
Spearman's rho figure shares:

- The unified model roster (no sigma in {0.75, 0.5, 0.25} sweep points; p0 at sigma = 1 for ESM-MSR).
- Seed 1 only (index 0) for every multi-seed model.
- Dataset-appropriate head selection (singles models on singles-only datasets; multi models on
  multi-mutant datasets, with ThermoMPNN additive on PTMUL / GB1 and ThermoMPNN-D on 2-mutant sets).
- Mutation-level bootstrapping (``kind='mutation'`` in ``msr_analysis.boot.bootstrap_rho``), cached
  separately in ``cache/boot_s1_mutation.pkl`` without altering ``01_core_rho.py`` / ``rho_bootstrap.csv``.
"""
import pickle

from .boot import bootstrap_rho, ci
from .data import Frame, load_frame
from .registry import CACHE, DMS7, EXT12, MODELS

EXT_ORDER = [
    'ESM-MSR',
    'MutateEverything',
    'ThermoMPNN',
    'SPURS',
    'ESM3-small-open',
    'ProteinMPNN',
    'ProSST',
    'GeoDDG-3D',
    'Rosetta',
]

EXT_MODELS = {
    'ESM-MSR':          dict(label='ESM-MSR',                  color='#000000', marker='D', fill=True, group='msr'),
    'MutateEverything': dict(label='MutateEverything',         color='#1b8a4b', marker='h', fill=True, group='retrain'),
    'ThermoMPNN':       dict(label='ThermoMPNN',               color='#1f5f8b', marker='p', fill=True, group='retrain'),
    'SPURS':            dict(label='SPURS',                    color='#d35400', marker='s', fill=True, group='retrain'),
    'ESM3-small-open':  dict(label='ESM3-small-open (σ = 0)',  color='#7f8c8d', marker='^', fill=True, group='zeroshot'),
    'ProteinMPNN':      dict(label='ProteinMPNN',              color='#34495e', marker='d', fill=True, group='zeroshot'),
    'ProSST':           dict(label='ProSST',                   color='#8e44ad', marker='X', fill=True, group='zeroshot'),
    'GeoDDG-3D':        dict(label='GeoDDG-3D',                color='#117a65', marker='v', fill=True, group='public'),
    'Rosetta':          dict(label='Rosetta',                  color='#8c564b', marker='P', fill=True, group='public'),
}

_CACHE_VERSION = 2


def register_ext_styles():
    """Register unified display styles into ``msr_analysis.registry.MODELS``."""
    for name, spec in EXT_MODELS.items():
        MODELS[name] = dict(spec)


def model_key_for_dataset(model_name, fr):
    """Return the prediction key in ``fr.preds`` appropriate for ``model_name`` on dataset ``fr``."""
    sp = fr.spine[fr.spine['use']]
    is_multi = bool((sp['n_mut'] > 1).any())
    has_higher = bool((sp['n_mut'] > 2).any())
    mapping = {
        'ESM-MSR':         'msrp0:comb' if is_multi else 'msrp0:wt',
        'MutateEverything': 'me2' if is_multi else 'me1',
        'ThermoMPNN':      ('tmpnn' if has_higher else 'tmpnnd') if is_multi else 'tmpnn',
        'SPURS':           'spursm' if is_multi else 'spurs',
        'ESM3-small-open': 'msrrel@0:comb' if is_multi else 'msrrel@0:wt',
        'ProteinMPNN':     'pmpnn',
        'ProSST':          'prosst',
        'GeoDDG-3D':       'geoddg',
        'Rosetta':         'rosetta',
    }
    key = mapping[model_name]
    return key if key in fr.preds else None


def _seed1_frame(ds, fr):
    """Return (Frame with seed-1 arrays keyed by unified model names, {model_name: raw_key})."""
    preds = {}
    used = {}
    for m in EXT_ORDER:
        k = model_key_for_dataset(m, fr)
        if k is not None:
            preds[m] = fr.preds[k][:1]
            used[m] = k
    return Frame(ds, fr.spine, preds), used


def load_s1_mutation_boot(refresh=False):
    """Compute (or load cached) seed-1 mutation-level bootstrap intervals across EXT12, DMS7, and Domainome.

    Returns dict with:
      - ``point``: ``{(dataset, model_name): (rho, lo, hi)}``
      - ``reps``: ``{dataset: {model_name: ndarray(B,)}}``
      - ``keys``: ``{(dataset, model_name): raw_key}``
    """
    cache_path = CACHE / 'boot_s1_mutation.pkl'
    if cache_path.exists() and not refresh:
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('version') == _CACHE_VERSION:
            return cached

    point_out = {}
    reps_out = {}
    keys_out = {}

    def _run_group(frames, keys, kind, B, seed, per_protein):
        pt, rp = bootstrap_rho(frames, keys, kind=kind, B=B, seed=seed, per_protein=per_protein)
        for ds, kmap in pt.items():
            reps_out[ds] = {}
            for m, (r, _ps, _n, _ng) in kmap.items():
                lo, hi = ci(rp[ds][m])
                point_out[(ds, m)] = (r, lo, hi)
                reps_out[ds][m] = rp[ds][m]

    ext_frames = {}
    for ds in EXT12:
        fr, used = _seed1_frame(ds, load_frame(ds))
        ext_frames[ds] = fr
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
    ext_keys = sorted({m for fr in ext_frames.values() for m in fr.preds})
    _run_group(ext_frames, ext_keys, kind='mutation', B=1000, seed=11, per_protein=False)

    for i, ds in enumerate(DMS7):
        fr, used = _seed1_frame(ds, load_frame(ds))
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
        _run_group({ds: fr}, list(fr.preds), kind='mutation', B=500, seed=12 + i, per_protein=False)

    dom_fr, dom_used = _seed1_frame('domainome', load_frame('domainome'))
    for m, rk in dom_used.items():
        keys_out[('domainome', m)] = rk
    _run_group({'domainome': dom_fr}, list(dom_fr.preds), kind='mutation', B=250, seed=30, per_protein=True)

    payload = dict(version=_CACHE_VERSION, point=point_out, reps=reps_out, keys=keys_out)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload
