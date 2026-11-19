"""Unified model selection (all seeds) and mutation-level bootstrap helpers for external benchmarks.

Used by ``02_thermostability.py``, ``04_dms_domainome.py``, and ``05_proteingym.py`` so every external
Spearman's rho figure shares:

- The unified model roster (no sigma in {0.75, 0.5, 0.25} sweep points; p0 at sigma = 1 for ESM-MSR).
- All three seeds of every retrained model (FINAL_p0, MutateEverything, ThermoMPNN, SPURS): the point estimate is the seed mean, the bootstrap resamples the seeds with the draws.
- Dataset-appropriate head selection (singles models on singles-only datasets; multi models on
  multi-mutant datasets, with ThermoMPNN additive on PTMUL / GB1 and ThermoMPNN-D on 2-mutant sets).
- Mutation-level bootstrapping (``kind='mutation'`` in ``msr_analysis.boot.bootstrap_rho``), cached
  separately in ``cache/boot_ext_mutation.pkl`` without altering ``01_core_rho.py`` / ``rho_bootstrap.csv``.
"""
import pickle

import numpy as np
from scipy.stats import spearmanr

from .boot import bootstrap_rho, ci
from .data import Frame, load_frame
from .registry import CACHE, DMS3, EXT12, MODELS

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

_CACHE_VERSION = 5                      # 3: the Domainome frame is held-out (msr_analysis.heldout); 4: all three seeds (was seed 1); 5: B = 1000 (was 100)


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


def seed_frame(ds, fr):
    """Return (Frame with the arrays of every seed, keyed by unified model names, {model_name: raw_key}).  A model with one run (zero-shot models, the public checkpoints) has one row."""
    preds = {}
    used = {}
    for m in EXT_ORDER:
        k = model_key_for_dataset(m, fr)
        if k is not None:
            preds[m] = fr.preds[k]
            used[m] = k
    return Frame(ds, fr.spine, preds), used


def load_mutation_boot(refresh=False):
    """Compute (or load cached) mutation-level bootstrap intervals (seed mean, seeds resampled) across EXT12, DMS3, and Domainome.

    Returns dict with:
      - ``point``: ``{(dataset, model_name): (rho, lo, hi)}``
      - ``reps``: ``{dataset: {model_name: ndarray(B,)}}``
      - ``keys``: ``{(dataset, model_name): raw_key}``
    """
    cache_path = CACHE / 'boot_ext_mutation.pkl'
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
        fr, used = seed_frame(ds, load_frame(ds))
        ext_frames[ds] = fr
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
    ext_keys = sorted({m for fr in ext_frames.values() for m in fr.preds})
    _run_group(ext_frames, ext_keys, kind='mutation', B=1000, seed=11, per_protein=False)

    for i, ds in enumerate(DMS3):
        fr, used = seed_frame(ds, load_frame(ds))
        for m, rk in used.items():
            keys_out[(ds, m)] = rk
        _run_group({ds: fr}, list(fr.preds), kind='mutation', B=1000, seed=12 + i, per_protein=False)

    dom_fr, dom_used = seed_frame('domainome', load_frame('domainome'))
    for m, rk in dom_used.items():
        keys_out[('domainome', m)] = rk
    _run_group({'domainome': dom_fr}, list(dom_fr.preds), kind='mutation', B=1000, seed=30, per_protein=True)

    payload = dict(version=_CACHE_VERSION, point=point_out, reps=reps_out, keys=keys_out)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload


def point_rho(fr, key):
    """Full-sample seed-mean rho of prediction ``key`` (the mean over the seeds of the pooled rho over the rows of ``use`` with a finite prediction: the point estimate of ``bootstrap_rho``), and the number of rows."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    rho, n = [], 0
    for pr in np.atleast_2d(fr.preds[key]):
        ok = (sp['use'] & np.isfinite(sp['y'])).to_numpy() & np.isfinite(pr)
        rho.append(float(spearmanr(pr[ok], y[ok])[0]))
        n = int(ok.sum())
    return float(np.mean(rho)), n


def fresh_frames(S, tol=5e-4):
    """{dataset: Frame} of the datasets of ``S`` (rows of ``02_thermostability/thermostability_rho.csv``: dataset, key, rho) that reproduce ``S``.

    The frame cache of ``analysis/cache`` can be older than the prediction files (SPURS: the trained-adapter predictions replaced the frozen ones on 2026-11-16), so when a seed-mean rho of a frame differs
    from the one in ``S`` by more than ``tol`` the frames are rebuilt once; if they still differ the error names the cells."""
    out = {}
    for attempt in (0, 1):
        bad = []
        for ds in sorted(S.dataset.unique()):
            if attempt:
                load_frame(ds, refresh=True)
            out[ds] = load_frame(ds)
            bad += [(ds, r.key, round(point_rho(out[ds], r.key)[0], 4), round(r.rho, 4)) for r in S[S.dataset == ds].itertuples() if abs(point_rho(out[ds], r.key)[0] - r.rho) > tol]
        if not bad:
            return out
        print(('frame cache disagrees with the figure data for %d cells, rebuilding: ' % len(bad)) if not attempt else 'STILL DISAGREES:', bad[:6], flush=True)
    raise SystemExit('the frames do not reproduce thermostability_rho.csv: rerun 02_thermostability.py --refresh first')
