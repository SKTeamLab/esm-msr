"""Unified model selection (all seeds) and mutation-level bootstrap helpers for external benchmarks."""
from __future__ import annotations

import pickle
from types import MappingProxyType

import numpy as np
from scipy.stats import spearmanr

from . import models as R
from .boot import bootstrap_rho, ci
from .data import Frame, load_frame
from .paths import CACHE, atomic_pickle_dump
from .registry import DMS3, EXT12, MODELS

EXT_ORDER = list(R.UNIFIED)
EXT_MODELS = MappingProxyType({m.name: MODELS[m.name] for m in R.ROSTER})

_CACHE_VERSION = 6


def register_ext_styles() -> None:
    """No-op kept for backwards compatibility: ``registry.MODELS`` already includes all canonical roster names."""
    return None


def model_key_for_dataset(model_name: str, fr: Frame) -> str | None:
    """Return the prediction key in ``fr.preds`` appropriate for ``model_name`` on dataset ``fr``."""
    sp = fr.spine[fr.spine['use']]
    is_multi = bool((sp['n_mut'] > 1).any())
    has_higher = bool((sp['n_mut'] > 2).any())
    m = R.BY_NAME[model_name]
    if model_name == 'ThermoMPNN':
        key = ('tmpnn' if has_higher else 'tmpnnd') if is_multi else 'tmpnn'
    else:
        key = m.multi if is_multi else m.single
    return key if key in fr.preds else None


def seed_frame(ds: str, fr: Frame, models=None) -> tuple[Frame, dict[str, str]]:
    """Return ``(Frame keyed by roster model names, {model_name: raw_key})``."""
    preds = {}
    used = {}
    for m in (EXT_ORDER if models is None else models):
        k = model_key_for_dataset(m, fr)
        if k is not None:
            preds[m] = fr.preds[k]
            used[m] = k
    return Frame(ds, fr.spine, preds), used


def load_mutation_boot(refresh: bool = False) -> dict:
    """Compute (or load cached) mutation-level bootstrap intervals across ``EXT12``, ``DMS3``, and ``domainome``."""
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
    atomic_pickle_dump(payload, cache_path)
    return payload


def point_rho(fr: Frame, key: str) -> tuple[float, int]:
    """Return ``(seed_mean_pooled_rho, n_rows)`` for ``key`` on ``fr``."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    rho, n = [], 0
    for pr in np.atleast_2d(fr.preds[key]):
        ok = (sp['use'] & np.isfinite(sp['y'])).to_numpy() & np.isfinite(pr)
        rho.append(float(spearmanr(pr[ok], y[ok])[0]))
        n = int(ok.sum())
    return float(np.mean(rho)), n


def fresh_frames(S: pd.DataFrame, tol: float = 5e-4) -> dict[str, Frame]:
    """Verify ``{dataset: Frame}`` reproduces ``S`` (columns ``dataset, key, rho``), refreshing once if stale."""
    out = {}
    for attempt in (0, 1):
        bad = []
        for ds in sorted(S.dataset.unique()):
            if attempt:
                load_frame(ds, refresh=True)
            out[ds] = load_frame(ds)
            bad += [
                (ds, r.key, round(point_rho(out[ds], r.key)[0], 4), round(r.rho, 4))
                for r in S[S.dataset == ds].itertuples()
                if abs(point_rho(out[ds], r.key)[0] - r.rho) > tol
            ]
        if not bad:
            return out
        print(('frame cache disagrees with figure data for %d cells, rebuilding: ' % len(bad)) if not attempt else 'STILL DISAGREES:', bad[:6], flush=True)
    raise SystemExit('frames do not reproduce thermostability_rho.csv: rerun 02_thermostability.py --refresh first')
