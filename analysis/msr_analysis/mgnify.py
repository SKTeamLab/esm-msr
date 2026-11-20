"""MGnify & Rocklin de novo stability sets scored with ESM-MSR (``FINAL_p0``): latent & Tobit-linked frames."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from esm_msr.val_stats import tobit_link

from .data import Frame
from .paths import (
    CACHE,
    FINAL_MGNIFY as RUNS,
    FINAL_MGNIFY_BY_SIGMA as RUNS_BY_SIGMA,
    MGNIFY_TABLE as TABLE,
    MSR_ROOT as ROOT,
    REFERENCE,
    atomic_pickle_dump,
    source_signature,
)
from .tobit import TOBIT

CACHE_SCHEMA_VERSION = 2
CLASSES = REFERENCE / 'mgnify_homology_classes.csv.gz'
HEADS = dict(wt='wt_lora_pred', mt='mt_lora_pred', comb='combined_pred')
PRED_COLS = [
    'uid', 'code_wt', 'mut_type', 'n_mut', 'mgnify_split', 'has_all_singles',
    'wt_lora_pred', 'mt_lora_pred', 'combined_pred', 'wt_lora_pred_additive', 'combined_pred_additive',
]
LABEL_COLS = ['dG_wt', 'dG_ML', 'ddG_ML', 'dddG_ML', 'cens']
CENS_LEGACY = {'default': 0, 'mut_below_neg1': -1, 'wt_out_of_range': -2, 'mut_above_pos5': 1}
STRATA = {
    'all': None,
    'lowid': ['none', 'low', 'h'],
    'clean': ['none', 'low'],
    'h': ['h'],
    'strong': ['H', 'sameprot'],
}


def legacy_cens(series: pd.Series) -> pd.Series:
    """Map v2 string censoring codes to legacy integers ``{0, -1, -2, 1}``."""
    return series.map(lambda v: CENS_LEGACY.get(v, v)).astype(int) if series.dtype == object else series


def finished_seeds(family: str = 'FINAL_p0', run: str = 'all', sigma: float = 1.0) -> list[int]:
    """Return seeds whose MGnify run wrote both ``summary.json`` and ``predictions.csv.gz``."""
    root = RUNS_BY_SIGMA[float(sigma)]
    return [
        s for s in (1, 2, 3)
        if (root / family / f's{s}' / run / 'summary.json').exists()
        and (root / family / f's{s}' / run / 'predictions.csv.gz').exists()
    ]


def _labels() -> pd.DataFrame:
    t = pd.read_csv(TABLE, usecols=['uid'] + LABEL_COLS, low_memory=False).set_index('uid')
    t['cens'] = legacy_cens(t['cens'])
    return t


def load_mgnify(
    family: str = 'FINAL_p0',
    run: str = 'all',
    seeds=None,
    refresh: bool = False,
    heldout: bool = True,
    sigma: float = 1.0,
    check_sources: bool = False,
) -> Frame:
    """Load MGnify (``run='all'``) or Rocklin de novo (``run='rdenovo'``) :class:`Frame`.

    When ``heldout=True``, scaffolds homologous to training libraries under the 4-part rule
    (``msr_analysis.heldout``) have ``spine['use'] = False``.
    """
    fr = _load_mgnify(family, run, seeds, refresh, sigma, check_sources=check_sources)
    if heldout:
        from .heldout import excluded
        bad = excluded('rocklin_denovo' if run == 'rdenovo' else 'mgnify')
        sp = fr.spine.copy()
        sp['use'] = sp['use'].to_numpy() & ~sp['code_wt'].astype(str).isin(bad).to_numpy()
        return Frame(fr.name, sp, fr.preds, notes=dict(fr.notes, heldout=True))
    return fr


def _load_mgnify(
    family: str = 'FINAL_p0',
    run: str = 'all',
    seeds=None,
    refresh: bool = False,
    sigma: float = 1.0,
    check_sources: bool = False,
) -> Frame:
    root = RUNS_BY_SIGMA[float(sigma)]
    seeds = list(seeds or finished_seeds(family, run, sigma))
    if not seeds:
        raise FileNotFoundError(f'No finished MGnify run under {root}/{family}')
    sig_sfx = '' if float(sigma) == 1.0 else f'_sigma{sigma}'
    pkl = CACHE / f'frame_mgnify_{family}_{run}_s{"".join(map(str, seeds))}{sig_sfx}.pkl'
    pred_paths = [root / family / f's{s}' / run / 'predictions.csv.gz' for s in seeds]
    sources: list[Path] = [*pred_paths, TABLE]
    if run != 'rdenovo' and CLASSES.exists():
        sources.append(CLASSES)

    if pkl.exists() and not refresh:
        with open(pkl, 'rb') as fh:
            cached = pickle.load(fh)
        if not check_sources:
            return cached
        stored_sig = cached.notes.get('source_sig')
        if stored_sig is not None and source_signature(p for p, _, _ in stored_sig) == stored_sig:
            return cached

    base, P = None, {h: [] for h in HEADS}
    for p in pred_paths:
        d = pd.read_csv(p, usecols=lambda c: c in PRED_COLS, low_memory=False)
        if base is None:
            base = d
        else:
            d = d.set_index('uid').loc[base['uid']].reset_index()
        for h, c in HEADS.items():
            P[h].append(d[c].to_numpy(float))
    lab = _labels()
    sp = pd.DataFrame({'id': base['uid'].astype(str), 'row_id': np.arange(len(base))})
    for c in LABEL_COLS:
        sp[c] = base['uid'].map(lab[c]).to_numpy()
    sp['y'] = sp['ddG_ML'].astype(float)
    sp['mut'] = base['mut_type'].astype(str).to_numpy()
    sp['n_mut'] = base['n_mut'].to_numpy()
    sp['site1'] = sp['mut'].str.extract(r'(-?\d+)')[0].astype(float).fillna(-1).astype(int)
    sp['group'] = sp['cluster'] = sp['code'] = sp['code_wt'] = base['code_wt'].to_numpy()
    sp['split'] = base['mgnify_split'].to_numpy()
    sp['has_all_singles'] = base['has_all_singles'].to_numpy()
    sp['use'] = (np.isfinite(sp['y']) & (sp['cens'] == 0)).to_numpy()
    if run == 'rdenovo':
        sp['cls'] = 'n/a'
    else:
        cls_path = CLASSES if CLASSES.exists() else ROOT / 'leak_audit' / 'out' / 'mgnify_homology_classes.csv'
        cls = pd.read_csv(cls_path, usecols=['code_wt', 'cls']).set_index('code_wt')['cls']
        sp['cls'] = sp['code_wt'].map(cls).fillna('none')
    h = tobit_link(**TOBIT)
    dg = sp['dG_wt'].to_numpy(float)
    preds = {}
    for k, a in P.items():
        arr = np.vstack(a)
        preds[k] = arr
        preds[k + '~'] = h(dg[None, :] + arr) - dg[None, :]
    notes = dict(
        seeds=seeds,
        family=family,
        run=run,
        sigma=float(sigma),
        tobit=TOBIT,
        cache_schema=CACHE_SCHEMA_VERSION,
        source_sig=source_signature(sources),
    )
    fr = Frame(f'mgnify_{family}', sp, preds, notes=notes)
    atomic_pickle_dump(fr, pkl)
    return fr
