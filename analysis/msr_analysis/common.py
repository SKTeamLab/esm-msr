"""Shared frame-selection, output-directory, epistasis-row, and leak-flag helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Frame, singles_of_doubles
from .paths import REFERENCE, results_dir
from .registry import TSU_CAPPED, TSU_MEGA

__all__ = [
    'epi_rows',
    'has_multi',
    'leak_table',
    'load_rho_df',
    'out_dir',
    'primary_ci',
    'primary_key',
    'singles_of_doubles',
    'subframe',
]


def has_multi(fr: Frame) -> bool:
    """Return True if the scored rows of ``fr`` include multi-mutants."""
    sp = fr.spine
    return bool(((sp['n_mut'] > 1) & sp['use']).any())


def primary_key(fr: Frame, family: str = 'msrp0') -> str:
    """Return the deployed ESM-MSR readout key for ``fr`` (``<family>:comb`` if multi-mutants, else ``<family>:wt``)."""
    return f'{family}:comb' if has_multi(fr) else f'{family}:wt'


def out_dir(name: str, split: str = 'capped'):
    """Return ``results[/03_tsuboyama/mega_splits]/<name>`` (created if needed)."""
    return results_dir(name, split=split, create=True)


def load_rho_df(path=None) -> pd.DataFrame:
    """Load ``results/00_core/rho_bootstrap.csv`` with ``esm3dg*`` stripped from Megascale/Tsuboyama sets."""
    p = path or (results_dir('00_core', create=False) / 'rho_bootstrap.csv')
    d = pd.read_csv(p)
    return d[~(d['model'].str.startswith('esm3dg') & d['dataset'].isin(TSU_CAPPED + TSU_MEGA))].reset_index(drop=True)


def primary_ci(df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Return ``{(dataset, model): (rho, lo, hi)}`` preferring protein-level CIs, then site, then mutation."""
    rank = {'protein': 0, 'site': 1, 'mutation': 2}
    d = df.assign(r=df['kind'].map(rank)).sort_values('r').drop_duplicates(['dataset', 'model'])
    return {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in d.itertuples()}


def epi_rows(fr: Frame) -> np.ndarray:
    """Boolean mask of measured doubles with finite ``dddG_ML`` and both constituent singles uncensored (``cens == 0``)."""
    sp = fr.spine
    m = ((sp['n_mut'] == 2) & sp['use'] & np.isfinite(sp['dddG_ML'])).to_numpy()
    if 'cens' in sp:
        single = sp[sp['n_mut'] == 1]
        cens: dict[str, int] = {}
        for sid, cv in zip(single['id'], single['cens']):
            if sid not in cens:
                cens[sid] = cv
        ids = sp['id'].to_numpy()
        for i in np.flatnonzero(m):
            pre, mut = ids[i].rsplit('_', 1)
            if not all(cens.get(f'{pre}_{x}', np.nan) == 0 for x in mut.split(':')):
                m[i] = False
    return m


def subframe(fr: Frame, mask, y=None, keys=None, rename=None, name: str | None = None) -> Frame:
    """Return a :class:`Frame` restricted to ``mask`` rows with optional label replacement ``y``."""
    sp = fr.spine.copy()
    if y is not None:
        sp['y'] = y
    sp['use'] = mask
    out = Frame(name or fr.name, sp, {})
    for k in (keys or fr.preds):
        if k in fr.preds:
            out.preds[(rename or {}).get(k, k)] = fr.preds[k]
    return out


LEAK_NAME = {
    'esmtherm': 'ESMtherm', 'ddgemb': 'DDGemb', 'ddmut': 'DDMut',
    'ifum_wt': 'IFUM (WT backbone)', 'ifum_esmfold': 'IFUM (native, ESMFold)',
    'geoddg': 'GeoDDG-3D', 'esm3dg': 'ESM3dG',
}
_LEAK_DS = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}


def leak_table() -> dict[tuple[str, str], bool]:
    """Return ``{(model_key, dataset): bool}`` flagging documented training/homolog overlap (>= 0.5% or ``#``)."""
    d = pd.read_csv(REFERENCE / 'leak_audit_matrix.csv')
    d['frac'] = d[['exact', 'constituent', 'P', 'H', 'D']].fillna(0).sum(axis=1) / d['n'].replace(0, np.nan)
    d['flag'] = (d['frac'] >= 0.005) | d['code'].astype(str).str.contains('#')
    out = {}
    inv = {v: k for k, v in LEAK_NAME.items()}
    inv_ds = {v: k for k, v in _LEAK_DS.items()}
    for r in d.itertuples():
        if r.model in inv:
            ds = inv_ds.get(r.dataset, r.dataset)
            out[(inv[r.model], ds)] = bool(r.flag)
    return out
