"""Small shared helpers for the figure scripts."""
import numpy as np
import pandas as pd

from .registry import RESULTS


def has_multi(fr):
    """True if the rows that are scored include multi-mutants."""
    sp = fr.spine
    return bool(((sp['n_mut'] > 1) & sp['use']).any())


def primary_key(fr, family='msr'):
    """The ESM-MSR readout a dataset calls for: the WT head when only single mutants are scored, the combined
    prediction when multi-mutants are (the WT head is additive and cannot represent an interaction)."""
    return f'{family}:comb' if has_multi(fr) else f'{family}:wt'


def out_dir(name):
    p = RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rho(kind='protein'):
    """results/00_core/rho_bootstrap.csv as {(dataset, model): (rho, lo, hi)} for one resampling unit."""
    d = pd.read_csv(RESULTS / '00_core' / 'rho_bootstrap.csv')
    d = d[d['kind'] == kind]
    return {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in d.itertuples()}


def load_rho_df():
    return pd.read_csv(RESULTS / '00_core' / 'rho_bootstrap.csv')


def best_ci(dataset, model, tables):
    """CI for a cell: protein-level where proteins can be resampled, site-level for the single-protein DMS."""
    for kind in ('protein', 'site', 'mutation'):
        v = tables.get(kind, {}).get((dataset, model))
        if v is not None:
            return v, kind
    return None, None


def primary_ci(df):
    """One (rho, lo, hi) per (dataset, model): the protein-level interval where defined, else site-level, else mutation."""
    rank = {'protein': 0, 'site': 1, 'mutation': 2}
    d = df.assign(r=df['kind'].map(rank)).sort_values('r').drop_duplicates(['dataset', 'model'])
    return {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in d.itertuples()}


def epi_rows(fr):
    """Doubles that enter the epistasis tables: measured, with a measured epistasis (dddG), and BOTH constituent singles
    measured (cens == 0).  Reproduces the 8,494 validation / 12,455 test doubles of BENCHMARKS section 2."""
    sp = fr.spine
    m = ((sp['n_mut'] == 2) & sp['use'] & np.isfinite(sp['dddG_ML'])).to_numpy()
    if 'cens' in sp:
        single = sp[sp['n_mut'] == 1]
        cens = dict(zip(single['id'], single['cens']))
        ids = sp['id'].to_numpy()
        for i in np.flatnonzero(m):
            pre, mut = ids[i].rsplit('_', 1)
            if not all(cens.get(f'{pre}_{x}', np.nan) == 0 for x in mut.split(':')):
                m[i] = False
    return m


def subframe(fr, mask, y=None, keys=None, rename=None, name=None):
    """A Frame restricted to ``mask`` rows whose label is replaced by ``y`` (e.g. dddG) -- lets the bootstrap engine score
    epistasis exactly as it scores ddG."""
    import copy
    from .data import Frame
    sp = fr.spine.copy()
    if y is not None:
        sp['y'] = y
    sp['use'] = mask
    out = Frame(name or fr.name, sp, {})
    for k in (keys or fr.preds):
        if k in fr.preds:
            out.preds[(rename or {}).get(k, k)] = fr.preds[k]
    return out


# ---- leak flags -----------------------------------------------------------------------------------------------
LEAK_NAME = {'esmtherm': 'ESMtherm', 'ddgemb': 'DDGemb', 'ddmut': 'DDMut', 'ifum_wt': 'IFUM (WT backbone)', 'ifum_esmfold': 'IFUM (native, ESMFold)',
             'geoddg': 'GeoDDG-3D', 'esm3dg': 'ESM3dG'}
LEAK_DS = {'tsuboyama_capped_val': 'tsu_capped_val', 'tsuboyama_capped_test': 'tsu_capped_test'}


def leak_table():
    """{(model key, dataset): fraction of rows with an identical / constituent mutation, same protein, strong homolog, documented
    training on the source set (classes M m P H D U #) or the cell is a training-set score}; the retrained models, ESM-MSR and
    the zero-shot models are held out by construction (empty).  From the leak audit of BENCHMARKS section 6."""
    from .registry import REFERENCE
    d = pd.read_csv(REFERENCE / 'leak_audit_matrix.csv')
    d['frac'] = (d[['exact', 'constituent', 'P', 'H', 'D', 'U']].fillna(0).sum(axis=1)) / d['n'].replace(0, np.nan)
    d['flag'] = (d['frac'] >= 0.005) | d['code'].astype(str).str.contains('#')
    out = {}
    inv = {v: k for k, v in LEAK_NAME.items()}
    for r in d.itertuples():
        if r.model in inv:
            ds = {v: k for k, v in LEAK_DS.items()}.get(r.dataset, r.dataset)
            out[(inv[r.model], ds)] = bool(r.flag)
    return out
