"""Load every model's predictions on one dataset, aligned to that dataset's label table (the "spine").

``load_frame(dataset)`` returns a :class:`Frame`: the spine (labels, protein clusters, mutation counts, sequence
length, the row filter the benchmark tables apply) and ``preds``, a dict ``model key -> (n_seeds, n_rows) array``
that is NaN wherever a model has no prediction.  All predictions are stabilizing-positive, like the label ``y``
(= ESM-MSR's ``ddG_ML``).  Frames are cached as pickles under ``analysis/cache``.
"""
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .registry import (BASE, CACHE, MSR, PUBLIC, RETRAINED, TASKS, PER_PROTEIN, DMS7, EXT12, TSU)

TSU_SETS = set(TSU)

SEEDS = (1, 2, 3)
SPINE_COLS = ['id', 'row_id', 'code', 'group', 'truth', 'mut_type', 'ref_seq', 'cens', 'mut_structure', 'code_wt',
              'ddG_ML', 'dddG_ML', 'ddG_additive_ML', 'ddG_dir', 'wt_code', 'dG_ML']


@dataclass
class Frame:
    name: str
    spine: pd.DataFrame
    preds: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.spine)

    def keys(self):
        return list(self.preds)

    def pred(self, key):
        """(n_seeds, n) array for a model key."""
        return self.preds[key]

    def has(self, key):
        return key in self.preds and np.isfinite(self.preds[key]).any()


# ----------------------------------------------------------------------------------------------
# spine
# ----------------------------------------------------------------------------------------------
def _first_site(m):
    mm = re.search(r'(-?\d+)', str(m).split(':')[0])
    return int(mm.group(1)) if mm else -1


def build_spine(ds):
    t = pd.read_csv(TASKS / f'{ds}.csv.gz', usecols=lambda c: c in SPINE_COLS, low_memory=False)
    s = pd.DataFrame({'id': t['id'].astype(str), 'row_id': t['row_id'].astype(int)})
    s['y'] = t['truth'].astype(float)
    s['code'] = t['code'].astype(str) if 'code' in t else s['id'].str[:4]
    s['group'] = t['group'].astype(str)
    s['mut'] = t['mut_type'].astype(str) if 'mut_type' in t else ''
    s['n_mut'] = s['mut'].str.count(':') + 1
    s['site1'] = s['mut'].map(_first_site)
    s['seq_len'] = t['ref_seq'].astype(str).str.len() if 'ref_seq' in t else np.nan
    if 'dG_ML' in t and 'ddG_ML' in t:
        s['dG_wt'] = (t['dG_ML'] - t['ddG_ML']).astype(float)     # the library constant (BENCHMARKS section 5.2)
    for c in ('cens', 'mut_structure', 'code_wt', 'dddG_ML', 'ddG_additive_ML'):
        if c in t:
            s[c] = t[c]
    # the benchmark tables score measured rows only; the Tsuboyama test table also drops destabilized backbones
    use = np.isfinite(s['y'])
    if 'cens' in s:
        use &= (s['cens'] == 0)             # measured rows only (BENCHMARKS section 5.3)
    s['use'] = use.values
    if 'mut_structure' in s:               # the epistasis tables score wild-type backbones only (section 5.5)
        s['wt_backbone'] = (s['mut_structure'] == '-').values
    if ds in TSU_SETS:                      # the per-protein mean is over libraries = wild-type codes (23 val, 33 test)
        s['group'] = s['code_wt'].astype(str)
    # the cluster a protein-level bootstrap resamples: wild-type protein, not variant structure
    if ds in PER_PROTEIN:
        s['cluster'] = s['group']
    else:
        s['cluster'] = _wt_cluster(ds, s)
    return s


def _wt_cluster(ds, s):
    """External sets name a protein by PDB entry; Ssym uses one entry per variant, so cluster on the
    wild-type code the ESM-MSR file carries (`wt_code`) when it is there."""
    p = MSR / 'baseline_w2a4_m16a16' / 'seed1' / 'sigma1' / f'{ds}.csv'
    if p.exists() and ds in EXT12:
        h = pd.read_csv(p, usecols=['id', 'wt_code'] if 'wt_code' in pd.read_csv(p, nrows=1).columns else ['id'])
        if 'wt_code' in h:
            m = dict(zip(h['id'], h['wt_code'].astype(str)))
            c = s['id'].map(m)
            return c.fillna(s['code']).astype(str)
    return s['code'].astype(str)


# ----------------------------------------------------------------------------------------------
# alignment helpers
# ----------------------------------------------------------------------------------------------
def _align_id(spine_ids, ids, vals):
    """vals indexed by file ids, laid out on the spine's ids (first occurrence wins; NaN when absent)."""
    ids = np.asarray(ids, dtype=object)
    vals = np.asarray(vals, dtype=float)
    if len(ids) == len(spine_ids) and (ids == spine_ids).all():
        return vals.copy()
    ser = pd.Series(vals, index=ids)
    ser = ser[~ser.index.duplicated(keep='first')]
    return ser.reindex(spine_ids).to_numpy(dtype=float)


def _align_rowid(spine_rowid, rowid, vals):
    ser = pd.Series(np.asarray(vals, dtype=float), index=np.asarray(rowid))
    ser = ser[~ser.index.duplicated(keep='first')]
    return ser.reindex(spine_rowid).to_numpy(dtype=float)


def sum_singles(spine, pred):
    """Additive prediction of every multi-mutant: the sum of its constituent singles' predictions taken from
    the same file (NaN when a constituent has no single row).  Singles keep their own prediction."""
    ids = spine['id'].to_numpy(dtype=object)
    nm = spine['n_mut'].to_numpy()
    single = {i: p for i, p in zip(ids[nm == 1], pred[nm == 1])}
    out = pred.copy()
    for k in np.flatnonzero(nm > 1):
        prefix, mut = ids[k].rsplit('_', 1) if '_' in ids[k] else ('', ids[k])
        tot = 0.0
        for m in mut.split(':'):
            tot += single.get(f'{prefix}_{m}', np.nan)
        out[k] = tot
    return out


# ----------------------------------------------------------------------------------------------
# families
# ----------------------------------------------------------------------------------------------
def _msr_dir(spec, seed, sigma):
    if spec == 'msr':
        return MSR / 'baseline_w2a4_m16a16' / f'seed{seed}' / f'sigma{sigma}'
    if spec == 'msrdev':
        return MSR / 'devel_dc8_floorlink_w2a4_m16a16' / f'seed{seed}' / f'sigma{sigma}'
    if spec == 'msrrel':
        return MSR / 'released_capped_w2a4_m16a16' / f'sigma{sigma}'
    raise KeyError(spec)


def _load_msr(frame, ds, out):
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    heads = dict(wt='wt_lora_pred', comb='combined_pred', mt='mt_lora_pred', epi='epi_pred', cdddg='combined_dddg_pred')
    for spec, sigmas, seeds in (('msr', ['1'], SEEDS), ('msrdev', ['1', '0.75', '0.5', '0.25'], SEEDS),
                                ('msrrel', ['1', '0.75', '0.5', '0.25', '0'], (None,))):
        for sg in sigmas:
            arrs = {h: [] for h in heads}
            ok = True
            for sd in seeds:
                p = _msr_dir(spec, sd, sg) / f'{ds}.csv'
                if not p.exists():
                    ok = False
                    break
                hdr = pd.read_csv(p, nrows=0).columns
                use = ['id'] + [c for c in heads.values() if c in hdr]
                d = pd.read_csv(p, usecols=use, low_memory=False)
                for h, c in heads.items():
                    if c in d:
                        arrs[h].append(_align_id(ids, d['id'].astype(str).to_numpy(), d[c].to_numpy()))
            if not ok:
                continue
            tag = spec if sg == '1' else f'{spec}@{sg}'
            for h, lst in arrs.items():
                if lst and (h in ('wt', 'comb') or sg == '1'):
                    out[f'{tag}:{h}'] = np.vstack(lst)


def _read_seeded(path_fmt, seeds=SEEDS):
    for sd in seeds:
        p = path_fmt.format(seed=sd)
        if not os.path.exists(p):
            return
        yield sd, pd.read_csv(p, low_memory=False)


PARENT = {'s461': 's669', 'k2369': 'k3822', 'ptmuld': 'ptmul'}      # S461 within S669, K2369 within K3822, PTMUL-D within PTMUL


def _me_stage_arrays(ds, spine, additive=False):
    """{stage: (n_seeds, n) stabilizing-positive ME prediction on ``spine``'s rows}; with ``additive`` also {stage + '+': the sum of the model's own singles}.
    The additive readout is built from ME's own file (complete GB1, 149,360 rows), not from the harness label table, which lacks three GB1 singles."""
    ids = spine['id'].to_numpy(dtype=object)
    idset = set(ids)
    res = {}
    for stage in ('stage1', 'stage2'):
        arrs, adds = [], []
        for sd, d in _read_seeded(str(RETRAINED / 'mutate_everything' / 'predictions' / stage / 'seed{seed}' / f'{ds}.csv')):
            k1 = d['pdb_id'].astype(str) + '_' + d['mut_info'].astype(str)
            k2 = d['pdb_id'].astype(str).str[:-1] + '_' + d['mut_info'].astype(str)
            key = (k1 if k1.isin(idset).mean() >= k2.isin(idset).mean() else k2).to_numpy(dtype=object)
            v = -d['ddg_pred'].to_numpy()
            arrs.append(_align_id(ids, key, v))
            if additive:
                own = pd.DataFrame({'id': key, 'n_mut': pd.Series(key).str.rsplit('_', n=1).str[-1].str.count(':').to_numpy() + 1})
                adds.append(_align_id(ids, key, sum_singles(own, v)))
        if arrs:
            res[stage] = np.vstack(arrs)
        if adds:
            res[stage + '+'] = np.vstack(adds)
    return res


def _load_me(frame, ds, out):
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    if ds in PARENT:
        # S461, K2369 and PTMUL-D are scored from their parent set's predictions, matched on (PDB entry + mutation), against the subset's own labels (BENCHMARKS
        # sections 1.1 and 5.7).  ME's own S461 file is a different input (other constructs and MSAs) and must not be used; K2369 and PTMUL-D have no ME file.
        psp = build_spine_ids(PARENT[ds])
        par = _me_stage_arrays(PARENT[ds], psp, additive=ds in DMS7)
        res = {st: np.vstack([_align_id(ids, psp['id'].to_numpy(dtype=object), r) for r in v]) for st, v in par.items()}
    else:
        res = _me_stage_arrays(ds, sp, additive=ds in DMS7)
    for stage, key in (('stage1', 'me1'), ('stage2', 'me2')):
        if stage in res:
            out[f'{key}:'] = res[stage]
    if 'stage1' in res and 'stage2' in res:
        # predicted epistasis of stage 2: direct double minus the same-seed stage-1 additive expectation
        out['me2:epi'] = res['stage2'] - np.vstack([sum_singles(sp, r) for r in res['stage1']])
    for stage, key in (('stage1+', 'me1+'), ('stage2+', 'me2+')):
        if stage in res:
            out[f'{key}:'] = res[stage]


def build_spine_ids(ds):
    t = pd.read_csv(TASKS / f'{ds}.csv.gz', usecols=['id'], low_memory=False)
    return pd.DataFrame({'id': t['id'].astype(str)})


def _keycol(d):
    if 'id' in d:
        return d['id'].astype(str).to_numpy()
    if 'uid' in d:
        return d['uid'].astype(str).to_numpy()
    return (d['code'].astype(str) + '_' + d['mut_type'].astype(str)).to_numpy()


def _load_tmpnn(frame, ds, out):
    ids = frame.spine['id'].to_numpy(dtype=object)
    a = []
    for sd, d in _read_seeded(str(RETRAINED / 'thermompnn' / 'predictions' / 'single_additive' / 'seed{seed}' / f'{ds}.csv')):
        col = 'pred_single_additive'
        a.append(_align_id(ids, _keycol(d), d[col].to_numpy()))
    if a:
        out['tmpnn:'] = np.vstack(a)
    b, e = [], []
    for sd, d in _read_seeded(str(RETRAINED / 'thermompnn' / 'predictions' / 'epistatic_corrected' / 'seed{seed}' / f'{ds}.csv')):
        k = _keycol(d)
        b.append(_align_id(ids, k, d['pred_combined'].to_numpy()))
        e.append(_align_id(ids, k, d['pred_combined'].to_numpy() - d['pred_single_additive'].to_numpy()))
    if b:
        out['tmpnnd:'] = np.vstack(b)
        out['tmpnnd:epi'] = np.vstack(e)


def _load_spurs(frame, ds, out):
    ids = frame.spine['id'].to_numpy(dtype=object)
    s, m, e = [], [], []
    for sd, d in _read_seeded(str(RETRAINED / 'spurs' / 'predictions' / 'seed{seed}' / f'{ds}.csv')):
        k = (d['id'] if 'id' in d else d['uid']).astype(str).to_numpy()
        s.append(_align_id(ids, k, d['SPURS_single_additive'].to_numpy()))
        m.append(_align_id(ids, k, d['SPURS_multi'].to_numpy()))
        e.append(_align_id(ids, k, d['SPURS_multi'].to_numpy() - d['SPURS_single_additive'].to_numpy()))
    if s:
        out['spurs:'] = np.vstack(s)
        out['spursm:'] = np.vstack(m)
        out['spursm:epi'] = np.vstack(e)


PUBLIC_DIRS = {'esmtherm': 'esmtherm', 'ddgemb': 'ddgemb', 'ddmut': 'ddmut', 'ifum_wt': 'ifum/wt_backbone',
               'ifum_esmfold': 'ifum/native_esmfold', 'geoddg': 'geoddg/faspr', 'esm3dg': 'esm3dg',
               'prosst': 'prosst', 'pmpnn': 'proteinmpnn', 'rosetta': 'rosetta'}


def _load_public(frame, ds, out):
    sp = frame.spine
    for key, sub in PUBLIC_DIRS.items():
        p = PUBLIC / sub / f'{ds}.csv'
        if key == 'ifum_esmfold' and ds not in TSU_SETS and (PUBLIC / 'ifum/native_esmfold_full' / f'{ds}.csv').exists():
            p = PUBLIC / 'ifum/native_esmfold_full' / f'{ds}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        if 'pred' not in d or 'row_id' not in d:
            continue
        out[f'{key}:'] = _align_rowid(sp['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred'].to_numpy())[None, :]


FORGE = {'esm3s': 'esm3-small-2024-08', 'esm3m': 'esm3-medium-2024-08', 'esm3l': 'esm3-large-2024-03'}


def _load_forge(frame, ds, out):
    """ESM3 through Forge has files for the parent sets only (S669, K3822, PTMUL); S461, K2369 and PTMUL-D are
    scored from their parent's predictions, matched on (PDB entry, mutation) (BENCHMARKS section 1.1)."""
    sp = frame.spine
    parent = {'s461': 's669', 'k2369': 'k3822', 'ptmuld': 'ptmul'}.get(ds, ds)
    for key, folder in FORGE.items():
        p = PUBLIC / 'esm3_forge_zero_shot' / folder / f'{parent}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        if 'esm3_score' not in d:
            continue
        mcol = 'mut_type' if 'mut_type' in d else ('mutations' if 'mutations' in d else None)
        if mcol is None:
            continue
        direct = _align_id(sp['id'].to_numpy(dtype=object), d['id'].astype(str).to_numpy(), d['esm3_score'].to_numpy())
        if np.isfinite(direct).mean() > 0.95 and parent == ds:
            out[f'{key}:'] = direct[None, :]
            continue
        fk = d['id'].astype(str).str[:4].str.upper() + '|' + d[mcol].astype(str)
        sk = sp['id'].str[:4].str.upper() + '|' + sp['mut']
        ser = pd.Series(d['esm3_score'].to_numpy(dtype=float), index=fk.to_numpy())
        ser = ser[~ser.index.duplicated(keep='first')]
        v = ser.reindex(sk.to_numpy()).to_numpy(dtype=float)
        if np.isfinite(v).any():
            out[f'{key}:'] = v[None, :]


def _load_small(frame, ds, out):
    p = MSR / 'esm-msr-small_march_crystal' / f'{ds}.csv'
    if p.exists():
        d = pd.read_csv(p, low_memory=False)
        if 'pred' in d and 'row_id' in d:
            out['msrsmall:comb'] = _align_rowid(frame.spine['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred'].to_numpy())[None, :]


def _tidy(out):
    """Registry keys are 'family:head'; heads that carry no suffix are the model's main readout."""
    fixed = {}
    for k, v in out.items():
        fixed[k.rstrip(':') if k.endswith(':') else k] = v
    return fixed


def load_frame(ds, refresh=False):
    pkl = CACHE / f'frame_{ds}.pkl'
    if pkl.exists() and not refresh:
        with open(pkl, 'rb') as fh:
            return pickle.load(fh)
    spine = build_spine(ds)
    fr = Frame(ds, spine)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _load_msr(fr, ds, out)
        _load_me(fr, ds, out)
        _load_tmpnn(fr, ds, out)
        _load_spurs(fr, ds, out)
        _load_public(fr, ds, out)
        _load_forge(fr, ds, out)
        _load_small(fr, ds, out)
    fr.preds = {k: np.asarray(v, dtype=np.float64) for k, v in _tidy(out).items()}
    fr.notes['coverage'] = {k: float(np.isfinite(v).all(axis=0)[spine['use'].to_numpy()].mean()) for k, v in fr.preds.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(pkl, 'wb') as fh:
        pickle.dump(fr, fh, protocol=4)
    return fr
