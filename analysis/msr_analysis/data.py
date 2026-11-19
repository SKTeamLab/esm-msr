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
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (BASE, ROOT, CACHE, MEGA_RAW, MSR, P0, PUBLIC, RETRAINED, TASKS, PER_PROTEIN, DMS3, EXT12, TSU, TSU_CAPPED, TSU_MEGA)

TSU_SETS = set(TSU_CAPPED) | set(TSU_MEGA)

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
    the same file (NaN when a constituent has no single row).  Singles keep their own prediction.
    Accepts 1D ``(n,)`` or 2D ``(n_seeds, n)`` arrays."""
    arr = np.asarray(pred, dtype=float)
    if arr.ndim == 2:
        return np.vstack([sum_singles(spine, r) for r in arr])
    ids = spine['id'].to_numpy(dtype=object)
    nm = spine['n_mut'].to_numpy()
    single = {i: p for i, p in zip(ids[nm == 1], arr[nm == 1])}
    out = arr.copy()
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


_P0_FOLDER = {'tsuboyama_capped_val': 'splits_oct06_capped-val', 'tsuboyama_capped_test': 'splits_oct06_capped-test', 'ptmul': 'PTMUL',
              'tsuboyama_mega_val': 'mega_splits-val', 'tsuboyama_mega_test': 'mega_splits-test'}


def _msr_file(spec, ds, seed, sigma):
    if spec == 'msrp0':
        fam = 'FINAL_p0_mega' if ds in TSU_MEGA else 'FINAL_p0'          # mega_splits: the FINAL_p0 recipe retrained on mega_splits
        base = P0 / _P0_FOLDER.get(ds, ds) / 'final_checkpoints' / fam / f's{seed}'
        p = base / f'best.ckpt_sigma{float(sigma):.1f}_unmasked_predictions.csv'
        if not p.exists():
            p = base / f'best.ckpt_sigma{float(sigma):.1f}_skip_reverse_unmasked_predictions.csv'
        return p
    return _msr_dir(spec, seed, sigma) / f'{ds}.csv'


def _load_msr(frame, ds, out):
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    heads = dict(wt='wt_lora_pred', comb='combined_pred', mt='mt_lora_pred')
    for spec, sigmas, seeds in (('msr', ['1'], SEEDS), ('msrdev', ['1', '0.75', '0.5', '0.25'], SEEDS),
                                ('msrp0', ['1'], SEEDS), ('msrrel', ['1', '0.75', '0.5', '0.25', '0'], (None,))):
        for sg in sigmas:
            arrs = {h: [] for h in heads}
            ok = True
            for sd in seeds:
                p = _msr_file(spec, ds, sd, sg)
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
                if lst and (h in ('wt', 'comb', 'mt') or sg == '1'):
                    out[f'{tag}:{h}'] = np.vstack(lst)
            if f'{tag}:comb' not in out and f'{tag}:wt' in out and f'{tag}:mt' in out:
                out[f'{tag}:comb'] = 0.5 * (out[f'{tag}:mt'] + out[f'{tag}:wt'])
            if (sp['n_mut'] > 1).any() and f'{tag}:comb' in out and f'{tag}:wt' in out:
                wt_add = sum_singles(sp, out[f'{tag}:wt'])
                out[f'{tag}:wt+'] = wt_add
                if sg in ('1', '0'):
                    # Consistent 1x generalized epistasis: 1/2(MT_AB + WT_AB) - WT_A - WT_B = comb_AB - WT_A - WT_B
                    out[f'{tag}:epi'] = out[f'{tag}:comb'] - wt_add


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
        par = _me_stage_arrays(PARENT[ds], psp, additive=ds in DMS3)
        res = {st: np.vstack([_align_id(ids, psp['id'].to_numpy(dtype=object), r) for r in v]) for st, v in par.items()}
    else:
        res = _me_stage_arrays(ds, sp, additive=ds in DMS3)
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
        scale = (1.0 / 2.94) if key == 'rosetta' else 1.0
        out[f'{key}:'] = _align_rowid(sp['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred'].to_numpy(dtype=float) * scale)[None, :]
        if 'pred_additive' in d:      # the model summed over a multi-mutant's single substitutions (harness column); equals `pred` for additive models
            out[f'{key}+:'] = _align_rowid(sp['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred_additive'].to_numpy(dtype=float) * scale)[None, :]


_ROSETTA_FOLDER = {'ptmul': 'PTMUL'}


def _rosetta_ext_root():
    for cand in (
        os.environ.get('MSR_ROSETTA_EXT', str(ROOT / 'rosetta_external_predictions')),     # <set>/rosetta/cartesian_ddg.csv, copied 2026-11-16 from esm-msr-data-oct-14/analysis_notebooks/additional/predictions
        '~/software/esm-msr-release/analysis_notebooks/predictions',
        '~/software/esm-msr/analysis_notebooks/predictions',
    ):
        p = Path(cand).expanduser()
        if p.is_dir():
            return p
    return None


def _rosetta_ext_saved(frame, ds, out):
    """Fallback when the notebook prediction tree is gone: the Rosetta predictions of the external sets saved from the last frames built while it existed
    (``public/rosetta/external_rosetta_preds.pkl``: per set, the spine ids and the ``rosetta`` / ``rosetta+`` / ``rosetta:epi`` arrays, already in kcal/mol)."""
    p = PUBLIC / 'rosetta' / 'external_rosetta_preds.pkl'
    if not p.exists():
        return
    with open(p, 'rb') as fh:
        saved = pickle.load(fh)
    if ds not in saved:
        return
    ids = frame.spine['id'].to_numpy(dtype=object)
    for k, arr in saved[ds]['preds'].items():
        out[k if k.endswith(':epi') else k + ':'] = np.vstack([_align_id(ids, saved[ds]['ids'], row) for row in np.atleast_2d(arr)])


def _load_rosetta_ext(frame, ds, out):
    """Load Rosetta cartesian_ddg predictions for external sets from the notebook prediction tree.

    Files are already stabilizing-positive (cartesian_ddg_1..3 in REU; divided by 2.94 for kcal/mol).
    Skipped when ``rosetta:`` is already populated (e.g. Tsuboyama sets from PUBLIC / 'rosetta').
    """
    if 'rosetta:' in out:
        return
    root = _rosetta_ext_root()
    if root is None:
        return _rosetta_ext_saved(frame, ds, out)
    p = root / _ROSETTA_FOLDER.get(ds, ds) / 'rosetta' / 'cartesian_ddg.csv'
    if not p.exists():
        return _rosetta_ext_saved(frame, ds, out)
    d = pd.read_csv(p, index_col=0)
    uids = d['uid_true'].astype(str) if 'uid_true' in d.columns else d.index.astype(str)
    rep_cols = [f'cartesian_ddg_{s}' for s in SEEDS]
    add_cols = [f'cartesian_ddg_{s}_additive' for s in SEEDS]
    cols = [c for c in rep_cols + add_cols if c in d.columns]
    if not cols:
        return
    d = d[cols].copy()
    d['_uid'] = uids.to_numpy()
    d = d.groupby('_uid', sort=False)[cols].mean()
    ids = frame.spine['id'].to_numpy(dtype=object)
    d_ids = d.index.to_numpy(dtype=object)
    if all(c in d.columns for c in rep_cols):
        out['rosetta:'] = np.vstack([_align_id(ids, d_ids, d[c].to_numpy(dtype=float) / 2.94) for c in rep_cols])
    if all(c in d.columns for c in add_cols) and 'rosetta:' in out:
        out['rosetta+:'] = np.vstack([_align_id(ids, d_ids, d[c].to_numpy(dtype=float) / 2.94) for c in add_cols])
        out['rosetta:epi'] = out['rosetta:'] - out['rosetta+:']


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


# ESM3-small-open with masked scoring: the ESM-MSR checkpoint at sigma = 0 (adapters off, calibration head kept), run by
# cluster/esm3_masked_gamut.sh once per strategy and copied here as <strategy>/analysis/predictions/<folder>/*_predictions.csv.
ESM3_MASKED = MSR / 'esm3_small_open_sigma0_masked'
_ESM3_FOLDER = {'tsuboyama_capped_val': 'splits_oct06_capped-val', 'tsuboyama_capped_test': 'splits_oct06_capped-test', 'ptmul': 'PTMUL'}


def _esm3_file(strategy, ds):
    parent = PARENT.get(ds, ds)
    d = ESM3_MASKED / strategy / 'analysis' / 'predictions' / _ESM3_FOLDER.get(parent, parent)
    hits = sorted(p for p in d.rglob(f'*_sigma0.0*_{strategy}_predictions.csv') if '_smoke' not in p.name) if d.is_dir() else []   # <folder>/<checkpoint directory>/<file>
    return hits[-1] if hits else None


def add_esm3_masked(frame, ds):
    """Add ESM3-small-open scored with masking to ``frame.preds`` (not cached with the frame: the runs may finish after it was built).

    ``esm3mm``       canonical masked marginal: every mutated position masked at once in the wild-type sequence
    ``esm3mm+``      additive sum of single-mutant masked marginals: WT_A + WT_B (each single position masked on its own)
    ``esm3mm:epi``   ``esm3mm`` - ``esm3mm+`` (masked-marginal double minus sum of masked-marginal singles)
    ``esm3ind``      independent masking, mutant pass: each mutated position masked on its own with the other mutations visible
    ``esm3ind:wt``   independent masking, WT pass
    ``esm3ind+``     additive sum of independent-masked WT singles: WT_A + WT_B
    ``esm3avg``      1/2(MT + WT) under independent masking: the combined readout of the checkpoint
    ``esm3avg:epi``  1/2(MT_AB + WT_AB) - WT_A - WT_B (consistent 1x generalized epistasis)
    ``esm3ind:epi``  MT_AB - WT_A - WT_B
    Returns the keys added (none until the cluster jobs are copied)."""
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    added = []
    for strategy, cols in (('marginal', {'esm3mm': 'wt_lora_pred'}),
                           ('independent', {'esm3ind': 'mt_lora_pred', 'esm3ind:wt': 'wt_lora_pred'})):
        p = _esm3_file(strategy, ds)
        if p is None:
            continue
        hdr = pd.read_csv(p, nrows=0).columns
        d = pd.read_csv(p, usecols=['id'] + [c for c in cols.values() if c in hdr], low_memory=False)
        for key, col in cols.items():
            if col in d:
                frame.preds[key] = _align_id(ids, d['id'].astype(str).to_numpy(), d[col].to_numpy())[None, :]
                added.append(key)
    if 'esm3mm' in added:
        frame.preds['esm3mm+'] = sum_singles(sp, frame.preds['esm3mm'])
        frame.preds['esm3mm:epi'] = frame.preds['esm3mm'] - frame.preds['esm3mm+']
        added.extend(['esm3mm+', 'esm3mm:epi'])
    if 'esm3ind' in added and 'esm3ind:wt' in added:
        frame.preds['esm3ind+'] = sum_singles(sp, frame.preds['esm3ind:wt'])
        frame.preds['esm3avg'] = 0.5 * (frame.preds['esm3ind'] + frame.preds['esm3ind:wt'])
        frame.preds['esm3avg:epi'] = frame.preds['esm3avg'] - frame.preds['esm3ind+']
        frame.preds['esm3ind:epi'] = frame.preds['esm3ind'] - frame.preds['esm3ind+']
        added.extend(['esm3ind+', 'esm3avg', 'esm3avg:epi', 'esm3ind:epi'])
    return added


# ----------------------------------------------------------------------------------------------
# mega_splits retrains (MSR_TSU_SPLIT=mega): cluster tracked runs outputs/tracked/<model>_mega*_s<k>, collected to MEGA_RAW
# ----------------------------------------------------------------------------------------------
_MEGA_SPLIT = {'tsuboyama_mega_val': 'val', 'tsuboyama_mega_test': 'test'}


def _mega_seeded(fmt):
    paths = [MEGA_RAW / fmt.format(seed=s) for s in SEEDS]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(f'mega_splits retrain files missing (collect them from the cluster): {missing}')
    return [pd.read_csv(p, low_memory=False) for p in paths]


def _load_mega(frame, ds, out):
    """The mega_splits retrains, with the same keys and epistasis definitions as their capped-split counterparts:
    ``me1`` / ``me2`` (``me2:epi`` = st2 double - same-seed st1 singles), ``tmpnn`` / ``tmpnnd`` (``tmpnnd:epi`` = combined -
    single-additive), ``spurs`` / ``spursm`` (``spursm:epi`` = multi - single-additive).  All stabilizing-positive."""
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    split = _MEGA_SPLIT[ds]
    me = {}
    for stage, key in (('st1', 'me1'), ('st2', 'me2')):
        main = _mega_seeded(f'me_mega_{stage}_s{{seed}}/results_mega_splits_allbl_{split}.csv')
        # the test run also scores the constituent singles of every double (unmeasured ones included) in a separate file
        comp = _mega_seeded(f'me_mega_{stage}_s{{seed}}/results_mega_splits_allbl_{split}comp.csv') if split == 'test' else [None] * len(main)
        arr = []
        for d, c in zip(main, comp):
            d = d if c is None else pd.concat([d, c], ignore_index=True)       # _align_id keeps the first (main-file) row on overlaps
            arr.append(_align_id(ids, (d['pdb_id'].astype(str) + '_' + d['mut_info'].astype(str)).to_numpy(dtype=object), -d['ddg_pred'].to_numpy()))
        me[key] = np.vstack(arr)
        out[f'{key}:'] = me[key]
    out['me2:epi'] = me['me2'] - sum_singles(sp, me['me1'])
    tm = _mega_seeded(f'thermompnn_mega_s{{seed}}/predictions_{split}.csv')
    key = [(d['code'].astype(str) + '_' + d['mut_type'].astype(str)).to_numpy(dtype=object) for d in tm]
    out['tmpnn:'] = np.vstack([_align_id(ids, k, d['pred_single_additive'].to_numpy()) for k, d in zip(key, tm)])
    out['tmpnnd:'] = np.vstack([_align_id(ids, k, d['pred_combined'].to_numpy()) for k, d in zip(key, tm)])
    out['tmpnnd:epi'] = out['tmpnnd:'] - out['tmpnn:']
    sps = _mega_seeded(f'spurs_mega_s{{seed}}/tsuboyama_{split}.csv')
    uid = [d['uid'].astype(str).to_numpy(dtype=object) for d in sps]
    out['spurs:'] = np.vstack([_align_id(ids, k, d['SPURS_single_additive'].to_numpy()) for k, d in zip(uid, sps)])
    out['spursm:'] = np.vstack([_align_id(ids, k, d['SPURS_multi'].to_numpy()) for k, d in zip(uid, sps)])
    out['spursm:epi'] = out['spursm:'] - out['spurs:']
    # SPURS with the adapter trained (spurs_mega_ta_s<k>: the paper's configuration; ``spurs`` above is the frozen random adapter, the released-checkpoint
    # configuration), and the published checkpoint (spurs_mega_public, one run): same files, keys spursu* / spurspub*. The published ThermoMPNN / ThermoMPNN-D
    # ensemble members (thermompnn_mega_public_ens<k>, trained by their authors on mega_splits) stand in for three seeds: tmpnnpub / tmpnndpub.
    # Present only once collected; the mega scripts use whichever exist.
    def _opt(fmt, seeds=SEEDS):
        paths = [MEGA_RAW / fmt.format(seed=s) for s in seeds]
        return [pd.read_csv(q, low_memory=False) for q in paths] if all(q.exists() for q in paths) else None
    for tag, fmt in (('spursu', f'spurs_mega_ta_s{{seed}}/tsuboyama_{split}.csv'), ('spurspub', f'spurs_mega_public/tsuboyama_{split}.csv')):
        d = _opt(fmt, SEEDS if tag == 'spursu' else [0])
        if d is None:
            continue
        u = [x['uid'].astype(str).to_numpy(dtype=object) for x in d]
        a = np.vstack([_align_id(ids, k, x['SPURS_single_additive'].to_numpy()) for k, x in zip(u, d)])
        m = np.vstack([_align_id(ids, k, x['SPURS_multi'].to_numpy()) for k, x in zip(u, d)])
        out[f'{tag}:'], out[f'{tag}m:'], out[f'{tag}m:epi'] = a, m, m - a
    d = _opt(f'thermompnn_mega_public_ens{{seed}}/predictions_{split}.csv')
    if d is not None:
        k = [(x['code'].astype(str) + '_' + x['mut_type'].astype(str)).to_numpy(dtype=object) for x in d]
        a = np.vstack([_align_id(ids, kk, x['pred_single_additive'].to_numpy()) for kk, x in zip(k, d)])
        c = np.vstack([_align_id(ids, kk, x['pred_combined'].to_numpy()) for kk, x in zip(k, d)])
        out['tmpnnpub:'], out['tmpnndpub:'], out['tmpnndpub:epi'] = a, c, c - a


def _tidy(out):
    """Registry keys are 'family:head'; heads that carry no suffix are the model's main readout."""
    fixed = {}
    for k, v in out.items():
        fixed[k.rstrip(':') if k.endswith(':') else k] = v
    return fixed


def load_frame(ds, refresh=False, heldout=True):
    """The frame of dataset ``ds``.  Domainome comes without the domains that are homologous to the training set under the split's four-part rule (``msr_analysis.heldout``; ``heldout=False`` keeps all 522)."""
    fr = _load_frame(ds, refresh)
    if heldout and ds == 'domainome':
        from .heldout import excluded
        keep = ~fr.spine['group'].astype(str).isin(excluded('domainome')).to_numpy()
        fr = Frame(fr.name, fr.spine.loc[keep].reset_index(drop=True), {k: v[:, keep] for k, v in fr.preds.items()}, notes=dict(fr.notes, heldout=True))
    return fr


def _load_frame(ds, refresh=False):
    pkl = CACHE / f'frame_{ds}.pkl'
    if pkl.exists() and not refresh:
        with open(pkl, 'rb') as fh:
            return pickle.load(fh)
    spine = build_spine(ds)
    fr = Frame(ds, spine)
    if ds in PARENT:
        # S461 (subset of S669), K2369 (subset of K3822), and PTMUL-D (subset of PTMUL) ALWAYS slice every model's
        # predictions from their parent superset and merge by variant id onto the child's own corrected label table.
        pfr = _load_frame(PARENT[ds], refresh=refresh)
        ids_c = spine['id'].to_numpy(dtype=object)
        ids_p = pfr.spine['id'].to_numpy(dtype=object)
        fr.preds = {k: np.vstack([_align_id(ids_c, ids_p, r) for r in v]).astype(np.float64) for k, v in pfr.preds.items()}
    else:
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _load_msr(fr, ds, out)
            if ds in TSU_MEGA:
                _load_mega(fr, ds, out)
                fr.preds = {k: np.asarray(v, dtype=np.float64) for k, v in _tidy(out).items()}
                return _finish(fr, pkl)
            _load_me(fr, ds, out)
            _load_tmpnn(fr, ds, out)
            _load_spurs(fr, ds, out)
            _load_public(fr, ds, out)
            _load_rosetta_ext(fr, ds, out)
            _load_forge(fr, ds, out)
            _load_small(fr, ds, out)
        fr.preds = {k: np.asarray(v, dtype=np.float64) for k, v in _tidy(out).items()}
    return _finish(fr, pkl)


def _finish(fr, pkl):
    fr.notes['coverage'] = {k: float(np.isfinite(v).all(axis=0)[fr.spine['use'].to_numpy()].mean()) for k, v in fr.preds.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(pkl, 'wb') as fh:
        pickle.dump(fr, fh, protocol=4)
    return fr
