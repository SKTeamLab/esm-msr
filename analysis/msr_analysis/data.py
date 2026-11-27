"""Load every model's predictions on one dataset, aligned to that dataset's canonical label spine.

``load_frame(dataset)`` returns a :class:`Frame` with stabilizing-positive predictions aligned to ``spine``.
Missing ``<model>:epi`` arrays are synthesized on the fly whenever ``<model>`` and ``<model>+`` exist, so
existing cached pickles immediately expose all roster epistasis keys without a full cache rebuild.
All new cache writes use atomic temp-file replacement and record ``(mtime_ns, size)`` source metadata.
"""
from __future__ import annotations

import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import (
    CACHE,
    FINAL_BENCH as P0,
    MEGA_RAW,
    MSR,
    MSR_ROOT as ROOT,
    PARENT_DATASET as PARENT,
    PUBLIC,
    RETRAINED,
    ROSETTA_EXT_DIR,
    TASKS,
    atomic_pickle_dump,
    runner_folder,
    source_signature,
)
from .registry import DMS3, EXT12, PER_PROTEIN, TSU_CAPPED, TSU_MEGA

CACHE_SCHEMA_VERSION = 2
TSU_SETS = frozenset(TSU_CAPPED) | frozenset(TSU_MEGA)
NO_ESM3DG = frozenset(TSU_CAPPED) | frozenset(TSU_MEGA)

SEEDS = (1, 2, 3)
SPINE_COLS = [
    'id', 'row_id', 'code', 'group', 'truth', 'mut_type', 'ref_seq', 'cens', 'mut_structure', 'code_wt',
    'ddG_ML', 'dddG_ML', 'ddG_additive_ML', 'ddG_dir', 'wt_code', 'dG_ML',
]


@dataclass
class Frame:
    name: str
    spine: pd.DataFrame
    preds: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.spine)

    def keys(self) -> list[str]:
        return list(self.preds)

    def pred(self, key: str):
        """Return ``(n_seeds, n)`` array for model key ``key``."""
        return self.preds[key]

    def has(self, key: str) -> bool:
        return key in self.preds and bool(np.isfinite(self.preds[key]).any())


# ----------------------------------------------------------------------------------------------
# spine & index helpers
# ----------------------------------------------------------------------------------------------
def _first_site(m) -> int:
    mm = re.search(r'(-?\d+)', str(m).split(':')[0])
    return int(mm.group(1)) if mm else -1


def build_spine(ds: str) -> pd.DataFrame:
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
        s['dG_wt'] = (t['dG_ML'] - t['ddG_ML']).astype(float)
    for c in ('cens', 'mut_structure', 'code_wt', 'dddG_ML', 'ddG_additive_ML'):
        if c in t:
            s[c] = t[c]
    use = np.isfinite(s['y'])
    if 'cens' in s:
        use &= (s['cens'] == 0)
    s['use'] = use.to_numpy()
    if 'mut_structure' in s:
        s['wt_backbone'] = (s['mut_structure'] == '-').to_numpy()
    if ds in TSU_SETS:
        s['group'] = s['code_wt'].astype(str)
    if ds in PER_PROTEIN:
        s['cluster'] = s['group']
    else:
        s['cluster'] = _wt_cluster(ds, s)
    return s


def _wt_cluster(ds: str, s: pd.DataFrame) -> pd.Series:
    """Cluster Ssym and external sets on WT protein code rather than variant PDB entry where available."""
    p = MSR / 'baseline_w2a4_m16a16' / 'seed1' / 'sigma1' / f'{ds}.csv'
    if p.exists() and ds in EXT12:
        hdr = pd.read_csv(p, nrows=0).columns
        if 'wt_code' in hdr:
            h = pd.read_csv(p, usecols=['id', 'wt_code'])
            m = dict(zip(h['id'], h['wt_code'].astype(str)))
            return s['id'].map(m).fillna(s['code']).astype(str)
    return s['code'].astype(str)


def _align_id(spine_ids, ids, vals) -> np.ndarray:
    """Lay out ``vals`` (indexed by ``ids``) onto ``spine_ids`` (first occurrence wins; NaN when absent)."""
    ids = np.asarray(ids, dtype=object)
    vals = np.asarray(vals, dtype=float)
    if len(ids) == len(spine_ids) and (ids == spine_ids).all():
        return vals.copy()
    ser = pd.Series(vals, index=ids)
    ser = ser[~ser.index.duplicated(keep='first')]
    return ser.reindex(spine_ids).to_numpy(dtype=float)


def _align_rowid(spine_rowid, rowid, vals) -> np.ndarray:
    ser = pd.Series(np.asarray(vals, dtype=float), index=np.asarray(rowid))
    ser = ser[~ser.index.duplicated(keep='first')]
    return ser.reindex(spine_rowid).to_numpy(dtype=float)


def singles_of_doubles(spine: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ia, ib)``: for every double row in ``spine``, the row indices of its two constituent singles (-1 when absent)."""
    code = spine['code'].astype(str).to_numpy()
    mut = spine['mut'].to_numpy(dtype=object)
    nm = spine['n_mut'].to_numpy()
    where: dict[tuple[str, str], int] = {}
    for i, (c, m, n) in enumerate(zip(code, mut, nm)):
        if n == 1 and (c, m) not in where:
            where[(c, m)] = i
    ia = np.full(len(spine), -1, dtype=int)
    ib = np.full(len(spine), -1, dtype=int)
    for i in np.flatnonzero(nm == 2):
        parts = str(mut[i]).split(':')
        if len(parts) == 2:
            ia[i] = where.get((code[i], parts[0]), -1)
            ib[i] = where.get((code[i], parts[1]), -1)
    return ia, ib


def sum_singles(spine: pd.DataFrame, pred) -> np.ndarray:
    """Sum constituent single-mutant predictions for every multi-mutant (first single row wins on duplicates)."""
    arr = np.asarray(pred, dtype=float)
    if arr.ndim == 2:
        return np.vstack([sum_singles(spine, r) for r in arr])
    ids = spine['id'].to_numpy(dtype=object)
    nm = spine['n_mut'].to_numpy()
    single: dict[str, float] = {}
    for i, p in zip(ids[nm == 1], arr[nm == 1]):
        if i not in single:
            single[i] = p
    out = arr.copy()
    for k in np.flatnonzero(nm > 1):
        prefix, mut = ids[k].rsplit('_', 1) if '_' in ids[k] else ('', ids[k])
        tot = 0.0
        for m in mut.split(':'):
            tot += single.get(f'{prefix}_{m}', np.nan)
        out[k] = tot
    return out


MSR_HEAD_COLS = {'wt': 'wt_lora_pred', 'comb': 'combined_pred', 'mt': 'mt_lora_pred'}


def add_msr_family_preds(out: dict, sp: pd.DataFrame, tag: str, files: list[Path], compute_epi: bool = True):
    """Shared ESM-MSR CSV reader/aligner computing ``wt``, ``mt``, ``comb`` (= 0.5*(wt+mt) fallback), ``wt+``, and ``epi``."""
    ids = sp['id'].to_numpy(dtype=object)
    arrs = {h: [] for h in MSR_HEAD_COLS}
    for p in files:
        hdr = pd.read_csv(p, nrows=0).columns
        use = ['id'] + [c for c in MSR_HEAD_COLS.values() if c in hdr]
        d = pd.read_csv(p, usecols=use, low_memory=False)
        fid = d['id'].astype(str).to_numpy()
        for h, c in MSR_HEAD_COLS.items():
            arrs[h].append(_align_id(ids, fid, d[c].to_numpy()) if c in d else np.full(len(ids), np.nan))
    for h, lst in arrs.items():
        if lst:
            a = np.vstack(lst)
            if np.isfinite(a).any():
                out[f'{tag}:{h}'] = a
    if f'{tag}:comb' not in out and f'{tag}:wt' in out and f'{tag}:mt' in out:
        out[f'{tag}:comb'] = 0.5 * (out[f'{tag}:mt'] + out[f'{tag}:wt'])
    if (sp['n_mut'] > 1).any() and f'{tag}:wt' in out:
        wt_add = sum_singles(sp, out[f'{tag}:wt'])
        out[f'{tag}:wt+'] = wt_add
        if compute_epi and f'{tag}:comb' in out:
            out[f'{tag}:epi'] = out[f'{tag}:comb'] - wt_add


def synthesize_missing_epi(preds: dict) -> dict:
    """Ensure ``f'{k}:epi' = preds[k] - preds[f'{k}+']`` exists for every model with both direct and additive keys."""
    out = dict(preds)
    for k in list(out):
        if k.endswith('+') and ':' not in k:
            base = k[:-1]
            epi_key = f'{base}:epi'
            if base in out and epi_key not in out:
                out[epi_key] = out[base] - out[k]
    for spec in ('msrp0', 'msrp0@0.5', 'msrrel@0'):
        if f'{spec}:comb' in out and f'{spec}:wt+' in out and f'{spec}:epi' not in out:
            out[f'{spec}:epi'] = out[f'{spec}:comb'] - out[f'{spec}:wt+']
    return out


# ----------------------------------------------------------------------------------------------
# model family loaders
# ----------------------------------------------------------------------------------------------
def _msr_dir(spec: str, seed, sigma: str) -> Path:
    if spec == 'msr':
        return MSR / 'baseline_w2a4_m16a16' / f'seed{seed}' / f'sigma{sigma}'
    if spec == 'msrdev':
        return MSR / 'devel_dc8_floorlink_w2a4_m16a16' / f'seed{seed}' / f'sigma{sigma}'
    if spec == 'msrrel':
        return MSR / 'released_capped_w2a4_m16a16' / f'sigma{sigma}'
    raise KeyError(spec)


def _msr_file(spec: str, ds: str, seed, sigma: str) -> Path:
    if spec == 'msrp0':
        fam = 'FINAL_p0_mega' if ds in TSU_MEGA else 'FINAL_p0'
        base = P0 / runner_folder(ds) / 'final_checkpoints' / fam / f's{seed}'
        sg = f'{float(sigma):.1f}'
        p = base / f'best.ckpt_sigma{sg}_unmasked_predictions.csv'
        if not p.exists():
            p = base / f'best.ckpt_sigma{sg}_skip_reverse_unmasked_predictions.csv'
        return p
    return _msr_dir(spec, seed, sigma) / f'{ds}.csv'


def _load_msr(frame: Frame, ds: str, out: dict, sources: list[Path]):
    sp = frame.spine
    specs = (
        ('msr', ['1'], SEEDS),
        ('msrdev', ['1', '0.75', '0.5', '0.25'], SEEDS),
        ('msrp0', ['1', '0.5'], SEEDS),
        ('msrrel', ['1', '0.75', '0.5', '0.25', '0'], (None,)),
    )
    for spec, sigmas, seeds in specs:
        for sg in sigmas:
            files = [_msr_file(spec, ds, sd, sg) for sd in seeds]
            if not all(p.exists() for p in files):
                continue
            sources.extend(files)
            tag = spec if sg == '1' else f'{spec}@{sg}'
            add_msr_family_preds(out, sp, tag, files, compute_epi=(sg in ('1', '0.5', '0')))


def _read_seeded(path_fmt: str, seeds=SEEDS, sources: list[Path] | None = None):
    paths = [Path(path_fmt.format(seed=sd)) for sd in seeds]
    if not all(p.exists() for p in paths):
        return
    if sources is not None:
        sources.extend(paths)
    for sd, p in zip(seeds, paths):
        yield sd, pd.read_csv(p, low_memory=False)


def _me_stage_arrays(ds: str, spine: pd.DataFrame, additive: bool = False, sources: list[Path] | None = None) -> dict:
    """Load stabilizing-positive MutateEverything predictions; builds ``stage+`` from ME's complete file (+ ``testcomp`` on Tsuboyama test)."""
    ids = spine['id'].to_numpy(dtype=object)
    idset = set(ids)
    res = {}
    for stage in ('stage1', 'stage2'):
        arrs, adds = [], []
        fmt = str(RETRAINED / 'mutate_everything' / 'predictions' / stage / 'seed{seed}' / f'{ds}.csv')
        comp_fmt = (
            str(RETRAINED / 'mutate_everything' / 'predictions' / stage / 'seed{seed}' / 'tsuboyama_capped_testcomp.csv')
            if ds == 'tsuboyama_capped_test'
            else None
        )
        for sd, d in _read_seeded(fmt, sources=sources):
            if comp_fmt is not None:
                cp = Path(comp_fmt.format(seed=sd))
                if cp.exists():
                    if sources is not None:
                        sources.append(cp)
                    d = pd.concat([d, pd.read_csv(cp, low_memory=False)], ignore_index=True)
            k1 = d['pdb_id'].astype(str) + '_' + d['mut_info'].astype(str)
            k2 = d['pdb_id'].astype(str).str[:-1] + '_' + d['mut_info'].astype(str)
            key = (k1 if k1.isin(idset).mean() >= k2.isin(idset).mean() else k2).to_numpy(dtype=object)
            v = -d['ddg_pred'].to_numpy()
            arrs.append(_align_id(ids, key, v))
            if additive:
                dedup = ~pd.Index(key).duplicated(keep='first')
                own_keys = key[dedup]
                own_vals = v[dedup]
                own = pd.DataFrame({
                    'id': own_keys,
                    'n_mut': pd.Series(own_keys).str.rsplit('_', n=1).str[-1].str.count(':').to_numpy() + 1,
                })
                adds.append(_align_id(ids, own_keys, sum_singles(own, own_vals)))
        if arrs:
            res[stage] = np.vstack(arrs)
        if adds:
            res[stage + '+'] = np.vstack(adds)
    return res


def _load_me(frame: Frame, ds: str, out: dict, sources: list[Path]):
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    want_add = bool((sp['n_mut'] > 1).any())
    if ds in PARENT:
        t = pd.read_csv(TASKS / f'{PARENT[ds]}.csv.gz', usecols=['id', 'mut_type'], low_memory=False)
        psp = pd.DataFrame({'id': t['id'].astype(str), 'n_mut': t['mut_type'].astype(str).str.count(':') + 1})
        par = _me_stage_arrays(PARENT[ds], psp, additive=want_add, sources=sources)
        res = {st: np.vstack([_align_id(ids, psp['id'].to_numpy(dtype=object), r) for r in v]) for st, v in par.items()}
    else:
        res = _me_stage_arrays(ds, sp, additive=want_add, sources=sources)
    for stage, key in (('stage1', 'me1'), ('stage2', 'me2')):
        if stage in res:
            out[key] = res[stage]
    if 'stage1' in res and 'stage2' in res:
        st1_add = res['stage1+'] if 'stage1+' in res else sum_singles(sp, res['stage1'])
        out['me2:epi'] = res['stage2'] - st1_add
    for stage, key in (('stage1+', 'me1+'), ('stage2+', 'me2+')):
        if stage in res:
            out[key] = res[stage]


def _keycol(d: pd.DataFrame) -> np.ndarray:
    if 'id' in d:
        return d['id'].astype(str).to_numpy()
    if 'uid' in d:
        return d['uid'].astype(str).to_numpy()
    return (d['code'].astype(str) + '_' + d['mut_type'].astype(str)).to_numpy()


def _load_tmpnn(frame: Frame, ds: str, out: dict, sources: list[Path]):
    ids = frame.spine['id'].to_numpy(dtype=object)
    a = []
    for _, d in _read_seeded(str(RETRAINED / 'thermompnn' / 'predictions' / 'single_additive' / 'seed{seed}' / f'{ds}.csv'), sources=sources):
        a.append(_align_id(ids, _keycol(d), d['pred_single_additive'].to_numpy()))
    if a:
        out['tmpnn'] = np.vstack(a)
    b, e = [], []
    for _, d in _read_seeded(str(RETRAINED / 'thermompnn' / 'predictions' / 'epistatic_corrected' / 'seed{seed}' / f'{ds}.csv'), sources=sources):
        k = _keycol(d)
        b.append(_align_id(ids, k, d['pred_combined'].to_numpy()))
        e.append(_align_id(ids, k, d['pred_combined'].to_numpy() - d['pred_single_additive'].to_numpy()))
    if b:
        out['tmpnnd'] = np.vstack(b)
        out['tmpnnd:epi'] = np.vstack(e)


def _load_spurs(frame: Frame, ds: str, out: dict, sources: list[Path]):
    ids = frame.spine['id'].to_numpy(dtype=object)
    s, m, e = [], [], []
    for _, d in _read_seeded(str(RETRAINED / 'spurs' / 'predictions' / 'seed{seed}' / f'{ds}.csv'), sources=sources):
        k = (d['id'] if 'id' in d else d['uid']).astype(str).to_numpy()
        s.append(_align_id(ids, k, d['SPURS_single_additive'].to_numpy()))
        m.append(_align_id(ids, k, d['SPURS_multi'].to_numpy()))
        e.append(_align_id(ids, k, d['SPURS_multi'].to_numpy() - d['SPURS_single_additive'].to_numpy()))
    if s:
        out['spurs'] = np.vstack(s)
        out['spursm'] = np.vstack(m)
        out['spursm:epi'] = np.vstack(e)


PUBLIC_DIRS = {
    'esmtherm': 'esmtherm', 'ddgemb': 'ddgemb', 'ddmut': 'ddmut',
    'ifum_wt': 'ifum/wt_backbone', 'ifum_esmfold': 'ifum/native_esmfold',
    'geoddg': 'geoddg/faspr', 'esm3dg': 'esm3dg',
    'prosst': 'prosst', 'pmpnn': 'proteinmpnn', 'rosetta': 'rosetta',
}


def _load_public(frame: Frame, ds: str, out: dict, sources: list[Path]):
    sp = frame.spine
    for key, sub in PUBLIC_DIRS.items():
        p = PUBLIC / sub / f'{ds}.csv'
        if key == 'ifum_esmfold' and ds not in TSU_SETS and (PUBLIC / 'ifum/native_esmfold_full' / f'{ds}.csv').exists():
            p = PUBLIC / 'ifum/native_esmfold_full' / f'{ds}.csv'
        if not p.exists():
            continue
        sources.append(p)
        d = pd.read_csv(p, low_memory=False)
        if 'pred' not in d or 'row_id' not in d:
            continue
        scale = (1.0 / 2.94) if key == 'rosetta' else 1.0
        out[key] = _align_rowid(sp['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred'].to_numpy(dtype=float) * scale)[None, :]
        if 'pred_additive' in d:
            out[f'{key}+'] = _align_rowid(sp['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred_additive'].to_numpy(dtype=float) * scale)[None, :]
            out[f'{key}:epi'] = out[key] - out[f'{key}+']


_PROSST_MULTI_MAP = {
    'tsuboyama_capped_val': 'megascale/splits_oct06_capped_val.csv.gz',
    'tsuboyama_capped_test': 'megascale/splits_oct06_capped_test.csv.gz',
    'tsuboyama_mega_val': 'megascale/mega_splits_val.csv.gz',
    'tsuboyama_mega_test': 'megascale/mega_splits_test.csv.gz',
    'ptmul': 'external/ptmul.csv.gz',
    'ptmuld': 'external/ptmuld.csv.gz',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'dms/DLG4_HUMAN_Faure_2021_binding_domain.csv.gz',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'dms/GRB2_HUMAN_Faure_2021_binding_domain.csv.gz',
    'MYO_HUMAN_Kung_2025_display': 'dms/MYO_HUMAN_Kung_2025_display.csv.gz',
}


def _load_prosst_multi(frame: Frame, ds: str, out: dict, sources: list[Path]):
    """Load ProSST dual-view unmasked predictions (comb_pred, mt_pred) and self-bias-cancelled epistasis."""
    rel = _PROSST_MULTI_MAP.get(ds)
    if rel is None:
        return
    p = PUBLIC / 'prosst_multi' / 'dual_unmasked' / rel
    if not p.exists():
        return
    sources.append(p)
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    hdr = pd.read_csv(p, nrows=0).columns
    kcol = 'uid' if 'uid' in hdr else ('id' if 'id' in hdr else None)
    if kcol is None or 'comb_pred' not in hdr:
        return
    use = [c for c in (kcol, 'wt_pred', 'mt_pred', 'comb_pred', 'epi_pred') if c in hdr]
    d = pd.read_csv(p, usecols=use, low_memory=False)
    k = d[kcol].astype(str).to_numpy()
    comb = _align_id(ids, k, d['comb_pred'].to_numpy(dtype=float))[None, :]
    out['prosst:comb'] = comb
    if 'mt_pred' in d:
        out['prosst:mt'] = _align_id(ids, k, d['mt_pred'].to_numpy(dtype=float))[None, :]
    if 'prosst' not in out and 'wt_pred' in d:
        out['prosst'] = _align_id(ids, k, d['wt_pred'].to_numpy(dtype=float))[None, :]
    if (sp['n_mut'] > 1).any():
        comb_add = sum_singles(sp, comb)
        out['prosst:comb+'] = comb_add
        # Subtracting comb_A + comb_B cancels ProSST's unmasked single-site self-bias (+2.9/site),
        # matching direct_AB - (direct_A + direct_B) used by ProteinMPNN, GeoDDG-3D, Rosetta, and ESM3dG.
        out['prosst:epi'] = comb - comb_add
        if 'prosst' in out:
            out['prosst:epi_wt'] = comb - out['prosst']


def _rosetta_ext_saved(frame: Frame, ds: str, out: dict, sources: list[Path]):
    p = PUBLIC / 'rosetta' / 'external_rosetta_preds.pkl'
    if not p.exists():
        return
    sources.append(p)
    with open(p, 'rb') as fh:
        saved = pickle.load(fh)
    if ds not in saved:
        return
    ids = frame.spine['id'].to_numpy(dtype=object)
    for k, arr in saved[ds]['preds'].items():
        out[k] = np.vstack([_align_id(ids, saved[ds]['ids'], row) for row in np.atleast_2d(arr)])


def _load_rosetta_ext(frame: Frame, ds: str, out: dict, sources: list[Path]):
    if 'rosetta' in out:
        return
    p = ROSETTA_EXT_DIR / runner_folder(ds) / 'rosetta' / 'cartesian_ddg.csv'
    if not p.exists():
        return _rosetta_ext_saved(frame, ds, out, sources)
    sources.append(p)
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
        out['rosetta'] = np.vstack([_align_id(ids, d_ids, d[c].to_numpy(dtype=float) / 2.94) for c in rep_cols])
    if all(c in d.columns for c in add_cols) and 'rosetta' in out:
        out['rosetta+'] = np.vstack([_align_id(ids, d_ids, d[c].to_numpy(dtype=float) / 2.94) for c in add_cols])
        out['rosetta:epi'] = out['rosetta'] - out['rosetta+']


FORGE = {'esm3s': 'esm3-small-2024-08', 'esm3m': 'esm3-medium-2024-08', 'esm3l': 'esm3-large-2024-03'}


def _load_forge(frame: Frame, ds: str, out: dict, sources: list[Path]):
    sp = frame.spine
    parent = PARENT.get(ds, ds)
    for key, folder in FORGE.items():
        p = PUBLIC / 'esm3_forge_zero_shot' / folder / f'{parent}.csv'
        if not p.exists():
            continue
        sources.append(p)
        d = pd.read_csv(p, low_memory=False)
        if 'esm3_score' not in d:
            continue
        mcol = 'mut_type' if 'mut_type' in d else ('mutations' if 'mutations' in d else None)
        if mcol is None:
            continue
        direct = _align_id(sp['id'].to_numpy(dtype=object), d['id'].astype(str).to_numpy(), d['esm3_score'].to_numpy())
        if np.isfinite(direct).mean() > 0.95 and parent == ds:
            out[key] = direct[None, :]
            continue
        fk = d['id'].astype(str).str[:4].str.upper() + '|' + d[mcol].astype(str)
        sk = sp['id'].str[:4].str.upper() + '|' + sp['mut']
        ser = pd.Series(d['esm3_score'].to_numpy(dtype=float), index=fk.to_numpy())
        ser = ser[~ser.index.duplicated(keep='first')]
        v = ser.reindex(sk.to_numpy()).to_numpy(dtype=float)
        if np.isfinite(v).any():
            out[key] = v[None, :]


def _load_small(frame: Frame, ds: str, out: dict, sources: list[Path]):
    p = MSR / 'esm-msr-small_march_crystal' / f'{ds}.csv'
    if p.exists():
        sources.append(p)
        d = pd.read_csv(p, low_memory=False)
        if 'pred' in d and 'row_id' in d:
            out['msrsmall:comb'] = _align_rowid(frame.spine['row_id'].to_numpy(), d['row_id'].to_numpy(), d['pred'].to_numpy())[None, :]


ESM3_MASKED = MSR / 'esm3_small_open_sigma0_masked'


def _esm3_file(strategy: str, ds: str) -> Path | None:
    parent = PARENT.get(ds, ds)
    d = ESM3_MASKED / strategy / 'analysis' / 'predictions' / runner_folder(parent)
    hits = sorted(p for p in d.rglob(f'*_sigma0.0*_{strategy}_predictions.csv') if '_smoke' not in p.name) if d.is_dir() else []
    return hits[-1] if hits else None


def add_esm3_masked(frame: Frame, ds: str) -> list[str]:
    """Add ESM3-small-open masked-scoring readouts to ``frame.preds`` on demand."""
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    added = []
    for strategy, cols in (
        ('marginal', {'esm3mm': 'wt_lora_pred'}),
        ('independent', {'esm3ind': 'mt_lora_pred', 'esm3ind:wt': 'wt_lora_pred'}),
    ):
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
# mega_splits retrains
# ----------------------------------------------------------------------------------------------
_MEGA_SPLIT = {'tsuboyama_mega_val': 'val', 'tsuboyama_mega_test': 'test'}


def _mega_seeded(fmt: str, sources: list[Path]):
    paths = [MEGA_RAW / fmt.format(seed=s) for s in SEEDS]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(f'mega_splits retrain files missing: {missing}')
    sources.extend(paths)
    return [pd.read_csv(p, low_memory=False) for p in paths]


def _load_mega(frame: Frame, ds: str, out: dict, sources: list[Path]):
    sp = frame.spine
    ids = sp['id'].to_numpy(dtype=object)
    split = _MEGA_SPLIT[ds]
    me = {}
    for stage, key in (('st1', 'me1'), ('st2', 'me2')):
        main = _mega_seeded(f'me_mega_{stage}_s{{seed}}/results_mega_splits_allbl_{split}.csv', sources)
        comp = _mega_seeded(f'me_mega_{stage}_s{{seed}}/results_mega_splits_allbl_{split}comp.csv', sources) if split == 'test' else [None] * len(main)
        arr = []
        for d, c in zip(main, comp):
            d = d if c is None else pd.concat([d, c], ignore_index=True)
            arr.append(_align_id(ids, (d['pdb_id'].astype(str) + '_' + d['mut_info'].astype(str)).to_numpy(dtype=object), -d['ddg_pred'].to_numpy()))
        me[key] = np.vstack(arr)
        out[key] = me[key]
    out['me2:epi'] = me['me2'] - sum_singles(sp, me['me1'])
    tm = _mega_seeded(f'thermompnn_mega_s{{seed}}/predictions_{split}.csv', sources)
    key = [(d['code'].astype(str) + '_' + d['mut_type'].astype(str)).to_numpy(dtype=object) for d in tm]
    out['tmpnn'] = np.vstack([_align_id(ids, k, d['pred_single_additive'].to_numpy()) for k, d in zip(key, tm)])
    out['tmpnnd'] = np.vstack([_align_id(ids, k, d['pred_combined'].to_numpy()) for k, d in zip(key, tm)])
    out['tmpnnd:epi'] = out['tmpnnd'] - out['tmpnn']
    sps = _mega_seeded(f'spurs_mega_s{{seed}}/tsuboyama_{split}.csv', sources)
    uid = [d['uid'].astype(str).to_numpy(dtype=object) for d in sps]
    out['spurs'] = np.vstack([_align_id(ids, k, d['SPURS_single_additive'].to_numpy()) for k, d in zip(uid, sps)])
    out['spursm'] = np.vstack([_align_id(ids, k, d['SPURS_multi'].to_numpy()) for k, d in zip(uid, sps)])
    out['spursm:epi'] = out['spursm'] - out['spurs']

    def _opt(fmt: str, seeds=SEEDS):
        paths = [MEGA_RAW / fmt.format(seed=s) for s in seeds]
        if not all(q.exists() for q in paths):
            return None
        sources.extend(paths)
        return [pd.read_csv(q, low_memory=False) for q in paths]

    d = _opt(f'spurs_mega_ta_s{{seed}}/tsuboyama_{split}.csv', SEEDS)
    if d is not None:
        u = [x['uid'].astype(str).to_numpy(dtype=object) for x in d]
        a = np.vstack([_align_id(ids, k, x['SPURS_single_additive'].to_numpy()) for k, x in zip(u, d)])
        m = np.vstack([_align_id(ids, k, x['SPURS_multi'].to_numpy()) for k, x in zip(u, d)])
        out['spursu'], out['spursum'], out['spursum:epi'] = a, m, m - a
    d = _opt(f'thermompnn_mega_public_ens{{seed}}/predictions_{split}.csv')
    if d is not None:
        k = [(x['code'].astype(str) + '_' + x['mut_type'].astype(str)).to_numpy(dtype=object) for x in d]
        a = np.vstack([_align_id(ids, kk, x['pred_single_additive'].to_numpy()) for kk, x in zip(k, d)])
        c = np.vstack([_align_id(ids, kk, x['pred_combined'].to_numpy()) for kk, x in zip(k, d)])
        out['tmpnnpub'], out['tmpnndpub'], out['tmpnndpub:epi'] = a, c, c - a


def _tidy(out: dict) -> dict:
    return {(k.rstrip(':') if k.endswith(':') else k): np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _maybe_attach_msrp0_s05(fr: Frame, ds: str) -> dict:
    """Attach ``msrp0@0.5:*`` on the fly when present on disk and absent from a pre-existing cached frame."""
    if 'msrp0@0.5:wt' in fr.preds:
        return fr.preds
    target_ds = PARENT.get(ds, ds)
    files = [_msr_file('msrp0', target_ds, sd, '0.5') for sd in SEEDS]
    if not all(p.exists() for p in files):
        return fr.preds
    preds = dict(fr.preds)
    if target_ds != ds:
        tmp_sp = build_spine(target_ds)
        tmp_out: dict = {}
        add_msr_family_preds(tmp_out, tmp_sp, 'msrp0@0.5', files, compute_epi=True)
        ids_c = fr.spine['id'].to_numpy(dtype=object)
        ids_p = tmp_sp['id'].to_numpy(dtype=object)
        for k, v in tmp_out.items():
            preds[k] = np.vstack([_align_id(ids_c, ids_p, r) for r in v]).astype(np.float64)
    else:
        add_msr_family_preds(preds, fr.spine, 'msrp0@0.5', files, compute_epi=True)
    return preds


def load_frame(ds: str, refresh: bool = False, heldout: bool = True, check_sources: bool = False) -> Frame:
    """Load the :class:`Frame` of dataset ``ds``.

    - Synthesizes any missing ``<model>:epi`` arrays from ``<model>`` and ``<model>+`` on the fly.
    - Strips ``esm3dg*`` on Tsuboyama splits (training-set overlap).
    - Filters ``domainome`` to held-out domains when ``heldout=True``.
    - When ``check_sources=True``, also verifies stored ``source_sig`` ``(mtime_ns, size)`` metadata and rebuilds if stale.
    """
    fr = _load_frame(ds, refresh=refresh, check_sources=check_sources)
    preds = synthesize_missing_epi(_maybe_attach_msrp0_s05(fr, ds))
    if ds in NO_ESM3DG:
        preds = {k: v for k, v in preds.items() if not k.startswith('esm3dg')}
    notes = dict(fr.notes)
    if heldout and ds == 'domainome':
        from .heldout import excluded
        keep = ~fr.spine['group'].astype(str).isin(excluded('domainome')).to_numpy()
        return Frame(fr.name, fr.spine.loc[keep].reset_index(drop=True),
                     {k: v[:, keep] for k, v in preds.items()}, notes=dict(notes, heldout=True))
    return Frame(fr.name, fr.spine, preds, notes=notes)


def _load_frame(ds: str, refresh: bool = False, check_sources: bool = False) -> Frame:
    pkl = CACHE / f'frame_{ds}.pkl'
    if pkl.exists() and not refresh:
        with open(pkl, 'rb') as fh:
            cached = pickle.load(fh)
        if not check_sources:
            return cached
        stored_sig = cached.notes.get('source_sig')
        if stored_sig is not None:
            cur_sig = source_signature(p for p, _, _ in stored_sig)
            if cur_sig == stored_sig:
                return cached

    sources: list[Path] = [TASKS / f'{ds}.csv.gz']
    spine = build_spine(ds)
    fr = Frame(ds, spine)
    if ds in PARENT:
        pfr = _load_frame(PARENT[ds], refresh=refresh, check_sources=check_sources)
        ids_c = spine['id'].to_numpy(dtype=object)
        ids_p = pfr.spine['id'].to_numpy(dtype=object)
        fr.preds = synthesize_missing_epi(
            {k: np.vstack([_align_id(ids_c, ids_p, r) for r in v]).astype(np.float64) for k, v in pfr.preds.items()}
        )
        parent_sig = pfr.notes.get('source_sig', ())
        sources.extend(Path(p) for p, _, _ in parent_sig)
    else:
        out: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _load_msr(fr, ds, out, sources)
            if ds in TSU_MEGA:
                _load_mega(fr, ds, out, sources)
                _load_public(fr, ds, out, sources)
                _load_prosst_multi(fr, ds, out, sources)
                fr.preds = synthesize_missing_epi(_tidy(out))
                return _finish(fr, pkl, sources)
            _load_me(fr, ds, out, sources)
            _load_tmpnn(fr, ds, out, sources)
            _load_spurs(fr, ds, out, sources)
            _load_public(fr, ds, out, sources)
            _load_prosst_multi(fr, ds, out, sources)
            _load_rosetta_ext(fr, ds, out, sources)
            _load_forge(fr, ds, out, sources)
            _load_small(fr, ds, out, sources)
        fr.preds = synthesize_missing_epi(_tidy(out))
    return _finish(fr, pkl, sources)


def _finish(fr: Frame, pkl: Path, sources: list[Path]) -> Frame:
    fr.notes['coverage'] = {
        k: float(np.isfinite(v).all(axis=0)[fr.spine['use'].to_numpy()].mean())
        for k, v in fr.preds.items()
    }
    fr.notes['cache_schema'] = CACHE_SCHEMA_VERSION
    fr.notes['source_sig'] = source_signature(sources)
    atomic_pickle_dump(fr, pkl)
    return fr
