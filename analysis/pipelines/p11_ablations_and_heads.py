#!/usr/bin/env python
"""Unified 11-family ``FINAL_*`` ablation benchmark pipeline and Dual-view vs WT-view mechanistic study.

Consolidates ``98_final_ablation_table.py``, ``ablation_vs_p0/{01..06}``, and ``combined_vs_wt/{01..08}`` with:
  1. **Scientific fix on ``table_main_delta`` (`pg|avg`)**:
     - ``table_main_delta`` evaluates the true **deployed readout** on ProteinGym (`comb` on the 47 multi-mutant held-out
       assays + `wt` on the 126 single-mutant held-out assays = 173 held-out assays, matching Figure 3b) ONLY for families
       that have complete ``final_pgym_comb`` predictions (`FINAL_p0`, `FINAL_p1`), leaving families without multi-mutant
       ``comb`` runs as ``—`` rather than silently substituting ``wt`` or scoring MT-only ablations (`pw0`, `r2a4`, `p1_marg`)
       on their untouched WT heads.
     - ``table_wt_delta`` evaluates ``wt`` across all WT-relevant families with complete ``final_pgym_wt`` runs.
     - Both tables use **joint ``[category-stratified assay × triplicate-seed]`` paired bootstrap** (`B >= 1000`),
       eliminating false-positive significance flags caused by unresampled seed drift.
  2. **All 11 ``FINAL_*`` families** (including ``FINAL_p0_norange`` and ``FINAL_p0_mse``) + 2 cross-masking evaluation
     arms, styled with canonical ``msr_analysis.models.ABLATION_STYLES`` (never ``#c0392b`` red, ``'D'`` reserved for ``FINAL_p0``).
  3. **Two Consolidated Extended Data Figures**:
     - ``results/09_ablation/ablation_architecture_suite.{png,pdf}``
     - ``results/99_misc/dual_view/dual_view_summary.{png,pdf}`` (featuring the ``R² ≈ 0.81`` ``when_rule`` law)
"""
from __future__ import annotations

import os

for _blas_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_blas_var, '1')

import argparse
import copy
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import rankdata, spearmanr, wilcoxon

from esm_msr import epi_metrics as E
from msr_analysis import epistasis as EPI
from msr_analysis import models as M
from msr_analysis import proteingym as PG
from msr_analysis import stats as ST
from msr_analysis import style
from msr_analysis.boot import DatasetBoot, SpearmanEngine, _make_seed_sampler, ci, global_cluster_index
from msr_analysis.common import epi_rows, subframe
from msr_analysis.data import Frame, MSR_HEAD_COLS, _align_id, build_spine, sum_singles
from msr_analysis.final_runs import FAMILIES, MEGA_FOLDER, REFERENCE as REF, SEEDS, _pred_file, load_final_frame
from msr_analysis.heldout import excluded
from msr_analysis.paths import CACHE, FINAL_PGYM_COMB, atomic_pickle_dump, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.registry import DMS3, EXT12, LABEL, PER_PROTEIN
from msr_analysis.tables import write_stat_table

STAGE = 'ablations_and_heads'
DESCRIPTION = '11-family FINAL_* ablation tables, joint 3-seed paired bootstraps, and Dual-view vs WT-view mechanistic suite.'

EXT_SINGLE = [d for d in EXT12 if d not in ('ptmul', 'ptmuld')]
EXT_MULTI = ['ptmul', 'ptmuld']
DMS_MULTI = list(DMS3)
DMS3_SET = frozenset(DMS3)
DMS3_COLS = DMS3_SET | {f'{d}|multi' for d in DMS3} | {f'{d}|singles' for d in DMS3} | {'dms_mean', 'dms_multi_mean'}
PG_CATS = list(PG.PG_CATEGORIES_ALPHA)
ABL_ARMS = [f for f in FAMILIES if f != REF and '@' not in f]  # all 10 trained ablation arms
MAIN_11_FAMS = [REF] + ABL_ARMS
MT_ONLY = {'FINAL_p0_pw0', 'FINAL_p0_r2a4', 'FINAL_p1_marg'}
MULTI_COLS = set(EXT_MULTI) | set(DMS_MULTI) | {'tsuboyama_capped_val', 'tsuboyama_capped_test'}
EPI_METRICS = ('naive_rho', 'beyond_rho', 'cell_mag', 'cell_rank')

COLLABEL = {
    'tsuboyama_capped_test': 'Tsu test',
    'tsuboyama_capped_test|singles': 'Tsu test singles',
    'tsuboyama_capped_test|multi': 'Tsu test doubles',
    'tsuboyama_capped_test|dddG': 'Tsu test ΔΔΔG',
    'ext_singles_mean': 'Ext. singles (10)',
    'ext_multi_mean': 'PTMUL/-D',
    'dms_mean': 'DMS (3)',
    'dms_multi_mean': 'DMS multis (3)',
    'domainome': 'Domainome',
    'domainome|lowid': 'Domainome low-id',
    'pg|avg': 'PG avg',
    'pg|avg_lowid': 'PG avg low-id',
    'pg|Stability': 'PG Stab.',
    'pg|Stability_lowid': 'PG Stab. low-id',
}
for _d in EXT12 + DMS3:
    COLLABEL.setdefault(_d, LABEL.get(_d, _d))
    COLLABEL.setdefault(f'{_d}|multi', LABEL.get(_d, _d) + ' multi')
    COLLABEL.setdefault(f'{_d}|singles', LABEL.get(_d, _d) + ' singles')

HEADLINE = [
    'tsuboyama_capped_test', 'tsuboyama_capped_test|dddG', 'ext_singles_mean',
    'ext_multi_mean', 'domainome|lowid', 'pg|avg',
]
LAYOUTS = {
    'A_compact_primary': dict(heads=['primary'], cols=HEADLINE),
    'B_compact_wt_vs_comb': dict(
        heads=['wt', 'comb'],
        cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|singles', 'tsuboyama_capped_test|multi',
              'ext_singles_mean', 'ext_multi_mean', 'domainome|lowid', 'pg|avg'],
    ),
    'C1_in_distribution': dict(
        heads=['primary'],
        cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|singles', 'tsuboyama_capped_test|multi', 'tsuboyama_capped_test|dddG'],
    ),
    'C2_out_of_distribution': dict(
        heads=['primary'],
        cols=['ext_singles_mean', 'ext_multi_mean', 'domainome', 'domainome|lowid', 'pg|avg', 'pg|avg_lowid', 'pg|Stability_lowid'],
    ),
    'D_full_primary': dict(
        heads=['primary'],
        cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|dddG'] + EXT12 + ['domainome', 'domainome|lowid', 'pg|avg'],
    ),
    'E_multi_readouts': dict(
        heads=['wt', 'wt+', 'comb'],
        cols=['tsuboyama_capped_test|multi', 'ptmul', 'ptmuld'],
    ),
}
COL_LABEL = {
    'tsuboyama_capped_test': 'Tsuboyama test',
    'tsuboyama_capped_test|multi': 'Tsuboyama doubles',
    'tsuboyama_capped_test|singles': 'Tsuboyama singles',
    'ext_singles_mean': 'External singles (10)',
    'ext_multi_mean': 'PTMUL / PTMUL-D',
    'dms_mean': 'DMS (3)',
    'domainome|lowid': 'Domainome low-id',
    'pg|avg': 'ProteinGym avg',
    'tsuboyama_capped_test|beyond_rho': 'Epistasis beyond ρ',
    'tsuboyama_capped_test|cell_mag': 'Epistasis cell mag.',
}
GROUP_LABEL = {
    'masking': 'Masking',
    'loss': 'Loss',
    'architecture': 'Architecture',
    'data': 'Training data',
    'cross-mask': 'Cross-masking (evaluation only)',
}

_FALLBACK_ABL_STYLES = {
    'FINAL_p0_deepfloor': M.StyleSpec(
        label='Deep floor (−1, 1)',
        color='#148f77',
        marker='h',
        fill=True,
        group='loss',
        model='FINAL_p0_deepfloor',
        ls='-',
        lw=2.0,
    ),
}


def _abl_style(fam: str) -> M.StyleSpec:
    if fam in M.ABLATION_STYLES:
        return M.ABLATION_STYLES[fam]
    if fam in _FALLBACK_ABL_STYLES:
        return _FALLBACK_ABL_STYLES[fam]
    meta = FAMILIES.get(fam, {})
    return M.StyleSpec(
        label=meta.get('short', fam),
        color='#566573',
        marker='o',
        fill=True,
        group=meta.get('group', 'loss'),
        model=fam,
        ls='-',
        lw=2.0,
    )


def is_multi_col(c: str) -> bool:
    return c in MULTI_COLS or c.endswith('|multi')


# ----------------------------------------------------------------------------------------------
# Frame & column scoring
# ----------------------------------------------------------------------------------------------
def _load_all_final_frames(datasets: list[str], sigma: str | float = '1.0', procs: int = 8) -> dict[str, Frame]:
    """Load all ``FINAL_*`` frames across ``datasets`` with file-level thread parallelism (matching ``load_final_frame``)."""
    n_workers = max(int(procs), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        spines = dict(ex.map(lambda d: (d, build_spine(d)), datasets))

    # Schedule largest datasets first so worker threads stay saturated without a heavy tail
    ds_prio = sorted(datasets, key=lambda d: len(spines[d]), reverse=True)
    file_tasks: list[tuple[str, str, int, Path]] = []
    fam_seeds_by_ds: dict[str, dict[str, list[int]]] = {d: {} for d in datasets}
    for ds in ds_prio:
        for fam in FAMILIES:
            if fam == 'FINAL_p0_mega' and ds in MEGA_FOLDER:
                continue
            have = [(s, _pred_file(fam, s, ds, sigma)) for s in SEEDS]
            have = [(s, p) for s, p in have if p is not None]
            if not have:
                continue
            fam_seeds_by_ds[ds][fam] = [s for s, _ in have]
            for s, p in have:
                file_tasks.append((ds, fam, s, p))

    sp_ids = {d: sp['id'].to_numpy(dtype=object) for d, sp in spines.items()}

    def _read_and_align(item: tuple[str, str, int, Path]):
        ds, fam, s, p = item
        hdr = pd.read_csv(p, nrows=0).columns
        use = ['id'] + [c for c in MSR_HEAD_COLS.values() if c in hdr]
        df_p = pd.read_csv(p, usecols=use, engine='c', low_memory=False)
        fid = df_p['id'].astype(str).to_numpy()
        ids_ref = sp_ids[ds]
        aligned = {
            h: (_align_id(ids_ref, fid, df_p[c].to_numpy()) if c in df_p else np.full(len(ids_ref), np.nan))
            for h, c in MSR_HEAD_COLS.items()
        }
        return ds, fam, s, aligned

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        loaded_files = list(ex.map(_read_and_align, file_tasks))

    by_ds_fam: dict[tuple[str, str], list[tuple[int, dict[str, np.ndarray]]]] = {}
    for ds, fam, s, aligned in loaded_files:
        by_ds_fam.setdefault((ds, fam), []).append((s, aligned))

    frames: dict[str, Frame] = {}
    for ds in datasets:
        sp = spines[ds]
        has_mm = bool((sp['n_mut'] > 1).any())
        preds: dict[str, np.ndarray] = {}
        for fam, seed_list in fam_seeds_by_ds[ds].items():
            items = sorted(by_ds_fam[(ds, fam)], key=lambda x: seed_list.index(x[0]))
            for h in MSR_HEAD_COLS:
                a = np.vstack([al[h] for _, al in items])
                if np.isfinite(a).any():
                    preds[f'{fam}:{h}'] = a
            if f'{fam}:comb' not in preds and f'{fam}:wt' in preds and f'{fam}:mt' in preds:
                preds[f'{fam}:comb'] = 0.5 * (preds[f'{fam}:mt'] + preds[f'{fam}:wt'])
            if has_mm and f'{fam}:wt' in preds:
                wt_add = sum_singles(sp, preds[f'{fam}:wt'])
                preds[f'{fam}:wt+'] = wt_add
                if f'{fam}:comb' in preds:
                    preds[f'{fam}:epi'] = preds[f'{fam}:comb'] - wt_add
        fr = Frame(name=ds, spine=sp, preds=preds, notes={'seeds': fam_seeds_by_ds[ds]})
        if ds == 'domainome':
            keep = ~sp['group'].astype(str).isin(excluded('domainome')).to_numpy()
            fr = Frame(
                fr.name,
                sp.loc[keep].reset_index(drop=True),
                {k: v[..., keep] for k, v in preds.items()},
                notes=dict(fr.notes, heldout=True),
            )
        frames[ds] = fr
    return frames


def columns_for(ds: str, fr):
    sp = fr.spine
    pp = ds in PER_PROTEIN
    nm = sp['n_mut'].to_numpy()
    use = sp['use'].to_numpy()
    cols = [(ds, fr, pp)]
    if (nm[use] > 1).any() and (nm[use] == 1).any():
        cols.append((f'{ds}|singles', subframe(fr, use & (nm == 1)), pp))
        cols.append((f'{ds}|multi', subframe(fr, use & (nm > 1)), pp))
    if 'dddG_ML' in sp and ds.startswith('tsuboyama'):
        m = epi_rows(fr) & sp['wt_backbone'].to_numpy()
        efr = subframe(fr, m, y=sp['dddG_ML'].to_numpy(dtype=float))
        cols.append((f'{ds}|dddG', efr, False))
    if ds == 'domainome' and not fr.notes.get('heldout'):
        hom = excluded('domainome')
        cols.append((f'{ds}|lowid', subframe(fr, use & ~sp['group'].astype(str).isin(hom).to_numpy()), True))
    return cols


def score_columns(frames: dict) -> pd.DataFrame:
    rows = []
    for ds, fr in frames.items():
        if fr is None:
            continue
        for col, f, pp in columns_for(ds, fr):
            keys = [k for k in f.preds if (k.endswith(':epi') == col.endswith('|dddG'))]
            if not keys:
                continue
            b = DatasetBoot(f, keys, per_protein=pp)
            for k in b.keys:
                fam, head = k.rsplit(':', 1)
                _, per_seed, n, ng = b.point(k)
                for s, r in zip(fr.notes['seeds'][fam], per_seed):
                    rec = dict(
                        family=fam, seed=s, head=('epi' if head == 'epi' else head),
                        column=col, dataset=ds, rho=float(r), n=int(n), n_proteins=ng,
                    )
                    rows.append(rec)
                    if ds == 'domainome' and col == 'domainome' and fr.notes.get('heldout'):
                        rows.append(dict(rec, column='domainome|lowid'))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# ProteinGym per-family × seed × readout ('wt', 'comb', AND true 'deployed') + joint [assay × seed] boot
# ----------------------------------------------------------------------------------------------
def load_pg_ablation_data(refresh: bool = False, procs: int = 8) -> tuple[pd.Series, dict[tuple[str, str], pd.DataFrame]]:
    """Load per-seed Spearman rhos on the canonical 173 held-out ProteinGym assays for every family & readout.

    A family receives a ``'deployed'`` (`primary`) matrix ONLY when it has at least one complete ``final_pgym_comb``
    seed (so ``comb`` is genuinely evaluated on the 47 multi-mutant held-out assays and ``wt`` on the 126 single-mutant
    held-out assays).
    """
    cache_pkl = CACHE / 'pgym_ablation_matrices_v1.pkl'
    ref = PG.pgym_reference(heldout=True, drop_large=True, include_fallback_small=False)
    cat = ref['category']
    dms_ids = list(ref.index)
    if cache_pkl.exists() and not refresh:
        with open(cache_pkl, 'rb') as fh:
            cached = pickle.load(fh)
        if isinstance(cached, dict) and cached.get('dms_ids') == dms_ids:
            return cat, cached['matrices']

    min_complete = len(dms_ids)
    has_mm_mask = np.array([PG.assay_has_multi(d)[0] for d in dms_ids], dtype=bool)
    tasks = [(fam, 'wt') for fam in FAMILIES] + [(fam, 'comb') for fam in FAMILIES if '@' not in fam]

    def _load_one(item: tuple[str, str]):
        fam, mode = item
        return item, PG.load_final_pgym_seeds(fam, mode=mode, dms_ids=dms_ids, seeds=SEEDS)

    with ThreadPoolExecutor(max_workers=min(max(procs, 1), 8)) as ex:
        raw_loaded = dict(ex.map(_load_one, tasks))

    matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for fam in FAMILIES:
        wt_df = raw_loaded[(fam, 'wt')]
        wt_good = [s for s in wt_df.columns if int(wt_df[s].notna().sum()) >= min_complete]
        if wt_good:
            matrices[(fam, 'wt')] = wt_df[wt_good].rename(columns={s: int(s) for s in wt_good})

        if '@' in fam:
            continue
        comb_df = raw_loaded[(fam, 'comb')]
        comb_good = [s for s in comb_df.columns if int(comb_df[s].notna().sum()) >= min_complete]
        if comb_good:
            matrices[(fam, 'comb')] = comb_df[comb_good].rename(columns={s: int(s) for s in comb_good})
            # True deployed readout on seeds that have BOTH wt and comb complete (assembled without duplicate disk reads)
            dep_seeds = [s for s in comb_good if s in wt_good]
            if dep_seeds:
                dep_df = wt_df[dep_seeds].copy()
                c_sub = comb_df[dep_seeds]
                use_comb = has_mm_mask & c_sub.notna().any(axis=1).to_numpy()
                dep_df.loc[use_comb, dep_seeds] = c_sub.loc[use_comb, dep_seeds]
                matrices[(fam, 'deployed')] = dep_df.rename(columns={s: int(s) for s in dep_seeds})
    atomic_pickle_dump({'dms_ids': dms_ids, 'matrices': matrices}, cache_pkl)
    return cat, matrices


def pg_cells_from_matrices(cat: pd.Series, matrices: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (fam, mode), df in matrices.items():
        head_out = 'primary' if mode == 'deployed' else mode
        for s in df.columns:
            r = df[s]
            means = {c: float(np.nanmean(r[cat == c])) for c in PG_CATS}
            vals = {f'pg|{c}': v for c, v in means.items()}
            vals['pg|avg'] = float(np.mean(list(means.values())))
            vals['pg|Stability_lowid'] = means['Stability']
            vals['pg|avg_lowid'] = vals['pg|avg']
            for col, v in vals.items():
                rows.append(dict(
                    family=fam, seed=int(s), head=head_out, column=col,
                    dataset='proteingym', rho=v, n=int(r.notna().sum()), n_proteins=np.nan,
                ))
    return pd.DataFrame(rows)


def pg_paired_boot(cat: pd.Series, matrices: dict[tuple[str, str], pd.DataFrame], B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint ``[category-stratified assay × triplicate-seed]`` paired bootstrap vs ``FINAL_p0``.

    Returns ``(delta_98_rows, detailed_pg_ablation_rows)``.
    """
    rows_98 = []
    rows_detail = []
    cols = (*PG_CATS, 'PG avg', 'DMS mean')

    for mode in ('wt', 'comb', 'deployed'):
        if (REF, mode) not in matrices:
            continue
        fam_models = {fam: df for (fam, m), df in matrices.items() if m == mode}
        if len(fam_models) < 2:
            continue
        summ_df, reps_map = PG.boot_pgym_categories(fam_models, cat, B=B, seed=16, columns=cols)
        pts = summ_df.set_index(['model', 'column'])['value']
        head_out = 'primary' if mode == 'deployed' else mode

        for fam in fam_models:
            if fam == REF:
                continue
            # Headline pg|avg row for 09_ablation/delta_paired.csv
            t_avg = ST.paired_boot_test(reps_map[fam]['PG avg'], reps_map[REF]['PG avg'], point_a=pts.loc[(fam, 'PG avg')], point_b=pts.loc[(REF, 'PG avg')])
            rows_98.append(dict(
                family=fam, head=head_out, column='pg|avg',
                delta_boot_mean=float(reps_map[fam]['PG avg'].mean() - reps_map[REF]['PG avg'].mean()),
                delta=t_avg['delta'], lo=t_avg['lo'], hi=t_avg['hi'], p=t_avg['p'], B=t_avg['n_boot'],
            ))
            for col in cols:
                col_id = 'pg|avg' if col == 'PG avg' else ('pg|dms_mean' if col == 'DMS mean' else f'pg|{col}')
                tst = ST.paired_boot_test(reps_map[fam][col], reps_map[REF][col], point_a=pts.loc[(fam, col)], point_b=pts.loc[(REF, col)])
                rows_detail.append(dict(
                    family=fam, head=head_out, category=col, column=col_id,
                    n_assays=int((cat == col).sum()) if col in PG_CATS else len(cat),
                    n_seeds=fam_models[fam].shape[1],
                    rho_p0=float(pts.loc[(REF, col)]),
                    rho_family=float(pts.loc[(fam, col)]),
                    delta=tst['delta'], lo=tst['lo'], hi=tst['hi'], se=tst['se'], p_boot=tst['p'], B=tst['n_boot'],
                ))
    df_detail = pd.DataFrame(rows_detail)
    if len(df_detail):
        df_detail['p_holm'] = np.nan
        for h, g in df_detail.groupby('head'):
            df_detail.loc[g.index, 'p_holm'] = ST.holm(g['p_boot'])
    return pd.DataFrame(rows_98), df_detail


# ----------------------------------------------------------------------------------------------
# Non-ProteinGym primary readout, composites, epistasis hierarchy & paired bootstraps
# ----------------------------------------------------------------------------------------------
def add_primary_non_pg(cells: pd.DataFrame) -> pd.DataFrame:
    non_pg = cells[cells['dataset'] != 'proteingym']
    p = non_pg[
        ((non_pg['head'] == 'comb') & non_pg['column'].map(is_multi_col))
        | ((non_pg['head'] == 'wt') & ~non_pg['column'].map(is_multi_col) & ~non_pg['column'].str.endswith('|dddG'))
        | (non_pg['head'] == 'epi')
    ].copy()
    p['head'] = 'primary'
    return pd.concat([cells, p], ignore_index=True)


COMPOSITES = {
    'ext_singles_mean': EXT_SINGLE,
    'ext_multi_mean': EXT_MULTI,
    'dms_mean': DMS3,
    'dms_multi_mean': [f'{d}|multi' for d in DMS_MULTI],
}


def add_composites(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, cols in COMPOSITES.items():
        sub = cells[cells['column'].isin(cols)]
        g = sub.groupby(['family', 'seed', 'head'])
        agg = g['rho'].agg(['mean', 'count']).reset_index()
        agg = agg[agg['count'] == len(cols)]
        for r in agg.itertuples():
            out.append(dict(family=r.family, seed=r.seed, head=r.head, column=name, dataset='composite', rho=r.mean, n=np.nan, n_proteins=np.nan))
    return pd.concat([cells, pd.DataFrame(out)], ignore_index=True)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby(['family', 'head', 'column'])['rho']
    s = g.agg(mean='mean', sd=lambda x: ST.seed_sd(x), n_seeds='count').reset_index()
    seeds = cells.groupby(['family', 'head', 'column'])['seed'].apply(lambda x: ','.join(map(str, sorted(x)))).rename('seeds').reset_index()
    return s.merge(seeds, on=['family', 'head', 'column'])


def build_ablation_epi_tables(fr) -> tuple[dict[tuple[str, int], object], list[tuple[str, str, str]]]:
    sp = fr.spine
    tables = {}
    seen_fams = []
    t_idx = EPI.make_epi_table(sp, np.arange(len(sp), dtype=float), np.arange(len(sp), dtype=float), heads=('add',))
    t_idx.matrices()
    i_sp = t_idx.L['add'].astype(int)
    ia_sp = t_idx.LA['add'].astype(int)
    ib_sp = t_idx.LB['add'].astype(int)
    for fam, seeds in fr.notes['seeds'].items():
        if f'{fam}:wt' not in fr.preds or f'{fam}:comb' not in fr.preds:
            continue
        seen_fams.append(fam)
        for j, sd in enumerate(seeds):
            r = copy.copy(t_idx)
            w = np.asarray(fr.preds[f'{fam}:wt'][j], dtype=float)
            c = np.asarray(fr.preds[f'{fam}:comb'][j], dtype=float)
            r._explicit_epi = False
            r.heads = ['epi', 'add', 'mt', 'comb']
            r.L = {'add': w[i_sp], 'comb': c[i_sp], 'mt': c[i_sp], 'epi': c[i_sp]}
            r.LA = {'add': w[ia_sp], 'comb': c[ia_sp], 'mt': c[ia_sp], 'epi': c[ia_sp]}
            r.LB = {'add': w[ib_sp], 'comb': c[ib_sp], 'mt': c[ib_sp], 'epi': c[ib_sp]}
            r.P, r.PA, r.PB = dict(r.L), dict(r.LA), dict(r.LB)
            r.PX = {h: r.LA[h] + r.LB[h] for h in r.heads}
            tables[(fam, sd)] = r
    # Ensure REF is first
    ordered_fams = ([REF] if REF in seen_fams else []) + [f for f in seen_fams if f != REF]
    models_spec = [(f, f, 'epi') for f in ordered_fams]
    return tables, models_spec


def _unit_ranks(v: np.ndarray) -> np.ndarray | None:
    r = rankdata(v)
    r -= r.mean()
    nrm = float(np.linalg.norm(r))
    return r / nrm if nrm > 0.0 else None


def _smooth_eval_at(x_sample: np.ndarray, v_sample: np.ndarray, x_eval: np.ndarray, nbins: int = E.SMOOTH_BINS) -> np.ndarray:
    nb = nbins if len(x_sample) >= 2 * nbins else max(1, len(x_sample) // 2)
    if len(x_sample) < 2:
        return np.full(len(x_eval), np.nan)
    qs = np.quantile(x_sample, np.linspace(0.0, 1.0, nb + 1))
    idx = np.clip(np.searchsorted(qs, x_sample, side='right') - 1, 0, nb - 1)
    cx = np.array([x_sample[idx == k].mean() for k in range(nb) if (idx == k).any()])
    cv = np.array([v_sample[idx == k].mean() for k in range(nb) if (idx == k).any()])
    o = np.argsort(cx)
    cx, cv = cx[o], cv[o]
    out = np.interp(x_eval, cx, cv)
    if len(cx) >= 2:
        lo, hi = x_eval < cx[0], x_eval > cx[-1]
        if lo.any():
            out[lo] = cv[0] + (x_eval[lo] - cx[0]) * (cv[1] - cv[0]) / max(cx[1] - cx[0], 1e-12)
        if hi.any():
            out[hi] = cv[-1] + (x_eval[hi] - cx[-1]) * (cv[-1] - cv[-2]) / max(cx[-1] - cx[-2], 1e-12)
    return out


_P11_EPI_STATE: dict = {}
_P11_BENCH_STATE: dict = {}


def _epi_worker_draw(task: tuple[np.ndarray, dict[int, np.ndarray]]) -> np.ndarray:
    cnt, sw_map = task
    st = _P11_EPI_STATE
    t0 = st['t0']
    by_lib = st['by_lib']
    blocks = st['blocks']
    blk_libs = st['blk_libs']
    naive_eng = st['naive_eng']
    prep = st['prep']
    m_spec = st['m_spec']
    fam_seeds = st['fam_seeds']

    idx = np.concatenate([by_lib[l] for l, c in enumerate(cnt) if c > 0 for _ in range(int(c))])
    w_row = cnt[t0.lib].astype(float)
    ry_naive = naive_eng.ry.ranks(w_row)
    rm_full = t0.dddG - _smooth_eval_at(t0.x[idx], t0.dddG[idx], t0.x)
    ry_beyond = _unit_ranks(rm_full[idx])
    active_blks = [(sub, int(cnt[bl])) for sub, bl in blocks if cnt[bl] > 0]
    rm_blk_units = [_unit_ranks(E._dc(rm_full[sub]).ravel()) for sub, _c in active_blks]
    w_blk = cnt[blk_libs].astype(float)

    out_row = np.full((len(m_spec), len(EPI_METRICS)), np.nan, dtype=float)
    for mi, (_d, fam, _h) in enumerate(m_spec):
        sub_s = fam_seeds[fam]
        sw = sw_map.get(len(sub_s), np.ones(len(sub_s), dtype=float))
        vals, wts = [], []
        for s, ws in zip(sub_s, sw):
            if ws <= 0.0:
                continue
            d_lat, xh, d_obs, xr_n, cr_b = prep[(fam, s)]
            n_rho = 0.0 if np.std(d_obs[idx]) < E.FLAT else float(naive_eng.rho(xr_n, w_row, ry_naive)[0])
            d_sub = d_lat[idx]
            if np.std(d_sub) < E.FLAT:
                rp_full = np.zeros_like(d_lat)
                b_rho = 0.0
            else:
                rp_full = d_lat - _smooth_eval_at(xh[idx], d_sub, xh)
                rp_sub = rp_full[idx]
                if np.std(rp_sub) < E.FLAT or ry_beyond is None:
                    b_rho = 0.0
                else:
                    rx = _unit_ranks(rp_sub)
                    b_rho = float(np.dot(rx, ry_beyond)) if rx is not None else 0.0
            cm_num, cm_den = 0.0, 0.0
            for (sub, c_b), u_rm in zip(active_blks, rm_blk_units):
                if u_rm is None:
                    continue
                dp = E._dc(rp_full[sub]).ravel()
                if np.std(dp) < E.FLAT:
                    rk_v = 0.0
                else:
                    u_rp = _unit_ranks(dp)
                    rk_v = float(np.dot(u_rp, u_rm)) if u_rp is not None else 0.0
                cm_num += c_b * rk_v
                cm_den += c_b
            c_mag = cm_num / cm_den if cm_den > 0.0 else np.nan
            ok_cr = np.isfinite(cr_b) & (w_blk > 0.0)
            c_rnk = float(np.dot(w_blk[ok_cr], cr_b[ok_cr]) / w_blk[ok_cr].sum()) if ok_cr.any() else np.nan
            vals.append([n_rho, b_rho, c_mag, c_rnk])
            wts.append(ws)
        if not vals:
            continue
        V = np.asarray(vals, dtype=float)
        W = np.asarray(wts, dtype=float)[:, None]
        fin = np.isfinite(V)
        with np.errstate(divide='ignore', invalid='ignore'):
            den = np.sum(np.where(fin, W, 0.0), axis=0)
            num = np.sum(np.where(fin, W * V, 0.0), axis=0)
            out_row[mi] = np.where(den > 0.0, num / den, np.nan)
    return out_row


def _fast_boot_ablation_epistasis(
    tabs_test: dict[tuple[str, int], object],
    m_spec: list[tuple[str, str, str]],
    pt_by_fam_metric: dict[tuple[str, str], float],
    B: int,
    seed: int,
    procs: int,
) -> np.ndarray:
    """Exact domain-cloning + actual-``ns`` seed bootstrap matching ``EPI.boot_epistasis_models`` without redundant block searches."""
    first_fam = m_spec[0][1]
    t0 = next(t for (k, _s), t in sorted(tabs_test.items()) if k == first_fam)
    libs = np.unique(t0.lib)
    n_libs = len(libs)
    by_lib = [np.flatnonzero(t0.lib == l) for l in libs]

    blocks = []
    for M_mat in t0.matrices():
        blk = E._complete_block(np.where(M_mat >= 0, 0.0, np.nan))
        if blk is not None:
            sub = M_mat[np.ix_(*blk)]
            blocks.append((sub, int(t0.lib[sub[0, 0]])))
    blk_libs = np.array([b[1] for b in blocks], dtype=int)

    y_blk_units = []
    for sub, _l in blocks:
        Y = t0.y[sub]
        y_blk_units.append((_unit_ranks(E._dc(E._colrank(Y)).ravel()), _unit_ranks(E._dc(E._colrank(Y.T)).ravel())))

    naive_eng = SpearmanEngine(t0.dddG, np.zeros(t0.n, dtype=np.int64), 1)
    prep = {}
    for (fam, sd), t in tabs_test.items():
        d_lat, xh = E.predicted_dddg(t, 'epi')
        d_obs = t.P['epi'] - (t.PA['add'] + t.PB['add'])
        cr_b = []
        for (sub, _l), (uy0, uy1) in zip(blocks, y_blk_units):
            L = t.L['epi'][sub]
            rr = []
            for B_m, uy in ((L, uy0), (L.T, uy1)):
                sp_m = E._dc(E._colrank(B_m)).ravel()
                if uy is None:
                    rr.append(np.nan)
                elif np.std(sp_m) < E.FLAT:
                    rr.append(0.0)
                else:
                    ul = _unit_ranks(sp_m)
                    rr.append(float(np.dot(ul, uy)) if ul is not None else 0.0)
            cr_b.append(E._mean(rr))
        prep[(fam, sd)] = (d_lat, xh, d_obs, naive_eng.prepare(d_obs), np.array(cr_b, dtype=float))

    fam_seeds = {fam: sorted([s for (k_, s) in tabs_test if k_ == fam]) for _d, fam, _h in m_spec}
    distinct_ns = sorted({len(v) for v in fam_seeds.values()})
    rng = np.random.default_rng(seed)
    ordered_ns = [3] + [ns for ns in distinct_ns if ns not in (0, 1, 3)]
    tasks = []
    for _ in range(B):
        cnt = rng.multinomial(n_libs, np.full(n_libs, 1.0 / n_libs))
        sw_map = {1: np.ones(1, dtype=float)}
        for ns in ordered_ns:
            sw_map[ns] = rng.multinomial(ns, np.full(ns, 1.0 / ns)).astype(float)
        tasks.append((cnt, sw_map))

    _P11_EPI_STATE.clear()
    _P11_EPI_STATE.update(
        t0=t0, by_lib=by_lib, blocks=blocks, blk_libs=blk_libs,
        naive_eng=naive_eng, prep=prep, m_spec=m_spec, fam_seeds=fam_seeds,
    )
    try:
        if procs > 1 and B > 1:
            with get_context('fork').Pool(procs) as pool:
                draws = pool.map(_epi_worker_draw, tasks, chunksize=max(1, B // (procs * 4)))
        else:
            draws = [_epi_worker_draw(tk) for tk in tasks]
    finally:
        _P11_EPI_STATE.clear()
    return np.asarray(draws, dtype=float)


def _epi_point_worker(item: tuple[str, str, int]) -> list[dict]:
    ds, fam, sd = item
    t = _P11_EPI_STATE['point_tabs'][(ds, fam, sd)]
    vals = EPI.compute_epi_metrics(t, head='epi', extended=False)
    n_r = int(t.n)
    n_p = int(len(np.unique(t.lib)))
    return [
        dict(family=fam, seed=sd, head='epi', column=f'{ds}|{m}', dataset=ds, rho=float(vals[m]), n=n_r, n_proteins=n_p)
        for m in EPI_METRICS
    ]


def epi_cells_and_boot(
    frames: dict,
    B: int,
    procs: int,
    reuse_cells: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabs_test, m_spec = build_ablation_epi_tables(frames['tsuboyama_capped_test'])
    if reuse_cells is not None:
        epi_cols = {f'tsuboyama_capped_test|{m}' for m in EPI_METRICS}
        cell_df = reuse_cells[
            (reuse_cells['head'] == 'epi') & reuse_cells['column'].str.endswith(tuple(f'|{m}' for m in EPI_METRICS))
        ].copy()
        test_cells = cell_df[cell_df['column'].isin(epi_cols)]
    else:
        point_tabs = {('tsuboyama_capped_test', fam, sd): t for (fam, sd), t in tabs_test.items()}
        if 'tsuboyama_capped_val' in frames and frames['tsuboyama_capped_val'] is not None:
            tabs_val, _ = build_ablation_epi_tables(frames['tsuboyama_capped_val'])
            for (fam, sd), t in tabs_val.items():
                point_tabs[('tsuboyama_capped_val', fam, sd)] = t
        point_keys = list(point_tabs.keys())
        _P11_EPI_STATE.clear()
        _P11_EPI_STATE['point_tabs'] = point_tabs
        try:
            if procs > 1 and len(point_keys) > 1:
                with get_context('fork').Pool(procs) as pool:
                    nested = pool.map(_epi_point_worker, point_keys)
            else:
                nested = [_epi_point_worker(pk) for pk in point_keys]
        finally:
            _P11_EPI_STATE.clear()
        cell_rows = [r for sub in nested for r in sub]
        cell_df = pd.DataFrame(cell_rows)
        test_cells = cell_df[cell_df['dataset'] == 'tsuboyama_capped_test']

    pt_lookup = {
        (fam, col.split('|', 1)[1]): float(grp['rho'].mean())
        for (fam, col), grp in test_cells.groupby(['family', 'column'])
    }
    reps_epi = _fast_boot_ablation_epistasis(tabs_test, m_spec, pt_lookup, B=B, seed=16, procs=procs)
    ref_idx = [i for i, (_d, k, _h) in enumerate(m_spec) if k == REF][0]
    boot_rows = []
    for mi, (_d, fam, _h) in enumerate(m_spec):
        if fam == REF:
            continue
        for ji, m in enumerate(EPI_METRICS):
            tst = ST.paired_boot_test(
                reps_epi[:, mi, ji], reps_epi[:, ref_idx, ji],
                point_a=float(pt_lookup[(fam, m)]), point_b=float(pt_lookup[(REF, m)]),
            )
            for h_name in ('epi', 'primary'):
                boot_rows.append(dict(
                    family=fam, head=h_name, column=f'tsuboyama_capped_test|{m}',
                    delta_boot_mean=float(np.nanmean(reps_epi[:, mi, ji] - reps_epi[:, ref_idx, ji])),
                    delta=tst['delta'], lo=tst['lo'], hi=tst['hi'], p=tst['p'], B=tst['n_boot'],
                ))
    return cell_df, pd.DataFrame(boot_rows)


def _bench_worker_batch(task_idx: tuple[int, int]) -> tuple[int, int, dict[str, np.ndarray]]:
    grp_idx, b_idx = task_idx
    grp = _P11_BENCH_STATE['groups'][grp_idx]
    boots = grp['boots']
    ncl = grp['ncl']
    prepped_mk = grp['prepped_mk']
    counts, sw_map = grp['tasks'][b_idx]
    counts_f64 = counts.astype(np.float64) if ncl else None

    out: dict[str, np.ndarray] = {}
    for ds, b in boots.items():
        res_ds = np.full(len(b.keys), np.nan, dtype=float)
        for mk, gidx_sub, key_specs in prepped_mk[ds]:
            idx, _g, _G, eng = b._engines[mk]
            w = counts_f64[gidx_sub] if counts_f64 is not None else np.zeros(len(idx), dtype=np.float64)
            sw_tot = float(w.sum())
            if sw_tot < 3.0:
                continue
            ry = eng.ry.ranks(w)
            my = (sw_tot + 1.0) * 0.5
            dy = ry - my
            w_dy = w * dy
            s_yy = float(np.dot(w_dy, dy))
            if s_yy <= 0.0 or not np.isfinite(s_yy):
                continue
            s_xx_notie = (sw_tot * (sw_tot * sw_tot - 1.0) - float(np.dot(w, w * w - 1.0))) / 12.0
            denom_notie = float(np.sqrt(s_xx_notie * s_yy)) if s_xx_notie > 0.0 else 0.0
            for k_pos, ns, seed_entries in key_specs:
                sw_s = sw_map.get(ns)
                rs, wts = [], []
                for si, order_i32, site_ord, xr in seed_entries:
                    wt_s = 1.0 if sw_s is None else float(sw_s[si])
                    if wt_s <= 0.0:
                        continue
                    if order_i32 is not None:
                        if denom_notie <= 0.0:
                            continue
                        ws = counts_f64[site_ord] if counts_f64 is not None else w[order_i32]
                        s_xy = float(np.dot(w_dy[order_i32], np.cumsum(ws) - 0.5 * ws))
                        r_val = s_xy / denom_notie
                    else:
                        rx = xr.ranks(w)
                        dx = rx - my
                        s_xx = float(np.dot(w * dx, dx))
                        if s_xx <= 0.0 or not np.isfinite(s_xx):
                            continue
                        r_val = float(np.dot(w_dy, dx) / np.sqrt(s_xx * s_yy))
                    if np.isfinite(r_val):
                        rs.append(r_val)
                        wts.append(wt_s)
                if rs:
                    res_ds[k_pos] = float(np.dot(wts, rs) / np.sum(wts))
        out[ds] = res_ds
    return grp_idx, b_idx, out


def _fast_bootstrap_g1_multi(
    g1_specs: list[tuple[dict, list[str], str]],
    B: int,
    seed: int = 16,
    procs: int = 8,
) -> list[tuple[dict, dict]]:
    """Exact parallel ``G == 1`` (`per_protein=False`) bootstrap across multiple groups in a single process pool."""
    if not g1_specs:
        return []
    prepared_groups = []
    point_per_grp = []
    flat_tasks: list[tuple[int, int]] = []

    for gi, (fs, keys, kind) in enumerate(g1_specs):
        rng = np.random.default_rng(seed)
        boots = {ds: DatasetBoot(fr, keys, per_protein=False) for ds, fr in fs.items()}
        point = {ds: {k: b.point(k) for k in b.keys} for ds, b in boots.items()}
        point_per_grp.append(point)
        names = {ds: b.cluster_names[kind] for ds, b in boots.items()}
        gidx, ncl = global_cluster_index(names)
        site_dtype = np.int16 if ncl < 32767 else np.int32
        prepped_mk = {}
        for ds, b in boots.items():
            mk_groups: dict[bytes, list[tuple[int, int, list]]] = {}
            for ki, k in enumerate(b.keys):
                mk_groups.setdefault(b.valid[k].tobytes(), []).append((ki, k))
            mk_list = []
            for mk, k_items in mk_groups.items():
                idx = b._engines[mk][0]
                gidx_sub = gidx[ds][idx].astype(site_dtype) if ncl else np.zeros(len(idx), dtype=site_dtype)
                key_specs = []
                for ki, k in k_items:
                    xrs = b._xr[k]
                    seed_entries = []
                    for si, xr in enumerate(xrs):
                        if len(xr.tie_start) == len(xr.order):
                            ord_i32 = xr.order.astype(np.int32)
                            s_ord = gidx_sub[ord_i32] if ncl else None
                            seed_entries.append((si, ord_i32, s_ord, None))
                        else:
                            seed_entries.append((si, None, None, xr))
                    key_specs.append((ki, len(xrs), seed_entries))
                mk_list.append((mk, gidx_sub, key_specs))
            prepped_mk[ds] = mk_list

        tasks = []
        for bi in range(B):
            counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl)) if ncl else np.zeros(0, dtype=np.int64)
            sampler = _make_seed_sampler(rng)
            sw_map = {1: np.ones(1, dtype=float), 3: sampler(3)}
            for ds, b in boots.items():
                for k in b.keys:
                    ns = len(b._xr[k])
                    if ns > 1 and ns not in sw_map:
                        sw_map[ns] = sampler(ns)
            tasks.append((counts, sw_map))
            flat_tasks.append((gi, bi))

        prepared_groups.append(dict(boots=boots, ncl=ncl, prepped_mk=prepped_mk, tasks=tasks))

    _P11_BENCH_STATE.clear()
    _P11_BENCH_STATE['groups'] = prepared_groups
    try:
        if procs > 1 and len(flat_tasks) > 1:
            with get_context('fork').Pool(procs) as pool:
                raw_out = pool.map(_bench_worker_batch, flat_tasks, chunksize=max(1, len(flat_tasks) // (procs * 8)))
        else:
            raw_out = [_bench_worker_batch(tk) for tk in flat_tasks]
    finally:
        _P11_BENCH_STATE.clear()

    draws_by_grp: list[list[dict[str, np.ndarray] | None]] = [[None] * B for _ in g1_specs]
    for gi, bi, d_out in raw_out:
        draws_by_grp[gi][bi] = d_out

    results: list[tuple[dict, dict]] = []
    for gi, grp in enumerate(prepared_groups):
        boots = grp['boots']
        draws = draws_by_grp[gi]
        reps = {}
        for ds, b in boots.items():
            mat = np.asarray([d[ds] for d in draws], dtype=float)
            reps[ds] = {k: mat[:, ki].copy() for ki, k in enumerate(b.keys)}
        results.append((point_per_grp[gi], reps))
    return results


def boot_benchmark_deltas(frames: dict, B: int, procs: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap of ``(family - FINAL_p0)`` AND ``(comb - wt)`` with joint ``[cluster/site × 3-seed]`` draws."""
    tsu = frames['tsuboyama_capped_test']
    tsu_val = frames['tsuboyama_capped_val']
    dom = frames['domainome']

    groups = [
        ('external', 'protein', {d: frames[d] for d in EXT12 if d in frames}),
        ('dms', 'site', {d: frames[d] for d in DMS3 if d in frames}),
        ('tsu_test', 'protein', {'tsuboyama_capped_test': tsu}),
        ('tsu_val', 'protein', {'tsuboyama_capped_val': tsu_val}),
        ('domainome', 'protein', {'domainome': dom}),
    ]
    for ds_base, fr_base, tag in (('tsuboyama_capped_test', tsu, 'tsu_test'), ('tsuboyama_capped_val', tsu_val, 'tsu_val')):
        nm = fr_base.spine['n_mut'].to_numpy()
        use = fr_base.spine['use'].to_numpy()
        for sub_nm, mask in (('singles', use & (nm == 1)), ('multi', use & (nm > 1))):
            sf = subframe(fr_base, mask)
            sf.notes = fr_base.notes
            groups.append((f'{tag}_{sub_nm}', 'protein', {f'{ds_base}|{sub_nm}': sf}))
        m_epi = epi_rows(fr_base) & fr_base.spine['wt_backbone'].to_numpy()
        efr = subframe(fr_base, m_epi, y=fr_base.spine['dddG_ML'].to_numpy(dtype=float), keys=[k for k in fr_base.preds if k.endswith(':epi')])
        efr.notes = fr_base.notes
        groups.append((f'{tag}_dddG', 'protein', {f'{ds_base}|dddG': efr}))

    for d_dms in DMS3:
        fr_d = frames[d_dms]
        nm = fr_d.spine['n_mut'].to_numpy()
        use = fr_d.spine['use'].to_numpy()
        for sub_nm, mask in (('singles', use & (nm == 1)), ('multi', use & (nm > 1))):
            if mask.sum() >= 10:
                sf = subframe(fr_d, mask)
                sf.notes = fr_d.notes
                groups.append((f'{d_dms}_{sub_nm}', 'site', {f'{d_dms}|{sub_nm}': sf}))

    # Note: frames['domainome'] is already heldout=True (436 held-out domains), so domainome|lowid is an exact alias!
    pts_all: dict[str, dict[str, float]] = {}
    reps_all: dict[str, dict[str, np.ndarray]] = {}
    meta_all: dict[str, tuple[str, int, int]] = {}
    pp_specs: list[tuple[str, str, object, list[str]]] = []
    g1_specs: list[tuple[dict, list[str], str]] = []
    g1_kinds: list[str] = []

    for name, kind, fs in groups:
        is_dddg = all(c.endswith('|dddG') for c in fs)
        allowed_heads = ('epi',) if is_dddg else ('wt', 'comb', 'mt', 'wt+')
        keys = sorted({k for f in fs.values() for k in f.preds if k.rsplit(':', 1)[1] in allowed_heads})
        if not keys:
            continue
        pp_map = {c: (c.split('|')[0] in PER_PROTEIN and not c.endswith('|dddG')) for c in fs}
        per_p = None if name in ('external', 'dms') else next(iter(pp_map.values()))
        # Fast exact path for single-dataset per_protein=True groups (Domainome 451k rows & Tsuboyama):
        # within-protein Spearman rho per seed is invariant to protein resampling, so compute (n_domains, n_seeds) once!
        if per_p is True and kind == 'protein' and len(fs) == 1:
            col_only, f_only = next(iter(fs.items()))
            pp_specs.append((col_only, kind, f_only, keys))
            continue

        g1_specs.append((fs, keys, kind))
        g1_kinds.append(kind)

    def _run_pp_one(item: tuple[str, str, object, list[str]]):
        col_only, kind, f_only, keys = item
        db = DatasetBoot(f_only, keys, per_protein=True)
        if not db.keys:
            return None
        any_k = db.keys[0]
        _, _, n_r, n_g = db.point(any_k)
        all_groups = np.unique(f_only.spine['group'].to_numpy()[db.base].astype(str))
        grp_pos = {g_nm: gi for gi, g_nm in enumerate(all_groups)}
        n_g_all = len(all_groups)
        distinct_ns = tuple(sorted({len(db._xr[k]) for k in db.keys if len(db._xr[k]) > 1})) or (2, 3)
        dom_c, sw_c = ST.make_domain_seed_draws(n_g_all, B=B, seed=16, seed_counts=distinct_ns)
        pts_c, reps_c = {}, {}
        for k in db.keys:
            pt_exact = float(db.point(k)[0])
            nm_k, r_seeds = db.per_protein_rho_seeds(k)
            mat_sd = np.full((n_g_all, r_seeds.shape[0]), np.nan, dtype=float)
            idx_g = [grp_pos[str(x)] for x in nm_k]
            mat_sd[idx_g, :] = r_seeds.T
            _, rp_v = ST.boot_domain_seeds(mat_sd, B=B, counts=dom_c, seed_weights=sw_c)
            pts_c[k] = pt_exact
            reps_c[k] = rp_v
        return col_only, pts_c, reps_c, (kind, int(n_r), int(n_g))

    if pp_specs:
        with ThreadPoolExecutor(max_workers=min(max(procs, 1), len(pp_specs))) as ex:
            for res_pp in ex.map(_run_pp_one, pp_specs):
                if res_pp is not None:
                    col_only, pts_c, reps_c, meta_c = res_pp
                    pts_all[col_only] = pts_c
                    reps_all[col_only] = reps_c
                    meta_all[col_only] = meta_c

    for kind, (point, reps) in zip(g1_kinds, _fast_bootstrap_g1_multi(g1_specs, B=B, seed=16, procs=procs)):
        for c in reps:
            reps_all[c] = dict(reps[c])
            pts_all[c] = {k: float(point[c][k][0]) for k in point[c]}
            any_k = next(iter(point[c]))
            meta_all[c] = (kind, int(point[c][any_k][2]), int(point[c][any_k][3]))

    if 'domainome' in reps_all:
        reps_all['domainome|lowid'] = dict(reps_all['domainome'])
        pts_all['domainome|lowid'] = dict(pts_all['domainome'])
        meta_all['domainome|lowid'] = meta_all['domainome']

    # Add primary readout per column
    for c, d in list(reps_all.items()):
        h = 'epi' if c.endswith('|dddG') else ('comb' if is_multi_col(c) else 'wt')
        for k in list(d):
            fam, head = k.rsplit(':', 1)
            if head == h:
                d[f'{fam}:primary'] = d[k]
                pts_all[c][f'{fam}:primary'] = pts_all[c][k]

    # Composites across shared draws
    comp_defs = {
        'ext_singles_mean': EXT_SINGLE,
        'ext_multi_mean': EXT_MULTI,
        'dms_mean': DMS3,
        'dms_multi_mean': [f'{d}|multi' for d in DMS_MULTI if f'{d}|multi' in reps_all],
    }
    for nm, cols in comp_defs.items():
        ks = set.intersection(*[set(reps_all[c]) for c in cols if c in reps_all])
        reps_all[nm] = {k: np.nanmean([reps_all[c][k] for c in cols], axis=0) for k in ks}
        pts_all[nm] = {k: float(np.nanmean([pts_all[c][k] for c in cols])) for k in ks}
        meta_all[nm] = ('composite', 0, len(cols))

    # 1) Ablation vs FINAL_p0 rows
    rows_vs_p0 = []
    for col, d in reps_all.items():
        unit_kind, n_rows, n_units = meta_all[col]
        for k, r in d.items():
            fam, head = k.rsplit(':', 1)
            if head not in ('wt', 'comb', 'epi', 'primary') or fam == REF:
                continue
            ref_r = d.get(f'{REF}:{head}')
            if ref_r is None:
                continue
            pt_f = pts_all[col][k]
            pt_r = pts_all[col][f'{REF}:{head}']
            tst = ST.paired_boot_test(r, ref_r, point_a=pt_f, point_b=pt_r)
            rows_vs_p0.append(dict(
                family=fam, head=head, column=col, unit=unit_kind, n_rows=n_rows, n_units=n_units,
                rho_p0=pt_r, rho_family=pt_f,
                delta_boot_mean=float(np.nanmean(r - ref_r)),
                delta=tst['delta'], lo=tst['lo'], hi=tst['hi'], se=tst['se'], p=tst['p'], B=tst['n_boot'],
            ))

    # 2) Combined minus WT rows across all families
    rows_cvw = []
    for col, d in reps_all.items():
        if col.endswith('|dddG'):
            continue
        unit_kind, n_rows, n_units = meta_all[col]
        fams_here = sorted({k.rsplit(':', 1)[0] for k in d if ':' in k})
        for fam in fams_here:
            if f'{fam}:comb' not in d or f'{fam}:wt' not in d:
                continue
            pt_c = pts_all[col][f'{fam}:comb']
            pt_w = pts_all[col][f'{fam}:wt']
            pt_m = pts_all[col].get(f'{fam}:mt', np.nan)
            pt_wplus = pts_all[col].get(f'{fam}:wt+', np.nan)
            tst = ST.paired_boot_test(d[f'{fam}:comb'], d[f'{fam}:wt'], point_a=pt_c, point_b=pt_w)
            rows_cvw.append(dict(
                family=fam, column=col,
                dataset=col.split('|')[0],
                cell=(col.split('|', 1)[1] if '|' in col else 'all'),
                unit=unit_kind, n_rows=n_rows, n_units=n_units,
                rho_wt=pt_w, rho_mt=pt_m, rho_comb=pt_c, rho_wt_plus=pt_wplus,
                d_mt=(pt_m - pt_w) if np.isfinite(pt_m) else np.nan,
                delta=tst['delta'], lo=tst['lo'], hi=tst['hi'], se=tst['se'], p_boot=tst['p'], B=tst['n_boot'],
            ))
    return pd.DataFrame(rows_vs_p0), pd.DataFrame(rows_cvw)


# ----------------------------------------------------------------------------------------------
# 09_ablation markdown / LaTeX tables
# ----------------------------------------------------------------------------------------------
def pooled_seed_sd(summ: pd.DataFrame, col: str, head: str = 'primary') -> float:
    d = summ[
        (summ['head'] == head) & (summ['column'] == col) & (summ['n_seeds'] >= 2)
        & ~summ['family'].isin([REF, 'FINAL_p0_mega']) & ~summ['family'].str.contains('@')
    ]
    return float(np.sqrt(np.nanmean(d['sd'] ** 2))) if len(d) else np.nan


def render_delta_tables(
    out_98: Path,
    summ: pd.DataFrame,
    deltas: pd.DataFrame,
    B: int,
    n_pg_held: int = 173,
    n_pg_mm: int = 47,
    z: float = 2.0,
) -> None:
    noise_note = (
        f'Bold: |Δ| > {z:g} SE, where SE is the paired bootstrap over proteins / sites / assays with triplicate seeds '
        f'resampled jointly (B = {B:,}; pooled column seed SD fallback where no bootstrap exists). '
    )
    epi_note = (
        'Epistasis columns: comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG on the 12,455 Homology-Restricted MegaScale test '
        'doubles with both singles measured (beyond ρ = Spearman above global saturation curve; cell mag. = double-centred '
        'complete-block residuals across the 13 test domains). '
    )
    specs = {
        'main': dict(
            head='primary',
            exclude={'FINAL_p0_mega'},
            cols=[
                'tsuboyama_capped_test', 'tsuboyama_capped_test|multi',
                'tsuboyama_capped_test|beyond_rho', 'tsuboyama_capped_test|cell_mag',
                'ext_singles_mean', 'ext_multi_mean', 'domainome|lowid', 'pg|avg',
            ],
            note=(
                'ESM-MSR row: Spearman ρ (seed mean; seeds listed). Other rows: Δρ = family seed mean − ESM-MSR under the '
                'deployed readout (WT head on singles-only sets, ½(MT + WT) where multi-mutants are scored, including the '
                f'{n_pg_mm} multi-mutant held-out ProteinGym assays of {n_pg_held}; — where a multi-mutant combined run was not executed). '
                + noise_note + epi_note
            ),
        ),
        'wt': dict(
            head='wt',
            exclude={'FINAL_p0_mega'} | MT_ONLY,
            cols=[
                'tsuboyama_capped_test', 'tsuboyama_capped_test|singles', 'tsuboyama_capped_test|multi',
                'ext_singles_mean', 'ext_multi_mean', 'domainome|lowid', 'pg|avg',
            ],
            note=(
                'WT head alone on every row (wt_lora_pred). ESM-MSR row: Spearman ρ (seed mean). Other rows: Δρ = family '
                'seed mean − ESM-MSR under the WT head (joint [cluster/assay × 3-seed] bootstrap). Left out: MT-only '
                'ablations (no pair loss, MT LoRA rank 2, p1 MT marginal) and epistasis columns. ' + noise_note
            ),
        ),
    }

    for name, spec in specs.items():
        head, cols = spec['head'], spec['cols']
        csv_name = f'table_{name}_delta.csv'
        S = summ[(summ['head'] == head) | ((summ['head'] == 'primary') & summ['column'].str.endswith(('|beyond_rho', '|cell_mag')))]
        S = S[(S['head'] == head) | ~S.duplicated(['family', 'column'], keep=False)].set_index(['family', 'column'])
        D = deltas[deltas['head'] == head].set_index(['family', 'column']) if len(deltas) else None
        if D is not None and head != 'primary':
            D = pd.concat([D, deltas[deltas['head'] == 'primary'].set_index(['family', 'column'])]).loc[lambda x: ~x.index.duplicated()]

        rows = []
        for fam in FAMILIES:
            if fam in spec['exclude']:
                continue
            for col in cols:
                if (fam, col) not in S.index or (REF, col) not in S.index:
                    continue
                f, r = S.loc[(fam, col)], S.loc[(REF, col)]
                pooled = pooled_seed_sd(summ, col, head)
                sd_f = f['sd'] if f['n_seeds'] > 1 and np.isfinite(f['sd']) else pooled
                sd_r = r['sd'] if r['n_seeds'] > 1 and np.isfinite(r['sd']) else pooled
                se_seed = float(np.sqrt(sd_f ** 2 / f['n_seeds'] + sd_r ** 2 / r['n_seeds']))
                se_boot = np.nan
                if D is not None and (fam, col) in D.index:
                    b = D.loc[(fam, col)]
                    se_boot = float((b['hi'] - b['lo']) / 3.92)
                se = float(se_boot if np.isfinite(se_boot) else se_seed)
                delta = float(f['mean'] - r['mean']) if fam != REF else np.nan
                rows.append(dict(
                    family=fam, column=col, head=head, value=float(f['mean']),
                    n_seeds=int(f['n_seeds']), seeds=f['seeds'], delta=delta,
                    se_seed=se_seed, se_boot=se_boot, se=se,
                    flag=bool(fam != REF and np.isfinite(se) and se > 0 and abs(delta) > z * se),
                ))
        T = pd.DataFrame(rows)
        atomic_to_csv(T, out_98 / csv_name, index=False, float_format='%.5f')

        def _cell(fam, col, tex=False):
            r_sub = T[(T['family'] == fam) & (T['column'] == col)] if len(T) else T
            if r_sub.empty:
                return '—' if not tex else '--'
            r = r_sub.iloc[0]
            few = r['n_seeds'] < T.loc[T['family'] == fam, 'n_seeds'].max()
            mark = ('‡' if not tex else '$^\\ddagger$') if few else ''
            if fam == REF:
                return f"{r['value']:.3f}{mark}"
            s = '0.000' if round(r['delta'], 3) == 0 else f"{r['delta']:+.3f}".replace('-', '−' if not tex else '$-$')
            if r['flag']:
                s = f'**{s}**' if not tex else f'\\textbf{{{s}}}'
            return s + mark

        def _seeds(fam):
            x = T.loc[T['family'] == fam, 'seeds'].dropna() if len(T) else pd.Series(dtype=str)
            return x.iloc[0] if len(x) else '—'

        hdr = ['Ablation', 'Change from ESM-MSR', 'Seeds'] + [COL_LABEL[c] for c in cols]
        md = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        tex = ['\\begin{tabular}{ll' + 'r' * len(cols) + '}', '\\toprule', 'Ablation & Change & ' + ' & '.join(COL_LABEL[c] for c in cols) + ' \\\\', '\\midrule']
        md.append(f"| **{FAMILIES[REF]['short']}** | {FAMILIES[REF]['change']} | {_seeds(REF)} | " + ' | '.join(_cell(REF, c) for c in cols) + ' |')
        tex.append('ESM-MSR (reference) & -- & ' + ' & '.join(_cell(REF, c, True) for c in cols) + ' \\\\')
        for grp, glabel in GROUP_LABEL.items():
            fams = [f for f, m in FAMILIES.items() if m['group'] == grp and f not in spec['exclude']]
            if not fams:
                continue
            md.append(f'| *{glabel}* |' + ' |' * (len(hdr) - 1))
            tex.append('\\midrule\n\\multicolumn{' + str(len(cols) + 2) + '}{l}{\\textit{' + glabel + '}} \\\\')
            for fam in fams:
                short = FAMILIES[fam]['short']
                change = re.sub(r'\s*\((?:--|independent|block_|train split|not homology|evaluated)[^)]*\)\s*$', '', FAMILIES[fam]['change'])
                md.append(f"| {short} | {change} | {_seeds(fam)} | " + ' | '.join(_cell(fam, c) for c in cols) + ' |')
                change_tex = change.replace('_', '\\_').replace('->', '\x00').replace('<', '$<$').replace('>', '$>$').replace('\x00', '$\\to$')
                tex.append(f"{short} & {change_tex} & " + ' & '.join(_cell(fam, c, True) for c in cols) + ' \\\\')
        tex += ['\\bottomrule', '\\end{tabular}']
        full_note = spec['note'] + f"‡: fewer seeds than the rest of the row (exact seeds in {csv_name}). —: not run."
        atomic_write_text(out_98 / f'table_{name}_delta.md', '\n'.join(md) + '\n\n' + full_note + '\n')
        atomic_write_text(out_98 / f'table_{name}_delta.tex', '\n'.join(tex) + '\n% ' + full_note + '\n')


def render_candidate_layouts(out_98: Path, summ: pd.DataFrame, deltas: pd.DataFrame, mega_own: pd.DataFrame | None, B: int) -> None:
    S = summ.set_index(['family', 'head', 'column'])
    D = deltas.set_index(['family', 'head', 'column']) if len(deltas) else None

    def _fmt(r, dl=None):
        if r is None or not np.isfinite(r['mean']):
            return '—'
        s = f"{r['mean']:.3f}"
        if r['n_seeds'] > 1 and np.isfinite(r['sd']):
            s += f" ± {r['sd']:.3f}"
        if dl is not None:
            mark = '**' if (dl['lo'] > 0 or dl['hi'] < 0) else ''
            s += f" ({mark}{dl['delta']:+.3f}{mark})"
        return s

    for name, spec in LAYOUTS.items():
        heads, cols = spec['heads'], spec['cols']
        hdr = ['Family', 'Change', 'Seeds'] + [f'{COLLABEL.get(c, c)}' + (f' [{h}]' if len(heads) > 1 else '') for c in cols for h in heads]
        lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        for fam, meta in FAMILIES.items():
            cells, seeds = [], set()
            for c in cols:
                for h in heads:
                    hh = 'epi' if (c.endswith('|dddG') and h in ('primary', 'comb')) else h
                    key = (fam, hh if c.endswith('|dddG') else h, c)
                    r = S.loc[key] if key in S.index else None
                    if fam == 'FINAL_p0_mega' and c.startswith('tsuboyama') and mega_own is not None:
                        mk = ('FINAL_p0_mega', key[1], c)
                        r_m = mega_own.loc[mk] if mk in mega_own.index else None
                        cells.append((_fmt(r_m) + '†') if r_m is not None else '—')
                        if r_m is not None:
                            seeds.add(str(r_m['seeds']))
                        continue
                    dl = D.loc[key] if (D is not None and key in D.index) else None
                    cells.append(_fmt(r, dl))
                    if r is not None:
                        seeds.add(str(r['seeds']))
            sd = '/'.join(sorted(seeds)) if seeds else '—'
            lines.append(f"| {meta['short']} | {meta['change']} | {sd} | " + ' | '.join(cells) + ' |')
        note = (
            f'Cells: seed mean ± seed SD (ddof=1; Spearman ρ). Parentheses: paired bootstrap Δ vs ESM-MSR across seeds 1–3 '
            f'(B = {B:,}; bold: 95% CI excludes 0). † FINAL_p0_mega on its own mega_splits test set.'
        )
        atomic_write_text(out_98 / f'layout_{name}.md', f'### {name}\n\n' + '\n'.join(lines) + f'\n\n{note}\n')


def write_informativeness(out_abl: Path, out_dms: Path, summ: pd.DataFrame) -> None:
    fams = [f for f in ABL_ARMS if f != 'FINAL_p0_mega']
    d = summ[summ['family'].isin(fams) & (summ['n_seeds'] >= 2)]
    rows = []
    for (col, head), g in d.groupby(['column', 'head']):
        if len(g) < 4:
            continue
        noise = float(np.sqrt(np.nanmean(g['sd'] ** 2)))
        spread = float(g['mean'].std(ddof=1))
        n = float(g['n_seeds'].mean())
        best = g.loc[g['mean'].idxmax(), 'family']
        rows.append(dict(
            column=col, head=head, n_families=len(g), spread=spread, noise=noise,
            snr=(spread / (noise / np.sqrt(n))) if noise > 0 else np.nan,
            best=FAMILIES[best]['short'], range=float(g['mean'].max() - g['mean'].min()),
        ))
    inf_df = pd.DataFrame(rows).sort_values(['head', 'snr'], ascending=[True, False])
    is_dms_inf = inf_df['column'].isin(DMS3_COLS)
    atomic_to_csv(inf_df[~is_dms_inf], out_abl / 'informativeness.csv', index=False, float_format='%.4f')
    atomic_to_csv(inf_df[is_dms_inf], out_dms / 'ablation_informativeness.csv', index=False, float_format='%.4f')

    hv = []
    for col, g in summ[summ['family'].isin(MAIN_11_FAMS) & summ['head'].isin(['wt', 'comb'])].groupby('column'):
        w = g[g['head'] == 'wt'].set_index('family')['mean']
        c = g[g['head'] == 'comb'].set_index('family')['mean']
        common = w.index.intersection(c.index)
        if len(common) < 4:
            continue
        ref_diff = float(c[REF] - w[REF]) if REF in common else np.nan
        hv.append(dict(
            column=col, n_families=len(common), comb_minus_wt=float((c[common] - w[common]).mean()),
            comb_minus_wt_ref=ref_diff, rank_agreement=float(spearmanr(w[common], c[common])[0]),
        ))
    hv_df = pd.DataFrame(hv)
    is_dms_hv = hv_df['column'].isin(DMS3_COLS)
    atomic_to_csv(hv_df[~is_dms_hv], out_abl / 'head_choice.csv', index=False, float_format='%.4f')
    atomic_to_csv(hv_df[is_dms_hv], out_dms / 'ablation_head_choice.csv', index=False, float_format='%.4f')


# ----------------------------------------------------------------------------------------------
# Dual-view vs WT-view study tables & mechanistic `when_rule` law
# ----------------------------------------------------------------------------------------------
def build_dual_view_study(
    out_dv: Path,
    frames: dict,
    cvw_df: pd.DataFrame,
    B: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-assay ProteinGym head scores, category p-values, unit-level ``when_rule``, and 11-family tables."""
    atomic_to_csv(cvw_df, out_dv / 'benchmark_delta.csv', index=False, float_format='%.4f')

    # All-11-families comb minus wt table
    key_cols = [
        ('ext_singles_mean', 'External singles (10)'),
        ('tsuboyama_capped_test|singles', 'MegaScale test singles'),
        ('tsuboyama_capped_test|multi', 'MegaScale test doubles'),
        ('ptmul', 'PTMUL'),
        ('ptmuld', 'PTMUL-D'),
        ('dms_multi_mean', 'DMS multi-mutants (3)'),
        ('domainome', 'Domainome held-out'),
    ]
    sub_all = cvw_df[cvw_df['family'].isin(MAIN_11_FAMS) & cvw_df['column'].isin([c for c, _ in key_cols])].copy()
    sub_all['family_label'] = sub_all['family'].map(lambda f: _abl_style(f).label)
    atomic_to_csv(sub_all, out_dv / 'combined_minus_wt_all_families.csv', index=False, float_format='%.4f')

    piv_rows = []
    bold_rows = []
    for fam in MAIN_11_FAMS:
        r_d = {'Family': _abl_style(fam).label}
        b_d = {'Family': fam == REF}
        for col_id, col_lbl in key_cols:
            hit = sub_all[(sub_all['family'] == fam) & (sub_all['column'] == col_id)]
            if len(hit):
                h = hit.iloc[0]
                r_d[col_lbl] = float(h['delta'])
                b_d[col_lbl] = bool(h['lo'] > 0 or h['hi'] < 0)
            else:
                r_d[col_lbl] = np.nan
                b_d[col_lbl] = False
        piv_rows.append(r_d)
        bold_rows.append(b_d)
    write_stat_table(pd.DataFrame(piv_rows), out_dv / 'table_all_families', bold_mask=pd.DataFrame(bold_rows), float_fmt='%+.4f')

    # ProteinGym per-assay WT, MT, Comb + r(WT, MT) across all 3 seeds of FINAL_p0 and FINAL_p1 (173 held-out + 40 hom)
    cvw_cache = CACHE / 'pgym_cvw_per_assay_v1.pkl'
    if cvw_cache.exists():
        with open(cvw_cache, 'rb') as fh:
            pg_df = pickle.load(fh)
    else:
        ref_213 = PG.pgym_reference(heldout=False, drop_large=True, include_fallback_small=False)

        def _score_cvw_assay(item):
            dms, r_meta = item
            rec = dict(assay=dms, category=r_meta['category'], train_overlap=bool(r_meta['homologous']))
            for fam, tag in (('FINAL_p0', 'p0'), ('FINAL_p1', 'p1')):
                s_vals = {k: [] for k in ('all_wt', 'all_mt', 'all_comb', 'single_wt', 'single_mt', 'single_comb', 'multi_wt', 'multi_mt', 'multi_comb', 'r_wt_mt')}
                n_tot, n_mm = 0, 0
                for sd in SEEDS:
                    f_p = FINAL_PGYM_COMB / fam / f's{sd}' / f'{dms}.csv'
                    if not f_p.exists():
                        continue
                    d = pd.read_csv(f_p, usecols=['mutant', 'DMS_score', 'wt_lora_pred', 'mt_lora_pred', 'combined_pred'])
                    y = d['DMS_score'].to_numpy(float)
                    mm = d['mutant'].astype(str).str.contains(':').to_numpy()
                    n_tot, n_mm = len(d), int(mm.sum())
                    split_ok = (n_mm >= 20) and ((n_tot - n_mm) >= 20)
                    for h_k, c_nm in (('wt', 'wt_lora_pred'), ('mt', 'mt_lora_pred'), ('comb', 'combined_pred')):
                        xv = d[c_nm].to_numpy(float)
                        val_all = float(spearmanr(xv, y, nan_policy='omit')[0])
                        rec[f'{tag}_s{sd}_all_{h_k}'] = val_all
                        s_vals[f'all_{h_k}'].append(val_all)
                        if split_ok:
                            vs = float(spearmanr(xv[~mm], y[~mm], nan_policy='omit')[0])
                            vm = float(spearmanr(xv[mm], y[mm], nan_policy='omit')[0])
                            rec[f'{tag}_s{sd}_single_{h_k}'] = vs
                            rec[f'{tag}_s{sd}_multi_{h_k}'] = vm
                            s_vals[f'single_{h_k}'].append(vs)
                            s_vals[f'multi_{h_k}'].append(vm)
                    rwm = float(spearmanr(d['wt_lora_pred'], d['mt_lora_pred'], nan_policy='omit')[0])
                    rec[f'{tag}_s{sd}_r_wt_mt'] = rwm
                    s_vals['r_wt_mt'].append(rwm)
                rec['n'] = n_tot
                rec['n_multi'] = n_mm
                rec[f'{tag}_n_seeds'] = len(s_vals['all_wt'])
                for k, vlist in s_vals.items():
                    rec[f'{tag}_{k}'] = float(np.nanmean(vlist)) if vlist else np.nan
            return rec

        with ThreadPoolExecutor(max_workers=8) as ex:
            pg_rows = list(ex.map(_score_cvw_assay, list(ref_213.iterrows())))
        pg_df = pd.DataFrame(pg_rows).set_index('assay')
        atomic_pickle_dump(pg_df, cvw_cache)
    atomic_to_csv(pg_df, out_dv / 'pg_per_assay.csv', float_format='%.4f')

    # Joint [assay × triplicate-seed] category p-values for comb - wt on held-out 173 assays
    pg_held = pg_df[~pg_df['train_overlap']].copy()
    rng_pg = np.random.default_rng(7)
    pv_rows = []
    for fam, tag in (('FINAL_p0', 'p0'), ('FINAL_p1', 'p1')):
        for cat_nm in list(PG.PG_CATEGORIES) + ['All held-out']:
            sub = pg_held if cat_nm == 'All held-out' else pg_held[pg_held['category'] == cat_nm]
            if sub.empty:
                continue
            w_mat = sub[[f'{tag}_s{sd}_all_wt' for sd in SEEDS]].to_numpy(float)
            c_mat = sub[[f'{tag}_s{sd}_all_comb' for sd in SEEDS]].to_numpy(float)
            d_pt = np.nanmean(c_mat - w_mat, axis=1)
            n_a = len(sub)
            picks = rng_pg.integers(0, n_a, size=(max(B, 5000), n_a))
            sw = rng_pg.multinomial(3, [1 / 3] * 3, size=max(B, 5000)).astype(float) / 3.0
            d_seed_draw = np.einsum('bs,is->bi', sw, c_mat - w_mat)
            m_draws = np.nanmean(np.take_along_axis(d_seed_draw, picks, axis=1), axis=1)
            lo, hi = ci(m_draws)
            p_b = ST.boot_pvalue(m_draws, null=0.0, alternative='two-sided')
            p_w = float(wilcoxon(d_pt).pvalue) if n_a >= 5 and np.any(d_pt != 0) else np.nan
            pv_rows.append(dict(
                family=fam, category=cat_nm, n_assays=n_a,
                mean_delta=float(np.nanmean(d_pt)), median_delta=float(np.nanmedian(d_pt)),
                lo=lo, hi=hi, share_improved=float((d_pt > 0).mean()), p_boot=p_b, p_wilcoxon=p_w,
            ))
    pv_df = pd.DataFrame(pv_rows)
    for fam, g in pv_df[pv_df['category'] != 'All held-out'].groupby('family'):
        pv_df.loc[g.index, 'p_holm'] = ST.holm(g['p_boot'])
    atomic_to_csv(pv_df, out_dv / 'pg_delta_pvalues.csv', index=False, float_format='%.4g')

    # Unit-level `when_rule` across ProteinGym held-out assays + MegaScale (val+test) libraries + Domainome held-out domains
    def _rho_1d(a: np.ndarray, b: np.ndarray, ub: np.ndarray | None = None) -> float:
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return np.nan
        if ok.all():
            ua = _unit_ranks(a)
            ub_eff = _unit_ranks(b) if ub is None else ub
        else:
            ua = _unit_ranks(a[ok])
            ub_eff = _unit_ranks(b[ok])
        if ua is None or ub_eff is None:
            return np.nan
        return float(np.dot(ua, ub_eff))

    unit_rows = []
    for fam, tag in (('FINAL_p0', 'p0'), ('FINAL_p1', 'p1')):
        u_pg = pd.DataFrame({
            'family': fam,
            'kind': 'ProteinGym',
            'category': pg_held['category'].values,
            'd_comb': (pg_held[f'{tag}_all_comb'] - pg_held[f'{tag}_all_wt']).values,
            'd_mt': (pg_held[f'{tag}_all_mt'] - pg_held[f'{tag}_all_wt']).values,
            'r_wt_mt': pg_held[f'{tag}_r_wt_mt'].values,
        })
        unit_rows.append(u_pg)
        for ds_u, kind_u in (('tsuboyama_capped_val', 'MegaScale'), ('tsuboyama_capped_test', 'MegaScale'), ('domainome', 'Domainome')):
            fr_u = frames[ds_u]
            if any(f'{fam}:{h}' not in fr_u.preds for h in ('wt', 'mt', 'comb')):
                continue
            sp_u = fr_u.spine
            use_u = sp_u['use'].to_numpy()
            grp_u = sp_u['group'].astype(str).to_numpy()
            y_full = sp_u['y'].to_numpy(float)
            w_p, m_p, c_p = fr_u.preds[f'{fam}:wt'], fr_u.preds[f'{fam}:mt'], fr_u.preds[f'{fam}:comb']
            used_pos = np.flatnonzero(use_u)
            grp_slices = pd.Series(used_pos).groupby(grp_u[used_pos]).groups
            u_recs = []
            for _g_id, idx_vals in grp_slices.items():
                sel = np.asarray(idx_vals, dtype=int)
                if len(sel) < 15:
                    continue
                yv = y_full[sel]
                if np.ptp(yv) == 0:
                    continue
                uy = _unit_ranks(yv) if np.isfinite(yv).all() else None
                rw = float(np.nanmean([_rho_1d(w_p[si, sel], yv, uy) for si in range(w_p.shape[0])]))
                rm = float(np.nanmean([_rho_1d(m_p[si, sel], yv, uy) for si in range(m_p.shape[0])]))
                rc = float(np.nanmean([_rho_1d(c_p[si, sel], yv, uy) for si in range(c_p.shape[0])]))
                rwm = float(np.nanmean([_rho_1d(w_p[si, sel], m_p[si, sel]) for si in range(w_p.shape[0])]))
                u_recs.append(dict(family=fam, kind=kind_u, category=kind_u, d_comb=rc - rw, d_mt=rm - rw, r_wt_mt=rwm))
            if u_recs:
                unit_rows.append(pd.DataFrame(u_recs))

    units_df = pd.concat(unit_rows, ignore_index=True).dropna()
    fit_rows = []
    for fam in ('FINAL_p0', 'FINAL_p1'):
        u = units_df[units_df['family'] == fam]
        x1 = u['d_mt'].to_numpy(float)
        x2 = (1.0 - u['r_wt_mt']).to_numpy(float)
        y = u['d_comb'].to_numpy(float)
        A1 = np.column_stack([np.ones(len(x1)), x1])
        b1 = np.linalg.lstsq(A1, y, rcond=None)[0]
        r2_1 = float(1.0 - ((y - A1 @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        A2 = np.column_stack([np.ones(len(x1)), x1, x2])
        b2 = np.linalg.lstsq(A2, y, rcond=None)[0]
        r2_2 = float(1.0 - ((y - A2 @ b2) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        fit_rows.append(dict(
            family=fam, n_units=len(u), intercept=float(b1[0]), slope=float(b1[1]), r2=r2_1,
            breakeven_d_mt=float(-b1[0] / b1[1]),
            intercept2=float(b2[0]), slope_d_mt=float(b2[1]), slope_one_minus_r=float(b2[2]), r2_two_predictors=r2_2,
        ))
    when_df = pd.DataFrame(fit_rows)
    atomic_to_csv(when_df, out_dv / 'when_rule.csv', index=False, float_format='%.4f')
    return pg_held, pv_df, units_df, when_df


# ----------------------------------------------------------------------------------------------
# Consolidated Extended Data Figure 1: `ablation_architecture_suite`
# ----------------------------------------------------------------------------------------------
def draw_ablation_architecture_suite(
    out_abl: Path,
    deltas_98: pd.DataFrame,
    pg_detail: pd.DataFrame,
    B: int,
) -> None:
    # Build unified lookup (family, head, col_id) -> (delta, lo, hi)
    lookup: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for r in deltas_98.itertuples():
        d_val = float(getattr(r, 'delta', r.delta_boot_mean))
        lookup[(r.family, r.head, r.column)] = (d_val, float(r.lo), float(r.hi))
    for r in pg_detail.itertuples():
        lookup[(r.family, r.head, r.column)] = (float(r.delta), float(r.lo), float(r.hi))

    col_groups = [
        ('External\nsingles', [
            ('ext_singles_mean', 'Composite (10)', 'both'),
            ('s669', 'S669', 'both'),
            ('s461', 'S461', 'both'),
            ('k3822', 'K3822', 'both'),
            ('q3421', 'Q3421', 'both'),
        ]),
        ('External\nmulti', [
            ('ext_multi_mean', 'PTMUL/-D', 'both'),
            ('ptmul', 'PTMUL', 'both'),
            ('ptmuld', 'PTMUL-D', 'both'),
        ]),
        ('MegaScale\nheld-out test', [
            ('tsuboyama_capped_test|singles', 'Singles', 'both'),
            ('tsuboyama_capped_test|multi', 'Doubles', 'both'),
            ('tsuboyama_capped_test|beyond_rho', 'Beyond ρ', 'epi'),
            ('tsuboyama_capped_test|cell_mag', 'Cell mag.', 'epi'),
        ]),
        ('Domainome\nheld-out', [
            ('domainome|lowid', 'Domainome', 'both'),
        ]),
        ('ProteinGym\n(173 held-out)', [
            ('pg|avg', 'PG avg', 'both'),
            ('pg|Stability', 'Stability', 'both'),
            ('pg|Expression', 'Expression', 'both'),
            ('pg|Activity', 'Activity', 'both'),
            ('pg|Binding', 'Binding', 'both'),
            ('pg|OrganismalFitness', 'Org. fitness', 'both'),
        ]),
    ]

    vlim = 0.04
    cmap = plt.get_cmap('RdBu')
    norm = Normalize(-vlim, vlim)

    fig = plt.figure(figsize=(16.6, 13.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 0.95], width_ratios=[1.0, 0.018], hspace=0.52, wspace=0.025)

    for panel_idx, (head, title_str) in enumerate([
        ('wt', 'a   WT-view readout (wt_lora_pred): paired Δρ × 1000 vs ESM-MSR (FINAL_p0)'),
        ('comb', 'b   Dual-view / epistasis readout (½[MT + WT] & epistasis head): paired Δρ × 1000 vs ESM-MSR'),
    ]):
        ax = fig.add_subplot(gs[panel_idx, 0])
        gap, x = 0.55, 0.0
        ticks, labs, spans = [], [], []
        for gname, specs in col_groups:
            x0 = x
            for col_id, lbl, mode_req in specs:
                h_eff = 'epi' if mode_req == 'epi' else head
                for i, fam in enumerate(ABL_ARMS):
                    val_tup = None if (head == 'wt' and mode_req == 'epi') else lookup.get((fam, h_eff, col_id))
                    if val_tup is None:
                        ax.add_patch(Rectangle((x, i), 1, 1, fc='#f3f3f1', ec='white', lw=1.1, hatch='////', zorder=1))
                        continue
                    d_v, lo_v, hi_v = val_tup
                    sig = (lo_v > 0) or (hi_v < 0)
                    ax.add_patch(Rectangle((x, i), 1, 1, fc=cmap(norm(float(np.clip(d_v, -vlim, vlim)))), ec='white', lw=1.1, zorder=1))
                    if sig:
                        ax.add_patch(Rectangle((x + 0.04, i + 0.04), 0.92, 0.92, fc='none', ec='#111111', lw=1.35, zorder=3))
                    ax.text(
                        x + 0.5, i + 0.52, f'{d_v * 1000:+.0f}', ha='center', va='center', fontsize=7.4,
                        fontweight='bold' if sig else 'normal',
                        color='white' if abs(d_v) > 0.027 else ('#111111' if sig else '#555555'), zorder=4,
                    )
                ticks.append(x + 0.5)
                labs.append(lbl)
                x += 1.0
            spans.append((gname, x0, x))
            x += gap

        ax.set_xlim(-0.15, x - gap + 0.15)
        ax.set_ylim(len(ABL_ARMS), -0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, rotation=45, ha='right', rotation_mode='anchor', fontsize=8.0)
        ax.set_yticks(np.arange(len(ABL_ARMS)) + 0.5)
        ax.set_yticklabels([_abl_style(f).label for f in ABL_ARMS], fontsize=8.4)
        for tick_lbl, fam in zip(ax.get_yticklabels(), ABL_ARMS):
            tick_lbl.set_color(_abl_style(fam).color)
            tick_lbl.set_fontweight('semibold')
        ax.tick_params(length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        if panel_idx == 0:
            for gname, x0, x1 in spans:
                ax.text((x0 + x1) / 2, -0.35, gname, ha='center', va='bottom', fontsize=8.3, fontweight='bold', color='#333333')
                ax.plot([x0, x1], [-0.15, -0.15], color='#444444', lw=1.1, clip_on=False)
        ax.set_title(title_str, loc='left', fontsize=9.8, fontweight='bold', pad=28 if panel_idx == 0 else 8)

    sm = cm.ScalarMappable(norm=Normalize(-vlim * 1000, vlim * 1000), cmap=cmap)
    cb = fig.colorbar(sm, cax=fig.add_subplot(gs[0, 1]), orientation='vertical')
    cb.ax.set_title('Δρ\n×1000', fontsize=8.0, pad=5)
    cb.ax.tick_params(labelsize=7.8)
    fig.add_subplot(gs[1, 1]).axis('off')

    # Panel c: Headline composite forest across all non-mega ablation arms (deployed readout)
    axc_grid = gs[2, 0].subgridspec(1, 6, wspace=0.18)
    headline_panels = [
        ('tsuboyama_capped_test|singles', 'wt', 'MegaScale\nsingles'),
        ('tsuboyama_capped_test|multi', 'comb', 'MegaScale\ndoubles'),
        ('tsuboyama_capped_test|beyond_rho', 'epi', 'Epistasis\nbeyond ρ'),
        ('tsuboyama_capped_test|cell_mag', 'epi', 'Epistasis\ncell mag.'),
        ('ext_singles_mean', 'wt', 'External\nsingles (10)'),
        ('domainome|lowid', 'wt', 'Domainome\nheld-out'),
    ]
    forest_fams = [f for f in ABL_ARMS if f != 'FINAL_p0_mega']
    for j, (col_id, h_use, p_title) in enumerate(headline_panels):
        ax_f = fig.add_subplot(axc_grid[0, j])
        ax_f.axvline(0, color='#777777', lw=0.9, ls='--', zorder=1)
        ax_f.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
        for yi, fam in enumerate(forest_fams):
            spec = _abl_style(fam)
            tup = lookup.get((fam, h_use, col_id))
            if tup is None:
                ax_f.axhline(yi, color='#eaeaea', lw=0.7, ls=':', zorder=1)
                continue
            dv, lov, hiv = tup
            sig = (lov > 0) or (hiv < 0)
            ax_f.plot([lov, hiv], [yi, yi], color=spec.color, lw=1.6, zorder=3)
            ax_f.plot(
                dv, yi, marker=spec.marker, ms=5.8, color=spec.color,
                mfc=spec.color if sig else 'white', mew=1.3, zorder=4,
            )
        ax_f.set_ylim(len(forest_fams) - 0.4, -0.6)
        ax_f.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_f.tick_params(axis='x', labelsize=7.6)
        ax_f.set_title(('c   ' if j == 0 else '') + p_title, loc='left', fontsize=9.0, fontweight='bold')
        ax_f.set_xlabel('Δρ vs ESM-MSR', fontsize=8.4)
        if j == 0:
            ax_f.set_yticks(range(len(forest_fams)))
            ax_f.set_yticklabels([_abl_style(f).label for f in forest_fams], fontsize=8.0)
            for tl, fam in zip(ax_f.get_yticklabels(), forest_fams):
                tl.set_color(_abl_style(fam).color)
        else:
            ax_f.set_yticks([])
    fig.add_subplot(gs[2, 1]).axis('off')

    fig.suptitle(
        f'ESM-MSR 10-ablation architecture, masking, and loss suite (B = {B:,} joint cluster/assay × 3-seed paired bootstrap)',
        x=0.012, ha='left', fontsize=10.8, fontweight='bold', y=0.985,
    )
    note_txt = (
        f'Panels a–b: Δρ × 1000 = ρ(ablation) − ρ(FINAL_p0) on identical variants (+12 = +0.012; blue = gain, red = loss; '
        f'bold black frame = 95% joint [protein/site/assay × triplicate-seed] CI excludes 0, B = {B:,}; hatched = not applicable / not run). '
        'Panel c: 95% paired bootstrap intervals on headline benchmarks (filled marker = 95% CI excludes 0).'
    )
    fig.subplots_adjust(left=0.175, right=0.955, top=0.915, bottom=0.095)
    wrapped = style.wrap_to_px(fig, note_txt, 0.955 * fig.bbox.width, fontsize=7.5, color='#333333')
    fig.text(0.012, 0.012, wrapped, ha='left', va='bottom', fontsize=7.5, color='#333333')
    style.save(fig, out_abl / 'ablation_architecture_suite', close=True)


# ----------------------------------------------------------------------------------------------
# Consolidated Extended Data Figure 2: `dual_view_summary`
# ----------------------------------------------------------------------------------------------
def draw_dual_view_summary(
    out_dv: Path,
    cvw_df: pd.DataFrame,
    pg_held: pd.DataFrame,
    pv_df: pd.DataFrame,
    units_df: pd.DataFrame,
    when_df: pd.DataFrame,
    B: int,
) -> None:
    fig, axes_2x2 = plt.subplots(2, 2, figsize=(16.2, 11.2))
    axa, axb = axes_2x2[0, 0], axes_2x2[0, 1]
    axc, axd = axes_2x2[1, 0], axes_2x2[1, 1]

    # Panel a: Paired comb - wt forest across stability singles, doubles, and functional DMSs (FINAL_p0 vs FINAL_p1)
    bench_rows = [
        ('ext_singles_mean', 'External singles (10)'),
        ('tsuboyama_capped_test|singles', 'MegaScale test singles'),
        ('domainome', 'Domainome held-out (436)'),
        ('tsuboyama_capped_test|multi', 'MegaScale test doubles'),
        ('ptmul', 'PTMUL (multi)'),
        ('ptmuld', 'PTMUL-D (doubles)'),
        ('DLG4_HUMAN_Faure_2021_binding_domain|multi', 'DLG4 binding (multi)'),
        ('GRB2_HUMAN_Faure_2021_binding_domain|multi', 'GRB2 binding (multi)'),
        ('MYO_HUMAN_Kung_2025_display|multi', 'MYO display (multi)'),
    ]
    axa.axvline(0, color='#666666', lw=0.9, ls='--', zorder=1)
    axa.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    for yi in range(len(bench_rows)):
        if yi % 2 == 0:
            axa.axhspan(yi - 0.5, yi + 0.5, color=style.BAND, lw=0, zorder=0)
    for fi, (fam, y_off) in enumerate([('FINAL_p0', -0.15), ('FINAL_p1', +0.15)]):
        spec = _abl_style(fam)
        for yi, (col_id, _) in enumerate(bench_rows):
            hit = cvw_df[(cvw_df['family'] == fam) & (cvw_df['column'] == col_id)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            sig = (r['lo'] > 0) or (r['hi'] < 0)
            axa.plot([r['lo'], r['hi']], [yi + y_off, yi + y_off], color=spec.color, lw=1.7, zorder=3)
            axa.plot(
                r['delta'], yi + y_off, marker=spec.marker, ms=6.0, color=spec.color,
                mfc=spec.color if sig else 'white', mew=1.3, zorder=4,
            )
    axa.set_yticks(range(len(bench_rows)))
    axa.set_yticklabels([lbl for _, lbl in bench_rows], fontsize=8.5)
    axa.set_ylim(len(bench_rows) - 0.4, -0.6)
    axa.set_xlabel('Δρ = ρ(Dual-view ½[MT + WT]) − ρ(WT-view)')
    axa.set_title('a   Dual-view vs WT-view across benchmarks', loc='left', fontsize=10.0, fontweight='bold')
    axa.legend(
        handles=[
            Line2D([0], [0], marker=_abl_style(f).marker, color=_abl_style(f).color, lw=1.7, ms=6, label=_abl_style(f).label)
            for f in ('FINAL_p0', 'FINAL_p1')
        ],
        loc='lower right', fontsize=8.0, frameon=True, facecolor='white', edgecolor='#dddddd',
    )

    # Panel b: ProteinGym per-category paired Δρ = ρ(comb) - ρ(wt) raincloud on FINAL_p0 (173 held-out assays)
    pv_p0 = pv_df[(pv_df['family'] == 'FINAL_p0') & (pv_df['category'] != 'All held-out')].set_index('category')
    rng_b = np.random.default_rng(4)
    axb.axhline(0, color='#666666', lw=0.9, ls='--', zorder=1)
    axb.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    xt_labels = []
    for xi, cat in enumerate(PG.PG_CATEGORIES):
        if xi % 2 == 0:
            axb.axvspan(xi - 0.5, xi + 0.5, color=style.BAND, lw=0, zorder=0)
        sub = pg_held[pg_held['category'] == cat]
        w_seeds = sub[[f'p0_s{sd}_all_wt' for sd in SEEDS]].to_numpy(float).T
        c_seeds = sub[[f'p0_s{sd}_all_comb' for sd in SEEDS]].to_numpy(float).T
        d_seeds = c_seeds - w_seeds
        r_pv = pv_p0.loc[cat]
        style.raincloud(
            axb, float(xi), d_seeds, M.CATEGORY_COLOR[cat],
            mean=float(r_pv['mean_delta']), lo=float(r_pv['lo']), hi=float(r_pv['hi']),
            width=0.62, rng=rng_b, annotate=True, fs=7.8,
        )
        p_txt = ST.fmt_p(float(r_pv['p_holm']), prefix='p=')
        xt_labels.append(f'{PG.PG_CATEGORY_SHORT[cat]}\n({len(sub)})\n{p_txt}')
    axb.set_xticks(range(len(PG.PG_CATEGORIES)))
    axb.set_xticklabels(xt_labels, fontsize=8.2)
    axb.set_xlim(-0.5, len(PG.PG_CATEGORIES) - 0.5)
    axb.set_ylabel('Δρ = ρ(Dual-view) − ρ(WT-view)')
    axb.set_title(f'b   ProteinGym Dual-view gain by property ({len(pg_held)} held-out assays)', loc='left', fontsize=10.0, fontweight='bold')

    # Panel c: Mechanistic `when_rule` law on FINAL_p0 (Δ_comb vs Δ_mt colored by 1 - r(WT, MT))
    u0 = units_df[units_df['family'] == 'FINAL_p0'].copy()
    w0 = when_df[when_df['family'] == 'FINAL_p0'].iloc[0]
    div_val = np.clip(1.0 - u0['r_wt_mt'].to_numpy(float), 0.0, 0.28)
    sc = axc.scatter(
        u0['d_mt'], u0['d_comb'], c=div_val, cmap='viridis', s=18, alpha=0.80, edgecolors='none', zorder=3, rasterized=True,
    )
    xs_line = np.array([-0.38, 0.22])
    axc.plot(xs_line, w0['intercept'] + w0['slope'] * xs_line, color='#111111', lw=1.6, zorder=4)
    axc.axhline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    axc.axvline(0, color='#777777', lw=0.85, ls='--', zorder=1)
    cb = fig.colorbar(sc, ax=axc, pad=0.02, fraction=0.046)
    cb.set_label('1 − ρ(WT, MT)  (head diversity)', fontsize=8.3)
    cb.ax.tick_params(labelsize=7.8)
    stat_box = (
        f"Univariate: Δ_comb = {w0['intercept']:+.3f} + {w0['slope']:.2f} Δ_mt  (R² = {w0['r2']:.2f})\n"
        f"Break-even MT deficit: Δ_mt = {w0['breakeven_d_mt']:+.3f}\n"
        f"With head diversity (1 − r): R² = {w0['r2_two_predictors']:.2f}  (n = {int(w0['n_units'])} units)"
    )
    axc.text(
        0.03, 0.96, stat_box, transform=axc.transAxes, va='top', ha='left', fontsize=8.0,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.92), zorder=5,
    )
    axc.set_xlim(-0.38, 0.22)
    axc.set_ylim(-0.16, 0.20)
    axc.grid(axis='both', color=style.GRID, lw=0.5, zorder=0)
    axc.set_xlabel('Δ_mt = ρ(MT head) − ρ(WT head)')
    axc.set_ylabel('Δ_comb = ρ(Dual-view) − ρ(WT head)')
    axc.set_title('c   Mechanistic law: when averaging MT + WT improves ranking', loc='left', fontsize=10.0, fontweight='bold')

    # Panel d: Dual-view minus WT-view across all 11 FINAL_* families
    d_cols = [
        ('tsuboyama_capped_test|multi', 'MegaScale doubles', '#1f5f8b'),
        ('ptmuld', 'PTMUL-D', '#1b8a4b'),
        ('dms_multi_mean', 'DMS multis (3)', '#eb6834'),
        ('ext_singles_mean', 'External singles (10)', '#7f8c8d'),
    ]
    axd.axvline(0, color='#666666', lw=0.9, ls='--', zorder=1)
    axd.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    y_fams = np.arange(len(MAIN_11_FAMS))
    bar_h = 0.19
    for ci_idx, (col_id, col_lbl, col_hex) in enumerate(d_cols):
        offs = (ci_idx - 1.5) * bar_h
        vals = []
        for fam in MAIN_11_FAMS:
            hit = cvw_df[(cvw_df['family'] == fam) & (cvw_df['column'] == col_id)]
            vals.append(float(hit['delta'].iloc[0]) if len(hit) else np.nan)
        axd.barh(y_fams + offs, vals, bar_h * 0.92, color=col_hex, alpha=0.88, label=col_lbl, zorder=3)
    axd.set_yticks(y_fams)
    axd.set_yticklabels([_abl_style(f).label for f in MAIN_11_FAMS], fontsize=8.0)
    for tl, fam in zip(axd.get_yticklabels(), MAIN_11_FAMS):
        if fam == REF:
            tl.set_fontweight('bold')
    axd.set_ylim(len(MAIN_11_FAMS) - 0.4, -0.6)
    axd.set_xlabel('Δρ = ρ(Dual-view) − ρ(WT-view)')
    axd.set_title('d   Dual-view gain across all 11 FINAL_* training recipes', loc='left', fontsize=10.0, fontweight='bold')
    axd.legend(loc='lower right', fontsize=7.8, frameon=True, facecolor='white', edgecolor='#dddddd')

    n_u_pg = int((u0['kind'] == 'ProteinGym').sum())
    n_u_mg = int((u0['kind'] == 'MegaScale').sum())
    n_u_dom = int((u0['kind'] == 'Domainome').sum())
    be_dmt = abs(float(w0['breakeven_d_mt']))
    style.finish_figure(
        fig,
        out_dv / 'dual_view_summary',
        title=f'Dual-view ½(MT + WT) vs WT-view readout: empirical gains and mechanistic variance-reduction law (B = {B:,})',
        note=(
            f'Joint [protein/site/assay × triplicate-seed] paired bootstraps (B = {B:,}; Holm-adjusted across the 5 ProteinGym categories in panel b). '
            f'Panel c pools held-out ProteinGym assays ({n_u_pg}), MegaScale val/test libraries ({n_u_mg}), and held-out Domainome domains ({n_u_dom}): '
            f'averaging the mutant-context pass improves ρ on average even when the MT head is up to {be_dmt:.3f} ρ weaker than the WT head, driven by head decorrelation 1 − ρ(WT, MT).'
        ),
        note_fontsize=7.5,
    )


# ----------------------------------------------------------------------------------------------
# Main Entrypoint
# ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--B', type=int, default=1000, help='Bootstrap replicates (default: 1000)')
    ap.add_argument('--fast', action='store_true', help='Fast smoke mode (B = 100)')
    ap.add_argument('--reuse-cells', action='store_true', help='Reuse fresh cells_long.csv and cells_long_mega_own_split.csv when present')
    ap.add_argument('--sigma', default='1.0', help='LoRA scale (default: 1.0)')
    ap.add_argument('--procs', type=int, default=8, help='Worker processes for epistasis and benchmark bootstraps')
    ap.add_argument('--split', default='capped', choices=('capped', 'mega'), help='Accepted for runner uniformity')
    args = ap.parse_args(argv)
    B = 100 if args.fast else int(args.B)

    out_abl = results_dir('ablation')
    out_dv = results_dir('dual_view')
    out_dms = results_dir('dms3')

    datasets = EXT12 + DMS3 + ['domainome', 'tsuboyama_capped_val', 'tsuboyama_capped_test']
    frames = _load_all_final_frames(datasets, sigma=args.sigma, procs=args.procs)
    cov_df = pd.DataFrame([
        dict(dataset=ds, family=fam, seeds=','.join(map(str, ss)))
        for ds in datasets
        for fam, ss in frames[ds].notes['seeds'].items()
    ])
    atomic_to_csv(cov_df[~cov_df['dataset'].isin(DMS3_SET)], out_abl / 'coverage.csv', index=False)
    atomic_to_csv(cov_df[cov_df['dataset'].isin(DMS3_SET)], out_dms / 'ablation_coverage.csv', index=False)

    pg_cat, pg_matrices = load_pg_ablation_data(procs=args.procs)
    n_pg_held = len(pg_cat)
    n_pg_mm = int(sum(PG.assay_has_multi(d)[0] for d in pg_cat.index))

    cells_path = out_abl / 'cells_long.csv'
    dms_cells_path = out_dms / 'ablation_cells_long.csv'
    mega_path = out_abl / 'cells_long_mega_own_split.csv'
    expected_fams = {fam for ds in datasets for fam in frames[ds].notes['seeds']}
    can_reuse = bool(args.reuse_cells and cells_path.exists() and mega_path.exists())
    if can_reuse:
        cells_main = pd.read_csv(cells_path)
        if not expected_fams.issubset(set(cells_main['family'].unique())):
            can_reuse = False

    if can_reuse:
        cells_parts = [cells_main]
        if dms_cells_path.exists():
            cells_parts.append(pd.read_csv(dms_cells_path))
        cells = pd.concat(cells_parts, ignore_index=True).drop_duplicates(['family', 'seed', 'head', 'column'])
        mega_cells = pd.read_csv(mega_path)
        _, epi_boot_df = epi_cells_and_boot(frames, B=B, procs=args.procs, reuse_cells=cells_main)
    else:
        cells = score_columns(frames)
        epi_cell_df, epi_boot_df = epi_cells_and_boot(frames, B=B, procs=args.procs)
        cells = pd.concat([cells, epi_cell_df], ignore_index=True)

        mega_frames = {
            ds: load_final_frame(ds, sigma=args.sigma, mega_own_split=True)
            for ds in ('tsuboyama_capped_val', 'tsuboyama_capped_test')
        }
        mega_cells = add_primary_non_pg(score_columns({k: v for k, v in mega_frames.items() if v is not None}))
        atomic_to_csv(mega_cells, mega_path, index=False, float_format='%.5f')

        cells = add_primary_non_pg(cells)
        cells = pd.concat([cells, pg_cells_from_matrices(pg_cat, pg_matrices)], ignore_index=True)
        cells = add_composites(cells)
        is_dms_cell = cells['dataset'].isin(DMS3_SET) | cells['column'].isin(DMS3_COLS)
        atomic_to_csv(cells[~is_dms_cell], cells_path, index=False, float_format='%.5f')
        atomic_to_csv(cells[is_dms_cell], dms_cells_path, index=False, float_format='%.5f')

    mega_own = summarize(mega_cells).set_index(['family', 'head', 'column']) if len(mega_cells) else None
    summ = summarize(cells)
    summ_main = summ[~summ['column'].isin(DMS3_COLS)].copy()
    summ_dms = summ[summ['column'].isin(DMS3_COLS)].copy()
    atomic_to_csv(summ_main, out_abl / 'summary.csv', index=False, float_format='%.5f')
    atomic_to_csv(summ_dms, out_dms / 'ablation_summary.csv', index=False, float_format='%.5f')
    write_informativeness(out_abl, out_dms, summ)

    t_b0 = time.perf_counter()
    deltas_bench, cvw_df = boot_benchmark_deltas(frames, B=B, procs=args.procs)
    t_b1 = time.perf_counter()
    pg_98_boot, pg_detail = pg_paired_boot(pg_cat, pg_matrices, B=B)
    deltas = pd.concat([deltas_bench, epi_boot_df, pg_98_boot], ignore_index=True)
    deltas_main = deltas[~deltas['column'].isin(DMS3_COLS)].copy()
    deltas_dms = deltas[deltas['column'].isin(DMS3_COLS)].copy()
    atomic_to_csv(deltas_main, out_abl / 'delta_paired.csv', index=False, float_format='%.5f')
    atomic_to_csv(deltas_dms, out_dms / 'ablation_delta_paired.csv', index=False, float_format='%.5f')

    render_delta_tables(out_abl, summ_main, deltas_main, B=B, n_pg_held=n_pg_held, n_pg_mm=n_pg_mm)
    render_candidate_layouts(out_abl, summ_main, deltas_main, mega_own, B=B)

    # Write merged 09_ablation/ tables & Extended Data Figure 1 (zero 3-DMS columns!)
    is_dms_bench = deltas_bench['column'].isin(DMS3_COLS)
    atomic_to_csv(deltas_bench[~is_dms_bench], out_abl / 'paired_vs_p0.csv', index=False, float_format='%.5f')
    atomic_to_csv(deltas_bench[is_dms_bench], out_dms / 'ablation_paired_vs_p0.csv', index=False, float_format='%.5f')
    atomic_to_csv(pg_detail, out_abl / 'paired_vs_p0_proteingym.csv', index=False, float_format='%.5f')

    comp_main = deltas_main[
        deltas_main['column'].isin(['ext_singles_mean', 'ext_multi_mean', 'tsuboyama_capped_test', 'tsuboyama_capped_test|beyond_rho', 'domainome|lowid', 'pg|avg'])
        & (deltas_main['head'] == 'primary')
    ].copy()
    write_stat_table(comp_main[['family', 'column', 'delta', 'lo', 'hi', 'p', 'B']], out_abl / 'table_composites', float_fmt='%.4f')
    comp_dms = deltas_dms[(deltas_dms['column'] == 'dms_mean') & (deltas_dms['head'] == 'primary')].copy()
    atomic_to_csv(comp_dms[['family', 'column', 'delta', 'lo', 'hi', 'p', 'B']], out_dms / 'ablation_table_composites.csv', index=False, float_format='%.4f')

    cnt_rows = []
    for fam in ABL_ARMS:
        sub_f = deltas_main[(deltas_main['family'] == fam) & (deltas_main['head'].isin(['wt', 'comb']))]
        cnt_rows.append(dict(
            family=fam, label=_abl_style(fam).label,
            n_cells=len(sub_f),
            sig_better=int((sub_f['lo'] > 0).sum()),
            sig_worse=int((sub_f['hi'] < 0).sum()),
        ))
    write_stat_table(pd.DataFrame(cnt_rows), out_abl / 'table_counts', float_fmt='%.4f')
    draw_ablation_architecture_suite(out_abl, deltas_main, pg_detail, B=B)

    # Write 99_misc/dual_view/ tables & Extended Data Figure 2
    pg_held, pv_df, units_df, when_df = build_dual_view_study(out_dv, frames, cvw_df, B=B)
    draw_dual_view_summary(out_dv, cvw_df, pg_held, pv_df, units_df, when_df, B=B)

    p0_pg_dep = summ.query("family == 'FINAL_p0' and head == 'primary' and column == 'pg|avg'")['mean'].iloc[0]
    p0_pg_wt = summ.query("family == 'FINAL_p0' and head == 'wt' and column == 'pg|avg'")['mean'].iloc[0]
    elapsed = time.perf_counter() - t_start
    print(
        f'[p11_ablations_and_heads] B={B} complete in {elapsed:.1f}s '
        f'(boot_benchmark_deltas={t_b1 - t_b0:.1f}s). '
        f'FINAL_p0 PG avg: deployed={p0_pg_dep:.4f}, wt={p0_pg_wt:.4f}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
