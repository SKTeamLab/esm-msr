#!/usr/bin/env python3
"""ProteinGym multi-property Pareto-frontier evaluation for ESM-MSR (σ = 1.0, 0.5) vs all 95 public models + stability baselines.

Evaluates all models on the single unified set of **176 unseen ProteinGym assays** (`217 - 41` homologous under
the repository's four-part sequence + structure homology filter, `analysis/reference/tsuboyama_rule_flags.csv`:
MMseqs2 E <= 1e-3, BLOSUM62 local identity > 0.40, Foldseek E <= 1e-3, or Foldseek TM-score > 0.70 against the
172 capped MegaScale training libraries). No `PG_DROP` subsetting is applied anywhere.

Usage:
    PYTHONPATH=src:analysis python -m studies.pgym_pareto.run --B 10000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr

from msr_analysis import heldout
from msr_analysis import proteingym as PG
from msr_analysis import style
from msr_analysis.boot import ci
from msr_analysis.paths import (
    CACHE,
    ESM3_S0_PGYM_COMB,
    FINAL_PGYM_COMB,
    FINAL_PGYM_COMB_S05,
    FINAL_PGYM_WT,
    FINAL_PGYM_WT_S05,
    MSR_ROOT,
    PGYM,
    PGYM_REF_CSV,
    PUBLIC,
    REFERENCE,
    RETRAINED,
    atomic_to_csv,
    results_dir,
)
from msr_analysis.stats import boot_pvalue

CATS = ('Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness')
CAT_SHORT = {
    'Stability': 'Stability',
    'Expression': 'Expression',
    'Activity': 'Activity',
    'Binding': 'Binding',
    'OrganismalFitness': 'Org. fitness',
}
SEEDS = (1, 2, 3)

# Official ProteinGym v1.3 inference-time input modality classification for the 95 public zero-shot models
MSA_OR_RETRIEVAL = frozenset({
    'Site_Independent', 'EVmutation', 'DeepSequence_single', 'DeepSequence_ensemble',
    'EVE_single', 'EVE_ensemble', 'Unirep_evotune',
    'MSA_Transformer_single', 'MSA_Transformer_ensemble',
    'GEMME', 'ESCOTT', 'RSALOR', 'S2F_MSA', 'S3F_MSA', 'SiteRM',
    'Tranception_S', 'Tranception_M', 'Tranception_L',
    'TranceptEVE_S', 'TranceptEVE_M', 'TranceptEVE_L',
    'PoET', 'VenusREM',
})
STRUCT_NO_MSA = frozenset({
    'MIF', 'MIFST', 'ESM-IF1', 'ProteinMPNN',
    'ProtSSN_k10_h512', 'ProtSSN_k10_h768', 'ProtSSN_k10_h1280',
    'ProtSSN_k20_h512', 'ProtSSN_k20_h768', 'ProtSSN_k20_h1280',
    'ProtSSN_k30_h512', 'ProtSSN_k30_h768', 'ProtSSN_k30_h1280', 'ProtSSN_ensemble',
    'SaProt_650M_AF2', 'SaProt_35M_AF2', 'MULAN_small',
    'ProSST-20', 'ProSST-128', 'ProSST-512', 'ProSST-1024', 'ProSST-2048', 'ProSST-4096',
    'S2F', 'S3F', 'ESM3',
})

STABILITY_BASELINES_CLEAN = ('SPURS', 'ThermoMPNN', 'MutateEverything', 'GeoDDG-3D', 'Rosetta')
STABILITY_BASELINES_LEAKED = ('ESM3ΔG', 'ESMtherm')

PAIRED_PG_PROTEINS = [
    ('KRAS', 'RASK_HUMAN_Weng_2022_abundance', 'RASK_HUMAN_Weng_2022_binding-DARPin_K55'),
    ('PTEN', 'PTEN_HUMAN_Matreyek_2021', 'PTEN_HUMAN_Mighell_2018'),
    ('KCNE1', 'KCNE1_HUMAN_Muhammad_2023_expression', 'KCNE1_HUMAN_Muhammad_2023_function'),
    ('Kir2.1', 'KCNJ2_MOUSE_Coyote-Maestas_2022_surface', 'KCNJ2_MOUSE_Coyote-Maestas_2022_function'),
    ('CYP2C9', 'CP2C9_HUMAN_Amorosi_2021_abundance', 'CP2C9_HUMAN_Amorosi_2021_activity'),
    ('GCK', 'HXK4_HUMAN_Gersing_2023_abundance', 'HXK4_HUMAN_Gersing_2022_activity'),
    ('OCT1', 'S22A1_HUMAN_Yee_2023_abundance', 'S22A1_HUMAN_Yee_2023_activity'),
    ('VKOR1', 'VKOR1_HUMAN_Chiasson_2020_abundance', 'VKOR1_HUMAN_Chiasson_2020_activity'),
    ('DAAO', 'OXDA_RHOTO_Vanella_2023_expression', 'OXDA_RHOTO_Vanella_2023_activity'),
    ('HLA-A*02', 'Q53Z42_HUMAN_McShan_2019_expression', 'Q53Z42_HUMAN_McShan_2019_binding-TAPBPR'),
    ('Spike RBD', 'SPIKE_SARS2_Starr_2020_expression', 'SPIKE_SARS2_Starr_2020_binding'),
]


def modality_of(model: str) -> str:
    if model.startswith('ESM-MSR'):
        return 'ESM-MSR'
    if model in STABILITY_BASELINES_CLEAN:
        return 'Stability baseline (clean)'
    if model in STABILITY_BASELINES_LEAKED:
        return 'Stability baseline (leaked*)'
    if model in MSA_OR_RETRIEVAL:
        return 'MSA / Retrieval'
    if model in STRUCT_NO_MSA:
        return 'Structure + Sequence (no MSA)'
    return 'Sequence-only'


def _safe_rho(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    pv, yv = p[ok], y[ok]
    if np.ptp(pv) == 0 or np.ptp(yv) == 0:
        return np.nan
    return float(spearmanr(pv, yv)[0])


def _safe_spearman_cols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    out = np.full(X.shape[1], np.nan, dtype=float)
    y_fin = np.isfinite(y)
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = y_fin & np.isfinite(x)
        if ok.sum() < 3:
            continue
        xv, yv = x[ok], y[ok]
        if np.ptp(xv) == 0 or np.ptp(yv) == 0:
            continue
        out[j] = float(spearmanr(xv, yv)[0])
    return out


def load_public_95_per_assay(refresh: bool = False) -> pd.DataFrame:
    cache_csv = CACHE / 'pgym_pareto_pub95_217.csv'
    if cache_csv.exists() and not refresh:
        return pd.read_csv(cache_csv, index_col='DMS_id')

    zdir = MSR_ROOT / 'zero_shot'
    files = sorted(zdir.glob('*.csv'))
    base_cols = {'mutant', 'mutated_sequence', 'DMS_score', 'DMS_score_bin', 'DMS_bin_score'}
    mcols = [c for c in pd.read_csv(files[0], nrows=0).columns if c not in base_cols]
    rows = {}
    for f in files:
        df = pd.read_csv(f, usecols=['DMS_score'] + mcols)
        rows[f.stem] = _safe_spearman_cols(df['DMS_score'].to_numpy(float), df[mcols].to_numpy(float))
    pub_df = pd.DataFrame.from_dict(rows, orient='index', columns=mcols)
    pub_df.index.name = 'DMS_id'
    atomic_to_csv(pub_df, cache_csv)
    return pub_df


def load_stability_baselines_all217(dms_ids: list[str], refresh: bool = False) -> pd.DataFrame:
    """Score all 7 stability baselines directly across all 217 ProteinGym assays (no PG_DROP, no SPURS published)."""
    cache_csv = CACHE / 'pgym_pareto_stab_baselines_217.csv'
    if cache_csv.exists() and not refresh:
        return pd.read_csv(cache_csv, index_col='DMS_id').reindex(dms_ids)

    ro = pd.read_csv(PGYM / 'pgym_rosetta_singles_per_dms.csv').set_index('DMS_id')
    tmpd_ref = pd.read_csv(REFERENCE / 'pgym_thermompnn_corrected_s1_per_dms.csv').set_index('DMS_id')

    def _sd(*parts):
        return {k: RETRAINED.joinpath(*parts, f'seed{k}', 'proteingym') for k in SEEDS}

    tmp_sa_dir = _sd('thermompnn', 'predictions', 'single_additive')
    tmp_d_dir = _sd('thermompnn', 'predictions', 'epistatic_corrected')
    me1_dir = _sd('mutate_everything', 'predictions', 'stage1')
    me2_dir = _sd('mutate_everything', 'predictions', 'stage2')
    sp_dir = _sd('spurs', 'predictions')
    geo_dir = PUBLIC / 'geoddg' / 'faspr' / 'proteingym'
    esm3dg_dir = PUBLIC / 'esm3dg' / 'proteingym'
    esmtherm_dir = PUBLIC / 'esmtherm' / 'proteingym'

    def _align_me2(dm2: pd.DataFrame, dt_spine: pd.DataFrame) -> float:
        m_ren = dm2['mut_info'].isin(set(dt_spine['mut_type_renumbered'])).mean()
        m_mut = dm2['mut_info'].isin(set(dt_spine['mutant'])).mean()
        kcol = 'mut_type_renumbered' if m_ren >= m_mut else 'mutant'
        s_map = dm2.drop_duplicates('mut_info').set_index('mut_info')['ddg_pred']
        v2 = -dt_spine[kcol].map(s_map).to_numpy(dtype=float)
        return _safe_rho(v2, dt_spine['DMS_score'].to_numpy(dtype=float))

    def _score_rowid_dir(folder: Path, dms: str, y_full: np.ndarray) -> float:
        f = folder / f'{dms}.csv'
        if not f.exists():
            return np.nan
        dp = pd.read_csv(f, usecols=['row_id', 'pred'])
        rid = dp['row_id'].to_numpy(dtype=int)
        return _safe_rho(dp['pred'].to_numpy(dtype=float), y_full[rid])

    def _seed_mean(dirs: dict[int, Path], dms: str, pred_col: str) -> float:
        vals = []
        for k in SEEDS:
            f = dirs[k] / f'{dms}.csv'
            if f.exists():
                d = pd.read_csv(f, usecols=[pred_col, 'DMS_score'])
                vals.append(_safe_rho(d[pred_col].to_numpy(float), d['DMS_score'].to_numpy(float)))
        return float(np.nanmean(vals)) if vals else np.nan

    rows = {}
    for dms in dms_ids:
        dt = pd.read_csv(
            tmp_sa_dir[1] / f'{dms}.csv',
            usecols=['mut_type_renumbered', 'mutant', 'n_mut', 'pred_single_additive', 'DMS_score'],
        )
        y_full = dt['DMS_score'].to_numpy(dtype=float)
        has_mm = bool((dt['n_mut'] > 1).any())
        has_dbl = bool((dt['n_mut'] == 2).any())

        # ThermoMPNN / -D
        if has_dbl:
            d_seeds = []
            for k in SEEDS:
                f = tmp_d_dir[k] / f'{dms}.csv'
                if f.exists():
                    dtd = pd.read_csv(f, usecols=['n_mut', 'pred_single_additive', 'pred_combined', 'DMS_score'])
                    comb_t = np.where(dtd['n_mut'] == 2, dtd['pred_combined'], dtd['pred_single_additive'])
                    d_seeds.append(_safe_rho(comb_t, dtd['DMS_score'].to_numpy(float)))
            tmp_val = float(np.nanmean(d_seeds)) if d_seeds else (float(tmpd_ref.loc[dms, 'rho_tmpnnd']) if dms in tmpd_ref.index else np.nan)
        else:
            tmp_val = _seed_mean(tmp_sa_dir, dms, 'pred_single_additive')

        # MutateEverything
        if has_mm:
            me_vals = [
                _align_me2(pd.read_csv(me2_dir[k] / f'{dms}.csv', usecols=['mut_info', 'ddg_pred']), dt)
                for k in SEEDS
                if (me2_dir[k] / f'{dms}.csv').exists()
            ]
            me_val = float(np.nanmean(me_vals)) if me_vals else np.nan
        else:
            me_val = _seed_mean(me1_dir, dms, 'pred_combined')

        # SPURS (retrained)
        sp_val = _seed_mean(sp_dir, dms, 'SPURS_multi' if has_mm else 'SPURS_single_additive')

        rows[dms] = {
            'SPURS': sp_val,
            'ThermoMPNN': tmp_val,
            'MutateEverything': me_val,
            'GeoDDG-3D': _score_rowid_dir(geo_dir, dms, y_full),
            'Rosetta': float(ro.loc[dms, 'rho']) if dms in ro.index else np.nan,
            'ESM3ΔG': _score_rowid_dir(esm3dg_dir, dms, y_full),
            'ESMtherm': _score_rowid_dir(esmtherm_dir, dms, y_full),
        }
    out_df = pd.DataFrame.from_dict(rows, orient='index')
    out_df.index.name = 'DMS_id'
    atomic_to_csv(out_df, cache_csv)
    return out_df.reindex(dms_ids)


def load_esm_msr_sigma0_all217(dms_ids: list[str], has_multi_map: pd.Series) -> pd.Series:
    """Load complete 217-assay ESM-MSR σ=0.0 deployed scores (final_pgym_wt_sigma0/FINAL_p0/s1 + combined on multi)."""
    s0_wt_dir = MSR_ROOT / 'final_pgym_wt_sigma0' / 'FINAL_p0' / 's1'
    vals = {}
    for dms in dms_ids:
        if bool(has_multi_map.loc[dms]):
            cp = ESM3_S0_PGYM_COMB / f'{dms}.csv'
            if cp.exists():
                d = pd.read_csv(cp, usecols=['combined_pred', 'DMS_score'])
                vals[dms] = _safe_rho(d['combined_pred'].to_numpy(float), d['DMS_score'].to_numpy(float))
                continue
        wp = s0_wt_dir / f'{dms}.csv'
        d = pd.read_csv(wp, usecols=['wt_lora_pred', 'DMS_score'])
        vals[dms] = _safe_rho(d['wt_lora_pred'].to_numpy(float), d['DMS_score'].to_numpy(float))
    return pd.Series(vals, name='ESM-MSR (σ=0.0)')


def verify_alignment(held_ids: list[str]) -> dict[str, int]:
    zdir = MSR_ROOT / 'zero_shot'
    s0_wt_dir = MSR_ROOT / 'final_pgym_wt_sigma0' / 'FINAL_p0' / 's1'
    exact_wt = 0
    exact_comb = 0
    total_comb = 0
    for aid in held_ids:
        zdf = pd.read_csv(zdir / f'{aid}.csv', usecols=['mutant', 'DMS_score']).drop_duplicates('mutant').set_index('mutant')
        p1 = FINAL_PGYM_WT / 'FINAL_p0' / 's1' / f'{aid}.csv'
        p05 = FINAL_PGYM_WT_S05 / 'FINAL_p0' / 's1' / f'{aid}.csv'
        p0 = s0_wt_dir / f'{aid}.csv'
        d1 = pd.read_csv(p1, usecols=['mutant', 'DMS_score']).drop_duplicates('mutant').set_index('mutant')
        d05 = pd.read_csv(p05, usecols=['mutant', 'DMS_score']).drop_duplicates('mutant').set_index('mutant')
        d0 = pd.read_csv(p0, usecols=['mutant', 'DMS_score']).drop_duplicates('mutant').set_index('mutant')
        if (
            len(zdf) == len(d1) == len(d05) == len(d0)
            and set(zdf.index) == set(d1.index) == set(d05.index) == set(d0.index)
            and np.allclose(zdf['DMS_score'].to_numpy(float), d1.loc[zdf.index, 'DMS_score'].to_numpy(float), rtol=1e-5, atol=1e-5, equal_nan=True)
        ):
            exact_wt += 1
        for base_d in (FINAL_PGYM_COMB / 'FINAL_p0' / 's1', FINAL_PGYM_COMB_S05 / 'FINAL_p0' / 's1'):
            cp = base_d / f'{aid}.csv'
            if cp.exists():
                total_comb += 1
                cd = pd.read_csv(cp, usecols=['mutant']).drop_duplicates('mutant')
                if len(cd) == len(zdf) and set(cd['mutant']) == set(zdf.index):
                    exact_comb += 1
    return dict(n_assays=len(held_ids), exact_wt=exact_wt, total_comb=total_comb, exact_comb=exact_comb)


def pareto_nondominated(tab: pd.DataFrame, cols: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    A = tab[cols].to_numpy(float)
    names = tab.index.tolist()
    nd, dom = [], {}
    for i, nm in enumerate(names):
        if np.isnan(A[i]).any():
            continue
        db = [
            names[j]
            for j in range(len(names))
            if j != i and not np.isnan(A[j]).any() and np.all(A[j] >= A[i]) and np.any(A[j] > A[i])
        ]
        if not db:
            nd.append(nm)
        else:
            dom[nm] = db
    return nd, dom


def convex_hull_supported(tab: pd.DataFrame, cols: list[str]) -> list[tuple[str, float, np.ndarray]]:
    """Exact LP check for strictly supported Pareto points (vertices of upper convex hull)."""
    A = tab[cols].to_numpy(float)
    names = tab.index.tolist()
    d = len(cols)
    sup = []
    for i, nm in enumerate(names):
        if np.isnan(A[i]).any():
            continue
        c_obj = np.zeros(d + 1)
        c_obj[-1] = -1.0
        A_ub, b_ub = [], []
        for j in range(len(names)):
            if j == i or np.isnan(A[j]).any():
                continue
            A_ub.append(np.append(A[j] - A[i], 1.0))
            b_ub.append(0.0)
        A_eq = np.zeros((1, d + 1))
        A_eq[0, :d] = 1.0
        res = linprog(
            c_obj,
            A_ub=np.asarray(A_ub),
            b_ub=np.asarray(b_ub),
            A_eq=A_eq,
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * d + [(-1.0, 1.0)],
            method='highs',
        )
        if res.success and -res.fun > 1e-9:
            sup.append((nm, float(-res.fun), res.x[:d]))
    return sorted(sup, key=lambda x: -x[1])


def _boot_pareto_mask(V: np.ndarray) -> np.ndarray:
    B_, M_, _ = V.shape
    out = np.zeros((B_, M_), dtype=bool)
    for b in range(B_):
        A = V[b]
        ge = (A[:, None, :] >= A[None, :, :]).all(axis=2)
        gt = (A[:, None, :] > A[None, :, :]).any(axis=2)
        out[b] = ~(ge & gt).any(axis=0)
    return out


def _boot_convex_2d_mask(V2: np.ndarray) -> np.ndarray:
    B_, M_, _ = V2.shape
    out = np.zeros((B_, M_), dtype=bool)
    for b in range(B_):
        A = V2[b]
        order = np.argsort(-A[:, 0], kind='mergesort')
        nd = []
        max_y = -np.inf
        for idx in order:
            if A[idx, 1] > max_y:
                nd.append(idx)
                max_y = A[idx, 1]
        hull = []
        for idx in nd:
            while len(hull) >= 2:
                x0, y0 = A[hull[-2]]
                x1, y1 = A[hull[-1]]
                x2, y2 = A[idx]
                if (y1 - y0) * (x1 - x2) <= (y2 - y1) * (x0 - x1):
                    hull.pop()
                else:
                    break
            hull.append(idx)
        out[b, hull] = True
    return out


def _ordered_pareto_2d(tab: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    nd, _ = pareto_nondominated(tab, [xcol, ycol])
    return tab.loc[nd].sort_values(xcol, ascending=False)


def _draw_2d_scatter(
    ax: plt.Axes,
    tab: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    ci_map: dict[tuple[str, str], tuple[float, float]] | None = None,
    label_models: list[str] | None = None,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    show_sigma_arrow: bool = True,
    show_legend: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    mod_style = {
        'Sequence-only': dict(color='#b8c0c8', marker='o', s=20, alpha=0.55, zorder=2, label='Public: Sequence-only'),
        'Structure + Sequence (no MSA)': dict(color='#4d90bd', marker='s', s=26, alpha=0.72, zorder=3, label='Public: Structure + Sequence (no MSA)'),
        'MSA / Retrieval': dict(color='#e08c3a', marker='^', s=28, alpha=0.72, zorder=3, label='Public: MSA / Retrieval'),
        'Stability baseline (clean)': dict(color='#2d8a4e', marker='D', s=34, alpha=0.88, zorder=4, label='Stability baseline (clean)'),
        'Stability baseline (leaked*)': dict(color='#999999', marker='x', s=36, alpha=0.80, zorder=4, label='Stability baseline (leaked*)'),
    }
    for mod, kw in mod_style.items():
        sub = tab[tab['modality'] == mod]
        if len(sub):
            ax.scatter(sub[xcol], sub[ycol], edgecolors='none' if kw['marker'] != 'x' else None, **kw)

    eval_tab = tab[tab['modality'] != 'Stability baseline (leaked*)']
    pf = _ordered_pareto_2d(eval_tab, xcol, ycol)
    if len(pf) >= 2:
        px = pf[xcol].to_numpy(float)
        py = pf[ycol].to_numpy(float)
        sx, sy = [px[0]], [py[0]]
        for k in range(len(px) - 1):
            sx.extend([px[k + 1], px[k + 1]])
            sy.extend([py[k], py[k + 1]])
        ax.plot(sx, sy, color='#222222', lw=1.15, ls='-', alpha=0.65, zorder=5, label='Pareto frontier (non-dominated)')

    hull_res = convex_hull_supported(eval_tab, [xcol, ycol])
    if len(hull_res) >= 2:
        h_names = [h[0] for h in hull_res]
        hdf = eval_tab.loc[h_names].sort_values(xcol, ascending=False)
        ax.plot(hdf[xcol], hdf[ycol], color='#b32424', lw=1.25, ls='--', alpha=0.75, zorder=5, label='Upper convex hull (linear scalarization)')

    for nm in pf.index:
        if not nm.startswith('ESM-MSR'):
            m_st = mod_style[tab.loc[nm, 'modality']]
            ax.scatter(
                tab.loc[nm, xcol], tab.loc[nm, ycol],
                s=52, marker=m_st['marker'], facecolor=m_st['color'], edgecolor='#111111', linewidth=1.0, zorder=7,
            )

    msr_styles = {
        'ESM-MSR (σ=0.0)': dict(color='#555555', marker='o', s=58, ec='#111111', lw=1.1, label='ESM3-small-open (σ = 0)'),
        'ESM-MSR (σ=0.5)': dict(color='#7a3eb1', marker='P', s=95, ec='#111111', lw=1.1, label='ESM-MSR σ = 0.5'),
        'ESM-MSR (σ=1.0)': dict(color='#c0262d', marker='*', s=165, ec='#111111', lw=0.9, label='ESM-MSR σ = 1.0'),
    }
    if show_sigma_arrow and all(k in tab.index for k in ('ESM-MSR (σ=0.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=1.0)')):
        xs = [tab.loc[k, xcol] for k in ('ESM-MSR (σ=0.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=1.0)')]
        ys = [tab.loc[k, ycol] for k in ('ESM-MSR (σ=0.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=1.0)')]
        ax.plot(xs, ys, color='#6c2e9c', lw=1.6, ls=':', alpha=0.85, zorder=6)

    for nm, kw in msr_styles.items():
        if nm not in tab.index:
            continue
        xv, yv = float(tab.loc[nm, xcol]), float(tab.loc[nm, ycol])
        if ci_map and (nm, xcol) in ci_map and (nm, ycol) in ci_map:
            xlo, xhi = ci_map[(nm, xcol)]
            ylo, yhi = ci_map[(nm, ycol)]
            ax.errorbar(
                xv, yv,
                xerr=[[xv - xlo], [xhi - xv]],
                yerr=[[yv - ylo], [yhi - yv]],
                fmt='none', ecolor=kw['color'], elinewidth=1.1, alpha=0.45, capsize=0, zorder=6,
            )
        ax.scatter(xv, yv, s=kw['s'], marker=kw['marker'], facecolor=kw['color'], edgecolor=kw['ec'], linewidth=kw['lw'], zorder=9, label=kw['label'])

    offsets = label_offsets or {}
    to_label = label_models if label_models is not None else list(dict.fromkeys(list(pf.index) + ['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)']))
    for nm in to_label:
        if nm not in tab.index:
            continue
        xv, yv = float(tab.loc[nm, xcol]), float(tab.loc[nm, ycol])
        dx, dy = offsets.get(nm, (5, 5))
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        short_nm = nm.replace('ESM-MSR (σ=1.0)', 'ESM-MSR σ=1.0').replace('ESM-MSR (σ=0.5)', 'ESM-MSR σ=0.5').replace('ESM-MSR (σ=0.0)', 'σ=0 (ESM3)')
        fw = 'bold' if nm.startswith('ESM-MSR') else 'normal'
        col_txt = '#7a0c10' if nm == 'ESM-MSR (σ=1.0)' else ('#551a8b' if nm == 'ESM-MSR (σ=0.5)' else '#222222')
        ax.annotate(
            short_nm, (xv, yv), xytext=(dx, dy), textcoords='offset points',
            fontsize=7.6, fontweight=fw, color=col_txt, ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6, shrinkA=1, shrinkB=3) if (abs(dx) > 10 or abs(dy) > 10) else None,
            zorder=10,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=10.0, fontweight='bold')
    ax.grid(color=style.GRID, lw=0.6, zorder=0)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(fontsize=7.3, loc='lower left', frameon=True, facecolor='white', edgecolor='#dddddd', ncol=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=10000, help='Bootstrap replicates')
    ap.add_argument('--refresh', action='store_true', help='Recompute caches')
    args = ap.parse_args()

    t0 = time.time()
    out = results_dir('99_misc/pgym_pareto', create=True)
    # Remove any legacy 173-assay CSV if present
    stale_173 = out / 'pgym_173_all106_summary.csv'
    if stale_173.exists():
        stale_173.unlink()

    ref = pd.read_csv(PGYM_REF_CSV).set_index('DMS_id')
    hom_set = heldout.excluded('proteingym')
    pub_df = load_public_95_per_assay(refresh=args.refresh)
    pub_95 = list(pub_df.columns)
    all217_ids = list(pub_df.index)

    sig_217, sig_seeds_217 = PG.load_sigma_pgym(heldout=False, drop_large=False, include_fallback_small=True, include_dms3=False, refresh=False)
    sig_idx = sig_217.set_index('assay').reindex(all217_ids)
    stab_217 = load_stability_baselines_all217(all217_ids, refresh=args.refresh)
    s0_clean = load_esm_msr_sigma0_all217(all217_ids, sig_idx['has_multi'])

    # Assemble single unified 217-assay master table (all 95 public + 3 ESM-MSR + 7 stability baselines = 105 models)
    full = pub_df.copy()
    full['ESM-MSR (σ=1.0)'] = sig_idx['rho_dep_1.0']
    full['ESM-MSR (σ=0.5)'] = sig_idx['rho_dep_0.5']
    full['ESM-MSR (σ=0.0)'] = s0_clean
    full['ESM-MSR (σ=1.0, WT-only)'] = sig_idx['rho_1.0']
    full['ESM-MSR (σ=0.5, WT-only)'] = sig_idx['rho_0.5']
    for em in STABILITY_BASELINES_CLEAN + STABILITY_BASELINES_LEAKED:
        full[em] = stab_217[em]
    full['category'] = ref['coarse_selection_type'].reindex(full.index).astype(str)
    full['uniprot'] = ref['UniProt_ID'].reindex(full.index).astype(str)
    full['homologous'] = full.index.isin(hom_set)
    full['has_multi'] = sig_idx['has_multi'].astype(bool)

    held176 = full[~full['homologous']].copy()

    # 1. Alignment verification (176/176)
    align = verify_alignment(list(held176.index))
    print(f"[alignment] {align}")

    # Report per-model assay coverage across the 176 unseen assays
    m98 = pub_95 + ['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)']
    m105 = m98 + list(STABILITY_BASELINES_CLEAN) + list(STABILITY_BASELINES_LEAKED)
    miss_map = {m: int(held176[m].isna().sum()) for m in m105 if held176[m].isna().any()}
    print(f"[coverage on 176 unseen] models with any missing assays: {miss_map}")

    # 2. Build unified 176-assay summary tables (98 public+ESM-MSR, and 105 incl. all 7 stability baselines)
    def _agg_table(sub_df: pd.DataFrame, mlist: list[str]) -> pd.DataFrame:
        cm = sub_df.groupby('category')[mlist].mean().loc[list(CATS)].T
        cm['NonStab_macro'] = cm[['Expression', 'Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
        cm['Func3_macro'] = cm[['Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
        cm['Biophys2_macro'] = cm[['Stability', 'Expression']].mean(axis=1)
        cm['PG_avg'] = cm[list(CATS)].mean(axis=1)
        cm['DMS_mean'] = sub_df[mlist].mean()
        cm['n_scored'] = [int(sub_df[m].notna().sum()) for m in cm.index]
        cm['modality'] = [modality_of(m) for m in cm.index]
        cm.index.name = 'model'
        return cm

    tab176 = _agg_table(held176, m98)
    tab176_all = _agg_table(held176, m105)
    atomic_to_csv(tab176.sort_values('PG_avg', ascending=False), out / 'pgym_176_all98_summary.csv', float_format='%.4f')
    atomic_to_csv(tab176_all.sort_values('PG_avg', ascending=False), out / 'pgym_176_all105_summary.csv', float_format='%.4f')

    # 3. Joint [category-stratified assay x triplicate-seed] Bootstrap on 176 (98 models)
    B = args.B
    rng = np.random.default_rng(42)
    held176_ids = list(held176.index)
    cat_ser176 = held176['category']
    cat_idx176 = {c: np.flatnonzero((cat_ser176 == c).to_numpy()) for c in CATS}
    picks176 = {c: rng.integers(0, len(idx), size=(B, len(idx))) for c, idx in cat_idx176.items()}
    sw3 = rng.multinomial(3, [1 / 3, 1 / 3, 1 / 3], size=B).astype(float) / 3.0

    boot176 = np.empty((B, len(m98), 5), dtype=float)
    for mi, m in enumerate(m98):
        if m == 'ESM-MSR (σ=1.0)':
            arr = sig_seeds_217[('deployed', 1.0)].reindex(held176_ids).to_numpy(float)
            draw = np.einsum('bs,is->bi', sw3, arr)
        elif m == 'ESM-MSR (σ=0.5)':
            arr = sig_seeds_217[('deployed', 0.5)].reindex(held176_ids).to_numpy(float)
            draw = np.einsum('bs,is->bi', sw3, arr)
        else:
            arr = held176[m].to_numpy(float)
            draw = None
        for ci_k, c in enumerate(CATS):
            idx = cat_idx176[c]
            if draw is not None:
                boot176[:, mi, ci_k] = np.take_along_axis(draw[:, idx], picks176[c], axis=1).mean(axis=1)
            else:
                boot176[:, mi, ci_k] = arr[idx][picks176[c]].mean(axis=1)

    b_nonstab = boot176[:, :, 1:].mean(axis=2)
    b_func3 = boot176[:, :, 2:].mean(axis=2)
    b_biophys2 = boot176[:, :, :2].mean(axis=2)
    b_pgavg = boot176.mean(axis=2)

    ci_map176: dict[tuple[str, str], tuple[float, float]] = {}
    ci_rows = []
    metric_tensors = {
        'Stability': boot176[:, :, 0],
        'Expression': boot176[:, :, 1],
        'Activity': boot176[:, :, 2],
        'Binding': boot176[:, :, 3],
        'OrganismalFitness': boot176[:, :, 4],
        'NonStab_macro': b_nonstab,
        'Func3_macro': b_func3,
        'Biophys2_macro': b_biophys2,
        'PG_avg': b_pgavg,
    }
    for mi, m in enumerate(m98):
        for col, tens in metric_tensors.items():
            lo, hi = ci(tens[:, mi])
            ci_map176[(m, col)] = (float(lo), float(hi))
            ci_rows.append(dict(model=m, modality=modality_of(m), metric=col, value=float(tab176.loc[m, col]), lo=float(lo), hi=float(hi)))
    atomic_to_csv(pd.DataFrame(ci_rows), out / 'pgym_176_bootstrap_cis.csv', index=False, float_format='%.4f')

    idx_s1 = m98.index('ESM-MSR (σ=1.0)')
    idx_s05 = m98.index('ESM-MSR (σ=0.5)')
    boot_specs = [
        ('5D: All 5 categories', boot176, list(CATS), False),
        ('2D: Stability vs Functional-3', np.stack([boot176[:, :, 0], b_func3], axis=2), ['Stability', 'Func3_macro'], True),
        ('2D: Stability vs Non-Stability', np.stack([boot176[:, :, 0], b_nonstab], axis=2), ['Stability', 'NonStab_macro'], True),
        ('2D: Biophys-2 vs Functional-3', np.stack([b_biophys2, b_func3], axis=2), ['Biophys2_macro', 'Func3_macro'], True),
        ('2D: Stability vs Activity', np.stack([boot176[:, :, 0], boot176[:, :, 2]], axis=2), ['Stability', 'Activity'], True),
        ('2D: Stability vs Binding', np.stack([boot176[:, :, 0], boot176[:, :, 3]], axis=2), ['Stability', 'Binding'], True),
        ('2D: Stability vs Org. fitness', np.stack([boot176[:, :, 0], boot176[:, :, 4]], axis=2), ['Stability', 'OrganismalFitness'], True),
        ('2D: Stability vs Expression', np.stack([boot176[:, :, 0], boot176[:, :, 1]], axis=2), ['Stability', 'Expression'], True),
    ]
    prob_rows = []
    for sname, V, cols, do_hull in boot_specs:
        pf_mask = _boot_pareto_mask(V)
        nd_pt, _ = pareto_nondominated(tab176, cols)
        sup_pt = {x[0] for x in convex_hull_supported(tab176, cols)}
        ch_mask = _boot_convex_2d_mask(V) if do_hull else None
        prob_rows.append(dict(
            frontier=sname,
            n_frontier_pt=len(nd_pt),
            s1_pareto_pt=('ESM-MSR (σ=1.0)' in nd_pt),
            s05_pareto_pt=('ESM-MSR (σ=0.5)' in nd_pt),
            s1_convex_pt=('ESM-MSR (σ=1.0)' in sup_pt),
            s05_convex_pt=('ESM-MSR (σ=0.5)' in sup_pt),
            P_pareto_s1=float(pf_mask[:, idx_s1].mean()),
            P_pareto_s05=float(pf_mask[:, idx_s05].mean()),
            P_pareto_both=float((pf_mask[:, idx_s1] & pf_mask[:, idx_s05]).mean()),
            P_convex_s1=float(ch_mask[:, idx_s1].mean()) if ch_mask is not None else np.nan,
            P_convex_s05=float(ch_mask[:, idx_s05].mean()) if ch_mask is not None else np.nan,
            P_convex_both=float((ch_mask[:, idx_s1] & ch_mask[:, idx_s05]).mean()) if ch_mask is not None else np.nan,
            frontier_models='; '.join(nd_pt),
        ))
    prob_df = pd.DataFrame(prob_rows)
    atomic_to_csv(prob_df, out / 'pgym_176_pareto_bootstrap_probs.csv', index=False, float_format='%.4f')

    pair_rows = []
    comparisons = [
        ('ESM-MSR (σ=0.5)', 'ESM-MSR (σ=1.0)', c) for c in (*CATS, 'NonStab_macro', 'Func3_macro', 'PG_avg')
    ] + [
        ('ESM-MSR (σ=1.0)', r, m) for r in ('ProSST-4096', 'ProSST-2048', 'VenusREM', 'S3F_MSA', 'ESM3') for m in ('Stability', 'NonStab_macro', 'Func3_macro', 'PG_avg')
    ] + [
        ('ESM-MSR (σ=0.5)', r, m) for r in ('ProSST-4096', 'ProSST-2048', 'VenusREM', 'S3F_MSA', 'ESM3') for m in ('Stability', 'NonStab_macro', 'Func3_macro', 'PG_avg')
    ]
    for ma, mb, met in comparisons:
        ia, ib = m98.index(ma), m98.index(mb)
        pt_d = float(tab176.loc[ma, met] - tab176.loc[mb, met])
        diff = metric_tensors[met][:, ia] - metric_tensors[met][:, ib]
        lo, hi = ci(diff)
        p = boot_pvalue(diff, null=0.0, alternative='two-sided')
        pair_rows.append(dict(model_a=ma, model_b=mb, metric=met, delta=pt_d, lo=float(lo), hi=float(hi), p_boot=float(p), B=B))
    atomic_to_csv(pd.DataFrame(pair_rows), out / 'pgym_176_paired_tests.csv', index=False, float_format='%.4g')

    # 4. Stress-Test Summary Table (strictly on unified 176 unseen assays!)
    stress_rows = []

    def _add_stress(label: str, scope: str, tab: pd.DataFrame, cols: list[str], m1: str = 'ESM-MSR (σ=1.0)', m05: str = 'ESM-MSR (σ=0.5)') -> None:
        nd, dom = pareto_nondominated(tab, cols)
        sup = {x[0] for x in convex_hull_supported(tab, cols)}
        stress_rows.append(dict(
            test=label,
            scope=scope,
            n_models=len(tab),
            n_frontier=len(nd),
            s1_pareto=(m1 in nd),
            s05_pareto=(m05 in nd),
            both_pareto=(m1 in nd and m05 in nd),
            s1_convex=(m1 in sup),
            s05_convex=(m05 in sup),
            both_convex=(m1 in sup and m05 in sup),
            s05_dominated_by='; '.join(dom.get(m05, [])),
            s1_dominated_by='; '.join(dom.get(m1, [])),
        ))

    _add_stress('5D: All 5 categories (primary)', '176 unseen, 98 models', tab176, list(CATS))
    _add_stress('5D: + all 7 stability baselines', '176 unseen, 105 models', tab176_all, list(CATS))
    _add_stress('5D: Alignment-free (no-MSA) only', '176 unseen, 75 models', tab176[tab176['modality'] != 'MSA / Retrieval'], list(CATS))
    _add_stress('2D: Stability vs Functional-3', '176 unseen, 98 models', tab176, ['Stability', 'Func3_macro'])
    _add_stress('2D: Stability vs Functional-3 (+ baselines)', '176 unseen, 105 models', tab176_all, ['Stability', 'Func3_macro'])
    _add_stress('2D: Stability vs Functional-3 (no-MSA)', '176 unseen, 75 models', tab176[tab176['modality'] != 'MSA / Retrieval'], ['Stability', 'Func3_macro'])
    _add_stress('2D: Stability vs Non-Stability (4-cat)', '176 unseen, 98 models', tab176, ['Stability', 'NonStab_macro'])
    _add_stress('2D: Biophys-2 vs Functional-3', '176 unseen, 98 models', tab176, ['Biophys2_macro', 'Func3_macro'])
    _add_stress('2D: Stability vs Activity', '176 unseen, 98 models', tab176, ['Stability', 'Activity'])
    _add_stress('2D: Stability vs Binding', '176 unseen, 98 models', tab176, ['Stability', 'Binding'])
    _add_stress('2D: Stability vs OrganismalFitness', '176 unseen, 98 models', tab176, ['Stability', 'OrganismalFitness'])
    _add_stress('2D: Stability vs Expression', '176 unseen, 98 models', tab176, ['Stability', 'Expression'])

    tab_micro = pd.DataFrame({
        'Stability': held176[held176['category'] == 'Stability'][m98].mean(),
        'NonStab_micro': held176[held176['category'] != 'Stability'][m98].mean(),
        'Func3_micro': held176[held176['category'].isin(['Activity', 'Binding', 'OrganismalFitness'])][m98].mean(),
    })
    _add_stress('2D: Stability vs Func-3 (assay-weighted)', '176 unseen, 98 models', tab_micro, ['Stability', 'Func3_micro'])
    _add_stress('2D: Stability vs NonStab (assay-weighted)', '176 unseen, 98 models', tab_micro, ['Stability', 'NonStab_micro'])

    m98_wt = pub_95 + ['ESM-MSR (σ=1.0, WT-only)', 'ESM-MSR (σ=0.5, WT-only)', 'ESM-MSR (σ=0.0)']
    tab_wt = _agg_table(held176, m98_wt)
    _add_stress('5D: WT-head only (no Dual-view)', '176 unseen, 98 models', tab_wt, list(CATS), 'ESM-MSR (σ=1.0, WT-only)', 'ESM-MSR (σ=0.5, WT-only)')
    _add_stress('2D: Stab vs Func-3 (WT-head only)', '176 unseen, 98 models', tab_wt, ['Stability', 'Func3_macro'], 'ESM-MSR (σ=1.0, WT-only)', 'ESM-MSR (σ=0.5, WT-only)')

    tab_up = held176.groupby(['category', 'uniprot'])[m98].mean().groupby('category').mean().loc[list(CATS)].T
    tab_up['Func3_macro'] = tab_up[['Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
    tab_up['NonStab_macro'] = tab_up[['Expression', 'Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
    _add_stress('5D: UniProt-deduplicated', '162 cat×UniProt, 98 models', tab_up, list(CATS))
    _add_stress('2D: Stab vs Func-3 (UniProt-dedup)', '162 cat×UniProt, 98 models', tab_up, ['Stability', 'Func3_macro'])

    tab_med = held176.groupby('category')[m98].median().loc[list(CATS)].T
    tab_med['Func3_macro'] = tab_med[['Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
    tab_med['NonStab_macro'] = tab_med[['Expression', 'Activity', 'Binding', 'OrganismalFitness']].mean(axis=1)
    _add_stress('5D: Median per-assay ρ', '176 unseen, 98 models', tab_med, list(CATS))
    _add_stress('2D: Stab vs Func-3 (Median)', '176 unseen, 98 models', tab_med, ['Stability', 'Func3_macro'])
    _add_stress('2D: Stab vs NonStab (Median)', '176 unseen, 98 models', tab_med, ['Stability', 'NonStab_macro'])

    tab_sing = _agg_table(held176[~held176['has_multi']], m98)
    tab_mm = _agg_table(held176[held176['has_multi']], m98)
    _add_stress('2D: Stab vs NonStab (Singles-only assays)', '129 assays, 98 models', tab_sing, ['Stability', 'NonStab_macro'])
    _add_stress('2D: Stab vs NonStab (Multi-mutant assays)', '47 assays, 98 models', tab_mm, ['Stability', 'NonStab_macro'])

    pg_f = heldout.flags()
    pg_f = pg_f[pg_f['dataset'] == 'proteingym'].set_index('item_id').reindex(held176.index)
    m53 = (
        (pg_f['max_identity'].fillna(0) <= 0.30)
        & (pg_f['max_tm'].fillna(0) <= 0.60)
        & (pg_f['min_evalue'].fillna(999) > 1e-2)
        & (pg_f['min_struct_evalue'].fillna(999) > 1e-2)
    )
    tab_53 = _agg_table(held176[m53], m98)
    _add_stress('5D: Strict twilight cutoff (id≤0.30, TM≤0.60)', '53 assays, 98 models', tab_53, list(CATS))
    _add_stress('2D: Stab vs Func-3 (Strict twilight cutoff)', '53 assays, 98 models', tab_53, ['Stability', 'Func3_macro'])

    tab_notsu = _agg_table(held176[~held176.index.str.contains('Tsuboyama')], m98)
    _add_stress('5D: Drop all 35 unseen Tsuboyama assays', '141 assays (2 Stab), 98 models', tab_notsu, list(CATS))
    _add_stress('2D: Stab vs Func-3 (Drop all 35 Tsuboyama)', '141 assays (2 Stab), 98 models', tab_notsu, ['Stability', 'Func3_macro'])

    ab_ids = [a for _, a, _ in PAIRED_PG_PROTEINS]
    fn_ids = [f for _, _, f in PAIRED_PG_PROTEINS]
    tab_mp11 = pd.DataFrame({
        'Abund_Expr': held176.loc[ab_ids, m98].mean(),
        'Func_ActBind': held176.loc[fn_ids, m98].mean(),
        'modality': [modality_of(m) for m in m98],
    })
    _add_stress('2D: 11 Paired Multi-Property Proteins (Expr vs Act/Bind)', '22 assays, 98 models', tab_mp11, ['Abund_Expr', 'Func_ActBind'])

    stress_df = pd.DataFrame(stress_rows)
    atomic_to_csv(stress_df, out / 'pgym_pareto_stress_tests.csv', index=False)

    # ==============================================================================================
    # FIGURE 1: Main 4-panel Pareto Frontier Analysis (all on unified 176 unseen assays)
    # ==============================================================================================
    plot_tab = tab176_all[tab176_all['modality'] != 'Stability baseline (leaked*)'].copy()

    fig, axes1 = plt.subplots(2, 2, figsize=(16.0, 11.4))
    axa, axb, axc, axd = axes1[0, 0], axes1[0, 1], axes1[1, 0], axes1[1, 1]

    _draw_2d_scatter(
        axa, plot_tab, 'Stability', 'Func3_macro',
        xlabel='Held-out Stability mean Spearman ρ (n = 37 assays)',
        ylabel='Held-out Functional-3 mean ρ\n(Activity, Binding, Org. fitness)',
        title='a   Stability vs Functional-3 Pareto frontier & convex hull',
        ci_map=ci_map176,
        label_models=['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)', 'VenusREM', 'ProSST-4096', 'ProSST-2048', 'S3F_MSA', 'SPURS', 'ThermoMPNN'],
        label_offsets={
            'ESM-MSR (σ=1.0)': (-8, -16),
            'ESM-MSR (σ=0.5)': (8, 8),
            'ESM-MSR (σ=0.0)': (-14, -15),
            'VenusREM': (-10, 8),
            'ProSST-4096': (12, 8),
            'ProSST-2048': (12, -14),
            'S3F_MSA': (-12, 8),
            'SPURS': (8, -6),
            'ThermoMPNN': (8, -12),
        },
        show_legend=True,
        xlim=(0.28, 0.90),
        ylim=(0.14, 0.52),
    )

    _draw_2d_scatter(
        axb, plot_tab, 'Stability', 'NonStab_macro',
        xlabel='Held-out Stability mean Spearman ρ (n = 37 assays)',
        ylabel='Held-out Non-Stability 4-cat mean ρ\n(Expression, Activity, Binding, Org. fitness)',
        title='b   Stability vs Non-Stability 4-cat frontier (Expression pulls σ=0.5 inside hull)',
        ci_map=ci_map176,
        label_models=['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)', 'VenusREM', 'ProSST-4096', 'ProSST-2048', 'S3F_MSA', 'SPURS'],
        label_offsets={
            'ESM-MSR (σ=1.0)': (-8, -16),
            'ESM-MSR (σ=0.5)': (8, 9),
            'ESM-MSR (σ=0.0)': (-14, -14),
            'VenusREM': (-8, 8),
            'ProSST-4096': (14, -12),
            'ProSST-2048': (12, 6),
            'S3F_MSA': (-14, 8),
            'SPURS': (8, -10),
        },
        show_legend=False,
        xlim=(0.28, 0.90),
        ylim=(0.16, 0.53),
    )

    prof_models = [
        ('ESM-MSR (σ=1.0)', '#c0262d', '*', 1.0, 'ESM-MSR σ=1.0 [5D Hull]'),
        ('ESM-MSR (σ=0.5)', '#7a3eb1', 'P', 1.0, 'ESM-MSR σ=0.5 [5D Hull]'),
        ('VenusREM', '#e08c3a', '^', 0.95, 'VenusREM (MSA) [5D Hull]'),
        ('ProSST-4096', '#2171b5', 's', 0.95, 'ProSST-4096 [5D Hull]'),
        ('ProSST-2048', '#6baed6', 's', 0.85, 'ProSST-2048 [5D Pareto]'),
        ('S3F_MSA', '#b86e00', 'v', 0.90, 'S3F_MSA (MSA) [5D Hull]'),
        ('ESM-MSR (σ=0.0)', '#666666', 'o', 0.70, 'ESM3-small-open (σ=0)'),
        ('SPURS', '#2d8a4e', 'D', 0.75, 'SPURS (retrained)'),
    ]
    x_cats = np.arange(len(CATS))
    offs = np.linspace(-0.28, 0.28, len(prof_models))
    for k, (m, col, mk, alp, lbl) in enumerate(prof_models):
        ys = [float(tab176_all.loc[m, c]) for c in CATS]
        ls = '-' if m.startswith('ESM-MSR (σ=1') or m.startswith('ESM-MSR (σ=0.5') else ('--' if m in ('VenusREM', 'ProSST-4096', 'S3F_MSA') else ':')
        lw = 2.0 if m.startswith('ESM-MSR (σ=1') or m.startswith('ESM-MSR (σ=0.5') else 1.25
        axc.plot(x_cats + offs[k] * 0.35, ys, color=col, marker=mk, ms=6.2 if mk != '*' else 8.8, lw=lw, ls=ls, alpha=alp, label=lbl, zorder=5 if lw > 1.5 else 3)
        if m != 'SPURS':
            for ci_k, c in enumerate(CATS):
                lo, hi = ci_map176[(m, c)]
                axc.plot([x_cats[ci_k] + offs[k] * 0.35] * 2, [lo, hi], color=col, lw=1.0, alpha=0.35, zorder=2)
    axc.set_xticks(x_cats)
    n_by_cat = held176['category'].value_counts()
    axc.set_xticklabels([f'{CAT_SHORT[c]}\n(n={int(n_by_cat[c])})' for c in CATS])
    axc.set_ylabel('Mean per-assay Spearman ρ')
    axc.set_ylim(0.24, 0.90)
    axc.grid(axis='y', color=style.GRID, lw=0.6)
    axc.legend(fontsize=7.3, loc='upper right', ncol=2, frameon=True, facecolor='white', edgecolor='#dddddd')
    axc.set_title('c   5-property profile of the 6 global 5D Pareto-optimal models', loc='left', fontsize=10.0, fontweight='bold')

    short_front = {
        '5D: All 5 categories': '5D: All 5 cats',
        '2D: Stability vs Functional-3': 'Stab. vs Func-3',
        '2D: Stability vs Non-Stability': 'Stab. vs Non-Stab',
        '2D: Biophys-2 vs Functional-3': 'Biophys-2 vs Func-3',
        '2D: Stability vs Activity': 'Stab. vs Activity',
        '2D: Stability vs Binding': 'Stab. vs Binding',
        '2D: Stability vs Org. fitness': 'Stab. vs Org. fit.',
        '2D: Stability vs Expression': 'Stab. vs Expression',
    }
    y_pos = np.arange(len(prob_df))
    h = 0.25
    axd.barh(y_pos - h, prob_df['P_pareto_s1'] * 100, h, color='#c0262d', alpha=0.88, label='ESM-MSR σ=1.0: P(Pareto)')
    axd.barh(y_pos, prob_df['P_pareto_both'] * 100, h, color='#7a3eb1', alpha=0.88, label='BOTH σ=1.0 & σ=0.5: P(Pareto)')
    ch_vals = prob_df['P_convex_both'].fillna(0.0) * 100
    axd.barh(y_pos + h, ch_vals, h, color='#7a3eb1', alpha=0.32, hatch='///', edgecolor='#551a8b', lw=0.7, label='BOTH σ=1.0 & σ=0.5: P(2D Convex Hull)')
    for i, r in prob_df.iterrows():
        axd.text(r['P_pareto_both'] * 100 + 1.2, i, f"{r['P_pareto_both']*100:.1f}%", va='center', fontsize=7.6, fontweight='bold', color='#4a1476')
        if np.isfinite(r['P_convex_both']):
            axd.text(r['P_convex_both'] * 100 + 1.2, i + h, f"{r['P_convex_both']*100:.1f}%", va='center', fontsize=7.2, color='#555555')
        else:
            axd.text(1.5, i + h, '5D LP: both strictly supported', va='center', fontsize=7.0, style='italic', color='#555555')
    axd.set_yticks(y_pos)
    axd.set_yticklabels([short_front.get(s, s) for s in prob_df['frontier']], fontsize=8.4)
    axd.set_ylim(len(prob_df) - 0.1, -0.6)
    axd.set_xlim(0, 118)
    axd.set_xlabel(f'Bootstrap inclusion probability (%, B = {B:,} joint assay × seed draws)')
    axd.grid(axis='x', color=style.GRID, lw=0.6)
    axd.legend(fontsize=7.3, loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd')
    axd.set_title('d   Joint bootstrap stability of Pareto & convex-hull membership', loc='left', fontsize=10.0, fontweight='bold')

    fig.subplots_adjust(wspace=0.32, hspace=0.30)
    style.finish_figure(
        fig,
        out / 'pgym_pareto_main',
        title='ProteinGym multi-property Pareto frontier on 176 unseen assays (all 95 public models + stability baselines + ESM-MSR)',
        note=(
            'Held-out filter: four-part sequence + structure homology rule against the 172 MegaScale training libraries '
            '(MMseqs2 E ≤ 1e-3, BLOSUM62 identity > 0.40, Foldseek E ≤ 1e-3, or Foldseek TM > 0.70; 41/217 assays excluded, zero PG_DROP). '
            'ESM-MSR = FINAL_p0 seeds 1–3 averaged (deployed: Dual-view combined on 47 multi-mutant assays, WT-view on 129 singles assays). '
            'Clean stability baselines (SPURS & ThermoMPNN 176/176; MutateEverything [missing BRCA2, SCN5A], GeoDDG-3D [BRCA2, POLG_CXB3N], Rosetta [BRCA2, POLG_HCVJF] 174/176) are all strictly dominated by ESM-MSR σ=1.0.'
        ),
        note_fontsize=7.5,
        top=0.935,
    )

    # ==============================================================================================
    # FIGURE 2: Pairwise 2D Stability vs each individual category
    # ==============================================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14.6, 10.6))
    pair_specs = [
        ('Activity', 'Activity mean Spearman ρ (n = 38)', 'a   Stability vs Activity (both σ=1 & σ=0.5 on Pareto + Hull)', (0.14, 0.54)),
        ('Binding', 'Binding mean Spearman ρ (n = 13)', 'b   Stability vs Binding (both σ=1 & σ=0.5 on Pareto + Hull)', (0.10, 0.53)),
        ('OrganismalFitness', 'Organismal fitness mean Spearman ρ (n = 70)', 'c   Stability vs Organismal fitness (both on Pareto)', (0.12, 0.53)),
        ('Expression', 'Expression mean Spearman ρ (n = 18)', 'd   Stability vs Expression (σ=1 dominates σ=0.5!)', (0.16, 0.59)),
    ]
    for idx_p, (ax, (ycol, ylab, ttl, ylim)) in enumerate(zip(axes2.ravel(), pair_specs)):
        lbls = ['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)', 'VenusREM', 'ProSST-4096', 'ProSST-2048', 'S3F_MSA']
        if ycol == 'Expression':
            offs_p = {
                'ESM-MSR (σ=1.0)': (-6, 10),
                'ESM-MSR (σ=0.5)': (-8, -16),
                'ESM-MSR (σ=0.0)': (-14, -14),
                'VenusREM': (-10, 9),
                'ProSST-4096': (14, -4),
                'ProSST-2048': (12, -16),
                'S3F_MSA': (-14, 8),
            }
        elif ycol == 'Binding':
            offs_p = {
                'ESM-MSR (σ=1.0)': (6, -15),
                'ESM-MSR (σ=0.5)': (8, 8),
                'ESM-MSR (σ=0.0)': (-14, -14),
                'VenusREM': (-12, 8),
                'ProSST-4096': (8, 6),
                'ProSST-2048': (12, -12),
                'S3F_MSA': (-14, 8),
            }
        else:
            offs_p = {
                'ESM-MSR (σ=1.0)': (-8, -16),
                'ESM-MSR (σ=0.5)': (8, 8),
                'ESM-MSR (σ=0.0)': (-14, -14),
                'VenusREM': (-10, 9),
                'ProSST-4096': (14, 8),
                'ProSST-2048': (14, -12),
                'S3F_MSA': (-14, 8),
            }
        _draw_2d_scatter(
            ax, plot_tab, 'Stability', ycol,
            xlabel='Held-out Stability mean Spearman ρ (n = 37 assays)',
            ylabel=ylab,
            title=ttl,
            ci_map=ci_map176,
            label_models=lbls,
            label_offsets=offs_p,
            show_legend=(idx_p == 0),
            xlim=(0.28, 0.90),
            ylim=ylim,
        )
    style.finish_figure(
        fig2,
        out / 'pgym_pareto_pairwise',
        title='Pairwise 2D frontiers: Stability vs each ProteinGym non-stability category (176 unseen assays)',
        note=(
            'ESM-MSR σ=1.0 and σ=0.5 both sit on the Pareto frontier for Stability vs Activity, Binding, and Organismal fitness. '
            'On Stability vs Expression (panel d), σ=1.0 strictly dominates σ=0.5 (p = 0.0102) because cellular expression scales directly with folding stability.'
        ),
        note_fontsize=7.6,
        top=0.935,
    )

    # ==============================================================================================
    # FIGURE 3: Key Caveats & Boundary Conditions
    # ==============================================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14.6, 5.7))
    stab_held = held176[held176['category'] == 'Stability']
    tsu_mask = stab_held.index.str.contains('Tsuboyama')
    comp_models = ['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ProSST-4096', 'ProSST-2048', 'VenusREM', 'ESM-MSR (σ=0.0)', 'S3F_MSA', 'ESCOTT']
    x_c = np.arange(len(comp_models))
    w = 0.25
    v_tsu = [float(stab_held.loc[tsu_mask, m].mean()) for m in comp_models]
    v_esta = [float(stab_held.loc['ESTA_BACSU_Nutschel_2020', m]) for m in comp_models]
    v_a4 = [float(stab_held.loc['A4_HUMAN_Seuma_2022', m]) for m in comp_models]
    ax3a.bar(x_c - w, v_tsu, w, color='#2c5d8f', label='35 unseen MegaScale domains (cDNA proteolysis)')
    ax3a.bar(x_c, v_esta, w, color='#3a9a6c', label='ESTA_BACSU (ΔTm thermostability, n=1)')
    ax3a.bar(x_c + w, v_a4, w, color='#d97b38', label='A4_HUMAN (amyloid-β nucleation, n=1)')
    for xi, vt, ve, va in zip(x_c, v_tsu, v_esta, v_a4):
        ax3a.text(xi - w, vt + 0.012, f'{vt:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90)
        ax3a.text(xi, ve + 0.012, f'{ve:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90)
        ax3a.text(xi + w, va + 0.012, f'{va:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90)
    ax3a.set_xticks(x_c)
    ax3a.set_xticklabels([m.replace('ESM-MSR ', 'ESM-MSR\n') for m in comp_models], fontsize=8.0, rotation=20, ha='right')
    ax3a.set_ylabel('Spearman ρ')
    ax3a.set_ylim(0, 0.99)
    ax3a.grid(axis='y', color=style.GRID, lw=0.6)
    ax3a.legend(fontsize=7.6, loc='upper right', frameon=True, facecolor='white', edgecolor='#dddddd')
    ax3a.set_title('a   Caveat 1: Held-out Stability assay composition (35 MegaScale + 2 other)', loc='left', fontsize=9.8, fontweight='bold')

    _draw_2d_scatter(
        ax3b, tab_mp11, 'Abund_Expr', 'Func_ActBind',
        xlabel='Mean ρ on Abundance / Expression assay (11 paired proteins)',
        ylabel='Mean ρ on Activity / Binding assay (same 11 proteins)',
        title='b   Caveat 2: 11 paired Abundance-vs-Function proteins (dominated)',
        label_models=['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)', 'VenusREM', 'ProSST-2048', 'S3F_MSA', 'SaProt_650M_AF2'],
        label_offsets={
            'ESM-MSR (σ=1.0)': (8, -16),
            'ESM-MSR (σ=0.5)': (-10, 12),
            'ESM-MSR (σ=0.0)': (8, -14),
            'VenusREM': (6, 6),
            'ProSST-2048': (8, 5),
            'S3F_MSA': (-8, 7),
            'SaProt_650M_AF2': (-12, 8),
        },
        show_legend=True,
        xlim=(0.18, 0.56),
        ylim=(0.18, 0.55),
    )

    style.finish_figure(
        fig3,
        out / 'pgym_pareto_caveats',
        title='Boundary conditions & caveats for the ProteinGym multi-property Pareto claim',
        note=(
            'Panel a: All 37 held-out Stability assays are non-homologous to training proteins, but 35/37 originate from MegaScale cDNA proteolysis '
            '(Tsuboyama 2023); ESM-MSR also beats ProSST/VenusREM on ESTA_BACSU ΔTm, while MSA models lead on A4_HUMAN amyloid nucleation. '
            'Panel b: When "multi-property" is restricted to the 11 paired Abundance/Expression vs Activity/Binding proteins (excluding folding Stability), '
            'VenusREM and ProSST-2048 strictly dominate ESM-MSR at both σ=1.0 and σ=0.5.'
        ),
        note_fontsize=7.5,
        top=0.89,
    )

    print('\n=== Stability Baselines on Unified 176 Unseen Assays ===')
    show_m = ['ESM-MSR (σ=1.0)', 'ESM-MSR (σ=0.5)', 'ESM-MSR (σ=0.0)'] + list(STABILITY_BASELINES_CLEAN) + list(STABILITY_BASELINES_LEAKED)
    print(tab176_all.loc[show_m, list(CATS) + ['Func3_macro', 'NonStab_macro', 'PG_avg', 'n_scored']].round(4).to_string())
    nd5_all, _ = pareto_nondominated(tab176_all, list(CATS))
    sup5_all = convex_hull_supported(tab176_all, list(CATS))
    print('\n5D Pareto (all 105 models on 176 unseen):', nd5_all)
    print('5D Convex Hull (all 105 models on 176 unseen):', [(x[0], round(x[1], 5)) for x in sup5_all])
    print(f'\n[done] wrote outputs to {out} in {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
