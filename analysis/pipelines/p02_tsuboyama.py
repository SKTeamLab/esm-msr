#!/usr/bin/env python3
"""Pipeline p02: Tsuboyama / MegaScale rankings, NDCG@96, ESM3 scoring variants, and mega_splits comparison.

Consolidates:
  - 03_tsuboyama.py                          -> results/01_megascale/{megascale_overview, megascale_per_library, megascale_singles_doubles, megascale_scatter}
  - 11_publication_figures.py (fig1, fig2*)  -> results/01_megascale/{best_other_ddg_model.csv, ndcg96_singles_doubles*, spearman_stab_destab*}
  - 11e_esm3_scoring_variants.py (fig2 part) -> results/01_megascale/esm3_scoring_variants
  - 11i_doubles_vs_singles_pvalues.py        -> results/01_megascale/doubles_vs_singles_on_singles.{csv,md,png,pdf}
  - 11j_overall_vs_transfer.py               -> results/01_megascale/overall_vs_transfer_{domain_mean,pooled}.{csv,md,png,pdf}
  - 11k_ndcg96_supplement.py                 -> results/00_supplementary_figs/ndcg96_singles_doubles.{png,pdf} + *_values.csv
  - tsuboyama/mega_splits/{01,03}            -> results/01_megascale/mega_splits/*

Key methodological fixes vs legacy scripts:
  - Parameterizes split selection explicitly (--split capped|mega and --compare-splits) with zero MSR_TSU_SPLIT env mutation.
  - Fixes 11j pooled Spearman bootstrap to resample the 3 training seeds jointly with domains and mutations via SpearmanEngine.
  - Fixes 11k docstring/implementation alignment (shifted NDCG@96: relevance = y - min(y) per domain/subset).
"""
from __future__ import annotations

from itertools import combinations
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from msr_analysis import models as M
from msr_analysis import style
from msr_analysis.boot import SpearmanEngine, bootstrap_rho, ci
from msr_analysis.common import leak_table, load_rho_df, subframe
from msr_analysis.data import Frame, _align_id, load_frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.paths import CORE_RHO_BOOTSTRAP_CSV, atomic_to_csv, atomic_write_text, results_dir
from msr_analysis.registry import MEGASCALE_NOTE, MEGASCALE_TRAINED, MODELS, TSU_CAPPED, TSU_MEGA
from msr_analysis.stats import fmt_p, holm
from msr_analysis.tobit import linked

from . import base_parser, resolve_b
from ._tsu_common import (
    BEST_LABEL,
    F_COMPOSITE,
    K_NDCG,
    MIN_N,
    add_esm3_marginal_additive,
    add_holm_by_group,
    best_other,
    boot_reps,
    col_spec,
    compare_grouped,
    compare_pooled,
    count_text,
    diff_df,
    fig_width,
    finish_bar_fig,
    load_tsu_test_frame,
    main_columns,
    make_domain_counts,
    ndcg96_rectified,
    ndcg96_shifted,
    per_library,
    rho_fn,
    split_title,
    table_grouped,
    table_pooled,
)

STAGE = 'tsuboyama'
DESCRIPTION = 'Tsuboyama / MegaScale rankings, NDCG@96, ESM3 scoring variants, and mega_splits comparison'

CAPPED_KEYS = list(dict.fromkeys(
    k for m in M.ROSTER if not m.sigma for k in (m.single, m.multi) if k
))
RETR_KEYS = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
MEGA_EXTRA = ['spursu', 'spursum', 'tmpnnpub', 'tmpnndpub']
MEGA_BASE = ['msrp0:wt', 'msrp0:comb', *RETR_KEYS]
MEGA_KEYS = MEGA_BASE + MEGA_EXTRA


def _stars(p: float) -> str:
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def _render_stat_table(
    rows: list[list[str]],
    headers: list[str],
    widths: list[float],
    out_stem,
    note: str,
    cell_style_fn=None,
) -> None:
    fig = plt.figure(figsize=(sum(widths) + 0.4, 0.42 * (len(rows) + 1.6) + 1.0))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 0.8 / fig.get_figheight(), sum(widths) / fig.get_figwidth(), 1 - 0.88 / fig.get_figheight()])
    ax.axis('off')
    t = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='upper left',
        cellLoc='center',
        colLoc='center',
        colWidths=[w / sum(widths) for w in widths],
        bbox=[0, 0, 1, 1],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#cccccc')
        c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
            continue
        if j == 0:
            c.set_text_props(ha='left', fontweight='bold')
            c._loc = 'left'
            c.PAD = 0.04
        if cell_style_fn is not None:
            cell_style_fn(i - 1, j, c)
    fig.text(0.2 / fig.get_figwidth(), 0.68 / fig.get_figheight(), note, fontsize=8.5, color='#444444', va='top', ha='left')
    style.save(fig, out_stem)


def _write_md_table(rows: list[list[str]], headers: list[str], note: str, path) -> None:
    lines = (
        ['| ' + ' | '.join(h.replace('\n', ' ') for h in headers) + ' |']
        + ['|' + '---|' * len(headers)]
        + ['| ' + ' | '.join(r) + ' |' for r in rows]
    )
    atomic_write_text(path, '\n'.join(lines) + '\n\n' + note.replace('\n', ' ').strip() + '\n')


# ----------------------------------------------------------------------------------------------
# 1. Core Tsuboyama overview, per-library scatters, singles/doubles, and density scatters (capped)
# ----------------------------------------------------------------------------------------------
def run_capped_core(B: int) -> None:
    out = results_dir('megascale', split='capped')
    if CORE_RHO_BOOTSTRAP_CSV.exists():
        df = load_rho_df(CORE_RHO_BOOTSTRAP_CSV)
        df = df[df['dataset'].isin(TSU_CAPPED)]
    else:
        frames = {d: load_frame(d) for d in TSU_CAPPED}
        keys = sorted({k for f in frames.values() for k in MODELS if k in f.preds})
        r_rows = []
        for kind in ('protein', 'mutation'):
            pt, rp = bootstrap_rho(frames, keys, kind=kind, B=B, seed=11)
            for ds in TSU_CAPPED:
                for k, (r, _ps, _n, _ng) in pt[ds].items():
                    lo, hi = ci(rp[ds][k])
                    r_rows.append(dict(dataset=ds, model=k, kind=kind, rho=r, lo=lo, hi=hi))
        df = pd.DataFrame(r_rows)

    P = {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in df[df['kind'] == 'protein'].itertuples()}
    Mu = {(r.dataset, r.model): (r.rho, r.lo, r.hi) for r in df[df['kind'] == 'mutation'].itertuples()}
    leak = leak_table()
    keys = [k for k in CAPPED_KEYS if (TSU_CAPPED[1], k) in P]
    mega = [k for k in keys if k in MEGASCALE_TRAINED]

    fig, axes = plt.subplots(1, 2, figsize=(9, 0.33 * len(keys) + 1.4), sharey=True)
    for ax, ds, ttl in zip(axes, TSU_CAPPED, ('validation (23 libraries)', 'test (33 libraries)')):
        for i, k in enumerate(keys):
            if (ds, k) not in P:
                continue
            v, lo, hi = P[(ds, k)]
            kw = style.marker_kwargs(k, size=6.5)
            ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.4, alpha=0.45, solid_capstyle='butt')
            m = Mu.get((ds, k))
            if m:
                ax.plot([m[1], m[2]], [i, i], color=kw['color'], lw=0.8)
            ax.plot(v, i, **kw)
            if leak.get((k, ds), False):
                ax.plot(v, i, marker='o', ms=11, mfc='none', mec=style.RING, mew=1.0, ls='none')
            ax.text(max(hi, m[2] if m else hi) + 0.008, i, f'{v:.3f}', va='center', fontsize=7.5, color='#444')
        for i, k in enumerate(keys):
            if k in mega:
                ax.axhspan(i - 0.5, i + 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
        ax.set_title(ttl, loc='left')
        ax.grid(axis='x', color=style.GRID, lw=0.6)
        ax.set_xlabel('mean per-library Spearman ρ')
        ax.set_xlim(0.40, 0.93)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([MODELS[k]['label'] for k in keys], fontsize=8.5)
    axes[0].set_ylim(len(keys) - 0.4, -0.6)
    for t, k in zip(axes[0].get_yticklabels(), keys):
        t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555'}[MODELS[k]['group']])
        if MODELS[k]['group'] == 'msr':
            t.set_fontweight('bold')
    fig.suptitle(
        'thick bar: resampling libraries; thin line: resampling mutations; red ring: documented training overlap\n'
        + MEGASCALE_NOTE + ' (grey rows, at the bottom)',
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    style.save(fig, out / 'megascale_overview')

    # Per-library scatter on capped test
    fr_test = load_frame(TSU_CAPPED[1])
    sp = fr_test.spine
    ok = sp['use'].to_numpy()
    y = sp['y'].to_numpy()
    libs = sorted(sp['group'][ok].unique())

    def _lib_rho(k: str) -> pd.Series:
        vals = {}
        for g in libs:
            m = ok & (sp['group'] == g).to_numpy() & np.isfinite(fr_test.preds[k]).all(0)
            vals[g] = float(np.mean([spearmanr(p[m], y[m])[0] for p in fr_test.preds[k]]))
        return pd.Series(vals)

    R_lib = pd.DataFrame({'msrp0:comb': _lib_rho('msrp0:comb'), **{k: _lib_rho(k) for k in RETR_KEYS}})
    atomic_to_csv(R_lib, out / 'per_library_rho_test.csv', float_format='%.4f')
    fig_pl, axes_pl = plt.subplots(1, len(RETR_KEYS), figsize=(2.6 * len(RETR_KEYS), 3.0), sharex=True, sharey=True)
    for ax, k in zip(axes_pl, RETR_KEYS):
        ax.scatter(R_lib[k], R_lib['msrp0:comb'], s=22, color=MODELS[k]['color'], alpha=0.8, lw=0)
        ax.plot([0.3, 1.0], [0.3, 1.0], color='#999', lw=0.8, ls='--')
        ax.set_title(MODELS[k]['label'].replace(' (retrained)', ''), fontsize=8.5)
        ax.text(0.04, 0.95, f'ESM-MSR higher in {(R_lib["msrp0:comb"] > R_lib[k]).sum()} / {len(R_lib)}', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('ρ of the baseline')
    axes_pl[0].set_ylabel('ρ of ESM-MSR Dual-view')
    fig_pl.suptitle('Each dot is one test library', fontsize=9, y=1.0)
    fig_pl.tight_layout()
    style.save(fig_pl, out / 'megascale_per_library')

    # Singles & doubles separately
    sd_rows = []
    for ds in TSU_CAPPED:
        fr = load_frame(ds)
        sp_ds = fr.spine
        for nm, m in (('singles', (sp_ds['n_mut'] == 1).to_numpy() & sp_ds['use'].to_numpy()), ('doubles', (sp_ds['n_mut'] == 2).to_numpy() & sp_ds['use'].to_numpy())):
            sub = subframe(fr, m, keys=CAPPED_KEYS)
            point, reps = bootstrap_rho({ds: sub}, CAPPED_KEYS, kind='protein', B=B, seed=6, per_protein=True)
            for k, (r_val, _ps, n, ng) in point[ds].items():
                lo, hi = ci(reps[ds][k])
                sd_rows.append(dict(dataset=ds, subset=nm, model=k, rho=r_val, lo=lo, hi=hi, n=n, n_libraries=ng))
    R_sd = pd.DataFrame(sd_rows)
    atomic_to_csv(R_sd, out / 'megascale_singles_doubles.csv', index=False, float_format='%.4f')
    groups = [f'{d}|{s}' for d in TSU_CAPPED for s in ('singles', 'doubles')]
    glabel = [f'{"val" if d == TSU_CAPPED[0] else "test"}\n{s}' for d in TSU_CAPPED for s in ('singles', 'doubles')]
    sd_models = ['msrp0:wt', 'msrp0:comb', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
    table_sd = {(f'{r.dataset}|{r.subset}', r.model): r.rho for r in R_sd.itertuples()}
    cis_sd = {(f'{r.dataset}|{r.subset}', r.model): (r.lo, r.hi) for r in R_sd.itertuples()}
    fig_sd, ax_sd = plt.subplots(figsize=(8.5, 4))
    style.dot_panel(ax_sd, groups, sd_models, table_sd, cis_sd, highlight=['msrp0:wt', 'msrp0:comb'], ylabel='mean per-library ρ', xlabels=glabel, rot=0, size=6.5, ylim=(0.5, 0.95))
    style.model_legend(ax_sd, sd_models, loc='lower left', ncol=4, fontsize=7.5)
    fig_sd.tight_layout()
    style.save(fig_sd, out / 'megascale_singles_doubles')

    # Predicted vs measured ddG density scatters
    panels = [
        ('msrp0:wt', 'singles', 'ESM-MSR WT-view'),
        ('msrp0:wt', 'doubles', 'ESM-MSR WT-view'),
        ('msrp0:comb', 'doubles', 'ESM-MSR Dual-view'),
        ('me2', 'doubles', 'MutateEverything Stage 2'),
        ('tmpnnd', 'doubles', 'ThermoMPNN-D'),
        ('spursm', 'doubles', 'SPURS-multi'),
    ]
    fig_sc, axes_sc = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.2), sharey=True)
    for ax, (k, subset, ttl) in zip(axes_sc, panels):
        pr = fr_test.preds[k].mean(axis=0)
        m = ok & (sp['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy() & np.isfinite(pr)
        style.density_scatter(ax, pr[m], y[m], s=3, line=True, max_points=40000)
        ax.set_title(f'{ttl}\n{subset}', fontsize=8.5)
        ax.set_xlabel('predicted ddG (mean of seeds 1-3)')
    axes_sc[0].set_ylabel('measured ddG (kcal/mol)')
    fig_sc.tight_layout()
    style.save(fig_sc, out / 'megascale_scatter')


# ----------------------------------------------------------------------------------------------
# 2. Publication bar charts on MegaScale test (ndcg96, spearman_stab_destab*, esm3_scoring_variants, ndcg96 supplement)
# ----------------------------------------------------------------------------------------------
def run_publication_bars(B: int, split: str = 'capped') -> None:
    out = results_dir('megascale', split=split)
    fr, _ = load_tsu_test_frame(split=split)
    add_esm3_marginal_additive(fr, split=split)
    sp = fr.spine
    use = sp['use'].to_numpy()
    n_mut = sp['n_mut'].to_numpy()
    single, double = use & (n_mut == 1), use & (n_mut == 2)
    y = sp['y'].to_numpy()
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = make_domain_counts(libs, B=B, seed=5)
    B_mut = min(B, 200)
    B_pool = min(B, 200)

    best, _ = best_other(fr, out_csv_path=(out / 'best_other_ddg_model.csv') if split == 'capped' else None)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols = main_columns(fr, best, 'unmasked')
    name_map.update({c['name']: c['label'].replace('\n', ' ') for c in cols if c['name'].endswith('(alt)')})
    s_title = f'{split_title(split)} test'
    w_fig = fig_width(cols)

    def _save_tab(T: pd.DataFrame, stem: str) -> None:
        atomic_to_csv(T, out / f'{stem}.csv', index=False, float_format='%.4f')

    def _save_p(rows: list[dict], stem: str) -> None:
        df_p = add_holm_by_group(pd.DataFrame(rows))
        atomic_to_csv(df_p, out / f'{stem}_pvalues.csv', index=False, float_format='%.4g')

    # Rectified NDCG@96, singles vs doubles
    cells1 = {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}
    v1: dict = {}
    T1 = table_grouped(fr, cols, cells1, ndcg96_rectified, 1, counts, libs, vals=v1)
    ov1 = table_grouped(fr, cols, {'overall': (use, 'all')}, ndcg96_rectified, 1, counts, libs, vals=v1)
    _save_tab(pd.concat([T1, ov1]), 'ndcg96_singles_doubles')
    _save_p(compare_grouped(fr, v1, ['singles', 'doubles', 'overall'], counts, libs, name_map=name_map), 'ndcg96_singles_doubles')
    d1 = (ov1.set_index('column')['value'] - T1[T1['subset'] == 'singles'].set_index('column')['value']).dropna()
    d1 = d1[[c['name'] for c in cols if c.get('draw', True) and c['name'] in d1.index]]
    draw_ov1 = int((d1.abs() <= 0.01).sum()) < 0.9 * len(d1)
    note1 = [F_COMPOSITE]
    if draw_ov1 and len(d1):
        note1.insert(0, f'Overall = NDCG@96 of a library with its singles and doubles ranked together; differs from singles bar by {d1.min():+.2f} to {d1.max():+.2f}.')
    fig1, ax1 = plt.subplots(figsize=(12.5, 6.4))
    style.bar_panel(
        ax1, cols, T1, ('singles', 'doubles'),
        (count_text(T1, 'singles', 'singles'), count_text(T1, 'doubles', 'doubles')),
        f'Rectified NDCG@{K_NDCG} (mean over test domains)', (0.0, 1.12),
        overall_df=ov1 if draw_ov1 else None, overall_label=f'overall (n = {int(ov1.n_variants.max()):,})',
        tick_fs=8.5,
    )
    for t in ax1.texts:
        if t.get_fontsize() < 8:
            t.set_fontsize(8)
    fig1.tight_layout(pad=0.5)
    out_supp_rect = results_dir('supplementary_figs') if split == 'capped' else out
    stem_rect = '_ndcg96_rectified_singles_doubles' if split == 'capped' else 'ndcg96_singles_doubles'
    style.save(fig1, out_supp_rect / stem_rect)

    # Stabilizing vs destabilizing (domain and mutation bootstrap; per-domain and pooled)
    for kind in ('domain', 'mutation'):
        tag = '' if kind == 'domain' else '_mutboot'
        unit = 'resampling domains' if kind == 'domain' else 'resampling mutations within each domain'
        unit_p = 'resampling domains' if kind == 'domain' else 'resampling mutations'
        b_this = B if kind == 'domain' else B_mut

        cells2 = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}
        v2: dict = {}
        T2 = table_grouped(fr, cols, cells2, rho_fn, MIN_N, counts, libs, kind=kind, B_mut=B_mut, vals=v2)
        ov2 = table_grouped(fr, cols, {'overall': (use, 'all')}, rho_fn, 3, counts, libs, kind=kind, B_mut=B_mut, vals=v2)
        _save_tab(pd.concat([T2, ov2]), f'spearman_stab_destab{tag}')
        _save_p(compare_grouped(fr, v2, ['stabilizing', 'destabilizing', 'overall'], counts, libs, kind=kind, B_mut=B_mut, name_map=name_map), f'spearman_stab_destab{tag}')
        fig2, ax2 = plt.subplots(figsize=(w_fig, 4.9))
        style.bar_panel(
            ax2, cols, T2, ('stabilizing', 'destabilizing'),
            (count_text(T2, 'stabilizing', 'stabilizing, ddG > 0'), count_text(T2, 'destabilizing', 'destabilizing, ddG < 0')),
            "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
            overall_df=ov2, overall_label=f'overall (n = {int(ov2.n_variants.max()):,})', zero_line=True,
        )
        finish_bar_fig(fig2, f'{s_title}, all variants: Spearman ρ of stabilizing vs destabilizing variants (≥ {MIN_N}/subset); 95% intervals from {unit} (B = {b_this:,})', [F_COMPOSITE])
        style.save(fig2, out / f'spearman_stab_destab{tag}')

        pc = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all'), 'overall': (use, 'all')}
        rp2: dict = {}
        Tp2 = table_pooled(fr, cols, pc, y, B_=B_pool, kind=kind, reps_out=rp2)
        _save_tab(Tp2, f'spearman_stab_destab_pooled{tag}')
        _save_p(compare_pooled(Tp2, rp2, ['stabilizing', 'destabilizing', 'overall'], kind=kind, name_map=name_map), f'spearman_stab_destab_pooled{tag}')
        pv2, po2 = Tp2[Tp2['subset'] != 'overall'], Tp2[Tp2['subset'] == 'overall']
        fig2p, ax2p = plt.subplots(figsize=(w_fig, 4.9))
        style.bar_panel(
            ax2p, cols, pv2, ('stabilizing', 'destabilizing'),
            (count_text(pv2, 'stabilizing', 'stabilizing, ddG > 0'), count_text(pv2, 'destabilizing', 'destabilizing, ddG < 0')),
            "Spearman's ρ pooled over test domains", (-0.3, 1.12),
            overall_df=po2, overall_label=f'overall (n = {int(po2.n_variants.max()):,})', zero_line=True,
        )
        finish_bar_fig(fig2p, f'{s_title}, all variants: Spearman ρ pooled over domains, stabilizing vs destabilizing; 95% intervals from {unit_p} (B = {B_pool:,})', [F_COMPOSITE])
        style.save(fig2p, out / f'spearman_stab_destab_pooled{tag}')

        # Singles and doubles panels separately
        tabs, tabs_p, ps, ps_p = [], [], [], []
        fig2b, axes2b = plt.subplots(2, 1, figsize=(w_fig, 8.6))
        fig2bp, axes2bp = plt.subplots(2, 1, figsize=(w_fig, 8.6))
        for ax_b, ax_bp, (which, base_m) in zip(axes2b, axes2bp, (('singles', single), ('doubles', double))):
            tk = 'label_s' if which == 'singles' else 'label_d'
            c_sub = {'stabilizing': (base_m & (y > 0), which), 'destabilizing': (base_m & (y < 0), which)}
            vb: dict = {}
            Tb = table_grouped(fr, cols, c_sub, rho_fn, MIN_N, counts, libs, kind=kind, B_mut=B_mut, vals=vb)
            ovb = table_grouped(fr, cols, {'overall': (base_m, which)}, rho_fn, 3, counts, libs, kind=kind, B_mut=B_mut, vals=vb)
            tabs.append(pd.concat([Tb, ovb]).assign(panel=which))
            ps.append(pd.DataFrame(compare_grouped(fr, vb, ['stabilizing', 'destabilizing', 'overall'], counts, libs, kind=kind, B_mut=B_mut, name_map=name_map)).assign(panel=which))
            style.bar_panel(
                ax_b, cols, Tb, ('stabilizing', 'destabilizing'),
                (count_text(Tb, 'stabilizing', 'stabilizing, ddG > 0'), count_text(Tb, 'destabilizing', 'destabilizing, ddG < 0')),
                "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
                overall_df=ovb, overall_label=f'overall (n = {int(ovb.n_variants.max()):,})',
                tick=tk, headers=(which == 'singles'), zero_line=True,
            )
            ax_b.text(0.006, 0.985, which, transform=ax_b.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')

            pcells = {'stabilizing': (base_m & (y > 0), which), 'destabilizing': (base_m & (y < 0), which), 'overall': (base_m, which)}
            rpb: dict = {}
            Tbp = table_pooled(fr, cols, pcells, y, B_=B_pool, kind=kind, reps_out=rpb)
            tabs_p.append(Tbp.assign(panel=which))
            ps_p.append(pd.DataFrame(compare_pooled(Tbp, rpb, ['stabilizing', 'destabilizing', 'overall'], kind=kind, name_map=name_map)).assign(panel=which))
            pvb, pob = Tbp[Tbp['subset'] != 'overall'], Tbp[Tbp['subset'] == 'overall']
            style.bar_panel(
                ax_bp, cols, pvb, ('stabilizing', 'destabilizing'),
                (count_text(pvb, 'stabilizing', 'stabilizing, ddG > 0'), count_text(pvb, 'destabilizing', 'destabilizing, ddG < 0')),
                "Spearman's ρ pooled over test domains", (-0.3, 1.12),
                overall_df=pob, overall_label=f'overall (n = {int(pob.n_variants.max()):,})',
                tick=tk, headers=(which == 'singles'), zero_line=True,
            )
            ax_bp.text(0.006, 0.985, which, transform=ax_bp.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')

        _save_tab(pd.concat(tabs), f'spearman_stab_destab_singles_doubles{tag}')
        _save_tab(pd.concat(tabs_p), f'spearman_stab_destab_singles_doubles_pooled{tag}')
        _save_p(pd.concat(ps).to_dict('records'), f'spearman_stab_destab_singles_doubles{tag}')
        _save_p(pd.concat(ps_p).to_dict('records'), f'spearman_stab_destab_singles_doubles_pooled{tag}')
        finish_bar_fig(fig2b, f'{s_title}, Spearman ρ of stabilizing vs destabilizing variants by subset; 95% intervals from {unit} (B = {b_this:,})', [])
        style.save(fig2b, out / f'spearman_stab_destab_singles_doubles{tag}')
        finish_bar_fig(fig2bp, f'{s_title}, Spearman ρ pooled over domains, stabilizing vs destabilizing variants; 95% intervals from {unit_p} (B = {B_pool:,})', [])
        style.save(fig2bp, out / f'spearman_stab_destab_singles_doubles_pooled{tag}')

    if split != 'capped':
        return

    # ESM3 scoring variants (11e, MegaScale part)
    if 'esm3avg' in fr.preds and 'esm3mm' in fr.preds:
        cols2_esm3 = main_columns(fr, best, 'average')
        grey = [M.color('ESM3-small-open'), '#aab7b8', '#4d5656']
        three2 = [
            col_spec('ESM3 unmasked', 'zeroshot', grey[0], 'ESM3-small-open\nunmasked\n(combined)', 'msrrel@0:comb'),
            col_spec('ESM3 independent masking', 'zeroshot', grey[1], 'ESM3-small-open\nind. masking\n(WT, MT average)', 'esm3avg'),
            col_spec('ESM3 masked marginal', 'zeroshot', grey[2], 'ESM3-small-open\nmasked marginal', 'esm3mm'),
        ]
        idx_e = next(i for i, c in enumerate(cols2_esm3) if c['name'] == 'ESM3-small-open')
        cols2_esm3 = cols2_esm3[:idx_e] + three2 + cols2_esm3[idx_e + 1:]
        T2e = table_grouped(fr, cols2_esm3, {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}, rho_fn, MIN_N, counts, libs)
        ov2e = table_grouped(fr, cols2_esm3, {'overall': (use, 'all')}, rho_fn, 3, counts, libs)
        _save_tab(pd.concat([T2e, ov2e]), 'esm3_scoring_variants')
        fig_e, ax_e = plt.subplots(figsize=(1.2 * fig_width(cols2_esm3), 4.9))
        style.bar_panel(
            ax_e, cols2_esm3, T2e, ('stabilizing', 'destabilizing'),
            (count_text(T2e, 'stabilizing', 'stabilizing, ddG > 0'), count_text(T2e, 'destabilizing', 'destabilizing, ddG < 0')),
            "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
            overall_df=ov2e, overall_label=f'overall (n = {int(ov2e.n_variants.max()):,})',
            zero_line=True, tick_fs=7.8,
        )
        finish_bar_fig(fig_e, f'{s_title}, all variants: Spearman ρ of stabilizing vs destabilizing variants, ESM3-small-open scored three ways (B = {B:,})', [F_COMPOSITE])
        style.save(fig_e, out / 'esm3_scoring_variants')

    # Shifted NDCG@96 supplement (11k): relevance = y - min(y) per domain & subset, defined for all 33 domains
    out_supp = results_dir('supplementary_figs')
    Ts = table_grouped(fr, cols, {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}, ndcg96_shifted, MIN_N, counts, libs)
    ovs = table_grouped(fr, cols, {'overall': (use, 'all')}, ndcg96_shifted, MIN_N, counts, libs)
    ds_diff = (ovs.set_index('column')['value'] - Ts[Ts['subset'] == 'singles'].set_index('column')['value']).dropna()
    drawn_names = [c['name'] for c in cols if c.get('draw', True) and c['name'] in ds_diff.index]
    ds_diff = ds_diff[drawn_names]
    draw_ovs = int((ds_diff.abs() <= 0.01).sum()) < 0.9 * len(ds_diff)
    fig_s, ax_s = plt.subplots(figsize=(12.5, 6.4))
    style.bar_panel(
        ax_s, cols, Ts, ('singles', 'doubles'),
        (count_text(Ts, 'singles', 'singles'), count_text(Ts, 'doubles', 'doubles')),
        f'Shifted NDCG@{K_NDCG} (mean over test domains)', (0.0, 1.12),
        overall_df=ovs if draw_ovs else None, overall_label=f'overall (n = {int(ovs.n_variants.max()):,})',
        tick_fs=8.5,
    )
    for t in ax_s.texts:
        if t.get_fontsize() < 8:
            t.set_fontsize(8)
    fig_s.tight_layout(pad=0.5)
    fig_s.savefig(out_supp / 'ndcg96_singles_doubles.png', dpi=300, bbox_inches=None)
    fig_s.savefig(out_supp / 'ndcg96_singles_doubles.pdf', bbox_inches=None)
    fig_s.savefig(out_supp / 'supp_figure2.png', dpi=300, bbox_inches=None)
    fig_s.savefig(out_supp / 'supp_figure2.pdf', bbox_inches=None)
    plt.close(fig_s)
    vals_piv = pd.concat([Ts, ovs]).pivot(index='column', columns='subset', values='value').round(3).reset_index()
    atomic_to_csv(vals_piv, out_supp / 'ndcg96_singles_doubles_values.csv', index=False)


# ----------------------------------------------------------------------------------------------
# 3. Statistical hypothesis tables on MegaScale capped test (11i & 11j)
# ----------------------------------------------------------------------------------------------
def run_doubles_vs_singles_on_singles(B: int) -> pd.DataFrame:
    """11i: Do models trained with doubles lose accuracy on single mutants vs their singles-only versions?"""
    out = results_dir('megascale', split='capped')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    single = sp['use'].to_numpy() & (sp['n_mut'].to_numpy() == 1)
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    b_dom = B if B <= 200 else max(B, 10000)
    counts = make_domain_counts(libs, B=b_dom, seed=5)

    models_cmp = [
        (n, M.BY_NAME[n].single, M.BY_NAME[n].multi, True, M.version_label(n, 0), M.version_label(n, 1))
        for n in M.TRANSFER
    ]
    rows = []
    for name, ks, kd, head, ls, ld in models_cmp:
        vs = per_library(fr, ks, single, rho_fn, MIN_N)
        vd = per_library(fr, kd, single, rho_fn, MIN_N)
        both = vs.index.intersection(vd.index)
        d = diff_df(vd, vs, both)
        reps = boot_reps(d, libs, counts)
        reps = reps[np.isfinite(reps)]
        dom = d.mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        ds_, dd_ = fr.preds[ks][:, single], fr.preds[kd][:, single]
        same = bool(np.nanmax(np.abs(ds_ - dd_)) < 5e-3 * np.nanstd(ds_))
        rows.append(dict(
            model=name, same_predictions=same, headline=head,
            singles_version=ls, doubles_version=ld,
            n_domains=len(both), n_seeds=vs.shape[1],
            rho_singles=float(vs.loc[both].mean(axis=1).mean()),
            rho_doubles=float(vd.loc[both].mean(axis=1).mean()),
            delta=float(dom.mean()), lo=float(lo), hi=float(hi),
            domains_worse=int((dom < 0).sum()),
            p=float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1.0 / (len(reps) + 1))),
            p_loss_one_sided=float((reps >= 0).mean()),
        ))
    res = pd.DataFrame(rows)
    tested = (res['headline'] & ~res['same_predictions']).to_numpy()
    res['p_holm'] = np.nan
    res.loc[tested, 'p_holm'] = holm(res.loc[tested, 'p'].to_numpy())
    res['B'] = b_dom
    atomic_to_csv(res, out / 'doubles_vs_singles_on_singles.csv', index=False, float_format='%.5g')

    headers = [
        'Model', 'Singles version →\ntrained with doubles', 'ρ\nsingles', 'ρ trained\nwith doubles',
        'Δρ [95% CI]', 'Domains\nlower', 'p', 'p\n(Holm)', 'Doubles-trained\nversion',
    ]
    t_rows = []
    for r in res.itertuples():
        if r.same_predictions:
            t_rows.append([r.model, f'{r.singles_version} → {r.doubles_version}', f'{r.rho_singles:.3f}', f'{r.rho_doubles:.3f}', 'identical predictions', '–', '–', '–', 'same network on singles, not tested'])
            continue
        verdict = 'lower' if r.delta < 0 else 'higher'
        sig = _stars(r.p_holm)
        t_rows.append([
            r.model, f'{r.singles_version} → {r.doubles_version}', f'{r.rho_singles:.3f}', f'{r.rho_doubles:.3f}',
            f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]', f'{r.domains_worse} / {r.n_domains}',
            fmt_p(r.p, B=r.B, prefix=''), fmt_p(r.p_holm, B=r.B, prefix=''), f'{verdict}, {sig}',
        ])
    note = (
        f'Tsuboyama capped test set, single mutants. ρ: mean over test domains (≥ {MIN_N} singles) of the per-domain Spearman ρ, seeds averaged; '
        f'Δρ = trained with doubles − singles version.\np: two-sided, paired bootstrap over domains and seeds ({b_dom:,} draws shared by all models); '
        'Holm over the tested models (ThermoMPNN-D scores singles with the ThermoMPNN network, so it is not tested).\n'
        'Red: significantly lower; green: significantly higher (p < 0.05).'
    )
    _write_md_table(t_rows, headers, note, out / 'doubles_vs_singles_on_singles.md')

    def _style_cell(ri: int, cj: int, cell) -> None:
        r = res.iloc[ri]
        if r.same_predictions:
            return
        if cj == 8 and r.p_holm < 0.05:
            cell.set_facecolor('#f6d5d0' if r.delta < 0 else '#d6ecd9')

    _render_stat_table(
        t_rows, headers, [2.4, 4.5, 0.9, 1.2, 2.8, 1.1, 0.95, 0.95, 3.1],
        out / 'doubles_vs_singles_on_singles', note, cell_style_fn=_style_cell,
    )
    build_supp_figure3(B=B, res=res, fr=fr)
    return res


def build_supp_figure3(B: int = 1000, res: pd.DataFrame | None = None, fr=None) -> Path:
    """Build Supplementary Figure 3 (`00_supplementary_figs/supp_figure3.{png,pdf}`).

    Ranking performance of all doubles-trained models (Dual-view for ESM-MSR) on Homology-Restricted
    MegaScale test-set single mutants, paired against their singles-only counterparts:
      a. Mean per-domain Spearman's rho (singles-only hatched vs doubles-trained solid, 95% CI, Holm p).
      b. Per-domain paired difference Delta-rho = rho(doubles-trained) - rho(singles-only) across all 33 domains.
    """
    from matplotlib.patches import Patch

    out_supp = results_dir('supplementary_figs')
    if fr is None:
        fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    single = sp['use'].to_numpy() & (sp['n_mut'].to_numpy() == 1)
    n_singles = int(single.sum())
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    b_dom = B if B <= 200 else max(B, 10000)
    counts = make_domain_counts(libs, B=b_dom, seed=5)

    if res is None:
        csv_path = results_dir('megascale', split='capped', create=False) / 'doubles_vs_singles_on_singles.csv'
        res = pd.read_csv(csv_path) if csv_path.exists() else run_doubles_vs_singles_on_singles(B=B)

    res_by_name = res.set_index('model')
    models_order = list(M.TRANSFER)

    stats_a = []
    dom_deltas = {}
    for name in models_order:
        ks, kd = M.BY_NAME[name].single, M.BY_NAME[name].multi
        vs = per_library(fr, ks, single, rho_fn, MIN_N)
        vd = per_library(fr, kd, single, rho_fn, MIN_N)
        both = vs.index.intersection(vd.index)
        rs = boot_reps(vs.loc[both], libs, counts)
        rd = boot_reps(vd.loc[both], libs, counts)
        rs, rd = rs[np.isfinite(rs)], rd[np.isfinite(rd)]
        d_dom = diff_df(vd, vs, both).mean(axis=1).to_numpy(float)
        dom_deltas[name] = d_dom
        r_row = res_by_name.loc[name]
        stats_a.append(dict(
            name=name,
            col=M.color(name),
            ls=M.version_label(name, 0),
            ld=M.version_label(name, 1),
            rho_s=float(r_row['rho_singles']),
            lo_s=float(np.percentile(rs, 2.5)),
            hi_s=float(np.percentile(rs, 97.5)),
            rho_d=float(r_row['rho_doubles']),
            lo_d=float(np.percentile(rd, 2.5)),
            hi_d=float(np.percentile(rd, 97.5)),
            delta=float(r_row['delta']),
            d_lo=float(r_row['lo']),
            d_hi=float(r_row['hi']),
            same=bool(r_row['same_predictions']),
            p_holm=float(r_row['p_holm']) if np.isfinite(r_row['p_holm']) else np.nan,
            n_dom=int(r_row['n_domains']),
            n_higher=int((d_dom > 0).sum()),
            n_lower=int((d_dom < 0).sum()),
            B_p=int(r_row['B']) if 'B' in r_row.index and np.isfinite(r_row['B']) else b_dom,
        ))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw=dict(width_ratios=[1.18, 1.0]), facecolor='white')

    # --- Panel a: Paired Spearman rho bars (singles-only hatched vs doubles-trained solid) ---
    xs = np.arange(len(stats_a), dtype=float)
    w = 0.34
    for i, st in enumerate(stats_a):
        c = st['col']
        ax_a.bar(
            xs[i] - w / 2, st['rho_s'], width=w,
            facecolor=c, alpha=0.52, edgecolor=c, lw=1.0, hatch='///', zorder=3,
        )
        ax_a.errorbar(
            xs[i] - w / 2, st['rho_s'],
            yerr=[[max(st['rho_s'] - st['lo_s'], 0)], [max(st['hi_s'] - st['rho_s'], 0)]],
            fmt='none', ecolor='#b22222', elinewidth=1.2, capsize=3.0, zorder=5,
        )
        ax_a.text(xs[i] - w / 2, st['hi_s'] + 0.012, f'{st["rho_s"]:.3f}',
                  ha='center', va='bottom', fontsize=8.4, color='#333333', zorder=6)

        ax_a.bar(
            xs[i] + w / 2, st['rho_d'], width=w,
            facecolor=c, alpha=0.95, edgecolor='#1a1a1a', lw=0.9, zorder=3,
        )
        ax_a.errorbar(
            xs[i] + w / 2, st['rho_d'],
            yerr=[[max(st['rho_d'] - st['lo_d'], 0)], [max(st['hi_d'] - st['rho_d'], 0)]],
            fmt='none', ecolor='#b22222', elinewidth=1.2, capsize=3.0, zorder=5,
        )
        ax_a.text(xs[i] + w / 2, st['hi_d'] + 0.012, f'{st["rho_d"]:.3f}',
                  ha='center', va='bottom', fontsize=8.4, fontweight='bold', color='#111111', zorder=6)

        y_br = max(st['hi_s'], st['hi_d']) + 0.055
        h_br = 0.010
        x0, x1 = xs[i] - w / 2, xs[i] + w / 2
        ax_a.plot([x0, x0, x1, x1], [y_br - h_br, y_br, y_br, y_br - h_br], color='#333333', lw=1.0, zorder=7)
        if st['same']:
            p_lbl = 'identical\nnetwork'
        else:
            p_lbl = f'p_Holm {fmt_p(st["p_holm"], B=st["B_p"], prefix="= ")}'.replace('= <', '<')
        ax_a.text(xs[i], y_br + 0.008, p_lbl, ha='center', va='bottom', fontsize=7.8,
                  color='#1e6b34' if (not st['same'] and st['delta'] > 0) else ('#a61c1c' if not st['same'] else '#555555'),
                  fontweight='bold' if not st['same'] else 'normal', zorder=7)

    ax_a.set_ylim(0.50, 1.03)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([f'{st["ls"]}\n→ {st["ld"]}' for st in stats_a], fontsize=8.5, rotation=14, ha='right')
    ax_a.set_ylabel("Spearman's ρ on test single mutants\n(mean over 33 domains)", fontsize=10.0)
    ax_a.set_title(f'Test-set single-mutant ranking (n = {n_singles:,} singles)', fontsize=10.4, fontweight='bold', loc='left', pad=8)
    ax_a.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.legend(
        handles=[
            Patch(fc='#888888', alpha=0.52, ec='#555555', hatch='///', label='Singles-only model'),
            Patch(fc='#444444', alpha=0.95, ec='#1a1a1a', label='Doubles-trained model (Dual-view for ESM-MSR)'),
        ],
        loc='lower left', fontsize=8.4, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#cccccc',
    )

    # --- Panel b: Per-domain paired Delta-rho = doubles-trained - singles-only ---
    rng = np.random.default_rng(7)
    ax_b.axhline(0.0, color='#333333', lw=1.1, ls='--', zorder=2)
    for i, st in enumerate(stats_a):
        c = st['col']
        vals = dom_deltas[st['name']]
        jit = rng.uniform(-0.14, 0.14, size=len(vals))
        ax_b.scatter(xs[i] + jit, vals, s=18, color=c, alpha=0.50, edgecolors='white', lw=0.4, zorder=3)
        if st['same']:
            ax_b.plot([xs[i] - 0.18, xs[i] + 0.18], [0.0, 0.0], color='#222222', lw=2.2, zorder=5)
            ax_b.text(xs[i], 0.018, '0.000\n(identical)', ha='center', va='bottom', fontsize=8.0, color='#555555', zorder=6)
        else:
            ax_b.errorbar(
                xs[i], st['delta'],
                yerr=[[max(st['delta'] - st['d_lo'], 0)], [max(st['d_hi'] - st['delta'], 0)]],
                fmt='D', ms=6.2, color='#111111', mfc=c, mec='#111111', mew=1.1,
                ecolor='#111111', elinewidth=1.5, capsize=4.0, zorder=6,
            )
            frac_txt = f'{st["n_higher"]}/{st["n_dom"]} higher' if st['delta'] > 0 else f'{st["n_lower"]}/{st["n_dom"]} lower'
            txt_col = '#1e6b34' if st['delta'] > 0 else '#a61c1c'
            y_txt = st['d_hi'] + 0.014 if st['delta'] > 0 else st['d_lo'] - 0.014
            va_txt = 'bottom' if st['delta'] > 0 else 'top'
            ax_b.text(xs[i], y_txt, f'{st["delta"]:+.3f}\n({frac_txt})',
                      ha='center', va=va_txt, fontsize=8.2, fontweight='bold', color=txt_col, zorder=7)

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([st['ld'] for st in stats_a], fontsize=8.6, rotation=14, ha='right')
    ax_b.set_ylabel('Paired Δρ = ρ(doubles-trained) − ρ(singles-only)', fontsize=10.0)
    ax_b.set_title('Per-domain shift on single mutants (33 domains)', fontsize=10.4, fontweight='bold', loc='left', pad=8)
    ax_b.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ymin_b = min(min(v.min() for v in dom_deltas.values()), -0.14)
    ymax_b = max(max(v.max() for v in dom_deltas.values()), 0.09)
    ax_b.set_ylim(ymin_b - 0.035, ymax_b + 0.035)

    fig.subplots_adjust(left=0.068, right=0.985, top=0.905, bottom=0.255, wspace=0.22)
    fig.canvas.draw()
    pb = ax_b.get_position()
    fig.text(0.010, 0.975, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pb.x0 - 0.056, 0.975, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    note_f = (
        f'Homology-Restricted MegaScale test set, single mutants only (n = {n_singles:,} across 33 domains; 3 seeds averaged per model).  '
        f'a. Mean per-domain Spearman ρ comparing each family\'s singles-only model (hatched) against its doubles-trained version (solid; Dual-view for ESM-MSR); '
        f'whiskers show 95% joint domain × seed bootstrap CIs (B = {b_dom:,}); brackets show Holm-adjusted two-sided paired p-values.  '
        f'b. Per-domain paired difference Δρ = ρ(doubles-trained) − ρ(singles-only) (dots: individual domains; diamonds & whiskers: mean & 95% CI). '
        f'ESM-MSR Dual-view is the only doubles-trained model that significantly improves single-mutant ranking (+0.014, 30/33 domains higher), whereas MutateEverything Stage 2 (−0.015) and SPURS-multi (−0.043) degrade relative to their singles-only versions (ThermoMPNN-D reuses the ThermoMPNN network on singles).'
    )
    kw = dict(fontsize=7.8, color='#333333', linespacing=1.25)
    fig.text(0.014, 0.012, style.wrap_to_px(fig, note_f, 0.972 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'supp_figure3'
    style.save(fig, out_path)
    return out_path


def run_overall_vs_transfer(B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """11j: ESM-MSR vs transfer baselines on MegaScale capped test (domain mean & pooled with joint seed resampling)."""
    out = results_dir('megascale', split='capped')
    fr, _ = load_tsu_test_frame('capped')
    sp = fr.spine
    use, g, y = sp['use'].to_numpy(), sp['group'].to_numpy(), sp['y'].to_numpy()
    n1 = (sp['n_mut'] == 1).to_numpy()[None, :]
    others = [
        (M.BY_NAME[n].label_all.replace('\n', ' '), M.BY_NAME[n].single, M.BY_NAME[n].multi)
        for n in ('MutateEverything', 'ThermoMPNN', 'SPURS')
    ]
    msr_raw = np.where(n1, fr.preds['msrp0:wt'], fr.preds['msrp0:comb'])
    dep = {
        'ESM-MSR (Tobit-linked)': linked(msr_raw, sp['dG_wt'].to_numpy(float)),
        'ESM-MSR (raw)': msr_raw,
        **{name: np.where(n1, fr.preds[a], fr.preds[b]) for name, a, b in others},
    }
    for k, v in dep.items():
        fr.preds['dep:' + k] = v

    def _tails(r: np.ndarray) -> tuple[float, float, float, float]:
        r = r[np.isfinite(r)]
        one = float(max((r <= 0).mean(), 1.0 / (len(r) + 1)))
        two = float(max(2 * min((r <= 0).mean(), (r >= 0).mean()), 1.0 / (len(r) + 1)))
        lo, hi = np.percentile(r, [2.5, 97.5])
        return one, two, float(lo), float(hi)

    # Part A: domain-mean Spearman rho
    b_dom = B if B <= 200 else max(B, 10000)
    libs = sorted(np.unique(g[use]))
    counts = make_domain_counts(libs, B=b_dom, seed=5)
    V = {k: per_library(fr, 'dep:' + k, use, rho_fn, 3) for k in dep}
    d_rows = []
    for name, _, _ in others:
        both = V['ESM-MSR (raw)'].index.intersection(V[name].index)
        d = diff_df(V['ESM-MSR (raw)'], V[name], both)
        one, two, lo, hi = _tails(boot_reps(d, libs, counts))
        dom = d.mean(axis=1)
        d_rows.append(dict(
            model=name, n_domains=len(both),
            rho_msr=float(V['ESM-MSR (raw)'].loc[both].mean(axis=1).mean()),
            rho_other=float(V[name].loc[both].mean(axis=1).mean()),
            delta=float(dom.mean()), lo=lo, hi=hi,
            domains_msr_better=int((dom > 0).sum()),
            p_one=one, p=two, B=b_dom,
        ))
    df_dom = pd.DataFrame(d_rows)
    df_dom['p_one_holm'] = holm(df_dom['p_one'].to_numpy())
    df_dom['p_holm'] = holm(df_dom['p'].to_numpy())
    atomic_to_csv(df_dom, out / 'overall_vs_transfer_domain_mean.csv', index=False, float_format='%.5g')

    # Part B: pooled Spearman rho using O(n) SpearmanEngine + joint triplicate seed resampling!
    idx = np.flatnonzero(use & np.isfinite(y))
    gi = g[idx]
    libs_p = sorted(np.unique(gi))
    by = [np.flatnonzero(gi == lib) for lib in libs_p]
    sizes = np.array([len(b) for b in by])
    order = np.concatenate(by)
    dom_of = np.repeat(np.arange(len(libs_p)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    eng = SpearmanEngine(y[idx], np.zeros(len(idx), np.int64), 1)
    names = list(dep)
    xrs = {k: [eng.prepare(p[idx]) for p in np.atleast_2d(dep[k])] for k in names}
    w_one = np.ones(len(idx), dtype=float)
    ry_one = eng.ry.ranks(w_one)
    pt = {k: float(np.mean([eng.rho(xr, w_one, ry_one)[0] for xr in xrs[k]])) for k in names}

    rng = np.random.default_rng(9)
    reps_pool: dict[str, dict[str, np.ndarray]] = {
        'mutation': {k: np.empty(B, dtype=float) for k in names},
        'domain': {k: np.empty(B, dtype=float) for k in names},
    }
    for bi in range(B):
        sw3 = rng.multinomial(3, np.full(3, 1.0 / 3.0)).astype(float) / 3.0

        # Stratified within-domain mutation resampling + seed resampling
        pick_m = starts[dom_of] + np.floor(rng.random(len(order)) * sizes[dom_of]).astype(np.int64)
        w_mut = np.bincount(order[pick_m], minlength=len(idx)).astype(float)
        ry_mut = eng.ry.ranks(w_mut)
        for k in names:
            rs = np.array([eng.rho(xr, w_mut, ry_mut)[0] for xr in xrs[k]], dtype=float)
            reps_pool['mutation'][k][bi] = float(np.dot(sw3, rs)) if len(rs) == 3 else float(np.mean(rs))

        # Whole-domain resampling + seed resampling
        cnt_d = rng.multinomial(len(libs_p), np.full(len(libs_p), 1.0 / len(libs_p))).astype(float)
        w_dom = np.empty(len(idx), dtype=float)
        w_dom[order] = cnt_d[dom_of]
        ry_dom = eng.ry.ranks(w_dom)
        for k in names:
            rs = np.array([eng.rho(xr, w_dom, ry_dom)[0] for xr in xrs[k]], dtype=float)
            reps_pool['domain'][k][bi] = float(np.dot(sw3, rs)) if len(rs) == 3 else float(np.mean(rs))

    p_rows = []
    for a in ('ESM-MSR (Tobit-linked)', 'ESM-MSR (raw)'):
        for name, _, _ in others:
            row = dict(esm=a, model=name, rho_msr=pt[a], rho_other=pt[name], delta=pt[a] - pt[name], B=B)
            for unit in ('mutation', 'domain'):
                one, two, lo, hi = _tails(reps_pool[unit][a] - reps_pool[unit][name])
                row.update({f'p_one_{unit}': one, f'p_{unit}': two, f'lo_{unit}': lo, f'hi_{unit}': hi})
            p_rows.append(row)
    df_pool = pd.DataFrame(p_rows)
    for _a, sub in df_pool.groupby('esm'):
        for unit in ('mutation', 'domain'):
            df_pool.loc[sub.index, f'p_one_holm_{unit}'] = holm(sub[f'p_one_{unit}'].to_numpy())
            df_pool.loc[sub.index, f'p_holm_{unit}'] = holm(sub[f'p_{unit}'].to_numpy())
    atomic_to_csv(df_pool, out / 'overall_vs_transfer_pooled.csv', index=False, float_format='%.5g')

    head1 = [
        'ESM-MSR against', 'ρ\nESM-MSR', 'ρ\nother', 'Δρ [95% CI]', 'Domains where\nESM-MSR is higher',
        'p\none-sided', 'p one-sided\n(Holm)', 'p\ntwo-sided', 'p two-sided\n(Holm)',
    ]
    rows1 = [
        [
            r.model, f'{r.rho_msr:.3f}', f'{r.rho_other:.3f}', f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]',
            f'{r.domains_msr_better} / {r.n_domains}',
            fmt_p(r.p_one, B=r.B, prefix=''), fmt_p(r.p_one_holm, B=r.B, prefix=''),
            fmt_p(r.p, B=r.B, prefix=''), fmt_p(r.p_holm, B=r.B, prefix=''),
        ]
        for r in df_dom.itertuples()
    ]
    note1 = (
        'Tsuboyama capped test set, all variants (singles and doubles), each model read as deployed (singles version on singles, doubles-trained version on doubles). '
        'ρ: mean over test domains of the per-domain Spearman ρ, seeds averaged;\n'
        f'Δρ = ESM-MSR − other. Paired bootstrap over domains and seeds ({b_dom:,} draws shared by all models). '
        'One-sided p = P(Δρ ≤ 0); two-sided p = twice the smaller tail. Holm over the three comparisons.'
    )
    _render_stat_table(rows1, head1, [3.6, 0.95, 0.95, 2.6, 1.7, 1.0, 1.3, 1.0, 1.3], out / 'overall_vs_transfer_domain_mean', note1)
    _write_md_table(rows1, head1, note1, out / 'overall_vs_transfer_domain_mean.md')

    L = df_pool[df_pool['esm'] == 'ESM-MSR (Tobit-linked)']
    head2 = [
        'ESM-MSR\n(Tobit-linked) against', 'ρ\nESM-MSR', 'ρ\nother',
        'Δρ [95% CI]\nmutation resampling', 'p one-sided\n(Holm)', 'p two-sided\n(Holm)',
        'Δρ [95% CI]\ndomain resampling', 'p one-sided\n(Holm)', 'p two-sided\n(Holm)',
    ]
    rows2 = [
        [
            r.model, f'{r.rho_msr:.3f}', f'{r.rho_other:.3f}',
            f'{r.delta:+.3f}  [{r.lo_mutation:+.3f}, {r.hi_mutation:+.3f}]',
            fmt_p(r.p_one_holm_mutation, B=r.B, prefix=''), fmt_p(r.p_holm_mutation, B=r.B, prefix=''),
            f'{r.delta:+.3f}  [{r.lo_domain:+.3f}, {r.hi_domain:+.3f}]',
            fmt_p(r.p_one_holm_domain, B=r.B, prefix=''), fmt_p(r.p_holm_domain, B=r.B, prefix=''),
        ]
        for r in L.itertuples()
    ]
    note2 = (
        'Tsuboyama capped test set, all variants, each model read as deployed. ρ: one Spearman ρ over all test variants, seeds averaged; '
        'ESM-MSR with the post hoc Tobit link h(dG_wt + latent) − dG_wt, the other models raw. Δρ = ESM-MSR − other.\n'
        f'Joint bootstrap of {B:,} draws shared by all models (resampling triplicate seeds jointly): mutation resampling resamples rows inside every domain '
        '(domain composition fixed); domain resampling resamples whole domains. One-sided p = P(Δρ ≤ 0); Holm over the three comparisons.'
    )
    _render_stat_table(rows2, head2, [3.6, 0.95, 0.95, 2.6, 1.2, 1.2, 2.6, 1.2, 1.2], out / 'overall_vs_transfer_pooled', note2)
    _write_md_table(rows2, head2, note2, out / 'overall_vs_transfer_pooled.md')
    return df_dom, df_pool


# ----------------------------------------------------------------------------------------------
# 4. mega_splits overview, capped-vs-mega comparison, and SPURS/ThermoMPNN version comparison
# ----------------------------------------------------------------------------------------------
def run_mega_splits(B: int) -> None:
    """Consolidate tsuboyama/mega_splits/01_rho_overview.py and 03_version_comparison.py."""
    out = results_dir('megascale', split='mega')
    ds_map = {'val': TSU_MEGA[0], 'test': TSU_MEGA[1]}
    fr = {s: load_frame(ds_map[s]) for s in ('val', 'test')}
    keys = [k for k in MEGA_KEYS if k in fr['test'].preds]
    label_map = {
        'msrp0:wt': 'ESM-MSR WT head',
        'msrp0:comb': 'ESM-MSR combined',
        **{k: MODELS[k]['label'] for k in keys if k not in ('msrp0:wt', 'msrp0:comb')},
        'spurs': 'SPURS, frozen adapter (retrain)',
        'spursm': 'SPURS multi, frozen adapter (retrain)',
    }

    def _score(frames_d: dict[str, Frame], kind: str = 'protein', use_keys=None) -> tuple[pd.DataFrame, dict]:
        ks = use_keys or keys
        point, reps = bootstrap_rho(frames_d, ks, kind=kind, B=B, seed=18, per_protein=True)
        rows = []
        for ds_k in frames_d:
            for k in ks:
                if k not in point[ds_k]:
                    continue
                r_v, seeds, n, ng = point[ds_k][k]
                lo, hi = ci(reps[ds_k][k])
                rows.append(dict(
                    dataset=ds_k, model=k, rho=r_v, lo=lo, hi=hi, n=n, n_libraries=ng,
                    n_seeds=len(np.atleast_1d(seeds)),
                    seed_rhos=';'.join(f'{x:.4f}' for x in np.atleast_1d(seeds)),
                ))
        return pd.DataFrame(rows), reps

    P, reps = _score(fr, kind='protein')
    Mu, _ = _score(fr, kind='mutation')
    P['kind'], Mu['kind'] = 'library', 'mutation'
    atomic_to_csv(pd.concat([P, Mu], ignore_index=True), out / 'megascale_mega_overall.csv', index=False, float_format='%.4f')

    p_lk = {(r.dataset, r.model): r for r in P.itertuples()}
    m_lk = {(r.dataset, r.model): r for r in Mu.itertuples()}
    ord_keys = sorted(keys, key=lambda k: p_lk[('test', k)].rho)
    fig_ov, axes_ov = plt.subplots(1, 2, figsize=(9, 0.36 * len(ord_keys) + 1.4), sharey=True)
    for ax, s in zip(axes_ov, ('val', 'test')):
        for i, k in enumerate(ord_keys):
            r, mm = p_lk[(s, k)], m_lk[(s, k)]
            kw = style.marker_kwargs(k, size=6.5)
            ax.plot([r.lo, r.hi], [i, i], color=kw['color'], lw=2.4, alpha=0.45, solid_capstyle='butt')
            ax.plot([mm.lo, mm.hi], [i, i], color=kw['color'], lw=0.8)
            ax.plot(r.rho, i, **kw)
            ax.text(max(r.hi, mm.hi) + 0.006, i, f'{r.rho:.3f}', va='center', fontsize=7.5, color='#444')
        ax.set_title(f'{s} ({p_lk[(s, ord_keys[0])].n_libraries} libraries)', loc='left')
        ax.grid(axis='x', color=style.GRID, lw=0.6)
        ax.set_xlabel('mean per-library Spearman ρ')
    lo_x = min(r.lo for r in Mu.itertuples()) - 0.02
    hi_x = max(r.hi for r in pd.concat([P, Mu]).itertuples()) + 0.045
    for ax in axes_ov:
        ax.set_xlim(lo_x, hi_x)
    axes_ov[0].set_yticks(range(len(ord_keys)))
    axes_ov[0].set_yticklabels([label_map[k] for k in ord_keys], fontsize=8.5)
    for t, k in zip(axes_ov[0].get_yticklabels(), ord_keys):
        if k in ('msrp0:wt', 'msrp0:comb'):
            t.set_fontweight('bold')
    fig_ov.suptitle(f'Every model trained on mega_splits (three seeds each, B = {B:,}). Thick bar: resampling libraries; thin line: resampling mutations', fontsize=8.5, y=1.01)
    fig_ov.tight_layout()
    style.save(fig_ov, out / 'megascale_mega_overview')

    # Per-library scatter on mega_splits test
    sp_t = fr['test'].spine
    ok_t = sp_t['use'].to_numpy()
    y_t = sp_t['y'].to_numpy()
    libs_t = sorted(sp_t['group'][ok_t].unique())
    retr_all = [k for k in RETR_KEYS + MEGA_EXTRA if k in fr['test'].preds]
    R_ml = pd.DataFrame({
        k: pd.Series({
            g: float(np.mean([spearmanr(p[m], y_t[m])[0] for p in fr['test'].preds[k]]))
            for g in libs_t
            for m in [ok_t & (sp_t['group'] == g).to_numpy() & np.isfinite(fr['test'].preds[k]).all(0)]
        })
        for k in keys
    })
    atomic_to_csv(R_ml, out / 'per_library_rho_mega_test.csv', float_format='%.4f')
    ncol = 6
    nrow = -(-len(retr_all) // ncol)
    fig_ml, axes_ml = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.0 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, k in zip(axes_ml.ravel(), retr_all):
        ax.scatter(R_ml[k], R_ml['msrp0:comb'], s=22, color=MODELS[k]['color'], alpha=0.8, lw=0)
        ax.plot([0.2, 1.0], [0.2, 1.0], color='#999', lw=0.8, ls='--')
        ax.set_title(label_map[k], fontsize=8.5)
        ax.text(0.04, 0.95, f'ESM-MSR higher in {(R_ml["msrp0:comb"] > R_ml[k]).sum()} / {len(R_ml)}', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('ρ of the baseline')
    axes_ml.ravel()[0].set_ylabel('ρ of ESM-MSR combined')
    for ax in axes_ml.ravel()[len(retr_all):]:
        ax.axis('off')
    fig_ml.suptitle('mega_splits test: each dot is one library', fontsize=9, y=1.0)
    fig_ml.tight_layout()
    style.save(fig_ml, out / 'megascale_mega_per_library')

    # Singles vs doubles on mega_splits
    sd_rows = []
    for s in ('val', 'test'):
        sp_s = fr[s].spine
        for nm, nmut in (('singles', 1), ('doubles', 2)):
            sub = subframe(fr[s], (sp_s['n_mut'] == nmut).to_numpy() & sp_s['use'].to_numpy(), keys=keys)
            pt_sd, rp_sd = bootstrap_rho({s: sub}, keys, kind='protein', B=B, seed=6, per_protein=True)
            for k, (r_v, _ps, n, ng) in pt_sd[s].items():
                lo, hi = ci(rp_sd[s][k])
                sd_rows.append(dict(dataset=s, subset=nm, model=k, rho=r_v, lo=lo, hi=hi, n=n, n_libraries=ng))
    SD = pd.DataFrame(sd_rows)
    atomic_to_csv(SD, out / 'megascale_mega_singles_doubles.csv', index=False, float_format='%.4f')
    groups = [f'{s}|{x}' for s in ('val', 'test') for x in ('singles', 'doubles')]
    glabel = [f'{s}\n{x}' for s in ('val', 'test') for x in ('singles', 'doubles')]
    fig_msd, ax_msd = plt.subplots(figsize=(8.5, 4))
    style.dot_panel(
        ax_msd, groups, keys,
        {(f'{r.dataset}|{r.subset}', r.model): r.rho for r in SD.itertuples()},
        {(f'{r.dataset}|{r.subset}', r.model): (r.lo, r.hi) for r in SD.itertuples()},
        highlight=['msrp0:wt', 'msrp0:comb'], ylabel='mean per-library ρ', xlabels=glabel, rot=0, size=6.5,
        ylim=(float(SD['lo'].min()) - 0.03, min(1.0, float(SD['hi'].max()) + 0.03)),
    )
    hs = []
    for k in keys:
        mk = style.marker_kwargs(k)
        mk.pop('ls', None)
        hs.append(Line2D([], [], ls='none', label=label_map[k], **mk))
    ax_msd.legend(handles=hs, loc='lower left', ncol=4, fontsize=7.5)
    fig_msd.tight_layout()
    style.save(fig_msd, out / 'megascale_mega_singles_doubles')

    # Density scatters on mega_splits test
    sc_panels = [
        ('msrp0:wt', 'singles', 'ESM-MSR WT head'),
        ('msrp0:comb', 'doubles', 'ESM-MSR combined'),
        ('me2', 'doubles', 'MutateEverything st2'),
        ('tmpnnd', 'doubles', 'ThermoMPNN + D'),
        ('spursm', 'doubles', 'SPURS multi'),
    ]
    fig_msc, axes_msc = plt.subplots(1, len(sc_panels), figsize=(2.9 * len(sc_panels), 3.2), sharey=True)
    for ax, (k, subset, ttl) in zip(axes_msc, sc_panels):
        pr = fr['test'].preds[k].mean(axis=0)
        m = ok_t & (sp_t['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy() & np.isfinite(pr)
        style.density_scatter(ax, pr[m], y_t[m], s=3, line=True, max_points=40000)
        ax.set_title(f'{ttl}\n{subset} (mean of seeds 1-3)', fontsize=8.5)
        ax.set_xlabel('predicted ddG')
    axes_msc[0].set_ylabel('measured ddG (kcal/mol)')
    fig_msc.tight_layout()
    style.save(fig_msc, out / 'megascale_mega_scatter')

    # Capped vs mega_splits test comparison
    cap_raw = load_frame(TSU_CAPPED[1])
    p0_cap = load_final_frame(TSU_CAPPED[1], families=['FINAL_p0'])
    cap_preds = {k: cap_raw.preds[k] for k in RETR_KEYS}
    ids_c = cap_raw.spine['id'].to_numpy(dtype=object)
    ids_p = p0_cap.spine['id'].to_numpy(dtype=object)
    for h in ('wt', 'comb'):
        cap_preds[f'msrp0:{h}'] = np.vstack([_align_id(ids_c, ids_p, r) for r in p0_cap.preds[f'FINAL_p0:{h}']])
    cap_fr = Frame(cap_raw.name, cap_raw.spine, cap_preds)
    C_df, _ = _score({'capped': cap_fr}, use_keys=MEGA_BASE)
    M_df, _ = _score({'mega': fr['test']}, use_keys=MEGA_BASE)
    CM = pd.concat([C_df, M_df], ignore_index=True)
    atomic_to_csv(CM, out / 'megascale_capped_vs_mega.csv', index=False, float_format='%.4f')

    c_lk = {r.model: r for r in C_df.itertuples()}
    m2_lk = {r.model: r for r in M_df.itertuples()}
    base_ord = sorted(MEGA_BASE, key=lambda k: m2_lk[k].rho)
    fig_cm, ax_cm = plt.subplots(figsize=(6.8, 0.36 * len(base_ord) + 1.3))
    for i, k in enumerate(base_ord):
        kw = style.marker_kwargs(k, size=6.5)
        ax_cm.plot([c_lk[k].rho, m2_lk[k].rho], [i, i], color=kw['color'], lw=1.0, alpha=0.6)
        for r, alpha in ((c_lk[k], 0.35), (m2_lk[k], 1.0)):
            ax_cm.plot([r.lo, r.hi], [i, i], color=kw['color'], lw=2.4, alpha=0.3 * alpha, solid_capstyle='butt')
            ax_cm.plot(r.rho, i, **style.marker_kwargs(k, size=6.5, alpha=alpha))
        ax_cm.text(
            1.0, i, f'{c_lk[k].rho:.3f} → {m2_lk[k].rho:.3f}  ({m2_lk[k].rho - c_lk[k].rho:+.3f})',
            va='center', fontsize=7.5, color='#444', transform=ax_cm.get_yaxis_transform(),
        )
    ax_cm.set_yticks(range(len(base_ord)))
    ax_cm.set_yticklabels([label_map[k] for k in base_ord], fontsize=8.5)
    ax_cm.grid(axis='x', color=style.GRID, lw=0.6)
    ax_cm.set_xlabel('test-set mean per-library Spearman ρ')
    ax_cm.set_title(
        f'faded: Homology-Restricted split ({c_lk[base_ord[0]].n_libraries} test libraries); solid: mega_splits ({m2_lk[base_ord[0]].n_libraries})',
        loc='left', fontsize=8.5, fontweight='normal',
    )
    fig_cm.tight_layout()
    style.save(fig_cm, out / 'megascale_capped_vs_mega')

    # Medians & summary markdown table
    sets_med = {'all': ok_t, 'singles_wt': ok_t & (sp_t['n_mut'] == 1).to_numpy() & sp_t['wt_backbone'].to_numpy().astype(bool)}
    med_rows = []
    for k in keys:
        row_m = dict(model=k)
        for nm, ok_m in sets_med.items():
            per_seed = []
            for p in np.atleast_2d(fr['test'].preds[k]):
                r_list = [
                    spearmanr(p[m], y_t[m])[0]
                    for g in sp_t['group'][ok_m].unique()
                    for m in [ok_m & (sp_t['group'] == g).to_numpy() & np.isfinite(p)] if m.sum() >= 5
                ]
                per_seed.append(np.median(r_list))
            row_m[f'median_{nm}'] = float(np.mean(per_seed))
            row_m[f'median_{nm}_min'] = float(np.min(per_seed))
            row_m[f'median_{nm}_max'] = float(np.max(per_seed))
        med_rows.append(row_m)
    MED = pd.DataFrame(med_rows).set_index('model')
    atomic_to_csv(MED, out / 'megascale_mega_test_medians.csv', float_format='%.4f')

    sd_lk = {(r.dataset, r.subset, r.model): r for r in SD.itertuples()}
    cm_lk = {(r.dataset, r.model): r for r in CM.itertuples()}
    ref = 'msrp0:comb'
    md_lines = [
        '| Model | val ρ | test ρ | test singles | test doubles | test median ρ (all variants) | test median ρ, WT-backbone singles (paper statistic) | Homology-Restricted test ρ | P(ESM-MSR comb > model), test |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for k in sorted(keys, key=lambda k: -p_lk[('test', k)].rho):
        fmt_ci = lambda r: f'{r.rho:.3f} [{r.lo:.3f}, {r.hi:.3f}]'
        d_rep = reps['test'][ref] - reps['test'][k]
        d_rep = d_rep[np.isfinite(d_rep)]
        pr_s = '' if k == ref else f'{float(np.mean(d_rep > 0)):.2f}'
        cap_s = f'{cm_lk[("capped", k)].rho:.3f}' if ('capped', k) in cm_lk else '–'
        md_lines.append(
            f'| {label_map[k]} | {fmt_ci(p_lk[("val", k)])} | {fmt_ci(p_lk[("test", k)])} | '
            f'{sd_lk[("test", "singles", k)].rho:.3f} | {sd_lk[("test", "doubles", k)].rho:.3f} | '
            f'{MED.loc[k, "median_all"]:.3f} | {MED.loc[k, "median_singles_wt"]:.3f} | {cap_s} | {pr_s} |'
        )
    n_ref = p_lk[('test', ref)]
    md_lines += [
        '',
        f'Mean per-library Spearman ρ, seed mean of three seeds per model; [95% interval] from resampling libraries and seeds (B = {B:,}). '
        f'mega_splits test: {n_ref.n_libraries} libraries, {n_ref.n:,} measured variants.',
    ]
    atomic_write_text(out / 'table_mega_split.md', '\n'.join(md_lines) + '\n')

    # Version comparison (03_version_comparison.py)
    ver_name = {
        'spurs': 'SPURS frozen', 'spursu': 'SPURS trained',
        'spursm': 'SPURS multi frozen', 'spursum': 'SPURS multi trained',
        'tmpnn': 'ThermoMPNN retrain', 'tmpnnpub': 'ThermoMPNN published',
        'tmpnnd': 'ThermoMPNN-D retrain', 'tmpnndpub': 'ThermoMPNN-D published',
    }
    ver_pairs = [
        ('spursu', 'spurs'),
        ('spursum', 'spursm'),
        ('tmpnnpub', 'tmpnn'), ('tmpnndpub', 'tmpnnd'),
    ]
    ver_pairs = [(a, b) for a, b in ver_pairs if a in fr['test'].preds and b in fr['test'].preds]
    rng = np.random.default_rng(3)
    v_rows = []
    for sp_name, fr_s in fr.items():
        nm_s = fr_s.spine['n_mut'].to_numpy()
        ok_s = fr_s.spine['use'].to_numpy()
        y_s = fr_s.spine['y'].to_numpy()
        for subset, mask in (('all variants', np.ones(len(nm_s), bool)), ('doubles', nm_s == 2)):
            for a, b in ver_pairs:
                def _lrho(key_v):
                    P_v = np.atleast_2d(fr_s.preds[key_v])
                    out_v = {}
                    for g_v in pd.unique(fr_s.spine['group'][ok_s & mask]):
                        mv = ok_s & mask & (fr_s.spine['group'] == g_v).to_numpy() & np.isfinite(P_v).all(0)
                        if mv.sum() >= 5:
                            out_v[g_v] = float(np.mean([spearmanr(pv[mv], y_s[mv])[0] for pv in P_v]))
                    return pd.Series(out_v)
                ra, rb = _lrho(a), _lrho(b)
                d = (ra - rb).dropna()
                boot = np.array([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
                v_rows.append(dict(
                    split=sp_name, subset=subset, a=a, b=b, mean_diff=float(d.mean()),
                    lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                    libs_a_higher=int((d > 0).sum()), libs_b_higher=int((d < 0).sum()),
                    n_libraries=len(d), rho_a=float(ra.mean()), rho_b=float(rb.mean()),
                ))
    VP = pd.DataFrame(v_rows)
    atomic_to_csv(VP, out / 'version_paired_rho.csv', index=False, float_format='%.4f')

    fig_vp, axes_vp = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    for ax, subset in zip(axes_vp, ('all variants', 'doubles')):
        sub = VP[VP['subset'] == subset]
        labels = [f'{ver_name[a]} − {ver_name[b]}' for a, b in ver_pairs]
        for sp_name, off, col in (('val', -0.12, '#7f8c8d'), ('test', 0.12, '#c0392b')):
            for i, (a, b) in enumerate(ver_pairs):
                r = sub[(sub['split'] == sp_name) & (sub['a'] == a) & (sub['b'] == b)].iloc[0]
                ax.plot([r.lo, r.hi], [i + off] * 2, color=col, lw=2.2, alpha=0.55, solid_capstyle='butt')
                ax.plot(r.mean_diff, i + off, 'o', color=col, ms=5, label=sp_name if i == 0 else None)
        ax.axvline(0, color='#555', lw=0.8)
        ax.set_yticks(range(len(ver_pairs)))
        ax.set_yticklabels(labels if ax is axes_vp[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Δ mean per-library ρ (first − second); 95% interval over libraries')
        ax.set_title(subset, loc='left', fontsize=9)
        ax.grid(axis='x', color=style.GRID, lw=0.6)
    axes_vp[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig_vp.tight_layout()
    style.save(fig_vp, out / 'version_paired_rho')

    # Variant-by-variant agreement matrices
    ag_sets = {
        'singles': (['spurs', 'spursu', 'tmpnn', 'tmpnnpub'], ok_t & (sp_t['n_mut'] == 1).to_numpy()),
        'doubles, epistasis term (multi - own-family singles)': (
            ['spursm:epi', 'spursum:epi', 'tmpnndpub:epi', 'tmpnnd:epi'],
            ok_t & (sp_t['n_mut'] == 2).to_numpy() & np.isfinite(sp_t['dddG_ML'].to_numpy()),
        ),
    }
    ag_rows, mats = [], {}
    for set_nm, (ks_ag, mask_ag) in ag_sets.items():
        ks_ag = [k for k in ks_ag if k in fr['test'].preds]
        mat = pd.DataFrame(index=ks_ag, columns=ks_ag, dtype=float)

        def _corr_diff_seeds(ka: str, kb: str) -> float:
            ra = [r[mask_ag] for r in np.atleast_2d(fr['test'].preds[ka])]
            rb = [r[mask_ag] for r in np.atleast_2d(fr['test'].preds[kb])]
            vals = []
            for ii, xa in enumerate(ra):
                for jj, xb in enumerate(rb):
                    if len(ra) > 1 and len(rb) > 1 and ii == jj:
                        continue
                    f = np.isfinite(xa) & np.isfinite(xb)
                    vals.append(spearmanr(xa[f], xb[f])[0])
            return float(np.mean(vals))

        for ka in ks_ag:
            for kb in ks_ag:
                mat.loc[ka, kb] = _corr_diff_seeds(ka, kb) if ka != kb else np.nan
        for ka, kb in combinations(ks_ag, 2):
            ag_rows.append(dict(set=set_nm, a=ka, b=kb, spearman=float(mat.loc[ka, kb])))
        for ka in ks_ag:
            Pa = [r[mask_ag] for r in np.atleast_2d(fr['test'].preds[ka])]
            if len(Pa) > 1:
                rs = [spearmanr(Pa[i][f], Pa[j][f])[0] for i, j in combinations(range(len(Pa)), 2) for f in [np.isfinite(Pa[i]) & np.isfinite(Pa[j])]]
                diag = float(np.mean(rs))
                ag_rows.append(dict(set=set_nm, a=ka, b=ka + ' (other seed)', spearman=diag))
                mat.loc[ka, ka] = diag
        mats[set_nm] = mat

    AG = pd.DataFrame(ag_rows)
    atomic_to_csv(AG, out / 'version_agreement.csv', index=False, float_format='%.4f')
    fig_ag, axes_ag = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4.2))
    for ax, (nm_ag, m_df) in zip(np.atleast_1d(axes_ag), mats.items()):
        ax.imshow(m_df.to_numpy(float), vmin=0.4, vmax=1.0, cmap='viridis')
        ax.set_xticks(range(len(m_df)))
        ax.set_yticks(range(len(m_df)))
        ax.set_xticklabels([ver_name.get(k.split(':')[0], k) for k in m_df.columns], rotation=60, ha='right', fontsize=7.5)
        ax.set_yticklabels([ver_name.get(k.split(':')[0], k) for k in m_df.index], fontsize=7.5)
        for i in range(len(m_df)):
            for j in range(len(m_df)):
                v_ij = m_df.iloc[i, j]
                if np.isfinite(v_ij):
                    ax.text(j, i, f'{v_ij:.2f}', ha='center', va='center', fontsize=7, color='w')
        ax.set_title(nm_ag, fontsize=8.5, loc='left')
    fig_ag.tight_layout()
    style.save(fig_ag, out / 'version_agreement')


def build_supp_backbone_and_origin_figure(split: str = 'capped', B: int = 10_000) -> Path:
    """Build 2-panel supplementary ranking figure on Homology-Restricted MegaScale test:
      - Panel a: Native vs destabilized backbones (per-library Spearman rho across 30 native vs 7 destabilized libraries)
      - Panel b: Natural vs synthetic (de novo) proteins (per-domain Spearman rho across 20 natural vs 13 synthetic domains)
    No overall figure title; exports *_sourcedata.csv (with Pearson r, RMSE, n) and *_pvalues.csv + annotation.
    """
    import re as _re
    from scipy.stats import pearsonr
    from msr_analysis.stats import boot_domain_seeds, make_domain_seed_draws

    fr, _ = load_tsu_test_frame(split=split)
    sp = fr.spine
    use = sp['use'].to_numpy(dtype=bool)
    y = sp['y'].to_numpy(float)
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols = [c for c in main_columns(fr, best, 'unmasked') if c.get('draw', True)]

    def _pearson(p: np.ndarray, yy: np.ndarray) -> float:
        return float(pearsonr(p, yy)[0]) if len(yy) >= 3 and np.ptp(p) > 0 and np.ptp(yy) > 0 else np.nan

    def _rmse(p: np.ndarray, yy: np.ndarray) -> float:
        return float(np.sqrt(np.mean((p - yy) ** 2))) if len(yy) >= 1 else np.nan

    def _per_unit(key: str, mask: np.ndarray, unit_col: str, fn) -> pd.DataFrame:
        g = sp[unit_col].astype(str).to_numpy()
        arr = fr.preds[key]
        out_d = {}
        for u in sorted(np.unique(g[mask])):
            m = mask & (g == u) & np.isfinite(arr).all(axis=0) & np.isfinite(y)
            if m.sum() < MIN_N:
                continue
            vals = [fn(pr[m], y[m]) for pr in arr]
            if np.isfinite(vals).any():
                out_d[u] = vals
        return pd.DataFrame.from_dict(out_d, orient='index', dtype=float) if out_d else pd.DataFrame(dtype=float)

    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    # Masks: Panel a (per library `code`) and Panel b (per domain `group`)
    wt_bb = sp['wt_backbone'].to_numpy(dtype=bool)
    grp_s = sp['group'].astype(str)
    is_nat = (grp_s.str.match(r'^[0-9][A-Za-z0-9]{3}$') | (grp_s == 'v2_2LC2')).to_numpy(dtype=bool)

    panel_specs = [
        (
            'backbone', 'code',
            [('native', use & wt_bb), ('destabilized', use & ~wt_bb), ('overall', use)],
        ),
        (
            'origin', 'group',
            [('natural', use & is_nat), ('synthetic', use & ~is_nat), ('overall', use)],
        ),
    ]

    summary_rows = []
    per_unit_rows = []
    p_rows = []

    for p_idx, (panel_key, unit_col, sub_list) in enumerate(panel_specs):
        for s_idx, (sub_name, m_sub) in enumerate(sub_list):
            vals_rho: dict[str, pd.DataFrame] = {}
            mean_r: dict[str, float] = {}
            mean_rmse: dict[str, float] = {}
            n_var = int((m_sub & np.isfinite(y)).sum())

            for c in cols:
                cname = c['name']
                disp = name_map.get(cname, cname)
                k = c.get('all', c['single'])
                df_rho = _per_unit(k, m_sub, unit_col, rho_fn)
                df_r = _per_unit(k, m_sub, unit_col, _pearson)
                df_e = _per_unit(k, m_sub, unit_col, _rmse)
                vals_rho[disp] = df_rho
                mean_r[disp] = float(df_r.mean(axis=1).mean()) if len(df_r) else np.nan
                mean_rmse[disp] = float(df_e.mean(axis=1).mean()) if len(df_e) else np.nan
                for u_id, row in df_rho.iterrows():
                    for si, v in enumerate(row.values):
                        per_unit_rows.append(dict(
                            panel=panel_key, subset=sub_name, unit_col=unit_col, unit=u_id,
                            column=cname, model=disp, key=k, seed=si + 1,
                            spearman_rho=float(v),
                            pearson_r=float(df_r.loc[u_id].iloc[si]) if u_id in df_r.index else np.nan,
                            rmse=float(df_e.loc[u_id].iloc[si]) if u_id in df_e.index else np.nan,
                        ))

            models_disp = [name_map.get(c['name'], c['name']) for c in cols]
            common_units = sorted(set.intersection(*(set(vals_rho[m].index) for m in models_disp)))
            n_u = len(common_units)
            # Parent-domain count for reporting
            parent_doms = sp.loc[m_sub & sp[unit_col].astype(str).isin(common_units), 'group'].astype(str).nunique()
            d_cnt, s_w = make_domain_seed_draws(n_u, B=B, seed=500 + 20 * p_idx + s_idx, seed_counts=(3,))
            pt_map, rep_map = {}, {}

            for c in cols:
                cname = c['name']
                disp = name_map.get(cname, cname)
                k = c.get('all', c['single'])
                arr = vals_rho[disp].reindex(common_units).to_numpy(float)
                pt, rp = boot_domain_seeds(arr, B=B, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
                pt_map[disp], rep_map[disp] = pt, rp
                lo, hi = ci(rp)
                summary_rows.append(dict(
                    panel=panel_key, subset=sub_name, column=cname, model=disp, key=k,
                    value=pt, lo=lo, hi=hi,
                    pearson_r=mean_r[disp], rmse=mean_rmse[disp],
                    n_libraries=n_u, n_domains=int(parent_doms), n_variants=n_var, n_seeds=int(arr.shape[1]),
                ))

            # Paired one-sided bootstrap p-values: ESM-MSR > X
            comps = [m for m in models_disp if m != 'ESM-MSR']
            m_clean = len(comps)
            p_by_c = {}
            d_list = []
            for comp in comps:
                d_arr = rep_map['ESM-MSR'] - rep_map[comp]
                p_one, nf, b_eff = _one_sided_p(d_arr)
                p_by_c[comp] = p_one
                d_list.append(d_arr)
                lo_d, hi_d = ci(d_arr)
                p_rows.append(dict(
                    benchmark=f'MegaScale_capped_test_{panel_key}',
                    subset_or_metric=f'per_{unit_col}_mean_spearman_{sub_name}',
                    resampling_unit=f'{unit_col}_x_seed',
                    test_type='pairwise',
                    msr_model='ESM-MSR',
                    comparator=comp,
                    direction='ESM-MSR > comparator',
                    value_msr=pt_map['ESM-MSR'],
                    value_comparator=pt_map[comp],
                    pearson_r_msr=mean_r['ESM-MSR'],
                    pearson_r_comparator=mean_r[comp],
                    rmse_msr=mean_rmse['ESM-MSR'],
                    rmse_comparator=mean_rmse[comp],
                    delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[comp],
                    ci95_lo=lo_d, ci95_hi=hi_d,
                    n_units=n_u, n_variants=n_var, B=b_eff, n_boot_fail=nf,
                    p_one_sided_uncorrected=p_one, bonferroni_m=m_clean,
                    p_one_sided_bonferroni=min(1.0, m_clean * p_one),
                ))
            best_c = max(comps, key=lambda x: pt_map[x])
            d_nb = rep_map['ESM-MSR'] - rep_map[best_c]
            p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
            lo_nb, hi_nb = ci(d_nb)
            p_rows.append(dict(
                benchmark=f'MegaScale_capped_test_{panel_key}',
                subset_or_metric=f'per_{unit_col}_mean_spearman_{sub_name}',
                resampling_unit=f'{unit_col}_x_seed',
                test_type='next_best_clean',
                msr_model='ESM-MSR', comparator=best_c, direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[best_c],
                pearson_r_msr=mean_r['ESM-MSR'], pearson_r_comparator=mean_r[best_c],
                rmse_msr=mean_rmse['ESM-MSR'], rmse_comparator=mean_rmse[best_c],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[best_c],
                ci95_lo=lo_nb, ci95_hi=hi_nb,
                n_units=n_u, n_variants=n_var, B=b_nb, n_boot_fail=nf_nb,
                p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
            ))
            worst_c = max(comps, key=lambda x: (p_by_c[x], pt_map[x]))
            d_w = rep_map['ESM-MSR'] - rep_map[worst_c]
            p_iut, nf_w, b_w = _one_sided_p(d_w)
            lo_w, hi_w = ci(d_w)
            p_rows.append(dict(
                benchmark=f'MegaScale_capped_test_{panel_key}',
                subset_or_metric=f'per_{unit_col}_mean_spearman_{sub_name}',
                resampling_unit=f'{unit_col}_x_seed',
                test_type='all_clean_IUT',
                msr_model='ESM-MSR', comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})', direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[worst_c],
                pearson_r_msr=mean_r['ESM-MSR'], pearson_r_comparator=mean_r[worst_c],
                rmse_msr=mean_rmse['ESM-MSR'], rmse_comparator=mean_rmse[worst_c],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[worst_c],
                ci95_lo=lo_w, ci95_hi=hi_w,
                n_units=n_u, n_variants=n_var, B=b_w, n_boot_fail=nf_w,
                p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
            ))
            d_min = np.nanmin(np.vstack(d_list), axis=0)
            p_jm, nf_jm, b_jm = _one_sided_p(d_min)
            lo_jm, hi_jm = ci(d_min)
            p_rows.append(dict(
                benchmark=f'MegaScale_capped_test_{panel_key}',
                subset_or_metric=f'per_{unit_col}_mean_spearman_{sub_name}',
                resampling_unit=f'{unit_col}_x_seed',
                test_type='all_clean_joint_max',
                msr_model='ESM-MSR', comparator=f'ALL_CLEAN_JOINT_MAX (max of {m_clean} clean models per draw)',
                direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=np.nan,
                pearson_r_msr=mean_r['ESM-MSR'], pearson_r_comparator=np.nan,
                rmse_msr=mean_rmse['ESM-MSR'], rmse_comparator=np.nan,
                delta_favoring_msr=pt_map['ESM-MSR'] - max(pt_map[x] for x in comps),
                ci95_lo=lo_jm, ci95_hi=hi_jm,
                n_units=n_u, n_variants=n_var, B=b_jm, n_boot_fail=nf_jm,
                p_one_sided_uncorrected=p_jm, bonferroni_m=1, p_one_sided_bonferroni=p_jm,
            ))

    df_sum = pd.DataFrame(summary_rows)
    df_a = df_sum[(df_sum['panel'] == 'backbone') & (df_sum['subset'] != 'overall')]
    ov_a = df_sum[(df_sum['panel'] == 'backbone') & (df_sum['subset'] == 'overall')]
    df_b = df_sum[(df_sum['panel'] == 'origin') & (df_sum['subset'] != 'overall')]
    ov_b = df_sum[(df_sum['panel'] == 'origin') & (df_sum['subset'] == 'overall')]

    w_fig = max(12.2, fig_width(cols))
    fig = plt.figure(figsize=(w_fig, 10.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, left=0.072, right=0.988, top=0.955, bottom=0.115, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    n_nat_bb = int(df_a.loc[df_a['subset'] == 'native', 'n_variants'].max())
    l_nat_bb = int(df_a.loc[df_a['subset'] == 'native', 'n_libraries'].max())
    n_dst_bb = int(df_a.loc[df_a['subset'] == 'destabilized', 'n_variants'].max())
    l_dst_bb = int(df_a.loc[df_a['subset'] == 'destabilized', 'n_libraries'].max())
    n_ov_a = int(ov_a['n_variants'].max())

    style.bar_panel(
        ax_a, cols, df_a, ('native', 'destabilized'),
        (
            f'native backbone (n = {n_nat_bb:,}, {l_nat_bb} libraries)',
            f'destabilized backbone (n = {n_dst_bb:,}, {l_dst_bb} libraries)',
        ),
        "Spearman's ρ (mean over test libraries)", (0.0, 1.15),
        overall_df=ov_a, overall_label=f'overall (n = {n_ov_a:,})',
        tick_fs=8.5,
    )

    n_nat_pr = int(df_b.loc[df_b['subset'] == 'natural', 'n_variants'].max())
    d_nat_pr = int(df_b.loc[df_b['subset'] == 'natural', 'n_libraries'].max())
    n_syn_pr = int(df_b.loc[df_b['subset'] == 'synthetic', 'n_variants'].max())
    d_syn_pr = int(df_b.loc[df_b['subset'] == 'synthetic', 'n_libraries'].max())
    n_ov_b = int(ov_b['n_variants'].max())

    style.bar_panel(
        ax_b, cols, df_b, ('natural', 'synthetic'),
        (
            f'natural proteins (n = {n_nat_pr:,}, {d_nat_pr} domains)',
            f'synthetic / de novo (n = {n_syn_pr:,}, {d_syn_pr} domains)',
        ),
        "Spearman's ρ (mean over test domains)", (0.0, 1.15),
        overall_df=ov_b, overall_label=f'overall (n = {n_ov_b:,})',
        tick_fs=8.5,
    )
    for ax_s in (ax_a, ax_b):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.012, pa.y1 + 0.030, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(0.012, pb.y1 + 0.030, 'b', ha='left', va='top', fontsize=16, fontweight='bold')

    supp_dir = results_dir('supplementary_figs')
    out_path = supp_dir / '_backbone_and_origin'
    style.save(fig, out_path)

    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(df_sum, src_dir / '_backbone_and_origin_summary_sourcedata.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(per_unit_rows), src_dir / '_backbone_and_origin_per_unit_sourcedata.csv', index=False, float_format='%.6f')
    atomic_to_csv(pd.DataFrame(p_rows), pval_dir / '_backbone_and_origin_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_backbone_and_origin_pvalues_annotation.txt',
        f"""Statistical Annotation: _backbone_and_origin_pvalues.csv
========================================================
Figure / Panel: Supplementary Figure (_backbone_and_origin) on Homology-Restricted MegaScale test (`tsuboyama_capped_test`).
  - Panel a: Native wild-type backbones (30 libraries, n = {n_nat_bb:,}) vs destabilized mutant backbones (7 libraries across 4 parent domains: 1YEZ_V46S, 2K52_F21S/L42A, 2K5H_G11S/G23A, 2MKX_L11A/N29S; n = {n_dst_bb:,}).
  - Panel b: Natural proteins (20 domains: 19 PDB domains + N-terminally trimmed construct v2_2LC2; n = {n_nat_pr:,}) vs synthetic / de novo designed proteins (13 domains: 12 Rocklin HEEH/HHH designs + 1 trRosetta hallucination; n = {n_syn_pr:,}).

Metric Tested:
  Mean per-library (Panel a) or per-domain (Panel b) Spearman's rank correlation (rho), with mean Pearson's r and RMSE (kcal/mol) reported alongside.

Statistical Test & Resampling Procedure:
  Paired nonparametric cluster x seed bootstrap (B = {B:,}, deterministic seed).
  In each replicate, units (libraries in Panel a; domains in Panel b) are resampled with replacement using one shared Multinomial weight vector across all 9 models, and 3-seed retrained models are resampled jointly across seeds.
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  Bonferroni adjusts within each subset over m = 8 clean comparators (ProSST, ProteinMPNN, ESM3-small-open, MutateEverything, ThermoMPNN, SPURS, GeoDDG-3D, Rosetta).
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j); `all_clean_joint_max` evaluates min_j(rho_ESM-MSR^(b) - rho_j^(b)) per draw.
""",
    )
    return out_path


def build_supp_ndcg96_combined(split: str = 'capped', B: int = 10_000) -> Path:
    """Build combined 2-panel supplementary NDCG@96 figure on Homology-Restricted MegaScale test:
      - Panel a: Rectified NDCG@96 (rel = max(ddG, 0); 33 singles domains, 8 doubles domains with >=1 ddG > 0)
      - Panel b: Shifted NDCG@96 (rel = ddG - min(ddG); all 33 singles domains and all 13 doubles domains)
    No overall figure title or footnote annotation; exports *_sourcedata.csv and *_pvalues.csv + annotation.
    """
    from scipy.stats import pearsonr
    from msr_analysis.stats import boot_domain_seeds, make_domain_seed_draws

    fr, _ = load_tsu_test_frame(split=split)
    sp = fr.spine
    use = sp['use'].to_numpy(dtype=bool)
    n_mut = sp['n_mut'].to_numpy()
    single, double = use & (n_mut == 1), use & (n_mut == 2)
    y = sp['y'].to_numpy(float)
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = make_domain_counts(libs, B=B, seed=5)

    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols = [c for c in main_columns(fr, best, 'unmasked') if c.get('draw', True)]

    def _pearson(p: np.ndarray, yy: np.ndarray) -> float:
        return float(pearsonr(p, yy)[0]) if len(yy) >= 3 and np.ptp(p) > 0 and np.ptp(yy) > 0 else np.nan

    def _rmse(p: np.ndarray, yy: np.ndarray) -> float:
        return float(np.sqrt(np.mean((p - yy) ** 2))) if len(yy) >= 1 else np.nan

    def _one_sided_p(diffs: np.ndarray) -> tuple[float, int, int]:
        fin = diffs[np.isfinite(diffs)]
        nf = int(np.sum(fin <= 0.0))
        return float((nf + 1) / (len(fin) + 1)), nf, int(len(fin))

    # Panel a: Rectified NDCG@96
    cells = {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}
    T_rect = table_grouped(fr, cols, cells, ndcg96_rectified, 1, counts, libs)
    ov_rect = table_grouped(fr, cols, {'overall': (use, 'all')}, ndcg96_rectified, 1, counts, libs)
    d_rect = (ov_rect.set_index('column')['value'] - T_rect[T_rect['subset'] == 'singles'].set_index('column')['value']).dropna()
    draw_ov_rect = int((d_rect.abs() <= 0.01).sum()) < 0.9 * len(d_rect)

    # Panel b: Shifted NDCG@96
    T_shft = table_grouped(fr, cols, cells, ndcg96_shifted, MIN_N, counts, libs)
    ov_shft = table_grouped(fr, cols, {'overall': (use, 'all')}, ndcg96_shifted, MIN_N, counts, libs)
    d_shft = (ov_shft.set_index('column')['value'] - T_shft[T_shft['subset'] == 'singles'].set_index('column')['value']).dropna()
    draw_ov_shft = int((d_shft.abs() <= 0.01).sum()) < 0.9 * len(d_shft)

    w_fig = max(12.5, fig_width(cols))
    fig = plt.figure(figsize=(w_fig, 10.6), facecolor='white')
    gs = fig.add_gridspec(2, 1, left=0.072, right=0.988, top=0.955, bottom=0.115, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    style.bar_panel(
        ax_a, cols, T_rect, ('singles', 'doubles'),
        (count_text(T_rect, 'singles', 'singles'), count_text(T_rect, 'doubles', 'doubles')),
        f'Rectified NDCG@{K_NDCG} (mean over test domains)', (0.0, 1.12),
        overall_df=ov_rect if draw_ov_rect else None,
        overall_label=f'overall (n = {int(ov_rect.n_variants.max()):,})',
        tick_fs=8.5,
    )
    style.bar_panel(
        ax_b, cols, T_shft, ('singles', 'doubles'),
        (count_text(T_shft, 'singles', 'singles'), count_text(T_shft, 'doubles', 'doubles')),
        f'Shifted NDCG@{K_NDCG} (mean over test domains)', (0.0, 1.12),
        overall_df=ov_shft if draw_ov_shft else None,
        overall_label=f'overall (n = {int(ov_shft.n_variants.max()):,})',
        tick_fs=8.5,
    )
    for ax_s in (ax_a, ax_b):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    fig.canvas.draw()
    pa, pb = ax_a.get_position(), ax_b.get_position()
    fig.text(0.012, pa.y1 + 0.030, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(0.012, pb.y1 + 0.030, 'b', ha='left', va='top', fontsize=16, fontweight='bold')

    supp_dir = results_dir('supplementary_figs')
    out_path = supp_dir / '_ndcg96_singles_doubles'
    style.save(fig, out_path)

    # Remove standalone rectified duplicate if present
    for ext in ('.png', '.pdf'):
        dup = supp_dir / f'_ndcg96_rectified_singles_doubles{ext}'
        if dup.exists():
            dup.unlink()

    # Export enriched source data + B=10,000 paired one-sided bootstrap p-values
    src_dir = supp_dir / 'source_data'
    pval_dir = supp_dir / 'pvalues'
    src_dir.mkdir(parents=True, exist_ok=True)
    pval_dir.mkdir(parents=True, exist_ok=True)

    cells_full = {'singles': (single, 'singles'), 'doubles': (double, 'doubles'), 'overall': (use, 'all')}
    per_dom_rows = []
    sum_parts = []
    p_rows = []

    for m_idx, (mode_name, metric_fn, mn, df_pair, df_ov) in enumerate((
        ('rectified', ndcg96_rectified, 1, T_rect, ov_rect),
        ('shifted', ndcg96_shifted, MIN_N, T_shft, ov_shft),
    )):
        df_m = pd.concat([df_pair, df_ov], ignore_index=True).copy()
        df_m.insert(0, 'mode', mode_name)
        df_m['model'] = df_m['column'].map(lambda x: name_map.get(x, x))
        r_col, e_col = [], []
        vals_sub: dict[tuple[str, str], pd.DataFrame] = {}
        r_map, e_map = {}, {}
        n_var_map = {}

        for sub_name, (m_sub, which) in cells_full.items():
            n_var_map[sub_name] = int((m_sub & np.isfinite(y)).sum())
            for c in cols:
                disp = name_map.get(c['name'], c['name'])
                k = c.get('all', c['single']) if which == 'all' else (c['single'] if which == 'singles' else c['double'])
                df_v = per_library(fr, k, m_sub, metric_fn, mn)
                df_r = per_library(fr, k, m_sub, _pearson, mn)
                df_e = per_library(fr, k, m_sub, _rmse, mn)
                vals_sub[(sub_name, disp)] = df_v
                r_map[(sub_name, disp)] = float(df_r.mean(axis=1).mean()) if len(df_r) else np.nan
                e_map[(sub_name, disp)] = float(df_e.mean(axis=1).mean()) if len(df_e) else np.nan
                for dom, row in df_v.iterrows():
                    for si, v in enumerate(row.values):
                        per_dom_rows.append(dict(
                            mode=mode_name, subset=sub_name, model=disp, column=c['name'], key=k,
                            domain=dom, seed=si + 1, ndcg96=float(v),
                            pearson_r=float(df_r.loc[dom].iloc[si]) if dom in df_r.index else np.nan,
                            rmse=float(df_e.loc[dom].iloc[si]) if dom in df_e.index else np.nan,
                        ))

        for row in df_m.itertuples(index=False):
            r_col.append(r_map.get((row.subset, row.model), np.nan))
            e_col.append(e_map.get((row.subset, row.model), np.nan))
        df_m['pearson_r'] = r_col
        df_m['rmse'] = e_col
        sum_parts.append(df_m)

        models_disp = [name_map.get(c['name'], c['name']) for c in cols]
        comps = [m for m in models_disp if m != 'ESM-MSR']
        m_clean = len(comps)
        for s_idx, sub_name in enumerate(('singles', 'doubles', 'overall')):
            common_doms = sorted(set.intersection(*(set(vals_sub[(sub_name, m)].index) for m in models_disp)))
            d_cnt, s_w = make_domain_seed_draws(len(common_doms), B=B, seed=700 + 20 * m_idx + s_idx, seed_counts=(3,))
            pt_map, rep_map = {}, {}
            for m in models_disp:
                arr = vals_sub[(sub_name, m)].reindex(common_doms).to_numpy(float)
                pt_map[m], rep_map[m] = boot_domain_seeds(arr, B=B, counts=d_cnt, seed_weights=s_w if arr.shape[1] > 1 else None)
            p_by_c, d_list = {}, []
            for comp in comps:
                d_arr = rep_map['ESM-MSR'] - rep_map[comp]
                p_one, nf, b_eff = _one_sided_p(d_arr)
                p_by_c[comp] = p_one
                d_list.append(d_arr)
                lo_d, hi_d = ci(d_arr)
                p_rows.append(dict(
                    benchmark='MegaScale_capped_test', subset_or_metric=f'ndcg96_{mode_name}_{sub_name}',
                    resampling_unit='domain_cluster_x_seed', test_type='pairwise',
                    msr_model='ESM-MSR', comparator=comp, direction='ESM-MSR > comparator',
                    value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[comp],
                    pearson_r_msr=r_map[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_map[(sub_name, comp)],
                    rmse_msr=e_map[(sub_name, 'ESM-MSR')], rmse_comparator=e_map[(sub_name, comp)],
                    delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[comp], ci95_lo=lo_d, ci95_hi=hi_d,
                    n_units=len(common_doms), n_variants=n_var_map[sub_name], B=b_eff, n_boot_fail=nf,
                    p_one_sided_uncorrected=p_one, bonferroni_m=m_clean,
                    p_one_sided_bonferroni=min(1.0, m_clean * p_one),
                ))
            best_c = max(comps, key=lambda x: pt_map[x])
            d_nb = rep_map['ESM-MSR'] - rep_map[best_c]
            p_nb, nf_nb, b_nb = _one_sided_p(d_nb)
            lo_nb, hi_nb = ci(d_nb)
            p_rows.append(dict(
                benchmark='MegaScale_capped_test', subset_or_metric=f'ndcg96_{mode_name}_{sub_name}',
                resampling_unit='domain_cluster_x_seed', test_type='next_best_clean',
                msr_model='ESM-MSR', comparator=best_c, direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[best_c],
                pearson_r_msr=r_map[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_map[(sub_name, best_c)],
                rmse_msr=e_map[(sub_name, 'ESM-MSR')], rmse_comparator=e_map[(sub_name, best_c)],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[best_c], ci95_lo=lo_nb, ci95_hi=hi_nb,
                n_units=len(common_doms), n_variants=n_var_map[sub_name], B=b_nb, n_boot_fail=nf_nb,
                p_one_sided_uncorrected=p_nb, bonferroni_m=m_clean,
                p_one_sided_bonferroni=min(1.0, m_clean * p_nb),
            ))
            worst_c = max(comps, key=lambda x: (p_by_c[x], pt_map[x]))
            d_w = rep_map['ESM-MSR'] - rep_map[worst_c]
            p_iut, nf_w, b_w = _one_sided_p(d_w)
            lo_w, hi_w = ci(d_w)
            p_rows.append(dict(
                benchmark='MegaScale_capped_test', subset_or_metric=f'ndcg96_{mode_name}_{sub_name}',
                resampling_unit='domain_cluster_x_seed', test_type='all_clean_IUT',
                msr_model='ESM-MSR', comparator=f'ALL_CLEAN_IUT (limiting: {worst_c})', direction='ESM-MSR > comparator',
                value_msr=pt_map['ESM-MSR'], value_comparator=pt_map[worst_c],
                pearson_r_msr=r_map[(sub_name, 'ESM-MSR')], pearson_r_comparator=r_map[(sub_name, worst_c)],
                rmse_msr=e_map[(sub_name, 'ESM-MSR')], rmse_comparator=e_map[(sub_name, worst_c)],
                delta_favoring_msr=pt_map['ESM-MSR'] - pt_map[worst_c], ci95_lo=lo_w, ci95_hi=hi_w,
                n_units=len(common_doms), n_variants=n_var_map[sub_name], B=b_w, n_boot_fail=nf_w,
                p_one_sided_uncorrected=p_iut, bonferroni_m=1, p_one_sided_bonferroni=p_iut,
            ))

    atomic_to_csv(pd.concat(sum_parts, ignore_index=True), src_dir / '_ndcg96_singles_doubles_summary_sourcedata.csv', index=False, float_format='%.4f')
    atomic_to_csv(pd.DataFrame(per_dom_rows), src_dir / '_ndcg96_singles_doubles_per_domain_sourcedata.csv', index=False, float_format='%.6f')
    atomic_to_csv(pd.DataFrame(p_rows), pval_dir / '_ndcg96_singles_doubles_pvalues.csv', index=False, float_format='%.6g')
    atomic_write_text(
        pval_dir / '_ndcg96_singles_doubles_pvalues_annotation.txt',
        f"""Statistical Annotation: _ndcg96_singles_doubles_pvalues.csv
===========================================================
Figure / Panel: Supplementary Figure (_ndcg96_singles_doubles) on Homology-Restricted MegaScale test (`tsuboyama_capped_test`).
  - Panel a: Rectified NDCG@96 (relevance = max(ddG, 0); 33 singles domains, 8 doubles domains containing >=1 stabilizing double, 33 overall domains).
  - Panel b: Shifted NDCG@96 (relevance = ddG - min(ddG) per domain & subset; all 33 singles domains, all 13 doubles domains, all 33 overall domains).

Statistical Test & Resampling Procedure:
  Paired nonparametric domain-cluster x seed bootstrap (B = {B:,}, deterministic seed) over the common test domain set for each subset, with 3-seed retrained models resampled jointly across training seeds.
  One-sided Phipson & Smyth (2010) p-value: p = (1 + #{{b : delta^(b) <= 0}}) / (B + 1).
  Bonferroni correction adjusts within each (mode, subset) over all m = 8 clean comparator models.
  `all_clean_IUT` gives Berger's (1982) Intersection-Union Test (p_IUT = max_j p_j).
""",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = base_parser(STAGE, DESCRIPTION, with_split=True)
    ap.add_argument('--compare-splits', action='store_true', default=True, help='Also generate mega_splits overview & capped-vs-mega comparison (default: True)')
    ap.add_argument('--no-compare-splits', dest='compare_splits', action='store_false', help='Skip mega_splits comparison when running --split capped')
    ap.add_argument('--only', choices=('all', 'supp_figure3', 'backbone_origin', 'ndcg96_combined'), default='all', help='Optional substage selector')
    args = ap.parse_args(argv)
    B = resolve_b(args)

    if args.only == 'ndcg96_combined':
        out = build_supp_ndcg96_combined(split=args.split, B=max(B, 10_000))
        print(f'[p02:tsuboyama] Saved {out}.{{png,pdf}}')
        return 0

    if args.only == 'backbone_origin':
        out = build_supp_backbone_and_origin_figure(split=args.split, B=max(B, 10_000))
        print(f'[p02:tsuboyama] Saved {out}.{{png,pdf}}')
        return 0

    if args.only == 'supp_figure3':
        out = build_supp_figure3(B=B)
        print(f'[p02:tsuboyama] Saved {out}.{{png,pdf}}')
        return 0

    if args.split == 'capped':
        run_capped_core(B=B)
        run_publication_bars(B=B, split='capped')
        run_doubles_vs_singles_on_singles(B=B)
        run_overall_vs_transfer(B=B)
        if args.compare_splits:
            run_mega_splits(B=B)
    else:
        run_mega_splits(B=B)
        run_publication_bars(B=B, split='mega')
    return 0


if __name__ == '__main__':
    sys.exit(main())
