#!/usr/bin/env python
"""Publication figure 6: the epistasis scatters of the notebooks (``compare_supervised_epistasis_rho_test``: density scatter, here without the marginal histograms), on the Homology-Restricted
MegaScale test doubles with both singles measured (12,455, the doubles of the epistasis tables; all backbones are wild type there).

  fig6a  the additive dG of the two measured singles, dG_additive = dG_wt + ddG_A + ddG_B (the double's stability if the substitutions did not interact), against the measured
         epistasis dddG = ddG_AB - ddG_A - ddG_B.  The double's own dG is dddG + dG_additive, so a double that sits at dG_AB = 0, where the unfolded doubles pile up (the median of the doubles at
         dG_additive <= -2 is -0.04), falls on the dashed line y = -x whatever the stability of its domain; the dotted line y = -x - 1 is dG_AB = -1, the lowest value the assay reports (0.2% of the
         doubles are there).  The rise of dddG as dG_additive falls is that floor, not interaction.  (In ddG the floor is at -dG_wt, a different line for every domain.)
  fig6b  ESM-MSR's predicted epistasis, 1/2(MT_AB+WT_AB) - WT_A - WT_B, against the
         measured dddG, with the line x = y.
  fig6c  fig6b with the doubles whose dG_additive is below 0 (already lost to the floor if the singles add) in green.
  fig6d  for the doubles that are not lost to the floor (dG_additive > 0): the additive ddG of the two singles against dddG, with the lines y = 0 (no epistasis) and y = -x (ddG_AB = 0).
  fig6e  fig6d coloured by the signs of the two singles.
  fig6f  the plane of the two singles (ddG_A the more destabilizing one on x, ddG_B on y), coloured by dddG: blue positive, red negative, purple no detectable epistasis, black saturation-dominated.

The dddG correlation plots of every model, with and without those doubles, are ``13_dddG_correlation_scatters.py``.  Outputs: ``results/06_epistasis`` (png, pdf, csv).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.link import singles_of_doubles  # noqa: E402

OUT = out_dir('06_epistasis')
DS = 'tsuboyama_capped_test'
Y_LABEL = 'measured δΔΔG = dddG (kcal/mol)'
TITLE = 'Homology-Restricted MegaScale test doubles with both singles measured'


def main():
    fr = load_frame(DS)
    sp = fr.spine
    m = epi_rows(fr)
    y = sp['dddG_ML'].to_numpy()
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)          # dG_wt is the domain's wild-type stability
    pred_base = fr.preds['msr:epi']
    pred_dev = fr.preds['msrdev:epi']
    pred_p0 = fr.preds['msrp0:epi']
    rows = []
    SCATTER_FONT = dict(
        label_fontsize=13.5,
        tick_fontsize=12.0,
        stats_fontsize=11.5,
        legend_fontsize=10.5,
        foot_fontsize=9.5,
        cbar_fontsize=12.5,
        cbar_tick_fontsize=10.5,
    )
    jobs = [('fig6a_dG_additive_vs_dddG', dG_add, 'ΔG additive = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', dict(ref='neg', ref_label='y = −x: dG_AB = 0,\nwhere the unfolded doubles pile up')),
            ('fig6b_esm_msr_dddG_pred_vs_true', pred_dev, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True)),
            ('fig6c_esm_msr_dddG_pred_vs_true_dG_additive_lt0', pred_dev, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True, highlight=dG_add < 0, highlight_label='ΔG additive < 0')),
            ('fig6b_esm_msr_baseline_dddG_pred_vs_true', pred_base, 'ESM-MSR (baseline) predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True)),
            ('fig6c_esm_msr_baseline_dddG_pred_vs_true_dG_additive_lt0', pred_base, 'ESM-MSR (baseline) predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True, highlight=dG_add < 0, highlight_label='ΔG additive < 0')),
            ('fig6b_esm_msr_p0_dddG_pred_vs_true', pred_p0, 'ESM-MSR (p0) predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True)),
            ('fig6c_esm_msr_p0_dddG_pred_vs_true_dG_additive_lt0', pred_p0, 'ESM-MSR (p0) predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', rmse=True, highlight=dG_add < 0, highlight_label='ΔG additive < 0'))]
    for name, x, xl, kw in jobs:
        if x.ndim == 2:
            ok = m & np.isfinite(x).all(axis=0) & np.isfinite(y)
            xs_in = x[:, ok]
            n_seeds = xs_in.shape[0]
            rho_v = float(np.mean([spearmanr(xs_in[s], y[ok])[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(xs_in[s], y[ok])[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((xs_in[s] - y[ok]) ** 2)) for s in range(n_seeds)])) if kw.get('ref') == 'xy' else np.nan
            rho_pool = float(spearmanr(xs_in.ravel(), np.tile(y[ok], n_seeds))[0])
            r_pool = float(pearsonr(xs_in.ravel(), np.tile(y[ok], n_seeds))[0])
        else:
            ok = m & np.isfinite(x) & np.isfinite(y)
            xs_in = x[ok]
            n_seeds = 1
            rho_v = rho_pool = float(spearmanr(xs_in, y[ok])[0])
            r_v = r_pool = float(pearsonr(xs_in, y[ok])[0])
            rmse_v = float(np.sqrt(np.mean((xs_in - y[ok]) ** 2))) if kw.get('ref') == 'xy' else np.nan
        if 'highlight' in kw:
            kw = {**kw, 'highlight': kw['highlight'][ok]}
        fig = style.density_scatter_plain(xs_in, y[ok], xl, Y_LABEL, title=None, foot=f'{TITLE} (n = {int(ok.sum()):,})', **SCATTER_FONT, **kw)
        if name.startswith('fig6a'):                  # the double's dG is dddG + the additive dG: 0 is where the unfolded doubles pile up, -1 the lowest value the assay reports
            ax = fig.axes[0]
            xl_, yl_ = ax.get_xlim(), ax.get_ylim()
            ax.plot(xl_, [-v - 1.0 for v in xl_], color='#c0392b', ls=':', lw=1.6, zorder=4)
            ax.set_xlim(xl_)
            ax.set_ylim(yl_)
            ax.legend(handles=ax.get_legend().legend_handles + [plt.Line2D([0], [0], color='#c0392b', ls=':', lw=1.6, label='y = −x − 1: dG_AB = −1,\nthe lowest value the assay reports')],
                      loc='lower left', fontsize=10.2, frameon=False, labelspacing=0.45)
        style.save(fig, OUT / name)
        rows.append(dict(figure=name, n=int(ok.sum()), n_seeds=n_seeds, spearman=rho_v, pearson=r_v, spearman_stacked=rho_pool, pearson_stacked=r_pool,
                         share_dG_additive_lt0=float(np.mean(dG_add[ok] < 0)), rmse_to_xy=rmse_v))
    compare_models_scatters(m, y, dG_add, [('ESM-MSR (baseline)', pred_base), ('ESM-MSR', pred_dev), ('ESM-MSR (p0)', pred_p0)], SCATTER_FONT,
                            suffix='', formula='1/2(MT_AB+WT_AB) − WT_A − WT_B (no linking)')
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'fig6_epistasis_scatters.csv', index=False, float_format='%.4f')
    print(T.to_string())
    kinds(fr, m, y, dG_add, SCATTER_FONT)
    singles_plane(fr, m, y, dG_add)


def compare_models_scatters(m, y, dG_add, model_preds, scatter_font, suffix='', formula='1/2(MT_AB+WT_AB) − WT_A − WT_B (no linking)'):
    """Side-by-side 1x3 density scatters comparing ESM-MSR (baseline), ESM-MSR, and ESM-MSR (p0) predicted dddG vs measured dddG across all triplicate seeds without ensemble averaging."""
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde
    ok = m & np.isfinite(y)
    for _, p in model_preds:
        ok &= np.isfinite(p).all(axis=0)
    yv_1d = y[ok]
    hl_1d = (dG_add < 0)[ok]
    n_var = len(yv_1d)
    all_vals = np.concatenate([yv_1d] + [p[:, ok].ravel() for _, p in model_preds])
    lo, hi = np.percentile(all_vals, [0.3, 99.7])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    cm = plt.get_cmap('plasma')

    for fname, use_hl in (
        (f'fig6b_esm_msr_models_dddG_pred_vs_true{suffix}', False),
        (f'fig6c_esm_msr_models_dddG_pred_vs_true_dG_additive_lt0{suffix}', True),
    ):
        fig, axes = plt.subplots(1, len(model_preds), figsize=(18.0, 5.8), sharex=True, sharey=True)
        for idx, (ax, (m_label, p)) in enumerate(zip(axes, model_preds)):
            p_ok = p[:, ok]
            n_seeds = p_ok.shape[0]
            xv = p_ok.ravel()
            yv = np.tile(yv_1d, n_seeds)
            n_pts = len(xv)
            sub = np.random.default_rng(0).choice(n_pts, min(n_pts, 15000), replace=False)
            z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
            o = np.argsort(z)
            norm = Normalize(z.min(), z.max())
            col = cm(norm(z))
            col[:, 3] = 0.3 / np.sqrt(n_seeds)
            hl = np.tile(hl_1d, n_seeds) if use_hl else np.zeros(n_pts, bool)
            keep = o[~hl[o]]
            ax.scatter(xv[keep], yv[keep], c=col[keep], s=8, edgecolors='none', rasterized=True)
            handles = []
            if hl.any():
                ax.scatter(xv[hl], yv[hl], c='#1a9850', s=8, alpha=0.55 / np.sqrt(n_seeds), edgecolors='none', rasterized=True, zorder=3)
                handles.append(plt.Line2D([0], [0], marker='o', ls='', color='#1a9850', alpha=0.8, label=f'ΔG additive < 0 (n = {int(hl_1d.sum()):,})'))
            ax.plot(lim, lim, color='#c0392b', ls='--', lw=1.6, zorder=4)
            handles.append(plt.Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='x = y'))
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f'{m_label} predicted δΔΔG = dddG (kcal/mol)', fontsize=scatter_font['label_fontsize'])
            if idx == 0:
                ax.set_ylabel(Y_LABEL, fontsize=scatter_font['label_fontsize'])
            ax.tick_params(axis='both', labelsize=scatter_font['tick_fontsize'])
            rho_v = float(np.mean([spearmanr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(p_ok[s], yv_1d)[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((p_ok[s] - yv_1d) ** 2)) for s in range(n_seeds)]))
            txt = f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {n_var:,} (×{n_seeds} seeds)\nRMSE = {rmse_v:.2f}'
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=scatter_font['stats_fontsize'], zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94))
            ax.legend(handles=handles, loc='lower right', fontsize=scatter_font['legend_fontsize'], framealpha=0.92)
            ax.set_title(m_label, fontsize=13.5, fontweight='bold', pad=8)
            for sp_ in ('top', 'right'):
                ax.spines[sp_].set_visible(False)
        fig.tight_layout()
        fig.text(0.012, -0.01, f'{TITLE} (n = {n_var:,}, 3 triplicate seeds integrated without ensemble averaging); predicted epistasis = {formula}',
                 fontsize=scatter_font['foot_fontsize'], va='top', ha='left', color='#444444')
        style.save(fig, OUT / fname)


def kinds(fr, m, y, dG_add, scatter_font):
    """fig6d / fig6e: the additive ddG of the two measured singles against the measured dddG for the doubles that are not already lost to the assay floor (dG_additive > 0): which kinds of epistasis
    the data hold.  A double on y = 0 does not interact; on y = -x it is as stable as the wild type (ddG_AB = 0), the strongest compensation the data can show.  fig6e colours the same points by the
    signs of the two singles (stabilizing = ddG > 0)."""
    from matplotlib.lines import Line2D
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    x = sp['ddG_additive_ML'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = m & (dG_add > 0) & np.isfinite(x) & np.isfinite(y) & (ia >= 0) & (ib >= 0)
    xs, ys, dab = x[ok], y[ok], ddg[ok]
    sa, sb = ddg[ia[ok]], ddg[ib[ok]]
    ttl = f'{TITLE}, dG additive > 0 (n = {int(ok.sum()):,})'
    xl = 'ΔΔG additive = ΔΔG_A + ΔΔG_B (kcal/mol)'
    fig = style.density_scatter_plain(xs, ys, xl, Y_LABEL, title=None, foot=ttl, ref='neg', ref_label='y = −x: ΔΔG_AB = 0', **scatter_font)
    ax = fig.axes[0]
    ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
    handles = ax.get_legend().legend_handles + [Line2D([0], [0], color='#555555', lw=1.2, ls=':', label='y = 0: no epistasis')]
    ax.legend(handles=handles, loc='lower left', fontsize=10.5, frameon=False)
    style.save(fig, OUT / 'fig6d_ddG_additive_vs_dddG_dG_additive_gt0')

    kind = np.where((sa > 0) & (sb > 0), 'both stabilizing', np.where((sa > 0) | (sb > 0), 'one stabilizing', 'both destabilizing'))
    colours = {'both destabilizing': '#2c3e50', 'one stabilizing': '#e67e22', 'both stabilizing': '#1a9850'}
    limx, limy = np.percentile(xs, [0.3, 99.7]), np.percentile(ys, [0.3, 99.7])
    padx, pady = 0.05 * (limx[1] - limx[0]), 0.05 * (limy[1] - limy[0])
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    handles, rows = [], []
    for name in ('both destabilizing', 'one stabilizing', 'both stabilizing'):
        k = kind == name
        if not k.any():
            continue
        ax.scatter(xs[k], ys[k], s=8, c=colours[name], alpha=0.3, edgecolors='none', rasterized=True)
        handles.append(Line2D([0], [0], marker='o', ls='', color=colours[name], alpha=0.8, label=f'{name} (n = {int(k.sum()):,})'))
        rows.append(dict(singles=name, n=int(k.sum()), mean_ddG_additive=xs[k].mean(), mean_dddG=ys[k].mean(), share_positive=float(np.mean(ys[k] > 0.5)), share_none=float(np.mean(np.abs(ys[k]) <= 0.5)),
                         share_negative=float(np.mean(ys[k] < -0.5)), share_double_stabilizing=float(np.mean(dab[k] > 0)), spearman=spearmanr(xs[k], ys[k])[0] if k.sum() > 2 else np.nan))
    ax.plot([limx[0] - padx, limx[1] + padx], [-(limx[0] - padx), -(limx[1] + padx)], color='#c0392b', ls='--', lw=1.6, zorder=4)
    ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
    handles += [Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='y = −x: ΔΔG_AB = 0'), Line2D([0], [0], color='#555555', lw=1.2, ls=':', label='y = 0: no epistasis')]
    ax.set_xlim(limx[0] - padx, limx[1] + padx)
    ax.set_ylim(limy[0] - pady, limy[1] + pady)
    ax.set_xlabel(xl, fontsize=13.5)
    ax.set_ylabel(Y_LABEL, fontsize=13.5)
    ax.tick_params(axis='both', labelsize=12.0)
    ax.legend(handles=handles, loc='upper right', fontsize=10.0, framealpha=0.92)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    fig.tight_layout()
    fig.text(0.015, -0.01, ttl, fontsize=9.5, va='top', ha='left', color='#444444')
    style.save(fig, OUT / 'fig6e_ddG_additive_vs_dddG_by_singles')
    K = pd.DataFrame(rows)
    K.to_csv(OUT / 'fig6de_ddG_additive_kinds.csv', index=False, float_format='%.4f')
    print(K.to_string())
    b = np.polyfit(xs, ys, 1)[0]
    print('OLS slope of dddG on ddG_additive %.3f (sd of ddG_additive %.2f); shared single-measurement noise would give about -%.3f if each measurement had sd 0.13' % (b, xs.std(), 2 * 0.13 ** 2 / xs.var()))
    print('dddG > 0.5: %.3f   |dddG| <= 0.5: %.3f   dddG < -0.5: %.3f   (n = %d)' % (np.mean(ys > 0.5), np.mean(np.abs(ys) <= 0.5), np.mean(ys < -0.5), len(ys)))
    print('double stabilizing (ddG_AB > 0): %.3f; compensation (dddG > -x - 0.5, x < 0): %.3f of the %d doubles with ddG_additive < 0' % (np.mean(dab > 0), np.mean(ys[xs < 0] > -xs[xs < 0] - 0.5), int((xs < 0).sum())))


SAT_DG = 0.0       # a double whose additive dG (measured singles) is at or below this is expected to sit on the assay floor whatever its interaction: saturation-dominated


def singles_plane(fr, m, y, dG_add):
    """fig6f drafts: the plane of the two measured singles, ddG_A (the more destabilizing of the pair) on x and ddG_B on y, coloured by the measured epistasis dddG (blue positive, white none, red negative).
    A  left the doubles that are clear of the assay floor (dG_additive > 0), right the same with the saturation-dominated ones (dG_additive <= 0) added back in black.
    B  the same as one hexagon per cell (at least 3 doubles): the mean dddG of the clear doubles on the left; on the right a cell is black when most of its doubles are saturation-dominated.
    C  every double, three panels by additive dG: > 1, 0 to 1 (close to the floor), <= 0 (saturation-dominated), all on one colour scale."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = m & np.isfinite(y) & np.isfinite(dG_add) & (ia >= 0) & (ib >= 0)
    a, b = ddg[ia[ok]], ddg[ib[ok]]
    xa, yb, z, g = np.minimum(a, b), np.maximum(a, b), y[ok], dG_add[ok]
    sat = g <= SAT_DG
    vmax = float(np.ceil(np.percentile(np.abs(z[~sat]), 98) * 4) / 4)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = 'RdBu'
    limx, limy = np.percentile(xa, [0.3, 99.7]), np.percentile(yb, [0.3, 99.7])
    padx, pady = 0.05 * (limx[1] - limx[0]), 0.05 * (limy[1] - limy[0])
    ext = (limx[0] - padx, limx[1] + padx, limy[0] - pady, limy[1] + pady)
    n_all, n_clear = len(z), int((~sat).sum())

    def frame(axes, titles):
        for ax, ttl in zip(np.atleast_1d(axes), titles):
            ax.plot([ext[0], ext[3]], [ext[0], ext[3]], color='#888888', ls=':', lw=1.0, zorder=0)
            ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
            ax.axvline(0, color='#cccccc', lw=0.8, zorder=0)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_xlabel('ΔΔG_A: the more destabilizing single (kcal/mol)')
            ax.set_title(ttl, fontsize=9, fontweight='bold', loc='left')
            for sp_ in ('top', 'right'):
                ax.spines[sp_].set_visible(False)
        np.atleast_1d(axes)[0].set_ylabel('ΔΔG_B: the other single (kcal/mol)')

    def coloured(ax, k):
        o = np.flatnonzero(k)
        o = o[np.argsort(np.abs(z[o]))]                                          # the strongest epistasis on top
        return ax.scatter(xa[o], yb[o], c=z[o], cmap=cmap, norm=norm, s=8, alpha=0.8, edgecolors='none', rasterized=True)

    def finish(fig, sc, name, title, right=0.93):
        fig.suptitle(title, fontsize=10, x=0.01, ha='left')
        fig.tight_layout(rect=(0, 0, right, 0.95))
        cax = fig.add_axes([right + 0.015, 0.12, 0.014, 0.74])
        fig.colorbar(sc, cax=cax).set_label('measured dddG (kcal/mol); blue positive, red negative')
        style.save(fig, OUT / name)

    # ---- A
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    sc = coloured(axes[0], ~sat)
    axes[1].scatter(xa[sat], yb[sat], c='black', s=7, alpha=0.4, edgecolors='none', rasterized=True)
    coloured(axes[1], ~sat)
    axes[1].legend(handles=[Line2D([0], [0], marker='o', ls='', color='black', alpha=0.6, label=f'dG additive ≤ {SAT_DG:g}: saturation-dominated (n = {int(sat.sum()):,})')], loc='lower right', fontsize=8, framealpha=0.92)
    frame(axes, (f'dG additive > {SAT_DG:g}: clear of the assay floor (n = {n_clear:,})', f'the same, with the saturation-dominated doubles added in black (n = {n_all:,})'))
    finish(fig, sc, 'fig6f_option_A_scatter_black_added', f'{TITLE}: measured epistasis over the plane of the two singles')

    # ---- B: hexagons
    gs = 24
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    hl = axes[0].hexbin(xa[~sat], yb[~sat], C=z[~sat], gridsize=gs, extent=ext, cmap=cmap, norm=norm, reduce_C_function=np.mean, mincnt=3, linewidths=0.2, edgecolors='#dddddd')
    tmp = plt.figure().gca()
    h_ns = tmp.hexbin(xa[~sat], yb[~sat], C=z[~sat], gridsize=gs, extent=ext, reduce_C_function=np.mean, mincnt=1)
    mean_ns = {tuple(np.round(k, 6)): v for k, v in zip(h_ns.get_offsets(), h_ns.get_array())}
    plt.close(tmp.figure)
    hb = axes[1].hexbin(xa, yb, C=sat.astype(float), gridsize=gs, extent=ext, reduce_C_function=np.mean, mincnt=3, linewidths=0.2, edgecolors='#dddddd')
    hb.set_facecolor([(0, 0, 0, 1) if sh > 0.5 else plt.get_cmap(cmap)(norm(mean_ns.get(tuple(np.round(k, 6)), 0.0))) for sh, k in zip(hb.get_array(), hb.get_offsets())])
    hb.set_array(None)
    axes[1].legend(handles=[Line2D([0], [0], marker='s', ls='', color='black', label=f'cell mostly dG additive ≤ {SAT_DG:g}: saturation-dominated')], loc='lower right', fontsize=8, framealpha=0.92)
    frame(axes, (f'dG additive > {SAT_DG:g}: mean dddG per cell (≥ 3 doubles)', 'all doubles: black = mostly saturation-dominated'))
    finish(fig, hl, 'fig6f_option_B_hexagons', f'{TITLE}: measured epistasis over the plane of the two singles')

    # ---- C: three panels by additive dG
    bands = [(g > 1, 'dG additive > 1'), ((g > SAT_DG) & (g <= 1), f'{SAT_DG:g} < dG additive ≤ 1: close to the floor'), (sat, f'dG additive ≤ {SAT_DG:g}: saturation-dominated')]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharex=True, sharey=True)
    for ax, (k, name) in zip(axes, bands):
        sc = coloured(ax, k)
    frame(axes, tuple(f'{name} (n = {int(k.sum()):,})' for k, name in bands))
    finish(fig, sc, 'fig6f_option_C_by_additive_dG', f'{TITLE} (n = {n_all:,}): measured epistasis over the plane of the two singles, by additive dG', right=0.94)
    print(f'plane of the singles: {n_all:,} doubles, {n_clear:,} clear of the floor, {int(sat.sum()):,} saturation-dominated; colour range ±{vmax:g}')

if __name__ == '__main__':
    main()
