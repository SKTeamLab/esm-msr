#!/usr/bin/env python
"""Predicted against measured epistasis (dddG) and true additive expectation (dG_additive) against predicted dddG of every model on the Homology-Restricted MegaScale test doubles
with both singles measured (12,455), density scatters without marginals:

  results/06_epistasis/dddG_scatters/all_dG_additive/<model>.png        every double (predicted dddG vs measured dddG, with line x = y)
  results/06_epistasis/dddG_scatters/dG_additive_positive/<model>.png   only doubles whose additive dG = dG_wt + ddG_A + ddG_B (the measured singles) is above 0
  results/06_epistasis/dddG_scatters/dG_additive_gt1/<model>.png        only those above 1 kcal/mol, where the link is the identity for practical purposes
  results/06_epistasis/dG_additive_vs_pred_dddG/<model>.png             true additive expectation dG_additive on x vs model's predicted dddG on y (with line y = -x)
"""
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

h12 = importlib.import_module('12_pair_epistasis_heatmaps')      # build(), MODELS, UNIT
EPI_OUT = out_dir('06_epistasis')
OUT = EPI_OUT / 'dddG_scatters'
OUT.mkdir(parents=True, exist_ok=True)
OUT_DG_ADD = EPI_OUT / 'dG_additive_vs_pred_dddG'
OUT_DG_ADD.mkdir(parents=True, exist_ok=True)
Y_LABEL = 'measured δΔΔG = dddG (kcal/mol)'
X_DG_ADD_LABEL = 'ΔG additive = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)'
TITLE = 'Homology-Restricted MegaScale test doubles with both singles measured'
VERSIONS = {'all_dG_additive': ('all doubles', None), 'dG_additive_positive': ('doubles with ΔG additive > 0', 0.0), 'dG_additive_gt1': ('doubles with ΔG additive > 1', 1.0)}
LINK_MODEL = 'ESM-MSR'
SCATTER_FONT = dict(
    label_fontsize=13.5,
    tick_fontsize=12.0,
    stats_fontsize=11.5,
    legend_fontsize=10.5,
    foot_fontsize=9.5,
    cbar_fontsize=12.5,
    cbar_tick_fontsize=10.5,
)


def slug(label):
    return re.sub(r'[^A-Za-z0-9-]+', '_', label.replace('*', '')).strip('_')


def rosetta_unit_check(fr, fit):
    """RMSE of Rosetta against the measurement in REU and in kcal/mol (REU / 2.94), for dddG and for ddG (rosetta in Frame is already scaled by 1/2.94)."""
    sp, p = fr.spine, fr.preds
    nm, use = sp['n_mut'].to_numpy(), sp['use'].to_numpy()
    y, d3 = sp['y'].to_numpy(float), sp['dddG_ML'].to_numpy(float)
    ros_kcal, add_kcal = p['rosetta'][0], p['rosetta+'][0]
    ros_reu, add_reu = ros_kcal * 2.94, add_kcal * 2.94
    rows = []
    for what, a_reu, a_kcal, b, mask in (
        ('dddG, doubles', ros_reu - add_reu, ros_kcal - add_kcal, d3, fit),
        ('ddG, singles', ros_reu, ros_kcal, y, use & (nm == 1)),
        ('ddG, doubles', ros_reu, ros_kcal, y, use & (nm == 2)),
    ):
        ok = mask & np.isfinite(a_reu) & np.isfinite(b)
        k = float((a_reu[ok] * b[ok]).sum() / (a_reu[ok] ** 2).sum())
        rows.append(dict(quantity=what, n=int(ok.sum()), rmse_reu=np.sqrt(np.mean((a_reu[ok] - b[ok]) ** 2)), rmse_div_2_94=np.sqrt(np.mean((a_kcal[ok] - b[ok]) ** 2)),
                         best_scale_through_origin=k, rmse_best_scale=np.sqrt(np.mean((k * a_reu[ok] - b[ok]) ** 2))))
    return pd.DataFrame(rows)


def diagnostics(fr, fit, model, dG_add, y):
    """Why does ESM-MSR's latent epistasis over-predict at dG_additive > 1?  The linked one is the same; a 2 x 2 of the measured against the model's own additive dG, the bias by domain, and
    the same on validation for the devel and the baseline checkpoints."""
    sp, p = fr.spine, fr.preds
    v = model[LINK_MODEL]
    c = sp['dG_wt'].to_numpy(float)
    x_model = c + np.nanmean(p['msrdev:wt+'], axis=0)                                 # the model's additive dG of the double: c + WT(A) + WT(B), latent
    df = pd.DataFrame(dict(domain=sp['group'].to_numpy(), dG_wt=c, measured_dG_additive=dG_add, model_dG_additive=x_model, measured=y, latent=v['latent'], linked=v['linked']))[fit].dropna()
    df['measured_gt1'], df['model_gt1'] = df.measured_dG_additive > 1, df.model_dG_additive > 1
    t = df.groupby(['measured_gt1', 'model_gt1']).agg(n=('measured', 'size'), measured_dddG=('measured', 'mean'), latent=('latent', 'mean'), linked=('linked', 'mean'),
                                                       mean_measured_dG_additive=('measured_dG_additive', 'mean'), mean_model_dG_additive=('model_dG_additive', 'mean')).reset_index()
    t['bias_latent'], t['bias_linked'] = t.latent - t.measured_dddG, t.linked - t.measured_dddG
    t.to_csv(OUT / 'bias_cross_tab_additive_dG.csv', index=False, float_format='%.4f')
    both = df[df.measured_gt1 & df.model_gt1].assign(bias=lambda d: d.latent - d.measured)
    g = both.groupby('domain').agg(n=('bias', 'size'), dG_wt=('dG_wt', 'mean'), measured=('measured', 'mean'), latent=('latent', 'mean'), bias=('bias', 'mean')).reset_index()
    g.to_csv(OUT / 'bias_by_domain_both_gt1.csv', index=False, float_format='%.4f')
    rows = []
    for ds, tag in (('tsuboyama_capped_test', 'test'), ('tsuboyama_capped_val', 'validation')):
        f2 = load_frame(ds)
        m2 = epi_rows(f2)
        s2, q = f2.spine, f2.preds
        c2 = s2['dG_wt'].to_numpy(float)
        xm = c2 + s2['ddG_additive_ML'].to_numpy(float)
        for key in ('msrdev', 'msr'):
            W = np.nanmean(q[f'{key}:wt+'], 0)
            lat, d3, xp = np.nanmean(q[f'{key}:epi'], 0), s2['dddG_ML'].to_numpy(float), c2 + W
            for sel_name, sel in (('measured dG_additive > 1', xm > 1), ('measured and model dG_additive > 1', (xm > 1) & (xp > 1))):
                k = m2 & sel & np.isfinite(lat) & np.isfinite(d3)
                rows.append(dict(split=tag, checkpoint='ESM-MSR' if key == 'msrdev' else 'ESM-MSR (baseline)', subset=sel_name, n=int(k.sum()), measured_dddG=d3[k].mean(), latent=lat[k].mean(), bias=(lat[k] - d3[k]).mean()))
    pd.DataFrame(rows).to_csv(OUT / 'bias_test_vs_validation.csv', index=False, float_format='%.4f')
    sel = df.measured_gt1
    fig = style.density_scatter_plain(df.linked[sel], df.latent[sel], 'ESM-MSR linked epistasis (kcal/mol)', 'ESM-MSR latent epistasis (kcal/mol)', ref='xy', rmse=True,
                                      title=f'ESM-MSR, doubles with ΔG additive > 1 (n = {int(sel.sum()):,}): latent against linked')
    fig.savefig(OUT / 'latent_vs_linked_gt1.png', dpi=150)
    plt.close(fig)
    print(t.round(3).to_string()); print(pd.DataFrame(rows).round(3).to_string())


def draw_dG_additive_vs_pred_dddG(fit, dG_add, model):
    """Subfolder results/06_epistasis/dG_additive_vs_pred_dddG/: true additive expectation dG_additive on x vs each model's predicted dddG on y."""
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde
    rows = []
    for label, v in model.items():
        star = '*' if label.endswith('*') else ''
        y_seeds = v['raw_seeds']
        ok = fit & np.isfinite(dG_add) & np.isfinite(y_seeds).all(axis=0)
        xs = dG_add[ok]
        ys = y_seeds[:, ok] if y_seeds.shape[0] > 1 else y_seeds[0, ok]
        n_seeds = y_seeds.shape[0]
        note = ' (latent)' if label.startswith('ESM-MSR') else ''
        yl = f'{label.rstrip("*")}{star} predicted δΔΔG = dddG (kcal/mol)'
        fig = style.density_scatter_plain(
            xs, ys, X_DG_ADD_LABEL, yl,
            ref='neg', ref_label='y = −x: dG_AB = 0,\nwhere the unfolded doubles pile up',
            title=None, foot=f'{TITLE}, {label}{note} (n = {int(ok.sum()):,})',
            legend_loc='lower left', **SCATTER_FONT,
        )
        ax = fig.axes[0]
        ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
        handles = ax.get_legend().legend_handles + [plt.Line2D([0], [0], color='#555555', lw=1.2, ls=':', label='y = 0: no epistasis')]
        ax.legend(handles=handles, loc='lower left', fontsize=10.2, frameon=False, labelspacing=0.45)
        style.save(fig, OUT_DG_ADD / slug(label))
        rho_all = float(np.mean([spearmanr(xs, y_seeds[s, ok])[0] for s in range(n_seeds)]))
        r_all = float(np.mean([pearsonr(xs, y_seeds[s, ok])[0] for s in range(n_seeds)]))
        ok_gt0 = ok & (dG_add > 0)
        rho_gt0 = float(np.mean([spearmanr(dG_add[ok_gt0], y_seeds[s, ok_gt0])[0] for s in range(n_seeds)]))
        r_gt0 = float(np.mean([pearsonr(dG_add[ok_gt0], y_seeds[s, ok_gt0])[0] for s in range(n_seeds)]))
        rows.append(dict(
            model=label, n=int(ok.sum()), n_seeds=n_seeds,
            spearman=rho_all, pearson=r_all,
            spearman_dG_add_gt0=rho_gt0, pearson_dG_add_gt0=r_gt0,
        ))
    T_dg = pd.DataFrame(rows)
    T_dg.to_csv(OUT_DG_ADD / 'stats.csv', index=False, float_format='%.4f')

    # Side-by-side 1x3 comparison for ESM-MSR (baseline), ESM-MSR, and ESM-MSR (p0)
    for msr_tuple, suffix, formula in (
        (('ESM-MSR (baseline)', 'ESM-MSR', 'ESM-MSR (p0)'), '', '1/2(MT_AB+WT_AB) − WT_A − WT_B'),
    ):
        msr_names = [m for m in msr_tuple if m in model]
        if len(msr_names) == 3:
            ok = fit & np.isfinite(dG_add)
            for m_name in msr_names:
                ok &= np.isfinite(model[m_name]['raw_seeds']).all(axis=0)
            xv_1d = dG_add[ok]
            n_var = len(xv_1d)
            limx = np.percentile(xv_1d, [0.3, 99.7])
            padx = 0.05 * (limx[1] - limx[0])
            xlim = (limx[0] - padx, limx[1] + padx)
            all_y = np.concatenate([model[m_name]['raw_seeds'][:, ok].ravel() for m_name in msr_names])
            limy = np.percentile(all_y, [0.3, 99.7])
            pady = 0.05 * (limy[1] - limy[0])
            ylim = (limy[0] - pady, limy[1] + pady)
            cm = plt.get_cmap('plasma')
            fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), sharex=True, sharey=True)
            for idx, (ax, m_name) in enumerate(zip(axes, msr_names)):
                y_ok = model[m_name]['raw_seeds'][:, ok]
                n_seeds = y_ok.shape[0]
                xv = np.tile(xv_1d, n_seeds)
                yv = y_ok.ravel()
                sub = np.random.default_rng(0).choice(len(xv), min(len(xv), 15000), replace=False)
                z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
                o = np.argsort(z)
                norm = Normalize(z.min(), z.max())
                col = cm(norm(z))
                col[:, 3] = 0.3 / np.sqrt(n_seeds)
                ax.scatter(xv[o], yv[o], c=col[o], s=8, edgecolors='none', rasterized=True)
                ax.plot(xlim, [-xlim[0], -xlim[1]], color='#c0392b', ls='--', lw=1.6, zorder=4)
                ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel(X_DG_ADD_LABEL, fontsize=SCATTER_FONT['label_fontsize'])
                if idx == 0:
                    ax.set_ylabel('predicted δΔΔG = dddG (kcal/mol)', fontsize=SCATTER_FONT['label_fontsize'])
                ax.tick_params(axis='both', labelsize=SCATTER_FONT['tick_fontsize'])
                rho_v = float(np.mean([spearmanr(xv_1d, y_ok[s])[0] for s in range(n_seeds)]))
                r_v = float(np.mean([pearsonr(xv_1d, y_ok[s])[0] for s in range(n_seeds)]))
                txt = f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {n_var:,} (×{n_seeds} seeds)'
                ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=SCATTER_FONT['stats_fontsize'], zorder=10,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94))
                handles = [
                    plt.Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label='y = −x: dG_AB = 0'),
                    plt.Line2D([0], [0], color='#555555', ls=':', lw=1.2, label='y = 0: no epistasis'),
                ]
                ax.legend(handles=handles, loc='lower left', fontsize=SCATTER_FONT['legend_fontsize'], frameon=False)
                ax.set_title(m_name, fontsize=13.5, fontweight='bold', pad=8)
                for sp_ in ('top', 'right'):
                    ax.spines[sp_].set_visible(False)
            fig.tight_layout()
            fig.text(0.012, -0.01, f'{TITLE} (n = {n_var:,}, 3 triplicate seeds integrated without ensemble averaging); true ΔG additive vs predicted epistasis = {formula}',
                     fontsize=SCATTER_FONT['foot_fontsize'], va='top', ha='left', color='#444444')
            style.save(fig, OUT_DG_ADD / f'esm_msr_models_dG_additive_vs_pred_dddG{suffix}')
    print('\ndG_additive vs predicted dddG:\n', T_dg.round(3).to_string())


def main():
    fr, d, fit, meas, model, singles = h12.build()
    sp = fr.spine
    y = meas['raw']
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    rosetta_unit_check(fr, fit).to_csv(OUT / 'rosetta_unit_check.csv', index=False, float_format='%.4f')
    rows = []
    for label, v in model.items():
        star = '*' if label.endswith('*') else ''
        x_seeds = v['raw_seeds']
        n_seeds = x_seeds.shape[0]
        for ver, (desc, cut) in VERSIONS.items():
            ok = fit & np.isfinite(x_seeds).all(axis=0) & np.isfinite(y) & (True if cut is None else (dG_add > cut))
            xs_in = x_seeds[:, ok] if n_seeds > 1 else x_seeds[0, ok]
            note = ' (latent)' if label.startswith('ESM-MSR') else ''
            fig = style.density_scatter_plain(xs_in, y[ok], f'{label.rstrip("*")}{star} predicted dddG (kcal/mol)', Y_LABEL, ref='xy', rmse=True,
                                              title=None, foot=f'{label}{note}: {desc} (n = {int(ok.sum()):,})', **SCATTER_FONT)
            folder = OUT / ver
            folder.mkdir(parents=True, exist_ok=True)
            style.save(fig, folder / slug(label))
            rho_v = float(np.mean([spearmanr(x_seeds[s, ok], y[ok])[0] for s in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(x_seeds[s, ok], y[ok])[0] for s in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((x_seeds[s, ok] - y[ok]) ** 2)) for s in range(n_seeds)]))
            rows.append(dict(model=label, version=ver, prediction='latent' if label.startswith('ESM-MSR') else 'as is', n=int(ok.sum()), n_seeds=n_seeds,
                             spearman=rho_v, pearson=r_v, rmse=rmse_v))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'stats.csv', index=False, float_format='%.4f')
    # ESM-MSR: observed against latent prediction, in every subset
    v = model[LINK_MODEL]
    cmp_rows = []
    for ver, (desc, cut) in VERSIONS.items():
        ok = fit & np.isfinite(v['linked']) & np.isfinite(v['latent']) & np.isfinite(y) & (True if cut is None else (dG_add > cut))
        o, l = v['linked'][ok], v['latent'][ok]
        cmp_rows.append(dict(version=ver, n=int(ok.sum()), pearson_observed_vs_latent=pearsonr(o, l)[0], spearman_observed_vs_latent=spearmanr(o, l)[0], mean_abs_diff=float(np.mean(np.abs(o - l))),
                             max_abs_diff=float(np.max(np.abs(o - l))), rho_observed=spearmanr(o, y[ok])[0], rho_latent=spearmanr(l, y[ok])[0],
                             rmse_observed=float(np.sqrt(np.mean((o - y[ok]) ** 2))), rmse_latent=float(np.sqrt(np.mean((l - y[ok]) ** 2)))))
    C = pd.DataFrame(cmp_rows)
    C.to_csv(OUT / 'latent_vs_observed.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    diagnostics(fr, fit, model, dG_add, y)
    draw_dG_additive_vs_pred_dddG(fit, dG_add, model)
    print(T.pivot(index='model', columns='version', values=['spearman', 'rmse']).round(3).reindex([m[0] for m in h12.MODELS if m[0] in model]).to_string())
    print(C.round(3).to_string())
    print(rosetta_unit_check(fr, fit).round(3).to_string())


if __name__ == '__main__':
    main()
