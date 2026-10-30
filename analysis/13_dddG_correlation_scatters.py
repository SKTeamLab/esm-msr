#!/usr/bin/env python
"""Predicted against measured epistasis (dddG) of every model on the capped Tsuboyama test doubles with both singles measured (12,455), density scatters without marginals and with the
line x = y, in three versions:

  results/13_dddG_scatters/all_dG_additive/<model>.png        every double
  results/13_dddG_scatters/dG_additive_positive/<model>.png   only doubles whose additive dG = dG_wt + ddG_A + ddG_B (the measured singles) is above 0, i.e. not already lost to the
                                                              assay floor if the singles add: where the measured dddG is not the floor's
  results/13_dddG_scatters/dG_additive_gt1/<model>.png        only those above 1 kcal/mol, where the link is the identity for practical purposes: ESM-MSR devel is drawn from its
                                                              latent prediction (the other two versions use the observed one)

The predicted epistasis of a model is the one of the pair heat maps (``12_pair_epistasis_heatmaps.py``): ESM-MSR MT - WT, ME stage 2 / ThermoMPNN-D / SPURS multi minus their singles
(seed means), ESM3-small-open independent mutant pass minus masked-marginal WT pass, the public models minus their sum of singles; Rosetta is divided by 2.94 (REU to kcal/mol, see
``rosetta_unit_check.csv``).  The measurement is on the observed scale (dddG = ddG_AB - ddG_A - ddG_B), so ESM-MSR devel, whose checkpoints write latent ddG, is put on it the way the
epistasis suite's naive epistasis does: obs(MT(AB)) - [obs(WT(A)) + obs(WT(B))] through each seed's link (the baseline triplicate has no link: MT - WT).  ``stats.csv`` has n, Spearman
rho, Pearson r and the RMSE of every panel; ``latent_vs_observed.csv`` compares the two predictions of ESM-MSR devel in each subset.
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
OUT = out_dir('13_dddG_scatters')
Y_LABEL = 'measured δΔΔG = dddG (kcal/mol)'
VERSIONS = {'all_dG_additive': ('all doubles', None), 'dG_additive_positive': ('doubles with ΔG additive > 0', 0.0), 'dG_additive_gt1': ('doubles with ΔG additive > 1', 1.0)}
LINK_MODEL = 'ESM-MSR (devel)'


def slug(label):
    return re.sub(r'[^A-Za-z0-9-]+', '_', label.replace('*', '')).strip('_')


def rosetta_unit_check(fr, fit):
    """RMSE of Rosetta against the measurement in REU and in kcal/mol (REU / 2.94), for dddG and for ddG."""
    sp, p = fr.spine, fr.preds
    nm, use = sp['n_mut'].to_numpy(), sp['use'].to_numpy()
    y, d3 = sp['y'].to_numpy(float), sp['dddG_ML'].to_numpy(float)
    ros, add = p['rosetta'][0], p['rosetta+'][0]
    rows = []
    for what, a, b, mask in (('dddG, doubles', ros - add, d3, fit), ('ddG, singles', ros, y, use & (nm == 1)), ('ddG, doubles', ros, y, use & (nm == 2))):
        ok = mask & np.isfinite(a) & np.isfinite(b)
        k = float((a[ok] * b[ok]).sum() / (a[ok] ** 2).sum())
        rows.append(dict(quantity=what, n=int(ok.sum()), rmse_reu=np.sqrt(np.mean((a[ok] - b[ok]) ** 2)), rmse_div_2_94=np.sqrt(np.mean((a[ok] / 2.94 - b[ok]) ** 2)),
                         best_scale_through_origin=k, rmse_best_scale=np.sqrt(np.mean((k * a[ok] - b[ok]) ** 2))))
    return pd.DataFrame(rows)


def diagnostics(fr, fit, model, dG_add, y):
    """Why does ESM-MSR's latent epistasis over-predict at dG_additive > 1?  The linked one is the same; a 2 x 2 of the measured against the model's own additive dG, the bias by domain, and
    the same on validation for the devel and the baseline checkpoints."""
    sp, p = fr.spine, fr.preds
    v = model[LINK_MODEL]
    c = sp['dG_wt'].to_numpy(float)
    x_model = c + np.nanmean(p['msrdev:wt'], axis=0)                                  # the model's additive dG of the double: c + WT(A) + WT(B), latent
    df = pd.DataFrame(dict(domain=sp['group'].to_numpy(), dG_wt=c, measured_dG_additive=dG_add, model_dG_additive=x_model, measured=y, latent=v['latent'], linked=v['raw']))[fit].dropna()
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
            W, M = np.nanmean(q[f'{key}:wt'], 0), np.nanmean(q[f'{key}:mt'], 0)
            lat, d3, xp = M - W, s2['dddG_ML'].to_numpy(float), c2 + W
            for sel_name, sel in (('measured dG_additive > 1', xm > 1), ('measured and model dG_additive > 1', (xm > 1) & (xp > 1))):
                k = m2 & sel & np.isfinite(lat) & np.isfinite(d3)
                rows.append(dict(split=tag, checkpoint='ESM-MSR devel' if key == 'msrdev' else 'ESM-MSR baseline', subset=sel_name, n=int(k.sum()), measured_dddG=d3[k].mean(), latent=lat[k].mean(), bias=(lat[k] - d3[k]).mean()))
    pd.DataFrame(rows).to_csv(OUT / 'bias_test_vs_validation.csv', index=False, float_format='%.4f')
    sel = df.measured_gt1
    fig = style.density_scatter_plain(df.linked[sel], df.latent[sel], 'ESM-MSR devel linked epistasis (kcal/mol)', 'ESM-MSR devel latent epistasis (kcal/mol)', ref='xy', rmse=True,
                                      title=f'ESM-MSR devel, doubles with ΔG additive > 1 (n = {int(sel.sum()):,}): latent against linked')
    fig.savefig(OUT / 'latent_vs_linked_gt1.png', dpi=150)
    plt.close(fig)
    print(t.round(3).to_string()); print(pd.DataFrame(rows).round(3).to_string())


def main():
    fr, d, fit, meas, model, singles = h12.build()
    sp = fr.spine
    y = meas['raw']
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    rosetta_unit_check(fr, fit).to_csv(OUT / 'rosetta_unit_check.csv', index=False, float_format='%.4f')
    rows = []
    for label, v in model.items():
        star = '*' if label.endswith('*') else ''
        for ver, (desc, cut) in VERSIONS.items():
            latent_here = label == LINK_MODEL and ver == 'dG_additive_gt1'
            x = v['latent'] if latent_here else v['raw']
            ok = fit & np.isfinite(x) & np.isfinite(y) & (True if cut is None else (dG_add > cut))
            note = ' (latent)' if latent_here else (' (observed)' if label == LINK_MODEL else '')
            fig = style.density_scatter_plain(x[ok], y[ok], f'{label.rstrip("*")}{star} predicted dddG (kcal/mol)', Y_LABEL, ref='xy', rmse=True,
                                              title=f'{label}{note}: {desc} (n = {int(ok.sum()):,})')
            folder = OUT / ver
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f'{slug(label)}.png', dpi=150)
            plt.close(fig)
            rows.append(dict(model=label, version=ver, prediction='latent' if latent_here else ('observed' if label == LINK_MODEL else 'as is'), n=int(ok.sum()),
                             spearman=spearmanr(x[ok], y[ok])[0], pearson=pearsonr(x[ok], y[ok])[0], rmse=float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'stats.csv', index=False, float_format='%.4f')
    # ESM-MSR devel: observed against latent prediction, in every subset
    v = model[LINK_MODEL]
    cmp_rows = []
    for ver, (desc, cut) in VERSIONS.items():
        ok = fit & np.isfinite(v['raw']) & np.isfinite(v['latent']) & np.isfinite(y) & (True if cut is None else (dG_add > cut))
        o, l = v['raw'][ok], v['latent'][ok]
        cmp_rows.append(dict(version=ver, n=int(ok.sum()), pearson_observed_vs_latent=pearsonr(o, l)[0], spearman_observed_vs_latent=spearmanr(o, l)[0], mean_abs_diff=float(np.mean(np.abs(o - l))),
                             max_abs_diff=float(np.max(np.abs(o - l))), rho_observed=spearmanr(o, y[ok])[0], rho_latent=spearmanr(l, y[ok])[0],
                             rmse_observed=float(np.sqrt(np.mean((o - y[ok]) ** 2))), rmse_latent=float(np.sqrt(np.mean((l - y[ok]) ** 2)))))
    C = pd.DataFrame(cmp_rows)
    C.to_csv(OUT / 'latent_vs_observed.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    diagnostics(fr, fit, model, dG_add, y)
    print(T.pivot(index='model', columns='version', values=['spearman', 'rmse']).round(3).reindex([m[0] for m in h12.MODELS if m[0] in model]).to_string())
    print(C.round(3).to_string())
    print(rosetta_unit_check(fr, fit).round(3).to_string())


if __name__ == '__main__':
    main()
