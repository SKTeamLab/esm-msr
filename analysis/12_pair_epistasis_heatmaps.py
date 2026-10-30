#!/usr/bin/env python
"""Pair-epistasis heat maps (the notebooks' ``plot_joint_with_marginals``) for every test position pair whose identity pairs are all in the data, raw and saturation-corrected.

A position pair (i, j) of a capped Tsuboyama test domain is *fully characterized* when all 19 x 19 = 361 substitution pairs (a at i, b at j) are in the dataset (15 of the 45
test pairs; the others lack up to 64 cells).  Each figure is one such pair: a heat map of the epistasis dddG = ddG_AB - ddG_A - ddG_B over the identity pairs, one panel for the
measurement and one per model, in the notebooks' layout (residue order K R H D E S T N Q Y C A V I L M F W P G, bars of the single-substitution scores of each position on the
margins: measured ddG for the measurement, the model's single-mutant prediction for a model).  A grey × = no usable value (the double sits at an assay bound, or a single is
missing: the benchmark's rule; the wild-type residue of each position is a row and a column of ×); a model cell is drawn wherever the model predicts and scored only where the measurement exists.

  raw                    dddG as is (measured) and the model's predicted epistasis (its double minus its additive sum / WT-head double; BENCHMARKS table 2.2's definition)
  saturation_corrected   the MEASUREMENT minus E[dddG | its additive expectation x = dG_wt + ddG_A + ddG_B] (the 25-quantile-bin smoother of ``esm_msr.epi_metrics.smooth_on`` fitted
                         on all 12,455 test doubles: the assay floor and any diminishing returns, a function of x alone); the models' predictions are left exactly as in ``raw``,
                         so a model that carries the saturation in its predicted epistasis is compared with the measurement once the saturation is gone from it.

Colour: blue = positive epistasis, red = negative, × = no usable value.  One scale for every panel of every figure of a mode (symmetric about zero, the 98th percentile of
|measured value| over the usable cells of the 15 pairs: raw and saturation-corrected each have their own, printed on the colour bar); a model's values beyond it are clipped to the end
colour.  Each panel title carries the Spearman rho of the model against the measured cells of the pair.
Layout: ``results/12_pair_heatmaps/<raw|saturation_corrected>/<domain>/<wt><pos>_<wt><pos>.png``, ``pairs.csv`` (the pairs) and ``agreement.csv`` (rho of each model, per pair and mode).
ESM-MSR's epistasis is the MT head against the WT head: for the baseline triplicate, which has no link, MT(AB) - WT(AB); for the devel checkpoints (which write latent ddG) the raw panels are on the observed scale like the
measurement, obs(MT(AB)) - [obs(WT(A)) + obs(WT(B))] through each seed's link and library floors (``msr_analysis.link.observed_epistasis``, the epistasis suite's naive definition); the predictions are never saturation-corrected.
Rosetta is divided by 2.94 (REU to kcal/mol).  Multi-seed models are averaged over seeds first; ESM3ΔG is not shown (its fine-tune overlaps the test proteins), the Megascale-trained ESMtherm and IFUM carry an asterisk.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.registry import REFERENCE  # noqa: E402
from msr_analysis.data import add_esm3_masked, load_frame  # noqa: E402
from msr_analysis.link import observed_epistasis  # noqa: E402

DS = 'tsuboyama_capped_test'
OUT = out_dir('12_pair_heatmaps')
AA = list('KRHDESTNQYCAVILMFWPG')                    # the notebooks' order
IDX = {a: i for i, a in enumerate(AA)}
SMOOTH_BINS = 25                                      # esm_msr.epi_metrics.SMOOTH_BINS
FLAT = 0.02                                           # esm_msr.epi_metrics.FLAT: a flatter prediction scores 0 beyond the global curve
DPI = 85
# Rosetta Cartesian ddG is in Rosetta energy units; 1 kcal/mol = 2.94 REU.  Dividing by it takes the RMSE of the predicted dddG against the measured one on the 12,455 test doubles
# from 3.08 to 1.15 and that of ddG from 6.9 / 9.8 (singles / doubles) to 1.9 / 2.5; every other model is already in kcal/mol.
UNIT = {'Rosetta': 1 / 2.94}

# (label, epistasis builder, additive key, singles key); builders are ('diff', a, b) = a - b or ('key', k).  Order = the panels after the measurement.
MODELS = [('ESM-MSR', ('diff', 'msr:mt', 'msr:wt'), 'msr:wt', 'msr:wt'),
          ('ESM-MSR (devel)', ('diff', 'msrdev:mt', 'msrdev:wt'), 'msrdev:wt', 'msrdev:wt'),
          ('MutateEverything st2', ('key', 'me2:epi'), 'me1', 'me1'),
          ('ThermoMPNN-D', ('key', 'tmpnnd:epi'), 'tmpnn', 'tmpnn'),
          ('SPURS multi', ('key', 'spursm:epi'), 'spurs', 'spurs'),
          ('ESM3-small-open', ('diff', 'esm3ind', 'esm3mm'), 'esm3mm', 'esm3mm'),
          ('ProteinMPNN', ('diff', 'pmpnn', 'pmpnn+'), 'pmpnn+', 'pmpnn'),
          ('GeoDDG-3D', ('diff', 'geoddg', 'geoddg+'), 'geoddg+', 'geoddg'),
          ('DDGemb', ('diff', 'ddgemb', 'ddgemb+'), 'ddgemb+', 'ddgemb'),
          ('Rosetta', ('diff', 'rosetta', 'rosetta+'), 'rosetta+', 'rosetta'),
          ('ESMtherm*', ('diff', 'esmtherm', 'esmtherm+'), 'esmtherm+', 'esmtherm'),
          ('IFUM, WT backbone*', ('diff', 'ifum_wt', 'ifum_wt+'), 'ifum_wt+', 'ifum_wt'),
          ('IFUM, ESMFold*', ('diff', 'ifum_esmfold', 'ifum_esmfold+'), 'ifum_esmfold+', 'ifum_esmfold')]


def smooth_on(x, v, nbins=SMOOTH_BINS):
    """E[v | x] of ``esm_msr.epi_metrics.smooth_on``: piecewise-linear through the means of quantile bins of x, extended linearly beyond the outer bin centres."""
    x, v = np.asarray(x, float), np.asarray(v, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x) & np.isfinite(v)
    if ok.sum() < 2 * nbins:
        nbins = max(1, int(ok.sum()) // 2)
    if ok.sum() < 2:
        return out
    qs = np.quantile(x[ok], np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(qs, x[ok], side='right') - 1, 0, nbins - 1)
    cx = np.array([x[ok][idx == k].mean() for k in range(nbins) if (idx == k).any()])
    cv = np.array([v[ok][idx == k].mean() for k in range(nbins) if (idx == k).any()])
    o = np.argsort(cx)
    cx, cv = cx[o], cv[o]
    fin = np.isfinite(x)
    out[fin] = np.interp(x[fin], cx, cv)
    if len(cx) >= 2:
        lo, hi = fin & (x < cx[0]), fin & (x > cx[-1])
        out[lo] = cv[0] + (x[lo] - cx[0]) * (cv[1] - cv[0]) / max(cx[1] - cx[0], 1e-12)
        out[hi] = cv[-1] + (x[hi] - cx[-1]) * (cv[-1] - cv[-2]) / max(cx[-1] - cx[-2], 1e-12)
    return out


def beyond_global(v, x, fit):
    """v minus E[v | x], the smoother fitted on the rows ``fit`` and evaluated everywhere x is finite; a prediction flatter than FLAT scores 0 where it is defined."""
    if not np.isfinite(v[fit]).any() or np.nanstd(v[fit]) < FLAT:
        return np.where(np.isfinite(v), 0.0, np.nan)
    return v - smooth_on(x, np.where(fit, v, np.nan))


def mean_seeds(p):
    return np.nanmean(p, axis=0)


def parse_double(s):
    out = []
    for t in s.split(':'):
        m = re.match(r'^([A-Z])(\d+)([A-Z])$', t)
        out.append((m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None))
    return out


def build():
    fr = load_frame(DS)
    add_esm3_masked(fr, DS)
    sp, p = fr.spine, fr.preds
    n = len(sp)
    y_floor = json.load(open(REFERENCE / 'y_floor.json'))
    fit = epi_rows(fr)
    c = sp['dG_wt'].to_numpy(float)
    meas = {'raw': sp['dddG_ML'].to_numpy(float)}
    x_meas = c + sp['ddG_additive_ML'].to_numpy(float)
    meas['saturation_corrected'] = beyond_global(meas['raw'], x_meas, fit)
    model = {}
    for label, build_, addk, sk in MODELS:
        need = [build_[1]] + ([build_[2]] if build_[0] == 'diff' else []) + [addk, sk]
        if any(k not in p for k in need):
            print('skip (missing predictions):', label, [k for k in need if k not in p])
            continue
        u = UNIT.get(label, 1.0)
        e = u * (mean_seeds(p[build_[1]]) - (mean_seeds(p[build_[2]]) if build_[0] == 'diff' else 0.0))
        model[label] = dict(raw=e, saturation_corrected=e, single=u * mean_seeds(p[sk]), latent=e)       # predictions are not saturation-corrected
        if label == 'ESM-MSR (devel)':       # the raw panels are on the observed scale, like the measurement: obs(MT(AB)) - [obs(WT(A)) + obs(WT(B))] through each seed's link (the suite's naive definition)
            model[label]['raw'] = model[label]['saturation_corrected'] = np.nanmean([observed_epistasis(sp, p['msrdev:wt'][s], p['msrdev:mt'][s], s + 1, y_floor) for s in range(3)], axis=0)
    mut = sp['mut'].to_numpy(dtype=object)
    nm = sp['n_mut'].to_numpy()
    d = pd.DataFrame({'i': np.flatnonzero(nm == 2)})
    pp = [parse_double(mut[i]) for i in d['i']]
    d['code_wt'] = sp['code_wt'].to_numpy()[d['i']]
    d['wt1'], d['pos1'], d['mt1'] = [t[0][0] for t in pp], [t[0][1] for t in pp], [t[0][2] for t in pp]
    d['wt2'], d['pos2'], d['mt2'] = [t[1][0] for t in pp], [t[1][1] for t in pp], [t[1][2] for t in pp]
    singles = {(cw, m): i for i, (cw, m) in enumerate(zip(sp['code_wt'].to_numpy(), mut)) if nm[i] == 1}
    return fr, d, fit, meas, model, singles


def matrix(d, values):
    """20 x 20 (rows: residue at position 1, columns: residue at position 2) of ``values`` (indexed by spine row); NaN where the pair is not in the data."""
    M = np.full((20, 20), np.nan)
    for r in d.itertuples():
        M[IDX[r.mt1], IDX[r.mt2]] = values[r.i]
    return M


def margin(code, wt, pos, singles, vals):
    out = np.full(20, np.nan)
    for a in AA:
        if a != wt and (code, f'{wt}{pos}{a}') in singles:
            out[IDX[a]] = vals[singles[(code, f'{wt}{pos}{a}')]]
    return out


def heat_panel(ax, M, top, right, title, vmax, wt1, wt2, pos1, pos2, label_axes):
    cm = plt.get_cmap('RdBu').copy()                   # blue = positive epistasis
    cm.set_bad('white')
    im = ax.imshow(np.ma.masked_invalid(M), origin='lower', interpolation='nearest', cmap=cm, vmin=-vmax, vmax=vmax, aspect='equal')
    yy, xx = np.where(~np.isfinite(M))                  # no value in this cell: a grey x on white
    ax.scatter(xx, yy, marker='x', s=6, color='#8c8c8c', linewidths=0.55, zorder=3)
    ax.set_xticks(range(20)); ax.set_xticklabels(AA, fontsize=5.2)
    ax.set_yticks(range(20)); ax.set_yticklabels(AA, fontsize=5.2)
    ax.tick_params(length=1.5, pad=1)
    if label_axes:
        ax.set_xlabel(f'residue at {wt2}{pos2}', fontsize=7)
        ax.set_ylabel(f'residue at {wt1}{pos1}', fontsize=7)
    div = make_axes_locatable(ax)
    at = div.append_axes('top', size='16%', pad=0.06, sharex=ax)
    ar = div.append_axes('right', size='16%', pad=0.06, sharey=ax)
    at.set_title(title, fontsize=7.8, pad=5)
    at.bar(range(20), top, color='#4682b4', width=0.8); ar.barh(range(20), right, color='#4682b4', height=0.8)
    for a_, o in ((at, 'x'), (ar, 'y')):
        a_.tick_params(labelbottom=False, labelleft=False, labelsize=5, length=1.5)
        a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
        a_.axhline(0, color='k', lw=0.4) if o == 'x' else a_.axvline(0, color='k', lw=0.4)
    at.tick_params(axis='y', labelleft=True, labelsize=4.8); ar.tick_params(axis='x', labelbottom=True, labelsize=4.8)
    ax.set_xlim(-0.5, 19.5); ax.set_ylim(-0.5, 19.5)
    return im


def figure(mode, pair, d, fit, meas, model, singles, sp, vmax):
    code, pos1, pos2 = pair
    sub = d[(d.code_wt == code) & (d.pos1 == pos1) & (d.pos2 == pos2)].drop_duplicates(['mt1', 'mt2'])
    wt1, wt2 = sub.wt1.iloc[0], sub.wt2.iloc[0]
    y_single = sp['y'].to_numpy(float)
    ok_single = sp['use'].to_numpy()
    y_s = np.where(ok_single, y_single, np.nan)
    Mm = matrix(sub, np.where(fit, meas[mode], np.nan))
    # the right margin is the singles at the first position (the rows), the top margin the singles at the second (the columns)
    right_m, top_m = margin(code, wt1, pos1, singles, y_s), margin(code, wt2, pos2, singles, y_s)
    cells_ok = np.isfinite(Mm)
    panels = [('Measured', Mm, top_m, right_m, np.nan)]
    rows = []
    for label, v in model.items():
        Mp = matrix(sub, v[mode])
        ts = margin(code, wt2, pos2, singles, v['single']); rs = margin(code, wt1, pos1, singles, v['single'])
        ok = cells_ok & np.isfinite(Mp)
        rho = spearmanr(Mm[ok], Mp[ok])[0] if ok.sum() > 10 and np.ptp(Mp[ok]) > 0 else np.nan
        panels.append((label, Mp, ts, rs, rho))
        rows.append(dict(mode=mode, domain=code, pos1=pos1, pos2=pos2, model=label, spearman=rho, n_cells=int(ok.sum())))
    nr, nc = 4, 4
    fig, axes = plt.subplots(nr, nc, figsize=(17, 16))
    for a_ in axes.ravel():
        a_.axis('off')
    for k, (label, M, top, right, rho) in enumerate(panels):
        ax = axes.ravel()[k]
        ax.axis('on')
        t = ('Measured dddG' + (' − E[dddG | additive]' if mode == 'saturation_corrected' else '')) if k == 0 else f'{label}  (ρ = {rho:.2f})'
        im = heat_panel(ax, M, top, right, t, vmax, wt1, wt2, pos1, pos2, label_axes=(k % nc == 0))
    info = axes.ravel()[len(panels)]
    mode_txt = 'measurement saturation-corrected,\npredictions as is' if mode == 'saturation_corrected' else 'raw'
    info.axis('on'); info.set_xticks([]); info.set_yticks([])
    for s in info.spines.values():
        s.set_visible(False)
    n_meas = int(cells_ok.sum())
    info.text(0.0, 1.0, f'{code}  {wt1}{pos1} × {wt2}{pos2}\n\n{mode_txt}\n\n'
                        f'{len(sub)} of 361 identity pairs in the data,\n{n_meas} with a usable measurement (× : none)\n\n'
                        'rows: residue at the first position;\ncolumns: residue at the second\n'
                        'margins: single-substitution scores\n(measured ddG, or the model\'s single)\n'
                        'colour: blue = positive epistasis, red = negative,\none scale for all panels (below)\n'
                        'ρ: Spearman against the measured cells\n* trained on Megascale data', va='top', ha='left', fontsize=9, transform=info.transAxes)
    cax = info.inset_axes([0.04, 0.10, 0.92, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(('measured dddG − E[dddG | additive]; predictions as is' if mode == 'saturation_corrected' else 'dddG') + f' (kcal/mol); values beyond ±{vmax:.2f} are clipped', fontsize=7.6)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(f'{code} {wt1}{pos1} × {wt2}{pos2}: pair epistasis dddG, {"measurement saturation-corrected, predictions as is" if mode == "saturation_corrected" else "raw"} (capped Tsuboyama test)', fontsize=12, y=0.995)
    fig.subplots_adjust(hspace=0.42, wspace=0.42, left=0.05, right=0.985, top=0.95, bottom=0.03)
    folder = OUT / mode / code
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{wt1}{pos1}_{wt2}{pos2}.png', dpi=DPI)
    plt.close(fig)
    return rows, dict(domain=code, pos1=pos1, pos2=pos2, wt1=wt1, wt2=wt2, identity_pairs=len(sub), usable_cells=n_meas)


def main():
    fr, d, fit, meas, model, singles = build()
    sp = fr.spine
    g = d.drop_duplicates(['code_wt', 'pos1', 'pos2', 'mt1', 'mt2']).groupby(['code_wt', 'pos1', 'pos2']).size()
    pairs = [k for k, n in g.items() if n >= 361]
    print(f'{len(g)} test position pairs, {len(pairs)} with all 361 identity pairs; models drawn: {list(model)}')
    agree, info = [], {}
    pair_rows = np.zeros(len(sp), bool)
    for pair in pairs:
        pair_rows[d.loc[(d.code_wt == pair[0]) & (d.pos1 == pair[1]) & (d.pos2 == pair[2]), 'i'].to_numpy()] = True
    for mode in ('raw', 'saturation_corrected'):
        v = meas[mode][pair_rows & fit]
        vmax = float(np.nanpercentile(np.abs(v[np.isfinite(v)]), 98))
        print(f'{mode}: colour scale ±{vmax:.2f} kcal/mol (98th percentile of |measured| over the usable cells of the {len(pairs)} pairs)')
        for pair in pairs:
            rows, meta = figure(mode, pair, d, fit, meas, model, singles, sp, vmax)
            agree += rows
            info[pair] = meta
        print(mode, 'done')
    pd.DataFrame(list(info.values())).to_csv(OUT / 'pairs.csv', index=False)
    A = pd.DataFrame(agree)
    A.to_csv(OUT / 'agreement.csv', index=False, float_format='%.4f')
    piv = A.pivot_table(index='model', columns='mode', values='spearman', aggfunc='mean').reindex([m[0] for m in MODELS if m[0] in model])
    print('mean over pairs of the per-pair Spearman against the measured cells\n', piv.round(3).to_string())


if __name__ == '__main__':
    main()
