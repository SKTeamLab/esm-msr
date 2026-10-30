#!/usr/bin/env python
"""RMSE on the kcal/mol scale, after recalibrating ESM-MSR (and, for a like-for-like comparison, every trained model).

ESM-MSR is trained on protease-derived stabilities, so its output is on the protease scale, not the calorimetric scale of the
external sets.  Rank metrics ignore that, RMSE does not.  Each model's two calibration numbers (scale, bias; OLS) are fitted on
ONE sacrificial set (run twice: S461, and Ssym for its larger dynamic range) and then frozen; every other set is scored out of sample.  S461 itself,
and S669 which contains it, carry an asterisk: their RMSE is not a held-out number.  The retrained baselines are trained on the same
Tsuboyama ddG scale as ESM-MSR, so the same recalibration is applied to them; the raw RMSE is shown faded for ESM-MSR.

Outputs (``results/07_rmse``): ``rmse_by_dataset.csv`` (raw and recalibrated RMSE with protein-level bootstrap intervals),
``calibration_candidates.csv`` (dynamic range of each set as a calibration donor), ``calibration_transfer.csv`` (fit on row set,
score on column set), and ``rmse_recalibrated.png/pdf``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import calib, style  # noqa: E402
from msr_analysis.boot import ci, global_cluster_index  # noqa: E402
from msr_analysis.common import out_dir, primary_key  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import EXT12, LABEL, MODELS  # noqa: E402

OUT = out_dir('07_rmse')
FAMILIES = ['msr', 'me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
# S571's labels are not on a kcal/mol scale (sd 8.9, 5th percentile -20): every model's RMSE there is label-scale noise, so it is left out
SETS = [d for d in EXT12 if d != 's571']
STAR = {'s461', 's669'}                      # sacrificial set and its superset


def key_for(fam, fr):
    return primary_key(fr) if fam == 'msr' else fam


def valid(fr, k):
    return fr.spine['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)


def fit(fam, ds):
    fr = load_frame(ds)
    k = key_for(fam, load_frame('s461'))
    if k not in fr.preds:
        return None
    m = valid(fr, k)
    y = fr.spine['y'].to_numpy()[m]
    return [calib.fit_scale_bias(p[m], y) for p in fr.preds[k]]       # one (a, b) per seed


def boot_sd(v, cluster, B=400, seed=0):
    """95% percentile interval of the standard deviation of ``v`` when the proteins (``cluster``) are resampled with replacement (per-protein n, sum and sum of squares)."""
    v, cl = np.asarray(v, float), np.asarray(cluster)
    u, inv = np.unique(cl, return_inverse=True)
    n = np.bincount(inv).astype(float); s1 = np.bincount(inv, v); s2 = np.bincount(inv, v * v)
    c = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1.0 / len(u)), size=B).astype(float)
    N, S1, S2 = c @ n, c @ s1, c @ s2
    sd = np.sqrt(np.maximum(S2 / N - (S1 / N) ** 2, 0))
    return tuple(np.percentile(sd, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=500)
    b = ap.parse_args()
    # the sacrificial set the document suggests (S461: one-sided, sd 1.25) and one with more dynamic range (Ssym: both directions of every mutation, 5-95% range
    # -3.1 to +3.1, sd 1.83); starred sets are the donor and the sets that contain it
    for calib, stars in (('s461', {'s461', 's669'}), ('ssym', {'ssym'})):
        globals()['STAR'] = stars
        run(argparse.Namespace(calib=calib, B=b.B))


def run(a):
    sfx = '' if a.calib == 's461' else f'_{a.calib}'
    frames = {d: load_frame(d) for d in EXT12}
    # --- calibration donors -------------------------------------------------------------------
    cand = []
    for d, fr in frames.items():
        y = fr.spine['y'].to_numpy()[fr.spine['use'].to_numpy()]
        k = primary_key(fr)
        m = valid(fr, k)
        ab = calib.fit_scale_bias(fr.preds[k].mean(0)[m], fr.spine['y'].to_numpy()[m])
        cand.append(dict(dataset=d, n=len(y), sd_y=y.std(), p05=np.percentile(y, 5), p95=np.percentile(y, 95), range_5_95=np.ptp(np.percentile(y, [5, 95])),
                         fit_scale=ab[0], fit_bias=ab[1], rho_in_set=spearmanr(fr.preds[k].mean(0)[m], fr.spine['y'].to_numpy()[m])[0]))
    pd.DataFrame(cand).to_csv(OUT / f'calibration_candidates{sfx}.csv', index=False, float_format='%.3f')
    frames = {d: fr for d, fr in frames.items() if d in SETS}

    # --- RMSE with frozen calibration, protein-level bootstrap -------------------------------------
    names = {d: fr.spine['cluster'].to_numpy() for d, fr in frames.items()}
    gidx, ncl = global_cluster_index(names)
    rng = np.random.default_rng(3)
    counts = rng.multinomial(ncl, np.full(ncl, 1.0 / ncl), size=a.B).astype(float)
    rows = []
    for fam in FAMILIES:
        ab = fit(fam, a.calib)
        if ab is None:
            continue
        for d, fr in frames.items():
            k = key_for(fam, fr)
            if k not in fr.preds:
                continue
            m = valid(fr, k)
            y = fr.spine['y'].to_numpy()[m]
            if m.sum() < 20:
                continue
            e2_cal = np.stack([(calib.apply(p[m], ab[s]) - y) ** 2 for s, p in enumerate(fr.preds[k])])
            e2_raw = np.stack([(p[m] - y) ** 2 for p in fr.preds[k]])
            g = gidx[d][m]
            res = {}
            for tag, e2 in (('cal', e2_cal), ('raw', e2_raw)):
                point = float(np.mean(np.sqrt(e2.mean(1))))
                reps = np.empty(a.B)
                for b in range(a.B):
                    w = counts[b][g]
                    reps[b] = np.mean(np.sqrt((e2 * w).sum(1) / w.sum()))
                res[tag] = (point,) + ci(reps)
            rows.append(dict(model=fam, key=k, dataset=d, rmse=res['cal'][0], lo=res['cal'][1], hi=res['cal'][2],
                             rmse_raw=res['raw'][0], lo_raw=res['raw'][1], hi_raw=res['raw'][2], n=int(m.sum()),
                             sacrificial=d == a.calib, contains_sacrificial=d in STAR))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / f'rmse_by_dataset{sfx}.csv', index=False, float_format='%.4f')

    # --- transfer matrix for ESM-MSR ------------------------------------------------------------------
    T = np.full((len(SETS), len(SETS)), np.nan)
    for i, dfit in enumerate(SETS):
        ab = fit('msr', dfit)
        for j, dev in enumerate(SETS):
            fr = frames[dev]
            k = primary_key(fr)
            m = valid(fr, k)
            y = fr.spine['y'].to_numpy()[m]
            T[i, j] = np.mean([calib.rmse(calib.apply(p[m], ab[min(s, len(ab) - 1)]), y) for s, p in enumerate(fr.preds[k])])
    pd.DataFrame(T, index=[LABEL[d] for d in SETS], columns=[LABEL[d] for d in SETS]).to_csv(OUT / f'calibration_transfer{sfx}.csv', float_format='%.3f')

    # --- figure ----------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[1, 1.05], hspace=0.5, wspace=0.28)
    fr = frames[a.calib]
    k = primary_key(fr)
    m = valid(fr, k)
    p = fr.preds[k].mean(0)[m]
    y = fr.spine['y'].to_numpy()[m]
    ab = calib.fit_scale_bias(p, y)
    axA = fig.add_subplot(gs[0, 0])
    style.density_scatter(axA, p, y, s=9, cmap='viridis', line=False)
    xs = np.array(axA.get_xlim())
    axA.plot(xs, xs, color='#999', lw=.9, ls=':', label='y = x (as trained)')
    axA.plot(xs, ab[0] * xs + ab[1], color='#c0392b', lw=1.4, label=f'recalibrated: {ab[0]:.2f}·pred {ab[1]:+.2f}')
    axA.set_xlabel('ESM-MSR prediction, protease scale'); axA.set_ylabel(f'measured ddG, {LABEL[a.calib]} (kcal/mol)')
    axA.legend(loc='lower right', fontsize=7.5)
    axA.set_title(f'A  Calibration set: {LABEL[a.calib]}*')
    axB = fig.add_subplot(gs[0, 1])
    sets = [d for d in SETS if d not in ('ptmul', 'ptmuld')]
    # raw ESM-MSR vs measured over all singles sets, by dataset: dynamic range comparison
    sd_pred, sd_y, ci_pred, ci_y = [], [], [], []
    for d in SETS:
        fd = frames[d]; kd = primary_key(fd); md = valid(fd, kd)
        vp, vy, cl = fd.preds[kd].mean(0)[md], fd.spine['y'].to_numpy()[md], fd.spine['cluster'].to_numpy()[md]
        sd_pred.append(vp.std()); sd_y.append(vy.std())
        ci_pred.append(boot_sd(vp, cl)); ci_y.append(boot_sd(vy, cl))
    x = np.arange(len(SETS))
    err = lambda sd, ci: [[max(v - c[0], 0) for v, c in zip(sd, ci)], [max(c[1] - v, 0) for v, c in zip(sd, ci)]]
    axB.bar(x - .2, sd_y, .4, color='#999999', label='measured ddG', yerr=err(sd_y, ci_y), error_kw=dict(ecolor='#c0392b', elinewidth=1.1, capsize=2))
    axB.bar(x + .2, sd_pred, .4, color=MODELS['msr:wt']['color'], label='ESM-MSR (raw)', yerr=err(sd_pred, ci_pred), error_kw=dict(ecolor='#c0392b', elinewidth=1.1, capsize=2))
    axB.set_xticks(x); axB.set_xticklabels([LABEL[d] for d in SETS], rotation=40, ha='right')
    axB.set_ylabel('standard deviation (kcal/mol)\n(whiskers: 95% bootstrap over proteins)'); axB.legend(); axB.set_title('B  Spread of raw vs measured ddG')
    axC = fig.add_subplot(gs[0, 2])
    im = axC.imshow(T, cmap='viridis_r', aspect='auto')
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            axC.text(j, i, f'{T[i, j]:.1f}', ha='center', va='center', fontsize=6.5, color='white' if T[i, j] > np.nanmedian(T) else 'black')
    axC.set_xticks(range(len(SETS))); axC.set_xticklabels([LABEL[d] for d in SETS], rotation=60, ha='right', fontsize=7.5)
    axC.set_yticks(range(len(SETS))); axC.set_yticklabels([LABEL[d] + ('*' if d == a.calib else '') for d in SETS], fontsize=7.5)
    axC.set_xlabel('scored on'); axC.set_ylabel('calibrated on')
    axC.set_title('C  RMSE hardly depends on the calibration set')
    fig.colorbar(im, ax=axC, fraction=.04)
    axD = fig.add_subplot(gs[1, :])
    models = [f for f in FAMILIES if f in set(R.model)]
    table = {(r.dataset, r.model): r.rmse for r in R.itertuples()}
    cis = {(r.dataset, r.model): (r.lo, r.hi) for r in R.itertuples()}
    reg = {f: ('msr:wt' if f == 'msr' else f) for f in models}
    style.dot_panel(axD, SETS, models, table, cis, highlight=['msr'], ylabel='RMSE (kcal/mol), recalibrated',
                    xlabels=[LABEL[d] + ('*' if d in STAR else '') for d in SETS], ylim=(0.6, None))
    # raw ESM-MSR faded
    for i, d in enumerate(SETS):
        r = R[(R.dataset == d) & (R.model == 'msr')]
        if len(r):
            axD.plot(i + (-0.36), r.rmse_raw.iloc[0], marker='x', color='#d62728', ms=9, mew=2.2, ls='none', zorder=12)
    # legend (registry styles; 'msr' stands for WT head / combined)
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], label=MODELS['msr']['label'], **style.marker_kwargs('msr', 7))]
    hs += [Line2D([0], [0], label=MODELS[f]['label'], **style.marker_kwargs(f, 7)) for f in models if f != 'msr']
    hs += [Line2D([0], [0], marker='x', color='#d62728', ms=8, mew=2, ls='none', label='ESM-MSR before recalibration')]
    axD.legend(handles=hs, ncol=4, loc='upper left', fontsize=8)
    axD.set_title(f'D  * calibrated on {LABEL[a.calib]} (and sets containing it, starred): not out-of-sample.  Whiskers: 95% protein-level bootstrap')
    style.save(fig, OUT / f'rmse_recalibrated{sfx}')
    s = R.pivot_table(index='model', columns='dataset', values='rmse')[SETS]
    print(s.round(2).to_string())
    print('scale/bias on', a.calib, ab)
    print(pd.read_csv(OUT / f'calibration_candidates{sfx}.csv').round(2).to_string(index=False))


if __name__ == '__main__':
    main()
