#!/usr/bin/env python
"""Protein-engineering view: what a fixed screening budget buys.

A *hit* is a variant measured more stabilizing than ``--tau`` (default 0.5 kcal/mol); for Domainome, whose label is a change in expression, a hit is any increase (> 0).  Three statistics (definitions in
``msr_analysis/screening.py``):

* **hit rate** - the fraction of the variants a model suggests that are hits (precision of the selection);
* **P(recover >= 1)** - the probability that a screen of the K best-scored variants of a protein contains at least one hit
  (over proteins that have a hit at all; the random-pick probability is drawn for reference);
* **best ddG in the top K** - the most stabilizing measured variant among the K suggested, against the oracle (the best any K
  could contain) and the average of K random picks.

Two designs: one pooled screen per external stability set (a budget of the top 10 % of the set, proteins resampled in the
bootstrap) and one screen per protein (Tsuboyama test libraries, Domainome domains) as the budget K grows.  The retrained
baselines are drawn with the head their training calls for on each set (singles models on singles-only sets, the doubles
models where the set has multi-mutants); ESM-MSR is its WT head on singles-only sets and its combined prediction elsewhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import screening as scr, style  # noqa: E402
from msr_analysis.common import has_multi, out_dir, primary_key  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import LABEL, MODELS  # noqa: E402

OUT = out_dir('08_screening')
POOLED_SETS = ['s783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']   # S571's labels are not ddG
FAM = ['msr', 'me', 'tm', 'spurs']
FAM_LABEL = {'msr': 'ESM-MSR', 'me': 'MutateEverything (st1 / st2)', 'tm': 'ThermoMPNN (/ -D)', 'spurs': 'SPURS (single / multi)'}
TAU_DS = {'domainome': 0.0}      # Domainome's label is a change in expression: a hit is any increase (> 0), not > 0.5 kcal/mol
FAM_STYLE = {'msr': 'msr', 'me': 'me2', 'tm': 'tmpnn', 'spurs': 'spurs'}
for _f, _k in FAM_STYLE.items():          # the dot panels look families up in the registry
    MODELS.setdefault(_f, dict(MODELS[_k]))


def fam_key(fam, fr):
    mm = has_multi(fr)
    k = {'msr': primary_key(fr), 'me': 'me2' if mm else 'me1', 'tm': 'tmpnnd' if mm else 'tmpnn', 'spurs': 'spursm' if mm else 'spurs'}[fam]
    return k if k in fr.preds else None


def pooled(tau, B, frac):
    rows, reps_store = [], {}
    for ds in POOLED_SETS:
        fr = load_frame(ds)
        sp = fr.spine
        for fam in FAM:
            k = fam_key(fam, fr)
            if k is None:
                continue
            ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
            y = sp['y'].to_numpy()[ok]
            cl = sp['cluster'].to_numpy()[ok]
            outs = [scr.pooled_screen(p[ok], y, cl, frac, tau, B, seed=5) for p in fr.preds[k]]
            reps = np.mean([o['reps'] for o in outs], axis=0)
            base = np.mean([o['base'] for o in outs], axis=0)
            reps_store[(ds, fam)] = reps
            for j, m in enumerate(('hit_rate', 'best', 'base_rate', 'gain')):
                lo, hi = np.nanquantile(reps[:, j], [.025, .975])
                rows.append(dict(dataset=ds, family=fam, key=k, metric=m, value=float(base[j]), lo=float(lo), hi=float(hi), n=int(ok.sum()),
                                 oracle_best=float(y.max())))
    return pd.DataFrame(rows), reps_store


def fig_pooled(P, reps_store, tau, frac):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.4), sharex=True)
    for ax, metric, yl in ((axes[0], 'hit_rate', f'hit rate (ddG > {tau})'),
                           (axes[1], 'best', 'best ddG in selection (kcal/mol)'),
                           (axes[2], 'gain', 'mean ddG of selection (kcal/mol)')):
        d = P[P.metric == metric]
        table = {(r.dataset, r.family): r.value for r in d.itertuples()}
        ci = {(r.dataset, r.family): (r.lo, r.hi) for r in d.itertuples()}
        for f in FAM:       # style lookup: draw each family with its registry marker
            MODELS.setdefault(f'fam_{f}', dict(MODELS[FAM_STYLE[f]]))
        style.dot_panel(ax, POOLED_SETS, FAM, {(g, m): v for (g, m), v in table.items()}, ci,
                        highlight=['msr'], ylabel=yl, size=6.5)
        if metric == 'hit_rate':
            br = P[P.metric == 'base_rate'].groupby('dataset').value.first()
            for i, ds in enumerate(POOLED_SETS):
                ax.plot([i - .42, i + .42], [br[ds]] * 2, color='#c0392b', lw=1.2, ls='--', zorder=2)
            ax.plot([], [], color='#c0392b', ls='--', label='hit rate of the whole set (random selection)')
        elif metric == 'gain':
            ax.axhline(0, color='#999', lw=.8, ls=':')
        else:
            orc = P.groupby('dataset').oracle_best.first()
            for i, ds in enumerate(POOLED_SETS):
                ax.plot([i - .42, i + .42], [orc[ds]] * 2, color='#2c3e50', lw=1.2, ls=':', zorder=2)
            ax.plot([], [], color='#2c3e50', ls=':', label='best variant in the set (oracle)')
        if metric != 'gain':
            ax.legend(loc='upper right', ncol=2, fontsize=8)
    # family legend
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], label=FAM_LABEL[f], **style.marker_kwargs(FAM_STYLE[f] if f != 'msr' else 'msr', 7)) for f in FAM]
    axes[0].legend(handles=hs + axes[0].get_legend_handles_labels()[0][-1:], loc='upper left', ncol=3, fontsize=8)
    axes[0].set_title(f'Pooled screen: the top {int(frac * 100)}% of each set by predicted ddG; whiskers 95% protein-level bootstrap')
    axes[2].set_xticklabels([LABEL[d] for d in POOLED_SETS], rotation=30, ha='right')
    fig.tight_layout()
    style.save(fig, OUT / 'screening_pooled_external')


def per_protein(ds, Ks, tau, B):
    fr = load_frame(ds)
    sp = fr.spine
    rows = []
    for fam in FAM:
        k = fam_key(fam, fr)
        if k is None:
            continue
        ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
        y = sp['y'].to_numpy()[ok]
        g = sp['group'].to_numpy()[ok]
        for K in Ks:
            tabs = [scr.per_protein_table(g, p[ok], y, K, tau) for p in fr.preds[k]]
            tab = tabs[0].copy()
            for c in ('hit', 'rec', 'best', 'oracle', 'rand_best', 'rand_rec', 'rand_hit'):
                tab[c] = np.mean([t[c].to_numpy() for t in tabs], axis=0)
            res = scr.boot_summary(tab, B=B, seed=K)
            for m, (v, lo, hi) in res.items():
                rows.append(dict(dataset=ds, family=fam, key=k, K=K, metric=m, value=v, lo=lo, hi=hi, n_proteins=len(tab)))
    return pd.DataFrame(rows)


def fig_per_protein(D, tau):
    tau_of = lambda ds: TAU_DS.get(ds, tau)
    sets = list(D.dataset.unique())
    fig, axes = plt.subplots(len(sets), 3, figsize=(12, 3.4 * len(sets)))
    axes = np.atleast_2d(axes)
    for r, ds in enumerate(sets):
        d = D[D.dataset == ds]
        for c, (metric, yl, ref) in enumerate((('hit_rate', f'hit rate (> {tau_of(ds)})', 'rand_hit'), ('p_recover', f'P(at least one hit in top K)', 'rand_rec'),
                                               ('best', 'best measured value in top K', 'rand_best'))):
            ax = axes[r, c]
            for f in FAM:
                x = d[(d.family == f) & (d.metric == metric)].sort_values('K')
                col = style.style_of(FAM_STYLE[f] if f != 'msr' else 'msr')['color']
                ax.plot(x.K, x.value, color=col, lw=2.2 if f == 'msr' else 1.3, marker=style.style_of(FAM_STYLE[f] if f != 'msr' else 'msr')['marker'], ms=4.5, label=FAM_LABEL[f], zorder=4 if f == 'msr' else 3)
                ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=.12, lw=0)
            rr = d[(d.family == 'msr') & (d.metric == ref)].sort_values('K')
            ax.plot(rr.K, rr.value, color='#c0392b', ls='--', lw=1.1, label='random picks')
            if metric == 'best':
                oo = d[(d.family == 'msr') & (d.metric == 'oracle')].sort_values('K')
                ax.plot(oo.K, oo.value, color='#2c3e50', ls=':', lw=1.2, label='oracle')
            ax.set_xscale('log'); ax.set_xlabel('variants screened per protein (K)'); ax.set_ylabel(yl)
            ax.set_xticks(sorted(d.K.unique())); ax.set_xticklabels(sorted(d.K.unique()))
            if c == 0:
                ax.set_title(f'{LABEL[ds]}  ({int(d.n_proteins.iloc[0])} proteins)', loc='left')
            if r == 0 and c == 2:
                ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    style.save(fig, OUT / 'screening_per_protein')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--B', type=int, default=500)
    a = ap.parse_args()
    P, reps = pooled(a.tau, a.B, a.frac)
    P.to_csv(OUT / 'screening_pooled_external.csv', index=False, float_format='%.4f')
    fig_pooled(P, reps, a.tau, a.frac)
    # paired differences ESM-MSR - baseline in hit rate, per set
    rows = []
    for ds in POOLED_SETS:
        for f in FAM[1:]:
            if (ds, 'msr') in reps and (ds, f) in reps:
                d = reps[(ds, 'msr')][:, 0] - reps[(ds, f)][:, 0]
                rows.append(dict(dataset=ds, baseline=f, d_hit_rate=float(np.nanmean(d)), lo=float(np.nanquantile(d, .025)), hi=float(np.nanquantile(d, .975))))
    pd.DataFrame(rows).to_csv(OUT / 'screening_pooled_paired_diff.csv', index=False, float_format='%.4f')
    D = pd.concat([per_protein('tsuboyama_capped_test', [1, 3, 5, 10, 20, 50], a.tau, a.B),
                   per_protein('domainome', [1, 3, 5, 10, 20, 50], TAU_DS['domainome'], a.B)])
    D.to_csv(OUT / 'screening_per_protein.csv', index=False, float_format='%.4f')
    fig_per_protein(D, a.tau)
    print(P[P.metric == 'hit_rate'].pivot_table(index='family', columns='dataset', values='value')[POOLED_SETS].round(3).to_string())
    print(P[P.metric == 'base_rate'].groupby('dataset').value.first().round(3).to_dict())
    print(D[(D.metric.isin(['hit_rate', 'p_recover'])) & (D.K == 10)].pivot_table(index=['dataset', 'metric'], columns='family', values='value').round(3))


if __name__ == '__main__':
    main()
