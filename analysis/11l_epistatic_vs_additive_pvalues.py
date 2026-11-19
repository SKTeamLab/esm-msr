#!/usr/bin/env python
"""Does each model's epistatic prediction beat its additive prediction on the Tsuboyama capped-test double mutants?  (figure 3 of ``11_publication_figures``)

Additive = the model's singles summed (ESM-MSR WT-view WT_A + WT_B; ThermoMPNN, SPURS, MutateEverything Stage 1 as the figure reads them; ProteinMPNN, GeoDDG-3D, Rosetta: the sum of the singles); epistatic = the model scoring the
double itself (ESM-MSR Dual-view, MutateEverything Stage 2, ThermoMPNN-D, SPURS-multi, ESM3-small-open Dual-view, ProteinMPNN / GeoDDG-3D / Rosetta direct).  ProSST has an additive version only.
Statistic: the mean over test domains (at least ``MIN_N`` doubles) of the per-domain Spearman rho on the doubles, seeds averaged; delta = rho(epistatic) - rho(additive); paired bootstrap that resamples **domains**
(``--B`` draws shared by all models).  One-sided p = P(delta <= 0) (the claim "epistatic beats additive"); two-sided p = max(2 min(P(delta <= 0), P(delta >= 0)), 1 / (B + 1)); Holm over the models.
ESM-MSR is FINAL_p0 (s1-s3), SPURS the retrain with the trained adapter; the models in the order of ``msr_analysis.models``.

``--metric ndcg96`` replaces the Spearman rho by NDCG@96 with relevance ddG - (the lowest ddG among the doubles of the domain): every double has a non-negative relevance, so the measure does not depend on any
double exceeding ddG = 0, and it is defined for every domain with at least ``MIN_N`` doubles.  ``--metric ndcg96_rectified`` is the older NDCG@96 of figure 1 of ``11_publication_figures`` (relevance
max(ddG, 0), only the domains with a stabilizing double).

Output ``results/06_epistasis/epistatic_vs_additive_pvalues.{csv,md,png,pdf}`` (``..._ndcg96_pvalues.*`` and ``..._ndcg96_rectified_pvalues.*`` for the other metrics).

    python analysis/11l_epistatic_vs_additive_pvalues.py [--B 100000] [--metric spearman|ndcg96|ndcg96_rectified]
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score

from msr_analysis.common import out_dir

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m11i', HERE / '11i_doubles_vs_singles_pvalues.py')
m11i = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11i)
pub, holm, fmt_p = m11i.pub, m11i.holm, m11i.fmt_p

OUT = out_dir('06_epistasis')
KEEP = ['ProteinMPNN', 'ESM3-small-open', 'ESM-MSR', 'MutateEverything', 'ThermoMPNN', 'SPURS', 'best other ddG', 'Rosetta']        # the columns of figure 3 with an epistatic version, in the roster order
HEAD0 = ['Model', 'Additive →\nepistatic', 'ρ\nadditive', 'ρ\nepistatic', 'Δρ [95% CI]', 'Domains\nepistatic higher', 'p\none-sided', 'p one-sided\n(Holm)', 'p\ntwo-sided']


def ndcg96_shifted(p, y):
    """NDCG@96 with relevance y - min(y) over the domain's doubles: every double has a non-negative relevance, so the measure does not depend on any double exceeding ddG = 0 (no rectification)."""
    rel = np.asarray(y, float) - np.min(y)
    if len(rel) < 2 or rel.max() <= 0 or np.ptp(p) == 0:
        return np.nan
    return float(ndcg_score(rel[None, :], np.asarray(p, float)[None, :], k=pub.K, ignore_ties=True))


METRICS = {'spearman': ('', pub.rho, pub.MIN_N), 'ndcg96': ('_ndcg96', ndcg96_shifted, pub.MIN_N), 'ndcg96_rectified': ('_ndcg96_rectified', pub.ndcg96, 1)}


def stem(metric):
    return f'epistatic_vs_additive{METRICS[metric][0]}_pvalues'


def compute(B, metric='spearman'):
    fr, _ = pub.load()
    best, _ = pub.best_other(fr)
    pub.NAMES['best other ddG'] = pub.BEST_LABEL[best]
    sp = fr.spine
    use, n_mut = sp['use'].to_numpy(), sp['n_mut'].to_numpy()
    double = use & (n_mut == 2)
    libs = sorted(np.unique(sp['group'].to_numpy()[double]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=B).astype(float)
    rows = []
    for c in pub.pair_columns(fr, best):
        if c['name'] not in KEEP or c['double'] is None:
            continue
        ka, ke = c['single'], c['double']
        _, fn, min_n = METRICS[metric]
        va, ve = pub.per_library(fr, ka, double, fn, min_n), pub.per_library(fr, ke, double, fn, min_n)
        both = va.index.intersection(ve.index)
        d = pub._diff_df(ve, va, both)
        reps = pub.boot_reps(d, libs, counts)
        reps = reps[np.isfinite(reps)]
        dom = d.mean(axis=1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        label = pub.NAMES.get(c['name'], c['name'])
        rows.append(dict(model=label, additive=ka, epistatic=ke, n_domains=len(both), n_seeds=va.shape[1], rho_additive=float(va.loc[both].mean(axis=1).mean()),
                         rho_epistatic=float(ve.loc[both].mean(axis=1).mean()), delta=float(dom.mean()), lo=lo, hi=hi, domains_higher=int((dom > 0).sum()),
                         p_one=float(max((reps <= 0).mean(), 1.0 / (len(reps) + 1))), p_two=float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1.0 / (len(reps) + 1))), B=B))
    res = pd.DataFrame(rows)
    res['p_one_holm'] = holm(res.p_one)
    res.to_csv(OUT / f'{stem(metric)}.csv', index=False, float_format='%.5g')
    return res


def table(res):
    fl = lambda p, B: fmt_p(p, 1 / (B + 1))
    return [[r.model, f'{r.additive_label} → {r.epistatic_label}', f'{r.rho_additive:.3f}', f'{r.rho_epistatic:.3f}', f'{r.delta:+.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]', f'{r.domains_higher} / {r.n_domains}',
             fl(r.p_one, r.B), fl(r.p_one_holm, r.B), fl(r.p_two, r.B)] for r in res.itertuples()]


LABELS = {'msrp0:wt+': 'WT_A + WT_B', 'msrp0:comb': 'Dual-view', 'me1': 'Stage 1', 'me2': 'Stage 2', 'tmpnn': 'ThermoMPNN', 'tmpnnd': 'ThermoMPNN-D', 'spurs': 'SPURS single', 'spursm': 'SPURS-multi',
          'msrrel@0:wt+': 'WT_A + WT_B', 'msrrel@0:comb': 'Dual-view', 'pmpnn+': 'sum of singles', 'pmpnn': 'direct double', 'geoddg+': 'sum of singles', 'geoddg': 'direct double',
          'rosetta+': 'sum of singles', 'rosetta': 'direct double', 'ddgemb+': 'sum of singles', 'ddgemb': 'direct double'}


def render(res, metric='spearman'):
    res = res.assign(additive_label=res.additive.map(LABELS), epistatic_label=res.epistatic.map(LABELS))
    rows = table(res)
    nd = metric != 'spearman'
    HEAD = [h.replace('ρ', 'NDCG@96') if nd else h for h in HEAD0]
    W = [2.5, 3.2, 1.25, 1.25, 2.8, 1.5, 1.05, 1.25, 1.05] if nd else [2.5, 3.2, 0.9, 0.9, 2.8, 1.5, 1.05, 1.25, 1.05]
    fig = plt.figure(figsize=(sum(W) + 0.4, 0.42 * (len(rows) + 1.6) + 0.85))
    ax = fig.add_axes([0.2 / fig.get_figwidth(), 0.62 / fig.get_figheight(), sum(W) / fig.get_figwidth(), 1 - 0.7 / fig.get_figheight()])
    ax.axis('off')
    t = ax.table(cellText=rows, colLabels=HEAD, loc='upper left', cellLoc='center', colLoc='center', colWidths=[w / sum(W) for w in W], bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor('#cccccc')
        c.set_linewidth(0.6)
        if i == 0:
            c.set_facecolor('#e6e6e6')
            c.set_text_props(fontweight='bold')
            continue
        r = res.iloc[i - 1]
        if j == 0:
            c.set_text_props(ha='left', fontweight='bold')
            c._loc = 'left'
        if j == 7 and r.p_one_holm < 0.05:
            c.set_facecolor('#d6ecd9')
        elif j == 7 and r.delta < 0 and r.p_two < 0.05:
            c.set_facecolor('#f6d5d0')
    fig.text(0.2 / fig.get_figwidth(), 0.5 / fig.get_figheight(), ((f'Tsuboyama capped test set, double mutants. NDCG@96 with relevance ΔΔG − (the domain\'s lowest double ΔΔG), so no double needs to exceed 0; mean over the {int(res.n_domains.iloc[0])} domains with ≥ {pub.MIN_N} doubles, seeds averaged; Δ = epistatic − additive. ' if metric == 'ndcg96' else
             f'Tsuboyama capped test set, double mutants. NDCG@96 (rectified relevance max(ΔΔG, 0)) of the doubles of each domain, mean over the {int(res.n_domains.iloc[0])} domains with a stabilizing double, seeds averaged; Δ = epistatic − additive. ') if nd else
             f'Tsuboyama capped test set, double mutants. ρ: mean over test domains (≥ {pub.MIN_N} doubles) of the per-domain Spearman ρ, seeds averaged; Δρ = epistatic − additive. ')+(
             f'Paired bootstrap over domains\n({int(res.B.iloc[0]):,} draws shared by all models). One-sided p = P(Δ ≤ 0); Holm over the models. Green: epistatic significantly better (one-sided, Holm p < 0.05); '
             'red: additive significantly better (two-sided p < 0.05). ProSST has an additive version only.'), fontsize=8.5, color='#444444', va='top', ha='left')
    fig.savefig(OUT / f'{stem(metric)}.png', dpi=200)
    fig.savefig(OUT / f'{stem(metric)}.pdf')
    plt.close(fig)
    lines = ['| ' + ' | '.join(h.replace('\n', ' ') for h in HEAD) + ' |', '|' + '---|' * len(HEAD)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    (OUT / f'{stem(metric)}.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=100000)
    ap.add_argument('--metric', choices=list(METRICS), default='spearman')
    ap.add_argument('--figures-only', action='store_true')
    a = ap.parse_args()
    res = pd.read_csv(OUT / f'{stem(a.metric)}.csv') if a.figures_only else compute(a.B, a.metric)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(res.round(5).to_string(index=False))
    render(res, a.metric)


if __name__ == '__main__':
    main()
