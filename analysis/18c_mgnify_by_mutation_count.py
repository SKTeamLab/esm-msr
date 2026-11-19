"""Supplementary figure: the MGnify and Rocklin results by number of mutations (1, 2, 3+).

Same models, readouts, homolog exclusion and seeds as ``18b_mgnify_models.py`` (the deployed multi-mutant version of every model on every stratum; ESM-MSR combined head, Tobit-linked),
with the measured variants split by ``n_mut``.  Per stratum, one Spearman rho over the stratum's variants (per seed, then averaged); whiskers resample scaffolds (``--B_pooled`` draws, the same for every model).

A stratum with fewer than ``MIN_STRATUM`` variants is not scored or drawn (MGnify holds 3 variants of 3 mutations).  Output: ``results/18_mgnify/mgnify_by_mutation_count.csv`` and
``results/18_mgnify/mgnify_by_mutation_count.{png,pdf}``; ``--figures-only`` redraws from the CSV.

    python analysis/18c_mgnify_by_mutation_count.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis import models as M  # noqa: E402
from msr_analysis.external import register_ext_styles  # noqa: E402
from msr_analysis.registry import MODELS, REFERENCE  # noqa: E402

spec = importlib.util.spec_from_file_location('mgnify_models', HERE / '18b_mgnify_models.py')
mm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm)

OUT = out_dir('18_mgnify')
MIN_STRATUM = 100
STRATA = [(1, '1 mutation'), (2, '2 mutations'), (3, '3+ mutations')]
ORDER = M.ordered(['ESM-MSR', 'ESM3-small-open', 'SPURS', 'ThermoMPNN', 'ProteinMPNN', 'ProSST'])          # the canonical order of the roster
LEGEND = {'ESM-MSR': 'ESM-MSR Dual-view', 'ESM3-small-open': 'ESM3-small-open (σ = 0, WT-view)', 'SPURS': 'SPURS-multi', 'ThermoMPNN': 'ThermoMPNN-D', 'ProteinMPNN': 'ProteinMPNN', 'ProSST': 'ProSST'}      # the doubles version of every model; ESM3-small-open has no Dual-view run on these sets
SETS = {'MGnify': 'MGnify', 'Rocklin designs': 'Rocklin designs'}


def compute(a):
    flags = pd.read_csv(REFERENCE / 'tsuboyama_rule_flags.csv')
    rng = np.random.default_rng(11)
    rows = []
    for name, cfg in mm.SETS.items():
        d, preds, _, _ = mm.prepare(name, cfg, flags)
        nm = np.clip(d['n_mut'].to_numpy(int), 1, 3)
        for k, lab in STRATA:
            sel = np.flatnonzero(nm == k)
            if len(sel) < MIN_STRATUM:
                print(f'{name} {lab}: {len(sel)} variants, not scored')
                rows.append(dict(dataset=name, n_mut=k, metric='n', model='', rho=np.nan, lo=np.nan, hi=np.nan, n_variants=len(sel), n_scaffolds=d.iloc[sel]['code_wt'].nunique()))
                continue
            code, uniq = pd.factorize(d['code_wt'].iloc[sel])
            y = d['y'].to_numpy(float)[sel]
            print(f'{name} {lab}: {len(sel):,} variants, {len(uniq):,} scaffolds', flush=True)
            # pooled: scaffold bootstrap re-ranks every model on each draw
            order = np.argsort(code, kind='stable')
            bounds = np.searchsorted(code[order], np.arange(len(uniq) + 1))
            pdraws = [np.concatenate([order[bounds[i]:bounds[i + 1]] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(a.B_pooled)]
            for mn in ORDER:
                if mn not in preds:
                    continue
                Pl = [p[sel] for p in preds[mn][2]]
                f = lambda idx: float(np.mean([mm.pooled_rho(p[idx], y[idx]) for p in Pl]))
                pt = f(np.arange(len(y)))
                lo, hi = np.percentile([f(dr) for dr in pdraws], [2.5, 97.5])
                rows.append(dict(dataset=name, n_mut=k, metric='pooled', model=mn, rho=pt, lo=lo, hi=hi, n_variants=len(sel), n_scaffolds=len(uniq)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'mgnify_by_mutation_count.csv', index=False, float_format='%.4f')
    return res


def figure(res):
    register_ext_styles()
    scored = res[res.metric == 'pooled']
    ks = {ds: [(k, lab) for k, lab in STRATA if ((scored.dataset == ds) & (scored.n_mut == k)).any()] for ds in SETS}      # a stratum that was not scored is not drawn
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True, gridspec_kw=dict(width_ratios=[len(ks[d]) + 0.25 for d in SETS]))
    for c, ds in enumerate(SETS):
        ax = axes[c]
        R = scored[scored.dataset == ds]
        tab, ci, labels = {}, {}, []
        for k, lab in ks[ds]:
            m = R[R.n_mut == k]
            labels.append(f'{lab}\n({int(m.n_variants.iloc[0]):,} variants)')
            for x in m.itertuples():
                tab[(k, x.model)] = x.rho
                ci[(k, x.model)] = (x.lo, x.hi)
        style.dot_panel(ax, [k for k, _ in ks[ds]], ORDER, tab, ci, highlight=['ESM-MSR'], hollow={(k, 'ESM3-small-open') for k, _ in ks[ds]}, ylabel=r'Spearman $\rho$ (pooled over variants)' if c == 0 else None,
                        xlabels=labels, rot=0, size=6.8, spread=0.78, ylim=(0.0, 0.85))
        ax.text(-0.02 if c else -0.12, 1.02, 'ab'[c], transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')
        ax.set_title(ds, fontsize=9.5, loc='right', color='#444444')
        if c == 0:
            hs = [Line2D([0], [0], label=LEGEND[m], **style.marker_kwargs(m, size=6.8, hollow=m == 'ESM3-small-open')) for m in ORDER]
            ax.legend(handles=hs, loc='upper right', ncol=2, fontsize=7.6, frameon=True, facecolor='white', edgecolor='#cccccc', columnspacing=1.0, handletextpad=0.3, borderpad=0.4)
    fig.tight_layout(w_pad=1.0)
    style.save(fig, OUT / 'mgnify_by_mutation_count')
    print('wrote', OUT / 'mgnify_by_mutation_count')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B_pooled', type=int, default=200)
    ap.add_argument('--figures-only', action='store_true')
    a = ap.parse_args()
    res = pd.read_csv(OUT / 'mgnify_by_mutation_count.csv') if a.figures_only else compute(a)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(res[res.metric == 'pooled'].pivot_table(index=['dataset', 'model'], columns='n_mut', values='rho').round(3).to_string())
    figure(res)


if __name__ == '__main__':
    main()
