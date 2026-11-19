#!/usr/bin/env python
"""Supplementary figure: how the inference masking scheme changes ESM3-small-open and ESM-MSR on the Tsuboyama capped test set.

Three schemes: **unmasked** (the sequence is fed as is); **one-at-a-time masking** (each mutated position masked on its own, so the model never sees its partner's mutation unmasked at the position it scores, but sees
the rest of the variant); **masked marginal** (all mutated positions of a variant masked at once, so a double is scored without either mutation visible: epistasis information is erased).

  ESM3-small-open (sigma 0, one seed)   unmasked: combined head of the sigma-0 checkpoint (``msrrel@0:comb``); one-at-a-time: mean of the independent-masking WT and mutant passes (``esm3avg``); masked marginal:
                                         ``esm3mm`` -- the three readouts of ``11e_esm3_scoring_variants.py``, every variant scored with one readout per scheme
  ESM-MSR (seeds 1-3)                    unmasked: FINAL_p0; one-at-a-time: FINAL_p1 (block_mask_wt/mt = 1); masked marginal: FINAL_p1_marg (independent-masked WT pass, mutant pass marginal).  The
                                         schemes are those the model was trained and tested with.  Readout of the doubles: the combined head

Statistic, on the double mutants only: the mean over test domains (at least ``MIN_N`` doubles) of the per-domain Spearman rho, with 95% intervals from resampling domains.  Brackets: paired domain bootstrap (``--B`` draws shared
by all bars) of the difference between two schemes, two-sided p = max(2 min(P(delta <= 0), P(delta >= 0)), 1 / (B + 1)); the table (``masking_schemes_pvalues.csv``) also has the one-sided p that the scheme
beats masked marginal.

Output ``results/00_supplementary_figs/masking_schemes.{png,pdf}`` and ``masking_schemes_pvalues.{csv,md}``.

    python analysis/11m_masking_schemes.py [--B 100000]
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from msr_analysis import models as M
from msr_analysis.common import out_dir
from msr_analysis.final_runs import load_final_frame

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m11i', HERE / '11i_doubles_vs_singles_pvalues.py')
m11i = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11i)
pub, fmt_p = m11i.pub, m11i.fmt_p

OUT = out_dir('00_supplementary_figs')
SCHEMES = [('unm', 'Unmasked'), ('ind', 'One-at-a-time masking'), ('mar', 'Masked marginal')]
SUBSETS = [('doubles', 'Doubles')]
MODELS = {n: M.color(n) for n in ('ESM3-small-open', 'ESM-MSR')}              # the roster's colours, in its order (zero-shot first)


def build():
    fr, _ = pub.load()
    sp = fr.spine
    n1 = (sp['n_mut'] == 1).to_numpy()[None, :]
    fr.preds['ESM3-small-open:unm'] = fr.preds['msrrel@0:comb']
    fr.preds['ESM3-small-open:ind'] = fr.preds['esm3avg']
    fr.preds['ESM3-small-open:mar'] = fr.preds['esm3mm']
    f = load_final_frame(pub.DS, families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
    for tag, fam in (('unm', 'FINAL_p0'), ('ind', 'FINAL_p1'), ('mar', 'FINAL_p1_marg')):
        fr.preds[f'ESM-MSR:{tag}'] = f.preds[f'{fam}:comb']                                              # combined head
    return fr


def compute(B):
    fr = build()
    sp = fr.spine
    use, n_mut = sp['use'].to_numpy(), sp['n_mut'].to_numpy()
    masks = {'doubles': use & (n_mut == 2)}
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    counts = np.random.default_rng(5).multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=B).astype(float)
    vals, bars, pairs = {}, [], []
    for model in MODELS:
        for sub, _ in SUBSETS:
            for sch, _ in SCHEMES:
                v = pub.per_library(fr, f'{model}:{sch}', masks[sub], pub.rho, pub.MIN_N)
                vals[(model, sub, sch)] = v
                mean, lo, hi, n = pub.boot_mean(v, libs, counts)
                bars.append(dict(model=model, subset=sub, scheme=sch, rho=mean, lo=lo, hi=hi, n_domains=n))
            for a, b in (('unm', 'mar'), ('ind', 'mar'), ('unm', 'ind')):
                va, vb = vals[(model, sub, a)], vals[(model, sub, b)]
                both = va.index.intersection(vb.index)
                d = pub._diff_df(va, vb, both)
                reps = pub.boot_reps(d, libs, counts)
                reps = reps[np.isfinite(reps)]
                lo, hi = np.percentile(reps, [2.5, 97.5])
                pairs.append(dict(model=model, subset=sub, scheme_a=a, scheme_b=b, delta=float(d.mean(axis=1).mean()), lo=lo, hi=hi, n_domains=len(both),
                                  p_two=float(max(2 * min((reps <= 0).mean(), (reps >= 0).mean()), 1.0 / (len(reps) + 1))), p_one_a_beats_b=float(max((reps <= 0).mean(), 1.0 / (len(reps) + 1))), B=B))
    bars, pairs = pd.DataFrame(bars), pd.DataFrame(pairs)
    bars.to_csv(OUT / 'masking_schemes_values.csv', index=False, float_format='%.5g')
    pairs.to_csv(OUT / 'masking_schemes_pvalues.csv', index=False, float_format='%.5g')
    return bars, pairs


def bracket(ax, x0, x1, y, text):
    h = 0.012
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color='#333333', lw=0.9, solid_capstyle='butt', clip_on=False, zorder=9)
    ax.text((x0 + x1) / 2, y + 0.006, text, ha='center', va='bottom', fontsize=7.6, color='#333333', zorder=9)


def draw(bars, pairs):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    off = {'unm': -0.28, 'ind': 0.0, 'mar': 0.28}
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=pub.style.GRID, lw=0.6, zorder=0)
    for i, (model, color) in enumerate(MODELS.items()):
        top = {}
        for sch, _ in SCHEMES:
            r = bars[(bars.model == model) & (bars.subset == 'doubles') & (bars.scheme == sch)].iloc[0]
            x = i + off[sch]
            kw = dict(color=color, alpha=1.0, edgecolor='black', lw=0.8) if sch == 'unm' else dict(color=color, alpha=0.62, edgecolor='black', lw=0.8) if sch == 'ind' else \
                dict(color=color, alpha=0.3, edgecolor=color, hatch='////', lw=0.8)
            ax.bar(x, r.rho, 0.26, zorder=3, **kw)
            ax.errorbar(x, r.rho, yerr=[[max(r.rho - r.lo, 0)], [max(r.hi - r.rho, 0)]], fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.5, zorder=5)
            ax.text(x, r.hi + 0.012, f'{r.rho:.2f}', ha='center', va='bottom', fontsize=8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
            top[sch] = r.hi
        q = lambda a_, b_: pairs[(pairs.model == model) & (pairs.subset == 'doubles') & (pairs.scheme_a == a_) & (pairs.scheme_b == b_)].iloc[0]
        y1 = max(top['ind'], top['mar']) + 0.115
        bracket(ax, i + off['ind'], i + off['mar'], y1, fmt_p(q('ind', 'mar').p_two))
        y2 = max(y1 + 0.085, top['unm'] + 0.115)
        bracket(ax, i + off['unm'], i + off['mar'], y2, fmt_p(q('unm', 'mar').p_two))
    n_dom = int(bars.n_domains.max())
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(list(MODELS), fontsize=10.5)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(f"Spearman's ρ on double mutants (mean over {n_dom} test domains)")
    h = [Patch(fc='#555555', ec='black', label='Unmasked'), Patch(fc='#555555', alpha=0.62, ec='black', label='One-at-a-time masking'), Patch(fc='#555555', alpha=0.3, ec='#555555', hatch='////', label='Masked marginal')]
    ax.legend(handles=h, ncol=1, loc='upper left', fontsize=8.5, frameon=True, facecolor='white', edgecolor='#cccccc')
    fig.tight_layout()
    fig.savefig(OUT / 'masking_schemes.png', dpi=300)
    fig.savefig(OUT / 'masking_schemes.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=100000)
    ap.add_argument('--figures-only', action='store_true')
    a = ap.parse_args()
    if a.figures_only:
        bars, pairs = pd.read_csv(OUT / 'masking_schemes_values.csv'), pd.read_csv(OUT / 'masking_schemes_pvalues.csv')
    else:
        bars, pairs = compute(a.B)
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(bars.pivot_table(index=['model', 'subset'], columns='scheme', values='rho').round(3).to_string())
        print(pairs.round(4).to_string(index=False))
    draw(bars, pairs)
    lines = ['| model | subset | A vs B | Δρ (A − B) [95% CI] | p two-sided | p one-sided (A beats B) |', '|---|---|---|---|---|---|']
    for r in pairs.itertuples():
        lines.append(f'| {r.model} | {r.subset} | {r.scheme_a} vs {r.scheme_b} | {r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {fmt_p(r.p_two, 1 / (r.B + 1))} | {fmt_p(r.p_one_a_beats_b, 1 / (r.B + 1))} |')
    (OUT / 'masking_schemes_pvalues.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
