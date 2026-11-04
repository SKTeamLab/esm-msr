#!/usr/bin/env python
"""Publication figure 7: which kinds of epistasis the Homology-Restricted MegaScale test doubles hold as the additive expectation moves into the assay floor (the figure of the Oct 21 epistasis
review, ``epi_types_by_additive_expectation``, redone on the current data with the link at floor 0, tau 0.59).

Every double with both singles measured (12,455).  With absolute dG of the wild type W, the singles A, B and the double Y (clipped at -1) and the link h(z) = LO + TAU softplus((z - LO) / TAU),
LO = 0, TAU = 0.59 (h^-1 clipped at LO + 0.05):

  z        the additive expectation in latent space, h^-1(A) + h^-1(B) - h^-1(W): the x axis, in bins of constant width 1 from -6 to 6 (z below -6 or above 6 goes to the end bins)
  eps      epistasis as measured, floor included: Y - (A + B - W) = ddG_AB - ddG_A - ddG_B
  net      the singles' summed effect, ddG_A + ddG_B = (A - W) + (B - W): only the counts below the bars are split by its sign (net < 0, net > 0)

The types are a ladder on eps (D = 0.5, S = 1.0 kcal/mol; D is about twice the noise of eps):

  compensation / positive sign  eps > S
  weak positive                 D < eps <= S
  near-neutral                  |eps| <= D
  weak negative                 -S <= eps < -D
  clash / negative sign         eps < -S

The part of each type that the floor decides, |h(z) - (A + B - W)| > D (the saturation the link expects of the singles alone is already larger than the epistasis threshold; it depends on the
singles only, not on how the double came out), is hatched in black.  95% intervals of the shares (resampling domains, the same draw for every bin) are in
``fig7_epistasis_types_by_additive_dG.csv``.  Outputs: ``results/06_epistasis``.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.link import singles_of_doubles  # noqa: E402

OUT = out_dir('06_epistasis')
DS = 'tsuboyama_capped_test'
LO, TAU, EPS, CLIP = 0.0, 0.59, 0.05, -1.0       # the link (floor, tau), the clip of h^-1, the clip of the measured dG
D, S = 0.5, 1.0                                  # epistasis is called beyond +-D; strong beyond +-S (kcal/mol)
TYPES = ['compensation / positive sign', 'weak positive', 'near-neutral', 'weak negative', 'clash / negative sign']
RANGES = [f'ε > {S:g}', f'{D:g} < ε ≤ {S:g}', f'|ε| ≤ {D:g}', f'−{S:g} ≤ ε < −{D:g}', f'ε < −{S:g}']
COLOURS = ['#2166ac', '#92c5de', '#d9d9d9', '#f4a582', '#b2182b']
HALVES = ['negative', 'positive']
EDGES = np.arange(-6, 7)
LABELS = [f'{a}..{b}' for a, b in zip(EDGES[:-1], EDGES[1:])]


def h(z):
    return LO + TAU * np.logaddexp(0, (z - LO) / TAU)


def hinv(y):
    return LO + TAU * np.log(np.expm1((np.maximum(y, LO + EPS) - LO) / TAU))


def ladder(eps):
    return np.where(eps > S, 0, np.where(eps > D, 1, np.where(eps >= -D, 2, np.where(eps >= -S, 3, 4))))


def shares(bi, half, cat, flag, dom, doms, counts):
    """Per bin and type: the share of the bin's doubles and the part of it that the floor decides, with 95% intervals over domains (``counts``: one multinomial draw per replicate); also the number of
    doubles of each bin with the singles' net ddG below and above 0."""
    nb, nc = len(LABELS), len(TYPES)
    N, F, H = np.zeros((len(doms), nb, nc)), np.zeros((len(doms), nb, nc)), np.zeros((len(doms), nb, 2))
    for k, d in enumerate(doms):
        m = dom == d
        np.add.at(N[k], (bi[m], cat[m]), 1)
        np.add.at(F[k], (bi[m & flag], cat[m & flag]), 1)
        np.add.at(H[k], (bi[m], half[m]), 1)
    tot, fl = N.sum(0), F.sum(0)
    nbin = np.maximum(tot.sum(1, keepdims=True), 1)
    rn, rf = np.einsum('rk,kbc->rbc', counts, N), np.einsum('rk,kbc->rbc', counts, F)
    rb = np.maximum(rn.sum(2, keepdims=True), 1)
    ci = lambda r: (np.nanpercentile(r / rb, 2.5, axis=0), np.nanpercentile(r / rb, 97.5, axis=0))
    return dict(share=tot / nbin, floor=fl / nbin, n=tot, nbin=tot.sum(1), half=H.sum(0), ci=ci(rn), ci_floor=ci(rf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=200)
    ap.add_argument('--x', choices=['latent', 'observed'], default='latent', help='x axis: the latent additive expectation z (default) or the observed additive dG = dG_wt + ddG_A + ddG_B (file fig7_alt_observed_additive_dG)')
    a = ap.parse_args()
    fr = load_frame(DS)
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    c = sp['dG_wt'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = epi_rows(fr) & (ia >= 0) & (ib >= 0)
    W = np.maximum(c[ok], CLIP)
    A, Bm = np.maximum(c[ok] + ddg[ia[ok]], CLIP), np.maximum(c[ok] + ddg[ib[ok]], CLIP)
    Y = np.maximum(c[ok] + ddg[ok], CLIP)
    dom = sp['group'].to_numpy()[ok]
    z = hinv(A) + hinv(Bm) - hinv(W)
    x = A + Bm - W
    eps = Y - x
    net = (A - W) + (Bm - W)
    if a.x == 'observed':
        z = x                                                                     # bin by the observed additive dG; the floor decides where h(x) - x is beyond D
    flag = np.abs(h(z) - x) > D
    bi = np.clip(np.floor(z).astype(int) + 6, 0, len(LABELS) - 1)
    half = (net > 0).astype(int)
    cat = ladder(eps)
    doms = np.unique(dom)
    counts = np.random.default_rng(5).multinomial(len(doms), np.full(len(doms), 1.0 / len(doms)), size=a.B).astype(float)
    r = shares(bi, half, cat, flag, dom, doms, counts)
    rows = []
    for b in range(len(LABELS)):
        for t in range(len(TYPES)):
            rows.append(dict(z_bin=LABELS[b], type=TYPES[t], n_bin=int(r['nbin'][b]), n_net_ddG_additive_negative=int(r['half'][b, 0]), n_net_ddG_additive_positive=int(r['half'][b, 1]), n=int(r['n'][b, t]), share=r['share'][b, t],
                             lo=r['ci'][0][b, t], hi=r['ci'][1][b, t], share_floor_decides=r['floor'][b, t], lo_floor=r['ci_floor'][0][b, t], hi_floor=r['ci_floor'][1][b, t]))
    stem = 'fig7_epistasis_types_by_additive_dG' if a.x == 'latent' else 'fig7_alt_observed_additive_dG'
    pd.DataFrame(rows).to_csv(OUT / f'{stem}.csv', index=False, float_format='%.4f')

    plt.rcParams['hatch.linewidth'] = 0.7
    fig, (ax, an) = plt.subplots(2, 1, figsize=(8.6, 7.4), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    nb, nc = len(LABELS), len(TYPES)
    xs = np.arange(nb)
    bottom = np.zeros(nb)
    for t in range(nc):
        rest = r['share'][:, t] - r['floor'][:, t]
        ax.bar(xs, r['floor'][:, t], bottom=bottom, color=COLOURS[t], width=0.9, edgecolor='black', lw=0, hatch='////')          # the hatched part at the bottom of each type
        bottom += r['floor'][:, t]
        ax.bar(xs, rest, bottom=bottom, color=COLOURS[t], width=0.9, edgecolor='white', lw=0.4)
        bottom += rest
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of doubles')
    ov = np.bincount(cat, minlength=nc) / len(cat)
    handles = [Patch(fc=COLOURS[t], ec='#999999', lw=0.4) for t in range(nc)] + [Patch(fc='white', ec='black', lw=0, hatch='////')]
    labels = [f'{TYPES[t]} ({RANGES[t]}; {100 * ov[t]:.0f}%)' for t in range(nc)] + [f'hatched: floor decides ({100 * flag.mean():.0f}%)']
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=6.5, handlelength=1.3, frameon=True, facecolor='white', framealpha=1.0, edgecolor='#999999', borderpad=0.45, labelspacing=0.3).set_zorder(10)
    wd = 0.45
    an.bar(xs - wd / 2, r['half'][:, 0], color='#888888', width=wd, edgecolor='white', lw=0.4, label='singles\' summed ΔΔG < 0')
    an.bar(xs + wd / 2, r['half'][:, 1], color='#222222', width=wd, edgecolor='white', lw=0.4, label='singles\' summed ΔΔG > 0')
    an.set_yscale('log')
    an.set_ylabel('n')
    an.legend(loc='upper right', fontsize=7.4, frameon=False, handlelength=1.2)
    an.set_xticks(range(nb))
    an.set_xticklabels(LABELS, rotation=45, fontsize=8)
    an.set_xlabel('additive expected dG: latent z (kcal/mol)' if a.x == 'latent' else 'additive dG = dG_wt + ΔΔG_A + ΔΔG_B of the measured singles (kcal/mol)')
    for a_ in (ax, an):
        for sp_ in ('top', 'right'):
            a_.spines[sp_].set_visible(False)
    fig.suptitle(f'Homology-Restricted MegaScale test doubles with both singles measured (n = {len(z):,}):\nkinds of epistasis as the additive expectation falls into the assay floor (uncorrected; hatched: the floor decides)',
                 fontsize=9.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    style.save(fig, OUT / stem)
    pd.set_option('display.width', 240)
    print(f'n = {len(z):,}; net ddG additive > 0: {int(half.sum())} ({100 * half.mean():.1f}%); floor decides {100 * flag.mean():.1f}%')
    print('overall shares of all doubles:', {TYPES[t]: round(float(ov[t]), 3) for t in range(nc)})
    print(pd.DataFrame({TYPES[t]: [f'{100 * r["share"][b, t]:.0f} ({100 * r["floor"][b, t]:.0f})' for b in range(nb)] for t in range(nc)}, index=LABELS).assign(n=r['nbin'].astype(int), n_net_pos=r['half'][:, 1].astype(int)).to_string())

if __name__ == '__main__':
    main()
