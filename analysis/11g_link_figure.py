#!/usr/bin/env python
"""Publication figure 8 (drafts): the saturating link against the data.  The doubles are the 12,455 Homology-Restricted MegaScale test doubles with both singles measured; the link is applied to
absolute dG (dG_wt + ddG).  Two links are drawn on every option:

  softplus  h(z) = f + tau softplus((z - f) / tau), floor f = 0, tau = 0.59: the link of the training runs (``--link_kind softplus``)
  leaky     u(z) = k + alpha (z - k) + (1 - alpha) tau softplus((z - k) / tau), h(z) = c + w softplus((u - c) / w), knee k = 0, tau = 0.59, alpha = 0.10, clamp c = -1, w = 0.25
            (``--link_kind leaky``, ``esm_msr.link.LeakyFloorLink``)

The additive expectation of a double is dG_wt + ddG_A + ddG_B (``x``, observed) or, undoing each link on the measured dG of the two singles and of the wild type, z = h^-1(A) + h^-1(B) - h^-1(W)
(``z``, latent; the leaky inverse is defined down to the clamp, the softplus one is clipped at 0.05).  A "distant" double has its two residues more than 8 A apart (CB-CB, AlphaFold models of the
Tsuboyama domains): no direct interaction expected, so their observed dG against the additive expectation draws the link itself.

  A  observed double dG against the observed additive dG x, distant doubles | all doubles, both links as curves h(x)
  B  observed double dG against the latent additive expectation z of each link (exact: the curve is h(z)), distant doubles over all doubles
  C  the same as the epistasis dddG = dG_AB - x against x, with the link's saturation epistasis h(x) - x as curves (what fig6a would show with the link on it); all | distant
  D  per-domain medians of dG_AB against x (13 lines) with the pooled median and the links
  E  the distribution of dG_AB in bins of x (ridgeline), the value each link predicts marked per row

Outputs: ``results/06_epistasis/fig8_link_option_*`` and ``fig8_link_binned.csv``.
"""
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.link import singles_of_doubles  # noqa: E402

OUT = out_dir('06_epistasis')
DS = 'tsuboyama_capped_test'
PDB = Path('/home/sareeves/software/esm-msr-release/data/tsuboyama/AlphaFold_model_PDBs')
TITLE = 'Homology-Restricted MegaScale test doubles with both singles measured'
TAU, KNEE, ALPHA, CLAMP, WIDTH = 0.59, 0.0, 0.10, -1.0, 0.25
DISTANT = 8.0                                     # CB-CB distance (A) beyond which a double counts as distant
C_SOFT, C_LEAKY = '#000000', '#e07b00'
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def softplus(v):
    return np.logaddexp(0.0, v)


def h_soft(z):
    return TAU * softplus(np.asarray(z, float) / TAU)


def h_leaky(z):
    z = np.asarray(z, float)
    u = KNEE + ALPHA * (z - KNEE) + (1 - ALPHA) * TAU * softplus((z - KNEE) / TAU)
    return CLAMP + WIDTH * softplus((u - CLAMP) / WIDTH)


def inv_soft(y):
    return TAU * np.log(np.expm1(np.maximum(np.asarray(y, float), 0.05) / TAU))


def inv_leaky(y):
    y = np.maximum(np.asarray(y, float), CLAMP + 0.05)
    v = (y - CLAMP) / WIDTH
    u = CLAMP + WIDTH * np.where(v > 20, v, np.log(np.expm1(np.maximum(v, 1e-12))))
    d = (u - KNEE) / ALPHA
    for _ in range(80):                                                    # Newton from the upper bound, as ``LeakyFloorLink.inverse``
        s = d / TAU
        f = ALPHA * d + (1 - ALPHA) * TAU * softplus(s) - (u - KNEE)
        d = d - f / (ALPHA + (1 - ALPHA) * (1.0 / (1.0 + np.exp(-s))))
    return KNEE + d


def load_pdb(fn):
    res = {}
    for line in open(fn):
        if line.startswith('ATOM'):
            res.setdefault(int(line[22:26]), {'aa': THREE.get(line[17:20], 'X')})[line[12:16].strip()] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    out = {}
    for i, r in res.items():
        if 'CB' in r:
            cb = r['CB']
        else:
            n, ca, c = r['N'], r['CA'], r['C']
            b_, c_ = ca - n, c - ca
            cb = -0.58273431 * np.cross(b_, c_) + 0.56802827 * b_ - 0.54067466 * c_ + ca
        out[i] = (r['aa'], cb)
    return out


def cb_distances(mut, code):
    cache, dist = {}, np.full(len(mut), np.nan)
    for k, (mm, cd) in enumerate(zip(mut, code)):
        f = PDB / f'{cd}.pdb'
        if not f.exists():
            continue
        s = cache.setdefault(cd, load_pdb(f))
        ga, gb = (re.match(r'([A-Z])(\d+)([A-Z])', t) for t in mm.split(':'))
        pa, pb = int(ga.group(2)), int(gb.group(2))
        if pa in s and pb in s and s[pa][0] == ga.group(1) and s[pb][0] == gb.group(1):
            dist[k] = np.linalg.norm(s[pa][1] - s[pb][1])
    return dist


def binned(x, y, edges, nmin=15):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x > lo) & (x <= hi)
        if m.sum() >= nmin:
            rows.append((x[m].mean(), m.sum(), np.median(y[m]), *np.quantile(y[m], [0.25, 0.75])))
    return pd.DataFrame(rows, columns=['x', 'n', 'median', 'q25', 'q75'])


def link_handles():
    return [Line2D([0], [0], color=C_SOFT, lw=1.8, label='softplus link, floor 0, τ 0.59'), Line2D([0], [0], color=C_LEAKY, lw=1.8, ls='--', label='leaky link (knee 0, τ 0.59, α 0.10, clamp −1)')]


def decorate(ax, xl, yl, title, legend=True):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontsize=9, fontweight='bold', loc='left')
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)


def main():
    fr = load_frame(DS)
    sp = fr.spine
    ddg = sp['y'].to_numpy(float)
    c_all = sp['dG_wt'].to_numpy(float)
    ia, ib = singles_of_doubles(sp)
    ok = epi_rows(fr) & (ia >= 0) & (ib >= 0)
    idx = np.flatnonzero(ok)
    W = c_all[ok]
    A, B = W + ddg[ia[ok]], W + ddg[ib[ok]]
    Y = W + ddg[ok]
    dom = sp['group'].to_numpy()[ok]
    dist = cb_distances(sp['mut'].to_numpy(dtype=object)[idx], sp['code'].astype(str).to_numpy()[idx])
    far = dist > DISTANT
    x = A + B - W
    zs = inv_soft(A) + inv_soft(B) - inv_soft(W)
    zl = inv_leaky(A) + inv_leaky(B) - inv_leaky(W)
    n_all, n_far = len(Y), int(far.sum())
    xs = np.linspace(-4.8, 4.8, 400)
    edges = np.arange(-4.5, 4.51, 0.5)
    b_all, b_far = binned(x, Y, edges), binned(x[far], Y[far], edges, 10)
    pd.concat({'all': b_all, 'distant': b_far}, names=['set']).to_csv(OUT / 'fig8_link_binned.csv', float_format='%.4f')
    XL, YL = 'additive dG = dG_wt + ΔΔG_A + ΔΔG_B of the measured singles (kcal/mol)', 'observed dG of the double (kcal/mol)'
    xlim, ylim = (-4.8, 4.8), (-1.25, 5.0)

    def hexpanel(ax, xx, yy, cmap='Blues'):
        return ax.hexbin(xx, yy, gridsize=40, extent=(*xlim, *ylim), mincnt=1, bins='log', cmap=cmap, linewidths=0.1, edgecolors='none')

    def curves(ax, xx=xs, shift=False):
        ax.plot(xx, h_soft(xx) - (xx if shift else 0), color=C_SOFT, lw=1.8, zorder=6)
        ax.plot(xx, h_leaky(xx) - (xx if shift else 0), color=C_LEAKY, lw=1.8, ls='--', zorder=6)

    def median_band(ax, b, col='#7f0000', shift=False):
        off = b['x'] if shift else 0
        ax.fill_between(b['x'], b['q25'] - off, b['q75'] - off, color=col, alpha=0.18, lw=0, zorder=4)
        ax.plot(b['x'], b['median'] - off, color=col, lw=1.4, marker='o', ms=2.5, zorder=5)

    # ---- A
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)
    for ax, (xx, yy, ttl, b) in zip(axes, ((x[far], Y[far], f'distant doubles (CB–CB > {DISTANT:g} Å, n = {n_far:,})', b_far), (x, Y, f'all doubles (n = {n_all:,})', b_all))):
        hexpanel(ax, xx, yy)
        ax.plot(xlim, xlim, color='#999999', ls=':', lw=1.0, zorder=3)
        median_band(ax, b)
        curves(ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        decorate(ax, XL, YL, ttl)
    axes[0].legend(handles=link_handles() + [Line2D([0], [0], color='#999999', ls=':', label='y = x'), Line2D([0], [0], color='#7f0000', marker='o', ms=3, label='median and IQR of the data')], loc='upper left', fontsize=8, framealpha=0.95)
    fig.suptitle(f'{TITLE}: observed double dG against the observed additive dG, the two links as curves', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save(fig, OUT / 'fig8_link_option_A_observed_additive')

    # ---- B: latent, exact
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=True)
    for ax, (z, h, col, ls, name) in zip(axes, ((zs, h_soft, C_SOFT, '-', 'softplus link, floor 0, τ 0.59'), (zl, h_leaky, C_LEAKY, '--', 'leaky link'))):
        lo, hi = max(np.percentile(z, 0.2), -9.0), np.percentile(z, 100)                  # the leaky inverse stretches the deepest readings (slope 1/alpha = 10): the panel stops at -9
        zz = np.linspace(lo - 0.3, hi + 0.3, 400)
        ax.scatter(z, Y, s=3, c='#c8c8c8', alpha=0.35, edgecolors='none', rasterized=True, label=f'all doubles (n = {n_all:,})')
        ax.scatter(z[far], Y[far], s=6, c='#2166ac', alpha=0.55, edgecolors='none', rasterized=True, label=f'distant doubles (n = {n_far:,})')
        ax.plot(zz, h(zz), color=col, lw=2.0, ls=ls, zorder=6, label=f'y = h(z), {name}')
        bz = binned(z[far], Y[far], np.arange(np.floor(lo), hi + 0.5, 0.5), 10)
        median_band(ax, bz)
        ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_ylim(*ylim)
        decorate(ax, 'latent additive expectation z = h⁻¹(dG_A) + h⁻¹(dG_B) − h⁻¹(dG_wt) (kcal/mol)', YL, f'{name}: the latent scale of this link')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.95, markerscale=2.5, title=f'{int((z < lo).sum()):,} double{"" if (z < lo).sum() == 1 else "s"} with z < {lo:.1f} off the panel' if (z < lo).any() else None, title_fontsize=7.5)
    fig.suptitle(f'{TITLE}: observed double dG against the latent additive expectation of each link (distant doubles should lie on the curve)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save(fig, OUT / 'fig8_link_option_B_latent_exact')

    # ---- C: epistasis form
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)
    ylim_c = (-1.6, 4.9)
    for ax, (xx, yy, ttl, b) in zip(axes, ((x, Y, f'all doubles (n = {n_all:,})', b_all), (x[far], Y[far], f'distant doubles (CB–CB > {DISTANT:g} Å, n = {n_far:,})', b_far))):
        ax.hexbin(xx, yy - xx, gridsize=40, extent=(*xlim, *ylim_c), mincnt=1, bins='log', cmap='Blues', linewidths=0.1, edgecolors='none')
        ax.axhline(0, color='#999999', ls=':', lw=1.0, zorder=3)
        median_band(ax, b, shift=True)
        curves(ax, shift=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim_c)
        decorate(ax, XL, 'measured epistasis dddG = dG_AB − additive dG (kcal/mol)', ttl)
    axes[0].legend(handles=[Line2D([0], [0], color=C_SOFT, lw=1.8, label='softplus: h(x) − x'), Line2D([0], [0], color=C_LEAKY, lw=1.8, ls='--', label='leaky: h(x) − x'), Line2D([0], [0], color='#7f0000', marker='o', ms=3, label='median and IQR of the data')],
                   loc='upper right', fontsize=8, framealpha=0.95)
    fig.suptitle(f'{TITLE}: dddG against the additive dG with the saturation epistasis the links predict (exact for unsaturated singles)', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save(fig, OUT / 'fig8_link_option_C_dddG_form')

    # ---- D: per-domain medians
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    cm = plt.get_cmap('tab20')
    for k, d in enumerate(np.unique(dom)):
        m = dom == d
        b = binned(x[m], Y[m], np.arange(-4.5, 4.51, 0.75), 8)
        ax.plot(b['x'], b['median'], color=cm(k % 20), lw=1.0, alpha=0.8, label=f'{d} (n = {int(m.sum()):,})')
    ax.plot(b_all['x'], b_all['median'], color='black', lw=2.6, label='all doubles', zorder=7)
    ax.plot(xs, h_soft(xs), color=C_SOFT, lw=1.4, ls='-', zorder=6)
    ax.plot(xs, h_leaky(xs), color=C_LEAKY, lw=2.2, ls='--', zorder=6)
    ax.plot(xlim, xlim, color='#999999', ls=':', lw=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, 4.6)
    decorate(ax, XL, 'median observed dG of the double (kcal/mol)', f'per-domain medians ({len(np.unique(dom))} domains)')
    ax.legend(handles=link_handles() + ax.get_legend_handles_labels()[0], fontsize=6.4, ncol=2, loc='upper left', framealpha=0.95)
    fig.suptitle(f'{TITLE}: does one link serve every domain?', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save(fig, OUT / 'fig8_link_option_D_per_domain')

    # ---- E: ridgeline of dG_AB by additive dG
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    edges_e = np.arange(-3.0, 4.01, 0.5)
    yy_ = np.linspace(-1.3, 5.2, 300)
    rows = []
    for k, (lo, hi) in enumerate(zip(edges_e[:-1], edges_e[1:])):
        m = (x > lo) & (x <= hi)
        if m.sum() < 25:
            continue
        mid = x[m].mean()
        dens = gaussian_kde(Y[m], bw_method=0.25)(yy_)
        dens = 1.7 * dens / dens.max()
        ax.fill_between(yy_, k, k + dens, color=plt.get_cmap('RdBu')(0.5 + 0.5 * np.tanh(mid / 3)), alpha=0.85, lw=0.6, edgecolor='white', zorder=10 - k * 0.01)
        ax.plot([h_soft(mid)] * 2, [k, k + 1.7], color=C_SOFT, lw=1.6, zorder=20)
        ax.plot([h_leaky(mid)] * 2, [k, k + 1.7], color=C_LEAKY, lw=1.6, ls='--', zorder=20)
        ax.text(5.15, k + 0.1, f'x = {mid:+.1f}  (n = {int(m.sum()):,})', fontsize=7, ha='right', va='bottom', zorder=30)
        rows.append(k)
    ax.axvline(0, color='#999999', ls=':', lw=1.0)
    ax.axvline(-1, color='#bbbbbb', ls=':', lw=1.0)
    ax.set_yticks([])
    ax.set_xlim(-1.3, 5.2)
    ax.set_ylim(-0.1, max(rows) + 2.2)
    decorate(ax, YL, '', 'distribution of the double\'s dG in bins of the additive dG')
    ax.spines['left'].set_visible(False)
    ax.legend(handles=link_handles()[:1] + [Line2D([0], [0], color=C_LEAKY, lw=1.6, ls='--', label='leaky link, value predicted at the bin mean'), Line2D([0], [0], color='#bbbbbb', ls=':', label='dG = −1, the lowest value the assay reports')], loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=7.5, framealpha=0.95)
    fig.suptitle(f'{TITLE}: where the doubles pile up as the additive dG falls', fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    style.save(fig, OUT / 'fig8_link_option_E_ridgeline')
    print(f'{n_all:,} doubles, {n_far:,} distant (CB-CB > {DISTANT:g} A); z range softplus {zs.min():.1f}..{zs.max():.1f}, leaky {zl.min():.1f}..{zl.max():.1f}; x range {x.min():.1f}..{x.max():.1f}')
    for nm, h, z in (('softplus', h_soft, zs), ('leaky', h_leaky, zl)):
        r = Y[far] - h(z[far])
        print(f'  distant doubles, dG_AB - h_{nm}(z): mean {r.mean():+.3f}, sd {r.std():.3f}, RMSE {np.sqrt(np.mean(r ** 2)):.3f}; all doubles RMSE {np.sqrt(np.mean((Y - h(z)) ** 2)):.3f}')
        r2 = Y[far] - h(x[far])
        print(f'  distant doubles, dG_AB - h_{nm}(x observed): mean {r2.mean():+.3f}, RMSE {np.sqrt(np.mean(r2 ** 2)):.3f}')


if __name__ == '__main__':
    main()
