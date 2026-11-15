#!/usr/bin/env python
"""Overview of the per-protein screening curves: Tsuboyama, Domainome and ProteinGym Activity side by side, and the rectified gain per variant as its own figure.

``screening_overview_{single,multi}*`` is a 2 x 3 grid, one column per category with two vertically stacked panels:

    Homology-Restricted MegaScale test set    Domainome                              ProteinGym Activity (single mutants)
    hit rate (> 0.5 kcal/mol)                 hit rate (abundance > wild type)       hurt rate, all 43 assays, drop > 0.5 assay SD
    P(at least one hit in the top K)          P(at least one hit in the top K)       hurt rate, the 18 enzymes

(a screen of K variants per protein / assay takes the K best-scored; the dashed line is the random-pick level, the dotted line the oracle; for Activity a variant hurts when it lies more than half an
assay standard deviation below the measured wild type, ``pg_activity/README.md``).  ``screening_gain_per_variant_{single,multi}*`` shows the rectified gain per variant, the rectified
gain @ K (the sum of max(y, 0) over the K picks) divided by K, of Tsuboyama and Domainome.

  single    single mutants only, scored by the single-mutant models
  multi     Tsuboyama singles + doubles, the singles scored by the single-mutant model and the doubles by the multi-mutant model of each family (the models are mixed where a family has both);
            Domainome has single mutants only and is the same in both versions

Each exists with the four model families (markers) or with all models (``_all_models``).  The shaded band is the 95% bootstrap interval of ESM-MSR only.  Every piece of text is wrapped to fit
its panel or the figure and checked to lie inside the figure box.

Reads the curve tables of ``08_screening.py`` (``results/08_screening/screening_per_protein_{singles,composite}*.csv``) and ``pg_activity/02_activity_screening.py``
(``pg_activity/results/screening_{all,enzymes}*.csv``).
"""
import importlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402

scr = importlib.import_module('08_screening')         # model families, labels and colours shared with the other screening figures
IN08 = out_dir('08_screening')
PGR = HERE / 'pg_activity' / 'results'
OUT = out_dir('16_screening_overview')
TSU, DOM = 'tsuboyama_capped_test', 'domainome'
BAND = ('msr', 'msrp0')                                # ESM-MSR: the only model whose CI band is drawn
VERSIONS = {'single': '_singles', 'multi': '_composite'}    # version -> suffix of the tables of 08_screening
MODEL_SETS = {'': True, '_all_models': False}          # suffix -> drawn with markers (four families) or not (all models)
TITLE_FS, NOTE_FS = 9.6, 7.2
ACT_NOTE = ('ProteinGym Activity column: single mutants only, scored by the single-mutant models; a variant hurts when its activity is more than half an assay standard deviation below the measured wild type '
            '(pg_activity/README.md); lower is better.  Shaded band: 95% bootstrap interval of ESM-MSR only.')


def title_of(ds, version, n):
    if ds == DOM:
        t = 'Domainome, single mutants only'
    elif version == 'single':
        t = 'Homology-Restricted MegaScale test set, single mutants only [excl. Cys]'
    else:
        t = 'Homology-Restricted MegaScale test set, singles + doubles [excl. Cys]: singles scored by the single-mutant model, doubles by the multi-mutant model'
    return f'{t} ({n} proteins)'


def labels(version):
    return scr.FAM_LABEL_SINGLES if version == 'single' else scr.FAM_LABEL_COMPOSITE


def draw(ax, d, fams, flab, markers, metric, ref, orc, ylabel):
    """One panel of curves from a table with columns family, metric, K, value, lo, hi."""
    for f in fams:
        x = d[(d.family == f) & (d.metric == metric)].sort_values('K')
        if x.empty:
            continue
        if markers:
            k_ = scr.FAM_STYLE[f] if f != 'msr' else 'msr'
            col, mk = style.style_of(k_)['color'], style.style_of(k_)['marker']
            ax.plot(x.K, x.value, color=col, lw=2.2 if f in BAND else 1.3, marker=mk, ms=4.5, label=flab[f], zorder=4 if f in BAND else 3)
            alpha = .18
        else:
            col, ls, lw, zo = scr.FAM_COLOR_ALL[f]
            ax.plot(x.K, x.value, color=col, ls=ls, lw=lw, label=flab[f], zorder=zo)
            alpha = .15
        if f in BAND:
            ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=alpha, lw=0, zorder=1)
    ref_fam = 'msrp0' if 'msrp0' in d.family.values else 'msr'
    rr = d[(d.family == ref_fam) & (d.metric == ref)].sort_values('K')
    ax.plot(rr.K, rr.value, color='#c0392b', ls=(0, (3, 2)), lw=1.4, zorder=3)
    if orc:
        oo = d[(d.family == ref_fam) & (d.metric == orc)].sort_values('K')
        ax.plot(oo.K, oo.value, color='#2c3e50', ls=':', lw=1.7, zorder=7)
    ax.set_xscale('log')
    ax.set_xticks(sorted(d.K.unique()))
    ax.set_xticklabels(sorted(d.K.unique()))
    ax.set_ylabel(ylabel)


def legend(fig, d, fams, flab, markers, oracle=True):
    handles = []
    for f in fams:
        if f in d.family.values:
            if markers:
                k_ = scr.FAM_STYLE[f] if f != 'msr' else 'msr'
                handles.append(Line2D([0], [0], color=style.style_of(k_)['color'], marker=style.style_of(k_)['marker'], lw=2.0, ms=5, label=flab[f]))
            else:
                col, ls, lw, _ = scr.FAM_COLOR_ALL[f]
                handles.append(Line2D([0], [0], color=col, ls=ls, lw=lw + 0.4, label=flab[f]))
    if oracle:
        handles.append(Line2D([0], [0], color='#2c3e50', ls=':', lw=2.0, label='oracle'))
    handles.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='random picks'))
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=6 if len(handles) > 8 else 5, fontsize=8.8, handlelength=3.4, frameon=True, facecolor='white', edgecolor='#cccccc')


def finish(fig, ax_titles, note, stem):
    """Lay the figure out and make the text fit: the footnote is wrapped to the figure width and reserved at the bottom, the legend at the top, every panel title is wrapped to its panel's
    width; then check that no text leaves the figure box and save."""
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=NOTE_FS, family='monospace', color='#333333')
    t = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = t.get_window_extent(r).height / fig.bbox.height
    top = 1 - fig.legends[0].get_window_extent(r).height / fig.bbox.height - 0.02
    for _ in range(3):
        fig.tight_layout(rect=(0.0, foot + 0.02, 1.0, top))
        for ax, raw in ax_titles:
            ax.set_title(style.wrap_to_px(fig, raw, 0.97 * ax.get_window_extent(r).width, fontsize=TITLE_FS, fontweight='bold'), loc='left', fontsize=TITLE_FS)
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, OUT / stem)


def fig_overview(D, A_all, A_enz, version, fams, markers, stem):
    flab = labels(version)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
    ax_titles = []
    for c, ds in enumerate((TSU, DOM)):
        d = D[D.dataset == ds]
        draw(axes[0, c], d, fams, flab, markers, 'hit_rate', 'rand_hit', 'oracle_hit', 'hit rate (abundance > wild type)' if ds == DOM else 'hit rate (> 0.5 kcal/mol)')
        draw(axes[1, c], d, fams, flab, markers, 'p_recover', 'rand_rec', 'oracle_rec', 'P(at least one hit in top K)')
        ax_titles.append((axes[0, c], title_of(ds, version, int(d.n_proteins.iloc[0]))))
        axes[1, c].set_xlabel('variants screened per protein (K)')
    for r, (A, nm) in enumerate(((A_all, 'all assays'), (A_enz, 'enzymes'))):
        draw(axes[r, 2], A, fams, flab, markers, 'hit_rate', 'rand_hit', None, 'fraction of the K picks\nthat hurt (lower is better)')
        ax_titles.append((axes[r, 2], f'ProteinGym Activity, {nm}, single mutants ({int(A.n_assays.iloc[0])} assays)'))
        axes[r, 2].set_ylim(0, None)
    axes[1, 2].set_xlabel('variants screened per assay (K)')
    legend(fig, D, fams, flab, markers)
    note = ACT_NOTE + ('' if version == 'single' else '  The legend names the models of the Tsuboyama panels (single | multi).')
    finish(fig, ax_titles, note, stem)


def per_variant(d):
    """The gain rows of a curve table divided by K: the rectified gain per variant of the K picks (value and interval)."""
    g = d[d.metric.isin(['gain', 'rand_gain', 'oracle_gain'])].copy()
    for c in ('value', 'lo', 'hi'):
        g[c] = g[c] / g['K']
    return g


def fig_gain(D, version, fams, markers, stem):
    flab = labels(version)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
    ax_titles = []
    for ax, ds in zip(axes, (TSU, DOM)):
        d = D[D.dataset == ds]
        draw(ax, per_variant(d), fams, flab, markers, 'gain', 'rand_gain', None if ds == DOM else 'oracle_gain',      # Domainome: no oracle line, its scale (0.6 at K = 1) hides the models
             'rectified gain per variant (abundance)' if ds == DOM else 'rectified gain per variant (kcal/mol)')
        ax.set_ylim(0, None)
        ax.set_xlabel('variants screened per protein (K)')
        ax_titles.append((ax, title_of(ds, version, int(d.n_proteins.iloc[0]))))
    legend(fig, D, fams, flab, markers)
    finish(fig, ax_titles, 'Rectified gain per variant = the sum of max(y, 0) over the K best-scored variants of a protein divided by K, averaged over proteins; shaded band: 95% bootstrap '
           'interval of ESM-MSR only.  The Domainome panel has no oracle line (it would be about 0.6 at K = 1, an order of magnitude above the models).', stem)


def main():
    for version, mtag in VERSIONS.items():
        for vtag, markers in MODEL_SETS.items():
            fams = scr.FAM if markers else [f for f in scr.FAM_ALL if f != 'geoddg']       # GeoDDG-3D is left out of the all-models version (close to Rosetta)
            D = pd.read_csv(IN08 / f'screening_per_protein{mtag}{vtag}.csv')
            acts = []
            for nm in ('all', 'enzymes'):
                t = pd.read_csv(PGR / f'screening_{nm}{vtag}.csv')
                acts.append(t[t.definition == 'sd'])
            fig_overview(D, acts[0], acts[1], version, fams, markers, f'screening_overview_{version}{vtag}')
            fig_gain(D, version, fams, markers, f'screening_gain_per_variant_{version}{vtag}')
            print('drew', f'screening_overview_{version}{vtag}', f'screening_gain_per_variant_{version}{vtag}')


if __name__ == '__main__':
    main()
