#!/usr/bin/env python
"""Publication figures: the notebooks' headline bar charts, recomputed from the collected predictions with protein-level bootstrap intervals.

All four are on the capped Tsuboyama **test** libraries (33 proteins; measured rows only, all backbones).  Error bars are 95% intervals from resampling
**libraries** (proteins) with replacement, the same draw for every model and subset (the notebooks' bars were the spread over three models/seeds).
Multi-seed models are seed means of the per-library value.

Columns are grouped in three labelled categories:
  retrained   ESM-MSR (WT head and combined head, both always shown: the WT head is the cheap one and the focus of the paper) and the baselines retrained on the
              capped split.  A baseline gets one column, scored by the model that fits the subset: MutateEverything stage 1 for singles and stage 2 for doubles,
              ThermoMPNN for singles and ThermoMPNN-D for doubles, SPURS for singles and SPURS multi for doubles ("X | Y" on the axis).
  zero-shot   ESM3-small-open (the sigma = 0 control of the ESM-MSR checkpoint, scored by masked marginal; ``--esm3 independent`` for independent masking), ProteinMPNN and ProSST.
  supervised  published ddG predictors that are not trained on Megascale: Rosetta and the better of GeoDDG-3D / DDGemb (mean per-library singles rho; ESM3dG is left out,
              its MGnify fine-tune is recent and overlaps the test proteins).

fig1  NDCG@96 (rectified relevance max(ddG, 0), sklearn ndcg_score as in the notebooks), per library then averaged, singles against doubles.
fig2  Spearman rho, stabilizing (measured ddG > 0) against destabilizing (< 0) mutations, one panel for singles and one for doubles (each column is the
      model that scores that subset); the overall rho of the panel is the black line.  ``*_pooled``: rho pooled over libraries instead of averaged.
fig3  Spearman rho on double mutants, additive version of a model against its epistatic version: the singles model summed over the double's two substitutions
      (ESM-MSR WT head, ME stage 1, ThermoMPNN, SPURS, Rosetta's sum of singles, ESM3-small-open's WT pass) against the model that scores the double itself (ESM-MSR
      combined, ME stage 2, ThermoMPNN-D, SPURS multi, Rosetta's direct double, ESM3-small-open's mutant pass).  ProSST is additive by construction (one bar).
fig4  Naive epistasis rho: predicted against measured dddG over doubles with both singles measured, positive (dddG > 0) against negative epistasis.  The predicted
      epistasis is always the double-mutant prediction of one model minus the sum of the single-mutant predictions of another (or another head): written under each column.

Until the ESM3-small-open masked runs (cluster/esm3_masked_gamut.sh) are copied to ``msr_predictions/esm3_small_open_sigma0_masked`` the ESM3 column falls back to the
unmasked sigma = 0 readout and says so.  Outputs: ``results/11_publication``.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import epi_rows, out_dir, subframe  # noqa: E402
from msr_analysis.data import add_esm3_masked, load_frame  # noqa: E402

OUT = out_dir('11_publication')
DS = 'tsuboyama_capped_test'
B = 2000
MIN_N = 10
K = 96

CATS = {'retrained': 'Retrained on the capped Tsuboyama split', 'zeroshot': 'Zero-shot', 'supervised': 'Supervised, not trained\non Megascale'}
BEST_LABEL = {'geoddg': 'GeoDDG-3D', 'ddgemb': 'DDGemb'}          # ESM3ΔG is left out on purpose: its MGnify fine-tune is recent and overlaps the test proteins (BENCHMARKS §6)
BEST_COLOR = {'geoddg': '#117a65', 'ddgemb': '#b7950b'}
F_COMPOSITE = 'X | Y: singles scored by X, doubles by Y.'
F_BEST = '† trained on S8754 (ProThermDB + ThermoMutDB), not on Megascale; chosen over DDGemb by mean per-library singles ρ. ESM3ΔG is not shown: homology to its MGnify fine-tune confounds the comparison.'
F_INTERIM = '‡ unmasked sigma = 0 readout: the independent-masking run (WT pass against mutant pass) has not been copied here yet.'


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load():
    fr = load_frame(DS)
    p = fr.preds
    added = add_esm3_masked(fr, DS)
    if 'msrrel@0:comb' in p and 'msrrel@0:wt' in p:
        p['msrrel@0:epi'] = 2.0 * (p['msrrel@0:comb'] - p['msrrel@0:wt'])         # comb = (WT + MT) / 2, so MT - WT = 2 (comb - WT)
    for k in ('rosetta', 'geoddg', 'ddgemb', 'pmpnn'):
        if k in p and k + '+' in p:
            p[k + ':epi'] = p[k] - p[k + '+']                                      # direct double minus the sum of the model's singles
    return fr, added


def per_library(fr, key, mask, fn, min_n, y=None):
    """{library: value}, the value of ``fn(pred, y)`` averaged over the model's seeds; libraries with fewer than ``min_n`` rows in ``mask`` are left out."""
    sp = fr.spine
    y = sp['y'].to_numpy() if y is None else y
    g = sp['group'].to_numpy()
    out = {}
    for lib in np.unique(g[mask]):
        m = mask & (g == lib) & np.isfinite(fr.preds[key]).all(axis=0)
        if m.sum() < min_n:
            continue
        vals = [fn(p[m], y[m]) for p in fr.preds[key]]
        v = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        if np.isfinite(v):
            out[lib] = v
    return pd.Series(out, dtype=float)


def rho(p, y):
    return spearmanr(p, y)[0] if len(y) >= 3 and np.ptp(p) > 0 else np.nan


def ndcg96(p, y):
    rel = np.where(y > 0, y, 0.0)
    if not (rel > 0).any():
        return np.nan
    return float(ndcg_score(rel[None, :], p[None, :], k=K, ignore_ties=True))


def boot_mean(values, libs, counts):
    """values: Series library -> value.  Returns (mean, lo, hi, n_libraries) under the shared library resamples ``counts`` (B x n_libs, columns = ``libs``)."""
    v = values.reindex(libs)
    ok = v.notna().to_numpy()
    point = float(v[ok].mean())
    c = counts[:, ok]
    den = c.sum(axis=1)
    reps = (c @ v[ok].to_numpy()) / np.where(den > 0, den, np.nan)
    lo, hi = ci(reps)
    return point, lo, hi, int(ok.sum())


def best_other(fr):
    """The supervised, not-Megascale-trained published ddG predictor with the highest mean per-library singles rho."""
    sp = fr.spine
    single = sp['use'].to_numpy() & (sp['n_mut'] == 1).to_numpy()
    sc = {k: per_library(fr, k, single, rho, MIN_N).mean() for k in BEST_LABEL if k in fr.preds}
    best = max(sc, key=sc.get)
    pd.Series(sc, name='mean per-library singles rho').rename_axis('model').to_csv(OUT / 'best_other_ddg_model.csv', float_format='%.4f')
    return best, sc


# ----------------------------------------------------------------------------------------------
# columns: name, category, colour, labels, and the prediction column to read for singles / doubles
# ----------------------------------------------------------------------------------------------
SAME = object()


def col(name, cat, color, label, single, double=SAME, **kw):
    """``double=SAME`` reads the same prediction column for doubles as for singles; ``None`` leaves the doubles cell empty."""
    return dict(name=name, cat=cat, color=color, label=label, single=single, double=single if double is SAME else double, **kw)


def esm3_choice(fr, want):
    """(key, label, interim?) for the ESM3-small-open column and the alternative strategy kept in the tables only."""
    p = fr.preds
    names = {'marginal': ('esm3mm', 'ESM3-small-open\nmasked marginal'), 'independent': ('esm3ind', 'ESM3-small-open\nindependent masking')}
    order = [want] + [s for s in names if s != want]
    have = [s for s in order if names[s][0] in p]
    if not have:
        return 'msrrel@0:wt', 'ESM3-small-open‡\nunmasked readout', True, None
    alt = names[have[1]] if len(have) > 1 else None
    return names[have[0]][0], names[have[0]][1], False, alt


def main_columns(fr, best, esm3_want):
    ek, el, interim, alt = esm3_choice(fr, esm3_want)
    cols = [col('ESM-MSR WT head', 'retrained', '#000000', 'ESM-MSR\nWT head', 'msr:wt'),
            col('ESM-MSR combined', 'retrained', '#666666', 'ESM-MSR\ncombined', 'msr:comb'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'ME\nst1 | st2', 'me1', 'me2', label_s='ME\nst1', label_d='ME\nst2'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\n| -D', 'tmpnn', 'tmpnnd', label_s='ThermoMPNN', label_d='ThermoMPNN-D'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\n| multi', 'spurs', 'spursm', label_s='SPURS', label_d='SPURS multi'),
            col('ESM3-small-open', 'zeroshot', '#7f8c8d', el, ek),
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN', 'pmpnn'),
            col('ProSST', 'zeroshot', '#8e44ad', 'ProSST', 'prosst'),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best] + '†', best),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta', 'rosetta')]
    if alt:
        cols.append(col('ESM3-small-open (alt)', 'zeroshot', '#7f8c8d', alt[1], alt[0], draw=False))
    return cols, interim


def pair_columns(fr, best):
    """Fig 3: single = the additive version's key, double = the epistatic version's key."""
    p = fr.preds
    if 'esm3ind' in p and 'esm3ind+' in p:
        es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open\nWT pass\n→ mutant pass\n(independent masking)', 'esm3ind+', 'esm3ind')
        interim = False
    else:
        es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open‡\nWT pass\n→ combined\n(unmasked)', 'msrrel@0:wt', 'msrrel@0:comb')
        interim = True
    cols = [col('ESM-MSR', 'retrained', '#000000', 'ESM-MSR\nWT head\n→ combined', 'msr:wt', 'msr:comb'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'ME\nst1\n→ st2', 'me1', 'me2'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\n→ ThermoMPNN-D', 'tmpnn', 'tmpnnd'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\n→ SPURS multi', 'spurs', 'spursm'),
            es,
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN\nsum of singles\n→ direct double', 'pmpnn+', 'pmpnn'),
            col('ProSST', 'zeroshot', '#8e44ad', 'ProSST\n(additive)', 'prosst', None),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best] + '†\nsum of singles\n→ direct double', best + '+', best),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta\nsum of singles\n→ direct double', 'rosetta+', 'rosetta')]
    return cols, interim


def epi_columns(fr, best):
    p = fr.preds
    if 'esm3ind:epi' in p:
        es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open\ndouble: mutant pass\nsingles: WT pass', None, 'esm3ind:epi')
        interim = False
    else:
        es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open‡\ndouble: combined\nsingles: WT pass', None, 'msrrel@0:epi')
        interim = True
    cols = [col('ESM-MSR', 'retrained', '#000000', 'ESM-MSR\ndouble: MT head\nsingles: WT head', None, 'msr:epi'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'MutateEverything\ndouble: stage 2\nsingles: stage 1', None, 'me2:epi'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\ndouble: ThermoMPNN-D\nsingles: ThermoMPNN', None, 'tmpnnd:epi'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\ndouble: SPURS multi\nsingles: SPURS', None, 'spursm:epi'),
            es,
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN\ndouble: direct\nsingles: summed', None, 'pmpnn:epi'),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best] + '†\ndouble: direct\nsingles: summed', None, best + ':epi'),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta\ndouble: direct\nsingles: summed', None, 'rosetta:epi')]
    return cols, interim


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
def key_for(c, which):
    return c['single'] if which == 'singles' else c['double']


def table_grouped(fr, cols, cells, fn, min_n, counts, libs, y=None):
    """Rows (column, subset): mean over libraries of ``fn``, with the library bootstrap.  ``cells``: subset -> (row mask, 'singles' | 'doubles' = which key of the column)."""
    g = fr.spine['group'].to_numpy()
    rows = []
    for c in cols:
        for sub, (m, which) in cells.items():
            key = key_for(c, which)
            if key is None or key not in fr.preds:
                continue
            v = per_library(fr, key, m, fn, min_n, y)
            if len(v) < 2:
                continue
            pt, lo, hi, n = boot_mean(v, libs, counts)
            rows.append(dict(column=c['name'], key=key, subset=sub, value=pt, lo=lo, hi=hi, n_libraries=n, n_variants=int((m & np.isin(g, v.index)).sum())))
    return pd.DataFrame(rows)


def table_pooled(fr, cols, cells, y, B_=1000, seed=7):
    """Rho pooled over libraries, with the library-level bootstrap, for the same (column, subset) cells."""
    rows = []
    for sub, (m, which) in cells.items():
        names = {key_for(c, which): c['name'] for c in cols if key_for(c, which) in fr.preds}
        keys = list(names)
        sf = subframe(fr, m & np.isfinite(y), y=y, keys=keys)
        point, reps = bootstrap_rho({DS: sf}, keys, kind='protein', B=B_, seed=seed, per_protein=False)
        for k, (r, per_seed, n, ng) in point[DS].items():
            lo, hi = ci(reps[DS][k])
            rows.append(dict(column=names[k], key=k, subset=sub, value=r, lo=lo, hi=hi, n_variants=n, n_libraries=ng))
    return pd.DataFrame(rows)


def save_tab(T, name):
    T.to_csv(OUT / f'{name}.csv', index=False, float_format='%.4f')


def count_text(T, sub, text):
    """'text (n = 1,234 variants, 33 libraries)' from the table's rows of one subset."""
    r = T[T.subset == sub]
    return f'{text} (n = {int(r.n_variants.max()):,}, {int(r.n_libraries.max())} libraries)'


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
def layout(cols):
    xs, spans, x, prev = [], {}, 0.0, None
    for c in cols:
        if prev is not None and c['cat'] != prev:
            x += 0.7
        xs.append(x)
        lo, hi = spans.get(c['cat'], (x, x))
        spans[c['cat']] = (min(lo, x), max(hi, x))
        prev = c['cat']
        x += 1.0
    return xs, spans


def bars(ax, cols, T, subsets, sub_labels, ylabel, ylim, overall=None, overall_label='overall', tick='label', headers=True, annotate=True, zero_line=False, tick_fs=8):
    """Two bars per column (first subset hatched), 95% intervals, the overall value as a black line behind them, three labelled categories."""
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = layout(cols)
    w = 0.38
    for c, x in zip(cols, xs):
        have = [s for s in subsets if len(T[(T.column == c['name']) & (T.subset == s)])]
        for j, sub in enumerate(subsets):
            r = T[(T.column == c['name']) & (T.subset == sub)]
            if not len(r):
                continue
            r = r.iloc[0]
            xx = x + (j - 0.5) * w * 1.08 if len(have) == 2 else x
            ax.bar(xx, r.value, w, color=c['color'], alpha=0.95 if j else 0.55, edgecolor=c['color'] if j == 0 else 'black', hatch='////' if j == 0 else None, lw=0.8, zorder=3)
            ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.5, zorder=5)
            if annotate:
                ax.text(xx, max(r.hi, r.value, 0) + 0.012 * (ylim[1] - ylim[0]), f'{r.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=6)
        if overall is not None:
            o = overall[overall.column == c['name']]
            if len(o):
                ax.plot([x - w * 1.1, x + w * 1.1], [o.iloc[0].value] * 2, color='black', lw=1.6, zorder=2.2, solid_capstyle='butt')
    ax.set_xticks(xs)
    ax.set_xticklabels([c.get('label_' + tick[-1], c['label']) if tick in ('label_s', 'label_d') else c['label'] for c in cols], fontsize=tick_fs)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in np.arange(np.ceil(ylim[0] * 5) / 5, 1.0001, 0.2)])
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    if zero_line:
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2:
            ax.axvspan(x0 - 0.5 - (0.35 if i else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers:
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, CATS[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
    handles = [Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=sub_labels[0]), Patch(fc='#999999', ec='black', label=sub_labels[1])]
    if overall is not None:
        handles.append(plt.Line2D([0], [0], color='black', lw=1.6, label=overall_label))
    ax.legend(handles=handles, loc='upper right', fontsize=8.3, ncol=len(handles), framealpha=0.92, borderpad=0.4, columnspacing=1.2, handlelength=1.6)


def finish(fig, title, foot, top=0.955):
    fig.suptitle(title, x=0.012, ha='left', fontsize=10, y=0.995)
    if foot:
        fig.text(0.012, 0.008, '\n'.join(foot), fontsize=7.2, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0, 0.014 + 0.021 * len(foot), 1, top))


def foots(*extra, composite=False, best=True, interim=False):
    out = []
    if composite:
        out.append(F_COMPOSITE)
    if best:
        out.append(F_BEST)
    if interim:
        out.append(F_INTERIM)
    return out + list(extra)


# ----------------------------------------------------------------------------------------------
# the four figures
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=B)
    ap.add_argument('--esm3', choices=['marginal', 'independent'], default='marginal', help='ESM3-small-open scoring drawn in figures 1 and 2 (the other stays in the tables)')
    a = ap.parse_args()
    fr, added = load()
    sp = fr.spine
    use = sp['use'].to_numpy()
    n_mut = sp['n_mut'].to_numpy()
    single, double = use & (n_mut == 1), use & (n_mut == 2)
    y = sp['y'].to_numpy()
    dd = sp['dddG_ML'].to_numpy()
    libs = sorted(np.unique(sp['group'].to_numpy()[use]))
    rng = np.random.default_rng(5)
    counts = rng.multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=a.B).astype(float)
    best, sc = best_other(fr)
    print(f'masked ESM3 keys found: {added or "none (interim unmasked readout)"}; best other ddG model: {best} ({sc[best]:.3f}; {", ".join(f"{k} {v:.3f}" for k, v in sc.items())})')

    # ---- fig 1: NDCG@96, singles against doubles -----------------------------------------------------
    cols, interim = main_columns(fr, best, a.esm3)
    cells = {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}
    T1 = table_grouped(fr, cols, cells, ndcg96, 1, counts, libs)
    save_tab(T1, 'fig1_ndcg96_singles_doubles')
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    bars(ax, cols, T1, ['singles', 'doubles'], [count_text(T1, 'singles', 'singles'), count_text(T1, 'doubles', 'doubles')], f'NDCG@{K} (mean over test libraries)', (0, 1.12))
    finish(fig, f'Capped Tsuboyama test, NDCG@{K} per library, 95% intervals from resampling libraries (doubles: the libraries with a stabilizing double)',
           foots(composite=True, interim=interim))
    style.save(fig, OUT / 'fig1_ndcg96_singles_doubles')

    # ---- fig 2: Spearman, stabilizing against destabilizing, singles and doubles panels ----------------------
    tabs, tabs_p = [], []
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.6))
    figp, axesp = plt.subplots(2, 1, figsize=(12.5, 8.6))
    for ax, axp, (which, base) in zip(axes, axesp, (('singles', single), ('doubles', double))):
        cells = {'stabilizing': (base & (y > 0), which), 'destabilizing': (base & (y < 0), which)}
        T = table_grouped(fr, cols, cells, rho, MIN_N, counts, libs)
        ov = table_grouped(fr, cols, {'overall': (base, which)}, rho, 3, counts, libs)
        tabs.append(pd.concat([T, ov]).assign(panel=which))
        ylab = "Spearman's ρ (mean over test libraries)"
        bars(ax, cols, T, ['stabilizing', 'destabilizing'], [count_text(T, 'stabilizing', 'stabilizing, ddG > 0'), count_text(T, 'destabilizing', 'destabilizing, ddG < 0')],
             ylab, (-0.2, 1.12), overall=ov, overall_label=f'overall (n = {int(ov.n_variants.max()):,})', tick='label_s' if which == 'singles' else 'label_d',
             headers=which == 'singles', zero_line=True)
        ax.text(0.006, 0.985, which, transform=ax.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')
        # pooled over libraries
        pcells = {'stabilizing': (base & (y > 0), which), 'destabilizing': (base & (y < 0), which), 'overall': (base, which)}
        Tp = table_pooled(fr, cols, pcells, y)
        tabs_p.append(Tp.assign(panel=which))
        pv = Tp[Tp.subset != 'overall']
        bars(axp, cols, pv, ['stabilizing', 'destabilizing'], [count_text(pv, 'stabilizing', 'stabilizing, ddG > 0'), count_text(pv, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ pooled over test libraries", (-0.3, 1.12), overall=Tp[Tp.subset == 'overall'], overall_label=f'overall (n = {int(Tp[Tp.subset == "overall"].n_variants.max()):,})',
             tick='label_s' if which == 'singles' else 'label_d', headers=which == 'singles', zero_line=True)
        axp.text(0.006, 0.985, which, transform=axp.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')
    save_tab(pd.concat(tabs), 'fig2_spearman_stab_destab')
    save_tab(pd.concat(tabs_p), 'fig2_spearman_stab_destab_pooled')
    finish(fig, f'Capped Tsuboyama test, Spearman ρ of stabilizing against destabilizing variants (libraries with at least {MIN_N} in the subset); each panel scores its subset with the model that fits it; 95% intervals from resampling libraries',
           foots(interim=interim))
    style.save(fig, OUT / 'fig2_spearman_stab_destab')
    finish(figp, 'Capped Tsuboyama test, Spearman ρ pooled over libraries, stabilizing against destabilizing variants; 95% intervals from resampling libraries',
           foots(interim=interim))
    style.save(figp, OUT / 'fig2_spearman_stab_destab_pooled')

    # ---- fig 3: doubles, additive version against epistatic version ---------------------------------------------
    cols3, interim3 = pair_columns(fr, best)
    cells = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    T3 = table_grouped(fr, cols3, cells, rho, MIN_N, counts, libs)
    save_tab(T3, 'fig3_spearman_doubles_additive_vs_epistatic')
    d_rows = []
    for c in cols3:
        if c['double'] is None or c['single'] not in fr.preds or c['double'] not in fr.preds:
            continue
        va, ve = per_library(fr, c['single'], double, rho, MIN_N), per_library(fr, c['double'], double, rho, MIN_N)
        both = va.index.intersection(ve.index)
        pt, lo, hi, n = boot_mean((ve - va).reindex(both), libs, counts)
        d_rows.append(dict(column=c['name'], delta=pt, lo=lo, hi=hi, n_libraries=n))
    save_tab(pd.DataFrame(d_rows), 'fig3_spearman_doubles_additive_vs_epistatic_paired_difference')
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    bars(ax, cols3, T3, ['additive', 'epistatic'], [count_text(T3, 'additive', 'additive: singles model summed'), count_text(T3, 'epistatic', 'epistatic: model scores the double')],
         "Spearman's ρ, double mutants", (0, 1.12))
    finish(fig, 'Capped Tsuboyama test doubles, additive version against epistatic version; 95% intervals from resampling libraries', foots(interim=interim3))
    style.save(fig, OUT / 'fig3_spearman_doubles_additive_vs_epistatic')

    # ---- fig 4: naive epistasis rho, positive against negative epistasis ------------------------------------------
    cols4, interim4 = epi_columns(fr, best)
    erows = epi_rows(fr)
    cells = {'positive': (erows & (dd > 0), 'doubles'), 'negative': (erows & (dd < 0), 'doubles')}
    Tg = table_grouped(fr, cols4, cells, rho, MIN_N, counts, libs, y=dd)
    Og = table_grouped(fr, cols4, {'overall': (erows, 'doubles')}, rho, MIN_N, counts, libs, y=dd)
    save_tab(pd.concat([Tg, Og]), 'fig4_naive_epistasis_pos_neg_grouped')
    Tp = table_pooled(fr, cols4, {**cells, 'overall': (erows, 'doubles')}, dd)
    save_tab(Tp, 'fig4_naive_epistasis_pos_neg_pooled')
    for tab, ov, name, yl in ((Tg, Og, 'fig4_naive_epistasis_pos_neg_grouped', 'naive epistasis ρ (mean over libraries)'),
                              (Tp[Tp.subset != 'overall'], Tp[Tp.subset == 'overall'], 'fig4_naive_epistasis_pos_neg_pooled', 'naive epistasis ρ, pooled over libraries')):
        fig, ax = plt.subplots(figsize=(13.5, 5.7))
        bars(ax, cols4, tab, ['positive', 'negative'], [count_text(tab, 'positive', 'positive epistasis, dddG > 0'), count_text(tab, 'negative', 'negative epistasis, dddG < 0')], yl, (-0.6, 1.0),
             overall=ov, overall_label=f'overall (n = {int(ov.n_variants.max()):,})', zero_line=True, tick_fs=7.8)
        ax.set_xlabel('predicted epistasis = prediction for the double − sum of the predictions for its two singles (models named under each column)', fontsize=8.3)
        finish(fig, 'Capped Tsuboyama test doubles with both singles measured: predicted against measured dddG; 95% intervals from resampling libraries', foots(interim=interim4), top=0.95)
        style.save(fig, OUT / name)

    # ---- console summary -------------------------------------------------------------------------------------
    pd.set_option('display.width', 220)
    order = [c['name'] for c in cols]
    print('fig1 NDCG@96\n', T1.pivot(index='column', columns='subset', values='value').reindex(order).round(3).to_string())
    print('fig3\n', T3.pivot(index='column', columns='subset', values='value').round(3).to_string())
    print('fig4 pooled\n', Tp.pivot(index='column', columns='subset', values='value').round(3).to_string())


if __name__ == '__main__':
    main()
