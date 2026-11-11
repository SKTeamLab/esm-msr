#!/usr/bin/env python
"""Publication figures: the notebooks' headline bar charts, recomputed from the collected predictions with protein-level bootstrap intervals.

All figures are on the Homology-Restricted MegaScale **test** domains (33 proteins; measured rows only, all backbones).  Error bars are 95% intervals from resampling
**domains** (proteins) with replacement, the same draw for every model and subset (the notebooks' bars were the spread over three models/seeds).
Multi-seed models are seed means of the per-library value.

Columns are grouped in three labelled categories:
  retrained   ESM-MSR (WT head and combined head, both always shown: the WT head is the cheap one and the focus of the paper) and the baselines retrained on
              Homology-Restricted MegaScale.  A baseline gets one column, scored by the model that fits the subset: MutateEverything stage 1 for singles and stage 2 for doubles,
              ThermoMPNN for singles and ThermoMPNN-D for doubles, SPURS for singles and SPURS multi for doubles ("X | Y" on the axis).
  zero-shot   ESM3-small-open (the sigma = 0 control of the ESM-MSR checkpoint, read unmasked like ESM-MSR: the combined head for every variant; ``--esm3 average`` / ``marginal`` / ``independent`` for the masked scorings, drawn side by side in ``11e_esm3_scoring_variants.py``), ProteinMPNN and ProSST.
  supervised  published ddG predictors that are not trained on Megascale: Rosetta and the better of GeoDDG-3D / DDGemb (mean per-library singles rho).

fig1  NDCG@96 (rectified relevance max(ddG, 0), sklearn ndcg_score as in the notebooks), per library then averaged, singles against doubles.
fig2  Spearman rho over all variants (singles and doubles; a baseline's singles model on the singles, its doubles model on the doubles), stabilizing (measured ddG > 0) against
      destabilizing (< 0) mutations.  fig2b: one panel for singles and one for doubles (each column is the
      model that scores that subset); the overall rho of the panel is the black line.  ``*_pooled``: rho pooled over domains instead of averaged.
fig3  Spearman rho on double mutants, additive version of a model against its epistatic version: the singles model summed over the double's two substitutions
      (ESM-MSR WT head, ME stage 1, ThermoMPNN, SPURS, Rosetta's sum of singles, ESM3-small-open's WT head) against the model that scores the double itself (ESM-MSR
      combined, ME stage 2, ThermoMPNN-D, SPURS multi, Rosetta's direct double, ESM3-small-open's combined head, unmasked).  ProSST is additive by construction (one bar).
fig4  Naive epistasis rho: predicted against measured dddG over doubles with both singles measured, positive (dddG > 0) against negative epistasis.  The predicted
      epistasis is always the double-mutant prediction of one model minus the sum of the single-mutant predictions of another (or another head): written under each column.

The masked ESM3-small-open runs (``cluster/esm3_masked_gamut.sh``, in ``msr_predictions/esm3_small_open_sigma0_masked``) feed the alternative scorings kept in the tables.  Outputs: figures 1 and 2 in ``results/03_tsuboyama``, figures 3 and 4 in ``results/06_epistasis`` (png, pdf, csv, p-value csv).
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

OUT_TSU = out_dir('03_tsuboyama')          # figures 1 and 2: the Tsuboyama test benchmark
OUT_EPI = out_dir('06_epistasis')          # figures 3 to 7: epistasis


class _Out:
    """``OUT / 'fig2_...'``: the figure files go to the theme folder of their figure number; ``results/11_publication`` is left for the figures chosen for the paper."""

    def __truediv__(self, name):
        return (OUT_TSU if str(name).startswith(('fig1', 'fig2', 'best_other')) else OUT_EPI) / name


OUT = _Out()
DS = 'tsuboyama_capped_test'
B = 200                                   # draft sizes; the figures of the paper want --B 2000 --B_mut 1000 --B_pooled 1000
B_MUT = 100
POOLED_B = 200
MIN_N = 10
K = 96

SPLIT = 'Homology-Restricted MegaScale'                            # the capped Tsuboyama split, under its paper name (dataset ids keep 'capped')
CATS = {'retrained': f'Retrained on {SPLIT}', 'zeroshot': 'Zero-shot', 'supervised': 'Supervised, not trained\non Megascale'}
BEST_LABEL = {'geoddg': 'GeoDDG-3D', 'ddgemb': 'DDGemb'}        
BEST_COLOR = {'geoddg': '#117a65', 'ddgemb': '#b7950b'}
F_COMPOSITE = 'MutateEverything(1+2), ThermoMPNN(-D), SPURS(-multi): singles are scored by the first model, doubles by the second (stage 1 / stage 2, ThermoMPNN / ThermoMPNN-D, SPURS / SPURS multi).'
# geometry shared by every bar figure (figures 1-5 sit beside or above each other): one x unit per column, the same bar width and gap, the same overall dash
BAR_W = 0.30
PAIR_OVERALL = (-0.27, 0.05)                                       # bar centres when the overall dash sticks out on the right
PAIR_CENTRED = (-0.16, 0.16)                                       # same width and gap without a dash
DASH = (-0.43, 0.46)                                               # the dash runs behind the bars and sticks out on the right
X_ERR = 0.38                                                       # its interval hangs off the protruding end
XPAD = (0.7, 0.75)                                                 # x margins of the axes, in column units
TICK_FS = 8
LEGEND_KW = dict(loc='upper right', fontsize=8.3, framealpha=0.92, borderpad=0.4, columnspacing=1.2, handlelength=1.6)
IN_PER_UNIT = 12.5 / 11.85                                         # inches per column unit: figure 1 is 12.5 in wide for its 11.85 units (printed by main)


def fig_width_units(cols):
    xs, _ = layout([c for c in cols if c.get('draw', True)])
    return xs[-1] - xs[0] + sum(XPAD)


def fig_width(cols):
    """Figure width that gives every figure figure 1's inches per column."""
    xs, _ = layout([c for c in cols if c.get('draw', True)])
    return IN_PER_UNIT * (xs[-1] - xs[0] + sum(XPAD))


F_BEST_PLAIN = '{m}: the better of GeoDDG-3D and DDGemb by mean per-library singles ρ; trained on S8754 (ProThermDB + ThermoMutDB), not on Megascale.'


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load():
    fr = load_frame(DS)
    p = fr.preds
    added = add_esm3_masked(fr, DS)
    if 'msrrel@0:comb' in p and 'msrrel@0:wt' in p:
        p['msrrel@0:epi'] = p['msrrel@0:comb'] - p['msrrel@0:wt']                   # the combined head's epistasis, no link (sigma = 0 checkpoint): comb(AB) - WT(A) - WT(B), WT(AB) = WT(A) + WT(B)
    # ESM-MSR's epistasis is always the combined head's latent interaction, comb(AB) - WT(AB) = (MT(AB) - WT(AB)) / 2, on the latent scale: no link
    for _spec in ('msr', 'msrdev', 'msrp0'):
        if f'{_spec}:comb' in p and f'{_spec}:wt' in p:
            p[f'{_spec}:epi'] = p[f'{_spec}:comb'] - p[f'{_spec}:wt']
    if 'esm3ind' in p and 'esm3ind+' in p:
        # esm3avg (data.add_esm3_masked): like ProteinMPNN and ESM-MSR's combined head, the mean of the independent-masking WT and mutant passes
        p['esm3:epi'] = p['esm3avg'] - p['esm3ind+']                              # the direct double minus the sum of the singles, as for the other models (= half of MT - WT)
    if 'esm3ind' in p and 'esm3mm' in p:
        p['esm3mm:epi'] = p['esm3ind'] - p['esm3mm']                              # the earlier definition: mutant pass minus the canonical masked marginal of the wild type (tables only)
    for k in ('rosetta', 'geoddg', 'ddgemb', 'pmpnn'):
        if k in p and k + '+' in p:
            p[k + ':epi'] = p[k] - p[k + '+']                                      # direct double minus the sum of the model's singles
    return fr, added


def per_library(fr, key, mask, fn, min_n, y=None):
    """DataFrame (index=library, columns=0..n_seeds-1) of ``fn(pred_seed, y)`` evaluated per seed without ensemble averaging; domains with fewer than ``min_n`` rows in ``mask`` are left out."""
    sp = fr.spine
    y = sp['y'].to_numpy() if y is None else y
    g = sp['group'].to_numpy()
    out = {}
    for lib in np.unique(g[mask]):
        m = mask & (g == lib) & np.isfinite(fr.preds[key]).all(axis=0)
        if m.sum() < min_n:
            continue
        vals = [fn(p[m], y[m]) for p in fr.preds[key]]
        if np.isfinite(vals).any():
            out[lib] = vals
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame.from_dict(out, orient='index', dtype=float)


def _mean_val(values):
    """Grand mean over domains and seeds of a Series or DataFrame."""
    if isinstance(values, pd.DataFrame):
        return float(values.mean(axis=1).mean())
    return float(values.mean())


def _diff_df(va, vb, both=None):
    """Per-domain, per-seed difference (va - vb), broadcasting 1-seed models against multi-seed models."""
    da = va.to_frame() if isinstance(va, pd.Series) else va
    db = vb.to_frame() if isinstance(vb, pd.Series) else vb
    if both is not None:
        da, db = da.reindex(both), db.reindex(both)
    return pd.DataFrame(da.to_numpy(dtype=float) - db.to_numpy(dtype=float), index=da.index)


def rho(p, y):
    return spearmanr(p, y)[0] if len(y) >= 3 and np.ptp(p) > 0 else np.nan


def ndcg96(p, y):
    rel = np.where(y > 0, y, 0.0)
    if not (rel > 0).any():
        return np.nan
    return float(ndcg_score(rel[None, :], p[None, :], k=K, ignore_ties=True))


def boot_reps(values, libs, counts):
    """Replicates of the mean of ``values`` (Series or DataFrame library x seeds) under shared library resamples ``counts`` (B x n_libs) and triplicate seed resamples."""
    df = values.to_frame() if isinstance(values, pd.Series) else values
    v = df.reindex(libs)
    ok = v.notna().any(axis=1).to_numpy()
    c = counts[:, ok]
    V = v.loc[ok].to_numpy(dtype=float)
    n_seeds = V.shape[1]
    fin = np.isfinite(V).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dom_per_seed = np.where((c @ fin) > 0, (c @ np.where(np.isfinite(V), V, 0.0)) / (c @ fin), np.nan)
    if n_seeds == 1:
        return dom_per_seed[:, 0]
    seed_w = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
    return np.nansum(dom_per_seed * seed_w, axis=1)


def boot_mean(values, libs, counts):
    """values: Series or DataFrame (library x seeds).  Returns (mean, lo, hi, n_libraries) integrating domain and seed resamples."""
    df = values.to_frame() if isinstance(values, pd.Series) else values
    v = df.reindex(libs)
    ok = v.notna().any(axis=1)
    lo, hi = ci(boot_reps(values, libs, counts))
    return float(v.loc[ok].mean(axis=1).mean()), lo, hi, int(ok.sum())


def best_other(fr):
    """The supervised, not-Megascale-trained published ddG predictor with the highest mean per-library singles rho."""
    sp = fr.spine
    single = sp['use'].to_numpy() & (sp['n_mut'] == 1).to_numpy()
    sc = {k: _mean_val(per_library(fr, k, single, rho, MIN_N)) for k in BEST_LABEL if k in fr.preds}
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
    """(key, label, alternative) for the ESM3-small-open column; the alternative scoring stays in the tables only (drawn side by side in ``11e_esm3_scoring_variants.py``)."""
    p = fr.preds
    names = {'unmasked': ('msrrel@0:comb', 'ESM3-small-open\nunmasked'), 'average': ('esm3avg', 'ESM3-small-open\nindependent masking\n(WT, MT average)'),
             'marginal': ('esm3mm', 'ESM3-small-open\nmasked marginal'), 'independent': ('esm3ind', 'ESM3-small-open\nindependent masking\n(mutant pass)')}
    have = [s for s in [want] + [k for k in names if k != want] if names[s][0] in p]
    return names[have[0]][0], names[have[0]][1], (names[have[1]] if len(have) > 1 else None)


def main_columns(fr, best, esm3_want):
    ek, el, alt = esm3_choice(fr, esm3_want)
    cols = [col('ESM-MSR (baseline) WT head', 'retrained', '#7f8c8d', 'ESM-MSR\n(baseline)\nWT head', 'msr:wt'),
            col('ESM-MSR (baseline) combined', 'retrained', '#95a5a6', 'ESM-MSR\n(baseline)\ncombined', 'msr:comb'),
            col('ESM-MSR WT head', 'retrained', '#000000', 'ESM-MSR\nWT head', 'msrdev:wt'),
            col('ESM-MSR combined', 'retrained', '#555555', 'ESM-MSR\ncombined', 'msrdev:comb'),
            col('ESM-MSR (p0) WT head', 'retrained', '#6c3483', 'ESM-MSR\n(p0)\nWT head', 'msrp0:wt'),
            col('ESM-MSR (p0) combined', 'retrained', '#9b59b6', 'ESM-MSR\n(p0)\ncombined', 'msrp0:comb'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'MutateEverything\n(1+2)', 'me1', 'me2', label_s='ME\nst1', label_d='ME\nst2'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\n(-D)', 'tmpnn', 'tmpnnd', label_s='ThermoMPNN', label_d='ThermoMPNN-D'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\n(-multi)', 'spurs', 'spursm', label_s='SPURS', label_d='SPURS multi'),
            col('ESM3-small-open', 'zeroshot', '#7f8c8d', el, ek),
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN', 'pmpnn'),
            col('ProSST', 'zeroshot', '#8e44ad', 'ProSST', 'prosst'),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best], best),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta', 'rosetta')]
    if alt:
        cols.append(col('ESM3-small-open (alt)', 'zeroshot', '#7f8c8d', alt[1], alt[0], draw=False))
    n1 = (fr.spine['n_mut'] == 1).to_numpy()
    for c in cols:                         # the model that scores every variant of a library: its singles model on the singles and its doubles model on the doubles
        if c['single'] != c['double']:
            c['all'] = f"{c['single']}|{c['double']}"
            fr.preds[c['all']] = np.where(n1[None, :], fr.preds[c['single']], fr.preds[c['double']])
    return cols


def pair_columns(fr, best):
    """Fig 3: single = the additive version's key, double = the epistatic version's key."""
    p = fr.preds
    es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open\nunmasked\nWT → combined', 'msrrel@0:wt', 'msrrel@0:comb')
    cols = [col('ESM-MSR (baseline)', 'retrained', '#7f8c8d', 'ESM-MSR\n(baseline)\nWT → comb', 'msr:wt', 'msr:comb'),
            col('ESM-MSR', 'retrained', '#000000', 'ESM-MSR\nWT head\n→ combined', 'msrdev:wt', 'msrdev:comb'),
            col('ESM-MSR (p0)', 'retrained', '#6c3483', 'ESM-MSR\n(p0)\nWT → comb', 'msrp0:wt', 'msrp0:comb'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'ME\nst1\n→ st2', 'me1', 'me2'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\n→ ThermoMPNN-D', 'tmpnn', 'tmpnnd'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\n→ SPURS multi', 'spurs', 'spursm'),
            es,
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN\nsum of singles\n→ direct double', 'pmpnn+', 'pmpnn'),
            col('ProSST', 'zeroshot', '#8e44ad', 'ProSST\n(additive)', 'prosst', None),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best] + '\nsum of singles\n→ direct double', best + '+', best),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta\nsum of singles\n→ direct double', 'rosetta+', 'rosetta')]
    return cols


def epi_columns(fr, best):
    p = fr.preds
    es = col('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open\ncombined − WT\n(unmasked)', None, 'msrrel@0:epi')
    extra = [col('ESM3-small-open (independent masking)', 'zeroshot', '#7f8c8d', 'ESM3-small-open', None, 'esm3:epi', draw=False)] if 'esm3:epi' in p else []
    if 'esm3mm:epi' in p:
        extra.append(col('ESM3-small-open (MT independent − WT marginal)', 'zeroshot', '#7f8c8d', 'ESM3-small-open', None, 'esm3mm:epi', draw=False))
    cols = [col('ESM-MSR (baseline)', 'retrained', '#7f8c8d', 'ESM-MSR\n(baseline)\ncomb − WT', None, 'msr:epi'),
            col('ESM-MSR', 'retrained', '#000000', 'ESM-MSR\ncomb − WT', None, 'msrdev:epi'),
            col('ESM-MSR (p0)', 'retrained', '#6c3483', 'ESM-MSR\n(p0)\ncomb − WT', None, 'msrp0:epi'),
            col('MutateEverything', 'retrained', '#1b8a4b', 'ME\nst2 − st1', None, 'me2:epi'),
            col('ThermoMPNN', 'retrained', '#1f5f8b', 'ThermoMPNN\n-D − single', None, 'tmpnnd:epi'),
            col('SPURS', 'retrained', '#d35400', 'SPURS\nmulti − single', None, 'spursm:epi'),
            es,
            col('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN\ndirect −\nsum of singles', None, 'pmpnn:epi'),
            col('best other ddG', 'supervised', BEST_COLOR[best], BEST_LABEL[best] + '\ndirect −\nsum of singles', None, best + ':epi'),
            col('Rosetta', 'supervised', '#1abc9c', 'Rosetta\ndirect −\nsum of singles', None, 'rosetta:epi')] + extra
    return cols


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
def key_for(c, which):
    if which == 'all':
        return c.get('all', c['single'])
    return c['single'] if which == 'singles' else c['double']


def table_grouped(fr, cols, cells, fn, min_n, counts, libs, y=None, kind='domain', B_mut=1000, seed=11, vals=None):
    """Rows (column, subset): mean over domains of ``fn``, with the library bootstrap.  ``cells``: subset -> (row mask, 'singles' | 'doubles' = which key of the column).
    ``kind='mutation'`` draws the interval by resampling the rows of every domain with replacement instead (``fn`` must be Spearman's rho); the bar is the same.
    ``vals``, if given, collects {(column, subset): (key, per-library values, row mask, label vector)} for the p-value tables."""
    g = fr.spine['group'].to_numpy()
    yy = fr.spine['y'].to_numpy() if y is None else y
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
            if vals is not None:
                vals[(c['name'], sub)] = (key, v, m, yy)
            if kind == 'mutation':
                m2 = m & np.isin(g, v.index) & np.isfinite(yy) & np.isfinite(fr.preds[key]).all(axis=0)
                point, reps = bootstrap_rho({DS: subframe(fr, m2, y=yy, keys=[key])}, [key], kind='mutation', B=B_mut, seed=seed, per_protein=True)
                if abs(point[DS][key][0] - pt) > 1e-4:
                    print(f'  warning: {c["name"]} / {sub}: bootstrap point {point[DS][key][0]:.4f} differs from the per-library mean {pt:.4f}')
                lo, hi = ci(reps[DS][key])
            rows.append(dict(column=c['name'], key=key, subset=sub, value=pt, lo=lo, hi=hi, n_libraries=n, n_variants=int((m & np.isin(g, v.index)).sum())))
    return pd.DataFrame(rows)


def table_pooled(fr, cols, cells, y, B_=None, seed=7, kind='domain', reps_out=None):
    """Rho pooled over domains, with the library-level bootstrap (``kind='mutation'``: rows resampled instead), for the same (column, subset) cells."""
    B_ = POOLED_B if B_ is None else B_
    rows = []
    for sub, (m, which) in cells.items():
        names = {key_for(c, which): c['name'] for c in cols if key_for(c, which) in fr.preds}
        keys = list(names)
        sf = subframe(fr, m & np.isfinite(y), y=y, keys=keys)
        point, reps = bootstrap_rho({DS: sf}, keys, kind='protein' if kind == 'domain' else 'mutation', B=B_, seed=seed, per_protein=False)
        for k, (r, per_seed, n, ng) in point[DS].items():
            lo, hi = ci(reps[DS][k])
            if reps_out is not None:
                reps_out[(sub, names[k])] = reps[DS][k]
            rows.append(dict(column=names[k], key=k, subset=sub, value=r, lo=lo, hi=hi, n_variants=n, n_libraries=ng))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# p-value tables: every ESM-MSR model against every other model, on every comparison a figure draws
# ----------------------------------------------------------------------------------------------
NAMES = {}                                                              # column name -> name in the p-value tables (filled in main: the best other ddG model, the alternative ESM3 scoring)


def is_msr(name):
    return name.startswith('ESM-MSR')


def p_two_sided(d):
    """Two-sided bootstrap p of 'the difference is 0' from replicates ``d``: twice the smaller tail, never below 1 / (B + 1)."""
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (len(d) + 1)))


def add_holm(df, by=('comparison', 'model_a')):
    """Holm-adjusted p within each family (the comparisons of one ESM-MSR model on one subset, and one panel)."""
    by = [b for b in by + ('panel',) if b in df]
    df = df.reset_index(drop=True)
    df['p_holm'] = np.nan
    for _, idx in df.groupby(by).groups.items():
        p = df.loc[idx, 'p'].sort_values()
        adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p.to_numpy())
        df.loc[p.index, 'p_holm'] = np.maximum.accumulate(adj)
    return df


def pair_names(names):
    """(a, b) for every ESM-MSR column a against every other column b of one subset (a pair of two ESM-MSR columns once)."""
    for a in names:
        if is_msr(a):
            for b in names:
                if b != a and not (is_msr(b) and names.index(b) < names.index(a)):
                    yield a, b


def row_of(sub, a, b, va, vb, d, reps, n, bootstrap, what):
    lo, hi = ci(reps)
    return dict(comparison=sub, model_a=NAMES.get(a, a), model_b=NAMES.get(b, b), value_a=va, value_b=vb, delta=d, lo=lo, hi=hi, p=p_two_sided(reps), n_libraries=n, bootstrap=bootstrap, statistic=what)


def compare_grouped(fr, vals, subsets, counts, libs, kind='domain', B_mut=1000, seed=13, what='mean over domains'):
    """Paired bootstrap of (model a - model b) for the per-domain-mean bars.  Domain level: the two models' per-domain values on the domains both have, under the
    shared domain and seed resamples of the figure.  Mutation level: the rows of each domain are resampled, the same draw for both models (Spearman's rho only)."""
    rows = []
    for sub in subsets:
        names = [n for (n, s_) in vals if s_ == sub]
        for a, b in pair_names(names):
            ka, va, ma, ya = vals[(a, sub)]
            kb, vb, mb, yb = vals[(b, sub)]
            both = va.index.intersection(vb.index)
            if len(both) < 2:
                continue
            d = _diff_df(va, vb, both)
            if kind == 'domain':
                reps = boot_reps(d, libs, counts)
            else:
                g = fr.spine['group'].to_numpy()
                m2 = ma & np.isin(g, both) & np.isfinite(ya) & np.isfinite(fr.preds[ka]).all(axis=0) & np.isfinite(fr.preds[kb]).all(axis=0)
                _, r = bootstrap_rho({DS: subframe(fr, m2, y=ya, keys=[ka, kb])}, [ka, kb], kind='mutation', B=B_mut, seed=seed, per_protein=True)
                reps = r[DS][ka] - r[DS][kb]
            rows.append(row_of(sub, a, b, _mean_val(va.loc[both]), _mean_val(vb.loc[both]), _mean_val(d), reps, len(both), 'domains' if kind == 'domain' else 'mutations', what))
    return rows


def compare_pooled(T, reps, subsets, kind='domain'):
    """The same for the pooled bars: the replicates of every model come from one shared draw, so the difference of two replicate vectors is paired."""
    rows = []
    for sub in subsets:
        t = T[T.subset == sub].set_index('column')
        for a, b in pair_names(list(t.index)):
            rows.append(row_of(sub, a, b, t.value[a], t.value[b], t.value[a] - t.value[b], reps[(sub, a)] - reps[(sub, b)], int(min(t.n_libraries[a], t.n_libraries[b])),
                               'domains' if kind == 'domain' else 'mutations', 'pooled over domains'))
    return rows


def save_p(rows, name, **extra):
    df = add_holm(pd.DataFrame(rows).assign(**extra))
    df.to_csv(OUT / f'{name}_pvalues.csv', index=False, float_format='%.4g')
    return df


def save_tab(T, name):
    T.to_csv(OUT / f'{name}.csv', index=False, float_format='%.4f')


def count_text(T, sub, text):
    """'text (n = 1,234 variants, 33 domains)' from the table's rows of one subset."""
    r = T[T.subset == sub]
    return f'{text} (n = {int(r.n_variants.max()):,}, {int(r.n_libraries.max())} domains)'


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


def bars(ax, cols, T, subsets, sub_labels, ylabel, ylim, overall=None, overall_label='overall', tick='label', headers=True, annotate=True, zero_line=False, tick_fs=TICK_FS, cats=CATS):
    """Two bars per column (first subset hatched) with 95% intervals; the overall value is a black dash behind the pair that sticks out to its right, with its interval
    on the protruding end (one side of the dash, not the middle) and its value on top; three labelled categories."""
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = layout(cols)
    w, dash, x_err = BAR_W, DASH, X_ERR
    pair = PAIR_CENTRED if overall is None else PAIR_OVERALL           # x offsets of the two bars
    top = 0.012 * (ylim[1] - ylim[0])
    for c, x in zip(cols, xs):
        have = [s for s in subsets if len(T[(T.column == c['name']) & (T.subset == s)])]
        for j, sub in enumerate(subsets):
            r = T[(T.column == c['name']) & (T.subset == sub)]
            if not len(r):
                continue
            r = r.iloc[0]
            xx = x + pair[j] if len(have) == 2 else x
            ax.bar(xx, r.value, w, color=c['color'], alpha=0.95 if j else 0.55, edgecolor=c['color'] if j == 0 else 'black', hatch='////' if j == 0 else None, lw=0.8, zorder=3)
            ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]], fmt='none', ecolor='#c0392b', elinewidth=1.3, capsize=2.5, zorder=5)
            if annotate:
                ax.text(xx, max(r.hi, r.value, 0) + top, f'{r.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8,
                        bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))      # a label that runs into a bar or a line stays readable
        if overall is not None:
            o = overall[overall.column == c['name']]
            if len(o):
                o = o.iloc[0]
                ax.plot([x + dash[0], x + dash[1]], [o.value] * 2, color='black', lw=1.8, zorder=2.2, solid_capstyle='butt')
                ax.errorbar(x + x_err, o.value, yerr=[[max(o.value - o.lo, 0)], [max(o.hi - o.value, 0)]], fmt='none', ecolor='black', elinewidth=1.3, capsize=2.5, zorder=5)
                if annotate:
                    ax.text(x + x_err, max(o.hi, o.value, 0) + top, f'{o.value:.2f}', ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, fontweight='bold',
                            bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
    ax.set_xticks(xs)
    ax.set_xticklabels([c.get('label_' + tick[-1], c['label']) if tick in ('label_s', 'label_d') else c['label'] for c in cols], fontsize=tick_fs)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in np.arange(np.ceil(ylim[0] * 5) / 5, 1.0001, 0.2)])
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    if zero_line:
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
    ax.set_xlim(xs[0] - XPAD[0], xs[-1] + XPAD[1])
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2:
            ax.axvspan(x0 - 0.5 - (0.35 if i else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers:
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, cats[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
    handles = [Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=sub_labels[0]), Patch(fc='#999999', ec='black', label=sub_labels[1])]
    if overall is not None:
        handles.append(plt.Line2D([0], [0], color='black', lw=1.8, label=overall_label))
    ax.legend(handles=handles, ncol=len(handles), **LEGEND_KW)


def finish(fig, title, foot, top=0.955):
    fig.suptitle(title, x=0.012, ha='left', fontsize=10, y=0.995)
    if foot:
        fig.text(0.012, 0.008, '\n'.join(foot), fontsize=7.2, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0, 0.014 + 0.021 * len(foot), 1, top))


def foots(*extra, best, composite=False):
    out = []
    if composite:
        out.append(F_COMPOSITE)
    return out + list(extra)


# ----------------------------------------------------------------------------------------------
# fig 2 and 2b, with the interval from resampling domains or mutations
# ----------------------------------------------------------------------------------------------
def fig2(fr, cols, best, use, single, double, y, counts, libs, kind, B_mut):
    """Every figure of the fig 2 family; ``kind='mutation'`` redraws them with intervals from resampling the mutations of each domain (the bars are identical), files ``*_mutboot``."""
    tag = '' if kind == 'domain' else '_mutboot'
    unit = 'resampling domains' if kind == 'domain' else 'resampling mutations within each domain'
    unit_p = 'resampling domains' if kind == 'domain' else 'resampling mutations'
    kw = dict(kind=kind, B_mut=B_mut)
    foot = lambda **k: foots(best=best, **k)
    split = f'{SPLIT} test'
    nlab = lambda ov: f'overall (n = {int(ov.n_variants.max()):,})'
    w_fig = fig_width(cols)

    cells = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all')}
    v2 = {}
    T2 = table_grouped(fr, cols, cells, rho, MIN_N, counts, libs, vals=v2, **kw)
    ov2 = table_grouped(fr, cols, {'overall': (use, 'all')}, rho, 3, counts, libs, vals=v2, **kw)
    save_tab(pd.concat([T2, ov2]), f'fig2_spearman_stab_destab{tag}')
    save_p(compare_grouped(fr, v2, ['stabilizing', 'destabilizing', 'overall'], counts, libs, **kw), f'fig2_spearman_stab_destab{tag}')
    fig, ax = plt.subplots(figsize=(w_fig, 4.9))
    bars(ax, cols, T2, ['stabilizing', 'destabilizing'], [count_text(T2, 'stabilizing', 'stabilizing, ddG > 0'), count_text(T2, 'destabilizing', 'destabilizing, ddG < 0')],
         "Spearman's ρ (mean over test domains)", (-0.2, 1.12), overall=ov2, overall_label=nlab(ov2), zero_line=True, cats=CATS)
    finish(fig, f'{split}, all variants (singles and doubles): Spearman ρ of stabilizing against destabilizing variants (domains with at least {MIN_N} in the subset); '
                f'95% intervals from {unit}', foot(composite=True))
    style.save(fig, OUT / f'fig2_spearman_stab_destab{tag}')

    pc = {'stabilizing': (use & (y > 0), 'all'), 'destabilizing': (use & (y < 0), 'all'), 'overall': (use, 'all')}
    rp2 = {}
    Tp2 = table_pooled(fr, cols, pc, y, kind=kind, reps_out=rp2)
    save_tab(Tp2, f'fig2_spearman_stab_destab_pooled{tag}')
    save_p(compare_pooled(Tp2, rp2, ['stabilizing', 'destabilizing', 'overall'], kind), f'fig2_spearman_stab_destab_pooled{tag}')
    fig, ax = plt.subplots(figsize=(w_fig, 4.9))
    pv2, po2 = Tp2[Tp2.subset != 'overall'], Tp2[Tp2.subset == 'overall']
    bars(ax, cols, pv2, ['stabilizing', 'destabilizing'], [count_text(pv2, 'stabilizing', 'stabilizing, ddG > 0'), count_text(pv2, 'destabilizing', 'destabilizing, ddG < 0')],
         "Spearman's ρ pooled over test domains", (-0.3, 1.12), overall=po2, overall_label=nlab(po2), zero_line=True, cats=CATS)
    finish(fig, f'{split}, all variants (singles and doubles): Spearman ρ pooled over domains, stabilizing against destabilizing; 95% intervals from {unit_p}',
           foot(composite=True))
    style.save(fig, OUT / f'fig2_spearman_stab_destab_pooled{tag}')

    # fig 2b: the same for singles and for doubles, each panel with the model that fits it
    tabs, tabs_p, ps, ps_p = [], [], [], []
    fig, axes = plt.subplots(2, 1, figsize=(w_fig, 8.6))
    figp, axesp = plt.subplots(2, 1, figsize=(w_fig, 8.6))
    for ax, axp, (which, base) in zip(axes, axesp, (('singles', single), ('doubles', double))):
        tk = 'label_s' if which == 'singles' else 'label_d'
        cells = {'stabilizing': (base & (y > 0), which), 'destabilizing': (base & (y < 0), which)}
        vb = {}
        T = table_grouped(fr, cols, cells, rho, MIN_N, counts, libs, vals=vb, **kw)
        ov = table_grouped(fr, cols, {'overall': (base, which)}, rho, 3, counts, libs, vals=vb, **kw)
        tabs.append(pd.concat([T, ov]).assign(panel=which))
        ps.append(pd.DataFrame(compare_grouped(fr, vb, ['stabilizing', 'destabilizing', 'overall'], counts, libs, **kw)).assign(panel=which))
        bars(ax, cols, T, ['stabilizing', 'destabilizing'], [count_text(T, 'stabilizing', 'stabilizing, ddG > 0'), count_text(T, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ (mean over test domains)", (-0.2, 1.12), overall=ov, overall_label=nlab(ov), tick=tk, headers=which == 'singles', zero_line=True, cats=CATS)
        ax.text(0.006, 0.985, which, transform=ax.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')
        pcells = {'stabilizing': (base & (y > 0), which), 'destabilizing': (base & (y < 0), which), 'overall': (base, which)}
        rpb = {}
        Tp = table_pooled(fr, cols, pcells, y, kind=kind, reps_out=rpb)
        tabs_p.append(Tp.assign(panel=which))
        ps_p.append(pd.DataFrame(compare_pooled(Tp, rpb, ['stabilizing', 'destabilizing', 'overall'], kind)).assign(panel=which))
        pv, po = Tp[Tp.subset != 'overall'], Tp[Tp.subset == 'overall']
        bars(axp, cols, pv, ['stabilizing', 'destabilizing'], [count_text(pv, 'stabilizing', 'stabilizing, ddG > 0'), count_text(pv, 'destabilizing', 'destabilizing, ddG < 0')],
             "Spearman's ρ pooled over test domains", (-0.3, 1.12), overall=po, overall_label=nlab(po), tick=tk, headers=which == 'singles', zero_line=True, cats=CATS)
        axp.text(0.006, 0.985, which, transform=axp.transAxes, va='top', ha='left', fontsize=9.5, fontweight='bold')
    save_tab(pd.concat(tabs), f'fig2b_spearman_stab_destab_singles_doubles{tag}')
    save_tab(pd.concat(tabs_p), f'fig2b_spearman_stab_destab_singles_doubles_pooled{tag}')
    save_p(pd.concat(ps), f'fig2b_spearman_stab_destab_singles_doubles{tag}')
    save_p(pd.concat(ps_p), f'fig2b_spearman_stab_destab_singles_doubles_pooled{tag}')
    finish(fig, f'{split}, Spearman ρ of stabilizing against destabilizing variants (domains with at least {MIN_N} in the subset); each panel scores its subset with the model that fits it; '
                f'95% intervals from {unit}', foot())
    style.save(fig, OUT / f'fig2b_spearman_stab_destab_singles_doubles{tag}')
    finish(figp, f'{split}, Spearman ρ pooled over domains, stabilizing against destabilizing variants; 95% intervals from {unit_p}', foot())
    style.save(figp, OUT / f'fig2b_spearman_stab_destab_singles_doubles_pooled{tag}')


# ----------------------------------------------------------------------------------------------
# the figures
# ----------------------------------------------------------------------------------------------
def main():
    global POOLED_B
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=B)
    ap.add_argument('--boot', choices=['domain', 'mutation', 'both'], default='both', help='fig 2 family: intervals from resampling domains, mutations (files *_mutboot), or both')
    ap.add_argument('--B_mut', type=int, default=B_MUT, help='replicates of the mutation-level bootstrap')
    ap.add_argument('--B_pooled', type=int, default=POOLED_B, help='replicates of the bootstrap of the pooled rho')
    ap.add_argument('--esm3', choices=['unmasked', 'average', 'marginal', 'independent'], default='unmasked', help='ESM3-small-open scoring drawn in figures 1 and 2: the unmasked sigma = 0 readout (combined head), the mean of the independent-masking WT and mutant passes, the canonical masked marginal, or the independent mutant pass (the next one stays in the tables)')
    a = ap.parse_args()
    POOLED_B = a.B_pooled
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
    NAMES['best other ddG'] = BEST_LABEL[best]
    print(f'masked ESM3 keys found (tables and 11e only): {added or "none"}; best other ddG model: {best} ({sc[best]:.3f}; {", ".join(f"{k} {v:.3f}" for k, v in sc.items())})')

    # ---- fig 1: NDCG@96, singles against doubles -----------------------------------------------------
    cols = main_columns(fr, best, a.esm3)
    print(f'figure 1: {fig_width_units(cols):.2f} column units in 12.5 in')
    NAMES.update({c['name']: c['label'].replace('\n', ' ') for c in cols if c['name'].endswith('(alt)')})
    cells = {'singles': (single, 'singles'), 'doubles': (double, 'doubles')}
    v1 = {}
    T1 = table_grouped(fr, cols, cells, ndcg96, 1, counts, libs, vals=v1)
    # the overall NDCG@96 of a library ranks its singles and doubles together; it is drawn as the black line unless it sits on the singles bar
    # (singles are 73% of the rows, and the top 96 of a library are mostly singles)
    ov1 = table_grouped(fr, cols, {'overall': (use, 'all')}, ndcg96, 1, counts, libs, vals=v1)
    save_tab(pd.concat([T1, ov1]), 'fig1_ndcg96_singles_doubles')
    save_p(compare_grouped(fr, v1, ['singles', 'doubles', 'overall'], counts, libs), 'fig1_ndcg96_singles_doubles')
    d1 = (ov1.set_index('column').value - T1[T1.subset == 'singles'].set_index('column').value).dropna()
    d1 = d1[[c['name'] for c in cols if c.get('draw', True) and c['name'] in d1.index]]
    close = int((d1.abs() <= 0.01).sum())
    draw_overall = close < 0.9 * len(d1)
    print(f'fig1 overall NDCG@96 against singles: {close} of {len(d1)} columns within 0.01 (largest difference {d1.abs().max():.3f}); '
          f'{"drawn as the overall line" if draw_overall else "not drawn, footnote instead"}')
    note1 = ([f'Overall = NDCG@96 of a library with its singles and doubles ranked together (singles are 73% of the rows); it differs from the singles bar by {d1.min():+.2f} to {d1.max():+.2f}.']
             if draw_overall else [])
    fig, ax = plt.subplots(figsize=(fig_width(cols), 4.9))
    bars(ax, cols, T1, ['singles', 'doubles'], [count_text(T1, 'singles', 'singles'), count_text(T1, 'doubles', 'doubles')], f'NDCG@{K} (mean over test domains)', (0, 1.12),
         overall=ov1 if draw_overall else None, overall_label=f'overall (n = {int(ov1.n_variants.max()):,})')
    finish(fig, f'{SPLIT} test, NDCG@{K} per library, 95% intervals from resampling domains (doubles: the domains with a stabilizing double)',
           foots(*note1, best=best, composite=True))
    style.save(fig, OUT / 'fig1_ndcg96_singles_doubles')

    # ---- fig 2: Spearman over all variants, stabilizing against destabilizing (+ fig 2b) ----------------------------
    for kind in ('domain', 'mutation') if a.boot == 'both' else (a.boot,):
        fig2(fr, cols, best, use, single, double, y, counts, libs, kind, a.B_mut)

    # ---- fig 3: doubles, additive version against epistatic version ---------------------------------------------
    cols3 = pair_columns(fr, best)
    cells = {'additive': (double, 'singles'), 'epistatic': (double, 'doubles')}
    v3 = {}
    T3 = table_grouped(fr, cols3, cells, rho, MIN_N, counts, libs, vals=v3)
    save_tab(T3, 'fig3_spearman_doubles_additive_vs_epistatic')
    d_rows = []
    for c in cols3:
        if c['double'] is None or c['single'] not in fr.preds or c['double'] not in fr.preds:
            continue
        va, ve = per_library(fr, c['single'], double, rho, MIN_N), per_library(fr, c['double'], double, rho, MIN_N)
        both = va.index.intersection(ve.index)
        pt, lo, hi, n = boot_mean(_diff_df(ve, va, both), libs, counts)
        d_rows.append(dict(column=c['name'], delta=pt, lo=lo, hi=hi, n_libraries=n))
    save_tab(pd.DataFrame(d_rows), 'fig3_spearman_doubles_additive_vs_epistatic_paired_difference')
    # p-values: ESM-MSR's additive version against the other models' additive versions, epistatic against epistatic, and the gain (epistatic - additive) against the other gains
    gain = 'gain (epistatic - additive)'
    p3 = []
    for c in cols3:
        if (c['name'], 'additive') in v3 and (c['name'], 'epistatic') in v3:
            ka, va, ma, ya = v3[(c['name'], 'additive')]
            ke, ve, me, ye = v3[(c['name'], 'epistatic')]
            both = va.index.intersection(ve.index)
            d = _diff_df(ve, va, both)
            v3[(c['name'], gain)] = (None, d, None, None)
            p3.append(row_of('own gain', f"{NAMES.get(c['name'], c['name'])} (epistatic)", f"{NAMES.get(c['name'], c['name'])} (additive)", _mean_val(ve.loc[both]), _mean_val(va.loc[both]), _mean_val(d), boot_reps(d, libs, counts), len(both), 'domains', 'mean over domains'))
    p3 = compare_grouped(fr, v3, ['additive', 'epistatic', gain], counts, libs) + p3
    save_p(p3, 'fig3_spearman_doubles_additive_vs_epistatic')
    fig, ax = plt.subplots(figsize=(fig_width(cols3), 4.9))
    bars(ax, cols3, T3, ['additive', 'epistatic'], [count_text(T3, 'additive', 'additive: singles model summed'), count_text(T3, 'epistatic', 'epistatic: model scores the double')],
         "Spearman's ρ, double mutants", (0, 1.12))
    finish(fig, f'{SPLIT} test doubles, additive version against epistatic version; 95% intervals from resampling domains', foots(best=best))
    style.save(fig, OUT / 'fig3_spearman_doubles_additive_vs_epistatic')

    # ---- fig 4: naive epistasis rho, positive against negative epistasis ------------------------------------------
    cols4 = epi_columns(fr, best)
    erows = epi_rows(fr)
    cells = {'positive': (erows & (dd > 0), 'doubles'), 'negative': (erows & (dd < 0), 'doubles')}
    v4 = {}
    Tg = table_grouped(fr, cols4, cells, rho, MIN_N, counts, libs, y=dd, vals=v4)
    Og = table_grouped(fr, cols4, {'overall': (erows, 'doubles')}, rho, MIN_N, counts, libs, y=dd, vals=v4)
    save_tab(pd.concat([Tg, Og]), 'fig4_naive_epistasis_pos_neg_grouped')
    save_p(compare_grouped(fr, v4, ['positive', 'negative', 'overall'], counts, libs), 'fig4_naive_epistasis_pos_neg_grouped')
    rp4 = {}
    Tp = table_pooled(fr, cols4, {**cells, 'overall': (erows, 'doubles')}, dd, reps_out=rp4)
    save_tab(Tp, 'fig4_naive_epistasis_pos_neg_pooled')
    save_p(compare_pooled(Tp, rp4, ['positive', 'negative', 'overall']), 'fig4_naive_epistasis_pos_neg_pooled')
    for tab, ov, name, yl in ((Tg, Og, 'fig4_naive_epistasis_pos_neg_grouped', 'naive epistasis ρ (mean over domains)'),
                              (Tp[Tp.subset != 'overall'], Tp[Tp.subset == 'overall'], 'fig4_naive_epistasis_pos_neg_pooled', 'naive epistasis ρ, pooled over domains')):
        drawn = [c['name'] for c in cols4 if c.get('draw', True)]
        lo_min = min(tab[tab.column.isin(drawn)].lo.min(), ov[ov.column.isin(drawn)].lo.min())
        ylo = np.floor((lo_min - 0.02) / 0.05) * 0.05                                         # just below the lowest interval: no white space under the bars
        fig, ax = plt.subplots(figsize=(fig_width(cols4), 5.15 - 2.0 * (1.6 - (1.0 - ylo))))      # the height follows the range, the horizontal scale stays that of the other figures
        bars(ax, cols4, tab, ['positive', 'negative'], [count_text(tab, 'positive', 'positive epistasis, dddG > 0'), count_text(tab, 'negative', 'negative epistasis, dddG < 0')], yl, (ylo, 1.0),
             overall=ov, overall_label=f'overall (n = {int(ov.n_variants.max()):,})', zero_line=True)
        ax.set_xlabel('predicted epistasis = prediction for the double − the additive reading of its two singles (as named under each column)', fontsize=8.3)
        finish(fig, f'{SPLIT} test doubles with both singles measured: predicted against measured dddG; 95% intervals from resampling domains', foots(best=best), top=0.95)
        style.save(fig, OUT / name)

    # ---- console summary -------------------------------------------------------------------------------------
    pd.set_option('display.width', 220)
    order = [c['name'] for c in cols]
    print('fig1 NDCG@96\n', T1.pivot(index='column', columns='subset', values='value').reindex(order).round(3).to_string())
    print('fig3\n', T3.pivot(index='column', columns='subset', values='value').round(3).to_string())
    print('fig4 pooled\n', Tp.pivot(index='column', columns='subset', values='value').round(3).to_string())


if __name__ == '__main__':
    main()
