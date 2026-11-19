#!/usr/bin/env python
"""MGnify: the Tsuboyama analyses (03_tsuboyama) repeated on a stability set ESM-MSR never saw.  ESM-MSR only (FINAL_p0 seeds 1-3; the other models were not run on MGnify).

MGnify = 766,790 variants of 180,367 scaffolds (60-80 residues; 616,655 singles, 150,127 doubles, 8 triples) with Megascale-style ddG labels (``data/mgnify``).  Rows: the measured ones (cens == 0,
dG of the wild type and of the variant inside the assay's range), 705,864 variants of 167,379 scaffolds.  Every statistic is given for the **latent** ddG of the model and for the **Tobit-linked** one
(``h(dG_wt + latent) - dG_wt``, the link of ``esm_msr.val_stats.tobit_link`` with the constants the checkpoints were fit with), for the WT head, the MT head and the combined prediction.

**Rocklin designs** (the same table, ``source == rocklin_denovo``: 1,000 de novo design scaffolds, 209,818 variants of one, two and three edits, 149,119 measured) are scored the same way
(``final_mgnify/FINAL_p0/s<seed>/rdenovo``, three seeds) and written to the same folder: dataset ``Rocklin designs`` in ``mgnify_summary.csv`` / ``mgnify_rmse.csv``, ``table_rocklin*.md``, figure ``rocklin_overview``.
They have no homology classes, and a mutation-count subset ``3 mutations`` (153,056 of the variants).  The designs' own within-design stats (``scoring/mgnify_rdenovo_stats.py``) are in ``rocklin_designs/``.

Subsets (the Tsuboyama analyses' cells):  overall; singles; doubles (the 8 triples are in overall only); dG > 0 (the variant's measured dG_ML above 0: the variants the assay saw folded), also dG <= 0;
stabilizing (measured ddG > 0) and destabilizing (ddG < 0), as in 11_publication_figures fig2.  Strata by the scaffold's homology to the capped training set (sequence evidence, MMseqs2): all, clean (none + low),
h, strong (H + same protein).

Units (as in 03_tsuboyama and 04_dms_domainome):
  pooled        one Spearman rho over all rows of the subset; 95% interval from resampling whole scaffolds (they share a backbone and a homology class) and the three seeds
  per scaffold  the mean over scaffolds of the Spearman rho of that scaffold's variants, scaffolds with at least ``MIN_VARIANTS`` variants in the subset; the interval resamples scaffolds and seeds.  The link is monotone
                and ``dG_wt`` is constant within a scaffold, so the per-scaffold rho of the latent and the linked reading are the same number; the link only moves pooled ranks (and the scale).

Reference rows: ESM-MSR on the Tsuboyama capped test libraries (33), the same statistics, per library and pooled.

  mgnify_summary.csv       every cell: stratum, subset, head, form (latent / linked), unit, rho, lo, hi, n rows, n scaffolds, per-seed rhos; ``delta_link`` rows: linked - latent with its paired interval and p
  mgnify_rmse.csv          pooled RMSE, MAE and Pearson r of the latent and the linked prediction against measured ddG (all measured rows and by subset)
  mgnify_overview          A: pooled rho of every subset, latent against linked, MGnify against Tsuboyama; B: per-scaffold mean rho
  mgnify_homology          the same by homology stratum
  mgnify_per_scaffold      per-scaffold rho of the combined prediction against the WT head, one dot per scaffold; distribution by head
  mgnify_scatter           predicted against measured ddG, latent and linked, singles and doubles

    python analysis/18_mgnify.py [--B 100] [--seeds 1 2 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci, paired_diff  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402
from msr_analysis.final_runs import load_final_frame  # noqa: E402
from msr_analysis.mgnify import HEADS, STRATA, TOBIT, finished_seeds, load_mgnify  # noqa: E402
from msr_analysis.registry import TASKS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402

OUT = out_dir('18_mgnify')
MIN_VARIANTS = 10
HEAD_LABEL = {'wt': 'WT head', 'mt': 'MT head', 'comb': 'combined'}
FORMS = [('latent', ''), ('linked', '~')]
SUBSETS = {                       # name -> mask on a spine (rows already measured)
    'overall': lambda s: np.ones(len(s), bool),
    'singles': lambda s: (s['n_mut'] == 1).to_numpy(),
    'doubles': lambda s: (s['n_mut'] == 2).to_numpy(),
    '3 mutations': lambda s: (s['n_mut'] == 3).to_numpy(),
    'dG > 0': lambda s: (s['dG_ML'] > 0).to_numpy(),
    'dG <= 0': lambda s: (s['dG_ML'] <= 0).to_numpy(),
    'stabilizing (ddG > 0)': lambda s: (s['y'] > 0).to_numpy(),
    'destabilizing (ddG < 0)': lambda s: (s['y'] < 0).to_numpy(),
}
MAIN_SUBSETS = ['overall', 'singles', 'doubles', 'dG > 0', 'stabilizing (ddG > 0)', 'destabilizing (ddG < 0)']
ROCKLIN_SUBSETS = ['overall', 'singles', 'doubles', '3 mutations', 'dG > 0', 'stabilizing (ddG > 0)', 'destabilizing (ddG < 0)']
DATASET = {'mgnify': 'MGnify', 'tsuboyama': 'Tsuboyama capped test', 'rocklin': 'Rocklin designs'}
COLORS = {'latent': '#2a78d6', 'linked': '#eb6834'}
TSU_COLOR = '#777777'


def sub_frame(fr, mask, min_variants=None, heads=('wt', 'comb'), name='x'):
    """Frame restricted (``use``) to ``mask``; with ``min_variants``, to scaffolds with at least that many rows in the mask."""
    sp = fr.spine.copy()
    m = sp['use'].to_numpy() & mask
    if min_variants:
        n = pd.Series(sp['group'].to_numpy()[m]).value_counts()
        keep = set(n[n >= min_variants].index)
        m &= sp['group'].isin(keep).to_numpy()
    sp['use'] = m
    return Frame(name, sp, {k: v for k, v in fr.preds.items() if k.rstrip('~') in heads})


def cell_stats(fr, mask, label, B, seed, heads=('wt', 'comb')):
    """Pooled and per-scaffold rho of every head, latent and linked, with intervals; returns rows and the replicate dictionaries."""
    rows = []
    keys = [h + f for h in heads for _, f in FORMS]
    for unit, mv in (('pooled', None), ('per scaffold', MIN_VARIANTS)):
        sf = sub_frame(fr, mask, mv, heads)
        n_rows = int(sf.spine['use'].sum())
        if n_rows < 30:
            continue
        point, reps = bootstrap_rho({'m': sf}, keys, kind='protein', B=B, seed=seed, per_protein=(unit == 'per scaffold'))
        for h in heads:
            for form, f in FORMS:
                k = h + f
                if k not in point['m']:
                    continue
                rho, per_seed, n, ng = point['m'][k]
                lo, hi = ci(reps['m'][k])
                rows.append(dict(subset=label, unit=unit, head=h, form=form, rho=rho, lo=lo, hi=hi, n_rows=n, n_scaffolds=ng, seeds='/'.join(f'{x:.3f}' for x in per_seed)))
            a, b = reps['m'].get(h), reps['m'].get(h + '~')
            if a is not None and b is not None:
                d = paired_diff(b, a)
                rows.append(dict(subset=label, unit=unit, head=h, form='linked - latent', rho=d['mean'], lo=d['lo'], hi=d['hi'], n_rows=n_rows, n_scaffolds=point['m'][h][3], seeds=(f"p={d['p']:.3g}" if unit == 'pooled' else 'identical (monotone within a scaffold)')))
    return rows


def tsuboyama_frame():
    """ESM-MSR on the capped Tsuboyama test libraries with the Tobit-linked reading (same link, per-library dG_wt)."""
    fr = load_final_frame('tsuboyama_capped_test', ['FINAL_p0'])
    sp = fr.spine.copy()
    t = pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', usecols=['id', 'dG_ML']).drop_duplicates('id').set_index('id')['dG_ML']
    sp['dG_ML'] = sp['id'].map(t).to_numpy()
    h = tobit_link(**TOBIT)
    dg = sp['dG_wt'].to_numpy(float)
    preds = {}
    for k in ('wt', 'comb', 'mt'):
        a = fr.preds[f'FINAL_p0:{k}']
        preds[k] = a
        preds[k + '~'] = h(dg[None, :] + a) - dg[None, :]
    return Frame('tsuboyama_p0', sp, preds)


def _job(args):
    which, stratum, name, B, seed, heads = args
    fr = _FR[which]
    sp = fr.spine
    classes = STRATA.get(stratum) if which == 'mgnify' else None             # homology strata: MGnify scaffolds only
    mask = SUBSETS[name](sp)
    if classes is not None:
        mask &= sp['cls'].isin(classes).to_numpy()
    dataset = DATASET[which]
    out = [dict(dataset=dataset, stratum=stratum, **r) for r in cell_stats(fr, mask, name, B, seed, heads)]
    print(dataset, stratum, name, flush=True)
    return out


_FR = {}


def run_stats(B, workers):
    import multiprocessing as mp
    seeds = finished_seeds()
    fr = load_mgnify(seeds=seeds, heldout=False)               # this script IS the homology analysis: all scaffolds, stratified by homology class
    tf = tsuboyama_frame()
    rk_seeds = finished_seeds(run='rdenovo')
    rk = load_mgnify(run='rdenovo', seeds=rk_seeds, heldout=False) if rk_seeds else None
    _FR.update(mgnify=fr, tsuboyama=tf, **({'rocklin': rk} if rk is not None else {}))
    if rk is not None:
        print(f'Rocklin designs FINAL_p0 seeds {rk_seeds}: {int(rk.spine["use"].sum()):,} measured variants, {rk.spine.loc[rk.spine["use"], "group"].nunique():,} designs', flush=True)
    print(f'MGnify FINAL_p0 seeds {seeds}: {int(fr.spine["use"].sum()):,} measured variants, {fr.spine.loc[fr.spine["use"], "group"].nunique():,} scaffolds', flush=True)
    jobs = []
    for stratum in STRATA:
        names = MAIN_SUBSETS + (['dG <= 0'] if stratum == 'all' else []) if stratum in ('all', 'clean') else ['overall', 'singles', 'doubles']
        for name in names:
            jobs.append(('mgnify', stratum, name, B, 11, ('wt', 'comb', 'mt') if stratum == 'all' else ('wt', 'comb')))
    for name in MAIN_SUBSETS:
        jobs.append(('tsuboyama', 'all', name, B, 12, ('wt', 'comb', 'mt')))
    if rk is not None:
        for name in ROCKLIN_SUBSETS:
            jobs.append(('rocklin', 'all', name, B, 13, ('wt', 'comb', 'mt')))
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    S = pd.DataFrame([r for part in res for r in part])
    S.to_csv(OUT / 'mgnify_summary.csv', index=False, float_format='%.4f')
    return fr, tf, rk, S


def run_rmse(fr, tf, rk=None):
    rows = []
    for ds, f in (('MGnify', fr), ('Tsuboyama capped test', tf), ('Rocklin designs', rk)):
        if f is None:
            continue
        sp = f.spine
        for name in (ROCKLIN_SUBSETS if ds == 'Rocklin designs' else MAIN_SUBSETS):
            m = sp['use'].to_numpy() & SUBSETS[name](sp)
            y = sp['y'].to_numpy()[m]
            for h in ('wt', 'comb'):
                for form, suf in FORMS:
                    p = f.preds[h + suf][:, m]
                    rows.append(dict(dataset=ds, subset=name, head=h, form=form, n=int(m.sum()), rmse=float(np.mean([np.sqrt(np.mean((q - y) ** 2)) for q in p])),
                                     mae=float(np.mean([np.mean(np.abs(q - y)) for q in p])), pearson=float(np.mean([pearsonr(q, y)[0] for q in p])), mean_pred=float(p.mean()), mean_measured=float(y.mean())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'mgnify_rmse.csv', index=False, float_format='%.4f')
    return R


# ---------------------------------------------------------------------------------------------- figures
def foot(fig, text):
    kw = dict(fontsize=7.6, color='#333333')
    t = fig.text(0.012, 0.008, style.wrap_to_px(fig, text, 0.96 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    return t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height


def finish(fig, text, stem, pad=0.04, top=1.0):
    fh = foot(fig, text)
    fig.tight_layout(rect=(0, fh + pad, 1, top))
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, OUT / stem)


def get(S, ds, stratum, subset, unit, head, form):
    t = S[(S.dataset == ds) & (S.stratum == stratum) & (S['subset'] == subset) & (S.unit == unit) & (S['head'] == head) & (S.form == form)]
    return (float(t.rho.iloc[0]), float(t.lo.iloc[0]), float(t.hi.iloc[0])) if len(t) else (np.nan,) * 3


def fig_overview(S, B, ds_main='MGnify', subsets=MAIN_SUBSETS, stem='mgnify_overview'):
    unit_word = 'scaffolds' if ds_main == 'MGnify' else 'designs'
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.9 * len(subsets) + 0.2), sharey=True)
    ys = np.arange(len(subsets))[::-1]
    for ax, unit, ttl in zip(axes, ('pooled', 'per scaffold'), ('A  Pooled Spearman rho', f'B  Mean per-{unit_word[:-1]} rho (>= {MIN_VARIANTS} variants)')):
        for j, (head, hl) in enumerate((('wt', 'WT head'), ('comb', 'combined'))):
            for ds, off0, mk in ((ds_main, 0.0, 'o'), ('Tsuboyama capped test', 0.0, 's')):
                for fi, (form, _) in enumerate(FORMS):
                    if unit == 'per scaffold' and form == 'linked':
                        continue                               # identical to latent per scaffold
                    off = (0.34 if ds == ds_main else -0.06) - 0.17 * j - (0.085 if fi else 0)
                    xs = [get(S, ds, 'all', s, unit, head, form) for s in subsets]
                    for y, (v, lo, hi) in zip(ys, xs):
                        if not np.isfinite(v):
                            continue
                        col = COLORS[form] if ds == ds_main else TSU_COLOR
                        ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, alpha=0.55, solid_capstyle='butt')
                        ax.plot(v, y + off, marker=mk, ms=6.4, mfc=col if head == 'comb' else 'white', mec=col, mew=1.3, ls='none')
        ax.set_yticks(ys)
        ax.set_yticklabels(subsets)
        ax.set_xlabel('Spearman rho')
        ax.set_title(ttl, loc='left')
        ax.grid(axis='x', color=style.GRID, lw=0.6)
        ax.set_xlim(0.0, 1.0)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker='o', color=COLORS['latent'], mfc=COLORS['latent'], ls='none', label=f'{ds_main}, latent'),
         Line2D([0], [0], marker='o', color=COLORS['linked'], mfc=COLORS['linked'], ls='none', label=f'{ds_main}, Tobit-linked'),
         Line2D([0], [0], marker='s', color=TSU_COLOR, mfc=TSU_COLOR, ls='none', label='Tsuboyama capped test (33 libraries)'),
         Line2D([0], [0], marker='o', color='#555555', mfc='#555555', ls='none', label='filled: combined'),
         Line2D([0], [0], marker='o', color='#555555', mfc='white', ls='none', label='open: WT head')]
    fig.legend(handles=h, loc='upper center', ncol=3, fontsize=8, handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    finish(fig, f'ESM-MSR, seed mean of FINAL_p0 s1-s3, measured variants (cens == 0). Pooled: one Spearman rho over all variants of the subset, 95% interval from resampling {unit_word} (libraries on Tsuboyama) and seeds, B = {B}. '
                f'Per {unit_word[:-1]}: mean over {unit_word} of the Spearman rho of the {unit_word[:-1]}\'s variants; the Tobit link (h(dG_wt + latent) - dG_wt) is monotone within a {unit_word[:-1]}, so latent and linked are the same number there. '
                'dG > 0: variants with measured dG_ML above 0; stabilizing / destabilizing: measured ddG above / below 0.', stem, top=0.9)


def fig_homology(S, B):
    strata = ['all', 'clean', 'h', 'strong']
    names = ['overall', 'singles', 'doubles']
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    w = 0.19
    cols = {'all': '#444444', 'clean': '#1baf7a', 'h': '#d6a02a', 'strong': '#c0392b'}
    for ax, unit, ttl in zip(axes, ('pooled', 'per scaffold'), ('A  Pooled rho, combined', 'B  Mean per-scaffold rho, combined')):
        for si, st in enumerate(strata):
            for ni, nm in enumerate(names):
                for fi, (form, _) in enumerate(FORMS):
                    if unit == 'per scaffold' and form == 'linked':
                        continue
                    v, lo, hi = get(S, 'MGnify', st, nm, unit, 'comb', form)
                    if not np.isfinite(v):
                        continue
                    x = ni + (si - 1.5) * w + (0.045 if fi else -0.045) * (unit == 'pooled')
                    ax.plot([x, x], [lo, hi], color=cols[st], lw=1.5, alpha=0.6)
                    ax.plot(x, v, marker='o' if form == 'latent' else 'D', ms=5.8, mfc=cols[st] if form == 'latent' else 'white', mec=cols[st], mew=1.3, ls='none')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(ttl, loc='left')
        ax.grid(axis='y', color=style.GRID, lw=0.6)
    axes[0].set_ylabel('Spearman rho')
    from matplotlib.lines import Line2D
    n_sc = {st: int(S[(S.dataset == 'MGnify') & (S.stratum == st) & (S['subset'] == 'overall') & (S.unit == 'pooled') & (S['head'] == 'comb') & (S.form == 'latent')].n_scaffolds.iloc[0]) for st in strata}
    lab = {'all': 'all scaffolds', 'clean': 'clean (none + low)', 'h': 'h (weak homolog)', 'strong': 'strong (H + same protein)'}
    h = [Line2D([0], [0], marker='o', color=cols[st], mfc=cols[st], ls='none', label=f'{lab[st]}, {n_sc[st]:,} scaffolds') for st in strata]
    h += [Line2D([0], [0], marker='o', color='#555555', mfc='#555555', ls='none', label='latent (filled)'), Line2D([0], [0], marker='D', color='#555555', mfc='white', ls='none', label='Tobit-linked (open, pooled)')]
    axes[0].legend(handles=h, loc='center', bbox_to_anchor=(0.5, 0.46), fontsize=7.6, handletextpad=0.3)
    finish(fig, f'ESM-MSR Dual-view prediction on MGnify, strata by the scaffold\'s sequence homology to the capped training set (MMseqs2 -s 7.5 against the training libraries\' wild types; classes none / low / h / H / same protein '
                f'of leak_audit lib.classify_hits, coverage over the MGnify scaffold). Seed mean of s1-s3, 95% intervals from resampling scaffolds and seeds, B = {B}.', 'mgnify_homology')


def fig_per_scaffold(fr):
    sp = fr.spine
    m0 = sp['use'].to_numpy() & (sp['n_mut'] >= 1).to_numpy()
    d = pd.DataFrame({'g': sp['group'].to_numpy()[m0], 'y': sp['y'].to_numpy()[m0]})
    for h in ('wt', 'comb'):
        d[h] = fr.preds[h][:, m0].mean(0)
    sc = {}
    for g, x in d.groupby('g'):
        if len(x) >= MIN_VARIANTS:
            sc[g] = (len(x), spearmanr(x['wt'], x['y'])[0], spearmanr(x['comb'], x['y'])[0])
    P = pd.DataFrame(sc, index=['n', 'wt', 'comb']).T
    P.to_csv(OUT / 'mgnify_per_scaffold_rho.csv', float_format='%.4f')
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ax = axes[0]
    sc_ = ax.scatter(P['wt'], P['comb'], s=7, c=np.log10(P['n']), cmap='viridis', alpha=0.7, lw=0)
    ax.plot([-0.2, 1], [-0.2, 1], color='#999', lw=0.8, ls='--')
    ax.set_xlabel('rho of the WT head')
    ax.set_ylabel('rho of the combined prediction')
    ax.text(0.04, 0.95, f'combined higher in {(P["comb"] > P["wt"]).sum():,} / {len(P):,} scaffolds', transform=ax.transAxes, va='top', fontsize=8.5)
    ax.set_title('A  Each dot is one scaffold', loc='left')
    cb = fig.colorbar(sc_, ax=ax, pad=0.02)
    cb.set_label('log10 variants in the scaffold')
    ax = axes[1]
    bins = np.linspace(-0.3, 1.0, 40)
    for h, c in (('wt', '#2a78d6'), ('comb', '#eb6834')):
        ax.hist(P[h], bins=bins, histtype='step', color=c, lw=1.7, label=f'{HEAD_LABEL[h]} (mean {P[h].mean():.3f}, median {P[h].median():.3f})')
    ax.set_xlabel('per-scaffold Spearman rho')
    ax.set_ylabel('scaffolds')
    ax.legend(fontsize=8)
    ax.set_title('B  Distribution over scaffolds', loc='left')
    finish(fig, f'MGnify scaffolds with at least {MIN_VARIANTS} measured variants (n = {len(P):,}), ESM-MSR predictions averaged over the three seeds, Spearman rho of the scaffold\'s variants against measured ddG. Latent and Tobit-linked '
                'are identical per scaffold.', 'mgnify_per_scaffold')
    return P


def fig_scatter(fr, tf):
    sp = fr.spine
    y = sp['y'].to_numpy()
    use = sp['use'].to_numpy()
    panels = [('wt', '', 'singles'), ('wt', '~', 'singles'), ('comb', '', 'doubles'), ('comb', '~', 'doubles')]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), sharey=True)
    for ax, (h, suf, subset) in zip(axes, panels):
        m = use & (sp['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy()
        x = fr.preds[h + suf][:, m].mean(0)
        style.density_scatter(ax, x, y[m], s=2, line=True, max_points=40000)
        ax.set_title(f'{HEAD_LABEL[h]}, {subset}\n{"Tobit-linked" if suf else "latent"}', fontsize=9, loc='left')
        ax.set_xlabel('predicted ddG (kcal/mol)')
    axes[0].set_ylabel('measured ddG (kcal/mol)')
    finish(fig, 'MGnify measured variants, ESM-MSR seed mean. The Tobit link h(dG_wt + latent) - dG_wt compresses predictions of very destabilized variants toward the unfolded floor (mu_f = -0.14).', 'mgnify_scatter')


def write_tables(S, R):
    """Markdown tables of the headline cells (``table_mgnify.md`` lean, ``table_mgnify_ci.md`` with 95% intervals, ``table_homology.md``)."""
    def cell(ds, st, sub, unit, head, form, ci_=False):
        t = S[(S.dataset == ds) & (S.stratum == st) & (S['subset'] == sub) & (S.unit == unit) & (S['head'] == head) & (S.form == form)]
        if not len(t):
            return '–'
        r = t.iloc[0]
        return f'{r.rho:.3f} [{r.lo:.3f}, {r.hi:.3f}]' if ci_ else f'{r.rho:.3f}'

    def n_of(ds, st, sub, unit, head='comb'):
        t = S[(S.dataset == ds) & (S.stratum == st) & (S['subset'] == sub) & (S.unit == unit) & (S['head'] == head) & (S.form == 'latent')]
        return (int(t.n_rows.iloc[0]), int(t.n_scaffolds.iloc[0])) if len(t) else (0, 0)

    def rmse(ds, sub, head, form):
        t = R[(R.dataset == ds) & (R['subset'] == sub) & (R['head'] == head) & (R.form == form)]
        return f'{t.rmse.iloc[0]:.3f}' if len(t) else '–'

    out = {}
    for tag, ci_ in (('table_mgnify', False), ('table_mgnify_ci', True)):
        lines = []
        for head in ('comb', 'wt', 'mt'):
            lines += [f'**{HEAD_LABEL[head]}**', '', '| set | subset | variants | scaffolds | pooled ρ, latent | pooled ρ, Tobit-linked | per-scaffold ρ | scaffolds (per-scaffold) | RMSE latent | RMSE linked |', '|---|---|---|---|---|---|---|---|---|---|']
            for ds, st, label in (('MGnify', 'all', 'MGnify'), ('Tsuboyama capped test', 'all', 'Tsuboyama capped test')):
                for sub in MAIN_SUBSETS:
                    if head != 'comb' and sub not in ('overall', 'singles', 'doubles'):
                        continue
                    if head == 'mt' and ds == 'MGnify' and st != 'all':
                        continue
                    n, ns = n_of(ds, st, sub, 'pooled', head)
                    _, nps = n_of(ds, st, sub, 'per scaffold', head)
                    if not n:
                        continue
                    lines.append(f'| {label} | {sub} | {n:,} | {ns:,} | {cell(ds, st, sub, "pooled", head, "latent", ci_)} | {cell(ds, st, sub, "pooled", head, "linked", ci_)} | {cell(ds, st, sub, "per scaffold", head, "latent", ci_)} | {nps:,} | '
                                 f'{rmse(ds, sub, head, "latent") if head != "mt" else "–"} | {rmse(ds, sub, head, "linked") if head != "mt" else "–"} |')
            lines.append('')
        (OUT / f'{tag}.md').write_text('\n'.join(lines) + '\n')
    for tag, ci_ in (('table_rocklin', False), ('table_rocklin_ci', True)):
        lines = []
        for head in ('comb', 'wt', 'mt'):
            lines += [f'**{HEAD_LABEL[head]}**', '', '| set | subset | variants | designs | pooled ρ, latent | pooled ρ, Tobit-linked | per-design ρ | designs (per-design) | RMSE latent | RMSE linked |', '|---|---|---|---|---|---|---|---|---|---|']
            for sub in ROCKLIN_SUBSETS:
                if head != 'comb' and sub not in ('overall', 'singles', 'doubles', '3 mutations'):
                    continue
                n, ns = n_of('Rocklin designs', 'all', sub, 'pooled', head)
                _, nps = n_of('Rocklin designs', 'all', sub, 'per scaffold', head)
                if not n:
                    continue
                lines.append(f'| Rocklin designs | {sub} | {n:,} | {ns:,} | {cell("Rocklin designs", "all", sub, "pooled", head, "latent", ci_)} | {cell("Rocklin designs", "all", sub, "pooled", head, "linked", ci_)} | '
                             f'{cell("Rocklin designs", "all", sub, "per scaffold", head, "latent", ci_)} | {nps:,} | {rmse("Rocklin designs", sub, head, "latent") if head != "mt" else "–"} | {rmse("Rocklin designs", sub, head, "linked") if head != "mt" else "–"} |')
            lines.append('')
        if len(lines) > 4:
            (OUT / f'{tag}.md').write_text('\n'.join(lines) + '\n')
    lines = ['| stratum | subset | scaffolds | pooled ρ, latent | pooled ρ, Tobit-linked | per-scaffold ρ |', '|---|---|---|---|---|---|']
    for st in STRATA:
        for sub in ('overall', 'singles', 'doubles'):
            n, ns = n_of('MGnify', st, sub, 'pooled')
            if n:
                lines.append(f'| {st} | {sub} | {ns:,} | {cell("MGnify", st, sub, "pooled", "comb", "latent", True)} | {cell("MGnify", st, sub, "pooled", "comb", "linked", True)} | {cell("MGnify", st, sub, "per scaffold", "comb", "latent", True)} |')
    (OUT / 'table_homology.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--figures-only', action='store_true', help='redraw the figures from the CSVs of an earlier run')
    a = ap.parse_args()
    if a.figures_only:
        fr, tf = load_mgnify(seeds=finished_seeds(), heldout=False), tsuboyama_frame()
        S = pd.read_csv(OUT / 'mgnify_summary.csv')
        S.loc[(S.form == 'linked - latent') & (S.unit == 'per scaffold'), 'seeds'] = 'identical (monotone within a scaffold)'
        S.to_csv(OUT / 'mgnify_summary.csv', index=False, float_format='%.4f')
    else:
        fr, tf, rk, S = run_stats(a.B, a.workers)
        run_rmse(fr, tf, rk)
    R = pd.read_csv(OUT / 'mgnify_rmse.csv')
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 500)
    show = S[(S.stratum == 'all') & S['head'].isin(['wt', 'comb']) & S['subset'].isin(ROCKLIN_SUBSETS)]
    print(show.drop(columns=['seeds']).round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))
    B = a.B
    write_tables(S, R)
    fig_overview(S, B)
    if (S.dataset == 'Rocklin designs').any():
        fig_overview(S, B, 'Rocklin designs', ROCKLIN_SUBSETS, 'rocklin_overview')
    fig_homology(S, B)
    fig_per_scaffold(fr)
    fig_scatter(fr, tf)


if __name__ == '__main__':
    main()
