#!/usr/bin/env python
"""Rosetta Cartesian ddG on multi-mutants: scoring the multi-mutant structure directly (``multi``) against summing the single-mutant predictions (``additive``).

``rosetta`` is the cartesian_ddg of the multi-mutant itself (FastRelaxed backbone, three iterations; doubles and higher-order variants scored directly); ``rosetta+`` is the same run summed over the single substitutions
of the variant (the harness' ``pred_additive``).  Both are in kcal/mol, stabilizing-positive, and are the same number on a single mutant.  Sets with multi-mutants that both are scored on:

  PTMUL (914 variants: 571 doubles, 343 with three or more mutations), PTMUL-D (536 doubles), the doubles of the Tsuboyama capped test (13,319; 13 libraries) and validation (8,547; 9 libraries) sets.

Spearman rho of each version against the measured ddG, and the paired difference ``multi - additive`` (same rows, same draws):

  unit          pooled: one rho over the rows of the subset, 95% interval resampling mutations (``mutation``) or proteins (``protein``; the libraries on Tsuboyama);
                ``library mean`` (Tsuboyama): the mean over libraries of the per-library rho, libraries resampled
  replicate     ``iteration 1`` is the one the figures use (``02_thermostability``, ``01_summary``); ``3 iterations`` the mean over the three Rosetta iterations (PTMUL, PTMUL-D; Tsuboyama has one)
  p             p_one_sided = P(multi - additive <= 0) over the draws (the test of "multi is better"), p_two_sided = max(2 min(P(d <= 0), P(d >= 0)), 1 / (B + 1))

Writes ``results/19_rosetta_multi_vs_additive/rosetta_multi_vs_additive.csv`` / ``.md`` and the figure ``rosetta_multi_vs_additive``.

    python analysis/19_rosetta_multi_vs_additive.py [--b-mut 200000] [--b-prot 100000] [--b-pooled 20000] [--workers 6]
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import Frame, load_frame  # noqa: E402

OUT = out_dir('19_rosetta_multi_vs_additive')
CHUNK = 5000
SPECS = [  # (dataset, subset label, mask on n_mut, units)
    ('ptmul', 'all', lambda n: n >= 2, ('mutation', 'protein')),
    ('ptmul', 'doubles', lambda n: n == 2, ('mutation', 'protein')),
    ('ptmul', '3+ mutations', lambda n: n >= 3, ('mutation', 'protein')),
    ('ptmuld', 'doubles', lambda n: n == 2, ('mutation', 'protein')),
    ('tsuboyama_capped_test', 'doubles', lambda n: n == 2, ('mutation', 'protein', 'library mean')),
    ('tsuboyama_capped_val', 'doubles', lambda n: n == 2, ('mutation', 'protein', 'library mean')),
]
LABEL = {'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D', 'tsuboyama_capped_test': 'Tsuboyama test', 'tsuboyama_capped_val': 'Tsuboyama val'}
_FR, _SUB = {}, {}


def _job(a):
    key, unit, B, seed, ns = a
    fr, mask = _FR[(key[0], ns)], _SUB[key]
    kind = 'mutation' if unit == 'mutation' else 'protein'
    pt, rp = bootstrap_rho({key[0]: fr}, ['multi', 'additive'], kind=kind, B=B, seed=seed, per_protein=(unit == 'library mean'), common=True, subsets={key[0]: mask})
    return key, unit, ns, {m: v for m, v in rp[key[0]].items()}, {m: (v[0], v[2], v[3]) for m, v in pt[key[0]].items()}


def p_two_sided(d):
    return float(max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / (len(d) + 1)))


def fmt_p(p, B):
    """p as text; a p at the resolution of the bootstrap (no draw on the other side) is given as a bound."""
    return f'<{1.0 / (B + 1):.0e}' if p * (B + 1) <= 1.0001 else f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}'


def write_md(R):
    lines = ['| set | subset | unit | replicate | rows | rho multi | rho additive | Δρ (multi − additive) | 95% interval | p, multi better (one-sided) | p two-sided |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R.itertuples():
        lines.append(f'| {LABEL[r.dataset]} | {r.subset} | {r.unit} | {r.replicate} | {r.n_rows:,} | {r.rho_multi:.3f} | {r.rho_additive:.3f} | {r.delta:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | '
                     f'{fmt_p(r.p_one_sided, r.B)} | {fmt_p(r.p_two_sided, r.B)} |')
    (OUT / 'rosetta_multi_vs_additive.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b-mut', type=int, default=1000)          # default size (paper: 200000)
    ap.add_argument('--b-prot', type=int, default=1000)         # default size (paper: 100000)
    ap.add_argument('--b-pooled', type=int, default=1000, help='draws for the pooled Tsuboyama rows (13k rows each)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--figures-only', action='store_true', help='redraw the figure and the table from the CSV of an earlier run')
    a = ap.parse_args()
    if a.figures_only:
        R = pd.read_csv(OUT / 'rosetta_multi_vs_additive.csv')
        write_md(R)
        figure(R)
        return
    for ds in {s[0] for s in SPECS}:
        fr = load_frame(ds)
        nrep = fr.preds['rosetta'].shape[0]
        for ns in sorted({1, nrep}):
            _FR[(ds, ns)] = Frame(ds, fr.spine, {'multi': fr.preds['rosetta'][:ns], 'additive': fr.preds['rosetta+'][:ns]})
    jobs = []
    for ds, lab, f, units in SPECS:
        _SUB[(ds, lab)] = f(_FR[(ds, 1)].spine['n_mut'].to_numpy())
        big = ds.startswith('tsuboyama')
        for ns in sorted({n for (d, n) in _FR if d == ds}):
            for unit in units:
                B = a.b_pooled if (big and unit != 'library mean') else (a.b_mut if unit == 'mutation' else a.b_prot)
                chunk = 25000 if unit == 'library mean' else CHUNK
                jobs += [((ds, lab), unit, min(chunk, B - i * chunk), 31 * i + 7 * ns + len(unit), ns) for i in range(int(np.ceil(B / chunk)))]
    print(f'{len(jobs)} chunks', flush=True)
    with mp.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    reps, pts = {}, {}
    for key, unit, ns, rp, pt in res:
        for m, v in rp.items():
            reps.setdefault((key, unit, ns, m), []).append(v)
        pts[(key, unit, ns)] = pt
    rows = []
    for ds, lab, f, units in SPECS:
        for ns in sorted({n for (d, n) in _FR if d == ds}):
            for unit in units:
                key = (ds, lab)
                mu, ad = (np.concatenate(reps[(key, unit, ns, m)]) for m in ('multi', 'additive'))
                ok = np.isfinite(mu) & np.isfinite(ad)                     # a library-mean draw can have no usable library
                mu, ad = mu[ok], ad[ok]
                d = mu - ad
                pt = pts[(key, unit, ns)]
                rows.append(dict(dataset=ds, subset=lab, unit=unit, replicate='iteration 1' if ns == 1 else f'{ns} iterations', n_rows=int(pt['multi'][1]), n_groups=int(pt['multi'][2]),
                                 rho_multi=pt['multi'][0], rho_additive=pt['additive'][0], delta=pt['multi'][0] - pt['additive'][0], lo=np.quantile(d, 0.025), hi=np.quantile(d, 0.975),
                                 multi_lo=np.quantile(mu, 0.025), multi_hi=np.quantile(mu, 0.975), add_lo=np.quantile(ad, 0.025), add_hi=np.quantile(ad, 0.975),
                                 p_one_sided=float(max(np.mean(d <= 0), 1.0 / (len(d) + 1))), p_two_sided=p_two_sided(d), B=len(d)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'rosetta_multi_vs_additive.csv', index=False, float_format='%.6g')
    write_md(R)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(R[['dataset', 'subset', 'unit', 'replicate', 'n_rows', 'n_groups', 'rho_multi', 'rho_additive', 'delta', 'lo', 'hi', 'p_one_sided', 'p_two_sided', 'B']].round(4).to_string(index=False))
    figure(R)


def figure(R):
    """ρ of both versions (95% intervals) and the difference, one row per set / subset; mutation resampling for PTMUL(-D) and pooled Tsuboyama, the library mean for Tsuboyama's per-library rows."""
    use = [('ptmul', 'all', 'mutation', 'PTMUL, all'), ('ptmul', 'doubles', 'mutation', 'PTMUL, doubles'), ('ptmul', '3+ mutations', 'mutation', 'PTMUL, 3+ mutations'),
           ('ptmuld', 'doubles', 'mutation', 'PTMUL-D, doubles'), ('tsuboyama_capped_test', 'doubles', 'library mean', 'Tsuboyama test, doubles, per library'),
           ('tsuboyama_capped_val', 'doubles', 'library mean', 'Tsuboyama val, doubles, per library')]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1]), sharey=True)
    ys = np.arange(len(use))[::-1]
    for y, (ds, sub, unit, lab) in zip(ys, use):
        r = R[(R.dataset == ds) & (R.subset == sub) & (R.unit == unit) & (R.replicate == 'iteration 1')].iloc[0]
        ax = axes[0]
        for off, v, lo, hi, filled in ((0.13, r.rho_multi, r.multi_lo, r.multi_hi, True), (-0.13, r.rho_additive, r.add_lo, r.add_hi, False)):
            ax.plot([lo, hi], [y + off] * 2, color='#8c564b', lw=1.5, alpha=0.6)
            ax.plot(v, y + off, marker='P', ms=7.5 * (1.0 if filled else 1.2), mfc='#8c564b' if filled else 'none', mec='#8c564b', mew=1.2 if filled else 2.0, ls='none')
        ax = axes[1]
        ax.plot([r.lo, r.hi], [y, y], color='#444444', lw=1.6)
        ax.plot(r.delta, y, marker='o', ms=6.5, color='#444444')
        ax.text(0.172, y, f'{r.delta:+.3f}   p {fmt_p(r.p_one_sided, r.B) if r.p_one_sided * (r.B + 1) <= 1.0001 else "= " + format(r.p_one_sided, ".2g")}', va='center', fontsize=8.2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([u[3] + ' (n = %s)' % f'{int(R[(R.dataset == u[0]) & (R.subset == u[1]) & (R.unit == u[2])].n_rows.iloc[0]):,}' for u in use])
    axes[0].set_xlabel(r'Spearman $\rho$')
    axes[0].set_title('A  filled: multi-mutant scored directly; hollow: singles summed', loc='left', fontsize=9.5)
    axes[1].axvline(0, color='#999999', lw=0.8)
    axes[1].set_xlim(-0.06, 0.30)
    axes[1].set_xlabel(r'$\Delta\rho$ (multi $-$ additive)')
    axes[1].set_title('B  Difference (95% interval), one-sided p', loc='left', fontsize=9.5)
    for ax in axes:
        ax.grid(axis='x', color=style.GRID, lw=0.6)
    kw = dict(fontsize=7.8, color='#333333')
    note = ('Rosetta Cartesian ddG, iteration 1 (the one the figures use). Both versions on the same rows (the variants with both predictions; 94 of the 13,319 Tsuboyama test doubles lack the summed one) and the same bootstrap draws. PTMUL / PTMUL-D: pooled rho, mutations resampled '
            '(B = 200,000); Tsuboyama: mean of the per-library rho, libraries resampled (B = 100,000). p = share of draws with multi - additive <= 0.')
    t = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.96 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = t.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    fig.tight_layout(rect=(0, foot + 0.05, 1, 1))
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box', bad)
    style.save(fig, OUT / 'rosetta_multi_vs_additive')


if __name__ == '__main__':
    main()
