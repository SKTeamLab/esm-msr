#!/usr/bin/env python
"""Every model retrained on the full MegaScale split (`mega_splits`), scored on that split's Tsuboyama validation (31 libraries)
and test (28 wild-type libraries) sets: the `03_tsuboyama.py` figures for the mega_splits retrains, and the capped-vs-mega comparison.

Models (three seeds each): ESM-MSR = FINAL_p0_mega (the FINAL_p0 recipe on mega_splits; testing queue, final_bench/), and the
MutateEverything st1 / st2, ThermoMPNN / ThermoMPNN-D and SPURS / SPURS multi retrains on mega_splits (cluster tracked runs
``outputs/tracked/{me_mega_st1,me_mega_st2,thermompnn_mega,spurs_mega}_s<k>``, collected to
``baselines/retrained/mega_splits/tracked/``), read by ``msr_analysis.data`` under ``MSR_TSU_SPLIT=mega`` (set here). The labels
are the task tables ``00_build_tasks.py`` writes from FINAL_p0_mega's prediction file (measured rows, ``cens == 0``: the same
49,048 test rows every baseline file holds). Outputs: ``results/03_tsuboyama/mega_splits/03_tsuboyama/``.

``tsuboyama_mega_overview``       mean per-library rho, validation and test; thick bar = resampling libraries, thin = mutations
``tsuboyama_mega_per_library``    per-library rho, ESM-MSR combined against each baseline (one dot per test library)
``tsuboyama_mega_singles_doubles``  singles and doubles separately
``tsuboyama_mega_scatter``        predicted vs measured ddG on the test set
``tsuboyama_capped_vs_mega``      test-set rho of each model trained and tested on the Homology-Restricted split
                                  (splits_oct06_capped) against the same model trained and tested on mega_splits

Scoring follows `03_tsuboyama.py`: stabilizing-positive predictions, Spearman rho per library (wild-type code, mutant-backbone
libraries pooled into their wild type), averaged over libraries and then over seeds; the bootstrap resamples libraries (and
seeds) with one draw shared by every model.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ['MSR_TSU_SPLIT'] = 'mega'          # before msr_analysis is imported: TSU, the loaders and RESULTS follow it

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import out_dir, subframe  # noqa: E402
from msr_analysis.data import Frame, _align_id, load_frame  # noqa: E402
from msr_analysis.final_runs import load_final_frame  # noqa: E402
from msr_analysis.registry import MODELS, TSU_CAPPED, TSU_MEGA  # noqa: E402

OUT = out_dir('03_tsuboyama')
SPLITS = ('val', 'test')
DS = {'val': TSU_MEGA[0], 'test': TSU_MEGA[1]}
MSR = ['msrp0:wt', 'msrp0:comb']
RETR = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
# mega_splits only (no capped-split counterpart): SPURS with the adapter trained, the published SPURS checkpoint, and the published ThermoMPNN / -D ensemble
# members (three members stand in for three seeds). ``spurs`` / ``spursm`` above are the retrains with the frozen random adapter.
EXTRA = ['spursu', 'spursum', 'spurspub', 'spurspubm', 'tmpnnpub', 'tmpnndpub']
BASE = MSR + RETR                     # the models that also exist on the capped split
RETR_ALL = RETR + EXTRA
KEYS = BASE + EXTRA
LABEL = {'msrp0:wt': 'ESM-MSR WT head', 'msrp0:comb': 'ESM-MSR combined', **{k: MODELS[k]['label'] for k in RETR_ALL}}
LABEL.update(spurs='SPURS, frozen adapter (retrain)', spursm='SPURS multi, frozen adapter (retrain)')


def mega_frame(split):
    fr = load_frame(DS[split])
    missing = [k for k in KEYS if k not in fr.preds or fr.preds[k].shape[0] not in ((1, 3) if k.startswith('spurspub') else (3,))]
    if missing:
        raise SystemExit(f'{DS[split]}: missing or not three seeds: {missing}')
    return fr


def capped_frame():
    """The Homology-Restricted test set with the capped-split models: FINAL_p0 (three seeds, testing queue) and the capped retrains."""
    fr = load_frame(TSU_CAPPED[1])
    p0 = load_final_frame(TSU_CAPPED[1], families=['FINAL_p0'])
    ids = fr.spine['id'].to_numpy(dtype=object)
    pid = p0.spine['id'].to_numpy(dtype=object)
    preds = {k: fr.preds[f'{k}:'] if f'{k}:' in fr.preds else fr.preds[k] for k in RETR}
    for h in ('wt', 'comb'):
        preds[f'msrp0:{h}'] = np.vstack([_align_id(ids, pid, r) for r in p0.preds[f'FINAL_p0:{h}']])
    return Frame(fr.name, fr.spine, preds)


def score(frames, B, subsets=None, kind='protein', keys=None):
    keys = keys or KEYS
    point, reps = bootstrap_rho(frames, keys, kind=kind, B=B, seed=18, per_protein=True)
    rows = []
    for ds in frames:
        for k in keys:
            if k not in point[ds]:
                continue
            rho, seeds, n, ng = point[ds][k]
            lo, hi = ci(reps[ds][k])
            rows.append(dict(dataset=ds, model=k, rho=rho, lo=lo, hi=hi, n=n, n_libraries=ng, n_seeds=len(np.atleast_1d(seeds)),
                             seed_rhos=';'.join(f'{x:.4f}' for x in np.atleast_1d(seeds))))
    return pd.DataFrame(rows), reps


def overview(fr, B):
    P, reps = score({s: fr[s] for s in SPLITS}, B)
    M, _ = score({s: fr[s] for s in SPLITS}, B, kind='mutation')
    P['kind'], M['kind'] = 'library', 'mutation'
    pd.concat([P, M]).to_csv(OUT / 'tsuboyama_mega_overall.csv', index=False, float_format='%.4f')
    p = {(r.dataset, r.model): r for r in P.itertuples()}
    m = {(r.dataset, r.model): r for r in M.itertuples()}
    keys = sorted(KEYS, key=lambda k: p[('test', k)].rho)
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.36 * len(keys) + 1.4), sharey=True)
    for ax, s in zip(axes, SPLITS):
        for i, k in enumerate(keys):
            r, mm = p[(s, k)], m[(s, k)]
            kw = style.marker_kwargs(k, size=6.5)
            ax.plot([r.lo, r.hi], [i, i], color=kw['color'], lw=2.4, alpha=.45, solid_capstyle='butt')
            ax.plot([mm.lo, mm.hi], [i, i], color=kw['color'], lw=0.8)
            ax.plot(r.rho, i, **kw)
            ax.text(max(r.hi, mm.hi) + .006, i, f'{r.rho:.3f}', va='center', fontsize=7.5, color='#444')
        ax.set_title(f'{s} ({p[(s, keys[0])].n_libraries} libraries)', loc='left')
        ax.grid(axis='x', color=style.GRID, lw=.6)
        ax.set_xlabel('mean per-library Spearman ρ')
    lo = min(r.lo for r in M.itertuples()) - .02
    hi = max(r.hi for r in pd.concat([P, M]).itertuples()) + .045
    for ax in axes:
        ax.set_xlim(lo, hi)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([LABEL[k] for k in keys], fontsize=8.5)
    for t, k in zip(axes[0].get_yticklabels(), keys):
        if k in MSR:
            t.set_fontweight('bold')
    fig.suptitle('Every model trained on mega_splits (three seeds each). Thick bar: resampling libraries; thin line: resampling mutations', fontsize=8.5, y=1.01)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_mega_overview')
    return P, reps


def paired(reps, ds, a, b):
    d = reps[ds][a] - reps[ds][b]
    d = d[np.isfinite(d)]
    return float(np.mean(d > 0)) if len(d) else np.nan


def medians(fr):
    """Median over libraries of the per-library Spearman rho, per seed, then the mean over seeds (and the seed range): on every measured variant (this table's population) and on the
    wild-type-backbone singles only, which is the population of the SPURS paper's statistic (median per protein, 28 proteins, 28,172 of its 28,312 mutants)."""
    sp = fr.spine
    y = sp['y'].to_numpy()
    base = sp['use'].to_numpy()
    sets = {'all': base, 'singles_wt': base & (sp['n_mut'] == 1).to_numpy() & sp['wt_backbone'].to_numpy().astype(bool)}
    rows = []
    for k in KEYS:
        row = dict(model=k)
        for nm, ok in sets.items():
            per_seed = []
            for p in np.atleast_2d(fr.preds[k]):
                r = [spearmanr(p[m], y[m])[0] for g in sp['group'][ok].unique()
                     for m in [ok & (sp['group'] == g).to_numpy() & np.isfinite(p)] if m.sum() >= 5]
                per_seed.append(np.median(r))
            row[f'median_{nm}'], row[f'median_{nm}_min'], row[f'median_{nm}_max'] = np.mean(per_seed), np.min(per_seed), np.max(per_seed)
        rows.append(row)
    M = pd.DataFrame(rows).set_index('model')
    M.to_csv(OUT / 'tsuboyama_mega_test_medians.csv', float_format='%.4f')
    return M


def per_library(fr):
    sp = fr.spine
    ok = sp['use'].to_numpy()
    y = sp['y'].to_numpy()
    libs = sorted(sp['group'][ok].unique())

    def lib_rho(k):
        out = {}
        for g in libs:
            m = ok & (sp['group'] == g).to_numpy() & np.isfinite(fr.preds[k]).all(0)
            out[g] = np.mean([spearmanr(p[m], y[m])[0] for p in fr.preds[k]])
        return pd.Series(out)
    R = pd.DataFrame({k: lib_rho(k) for k in KEYS})
    R.to_csv(OUT / 'per_library_rho_mega_test.csv', float_format='%.4f')
    ncol = 6
    fig, axes = plt.subplots(-(-len(RETR_ALL) // ncol), ncol, figsize=(2.6 * ncol, 3.0 * -(-len(RETR_ALL) // ncol)), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, k in zip(axes, RETR_ALL):
        ax.scatter(R[k], R['msrp0:comb'], s=22, color=MODELS[k]['color'], alpha=.8, lw=0)
        ax.plot([.2, 1], [.2, 1], color='#999', lw=.8, ls='--')
        ax.set_title(LABEL[k], fontsize=8.5)
        ax.text(.04, .95, f'ESM-MSR higher in {(R["msrp0:comb"] > R[k]).sum()} / {len(R)}', transform=ax.transAxes, va='top', fontsize=8)
        ax.set_xlabel('ρ of the baseline')
    axes[0].set_ylabel('ρ of ESM-MSR combined')
    for ax in axes[len(RETR_ALL):]:
        ax.axis('off')
    fig.suptitle('mega_splits test: each dot is one library', fontsize=9, y=1.0)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_mega_per_library')
    return R


def singles_doubles(fr, B):
    rows = []
    for s in SPLITS:
        sp = fr[s].spine
        for nm, nmut in (('singles', 1), ('doubles', 2)):
            sub = subframe(fr[s], (sp['n_mut'] == nmut).to_numpy() & sp['use'].to_numpy(), keys=KEYS)
            point, reps = bootstrap_rho({s: sub}, KEYS, kind='protein', B=B, seed=6, per_protein=True)
            for k, (rho, _, n, ng) in point[s].items():
                lo, hi = ci(reps[s][k])
                rows.append(dict(dataset=s, subset=nm, model=k, rho=rho, lo=lo, hi=hi, n=n, n_libraries=ng))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'tsuboyama_mega_singles_doubles.csv', index=False, float_format='%.4f')
    groups = [f'{s}|{x}' for s in SPLITS for x in ('singles', 'doubles')]
    glabel = [f'{s}\n{x}' for s in SPLITS for x in ('singles', 'doubles')]
    table = {(f'{r.dataset}|{r.subset}', r.model): r.rho for r in R.itertuples()}
    cis = {(f'{r.dataset}|{r.subset}', r.model): (r.lo, r.hi) for r in R.itertuples()}
    fig, ax = plt.subplots(figsize=(8.5, 4))
    style.dot_panel(ax, groups, KEYS, table, cis, highlight=MSR, ylabel='mean per-library ρ', xlabels=glabel, rot=0, size=6.5,
                    ylim=(min(R.lo) - .03, min(1.0, max(R.hi) + .03)))
    _legend(ax, KEYS, loc='lower left', ncol=4, fontsize=7.5)
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_mega_singles_doubles')
    return R


def _legend(ax, keys, **kw):
    from matplotlib.lines import Line2D
    h = []
    for k in keys:
        mk = style.marker_kwargs(k)
        mk.pop('ls', None)
        h.append(Line2D([], [], ls='none', label=LABEL[k], **mk))
    ax.legend(handles=h, **kw)


def scatter(fr):
    sp = fr.spine
    y = sp['y'].to_numpy()
    ok = sp['use'].to_numpy()
    panels = [('msrp0:wt', 'singles', 'ESM-MSR WT head'), ('msrp0:comb', 'doubles', 'ESM-MSR combined'), ('me2', 'doubles', 'MutateEverything st2'),
              ('tmpnnd', 'doubles', 'ThermoMPNN + D'), ('spursm', 'doubles', 'SPURS multi')]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.2), sharey=True)
    for ax, (k, subset, t) in zip(axes, panels):
        pr = fr.preds[k].mean(axis=0)                                     # averaged over the three seeds
        m = ok & (sp['n_mut'] == (1 if subset == 'singles' else 2)).to_numpy() & np.isfinite(pr)
        style.density_scatter(ax, pr[m], y[m], s=3, line=True, max_points=40000)
        ax.set_title(f'{t}\n{subset} (mean of seeds 1-3)', fontsize=8.5)
        ax.set_xlabel('predicted ddG')
    axes[0].set_ylabel('measured ddG (kcal/mol)')
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_mega_scatter')


def capped_vs_mega(mega_test, B):
    cap = capped_frame()
    C, _ = score({'capped': cap}, B, keys=BASE)
    M, _ = score({'mega': mega_test}, B, keys=BASE)
    R = pd.concat([C, M])
    R.to_csv(OUT / 'tsuboyama_capped_vs_mega.csv', index=False, float_format='%.4f')
    c = {r.model: r for r in C.itertuples()}
    m = {r.model: r for r in M.itertuples()}
    keys = sorted(BASE, key=lambda k: m[k].rho)
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(keys) + 1.3))
    for i, k in enumerate(keys):
        kw = style.marker_kwargs(k, size=6.5)
        ax.plot([c[k].rho, m[k].rho], [i, i], color=kw['color'], lw=1, alpha=.6)
        for r, alpha in ((c[k], .35), (m[k], 1.0)):
            ax.plot([r.lo, r.hi], [i, i], color=kw['color'], lw=2.4, alpha=.3 * alpha, solid_capstyle='butt')
            ax.plot(r.rho, i, **style.marker_kwargs(k, size=6.5, alpha=alpha))
        ax.text(1.0, i, f'{c[k].rho:.3f} → {m[k].rho:.3f}  ({m[k].rho - c[k].rho:+.3f})', va='center', fontsize=7.5, color='#444',
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABEL[k] for k in keys], fontsize=8.5)
    ax.grid(axis='x', color=style.GRID, lw=.6)
    ax.set_xlabel('test-set mean per-library Spearman ρ')
    ax.set_title(f'faded: Homology-Restricted split ({c[keys[0]].n_libraries} test libraries); solid: mega_splits ({m[keys[0]].n_libraries})',
                 loc='left', fontsize=8.5, fontweight='normal')
    fig.tight_layout()
    style.save(fig, OUT / 'tsuboyama_capped_vs_mega')
    return R


def table_md(P, reps, SD, CM, MED):
    p = {(r.dataset, r.model): r for r in P.itertuples()}
    sd = {(r.dataset, r.subset, r.model): r for r in SD.itertuples()}
    cm = {(r.dataset, r.model): r for r in CM.itertuples()}
    ref = 'msrp0:comb'
    lines = ['| Model | val ρ | test ρ | test singles | test doubles | test median ρ (all variants) | test median ρ, WT-backbone singles (paper statistic) | Homology-Restricted test ρ | P(ESM-MSR comb > model), test |', '|---|---|---|---|---|---|---|---|---|']
    for k in sorted(KEYS, key=lambda k: -p[('test', k)].rho):
        f = lambda r: f'{r.rho:.3f} [{r.lo:.3f}, {r.hi:.3f}]'
        pr = '' if k == ref else f'{paired(reps, "test", ref, k):.2f}'
        cap = f'{cm[("capped", k)].rho:.3f}' if ('capped', k) in cm else '–'
        lines.append(f'| {LABEL[k]} | {f(p[("val", k)])} | {f(p[("test", k)])} | {sd[("test", "singles", k)].rho:.3f} | {sd[("test", "doubles", k)].rho:.3f} | '
                     f'{MED.loc[k, "median_all"]:.3f} | {MED.loc[k, "median_singles_wt"]:.3f} | {cap} | {pr} |')
    n = p[('test', ref)]
    lines += ['', 'Medians: median over the test libraries of the per-library ρ (per seed, then the mean over seeds; the seed range is in `tsuboyama_mega_test_medians.csv`). The paper statistic is the SPURS paper\'s: '
              'median per protein over the wild-type-backbone singles (SPURS 0.83, ThermoMPNN 0.77 as reported there).', f'Mean per-library Spearman ρ, seed mean of three seeds per model; [95% interval] from resampling libraries and seeds '
              f'(B = {len(reps["test"][ref])}). mega_splits test: {n.n_libraries} libraries, {n.n:,} measured variants.']
    txt = '\n'.join(lines) + '\n'
    (OUT / 'table_mega_split.md').write_text(txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--B', type=int, default=1000, help='bootstrap replicates (default 1000; paper 2000)')
    a = ap.parse_args()
    fr = {s: mega_frame(s) for s in SPLITS}
    for s in SPLITS:
        cov = fr[s].notes['coverage']
        low = {k: round(v, 4) for k, v in cov.items() if v < 1 and k in KEYS}
        print(f'{s}: {fr[s].spine["use"].sum():,} measured rows, {fr[s].spine.loc[fr[s].spine["use"], "group"].nunique()} libraries; '
              f'coverage < 1: {low or "none"}')
    P, reps = overview(fr, a.B)
    per_library(fr['test'])
    SD = singles_doubles(fr, a.B)
    scatter(fr['test'])
    CM = capped_vs_mega(fr['test'], a.B)
    MED = medians(fr['test'])
    print(table_md(P, reps, SD, CM, MED))


if __name__ == '__main__':
    main()
