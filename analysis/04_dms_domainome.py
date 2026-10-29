#!/usr/bin/env python
"""Deep-mutational scans and Domainome: BENCHMARKS tables 1.3 (with the ME additive rows), 1.3 sigma sweep and 1.6 (DMS part).

``dms_sigma_vs_zero_shot``   ESM-MSR at LoRA scale sigma = 1, 0.75, 0.5, 0.25 and its sigma = 0 control against the other zero-shot models, on the
                             7 DMS, their mean, and Domainome.  The head follows the data: WT head where the assay holds only single mutants,
                             combined where it holds doubles (DLG4, GRB2, GB1).
``dms_sigma_sweep``          the same sweep as rho against sigma, assay by assay.
``dms_vs_retrains``          ESM-MSR against the retrained baselines (and ME's additive readouts) on the same columns.
Intervals: resampling mutation sites (first mutated position) for a DMS, whole domains for Domainome (the mean over 522 domains); the mean of the 7 DMS
combines the independent site-level resamples.  The best sigma per assay is picked on these assays: an upper bound, not a held-out result.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import ci  # noqa: E402
from msr_analysis.common import has_multi, load_rho_df, out_dir, primary_ci  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import CACHE, DMS7, LABEL, MODELS  # noqa: E402

OUT = out_dir('04_dms_domainome')
DS = DMS7[:6] + [DMS7[6], 'domainome']                    # GB1 last among the DMS, as in the tables
COLS = DS[:7] + ['mean of 7 DMS'] + ['domainome']
FAMS = {'devel σ = 1': ('msrdev', None), 'devel σ = 0.75': ('msrdev@0.75', None), 'devel σ = 0.5': ('msrdev@0.5', None), 'devel σ = 0.25': ('msrdev@0.25', None),
        'σ = 0 (zero-shot control)': ('msrrel@0', None)}


def head(base, ds):
    comb = has_multi(load_frame(ds))
    return f'{base}:{"comb" if comb else "wt"}'


def build():
    df = load_rho_df()
    P = primary_ci(df)
    site = pickle.load(open(CACHE / 'boot_site.pkl', 'rb'))
    prot = pickle.load(open(CACHE / 'boot_protein.pkl', 'rb'))

    def cell(key_fn, ds):
        k = key_fn(ds)
        return P.get((ds, k)), k

    def mean7(key_fn):
        ks = [key_fn(d) for d in DMS7]
        if not all((d, k) in P for d, k in zip(DMS7, ks)):
            return None
        pt = np.mean([P[(d, k)][0] for d, k in zip(DMS7, ks)])
        B = min(len(site[d][k]) for d, k in zip(DMS7, ks))
        r = np.mean([site[d][k][:B] for d, k in zip(DMS7, ks)], axis=0)
        return (pt,) + ci(r)
    return P, mean7


def table_for(models, P, mean7):
    """models: {display: key_fn(ds) -> key}; returns table, cis keyed (column, display)"""
    t, c = {}, {}
    for name, fn in models.items():
        for d in DMS7 + ['domainome']:
            v = P.get((d, fn(d)))
            if v is not None:
                t[(d, name)], c[(d, name)] = v[0], (v[1], v[2])
        m = mean7(fn)
        if m is not None:
            t[('mean of 7 DMS', name)], c[('mean of 7 DMS', name)] = m[0], (m[1], m[2])
    return t, c


def main():
    P, mean7 = build()
    # registry styles for the display names
    def reg(name, base, **kw):
        MODELS[name] = dict(label=name, group='dms', **kw)
    zs = {'ProSST (zero-shot)': 'prosst', 'ProteinMPNN (zero-shot)': 'pmpnn', 'ESM3-small (Forge)': 'esm3s', 'ESM3ΔG': 'esm3dg'}
    models1 = {}
    sig_style = {'devel σ = 1': ('#000000', 'D', True), 'devel σ = 0.75': ('#333333', 'D', False), 'devel σ = 0.5': ('#666666', 'D', False), 'devel σ = 0.25': ('#999999', 'D', False),
                 'σ = 0 (zero-shot control)': ('#7f8c8d', '^', True)}
    for n, (b, _) in FAMS.items():
        models1[n] = (lambda b: (lambda d: head(b, d)))(b)
        c, m, f = sig_style[n]
        MODELS[n] = dict(label=n if 'devel' not in n else 'ESM-MSR ' + n, color=c, marker=m, fill=f, group='dms')
    for n, k in zs.items():
        models1[n] = (lambda k: (lambda d: k))(k)
        MODELS[n] = dict(MODELS[k], label=n)
    t, c = table_for(models1, P, mean7)
    pd.DataFrame([dict(column=k[0], model=k[1], rho=t[k], lo=c[k][0], hi=c[k][1]) for k in t]).to_csv(OUT / 'dms_sigma_vs_zero_shot.csv', index=False, float_format='%.4f')
    names = list(models1)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    style.dot_panel(ax, COLS, names, t, c, highlight=['devel σ = 1'], ylabel='Spearman ρ', xlabels=[LABEL.get(x, x) for x in COLS], size=6.2, spread=.8, ylim=(0.0, 0.95))
    style.model_legend(ax, names, loc='lower left', ncol=3, fontsize=7.5)
    ax.set_title('ESM-MSR (devel seeds) at lower LoRA scale, against zero-shot models; whiskers: 95% bootstrap over sites (DMS) / domains (Domainome)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, OUT / 'dms_sigma_vs_zero_shot')

    # sweep
    fig, axes = plt.subplots(1, 8, figsize=(16, 2.9), sharey=False)
    sig = [0, 0.25, 0.5, 0.75, 1.0]
    for ax, col in zip(axes, DMS7 + ['domainome']):
        for fam, lab, colr in (('msrdev', 'devel (3 seeds)', 'k'), ('msrrel', 'released', '#888888')):
            ys, los, his, xs = [], [], [], []
            for s in sig:
                key = head('msrrel@0' if s == 0 else (fam if s == 1.0 else f'{fam}@{s:g}'), col)
                v = P.get((col, key))
                if v is not None:
                    xs.append(s); ys.append(v[0]); los.append(v[1]); his.append(v[2])
            if ys:
                ax.plot(xs, ys, '-o', color=colr, ms=3.5, lw=1.3, label=lab)
                ax.fill_between(xs, los, his, color=colr, alpha=.12, lw=0)
        ax.set_title(LABEL[col], fontsize=9); ax.set_xlabel('LoRA scale σ'); ax.set_xticks(sig); ax.set_xticklabels(['0', '', '.5', '', '1'])
    axes[0].set_ylabel('Spearman ρ'); axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    style.save(fig, OUT / 'dms_sigma_sweep')

    # retrains
    def fixed(k):
        return lambda d: k
    mod2 = {'ESM-MSR (baseline)': lambda d: head('msr', d), 'ESM-MSR (devel)': lambda d: head('msrdev', d),
            'MutateEverything st1': fixed('me1'), 'MutateEverything st1, additive': fixed('me1+'), 'MutateEverything st2': fixed('me2'),
            'MutateEverything st2, additive': fixed('me2+'), 'ThermoMPNN': fixed('tmpnn'), 'ThermoMPNN + ThermoMPNN-D': fixed('tmpnnd'),
            'SPURS': fixed('spurs'), 'SPURS multi': fixed('spursm')}
    sty = {'ESM-MSR (baseline)': ('msr', 'ESM-MSR (baseline)'), 'ESM-MSR (devel)': ('msrdev:comb', 'ESM-MSR (devel)'), 'MutateEverything st1': ('me1', None),
           'MutateEverything st1, additive': ('me1+', None), 'MutateEverything st2': ('me2', None), 'MutateEverything st2, additive': ('me2+', None), 'ThermoMPNN': ('tmpnn', None),
           'ThermoMPNN + ThermoMPNN-D': ('tmpnnd', None), 'SPURS': ('spurs', None), 'SPURS multi': ('spursm', None)}
    for n, (k, _) in sty.items():
        MODELS[n] = dict(MODELS[k], label=n)
    t2, c2 = table_for(mod2, P, mean7)
    pd.DataFrame([dict(column=k[0], model=k[1], rho=t2[k], lo=c2[k][0], hi=c2[k][1]) for k in t2]).to_csv(OUT / 'dms_vs_retrains.csv', index=False, float_format='%.4f')
    fig, ax = plt.subplots(figsize=(13, 4.8))
    style.dot_panel(ax, COLS, list(mod2), t2, c2, highlight=['ESM-MSR (baseline)'], ylabel='Spearman ρ', xlabels=[LABEL.get(x, x) for x in COLS], size=6.0, spread=.82, ylim=(0.15, 0.85))
    style.model_legend(ax, list(mod2), loc='upper right', ncol=5, fontsize=7.3)
    ax.set_title('ESM-MSR against the retrained baselines on the deep scans (ESM-MSR: WT head on single-mutant assays, combined where doubles are scored)', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, OUT / 'dms_vs_retrains')
    print(pd.DataFrame([dict(column=k[0], model=k[1], rho=t2[k]) for k in t2]).pivot(index='model', columns='column', values='rho')[COLS].round(3).to_string())


if __name__ == '__main__':
    main()
