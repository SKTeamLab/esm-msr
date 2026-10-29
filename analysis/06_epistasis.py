#!/usr/bin/env python
"""Epistasis: BENCHMARKS section 2 (tables 2.0-2.3, 2.5, 2.6), with the raw data shown.

* ``epistasis_hierarchy_val/test``  tables 2.1 / 2.2 as heat maps (the full suite of the document; the numbers are the
  document's because the suite itself lives with the training code), highlighting the ESM-MSR rows; ``epistasis_authoritative_metrics``
  is table 2.0.
* ``epistasis_naive_recomputed``  the headline column, naive rho (predicted against measured dddG over all doubles), and the RMSE of
  the predicted epistasis, recomputed from the per-variant predictions with library-level bootstrap intervals; the document's
  value is the cross.
* ``epistasis_scatter_dddG`` / ``epistasis_scatter_ddG_doubles``  the raw points: predicted against measured epistasis, and
  predicted against measured ddG of the double, for every model that scores a double itself.
* ``ddg_ranking``  table 2.3 (per-protein rho: all variants, doubles, dddG of doubles) with protein-level bootstrap intervals.
* ``lambda_scan``  table 2.6: rho of (1 - lam) * WT + lam * MT on multi-mutants, with mutation-level bootstrap bands.
* ``nobb_ablation``  table 2.5 (no predictions of the ablation runs are collected; drawn from the document).

Predicted epistasis: ESM-MSR ``epi_pred`` = MT(AB) - WT(AB); ME stage 2 = ME2(AB) - sum of the same seed's ME1 singles; ThermoMPNN-D =
TM-D(AB) - sum of ThermoMPNN singles; SPURS multi = multi(AB) - sum of SPURS singles.  Doubles: measured, both singles measured.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import SpearmanEngine, _Ranks, bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import epi_rows, out_dir, subframe  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import LABEL, MODELS, PER_PROTEIN  # noqa: E402
from msr_analysis.tables import clean_label, num, numeric, read_table  # noqa: E402

OUT = out_dir('06_epistasis')
TSU = ['tsuboyama_capped_val', 'tsuboyama_capped_test']
EPI_KEYS = ['msr:epi', 'msr:cdddg', 'me2:epi', 'tmpnnd:epi', 'spursm:epi']
DMS_D = ['GB1_Wu_2016_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
         'DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain']


# --------------------------------------------------------------------------------------------------
# 1. the document's hierarchy tables as heat maps
# --------------------------------------------------------------------------------------------------
def heat(ax, V, S, higher, title, hl=('ESM-MSR',), fs=7.2):
    cols = list(V.columns)
    Z = np.zeros(V.shape)
    for j, c in enumerate(cols):
        v = V[c].to_numpy(float)
        r = pd.Series(v).rank(pct=True, na_option='keep').to_numpy()
        Z[:, j] = r if higher[c] else 1 - r
    ax.imshow(Z, cmap='YlGnBu', vmin=0, vmax=1.15, aspect='auto')
    for i in range(V.shape[0]):
        for j, c in enumerate(cols):
            v, s = V.iloc[i, j], S.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}' + (f'\n±{s:.2f}' if np.isfinite(s) else ''), ha='center', va='center', fontsize=fs,
                        color='white' if Z[i, j] > .75 else 'black')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace('_', ' ') for c in cols], rotation=35, ha='right')
    ax.set_yticks(range(V.shape[0]))
    ax.set_yticklabels(V.index)
    for t in ax.get_yticklabels():
        if any(h in t.get_text() for h in hl):
            t.set_fontweight('bold')
    ax.set_title(title, loc='left')
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def hierarchy_figures():
    for tid, nm, ttl in (('2.1.1', 'val', 'Validation (8,494 doubles)'), ('2.2.1', 'test', 'Test (12,455 doubles)')):
        d = read_table(tid)
        d.iloc[:, 0] = [clean_label(x) for x in d.iloc[:, 0]]
        V, S = numeric(d)
        higher = {c: ('↓' not in c) for c in V.columns}
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        heat(ax, V, S, higher, f'{ttl}: the epistasis hierarchy (table {tid[:3]}); colour = rank within the column, darker = better; ± = seed sd')
        fig.tight_layout()
        style.save(fig, OUT / f'epistasis_hierarchy_{nm}')
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2))
    for ax, tid, ttl in ((axes[0], '2.0.1', 'Validation'), (axes[1], '2.0.2', 'Test')):
        d = read_table(tid)
        d.iloc[:, 0] = [clean_label(x) for x in d.iloc[:, 0]]
        V, S = numeric(d)
        heat(ax, V, S, {c: ('↓' not in c and 'bias' not in c) for c in V.columns}, f'{ttl}: authoritative epistasis metrics (table 2.0); bias is not ranked', fs=6.8)
    fig.tight_layout()
    style.save(fig, OUT / 'epistasis_authoritative_metrics')


# --------------------------------------------------------------------------------------------------
# 2. naive rho and RMSE recomputed
# --------------------------------------------------------------------------------------------------
def rmse_boot(fr, mask, key, B=500, seed=0):
    sp = fr.spine
    y = sp['dddG_ML'].to_numpy()
    cl = sp['cluster'].astype(str).to_numpy()
    ok = mask & np.isfinite(fr.preds[key]).all(axis=0)
    u, inv = np.unique(cl[ok], return_inverse=True)
    rng = np.random.default_rng(seed)
    e = np.stack([(p[ok] - y[ok]) for p in fr.preds[key]])
    pt = np.mean(np.sqrt((e ** 2).mean(1)))
    bias = e.mean()
    reps = []
    for _ in range(B):
        c = rng.multinomial(len(u), np.full(len(u), 1 / len(u))).astype(float)[inv]
        reps.append(np.mean(np.sqrt(((e ** 2) * c).sum(1) / c.sum())))
    return pt, ci(reps), float(bias)


def naive_recomputed():
    rows = []
    for ds in TSU + DMS_D:
        fr = load_frame(ds)
        m = epi_rows(fr)
        sub = subframe(fr, m, y=fr.spine['dddG_ML'].to_numpy(), keys=EPI_KEYS)
        kind = 'protein' if ds in TSU else 'mutation'
        B = 500 if ds in TSU else 100
        point, reps = bootstrap_rho({ds: sub}, EPI_KEYS, kind=kind, B=B, seed=3, per_protein=False)
        for k, (rho, per_seed, n, ng) in point[ds].items():
            lo, hi = ci(reps[ds][k])
            r = dict(dataset=ds, key=k, kind=kind, naive_rho=rho, lo=lo, hi=hi, n_doubles=n, n_libraries=ng, seed_sd=float(np.std(per_seed, ddof=1)))
            if ds in TSU:
                rm, (rlo, rhi), bias = rmse_boot(fr, m, k)
                r.update(global_err=rm, global_err_lo=rlo, global_err_hi=rhi, global_bias=bias)
            rows.append(r)
    R = pd.DataFrame(rows)
    # the document's values for the Tsuboyama rows
    doc = {}
    for tid, ds in (('2.1.1', TSU[0]), ('2.2.1', TSU[1])):
        d = read_table(tid)
        for _, r in d.iterrows():
            lab = clean_label(r.iloc[0])
            doc[(ds, lab)] = (num(r['naive ρ'])[0], num(r['global_err ↓'])[0])
    names = {'msr:epi': 'ESM-MSR (baseline)', 'me2:epi': 'MutateEverything stage 2', 'tmpnnd:epi': 'ThermoMPNN-D', 'spursm:epi': 'SPURS multi (multi-mutant model)'}
    R['doc_naive_rho'] = [doc.get((r.dataset, names.get(r.key, '')), (np.nan, np.nan))[0] for r in R.itertuples()]
    R['doc_global_err'] = [doc.get((r.dataset, names.get(r.key, '')), (np.nan, np.nan))[1] for r in R.itertuples()]
    R.to_csv(OUT / 'epistasis_naive_recomputed.csv', index=False, float_format='%.4f')
    print(R[R.dataset.isin(TSU)][['dataset', 'key', 'naive_rho', 'doc_naive_rho', 'global_err', 'doc_global_err', 'n_doubles']].round(3).to_string(index=False))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    groups = TSU + DMS_D
    glabels = ['Tsuboyama val', 'Tsuboyama test', 'GB1', 'GRB2 abund.', 'GRB2 bind.', 'DLG4 abund.', 'DLG4 bind.']
    keys = ['msr:epi', 'msr:cdddg', 'me2:epi', 'tmpnnd:epi', 'spursm:epi']
    table = {(r.dataset, r.key): r.naive_rho for r in R.itertuples()}
    cis = {(r.dataset, r.key): (r.lo, r.hi) for r in R.itertuples()}
    style.dot_panel(axes[0], groups, keys, table, cis, highlight=['msr:epi'], ylabel='naive ρ: predicted vs measured dddG (all doubles)', xlabels=glabels, size=6.5)
    for i, ds in enumerate(TSU):
        for j, k in enumerate(keys):
            v = R[(R.dataset == ds) & (R.key == k)].doc_naive_rho
            if len(v) and np.isfinite(v.iloc[0]):
                axes[0].plot(i + np.linspace(-.36, .36, len(keys))[j], v.iloc[0], marker='x', color='k', ms=5, mew=1)
    style.model_legend(axes[0], keys, loc='upper right', fontsize=7.5)
    axes[0].set_title("A  naive ρ (× = document's value); whiskers: library-level (Tsuboyama) / mutation-level (DMS) bootstrap", loc='left', fontsize=9)
    T = R[R.dataset.isin(TSU)]
    tbl = {(r.dataset, r.key): r.global_err for r in T.itertuples()}
    cs = {(r.dataset, r.key): (r.global_err_lo, r.global_err_hi) for r in T.itertuples()}
    style.dot_panel(axes[1], TSU, keys, tbl, cs, highlight=['msr:epi'], ylabel='RMSE of predicted epistasis (kcal/mol)', xlabels=['Tsuboyama val', 'Tsuboyama test'], size=6.5)
    axes[1].set_title('B  global error (lower is better)', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, OUT / 'epistasis_naive_recomputed')


# --------------------------------------------------------------------------------------------------
# 3. raw scatter
# --------------------------------------------------------------------------------------------------
def scatter_figures():
    sets = [('tsuboyama_capped_test', 'Tsuboyama test'), ('GB1_Wu_2016_binding_domain', 'GB1'), ('GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2 abund.')]
    keys = ['msr:epi', 'msr:cdddg', 'me2:epi', 'tmpnnd:epi', 'spursm:epi']
    fig, axes = plt.subplots(len(sets), len(keys), figsize=(3.0 * len(keys), 3.0 * len(sets)))
    for r, (ds, lab) in enumerate(sets):
        fr = load_frame(ds)
        m = epi_rows(fr)
        y = fr.spine['dddG_ML'].to_numpy()
        for c, k in enumerate(keys):
            ax = axes[r, c]
            p = fr.preds[k][0]
            ok = m & np.isfinite(p)
            style.density_scatter(ax, p[ok], y[ok], s=4, cmap='viridis', line=True, max_points=40000)
            ax.axhline(0, color='#bbb', lw=.6); ax.axvline(0, color='#bbb', lw=.6)
            if r == 0:
                ax.set_title(MODELS[k]['label'], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f'{lab}\nmeasured dddG (kcal/mol)')
            if r == len(sets) - 1:
                ax.set_xlabel('predicted epistasis (kcal/mol)')
    fig.suptitle('Predicted vs measured epistasis of double mutants (seed 1; ρ in each panel; dashed = y = x)', y=1.0, fontsize=10)
    fig.tight_layout()
    style.save(fig, OUT / 'epistasis_scatter_dddG')
    # ddG of the double
    fr = load_frame('tsuboyama_capped_test')
    m = epi_rows(fr)
    y = fr.spine['y'].to_numpy()
    cols = [('msr:wt', 'ESM-MSR WT head (additive)'), ('msr:comb', 'ESM-MSR combined'), ('me2', 'MutateEverything st2'), ('tmpnnd', 'ThermoMPNN + D'), ('spursm', 'SPURS multi')]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.3), sharey=True)
    for ax, (k, t) in zip(axes, cols):
        p = fr.preds[k][0]
        ok = m & np.isfinite(p)
        style.density_scatter(ax, p[ok], y[ok], s=4, cmap='viridis', line=True, max_points=40000)
        ax.set_title(t, fontsize=8.5); ax.set_xlabel('predicted ddG of the double')
    axes[0].set_ylabel('measured ddG of the double (kcal/mol)')
    fig.suptitle('Tsuboyama test doubles with both singles measured (n = %d, seed 1)' % int(m.sum()), y=1.02, fontsize=10)
    fig.tight_layout()
    style.save(fig, OUT / 'epistasis_scatter_ddG_doubles')


# --------------------------------------------------------------------------------------------------
# 4. table 2.3: ddG ranking
# --------------------------------------------------------------------------------------------------
def ddg_ranking():
    ms = ['msr:wt', 'msr:comb', 'me1', 'me2', 'tmpnn', 'tmpnnd']
    rows = []
    for ds, sp_ in zip(TSU, ('val', 'test')):
        fr = load_frame(ds)
        sp = fr.spine
        dbl = (sp['n_mut'] == 2).to_numpy() & sp['use'].to_numpy()
        e = (dbl & np.isfinite(sp['dddG_ML'].to_numpy()))      # table 2.3 uses every double with a measured dddG
        sets = {'all variants': (fr, ms), 'doubles': (subframe(fr, dbl, keys=ms), ms),
                'dddG of doubles': (subframe(fr, e, y=sp['dddG_ML'].to_numpy(), keys=['msr:cdddg', 'me2:epi', 'tmpnnd:epi', 'spursm:epi'],
                                             rename={'msr:cdddg': 'msr:comb', 'me2:epi': 'me2', 'tmpnnd:epi': 'tmpnnd', 'spursm:epi': 'spursm'}),
                                    ['msr:comb', 'me2', 'tmpnnd', 'spursm'])}
        for nm, (f, keys) in sets.items():
            point, reps = bootstrap_rho({ds: f}, keys, kind='protein', B=1000, seed=4, per_protein=True)
            for k, (rho, per_seed, n, ng) in point[ds].items():
                lo, hi = ci(reps[ds][k])
                rows.append(dict(split=sp_, metric=nm, model=k, rho=rho, lo=lo, hi=hi, seed_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan, n=n, n_proteins=ng))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'ddg_ranking_recomputed.csv', index=False, float_format='%.4f')
    groups = [(s, m) for s in ('val', 'test') for m in ('all variants', 'doubles', 'dddG of doubles')]
    table = {(g, k): v for g, k, v in ((f'{r.split} {r.metric}', r.model, r.rho) for r in R.itertuples())}
    cis = {(f'{r.split} {r.metric}', r.model): (r.lo, r.hi) for r in R.itertuples()}
    labels = [f'{s}\n{m}' for s, m in groups]
    gnames = [f'{s} {m}' for s, m in groups]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    style.dot_panel(ax, gnames, ms + ['spursm'], table, cis, highlight=['msr:comb'], ylabel='mean per-protein Spearman ρ', xlabels=labels, rot=0, size=6.5)
    style.model_legend(ax, ms + ['spursm'], loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=7, fontsize=7.5, columnspacing=1.0, handletextpad=0.3)
    ax.set_title('Table 2.3: ddG ranking; whiskers 95% protein-level bootstrap.  The WT head is additive, so its dddG is zero by construction (not shown)', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, OUT / 'ddg_ranking')
    print(R.pivot_table(index=['split', 'metric'], columns='model', values='rho').round(3).to_string())


# --------------------------------------------------------------------------------------------------
# 5. table 2.6: how much predicted epistasis to keep
# --------------------------------------------------------------------------------------------------
LAMBDAS = np.round(np.arange(0, 1.0001, 0.05), 2)
SCAN_SETS = [('tsuboyama_capped_test', 'Tsuboyama test'), ('ptmul', 'PTMUL'), ('ptmuld', 'PTMUL-D'), ('GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2 abund.'),
             ('DLG4_HUMAN_Faure_2021_binding_domain', 'DLG4 bind.'), ('GB1_Wu_2016_binding_domain', 'GB1')]


def lambda_scan():
    rows, curves = [], {}
    for fam, tag in (('msrrel', 'released (1 checkpoint; the document)'), ('msr', 'baseline triplicate')):
        for ds, lab in SCAN_SETS:
            fr = load_frame(ds)
            sp = fr.spine
            m = (sp['n_mut'] >= 2).to_numpy() & sp['use'].to_numpy()
            A, M = fr.preds[f'{fam}:wt'][:, m], fr.preds[f'{fam}:mt'][:, m]
            ok = np.isfinite(A).all(0) & np.isfinite(M).all(0)
            A, M, y = A[:, ok], M[:, ok], sp['y'].to_numpy()[m][ok]
            n = len(y)
            if ds in PER_PROTEIN:       # Tsuboyama: the mean over libraries of the per-library rho, as in the document
                g = pd.factorize(sp['group'].to_numpy()[m][ok])[0]
            else:
                g = np.zeros(n, np.int64)
            G = int(g.max()) + 1
            red = (lambda r: np.nanmean(r)) if G > 1 else (lambda r: r[0])
            eng = SpearmanEngine(y, g, G)
            ranks = [[eng.prepare((1 - l) * a + l * mm) for l in LAMBDAS] for a, mm in zip(A, M)]
            w1 = np.ones(n)
            pt = np.array([np.mean([red(eng.rho(ranks[s][i], w1)) for s in range(len(A))]) for i in range(len(LAMBDAS))])
            B = 200 if n < 20000 else 60
            rng = np.random.default_rng(1)
            reps = np.empty((B, len(LAMBDAS)))
            for b in range(B):
                w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
                ry = eng.ry.ranks(w)
                reps[b] = [np.mean([red(eng.rho(ranks[s][i], w, ry)) for s in range(len(A))]) for i in range(len(LAMBDAS))]
            curves[(fam, ds)] = (pt, reps)
            i0, i5, i1, ib = 0, int(np.where(LAMBDAS == 0.5)[0][0]), len(LAMBDAS) - 1, int(pt.argmax())
            d = reps[:, i5] - reps[:, ib]
            rows.append(dict(checkpoint=tag, dataset=ds, n=n, rho_lam0=pt[i0], rho_lam05=pt[i5], rho_lam1=pt[i1], best_lambda=LAMBDAS[ib], rho_best=pt[ib],
                             gap_05_to_best=pt[ib] - pt[i5], gap_lo=float(np.quantile(reps[:, ib] - reps[:, i5], .025)), gap_hi=float(np.quantile(reps[:, ib] - reps[:, i5], .975))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'lambda_scan.csv', index=False, float_format='%.4f')
    print(R.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, len(SCAN_SETS), figsize=(3.0 * len(SCAN_SETS), 3.2), sharey=False)
    for ax, (ds, lab) in zip(axes, SCAN_SETS):
        for fam, col, ls in (('msr', 'k', '-'), ('msrrel', '#888888', '--')):
            pt, reps = curves[(fam, ds)]
            lo, hi = np.quantile(reps, [.025, .975], axis=0)
            ax.fill_between(LAMBDAS, lo, hi, color=col, alpha=.15, lw=0)
            ax.plot(LAMBDAS, pt, color=col, ls=ls, lw=1.6, label='baseline triplicate' if fam == 'msr' else 'released checkpoint')
            ax.plot(LAMBDAS[pt.argmax()], pt.max(), 'o', color=col, ms=4)
        ax.axvline(0.5, color='#c0392b', lw=.8, ls=':')
        ax.set_title(lab, fontsize=9); ax.set_xlabel('λ  (fraction of MT epistasis kept)')
    axes[0].set_ylabel('ρ on multi-mutants'); axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    style.save(fig, OUT / 'lambda_scan')


# --------------------------------------------------------------------------------------------------
# 6. table 2.5
# --------------------------------------------------------------------------------------------------
def nobb():
    d = read_table('2.5.1')
    V, S = numeric(d)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(V.shape[1]); w = .8 / len(V)
    cols = {'MutateEverything stage 2': MODELS['me2']['color'], 'MutateEverything stage 1': MODELS['me1']['color'], 'ThermoMPNN': MODELS['tmpnn']['color']}
    for i, (name, row) in enumerate(V.iterrows()):
        ax.bar(x + (i - 1) * w, row.values, w, color=cols.get(name, '#999'), label=name)
    ax.axhline(0, color='k', lw=.7)
    ax.set_xticks(x); ax.set_xticklabels(V.columns, rotation=20, ha='right'); ax.set_ylabel('change when the mutant-background libraries are dropped')
    ax.legend(fontsize=8); ax.set_title('Table 2.5: mutant-background ablation (document values; the runs are not collected)', loc='left', fontsize=9)
    fig.tight_layout()
    style.save(fig, OUT / 'nobb_ablation')


if __name__ == '__main__':
    hierarchy_figures()
    naive_recomputed()
    scatter_figures()
    ddg_ranking()
    lambda_scan()
    nobb()
