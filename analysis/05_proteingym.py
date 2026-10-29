#!/usr/bin/env python
"""ProteinGym (213 DMS): BENCHMARKS tables 1.4 and 3, with intervals from a bootstrap over assays.

Per-DMS Spearman rho tables come from ``baselines/pgym_per_dms`` (the tables each BENCHMARKS row is built from); the new ESM-MSR
baseline triplicate and the retrained SPURS pair are scored here from their per-assay prediction files
(``msr_predictions/baseline_w2a4_m16a16/seed*/sigma1/proteingym``, ``baselines/retrained/spurs/predictions/seed*/proteingym``).
Summaries follow the document: the category mean is a plain mean over the DMS of a category, **PG avg** is the mean of
the five category means, **DMS mean** the mean over the 213 assays.  The bootstrap resamples assays *within* each category (so the
category sizes stay fixed) with the same draws for every model, so differences between models are paired.

Outputs (``results/05_proteingym``): ``pg_summary.csv`` (value and 95% interval per model and column), ``pg_per_dms.csv`` (the
raw per-DMS rhos), ``pg_sigma_paired.csv`` (paired shift of each LoRA scale against the released checkpoint),
``pg_categories.png`` (dot panels), ``pg_per_dms_scatter.png`` (raw per-assay rho, ESM-MSR against ProSST and against its
own zero-shot readout, and the per-assay effect of the LoRA scale).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.registry import BASE, CACHE, MSR, PGYM, REFERENCE, RETRAINED  # noqa: E402

OUT = out_dir('05_proteingym')
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']
CAT_LABEL = {'Activity': 'Activity (43)', 'Binding': 'Binding (13)', 'Expression': 'Expression (18)', 'OrganismalFitness': 'Org. fitness (73)',
             'Stability': 'Stability (66)', 'PG avg': 'PG avg', 'DMS mean': 'DMS mean'}
DROP = {'BRCA1_HUMAN_Findlay_2018', 'BRCA2_HUMAN_Erwood_2022_HEK293T', 'POLG_CXB3N_Mattenberger_2021', 'SCN5A_HUMAN_Glazer_2019'}
B = 2000


def score_files(folder_fmt, cols, seeds=(1, 2, 3)):
    """per-DMS rho (seed mean) of each column in per-assay files"""
    out = {c: {} for c in cols}
    for sd in seeds:
        for f in sorted(glob.glob(folder_fmt.format(seed=sd) + '/*.csv')):
            name = Path(f).stem
            if name in ('summary',):
                continue
            d = pd.read_csv(f, usecols=lambda c: c in cols + ['DMS_score'])
            for c in cols:
                if c in d:
                    out[c].setdefault(name, []).append(spearmanr(d[c], d['DMS_score'], nan_policy='omit')[0])
    return {c: {k: float(np.mean(v)) for k, v in dd.items()} for c, dd in out.items()}


def extra_models():
    cache = CACHE / 'pg_extra.csv'
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    msr = score_files(str(MSR / 'baseline_w2a4_m16a16' / 'seed{seed}' / 'sigma1' / 'proteingym'), ['combined_pred', 'wt_lora_pred'])
    sp = score_files(str(RETRAINED / 'spurs' / 'predictions' / 'seed{seed}' / 'proteingym'), ['SPURS_single_additive', 'SPURS_multi'])
    df = pd.DataFrame({'ESM-MSR (baseline) combined': msr['combined_pred'], 'ESM-MSR (baseline) WT head': msr['wt_lora_pred'],
                       'SPURS (retrained)': sp['SPURS_single_additive'], 'SPURS multi (retrained)': sp['SPURS_multi']})
    df.to_csv(cache)
    return df


def build():
    a = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
    dc = pd.read_csv(PGYM / 'pgym_dc8_per_dms.csv').set_index('DMS_id')
    pub = pd.read_csv(PGYM / 'pgym_public_per_dms.csv').set_index('DMS_id')
    ro = pd.read_csv(PGYM / 'pgym_rosetta_singles_per_dms.csv').set_index('DMS_id')
    X = extra_models()
    cat = a['category']
    T = pd.DataFrame(index=a.index)
    T['category'] = cat
    T['ESM-MSR (baseline) combined'] = X['ESM-MSR (baseline) combined']
    T['ESM-MSR (baseline) WT head'] = X['ESM-MSR (baseline) WT head']
    T['ESM-MSR (released) combined σ=1'] = a['ESM-MSR full']
    for s in ('1.0', '0.75', '0.5', '0.25', '0'):
        T[f'ESM-MSR (released) WT head σ={float(s):g}'] = a[f'ESM-MSR WT eps={s}']
    for s in ('1.0', '0.75', '0.5', '0.25'):
        T[f'ESM-MSR (devel) σ={float(s):g}'] = dc[f'dc8 eps={s} mean']
    T['SPURS (retrained)'] = X['SPURS (retrained)']
    T['SPURS multi (retrained)'] = X['SPURS multi (retrained)']
    T['SPURS (published, not held out)'] = a['SPURS single-additive']
    T['SPURS multi (published, not held out)'] = a['SPURS multi']
    T['MutateEverything st1'] = a['MutateEverything single']
    T['ThermoMPNN'] = a['ThermoMPNN']
    T['ProSST (published scores)'] = pub['ProSST-2048 rho_full']
    T['ProteinMPNN (ours)'] = a['ProteinMPNN']
    T['ProteinMPNN (published recipe)'] = pub['ProteinMPNN rho_full']
    T['ESM3-open (published recipe)'] = pub['ESM3 rho_full']
    T['Rosetta (singles only)'] = ro['rho']
    T = T[~T.index.isin(DROP)]
    return T


def summarize_cols(T, idx_by_cat):
    """value and bootstrap draws for the seven summary columns of one model; idx_by_cat[c] is (B, n_c) index arrays."""
    res = {}
    for m in T.columns[1:]:
        v = T[m]
        point = {c: float(v[T.category == c].mean()) for c in CATS}
        point['PG avg'] = float(np.mean([point[c] for c in CATS]))
        point['DMS mean'] = float(v.mean())
        reps = {c: [] for c in CATS}
        for c in CATS:
            vals = v[T.category == c].to_numpy()
            reps[c] = vals[idx_by_cat[c]].mean(axis=1) if np.isfinite(vals).all() else np.full(B, np.nan)
        reps['PG avg'] = np.mean([reps[c] for c in CATS], axis=0)
        n = np.array([(T.category == c).sum() for c in CATS])
        reps['DMS mean'] = sum(reps[c] * k for c, k in zip(CATS, n)) / n.sum()
        res[m] = (point, reps)
    return res


def main():
    T = build()
    T.to_csv(OUT / 'pg_per_dms.csv', float_format='%.4f')
    # keep assays where every model of the main figure has a score; ME double / ESM-MSR full have a few gaps
    full = T.drop(columns=['MutateEverything st1', 'Rosetta (singles only)']).dropna()     # Rosetta lacks one assay, ME a few
    print(len(T), 'assays;', len(full), 'with all main models;', T.isna().sum()[T.isna().sum() > 0].to_dict())
    T = T.loc[full.index]
    rng = np.random.default_rng(2)
    idx_by_cat = {}
    for c in CATS:
        n = int((T.category == c).sum())
        idx_by_cat[c] = rng.integers(0, n, (B, n))
    S = summarize_cols(T, idx_by_cat)
    rows = []
    cols = CATS + ['PG avg', 'DMS mean']
    for m, (pt, reps) in S.items():
        for c in cols:
            lo, hi = np.nanquantile(reps[c], [.025, .975]) if np.isfinite(reps[c]).all() else (np.nan, np.nan)
            rows.append(dict(model=m, column=c, value=pt[c], lo=lo, hi=hi, n_dms=int((T.category == c).sum()) if c in CATS else len(T)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'pg_summary.csv', index=False, float_format='%.4f')
    print(R[R.column.isin(['PG avg', 'Stability', 'DMS mean'])].pivot(index='model', columns='column', values='value').round(3).sort_values('PG avg', ascending=False).to_string())

    # ---- paired shift of each LoRA scale against the released checkpoint (T3.2) -----------------------
    rel = S['ESM-MSR (released) combined σ=1']
    prow = []
    for s in ('1', '0.75', '0.5', '0.25'):
        m = f'ESM-MSR (devel) σ={s}'
        d = T[m] - T['ESM-MSR (released) combined σ=1']
        dd = S[m][1]['DMS mean'] - rel[1]['DMS mean']
        prow.append(dict(setting=f'devel σ={s}', pg_avg=S[m][0]['PG avg'], paired_delta_mean_over_dms=float(d.mean()), delta_lo=float(np.quantile(dd, .025)),
                         delta_hi=float(np.quantile(dd, .975)), better_on=int((d > 0).sum()), of=len(d)))
    P = pd.DataFrame(prow)
    P.to_csv(OUT / 'pg_sigma_paired.csv', index=False, float_format='%.4f')
    print(P.round(4).to_string(index=False))

    # ---- figure: category dot panels --------------------------------------------------------------------
    top = ['ESM-MSR (devel) σ=1', 'ESM-MSR (devel) σ=0.75', 'ESM-MSR (devel) σ=0.5', 'ESM-MSR (devel) σ=0.25', 'ESM-MSR (released) WT head σ=0',
           'ProSST (published scores)', 'ProteinMPNN (published recipe)', 'ESM3-open (published recipe)']
    bot = ['ESM-MSR (baseline) WT head', 'ESM-MSR (baseline) combined', 'SPURS (retrained)', 'SPURS multi (retrained)', 'MutateEverything st1', 'ThermoMPNN',
           'SPURS (published, not held out)']
    sty = {
        'ESM-MSR (devel) σ=1': dict(color='#000000', marker='D', fill=True), 'ESM-MSR (devel) σ=0.75': dict(color='#333333', marker='D', fill=False),
        'ESM-MSR (devel) σ=0.5': dict(color='#666666', marker='D', fill=False), 'ESM-MSR (devel) σ=0.25': dict(color='#999999', marker='D', fill=False),
        'ESM-MSR (released) WT head σ=0': dict(color='#7f8c8d', marker='^', fill=True), 'ProSST (published scores)': dict(color='#8e44ad', marker='X', fill=True),
        'ProteinMPNN (published recipe)': dict(color='#34495e', marker='d', fill=True), 'ESM3-open (published recipe)': dict(color='#e74c3c', marker='o', fill=True),
        'ESM-MSR (baseline) WT head': dict(color='#000000', marker='o', fill=False), 'ESM-MSR (baseline) combined': dict(color='#000000', marker='D', fill=True),
        'SPURS (retrained)': dict(color='#d35400', marker='s', fill=True), 'SPURS multi (retrained)': dict(color='#e59866', marker='s', fill=False),
        'MutateEverything st1': dict(color='#2ecc71', marker='h', fill=True), 'ThermoMPNN': dict(color='#3498db', marker='p', fill=True),
        'SPURS (published, not held out)': dict(color='#d35400', marker='s', fill=False)}
    from msr_analysis.registry import MODELS
    for k, v in sty.items():
        MODELS[k] = dict(label=k, group='pg', **v)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.6), sharex=True)
    for ax, models, ttl in ((axes[0], top, 'ESM-MSR at lower LoRA scale σ, against zero-shot models'), (axes[1], bot, 'ESM-MSR against the models retrained on the capped split (σ = 1)')):
        table = {(c, m): float(R[(R.model == m) & (R.column == c)].value.iloc[0]) for m in models for c in cols}
        ci = {(c, m): (float(R[(R.model == m) & (R.column == c)].lo.iloc[0]), float(R[(R.model == m) & (R.column == c)].hi.iloc[0])) for m in models for c in cols}
        style.dot_panel(ax, cols, models, table, ci, highlight=['ESM-MSR (devel) σ=1', 'ESM-MSR (baseline) combined'], ylabel='Spearman ρ (mean over assays)',
                        xlabels=[CAT_LABEL[c] for c in cols], rot=0, size=6)
        style.model_legend(ax, models, loc='upper left', ncol=2, fontsize=8)
        ax.set_title(ttl, loc='left')
        ax.set_ylim(0.1, 1.0)
    fig.tight_layout()
    style.save(fig, OUT / 'pg_categories')

    # ---- figure: every row of tables 1.4 / 3.1 as a heat map -------------------------------------------------------
    Z = R.pivot(index='model', columns='column', values='value')[cols]
    Z = Z.loc[Z['PG avg'].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.6, 0.29 * len(Z) + 1.5))
    rk = Z.rank(pct=True)
    ax.imshow(rk.to_numpy(), cmap='YlGnBu', vmin=0, vmax=1.15, aspect='auto')
    for i in range(len(Z)):
        for j, c in enumerate(cols):
            lo, hi = R[(R.model == Z.index[i]) & (R.column == c)][['lo', 'hi']].iloc[0]
            ax.text(j, i, f'{Z.iloc[i, j]:.3f}', ha='center', va='center', fontsize=7.2, color='white' if rk.iloc[i, j] > .75 else 'black')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([CAT_LABEL[c] for c in cols], rotation=30, ha='right')
    ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index, fontsize=8)
    ax.set_title('ProteinGym, every model (tables 1.4 and 3): rho by category, PG avg and DMS mean; colour = rank in column', loc='left', fontsize=9)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    fig.tight_layout()
    style.save(fig, OUT / 'pg_table_heatmap')

    # ---- figure: per-assay raw data -------------------------------------------------------------------
    colors = dict(zip(CATS, ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (xm, ym, xl, yl) in zip(axes[:2], [('ProSST (published scores)', 'ESM-MSR (devel) σ=1', 'ProSST (zero-shot)', 'ESM-MSR (devel) σ=1'),
                                               ('ESM-MSR (released) WT head σ=0', 'ESM-MSR (devel) σ=1', 'σ = 0 (LoRA off, zero-shot readout)', 'ESM-MSR (devel) σ=1')]):
        for c in CATS:
            s = T[T.category == c]
            ax.scatter(s[xm], s[ym], s=16, color=colors[c], alpha=.75, lw=0, label=CAT_LABEL[c])
        lim = [-0.2, 1.0]
        ax.plot(lim, lim, color='#999', lw=.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'ρ, {xl}'); ax.set_ylabel(f'ρ, {yl}')
        ax.text(.03, .97, f'ESM-MSR higher on {int((T[ym] > T[xm]).sum())} of {len(T)} assays', transform=ax.transAxes, va='top', fontsize=8.5)
    axes[0].legend(fontsize=7.5, loc='lower right')
    ax = axes[2]
    for i, s in enumerate(('0.75', '0.5', '0.25')):
        d = (T[f'ESM-MSR (devel) σ={s}'] - T['ESM-MSR (devel) σ=1'])
        for c in CATS:
            dc_ = d[T.category == c]
            ax.scatter(np.full(len(dc_), i) + np.random.default_rng(i).uniform(-.28, .28, len(dc_)), dc_, s=10, color=colors[c], alpha=.7, lw=0)
        ax.plot([i - .35, i + .35], [d.mean()] * 2, color='k', lw=2)
    ax.axhline(0, color='#999', lw=.8, ls=':')
    ax.set_xticks(range(3)); ax.set_xticklabels(['σ=0.75', 'σ=0.5', 'σ=0.25']); ax.set_ylabel('per-assay ρ(σ) − ρ(σ=1), devel seeds')
    ax.set_title('effect of lowering the LoRA scale, assay by assay')
    fig.tight_layout()
    style.save(fig, OUT / 'pg_per_dms_scatter')


if __name__ == '__main__':
    main()
