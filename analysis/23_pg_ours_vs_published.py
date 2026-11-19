#!/usr/bin/env python
"""Our runs against ProteinGym's published scores of the same zero-shot models, on every ProteinGym-related figure.

ESM3-small-open is our sigma = 0 read of the ESM-MSR checkpoint (adapters off, unmasked: WT-view on singles assays, Dual-view on the assays with multi-mutants); the published ESM3 is the score ProteinGym ships
(``baselines/pgym_per_dms/pgym_public_per_dms.csv``, column ``ESM3 rho_full``).  ProteinMPNN is our three runs (``proteinmpnn_020_unmasked_fast_ensemble3_per_dms.csv``; per variant in ``baselines/public/proteinmpnn/proteingym``) against ProteinGym's own.  ProSST is the published
ProSST is the published score in both until our run (harness job on the cluster) is collected.  All on the held-out assays (the four-part filter of ``msr_analysis.heldout``), 173 of the 213 assays.

  a  the columns of Figure 3b / ``pg_categories``: mean per-assay Spearman rho per ProteinGym category and the average of the five (ours solid, published hatched), 95% bootstrap over assays (paired draws)
  b  the per-assay scatter of ``pg_per_dms_scatter``: ours against published, one dot per assay coloured by category
  c  the Activity screening curves of ``pg_activity`` (hurt rate of the K picks, single mutants, lower is better): ESM3-small-open and ProteinMPNN, ours against published (our ProteinMPNN: three per-variant runs)

    python analysis/23_pg_ours_vs_published.py [--B 1000]
Writes ``results/23_pg_ours_vs_published/pg_ours_vs_published.{png,pdf,csv}``.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from msr_analysis import models as M  # noqa: E402
from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402
from msr_analysis.heldout import excluded  # noqa: E402
from msr_analysis.registry import PGYM, RESULTS  # noqa: E402

OUT = out_dir('23_pg_ours_vs_published')
CATS = ['Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness']
CAT_LABEL = {'Stability': 'Stability', 'Expression': 'Expression', 'Activity': 'Activity', 'Binding': 'Binding', 'OrganismalFitness': 'Org. fitness', 'PG avg': 'ProteinGym\naverage'}
MODELS = [('ESM3-small-open', 'ESM3 rho_full'), ('ProteinMPNN', 'ProteinMPNN rho_full')]            # (our column of pg_per_dms.csv, the published column)
_src = (RESULTS / '05_proteingym' / 'pg_zero_shot_sources.txt')
if _src.exists() and _src.read_text().startswith('ProSST: ours'):                                       # our ProSST run on every assay (harness job; ``05_proteingym`` uses it once it is there)
    MODELS.append(('ProSST', 'ProSST-2048 rho_full'))


def per_assay():
    """One row per held-out assay: category, our rho and the published rho of each model."""
    T = pd.read_csv(RESULTS / '05_proteingym' / 'pg_per_dms.csv').set_index('DMS_id')
    P = pd.read_csv(PGYM / 'pgym_public_per_dms.csv').set_index('DMS_id')
    T = T[~T.index.isin(excluded('proteingym'))]
    out = T[['category']].copy()
    for ours, theirs in MODELS:
        out[f'{ours} ours'] = T[ours]
        out[f'{ours} published'] = P[theirs].reindex(T.index)
    return out


def category_table(D, B):
    """Mean rho per category / average, ours and published, with a paired bootstrap over assays within the category (the draws are shared by the two versions)."""
    rng = np.random.default_rng(2)
    idx = {c: rng.integers(0, int((D.category == c).sum()), (B, int((D.category == c).sum()))) for c in CATS}
    rows = []
    for name, _ in MODELS:
        reps = {}
        for ver in ('ours', 'published'):
            v = D[f'{name} {ver}']
            point = {c: float(np.nanmean(v[D.category == c])) for c in CATS}
            point['PG avg'] = float(np.mean([point[c] for c in CATS]))
            rr = {c: np.nanmean(v[D.category == c].to_numpy(float)[idx[c]], axis=1) for c in CATS}
            rr['PG avg'] = np.mean([rr[c] for c in CATS], axis=0)
            reps[ver] = (point, rr)
        for c in CATS + ['PG avg']:
            d = reps['ours'][1][c] - reps['published'][1][c]
            for ver in ('ours', 'published'):
                lo, hi = np.nanquantile(reps[ver][1][c], [0.025, 0.975])
                rows.append(dict(model=name, version=ver, column=c, value=reps[ver][0][c], lo=lo, hi=hi, delta_ours_minus_published=reps['ours'][0][c] - reps['published'][0][c],
                                 delta_lo=np.nanquantile(d, 0.025), delta_hi=np.nanquantile(d, 0.975), n_assays=int((D.category == c).sum()) if c in CATS else len(D)))
    return pd.DataFrame(rows)


def activity_curves(B):
    """Hurt-rate curves (K picks per assay, 0.5 assay SD definition, all held-out Activity assays) of ESM3-small-open and ProteinMPNN, ours and published: {(model, version): (table, n assays)}."""
    spec = importlib.util.spec_from_file_location('pg_act', HERE / 'pg_activity' / '02_activity_screening.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['pg_act'] = mod
    spec.loader.exec_module(mod)
    fr, info = mod.load_assays()
    out = {}
    for ver, (e3, pm) in (('ours', ('msrrel@0:wt', 'pmpnn')), ('published', ('esm3_pub', 'pmpnn_pub'))):
        f = Frame(fr.name, fr.spine, {**fr.preds, 'msrrel@0:wt': fr.preds[e3], 'pmpnn': fr.preds[pm]})           # the families 'esm3' and 'pmpnn' read these keys
        t, n = mod.curves(f, info, set(info.index), ['esm3', 'pmpnn'], B)['sd']
        for model, fam in (('ESM3-small-open', 'esm3'), ('ProteinMPNN', 'pmpnn')):
            out[(model, ver)] = (t[(t.family == fam) & (t.metric == 'hit_rate')].sort_values('K'), n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000)
    a = ap.parse_args()
    D = per_assay()
    C = category_table(D, a.B)
    C.to_csv(OUT / 'pg_ours_vs_published.csv', index=False, float_format='%.4f')
    act = activity_curves(a.B)

    fig = plt.figure(figsize=(15.0 + 3.6 * (len(MODELS) - 2), 9.6))
    gs = fig.add_gridspec(2, 1 + len(MODELS), height_ratios=[1, 1], width_ratios=[1.5] + [1] * len(MODELS), hspace=0.38, wspace=0.28)
    axa = fig.add_subplot(gs[0, :])
    cols = CATS + ['PG avg']
    w = 0.19
    for j, (name, _) in enumerate(MODELS):
        col = M.color(name)
        for i, ver in enumerate(('ours', 'published')):
            x = np.arange(len(cols)) + (j * 2 + i - 1.5) * w
            r = C[(C.model == name) & (C.version == ver)].set_index('column').loc[cols]
            axa.bar(x, r.value, w * 0.92, color=col if ver == 'ours' else M.tint(col, 0.55), edgecolor='black' if ver == 'ours' else col, hatch=None if ver == 'ours' else '////', lw=0.8, zorder=3)
            axa.errorbar(x, r.value, yerr=[r.value - r.lo, r.hi - r.value], fmt='none', ecolor=style.RING, elinewidth=1.2, capsize=2.2, zorder=5)
            for xx, v, hi in zip(x, r.value, r.hi):
                axa.text(xx, hi + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)
    axa.set_xticks(range(len(cols)))
    n_ = C[C.model == MODELS[0][0]].drop_duplicates('column').set_index('column').n_assays
    axa.set_xticklabels([f'{CAT_LABEL[c]}\n({n_[c]} assays)' for c in cols], fontsize=9)
    axa.set_ylabel('mean per-assay Spearman ρ')
    axa.set_ylim(0, 0.88)
    axa.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    axa.legend(handles=[Patch(fc=M.color('ESM3-small-open'), ec='black', label='ESM3-small-open, ours (σ = 0 read of the ESM-MSR checkpoint)'),
                        Patch(fc=M.tint(M.color('ESM3-small-open'), 0.55), ec=M.color('ESM3-small-open'), hatch='////', label='ESM3, ProteinGym published'),
                        Patch(fc=M.color('ProteinMPNN'), ec='black', label='ProteinMPNN, ours'),
                        Patch(fc=M.tint(M.color('ProteinMPNN'), 0.55), ec=M.color('ProteinMPNN'), hatch='////', label='ProteinMPNN, ProteinGym published')], ncol=2, loc='upper right', fontsize=8.5, frameon=True, facecolor='white', edgecolor='#cccccc')
    axa.set_title('a   Figure 3b columns: mean ρ per ProteinGym category (held-out assays; 95% bootstrap over assays)', loc='left', fontsize=10, fontweight='bold')

    for k, (name, _) in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, 1 + k])
        for c in CATS:
            s = D[D.category == c]
            ax.scatter(s[f'{name} published'], s[f'{name} ours'], s=16, color=style.CATEGORY_COLOR[c], alpha=0.8, lw=0, label=CAT_LABEL[c])
        ax.plot([-0.2, 1], [-0.2, 1], color='#999', lw=0.8, ls='--')
        ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1)
        ok = D[[f'{name} ours', f'{name} published']].dropna()
        ax.text(0.03, 0.97, f'ρ between the two = {spearmanr(ok.iloc[:, 0], ok.iloc[:, 1])[0]:.2f}\nours higher on {int((ok.iloc[:, 0] > ok.iloc[:, 1]).sum())} of {len(ok)} assays', transform=ax.transAxes, va='top', fontsize=8.5)
        ax.set_xlabel(f'ρ, {name} published'); ax.set_ylabel(f'ρ, {name} ours')
        ax.set_title(f'{"cd"[k]}   {name}: per assay', loc='left', fontsize=10, fontweight='bold')
        if k == 0:
            ax.legend(fontsize=7.6, loc='lower right')

    axc = fig.add_subplot(gs[1, 0])
    for (name, ver), (t, n) in act.items():
        col = M.color(name)
        axc.plot(t.K, t.value, color=col if ver == 'ours' else M.tint(col, 0.35), ls='-' if ver == 'ours' else '--', lw=2.2, marker='o', ms=4, label=f'{name}, {ver}')
        axc.fill_between(t.K, t.lo, t.hi, color=col, alpha=0.10 if ver == 'ours' else 0.05, lw=0)
    axc.set_xscale('log'); axc.set_xlabel('variants screened per assay (K)'); axc.set_ylabel('fraction of the K picks that hurt (lower is better)')
    ks = sorted(next(iter(act.values()))[0].K.unique())
    axc.set_xticks(ks); axc.set_xticklabels(ks)
    axc.legend(fontsize=8)
    axc.set_title(f'b   Activity screening ({next(iter(act.values()))[1]} held-out assays, single mutants)', loc='left', fontsize=10, fontweight='bold')
    fig.text(0.01, 0.005, 'Ours: ESM3-small-open = sigma-0 read of the ESM-MSR checkpoint (WT-view on singles assays, Dual-view on assays with multi-mutants); ProteinMPNN = our three runs. Published: the per-assay scores ProteinGym ships. '
             'ProSST is published in both until our run (harness job) is collected. Held-out = homology-filtered assays (173 of 213).', fontsize=7.8, color=style.INK_NOTE)
    style.save(fig, OUT / 'pg_ours_vs_published')
    print(C[C.column.isin(['PG avg', 'Activity'])].round(3).to_string(index=False))
    print('wrote', OUT / 'pg_ours_vs_published.png')


if __name__ == '__main__':
    main()
