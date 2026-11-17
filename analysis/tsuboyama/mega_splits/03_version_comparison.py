#!/usr/bin/env python
"""How the SPURS versions (and published vs retrained ThermoMPNN / -D) compare on mega_splits, library by library and variant by variant.

Versions of SPURS: ``spurs`` / ``spursm`` retrained with the frozen random adapter (the released checkpoint's configuration), ``spursu`` / ``spursum`` retrained with
the adapter trained (the paper's configuration), ``spurspub`` / ``spurspubm`` the published checkpoint (one run). ThermoMPNN: ``tmpnn`` / ``tmpnnd`` retrained on this
data (3 seeds), ``tmpnnpub`` / ``tmpnndpub`` the published ensemble members (3).

* ``version_paired_rho.csv`` / ``version_paired_rho``  per-library difference of mean-over-seeds Spearman rho between two versions (positive: first higher), the mean over
  libraries with a library-resampling 95 % interval, and the number of libraries each way; validation and test, all variants and doubles only
* ``version_agreement.csv`` / ``version_agreement``    per-variant agreement between versions on the test set: Spearman correlation of the seed-mean predictions (singles,
  and the epistasis term of the multi models on doubles), with the seed-to-seed correlation inside a version as the reference for "the same model"
"""
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ['MSR_TSU_SPLIT'] = 'mega'

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.registry import TSU_MEGA  # noqa: E402

OUT = out_dir('03_tsuboyama')
DS = {'val': TSU_MEGA[0], 'test': TSU_MEGA[1]}
NAME = {'spurs': 'SPURS frozen', 'spursu': 'SPURS trained', 'spurspub': 'SPURS published',
        'spursm': 'SPURS multi frozen', 'spursum': 'SPURS multi trained', 'spurspubm': 'SPURS multi published',
        'tmpnn': 'ThermoMPNN retrain', 'tmpnnpub': 'ThermoMPNN published', 'tmpnnd': 'ThermoMPNN-D retrain', 'tmpnndpub': 'ThermoMPNN-D published'}
PAIRS = [('spursu', 'spurs'), ('spurspub', 'spurs'), ('spurspub', 'spursu'),
         ('spursum', 'spursm'), ('spurspubm', 'spursm'), ('spurspubm', 'spursum'),
         ('tmpnnpub', 'tmpnn'), ('tmpnndpub', 'tmpnnd')]
B = 200               # draft size: this is a cheap resample of per-library differences, so it can be larger than the model bootstraps; paper 2000
rng = np.random.default_rng(3)


def lib_rho(fr, key, mask):
    sp = fr.spine
    ok = sp['use'].to_numpy() & mask
    P = np.atleast_2d(fr.preds[key])
    out = {}
    for g in pd.unique(sp['group'][ok]):
        m = ok & (sp['group'] == g).to_numpy() & np.isfinite(P).all(0)
        if m.sum() >= 5:
            out[g] = np.mean([spearmanr(p[m], sp['y'].to_numpy()[m])[0] for p in P])
    return pd.Series(out)


def paired_rows():
    rows = []
    for split, ds in DS.items():
        fr = load_frame(ds)
        n_mut = fr.spine['n_mut'].to_numpy()
        for subset, mask in (('all variants', np.ones(len(n_mut), bool)), ('doubles', n_mut == 2)):
            for a, b in PAIRS:
                ra, rb = lib_rho(fr, a, mask), lib_rho(fr, b, mask)
                d = (ra - rb).dropna()
                boot = np.array([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
                rows.append(dict(split=split, subset=subset, a=a, b=b, mean_diff=d.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                                 libs_a_higher=int((d > 0).sum()), libs_b_higher=int((d < 0).sum()), n_libraries=len(d), rho_a=ra.mean(), rho_b=rb.mean()))
    return pd.DataFrame(rows)


def agreement():
    fr = load_frame(DS['test'])
    sp = fr.spine
    n_mut, ok = sp['n_mut'].to_numpy(), sp['use'].to_numpy()
    out = []
    def runs(k, m):                                 # one prediction vector per seed (one for the published SPURS)
        return [r[m] for r in np.atleast_2d(fr.preds[k])]
    def corr(a, b, m):
        """Spearman between the runs of two versions, over run pairs of DIFFERENT seeds (all pairs when one side is a single run), the same footing as the
        seed-to-seed reference inside one version: no two runs share a seed, so shared initialisation does not inflate the agreement."""
        ra, rb = runs(a, m), runs(b, m)
        vals = []
        for i, x in enumerate(ra):
            for j, y in enumerate(rb):
                if len(ra) > 1 and len(rb) > 1 and i == j:
                    continue
                f = np.isfinite(x) & np.isfinite(y)
                vals.append(spearmanr(x[f], y[f])[0])
        return float(np.mean(vals))
    sets = {'singles': (['spurs', 'spursu', 'spurspub', 'tmpnn', 'tmpnnpub'], ok & (n_mut == 1)),
            'doubles, epistasis term (multi - own-family singles)': (['spursm:epi', 'spursum:epi', 'spurspubm:epi', 'tmpnndpub:epi', 'tmpnnd:epi'],
                                                                     ok & (n_mut == 2) & np.isfinite(sp['dddG_ML'].to_numpy()))}
    mats = {}
    for name, (keys, m) in sets.items():
        keys = [k for k in keys if k in fr.preds]
        mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
        for a in keys:
            for b in keys:
                mat.loc[a, b] = corr(a, b, m) if a != b else np.nan
        mats[name] = mat
        for a, b in combinations(keys, 2):
            out.append(dict(set=name, a=a, b=b, spearman=mat.loc[a, b]))
        for k in keys:                              # same model, different seed
            P = [r[m] for r in np.atleast_2d(fr.preds[k])]
            if len(P) > 1:
                r = []
                for i, j in combinations(range(len(P)), 2):
                    f = np.isfinite(P[i]) & np.isfinite(P[j])
                    r.append(spearmanr(P[i][f], P[j][f])[0])
                out.append(dict(set=name, a=k, b=k + ' (other seed)', spearman=float(np.mean(r))))
                mat.loc[k, k] = float(np.mean(r))   # the diagonal: the same version, two seeds
            else:
                mat.loc[k, k] = np.nan
    return pd.DataFrame(out), mats


def main():
    P = paired_rows()
    P.to_csv(OUT / 'version_paired_rho.csv', index=False, float_format='%.4f')
    t = P[(P.split == 'test') & (P.subset == 'all variants')]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    for ax, subset in zip(axes, ('all variants', 'doubles')):
        sub = P[P.subset == subset]
        labels = [f'{NAME[a]} − {NAME[b]}' for a, b in PAIRS]
        for j, (split, off, col) in enumerate((('val', -.12, '#7f8c8d'), ('test', .12, '#c0392b'))):
            for i, (a, b) in enumerate(PAIRS):
                r = sub[(sub.split == split) & (sub.a == a) & (sub.b == b)].iloc[0]
                ax.plot([r.lo, r.hi], [i + off] * 2, color=col, lw=2.2, alpha=.55, solid_capstyle='butt')
                ax.plot(r.mean_diff, i + off, 'o', color=col, ms=5, label=split if i == 0 else None)
        ax.axvline(0, color='#555', lw=.8)
        ax.set_yticks(range(len(PAIRS)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Δ mean per-library ρ (first − second); 95 % interval over libraries')
        ax.set_title(subset, loc='left', fontsize=9)
        ax.grid(axis='x', color=style.GRID, lw=.6)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    style.save(fig, OUT / 'version_paired_rho')
    A, mats = agreement()
    A.to_csv(OUT / 'version_agreement.csv', index=False, float_format='%.4f')
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4.2))
    for ax, (name, m) in zip(np.atleast_1d(axes), mats.items()):
        ax.imshow(m.to_numpy(float), vmin=.4, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(m))); ax.set_yticks(range(len(m)))
        ax.set_xticklabels([NAME.get(k.split(':')[0], k) for k in m.columns], rotation=60, ha='right', fontsize=7.5)
        ax.set_yticklabels([NAME.get(k.split(':')[0], k) for k in m.index], fontsize=7.5)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f'{m.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w')
        ax.set_title(name, fontsize=8.5, loc='left')
    fig.tight_layout()
    style.save(fig, OUT / 'version_agreement')
    pd.set_option('display.width', 200)
    print(t[['a', 'b', 'mean_diff', 'lo', 'hi', 'libs_a_higher', 'libs_b_higher']].to_string(index=False, float_format='%.4f'))
    print(P[(P.split == 'test') & (P.subset == 'doubles')][['a', 'b', 'mean_diff', 'lo', 'hi', 'libs_a_higher', 'libs_b_higher']].to_string(index=False, float_format='%.4f'))
    print(A.to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
    main()
