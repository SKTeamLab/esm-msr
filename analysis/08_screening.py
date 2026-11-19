#!/usr/bin/env python
"""Protein-engineering view: what a fixed screening budget buys.

A *hit* is a variant measured more stabilizing than ``--tau`` (default 0.5 kcal/mol); for Domainome, whose label is a change in expression, a hit is any increase (> 0).  Three statistics (definitions in
``msr_analysis/screening.py``):

* **hit rate** - the fraction of the variants a model suggests that are hits (precision of the selection);
* **P(recover >= 1)** - the probability that a screen of the K best-scored variants of a protein contains at least one hit
  (over proteins that have a hit at all; the random-pick probability is drawn for reference);
* **best ddG in the top K** - the most stabilizing measured variant among the K suggested, against the oracle (the best any K
  could contain) and the average of K random picks.

Two designs: one pooled screen per external stability set (a budget of the top 10 % of the set, proteins resampled in the
bootstrap) and one screen per protein (Tsuboyama test libraries, Domainome domains) as the budget K grows; the per-protein curve tables (``screening_per_protein*.csv``) are drawn by
``16_screening_overview.py``, together with the ProteinGym Activity curves.  The retrained
baselines are drawn with the head their training calls for on each set (singles models on singles-only sets, the doubles
models where the set has multi-mutants); ESM-MSR is its WT head on singles-only sets and its combined prediction elsewhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from msr_analysis import models as M  # noqa: E402
from msr_analysis import screening as scr, style  # noqa: E402
from msr_analysis.common import has_multi, out_dir, primary_key  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402
from msr_analysis.external import register_ext_styles  # noqa: E402
from msr_analysis.registry import HERE, LABEL, MODELS, REFERENCE  # noqa: E402
from msr_analysis.registry import TSU, TSU_SPLIT  # noqa: E402  (MSR_TSU_SPLIT=mega -> the mega_splits pair)

OUT = out_dir('08_screening')
POOLED_SETS = ['s783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']   # S571's labels are not ddG
# a family is a model of the roster (``msr_analysis.models``): the same colour, line style, label and position in every figure
FAM_NAME = {'prosst': 'ProSST', 'pmpnn': 'ProteinMPNN', 'esm3': 'ESM3-small-open', 'msr': 'ESM-MSR', 'me': 'MutateEverything', 'tm': 'ThermoMPNN', 'spurs': 'SPURS',
            'geoddg': 'GeoDDG-3D', 'rosetta': 'Rosetta', 'esm3dg': 'ESM3ΔG'}
FAM_ALL = sorted(FAM_NAME, key=lambda f: M.ORDER_INDEX[FAM_NAME[f]])                  # zero-shot, ESM-MSR and the other transfer models, supervised
FAM = [f for f in FAM_ALL if M.BY_NAME[FAM_NAME[f]].group == 'transfer']              # the four transfer models, drawn with markers
FAM_LABEL = {f: M.BY_NAME[n].label_all.replace('\n', ' ') for f, n in FAM_NAME.items()}                                # 'ESM-MSR WT-view (& Dual-view)', 'MutateEverything (Stage 1 & Stage 2)', ...
FAM_LABEL_SINGLES = {f: (M.version_label(n, 0) if M.BY_NAME[n].two_versions else n) for f, n in FAM_NAME.items()}      # the singles version
FAM_LABEL_COMPOSITE = dict(FAM_LABEL)                                                 # singles with the first version, multi-mutants with the second
TAU_DS = {'domainome': 0.0}      # Domainome's label is a change in expression: a hit is any increase (> 0), not > 0.5 kcal/mol
FAM_STYLE = dict(FAM_NAME)       # family -> roster name (the registry key of its style)
FAM_COLOR_ALL = {f: M.line_style(n)[:3] + (8 if n == 'ESM-MSR' else 6,) for f, n in FAM_NAME.items()}          # (colour, line style, line width, zorder) of the curves
register_ext_styles()            # the dot panels look models up in the registry by roster name


def domainome_overlap_domains():
    """Domainome domains homologous to the training set under the split's four-part rule (MMseqs2 E <= 1e-3, identity > 0.40, Foldseek E <= 1e-3, Foldseek TM > 0.70; ``msr_analysis.heldout``,
    ``leak_audit/21_tsuboyama_rule.py``).  ``load_frame('domainome')`` already leaves them out; the set is kept for the explicit exclusion below."""
    from msr_analysis.heldout import excluded
    return excluded('domainome')


# the version that fits the variant: ESM-MSR and ESM3-small-open WT-view on singles and Dual-view on multi-mutants, Stage 1 / Stage 2, ThermoMPNN / ThermoMPNN-D, SPURS / SPURS-multi (the models with one version have one key)
SINGLE_KEY = {f: M.BY_NAME[n].single for f, n in FAM_NAME.items()}
MULTI_KEY = {f: M.BY_NAME[n].multi for f, n in FAM_NAME.items()}


def fam_key(fam, fr, mode='default'):
    """The prediction key of family ``fam`` on frame ``fr``: ``singles`` = the singles version of every variant (singles only); otherwise the singles version on the singles and the doubles version on the multi-mutants
    (a set of singles only has no doubles: the singles version; a mixed set: a composite column built here)."""
    ks, kd = SINGLE_KEY[fam], MULTI_KEY[fam]
    if ks == kd or mode == 'singles' or not has_multi(fr):
        return ks if ks in fr.preds else None
    if ks not in fr.preds or kd not in fr.preds:
        return None
    ck = f'{ks}|{kd}'
    if ck not in fr.preds:
        n1 = (fr.spine['n_mut'] == 1).to_numpy()
        fr.preds[ck] = np.where(n1[None, :], fr.preds[ks], fr.preds[kd])
    return ck


def pooled(tau, B, frac):
    rows, reps_store = [], {}
    for ds in POOLED_SETS:
        fr = load_frame(ds)
        sp = fr.spine
        for fam in FAM:
            k = fam_key(fam, fr)
            if k is None:
                continue
            ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
            y = sp['y'].to_numpy()[ok]
            cl = sp['cluster'].to_numpy()[ok]
            outs = [scr.pooled_screen(p[ok], y, cl, frac, tau, B, seed=5) for p in fr.preds[k]]
            n_seeds = len(outs)
            if n_seeds == 1:
                reps = outs[0]['reps']
            else:
                sw = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=B).astype(float) / n_seeds
                reps = np.einsum('sbm,bs->bm', np.stack([o['reps'] for o in outs], axis=0), sw)
            base = np.mean([o['base'] for o in outs], axis=0)
            reps_store[(ds, fam)] = reps
            for j, m in enumerate(('hit_rate', 'best', 'base_rate', 'gain')):
                lo, hi = np.nanquantile(reps[:, j], [.025, .975])
                rows.append(dict(dataset=ds, family=fam, key=k, metric=m, value=float(base[j]), lo=float(lo), hi=float(hi), n=int(ok.sum()),
                                 oracle_best=float(y.max())))
    return pd.DataFrame(rows), reps_store


def fig_pooled(P, reps_store, tau, frac):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.4), sharex=True)
    for ax, metric, yl in ((axes[0], 'hit_rate', f'hit rate (ddG > {tau})'),
                           (axes[1], 'best', 'best ddG in selection (kcal/mol)'),
                           (axes[2], 'gain', 'mean ddG of selection (kcal/mol)')):
        d = P[P.metric == metric]
        table = {(r.dataset, FAM_NAME[r.family]): r.value for r in d.itertuples() if r.family in FAM_NAME}
        ci = {(r.dataset, FAM_NAME[r.family]): (r.lo, r.hi) for r in d.itertuples() if r.family in FAM_NAME}
        style.dot_panel(ax, POOLED_SETS, [FAM_NAME[f] for f in FAM], table, ci, highlight=['ESM-MSR'], ylabel=yl, size=6.5)
        if metric == 'hit_rate':
            br = P[P.metric == 'base_rate'].groupby('dataset').value.first()
            for i, ds in enumerate(POOLED_SETS):
                ax.plot([i - .42, i + .42], [br[ds]] * 2, color='#c0392b', lw=1.2, ls='--', zorder=2)
            ax.plot([], [], color='#c0392b', ls='--', label='hit rate of the whole set (random selection)')
        elif metric == 'gain':
            ax.axhline(0, color='#999', lw=.8, ls=':')
        else:
            orc = P.groupby('dataset').oracle_best.first()
            for i, ds in enumerate(POOLED_SETS):
                ax.plot([i - .42, i + .42], [orc[ds]] * 2, color='#2c3e50', lw=1.2, ls=':', zorder=2)
            ax.plot([], [], color='#2c3e50', ls=':', label='best variant in the set (oracle)')
        if metric != 'gain':
            ax.legend(loc='upper right', ncol=2, fontsize=8)
    # family legend
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], label=FAM_LABEL[f], **style.marker_kwargs(FAM_NAME[f], 7, hollow=False)) for f in FAM]
    axes[0].legend(handles=hs + axes[0].get_legend_handles_labels()[0][-1:], loc='upper left', ncol=3, fontsize=8)
    axes[0].set_title(f'Pooled screen: the top {int(frac * 100)}% of each set by predicted ddG; whiskers 95% protein-level bootstrap')
    axes[2].set_xticklabels([LABEL[d] for d in POOLED_SETS], rotation=30, ha='right')
    fig.tight_layout()
    style.save(fig, OUT / 'screening_pooled_external')


if TSU_SPLIT == 'mega':      # mega_splits: SPURS with the adapter trained, the published SPURS, the published ThermoMPNN / -D (data._load_mega); drawn in the all-models panels
    _X = {'spursu': ('SPURS, trained adapter', 'SPURS multi, trained adapter', 'spursu', 'spursum', 'spursu', (MODELS['spursu']['color'], '-', 2.0, 6)),
          'spurspub': ('SPURS, published', 'SPURS multi, published', 'spurspub', 'spurspubm', 'spurspub', (MODELS['spurspub']['color'], '--', 2.0, 6)),
          'tmpub': ('ThermoMPNN, published', 'ThermoMPNN-D, published', 'tmpnnpub', 'tmpnndpub', 'tmpnnpub', (MODELS['tmpnnpub']['color'], '--', 2.0, 6))}
    for _f, (_ls, _lc, _sk, _mk, _st, _col) in _X.items():
        FAM_LABEL[_f], FAM_LABEL_SINGLES[_f], FAM_LABEL_COMPOSITE[_f] = _ls, _ls, f'{_ls} | {_lc.split(",")[0]}'
        SINGLE_KEY[_f], MULTI_KEY[_f], FAM_STYLE[_f], FAM_COLOR_ALL[_f] = _sk, _mk, _st, _col
        FAM_NAME[_f] = {'spursu': 'SPURS', 'spurspub': 'SPURS published*', 'tmpub': 'ThermoMPNN'}[_f]
        MODELS[_f] = dict(MODELS[_st])
    FAM_ALL = FAM_ALL[:7] + list(_X) + FAM_ALL[7:]


def per_protein(ds, Ks, tau, B, exclude_groups=None, mode='default', exclude_cys=False, fams=FAM, fr=None):
    fr = load_frame(ds) if fr is None else fr
    sp = fr.spine
    keep = ~sp['group'].astype(str).isin(exclude_groups).to_numpy() if exclude_groups else np.ones(len(sp), dtype=bool)
    if exclude_cys:
        to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
        keep = keep & ~to_cys
    if mode == 'singles':
        keep = keep & (sp['n_mut'].to_numpy() == 1)
    rows = []
    for fam in fams:
        k = fam_key(fam, fr, mode=mode)
        if k is None:
            continue
        ok = sp['use'].to_numpy() & keep & np.isfinite(fr.preds[k]).all(axis=0)
        y = sp['y'].to_numpy()[ok]
        g = sp['group'].to_numpy()[ok]
        for K in Ks:
            tabs = [scr.per_protein_table(g, p[ok], y, K, tau) for p in fr.preds[k]]
            res = scr.boot_summary(tabs, B=B, seed=K)
            for m, (v, lo, hi) in res.items():
                rows.append(dict(dataset=ds, family=fam, key=k, K=K, metric=m, value=v, lo=lo, hi=hi, n_proteins=len(tabs[0])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--frac', type=float, default=0.10)
    ap.add_argument('--B', type=int, default=1000)
    a = ap.parse_args()
    if TSU_SPLIT == 'mega':           # mega_splits: only the Tsuboyama test tables (the external sets and Domainome do not depend on the split)
        for mode, stem in (('singles', 'screening_per_protein_singles'), ('composite', 'screening_per_protein_composite')):
            for fams, sfx in ((FAM, ''), (FAM_ALL, '_all_models')):
                D = per_protein(TSU[1], [1, 3, 5, 10, 20, 50, 100], a.tau, a.B, mode=mode, exclude_cys=True, fams=fams)
                D.to_csv(OUT / f'{stem}{sfx}.csv', index=False, float_format='%.4f')
            print(f'=== {stem} (K=10) ===')
            print(D[(D.metric.isin(['hit_rate', 'p_recover', 'gain'])) & (D.K == 10)].pivot_table(index=['dataset', 'metric'], columns='family', values='value').round(3))
        return
    P, reps = pooled(a.tau, a.B, a.frac)
    P.to_csv(OUT / 'screening_pooled_external.csv', index=False, float_format='%.4f')
    fig_pooled(P, reps, a.tau, a.frac)
    # paired differences ESM-MSR - baseline in hit rate, per set
    rows = []
    for ds in POOLED_SETS:
        for f in FAM[1:]:
            if (ds, 'msr') in reps and (ds, f) in reps:
                d = reps[(ds, 'msr')][:, 0] - reps[(ds, f)][:, 0]
                rows.append(dict(dataset=ds, baseline=f, d_hit_rate=float(np.nanmean(d)), lo=float(np.nanquantile(d, .025)), hi=float(np.nanquantile(d, .975))))
    pd.DataFrame(rows).to_csv(OUT / 'screening_pooled_paired_diff.csv', index=False, float_format='%.4f')
    ks = [1, 3, 5, 10, 20, 50, 100]
    dom_exclude = domainome_overlap_domains()
    dom_df = per_protein('domainome', ks, TAU_DS['domainome'], a.B, exclude_groups=dom_exclude, mode='default', fams=FAM)
    dom_df_all = per_protein('domainome', ks, TAU_DS['domainome'], a.B, exclude_groups=dom_exclude, mode='default', fams=FAM_ALL)
    for mode, stem in (('singles', 'screening_per_protein_singles'),
                       ('composite', 'screening_per_protein_composite')):
        tsu_df = per_protein(TSU[1], ks, a.tau, a.B, mode=mode, exclude_cys=True, fams=FAM)
        D = pd.concat([tsu_df, dom_df], ignore_index=True)
        D.to_csv(OUT / f'{stem}.csv', index=False, float_format='%.4f')

        tsu_df_all = per_protein(TSU[1], ks, a.tau, a.B, mode=mode, exclude_cys=True, fams=FAM_ALL)
        D_all = pd.concat([tsu_df_all, dom_df_all], ignore_index=True)
        stem_all = f'{stem}_all_models'
        D_all.to_csv(OUT / f'{stem_all}.csv', index=False, float_format='%.4f')
        print(f'=== {stem_all} (K=10) ===')
        print(D_all[(D_all.metric.isin(['hit_rate', 'p_recover'])) & (D_all.K == 10)].pivot_table(index=['dataset', 'metric'], columns='family', values='value').round(3))
    print(P[P.metric == 'hit_rate'].pivot_table(index='family', columns='dataset', values='value')[POOLED_SETS].round(3).to_string())
    print(P[P.metric == 'base_rate'].groupby('dataset').value.first().round(3).to_dict())


if __name__ == '__main__':
    main()
