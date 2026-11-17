#!/usr/bin/env python
"""How close are the two mutated residues of the double mutants of the four Faure datasets?

For every double mutant with both singles measured (the doubles that have an epistasis): the minimal heavy-atom distance between the two residues, the C-beta - C-beta distance (C-alpha for glycine) in the
AlphaFold2 model of the protein, and the sequence separation.  Baselines: the same distances over all pairs of positions that carry a measured single (``all pairs``), and for PDZ3, whose doubles are
random NNK substitutions on nine single-mutant backgrounds (Faure et al.), the pairs that contain one of the background positions (``background pairs``).  A contact = minimal heavy-atom distance < 5 A.

  proximity_distribution   A: cumulative distribution of the minimal heavy-atom distance (doubles; baselines dashed); B: share of doubles per distance bin; C: cumulative distribution of the sequence
                           separation; D: number of doubles per position pair.

Writes ``results/proximity_summary.csv`` (per set and baseline: n, percentiles and shares) and ``results/pair_table.csv`` (per set and position pair: doubles, distances).

    python analysis/faure_epistasis/01_proximity.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import faure as F

BINS = [0, 4, 5, 8, 12, 16, 99]
BIN_LABEL = ['< 4', '4-5', '5-8', '8-12', '12-16', '> 16']
BACKGROUND_MIN_DOUBLES = 300            # PDZ3: positions in at least this many doubles are the nine single-mutant backgrounds' sites


def cdf(ax, v, **kw):
    v = np.sort(np.asarray(v))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), **kw)


def summary_row(name, dist_min, dist_cb, sep, extra=None):
    r = dict(name=name, n=len(dist_min), min_p10=np.percentile(dist_min, 10), min_p25=np.percentile(dist_min, 25), min_median=np.median(dist_min), min_p75=np.percentile(dist_min, 75),
             min_p90=np.percentile(dist_min, 90), cb_median=np.median(dist_cb), sep_median=np.median(sep), contact_lt5=float((dist_min < 5).mean()), lt8=float((dist_min < 8).mean()),
             lt12=float((dist_min < 12).mean()))
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], BIN_LABEL):
        r[f'bin_{lab}'] = float(((dist_min >= lo) & (dist_min < hi)).mean())
    r.update(extra or {})
    return r


def main():
    rows, pairs = [], []
    store = {}
    for ds in F.SETS:
        single, dbl, raw = F.load(ds)
        dom, _ = F.structure(ds, raw.ref_seq.iloc[0])
        cb, mn = F.distances(dom)
        dbl['dmin'], dbl['dcb'] = mn[dbl.pos1, dbl.pos2], cb[dbl.pos1, dbl.pos2]
        dbl['sep'] = (dbl.pos2 - dbl.pos1).abs()
        dbl.to_pickle(F.CACHE / f'faure_doubles_{ds}.pkl')
        pc = dbl.groupby(['pos1', 'pos2']).size().rename('n_doubles').reset_index()
        pc['dmin'], pc['dcb'] = mn[pc.pos1, pc.pos2], cb[pc.pos1, pc.pos2]
        pc.insert(0, 'set', ds)
        pairs.append(pc)
        sp = sorted({int(m[1:-1]) for m in single})
        allp = [(i, j) for a, i in enumerate(sp) for j in sp[a + 1:]]
        ai, aj = np.array([p[0] for p in allp]), np.array([p[1] for p in allp])
        rows.append(summary_row(f'{ds}|doubles', dbl.dmin.to_numpy(), dbl.dcb.to_numpy(), dbl.sep.to_numpy(), dict(set=ds, kind='doubles', n_pairs=len(pc), doubles_per_pair_median=float(pc.n_doubles.median()))))
        rows.append(summary_row(f'{ds}|pairs', pc.dmin.to_numpy(), pc.dcb.to_numpy(), (pc.pos2 - pc.pos1).to_numpy(), dict(set=ds, kind='observed position pairs (each once)', n_pairs=len(pc))))
        rows.append(summary_row(f'{ds}|all pairs', mn[ai, aj], cb[ai, aj], aj - ai, dict(set=ds, kind='all pairs of measured positions', n_pairs=len(allp))))
        store[ds] = dict(dbl=dbl, pc=pc, allmin=mn[ai, aj], allsep=aj - ai)
        if ds.startswith('DLG4'):
            cnt = pd.concat([dbl.pos1, dbl.pos2]).value_counts()
            bg = sorted(cnt[cnt >= BACKGROUND_MIN_DOUBLES].index)
            bp = [(i, j) for (i, j) in allp if i in bg or j in bg]
            bi, bj = np.array([p[0] for p in bp]), np.array([p[1] for p in bp])
            rows.append(summary_row(f'{ds}|background pairs', mn[bi, bj], cb[bi, bj], bj - bi, dict(set=ds, kind=f'pairs with a background position {bg}', n_pairs=len(bp))))
            store[ds]['bgmin'], store[ds]['bg'] = mn[bi, bj], bg
    S = pd.DataFrame(rows)
    S.to_csv(F.OUT / 'proximity_summary.csv', index=False, float_format='%.4g')
    pd.concat(pairs).to_csv(F.OUT / 'pair_table.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(S[['name', 'n', 'n_pairs', 'min_median', 'min_p10', 'min_p90', 'contact_lt5', 'lt8', 'lt12', 'cb_median', 'sep_median']].round(3).to_string(index=False))
    with plt.rc_context(F.RC):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
        ax = axes[0, 0]
        for ds in F.SETS:
            cdf(ax, store[ds]['dbl'].dmin, color=F.COLOR[ds], lw=1.8, label=F.LABEL[ds])
        cdf(ax, store[F.SETS[0]]['allmin'], color='#888888', ls='--', lw=1.1, label='all pairs of PDZ3 positions')
        cdf(ax, store[F.SETS[0]]['bgmin'], color='#555555', ls=':', lw=1.4, label='PDZ3 pairs with a background position')
        cdf(ax, store[F.SETS[2]]['allmin'], color='#b5b5b5', ls='--', lw=1.1, label='all pairs of SH3 positions')
        ax.axvline(5, color=F.RC['axes.edgecolor'], lw=1)
        ax.set_xlabel('minimal heavy-atom distance between the two residues (A)')
        ax.set_ylabel('cumulative share of doubles')
        ax.set_xlim(0, 30)
        ax.set_title('A  Distance of the two mutated residues')
        ax.legend(fontsize=7.4, loc='lower right', handlelength=2.4)
        ax = axes[0, 1]
        w = 0.2
        for k, ds in enumerate(F.SETS):
            r = S[S.name == f'{ds}|doubles'].iloc[0]
            ax.bar(np.arange(len(BIN_LABEL)) + (k - 1.5) * w, [r[f'bin_{l}'] for l in BIN_LABEL], width=w, color=F.COLOR[ds], label=F.SHORT[ds])
        ax.set_xticks(range(len(BIN_LABEL)))
        ax.set_xticklabels(BIN_LABEL)
        ax.set_xlabel('minimal heavy-atom distance (A)')
        ax.set_ylabel('share of doubles')
        ax.set_title('B  Doubles by distance bin')
        ax.legend(fontsize=7.6, handlelength=1.2)
        ax = axes[1, 0]
        for ds in F.SETS:
            cdf(ax, store[ds]['dbl'].sep, color=F.COLOR[ds], lw=1.8, label=F.SHORT[ds])
        ax.set_xscale('log')
        ax.set_xlabel('sequence separation |i - j| (residues)')
        ax.set_ylabel('cumulative share of doubles')
        ax.set_title('C  Sequence separation')
        ax = axes[1, 1]
        for ds in F.SETS:
            c = store[ds]['pc'].n_doubles
            ax.hist(c, bins=np.unique(np.round(np.geomspace(1, c.max() + 1, 28)).astype(int)), histtype='step', color=F.COLOR[ds], lw=1.8, label=f'{F.SHORT[ds]} (median {int(c.median())})')
        ax.set_xscale('log')
        ax.set_xlabel('doubles per position pair')
        ax.set_ylabel('position pairs')
        ax.set_title('D  Doubles per position pair')
        ax.legend(fontsize=7.6, handlelength=1.2)
        F.finish(fig, 'Doubles with both singles measured (PDZ3 abundance %d, binding %d; SH3 abundance %d, binding %d). Distances in the AlphaFold2 models of the proteins (domain residues aligned to the model); '
                  'contact = minimal heavy-atom distance < 5 A (vertical line). Dashed: all pairs of positions that carry a measured single; dotted: the PDZ3 pairs that contain one of the %d background positions '
                  '(positions in at least %d doubles).' % tuple([len(store[d]['dbl']) for d in F.SETS] + [len(store[F.SETS[0]]['bg']), BACKGROUND_MIN_DOUBLES]), 'proximity_distribution')


if __name__ == '__main__':
    sys.exit(main())
