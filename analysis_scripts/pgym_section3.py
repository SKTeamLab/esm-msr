"""The ESM-MSR rows of docs/BENCHMARKS_OCT2026.md section 3 (ProteinGym in detail), and its paired devel-vs-released table, from the two
per-DMS tables in outputs/pgym (pgym_all_models_per_dms.csv, pgym_dc8_per_dms.csv).

  python analysis_scripts/pgym_section3.py --root <dir holding the two csv files>

Run it on the backed-up tables (outputs/pgym_pre_per_mutation_bias) to reproduce the numbers the doc printed before the
per-mutation-bias correction, then on the corrected ones (outputs/pgym). Categories and the 213-DMS set are those of
benchmark_matrix.py (PG avg = mean of the five category means; DMS mean = plain mean over the 213).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

PG_DROP = {'BRCA1_HUMAN_Findlay_2018', 'SCN5A_HUMAN_Glazer_2019', 'POLG_CXB3N_Mattenberger_2021', 'BRCA2_HUMAN_Erwood_2022_HEK293T'}
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']


def summarize(series, cat, common):
    s = series.reindex(common)
    assert s.notna().all(), f'{series.name}: {int(s.isna().sum())} of the {len(common)} common DMS missing'
    g = s.groupby(cat.reindex(common)).mean()
    return [g[c] for c in CATS] + [g.mean(), s.mean()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    a = ap.parse_args()
    allm = pd.read_csv(os.path.join(a.root, 'pgym_all_models_per_dms.csv')).set_index('DMS_id')
    dc8 = pd.read_csv(os.path.join(a.root, 'pgym_dc8_per_dms.csv')).set_index('DMS_id')
    cat = allm['category']
    common = [d for d in cat.index if d not in PG_DROP]
    assert len(common) == 213

    rows = {}
    for e, lab in (('1.0', '1'), ('0.75', '0.75'), ('0.5', '0.5'), ('0.25', '0.25')):
        rows[f'ESM-MSR (devel), σ = {lab} (3 seeds)'] = summarize(dc8[f'dc8 eps={e} mean'], cat, common)
    rows['ESM-MSR (released), σ = 1'] = summarize(allm['ESM-MSR full'], cat, common)
    for e, lab in (('1.0', '1'), ('0.75', '0.75'), ('0.5', '0.5'), ('0.25', '0.25')):
        rows[f'ESM-MSR (released), WT head only, σ = {lab}'] = summarize(allm[f'ESM-MSR WT eps={e}'], cat, common)
    rows['ESM-MSR, σ = 0 (no LoRA)'] = summarize(allm['ESM-MSR WT eps=0'], cat, common)
    print('| model | Activity | Binding | Expression | Org. fitness | Stability | PG avg | DMS mean |')
    for k, v in rows.items():
        print(f'| {k} | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')

    # Definitions as the doc has always used them (recovered from its printed numbers, which this reproduces): "paired delta" is the mean
    # over the 213 DMS of (devel seed-mean - released), i.e. a DMS-mean difference; the seed sd is the population sd (ddof = 0) of the three
    # seeds' PG averages; t is the paired t over DMS.
    print('\n| devel setting | PG avg | paired Δ vs released | better on | seed sd (PG avg) | paired t |')
    rel = allm['ESM-MSR full'].reindex(common)
    for e in ('1.0', '0.75', '0.5', '0.25'):
        seeds = [summarize(dc8[f'dc8 eps={e} s{i}'], cat, common)[5] for i in (1, 2, 3)]
        d = dc8[f'dc8 eps={e} mean'].reindex(common) - rel
        pg = summarize(dc8[f'dc8 eps={e} mean'], cat, common)[5]
        print(f'| σ = {e} | {pg:.4f} | {d.mean():+.4f} | {int((d > 0).sum())} / 213 | {np.std(seeds):.4f} | {ttest_1samp(d, 0).statistic:+.2f} |')


if __name__ == '__main__':
    main()
