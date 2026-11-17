#!/usr/bin/env python
"""The assays that lose most when the LoRA scale goes from sigma 0 to sigma 1: where does the loss come from?

For the worst assay of every category but Stability (and the runners-up named in ``WORST``), with y = the measured score and s0, s1 = the WT-adapter predictions at sigma 0 (ESM3-small-open, unmasked) and 1
(FINAL_p0 seed 1), delta = z(s1) - z(s0) the signal the adapter adds, Spearman rho of

  rho0, rho1         s0 and s1 with y
  rho_added          delta with y (negative: what the adapter adds runs against the phenotype)
  between0/1         the position means of the prediction with those of y (which positions matter)
  within0/1          the mean over positions of the rho between the substitutions of one position (positions with >= 5 variants)
  hydropathy columns the Kyte-Doolittle hydropathy of the mutant residue with y, s0, s1 and delta

Single mutants only (the multi-mutant rows of an assay are left out).  Writes ``results/worst_assays_sigma1.csv``.

    python analysis/proteingym/p0_sigma/05_worst_assays.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import pgsigma as P

WORST = ['KCNE1_HUMAN_Muhammad_2023_expression', 'KCNE1_HUMAN_Muhammad_2023_function', 'RPC1_LAMBD_Li_2019_low-expression', 'RPC1_LAMBD_Li_2019_high-expression',
         'B2L11_HUMAN_Dutta_2010_binding-Mcl-1', 'TADBP_HUMAN_Bolognesi_2019', 'IF1_ECOLI_Kelsic_2016', 'RNC_ECOLI_Weeks_2023', 'DLG4_RAT_McLaughlin_2012', 'Q837P4_ENTFA_Meier_2023']
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)


def z(a):
    return (a - np.nanmean(a)) / np.nanstd(a)


def rho(a, b):
    return float(spearmanr(a, b)[0])


def load(a):
    d = None
    for s in P.SIGMAS:
        x = pd.read_csv(P.PG[s] / f'{a}.csv', usecols=['mutant', 'DMS_score', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant').rename(columns={'wt_lora_pred': f's{s}', 'DMS_score': 'y'})
        d = x if d is None else d.join(x[[f's{s}']], how='inner')
    return d[~d.index.str.contains(':')]


def within(pos, y, c):
    v = []
    for _, h in pd.DataFrame(dict(pos=pos, y=y, c=c)).groupby('pos'):
        if len(h) >= 5 and h.y.nunique() > 2:
            v.append(rho(h.c, h.y))
    return float(np.nanmean(v))


def main():
    t = P.per_assay()
    t['d1'] = t['rho_1.0'] - t['rho_0.0']
    print('worst assay at sigma 1 per category:')
    print(t.sort_values('d1').groupby('category').head(1).query("category != 'Stability'")[['category', 'assay', 'n', 'rho_0.0', 'rho_1.0', 'd1', 'leaked']].round(3).to_string(index=False))
    rows = []
    for a in WORST:
        d = load(a)
        pos = d.index.str.extract(r'(\d+)')[0].astype(int).to_numpy()
        y, s0, s1 = d.y.to_numpy(), d['s0.0'].to_numpy(), d['s1.0'].to_numpy()
        dl = z(s1) - z(s0)
        g = pd.DataFrame(dict(pos=pos, y=y, s0=s0, s1=s1)).groupby('pos').mean()
        h = d.index.str[-1].map(KD).to_numpy(float)
        rows.append(dict(assay=a, n=len(d), n_positions=len(g), rho0=rho(s0, y), rho1=rho(s1, y), rho_added=rho(dl, y), between0=rho(g.s0, g.y), between1=rho(g.s1, g.y),
                         within0=within(pos, y, s0), within1=within(pos, y, s1), rho_s0_s1=rho(s0, s1), hydropathy_y=rho(h, y), hydropathy_s0=rho(h, s0), hydropathy_s1=rho(h, s1), hydropathy_added=rho(h, dl)))
    R = pd.DataFrame(rows)
    R.to_csv(P.OUT / 'worst_assays_sigma1.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
