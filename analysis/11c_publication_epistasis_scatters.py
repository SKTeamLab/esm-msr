#!/usr/bin/env python
"""Publication figure 6: the epistasis scatters of the notebooks (``compare_supervised_epistasis_rho_test``: density scatter, here without the marginal histograms), on the capped
Tsuboyama test doubles with both singles measured (12,455, the doubles of the epistasis tables; all backbones are wild type there).

  fig6a  the additive dG of the two measured singles, dG_additive = dG_wt + ddG_A + ddG_B (the double's stability if the substitutions did not interact), against the measured
         epistasis dddG = ddG_AB - ddG_A - ddG_B.  A double that is lost to the assay floor (dG about 0) has dddG = -dG_additive: it falls on the dashed line y = -x, whatever the
         stability of its domain, and the rise of dddG as dG_additive falls is that floor, not interaction.  (In ddG the floor is at -dG_wt, a different line for every domain.)
  fig6b  ESM-MSR's predicted epistasis (the MT head minus the WT head; mean of the three baseline seeds) against the measured dddG, with the line x = y.
  fig6c  fig6b with the doubles whose dG_additive is below 0 (already lost to the floor if the singles add) in green.

The dddG correlation plots of every model, with and without those doubles, are ``13_dddG_correlation_scatters.py``.  Outputs: ``results/11_publication`` (png, pdf, csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis import style  # noqa: E402
from msr_analysis.common import epi_rows, out_dir  # noqa: E402
from msr_analysis.data import load_frame  # noqa: E402

OUT = out_dir('11_publication')
DS = 'tsuboyama_capped_test'
Y_LABEL = 'measured δΔΔG = dddG (kcal/mol)'
TITLE = 'Capped Tsuboyama test doubles with both singles measured'


def main():
    fr = load_frame(DS)
    sp = fr.spine
    m = epi_rows(fr)
    y = sp['dddG_ML'].to_numpy()
    dG_add = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)          # dG_wt is the domain's wild-type stability
    pred = np.nanmean(fr.preds['msr:mt'] - fr.preds['msr:wt'], axis=0)                      # ESM-MSR's epistasis: the MT head minus the WT head
    rows = []
    jobs = [('fig6a_dG_additive_vs_dddG', dG_add, 'ΔG additive = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', dict(ref='neg', ref_label='y = −x: the double is lost to the assay floor')),
            ('fig6b_esm_msr_dddG_pred_vs_true', pred, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy')),
            ('fig6c_esm_msr_dddG_pred_vs_true_dG_additive_lt0', pred, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', dict(ref='xy', highlight=dG_add < 0, highlight_label='ΔG additive < 0'))]
    for name, x, xl, kw in jobs:
        ok = m & np.isfinite(x) & np.isfinite(y)
        if 'highlight' in kw:
            kw = {**kw, 'highlight': kw['highlight'][ok]}
        fig = style.density_scatter_plain(x[ok], y[ok], xl, Y_LABEL, title=f'{TITLE} (n = {int(ok.sum()):,})', **kw)
        style.save(fig, OUT / name)
        rows.append(dict(figure=name, n=int(ok.sum()), spearman=spearmanr(x[ok], y[ok])[0], pearson=pearsonr(x[ok], y[ok])[0], share_dG_additive_lt0=float(np.mean(dG_add[ok] < 0))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'fig6_epistasis_scatters.csv', index=False, float_format='%.4f')
    print(T.to_string())


if __name__ == '__main__':
    main()
