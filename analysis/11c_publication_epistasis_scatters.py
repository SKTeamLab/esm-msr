#!/usr/bin/env python
"""Publication figure 6: the epistasis scatters of the notebooks (``compare_supervised_epistasis_rho_test``: density scatter with marginal histograms), on the capped Tsuboyama
test doubles with both singles measured (12,455, the doubles of the epistasis tables; all backbones are wild type there).

  fig6a  minus the additive dG of the two measured singles, dG_additive = dG_wt + ddG_A + ddG_B (the double's stability if the substitutions did not interact), against the
         measured epistasis dddG = ddG_AB - ddG_A - ddG_B.  A double that is lost to the assay floor (dG about 0) has dddG = -dG_additive: it sits on the x = y line, whatever
         the stability of its domain, and the rise of dddG with -dG_additive is that floor, not interaction.  (In ddG the floor is at -dG_wt, a different line for every domain.)
  fig6b  ESM-MSR's predicted epistasis (the MT head minus the WT head; mean of the three baseline seeds) against the measured dddG.

The notebooks drew a dashed linear fit; here the only line is x = y, over equal axis limits.  Outputs: ``results/11_publication`` (png, pdf, and the numbers in csv).
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


def main():
    fr = load_frame(DS)
    sp = fr.spine
    m = epi_rows(fr)
    y = sp['dddG_ML'].to_numpy()
    minus_add = -(sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float))        # -dG_additive; dG_wt is the domain's wild-type stability
    pred = np.nanmean(fr.preds['msr:mt'] - fr.preds['msr:wt'], axis=0)          # ESM-MSR's epistasis: the MT head minus the WT head
    rows = []
    for name, x, xl, xn in (('fig6a_dG_additive_vs_dddG', minus_add, '−(ΔG additive) = −(ΔG_wt + ΔΔG_A + ΔΔG_B) (kcal/mol)', '−(ΔG additive)'),
                            ('fig6b_esm_msr_dddG_pred_vs_true', pred, 'ESM-MSR predicted δΔΔG = dddG (kcal/mol)', 'ESM-MSR δΔΔG')):
        ok = m & np.isfinite(x) & np.isfinite(y)
        fig = style.density_scatter_marginals(x[ok], y[ok], xl, Y_LABEL, xname=xn, yname='measured δΔΔG')
        fig.suptitle(f'Capped Tsuboyama test doubles with both singles measured (n = {int(ok.sum()):,})', fontsize=9.5, y=0.93)
        style.save(fig, OUT / name)
        rows.append(dict(figure=name, n=int(ok.sum()), spearman=spearmanr(x[ok], y[ok])[0], pearson=pearsonr(x[ok], y[ok])[0],
                         share_both_positive=float(np.mean((x[ok] > 0) & (y[ok] > 0)))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'fig6_epistasis_scatters.csv', index=False, float_format='%.4f')
    print(T.to_string())


if __name__ == '__main__':
    main()
