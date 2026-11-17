#!/usr/bin/env python
"""Figure of the ``cell_mag`` ceiling on the Faure datasets (from ``03_cell_ceiling.py``).

  cell_mag_ceiling   A: the number of complete blocks (>= 3 x 3) per dataset: none for PDZ3, hundreds for SH3; B: the noise ceiling of ``cell_mag`` (the mean block Spearman rho of a perfect predictor of
                     the true cell effects) for the SH3 datasets under three noise models of the double's own measurement, for all blocks and the blocks above the floor, with the value of ESM-MSR (FINAL_p0).

    python analysis/faure_epistasis/04_fig_ceiling.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import faure as F

MODELS = [('nominal', 'reported error of each double'), ('R=0.87', 'replicate R = 0.87'), ('R=0.92', 'replicate R = 0.92')]
MCOL = {'nominal': '#444444', 'R=0.87': '#eb6834', 'R=0.92': '#1baf7a'}


def main():
    C = pd.read_csv(F.OUT / 'cell_mag_ceiling.csv')
    Bk = pd.read_csv(F.OUT / 'cell_mag_blocks.csv')
    with plt.rc_context(F.RC):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw=dict(width_ratios=[0.8, 1.5]))
        ax = axes[0]
        x = np.arange(len(F.SETS))
        cells = [int(Bk[Bk.set == d].eval('rows * cols').sum()) if (Bk.set == d).any() else 0 for d in F.SETS]
        nb = [int((Bk.set == d).sum()) for d in F.SETS]
        ax.bar(x, nb, color=[F.COLOR[d] for d in F.SETS])
        for xi, (n, c) in enumerate(zip(nb, cells)):
            ax.text(xi, n + 12, f'{n} blocks\n{c:,} cells' if n else 'none', ha='center', va='bottom', fontsize=8.2)
        ax.set_xticks(x)
        ax.set_xticklabels([F.SHORT[d] for d in F.SETS], rotation=20, ha='right')
        ax.set_ylabel('complete blocks (>= 3 x 3)')
        ax.set_ylim(0, max(nb) * 1.25)
        ax.set_title('A  Blocks available for cell_mag')
        ax = axes[1]
        ax.axhline(0, color='#8a8982', lw=1)
        pos, ticks, labs = 0.0, [], []
        for ds in F.SETS[2:]:
            for stratum in ('all blocks', 'blocks above floor'):
                t = C[(C.set == ds) & (C.stratum == stratum) & C.noise_model.notna()]
                if not len(t):
                    continue
                v = {m: float(t[t.noise_model == m].ceiling_spearman.iloc[0]) for m, _ in MODELS}
                ax.plot([pos, pos], [min(v.values()), max(v.values())], color='#bbbbbb', lw=5, solid_capstyle='round', zorder=1)
                for m, lab in MODELS:
                    ax.plot(pos, v[m], 'o', color=MCOL[m], ms=7, zorder=3, label=lab if (pos == 0.0) else None)
                mv = float(t.model_cell_mag.iloc[0])
                ax.plot([pos - 0.28, pos + 0.28], [mv, mv], color=F.COLOR[ds], lw=3, zorder=4, solid_capstyle='butt', label='ESM-MSR (FINAL_p0), same blocks' if pos == 0.0 else None)
                ax.text(pos, mv - 0.045, f'{mv:.2f}', ha='center', va='top', fontsize=8.4, color=F.COLOR[ds], fontweight='bold')
                ticks.append(pos)
                labs.append(f'{F.SHORT[ds]}\n{stratum}\n({int(t.blocks.iloc[0])} blocks)')
                pos += 1.0
            pos += 0.5
        ax.set_xticks(ticks)
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylim(-0.12, 1.02)
        ax.set_xlim(-0.6, pos - 0.4)
        ax.set_ylabel('cell_mag (mean block Spearman rho)')
        ax.set_title('B  Noise ceiling of a perfect predictor, and ESM-MSR')
        ax.legend(fontsize=7.8, loc='upper right', handlelength=1.4, ncol=2)
        F.finish(fig, 'cell_mag = mean over complete blocks of the Spearman rho between the double-centred predicted and measured beyond-global residuals (esm_msr.epi_metrics). Ceiling: the true double-centred cell effects '
                 'are the observed ones shrunk to the signal variance (observed variance - noise variance), measured = true + noise of the double itself (the singles cancel in double-centring), mean block rho over 30 '
                 'simulations; noise models: the reported (DiMSum) error of each double, or a constant error calibrated to a between-replicate Pearson R of 0.87 or 0.92 (Faure et al. report 0.87-0.92) with the merged '
                 'value the mean of three replicates. Blocks above the floor: mean additive expectation f_A + f_B > -0.5. PDZ3: every position pair has one residue on one side (nine single-mutant backgrounds), so '
                 'no 3 x 3 block exists.', 'cell_mag_ceiling')


if __name__ == '__main__':
    sys.exit(main())
