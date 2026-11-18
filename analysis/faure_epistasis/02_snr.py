#!/usr/bin/env python
"""Signal and noise of the epistasis of the four Faure datasets.

Epistasis of a double: eps = f_AB - f_A - f_B (fitness scale, the benchmark's ``dddG_ML``), measurement error sigma_eps^2 = sigma_AB^2 + sigma_A^2 + sigma_B^2 (``sigma`` = DiMSum's count-based error of
the merged triplicate; independent across variants).  Two readings of the same doubles:

  raw          eps itself (``naive_rho`` of the benchmark): dominated by the assay's saturation, which is a function of the additive expectation x = f_A + f_B alone
  beyond-global r = eps - E[eps | x] (``esm_msr.epi_metrics.smooth_on``, 25 quantile bins), the residual every epistasis level above ``naive_rho`` is computed on

For each: the variance, the mean noise variance, the signal variance (variance - noise), SNR = signal / noise, the reliability = signal / variance and the ceiling correlation sqrt(reliability) of a perfect
predictor of the true epistasis against the measured one (also with a constant noise calibrated to a between-replicate Pearson R of 0.87 / 0.92, ``*_R0.87`` / ``*_R0.92``).  Each is given for all doubles and for the doubles above the floor (additive expectation x > -0.5: both substitutions tolerated), because the
reported errors of near-floor variants are larger than their actual scatter around the global curve (the noise of a dead variant cannot move it below the floor), which makes the signal of the
residual vanish in an all-doubles estimate.  The noise model is checked against the replicate reproducibility Faure et al. report (Pearson R = 0.87-0.92 between replicates).  Also:
the share of doubles with |r| / sigma_eps > 2 and > 3 against the share a noise-only Gaussian gives (4.55%, 0.27%), and the signal SD, by distance of the two residues.

  epistasis_snr   A: variance of eps by source (saturation / specific epistasis / noise); B: share of doubles with |z| > 3 by distance; C: signal SD and noise SD by distance; D: distribution of z.

Writes ``results/snr_summary.csv`` and ``results/snr_by_distance.csv`` (needs ``01_proximity.py``).

    python analysis/faure_epistasis/02_snr.py
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import faure as F
from esm_msr import epi_metrics as E

FLOOR_X = -0.5                  # doubles above the floor: additive expectation x = f_A + f_B above this
DBINS = [(0, 5, 'contact\n< 5 A'), (5, 8, '5-8 A'), (8, 12, '8-12 A'), (12, 99, '> 12 A')]


def stats(v, noise):
    sig = v.var() - noise
    return dict(var=v.var(), noise=noise, signal=sig, snr=sig / noise, reliability=sig / v.var(), ceiling=float(np.sqrt(max(sig, 0) / v.var())))


def main():
    rows, byd = [], []
    store = {}
    for ds in F.SETS:
        d0 = pd.read_pickle(F.CACHE / f'faure_doubles_{ds}.pkl')
        single, _, raw = F.load(ds)
        d0['g'] = E.smooth_on(d0.x.to_numpy(), d0.eps.to_numpy())
        d0['r'] = d0.eps - d0.g
        d0['z'] = d0.r / d0.sig_eps
        allv = raw[raw.NMUTS <= 2]
        vt = allv.truth.var() - (allv.sigma ** 2).mean()
        r_rep = vt / (vt + 3 * (allv.sigma ** 2).mean())            # merged triplicate: one replicate has about three times the error variance
        store[ds] = d0
        for stratum, m in (('all', np.ones(len(d0), bool)), ('above floor', (d0.x > FLOOR_X).to_numpy())):
            d = d0[m]
            noise = float((d.sig_eps ** 2).mean())
            a, b = stats(d.eps, noise), stats(d.r, noise)
            r = dict(set=ds, stratum=stratum, n=len(d), eps_sd=d.eps.std(), noise_sd=np.sqrt(noise), global_share=1 - d.r.var() / d.eps.var(), implied_replicate_r=r_rep, fitness_sd=allv.truth.std(),
                     median_sigma=allv.sigma.median(), z2=float((d.z.abs() > 2).mean()), z3=float((d.z.abs() > 3).mean()), z_sd=float(d.z.std()))
            r.update({f'raw_{k}': v for k, v in a.items()})
            r.update({f'resid_{k}': v for k, v in b.items()})
            for Rr in (0.87, 0.92):                                  # constant noise calibrated to a between-replicate Pearson R: var(eps noise) = 3 x var(merged noise) = var(true f) (1 - R) / R
                n_cal = vt * (1 - Rr) / Rr
                r[f'raw_snr_R{Rr}'] = (d.eps.var() - n_cal) / n_cal
                r[f'resid_snr_R{Rr}'] = (d.r.var() - n_cal) / n_cal
                r[f'resid_ceiling_R{Rr}'] = float(np.sqrt(max(d.r.var() - n_cal, 0) / d.r.var()))
            rows.append(r)
            for lo, hi, lab in DBINS:
                mm = (d.dmin >= lo) & (d.dmin < hi)
                sub = d[mm]
                if len(sub) < 30:
                    continue
                nz = float((sub.sig_eps ** 2).mean())
                byd.append(dict(set=ds, stratum=stratum, bin=lab.replace('\n', ' '), n=len(sub), resid_sd=sub.r.std(), noise_sd=np.sqrt(nz), signal_sd=np.sqrt(max(sub.r.var() - nz, 0)),
                                snr=(sub.r.var() - nz) / nz, z2=float((sub.z.abs() > 2).mean()), z3=float((sub.z.abs() > 3).mean()), z_sd=float(sub.z.std()), mean_abs_resid=float(sub.r.abs().mean())))
    S, Bd = pd.DataFrame(rows), pd.DataFrame(byd)
    S.to_csv(F.OUT / 'snr_summary.csv', index=False, float_format='%.4g')
    Bd.to_csv(F.OUT / 'snr_by_distance.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(S[['set', 'stratum', 'n', 'eps_sd', 'noise_sd', 'global_share', 'raw_snr', 'raw_ceiling', 'resid_signal', 'resid_snr', 'resid_reliability', 'resid_ceiling', 'z2', 'z3', 'z_sd', 'implied_replicate_r']].round(3).to_string(index=False))
    print(Bd[Bd.stratum == 'above floor'].round(3).to_string(index=False))
    Sa, Bda = S[S.stratum == 'above floor'].set_index('set'), Bd[Bd.stratum == 'above floor']
    with plt.rc_context(F.RC):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
        ax = axes[0, 0]
        x0 = np.arange(len(F.SETS))
        for k, (stratum, off) in enumerate((('all', -0.2), ('above floor', 0.2))):
            T = S[S.stratum == stratum].set_index('set').loc[F.SETS]
            tot = T.raw_var.to_numpy()
            glob = (T.global_share * T.raw_var).to_numpy()
            spec = T.resid_signal.clip(lower=0).to_numpy()
            nz = np.minimum(T.raw_noise, T.resid_var).to_numpy()               # when the residual is smaller than the noise estimate, all of it is attributed to noise
            xs = x0 + off
            ax.bar(xs, glob / tot, width=0.36, color='#9aa5b1', label='saturation (a function of f_A + f_B)' if k == 0 else None)
            ax.bar(xs, spec / tot, width=0.36, bottom=glob / tot, color='#eb6834', label='specific epistasis (signal)' if k == 0 else None)
            ax.bar(xs, nz / tot, width=0.36, bottom=(glob + spec) / tot, color='#444444', label='measurement noise' if k == 0 else None)
            for xi, g, s_, n_ in zip(xs, glob / tot, spec / tot, nz / tot):
                ax.text(xi, g / 2, f'{g:.0%}', ha='center', va='center', fontsize=7.6, color='white', fontweight='bold')
                if s_ > 0.03:
                    ax.text(xi, g + s_ / 2, f'{s_:.0%}', ha='center', va='center', fontsize=7.6, fontweight='bold')
                ax.text(xi, g + s_ + n_ / 2, f'{n_:.0%}', ha='center', va='center', fontsize=7.6, color='white', fontweight='bold')
            for xi in xs:
                ax.text(xi, 1.015, 'all' if stratum == 'all' else 'above\nfloor', ha='center', va='bottom', fontsize=6.8, color='#555555')
        ax.set_xticks(x0)
        ax.set_xticklabels([F.SHORT[d] for d in F.SETS])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('share of the variance of the epistasis eps')
        ax.set_title('A  What the variance of eps is made of')
        ax.legend(fontsize=7.6, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, handlelength=1.2)
        ax = axes[0, 1]
        for ds in F.SETS:
            t = Bda[Bda.set == ds]
            ax.plot(range(len(t)), t.z3 * 100, 'o-', color=F.COLOR[ds], label=F.SHORT[ds], ms=5)
        ax.axhline(2 * (1 - norm.cdf(3)) * 100, color='#444444', ls='--', lw=1)
        ax.text(len(DBINS) - 1.05, 2 * (1 - norm.cdf(3)) * 100 + 0.35, 'noise only (0.27%)', ha='right', fontsize=7.6, color='#444444')
        ax.set_xticks(range(len(DBINS)))
        ax.set_xticklabels([b[2] for b in DBINS])
        ax.set_ylabel('% of doubles with |r| / sigma_eps > 3')
        ax.set_title('B  Detectable epistasis by distance (above floor)')
        ax.legend(fontsize=7.6, handlelength=1.2, loc='center right')
        ax = axes[1, 0]
        for ds in F.SETS:
            t = Bda[Bda.set == ds]
            ax.plot(range(len(t)), t.signal_sd, 'o-', color=F.COLOR[ds], label=F.SHORT[ds], ms=5)
            ax.plot(range(len(t)), t.noise_sd, ':', color=F.COLOR[ds], lw=1.4)
        ax.set_xticks(range(len(DBINS)))
        ax.set_xticklabels([b[2] for b in DBINS])
        ax.set_ylabel('SD of the residual r (fitness units)')
        ax.set_title('C  Signal SD (solid), noise SD (dotted), above floor')
        ax.legend(fontsize=7.6, handlelength=1.2)
        ax = axes[1, 1]
        d = store[F.SETS[3]]
        d = d[d.x > FLOOR_X]
        grid = np.linspace(-6, 6, 61)
        for lo, hi, lab, col in ((0, 5, 'contact (< 5 A)', '#c0392b'), (12, 99, 'far (> 12 A)', '#2a78d6')):
            m = (d.dmin >= lo) & (d.dmin < hi)
            h, edges = np.histogram(d.z[m].clip(-6, 6), bins=grid, density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, h, color=col, lw=1.6, label=f'{lab}, n = {int(m.sum()):,}')
        ax.plot(grid, norm.pdf(grid), color='#444444', ls='--', lw=1.2, label='noise only, N(0, 1)')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel('z = r / sigma_eps')
        ax.set_ylabel('density')
        ax.set_title('D  z, SH3 (GRB2) binding, above floor')
        ax.legend(fontsize=7.6, handlelength=1.6)
        F.finish(fig, 'eps = f_AB - f_A - f_B, fitness scale; noise variance = mean of sigma_AB^2 + sigma_A^2 + sigma_B^2 (DiMSum merged-triplicate errors); saturation = the share of the variance of eps that is a function of '
                 'the additive expectation x = f_A + f_B (25-quantile-bin smoother); specific epistasis = variance of the residual r = eps - E[eps | x] minus the noise. B and C: doubles binned by the minimal heavy-atom '
                 'distance of the two residues (AlphaFold2 models). z = r / sigma_eps; a Gaussian noise-only z has 4.55% beyond 2 and 0.27% beyond 3.', 'epistasis_snr')


if __name__ == '__main__':
    sys.exit(main())
