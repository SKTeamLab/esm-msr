"""The post-hoc Tobit link of FINAL_p0's latent ddG, for any pooled (across-domain) statistic (``link.py`` is the older devel floor link).

``linked(latent, dG_wt)`` = ``h(dG_wt + latent) - dG_wt`` with ``esm_msr.val_stats.tobit_link`` and the constants FINAL_p0 s1-s3 were fit with (sigma_y 0.28, mu_f -0.14, sigma_f 0.5).  The link is monotone and
dG_wt is constant inside a domain, so a per-domain rank (the domain-mean rho) cannot change; a rho pooled over domains can, because the link saturates the predicted dG near the assay's range.  Apply it to
ESM-MSR (all heads, every seed) before any pooled rho / RMSE against ddG_ML; baselines are read raw.  On the Tsuboyama capped test set it moves ESM-MSR p0's pooled rho 0.748 -> 0.757
(``11j_overall_vs_transfer.py``); the MGnify loader (``mgnify.load_mgnify``) applies the same link (``preds['<head>~']``).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402

TOBIT = dict(sigma_y=0.28, mu_f=-0.14, sigma_f=0.5)


def linked(latent, dg_wt):
    """Tobit-linked ddG of ``latent`` (n,) or (n_seeds, n) given the per-row wild-type dG ``dg_wt`` (n,)."""
    dg = np.asarray(dg_wt, float)
    return tobit_link(**TOBIT)(dg + np.asarray(latent, float)) - dg
