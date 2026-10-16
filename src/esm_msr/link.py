"""The assay's saturation as a monotone link, for the epistasis metrics only (no training code reads this module).

The assay cannot report a dG outside roughly -1 to 5 kcal/mol, so a variant whose true stability is far below the floor is reported
near the floor, and the same happens at the ceiling. An additive model of mutation effects therefore sees "epistasis" that is just this
saturation. The metrics (esm_msr.epi_metrics) can score a head through such a link, observed dG = h(dG_wt + latent); with no link they
take the predictions as observed, which is how the released model is trained.

``h`` is two soft clamps in series: a soft floor at ``lo`` followed by a soft ceiling at ``hi``,

    u = lo + tau_lo * softplus((z - lo) / tau_lo)         # rises from lo, then follows z
    h(z) = hi - tau_hi * softplus((hi - u) / tau_hi)      # follows u, then levels off at hi

Each stage is increasing, so ``h`` is increasing for any positive temperatures. Trainable links are on the devel branches, not here.
"""
import numpy as np


def numpy_link(lo: float, hi: float, tau_lo: float, tau_hi: float):
    """h as a numpy function of absolute dG (also how baseline/epi_from_dump.py rescores a dump that carries a ``link_summary``)."""
    def h(z):
        z = np.asarray(z, dtype=np.float64)
        u = lo + tau_lo * np.logaddexp(0.0, (z - lo) / tau_lo)
        return hi - tau_hi * np.logaddexp(0.0, (hi - u) / tau_hi)
    return h
