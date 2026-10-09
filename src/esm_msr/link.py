"""A monotone saturating link between latent stability and what the assay reports.

The assay cannot report a dG outside roughly -1 to 5 kcal/mol, so a variant whose true stability is far below the floor is reported
near the floor, and the same happens at the ceiling. An additive model of mutation effects therefore sees "epistasis" that is just this
saturation (about 60-66% of the variance of ddG_AB - ddG_A - ddG_B in the measured doubles; see docs/anova_epistasis_report.md).

The model here is the standard one for global epistasis:

    observed dG  =  h( dG_wt + latent ),       latent = the model's additive-in-effects prediction (the calibrated LLR sum)

with one monotone ``h`` shared by every library. The latent scale is unsaturated: it can say a double is 7 kcal/mol less stable than
the wild type even though the assay can only report that it is at the floor. Regression is done on the observed scale (through ``h``);
ranking is done on the latent scale and is unaffected by ``h`` because ``h`` is monotone.

``h`` is two soft clamps in series: a soft floor at ``lo`` followed by a soft ceiling at ``hi``,

    u = lo + tau_lo * softplus((z - lo) / tau_lo)         # rises from lo, then follows z
    h(z) = hi - tau_hi * softplus((hi - u) / tau_hi)      # follows u, then levels off at hi

Each stage is increasing, so ``h`` is increasing for any positive temperatures. Inside the range it is close to the identity, so
latent and observed agree for variants well inside the dynamic range. ``lo`` and ``hi`` are where the knees sit. The upper plateau ``h(+inf)`` is exactly ``hi``; the lower
plateau ``h(-inf) = h(lo)`` equals ``lo - tau_hi * softplus(-(hi - lo) / tau_hi)``, which differs from ``lo`` only when ``tau_hi`` is not
small compared with the span (about 4e-3 at span 5.4, tau_hi 1). :meth:`summary` reports both as ``floor`` / ``ceiling``.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20.0 else y


def numpy_link(lo: float, hi: float, tau_lo: float, tau_hi: float):
    """The same h as a numpy function of absolute dG, for metrics computed off the GPU (``MonotoneLink.numpy()``, or a dump's
    ``link_summary``)."""
    def h(z):
        z = np.asarray(z, dtype=np.float64)
        u = lo + tau_lo * np.logaddexp(0.0, (z - lo) / tau_lo)
        return hi - tau_hi * np.logaddexp(0.0, (hi - u) / tau_hi)
    return h


def numpy_link_from_state_dict(state_dict, prefix: str = 'link_head.'):
    """The numpy link of a checkpoint trained with --link (its ``link_head.*`` parameters), or None when it has none. Inference returns the
    latent ddG; evaluation code uses this to put predictions on the assay's observed scale."""
    keys = [prefix + k for k in ('lo', 'raw_span', 'raw_tau_lo', 'raw_tau_hi')]
    if not all(k in state_dict for k in keys):
        return None
    sp = lambda v: float(F.softplus(torch.as_tensor(v, dtype=torch.float64)))
    lo = float(state_dict[keys[0]])
    return numpy_link(lo, lo + sp(state_dict[keys[1]]), sp(state_dict[keys[2]]) + 1e-4, sp(state_dict[keys[3]]) + 1e-4)


class MonotoneLink(nn.Module):
    def __init__(self, lo: float = -1.0, hi: float = 5.0, tau_lo: float = 0.5, tau_hi: float = 0.5,
                 learn_bounds: bool = True):
        super().__init__()
        if not hi > lo:
            raise ValueError(f"hi ({hi}) must exceed lo ({lo})")
        # the span hi - lo is kept positive by construction; temperatures likewise
        self.lo = nn.Parameter(torch.tensor(float(lo)), requires_grad=learn_bounds)
        self.raw_span = nn.Parameter(torch.tensor(_inv_softplus(float(hi - lo))), requires_grad=learn_bounds)
        self.raw_tau_lo = nn.Parameter(torch.tensor(_inv_softplus(float(tau_lo))))
        self.raw_tau_hi = nn.Parameter(torch.tensor(_inv_softplus(float(tau_hi))))

    def params(self):
        """(lo, hi, tau_lo, tau_hi) as tensors."""
        lo = self.lo
        hi = lo + F.softplus(self.raw_span)
        return lo, hi, F.softplus(self.raw_tau_lo) + 1e-4, F.softplus(self.raw_tau_hi) + 1e-4

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lo, hi, tl, th = self.params()
        u = lo + tl * F.softplus((z - lo) / tl)
        return hi - th * F.softplus((hi - u) / th)

    def obs_ddG(self, latent: torch.Tensor, dG_wt: torch.Tensor, bg_offset=0.0) -> torch.Tensor:
        """
        The ddG the assay would report: ``h(dG_wt + bg_offset + latent) - dG_wt``.

        ``dG_wt`` is the dG of the library's starting sequence. ``bg_offset`` is the (observed) ddG of a conditional item's
        background, ``ddG_AB - ddG(A|B)``, so a conditional item is scored as the double it came from; it is 0 for singles.
        NaN wherever ``dG_wt`` is unknown.
        """
        return self(dG_wt + bg_offset + latent) - dG_wt

    @torch.no_grad()
    def numpy(self):
        """This link as a numpy function of absolute dG (see ``numpy_link``)."""
        lo, hi, tl, th = (float(v) for v in self.params())
        return numpy_link(lo, hi, tl, th)

    @torch.no_grad()
    def summary(self) -> dict:
        lo, hi, tl, th = self.params()
        far = torch.tensor([-1e3, 1e3], device=lo.device)
        floor, ceiling = self(far).tolist()
        return {'lo': float(lo), 'hi': float(hi), 'tau_lo': float(tl), 'tau_hi': float(th), 'floor': floor, 'ceiling': ceiling}
