"""A monotone saturating link between latent stability and what the assay reports.

The assay cannot report a dG outside roughly -1 to 5 kcal/mol, so a variant whose true stability is far below the floor is reported
near the floor, and the same happens at the ceiling. An additive model of mutation effects therefore sees "epistasis" that is just this
saturation (about 60-66% of the variance of ddG_AB - ddG_A - ddG_B in the measured doubles; see docs/anova_epistasis_report.md).

The model here is the standard one for global epistasis:

    observed dG  =  h( dG_wt + latent ),       latent = the model's additive-in-effects prediction (the calibrated LLR sum)

with one monotone ``h`` whose shape is shared by every library, and (with ``--link_floor_file``) a per-library offset on the floor knee alone.

The floor is per-library because of how the assay infers dG: it compares the measured K50 against a K50 predicted for the UNFOLDED sequence
by a composition model, so an unfolded variant reads dG = the error of that prediction for its sequence, a constant of the library and of the
protease. Measured over 96 libraries with at least 20 deeply destabilised doubles: median +0.02, SD 0.31 kcal/mol, standard error of one
library's estimate 0.05 (split-half reliability 0.97), uncorrelated between the two proteases (-0.10) and essentially unpredictable from dG_wt
(0.16). One shared floor fitted to that mixture comes out both too low and too soft, so ``lo_offset`` supplies the library's own floor as data
(like ``dG_wt``) and adds no parameters. Only the floor knee moves; the ceiling and the in-range identity are untouched.

The latent scale is unsaturated: it can say a double is 7 kcal/mol less stable than
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


MAX_FLOOR_OFFSET = 2.0       # a per-library floor offset is clamped to +-this (kcal/mol); the measured spread is an SD of 0.31
MIN_SPAN = 0.5               # and the shifted floor knee is never allowed within this of the ceiling


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20.0 else y


def numpy_link(lo: float, hi: float, tau_lo: float, tau_hi: float):
    """The same h as a numpy function of absolute dG, for metrics computed off the GPU (``MonotoneLink.numpy()``, or a dump's
    ``link_summary``). ``lo_offset`` moves the floor knee only (see ``MonotoneLink.forward``); 0 is the shared-floor link."""
    def h(z, lo_offset=0.0):
        z = np.asarray(z, dtype=np.float64)
        lo_e = np.minimum(lo + np.nan_to_num(np.clip(np.asarray(lo_offset, dtype=np.float64), -MAX_FLOOR_OFFSET, MAX_FLOOR_OFFSET)),
                          hi - MIN_SPAN)
        u = lo_e + tau_lo * np.logaddexp(0.0, (z - lo_e) / tau_lo)
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

    def _lo_eff(self, lo, hi, lo_offset):
        """The floor knee after a per-item offset. Only the floor moves: the ceiling stays at ``hi``, so a library's offset cannot act as a
        calibration shift for variants inside the range (there h(z) ~ z whatever the offset)."""
        if lo_offset is None:
            return lo
        off = torch.as_tensor(lo_offset, dtype=lo.dtype, device=lo.device)
        off = torch.nan_to_num(off, nan=0.0).clamp(-MAX_FLOOR_OFFSET, MAX_FLOOR_OFFSET)
        return torch.minimum(lo + off, hi - MIN_SPAN)

    def forward(self, z: torch.Tensor, lo_offset=None) -> torch.Tensor:
        lo, hi, tl, th = self.params()
        lo = self._lo_eff(lo, hi, lo_offset)
        u = lo + tl * F.softplus((z - lo) / tl)
        return hi - th * F.softplus((hi - u) / th)

    def obs_ddG(self, latent: torch.Tensor, dG_wt: torch.Tensor, bg_offset=0.0, lo_offset=None) -> torch.Tensor:
        """
        The ddG the assay would report: ``h(dG_wt + bg_offset + latent) - dG_wt``.

        ``dG_wt`` is the dG of the library's starting sequence. ``bg_offset`` is the (observed) ddG of a conditional item's
        background, ``ddG_AB - ddG(A|B)``, so a conditional item is scored as the double it came from; it is 0 for singles.
        ``lo_offset`` is the library's floor offset (``--link_floor_file``). NaN wherever ``dG_wt`` is unknown.
        """
        return self(dG_wt + bg_offset + latent, lo_offset) - dG_wt

    @torch.no_grad()
    def inverse(self, dG_obs: torch.Tensor, below_lo: float = 2.0, above_hi: float = 2.0, lo_offset=None) -> torch.Tensor:
        """
        h^-1: the latent dG whose observed value is ``dG_obs`` (absolute dG). Exact inside the range; where h is flat it is not invertible, so
        values at or below the floor map to at most ``below_lo`` kcal/mol below ``lo`` and values at or above the ceiling to at most
        ``above_hi`` above ``hi``. With ``lo_offset`` the floor knee is the library's own (so a censored variant's bound lands on the knee of
        the library it belongs to). No gradient (used for teacher-forcing targets, which must not train the link).
        """
        lo, hi, tl, th = self.params()
        lo = self._lo_eff(lo, hi, lo_offset)
        sp_inv = lambda v: v + torch.log(-torch.expm1(-v))              # log(exp(v) - 1), stable for large v
        v = (hi - dG_obs) / th
        u = hi - th * sp_inv(v.clamp(min=1e-6))
        w = (u - lo) / tl
        z = lo + tl * sp_inv(w.clamp(min=1e-6))
        z = torch.where(w <= 1e-6, lo - below_lo, z)                    # at or below the floor
        z = torch.where(v <= 1e-6, hi + above_hi, z)                    # at or above the ceiling
        return torch.minimum(torch.maximum(z, lo - below_lo), hi + above_hi)

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


class NonspecificTerm(nn.Module):
    """
    Nonspecific epistasis of a multi-mutant as a function of its additive expectation (``--nonspecific_term fx``): f(x), with
    x = dG_wt + the additive latent ddG (the sum of the single effects). Piecewise linear on fixed knots, constant beyond the outer knots,
    initialised at 0. It is added to a multi-mutant's latent before the link and never to a single, so the link keeps modelling the
    readout alone (the analysis behind it: docs/epistasis_metrics_report.html on devel, "Saturation and nonspecific epistasis").
    """

    def __init__(self, lo: float = -3.0, hi: float = 5.0, step: float = 0.5):
        super().__init__()
        n = int(round((hi - lo) / step)) + 1
        self.register_buffer('knots', torch.linspace(lo, hi, n))
        self.values = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.knots
        xc = x.clamp(float(k[0]), float(k[-1]))
        i = torch.clamp(torch.searchsorted(k, xc.contiguous()) - 1, 0, len(k) - 2)
        t = (xc - k[i]) / (k[i + 1] - k[i])
        return self.values[i] + t * (self.values[i + 1] - self.values[i])

    @torch.no_grad()
    def table(self) -> dict:
        return {'knots': [round(float(v), 4) for v in self.knots], 'values': [round(float(v), 5) for v in self.values]}
