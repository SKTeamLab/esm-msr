"""A monotone saturating link between latent stability and what the assay reports.

The assay infers dG by comparing a variant's measured K50 against a K50 predicted for its UNFOLDED sequence from composition. A variant that
is unfolded anyway has K50 = its true unfolded K50, so its inferred dG is the error of that prediction for that sequence: a constant of the
library and of the protease. That constant is the library's practical FLOOR. No measurement can be reported below it, so a variant whose true
stability is far beneath it is reported at it, and an additive model of mutation effects sees the difference as "epistasis" that is only the
readout saturating.

The link states that explicitly, as a soft clamp on absolute dG:

    h(z) = f + tau * softplus((z - f) / tau)

with ``z = dG_wt + latent``, where ``latent`` is the model's additive-in-effects prediction (the calibrated LLR sum). ``h`` is increasing and,
above the floor, close to the identity. Regression is done on the observed scale (through ``h``); ranking is done on the latent scale and is
unaffected by ``h`` because ``h`` is monotone, so the model keeps an unsaturated opinion - it can say a double is 7 kcal/mol less stable than
the wild type even where the assay can only report the floor.

There is one floor (0.0, the unfolded state) and one knee width (0.59 kcal/mol, RT), both set rather than fitted. Per-library floors, a learned
floor, a learned width and an upper clamp were all tried; ``docs/trim_audit.md`` §2.2 has the evidence. ``numpy_link`` and
``numpy_link_from_state_dict`` still evaluate the older forms (per-library floors, a ceiling) so a checkpoint or dump from those runs can be rescored.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TAU_DEFAULT = 0.59        # kcal/mol; the knee width that fits the measured pile-up at the floor
FLOOR_BAND = 1e-3         # floor_max - floor_min: the clip band around the one floor. Kept at its old value, 0.001 (--link_floor_max 0.001)
CEILING_DEFAULT = 5.0     # the top of the assay's reported range ('>5'); see censoring.DG_RANGE_HIGH
CEILING_MIN_SPAN = 1e-3   # the ceiling stays at least this far above floor_max, as bounds() keeps the band ordered


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20.0 else y


def numpy_link(tau: float, floor_min: float, floor_max: float, global_floor: float = 0.0,
               ceiling: float = None, tau_hi: float = None):
    """The same ``h`` as a numpy function of absolute dG, for metrics computed off the GPU (``SaturatingLink.numpy()``) and for evaluating a
    trained checkpoint. Takes the item's cached ``y_floor`` and clips it exactly as training did. ``ceiling=None`` is the floor-only link."""
    th = tau if tau_hi is None else tau_hi

    def h(z, y_floor=0.0):
        z = np.asarray(z, dtype=np.float64)
        f = np.where(np.isfinite(np.asarray(y_floor, dtype=np.float64)),
                     np.asarray(y_floor, dtype=np.float64), global_floor)
        f = np.clip(f, floor_min, floor_max)
        u = f + tau * np.logaddexp(0.0, (z - f) / tau)
        if ceiling is None:
            return u
        return ceiling - th * np.logaddexp(0.0, (ceiling - u) / th)
    return h


def numpy_link_from_state_dict(state_dict, prefix: str = 'link_head.'):
    """The numpy link of a checkpoint trained with ``--link`` (its ``link_head.*`` entries), or None when it has none. Inference returns the
    latent ddG; evaluation code uses this to put predictions on the assay's observed scale."""
    keys = {k: prefix + k for k in ('raw_tau', 'floor_min', 'floor_max', 'global_floor')}
    if not all(v in state_dict for v in keys.values()):
        return None
    sp = lambda v: float(F.softplus(torch.as_tensor(v, dtype=torch.float64)))
    f = lambda k: float(torch.as_tensor(state_dict[keys[k]], dtype=torch.float64))
    tau = sp(state_dict[keys['raw_tau']])
    lo = f('floor_min')
    hi = max(f('floor_max'), lo + 1e-3)                      # the same ordering SaturatingLink.bounds() enforces
    ceiling = tau_hi = None
    if prefix + 'raw_ceiling_span' in state_dict:
        ceiling = hi + sp(state_dict[prefix + 'raw_ceiling_span']) + CEILING_MIN_SPAN
        tau_hi = sp(state_dict[prefix + 'raw_tau_hi']) if prefix + 'raw_tau_hi' in state_dict else tau
    return numpy_link(tau, lo, hi, f('global_floor'), ceiling, tau_hi)


class SaturatingLink(nn.Module):
    """
    A soft floor at ``floor`` with knee width ``tau``: ``h(z) = f + tau * softplus((z - f) / tau)``. Nothing in it is learned.

    Both numbers are set, not fitted: 0.0 is the physical floor (the unfolded state) and 0.59 kcal/mol is RT. Learning them was
    tried and dropped. A floor learned under a hinge is not identified (truncation pulls it up; scored on all items the best value
    is exactly 0), it drifted to -0.37 / -0.50 when free (dc3), and the 4-parameter link matched the floor-only one (dc7 against
    dc8). The values live in buffers named as the old parameters were (``raw_tau``, ``floor_min``, ``floor_max``, ``global_floor``),
    so a checkpoint's ``link_head.*`` entries load into this module and ``numpy_link_from_state_dict`` reads them unchanged.

    Items still carry ``y_floor`` (the dataset stamps ``floor`` on every one), and the methods still take it, so a dump written by this
    code is read by the same ``epi_metrics`` as an old one. The knee is ``clip(y_floor, floor_min, floor_max)``, which is ``floor``
    for every item of a run that uses one floor.
    """

    def __init__(self, tau: float = TAU_DEFAULT, floor: float = 0.0):
        super().__init__()
        if not tau > 0:
            raise ValueError(f"tau ({tau}) must be positive")
        self.register_buffer('raw_tau', torch.tensor(_inv_softplus(float(tau))))
        self.register_buffer('floor_min', torch.tensor(float(floor)))
        self.register_buffer('floor_max', torch.tensor(float(floor) + FLOOR_BAND))
        self.register_buffer('global_floor', torch.tensor(float(floor)))

    def tau(self) -> torch.Tensor:
        """Softplus keeps it positive for any raw value, and round-trips the configured tau exactly."""
        return F.softplus(self.raw_tau)

    def bounds(self):
        """(floor_min, floor_max) with the order enforced."""
        lo = self.floor_min
        return lo, torch.maximum(self.floor_max, lo + 1e-3)

    def effective_floor(self, y_floor) -> torch.Tensor:
        """The knee this item bends at: its ``y_floor`` clipped into the band, or ``global_floor`` where it carries none (NaN)."""
        lo, hi = self.bounds()
        f = torch.as_tensor(y_floor, dtype=lo.dtype, device=lo.device)
        f = torch.where(torch.isfinite(f), f, self.global_floor.expand_as(f))
        return torch.minimum(torch.maximum(f, lo), hi)

    def forward(self, z: torch.Tensor, y_floor) -> torch.Tensor:
        """``h`` on absolute dG."""
        f = self.effective_floor(y_floor)
        return f + self.tau() * F.softplus((z - f) / self.tau())

    def inverse(self, y: torch.Tensor, y_floor) -> torch.Tensor:
        """
        ``h^-1`` on absolute dG: the latent dG whose readout would be ``y``.

        Inverting the soft floor ``u = f + tau * softplus((z - f) / tau)`` gives ``z = f + tau * log(expm1((y - f) / tau))``.

        The floor is an ASYMPTOTE, so this is defined only for ``y > f``: a reading at or below the floor is consistent with any
        sufficiently negative latent value and carries no information about which. Those items return NaN rather than a clamped
        number, so a caller has to decide what to do with them instead of silently receiving a fabricated latent. That NaN set is
        essentially the one ``tf_saturated`` already drops.
        """
        f = self.effective_floor(y_floor)
        v = (y - f) / self.tau()
        # v > 20: log(expm1(v)) == v to float precision, and expm1 overflows past ~88 in float32.
        safe = torch.log(torch.expm1(v.clamp(min=1e-12)))
        z = f + self.tau() * torch.where(v > 20.0, v, safe)
        return torch.where(v > 0, z, torch.full_like(z, float('nan')))

    def latent_ddG(self, observed: torch.Tensor, dG_wt: torch.Tensor, y_floor) -> torch.Tensor:
        """
        The exact inverse of :meth:`obs_ddG`: the latent ddG that would be REPORTED as ``observed``.

        Use it to put a measured label on the latent scale before it is added to a latent prediction. Note that the inverse is
        nonlinear, so ``latent_ddG(a + b) != latent_ddG(a) + latent_ddG(b)``: a sum of two measured singles has to be de-linked ONE
        SINGLE AT A TIME and then summed, not de-linked as a sum.
        """
        return self.inverse(dG_wt + observed, y_floor) - dG_wt

    def obs_ddG(self, latent: torch.Tensor, dG_wt: torch.Tensor, y_floor) -> torch.Tensor:
        """
        The ddG the assay would report: ``h(dG_wt + latent) - dG_wt``. ``dG_wt`` is the dG of the library's starting sequence; NaN
        wherever it is unknown, which is how callers find the items the observed scale cannot be computed for.
        """
        return self(dG_wt + latent, y_floor) - dG_wt

    @torch.no_grad()
    def numpy(self):
        """This link as a numpy function of absolute dG (see ``numpy_link``)."""
        lo, hi = self.bounds()
        return numpy_link(float(self.tau()), float(lo), float(hi), float(self.global_floor))

    @torch.no_grad()
    def summary(self) -> dict:
        lo, hi = self.bounds()
        return {'tau': float(self.tau()), 'floor_min': float(lo), 'floor_max': float(hi),
                'global_floor': float(torch.minimum(torch.maximum(self.global_floor, lo), hi))}
