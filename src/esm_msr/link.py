"""A monotone saturating link between latent stability and what the assay reports.

The assay infers dG by comparing a variant's measured K50 against a K50 predicted for its UNFOLDED sequence from composition. A variant that
is unfolded anyway has K50 = its true unfolded K50, so its inferred dG is the error of that prediction for that sequence: a constant of the
library and of the protease. That constant is the library's practical FLOOR. No measurement can be reported below it, so a variant whose true
stability is far beneath it is reported at it, and an additive model of mutation effects sees the difference as "epistasis" that is only the
readout saturating. The same happens at the top of the range, where the assay cannot resolve a dG above about 5 kcal/mol.

The link states that explicitly, as two soft clamps in series on absolute dG (the construction used on ``devel``):

    u    = f + tau_lo * softplus((z - f) / tau_lo)        # rises from the floor, then follows z
    h(z) = c - tau_hi * softplus((c - u) / tau_hi)        # follows u, then levels off at the ceiling

with ``z = dG_wt + latent``, where ``latent`` is the model's additive-in-effects prediction (the calibrated LLR sum). Each stage is
increasing, so ``h`` is increasing for any positive temperatures, and inside the range it is close to the identity. Regression is done on the
observed scale (through ``h``); ranking is done on the latent scale and is unaffected by ``h`` because ``h`` is monotone, so the model keeps
an unsaturated opinion - it can say a double is 7 kcal/mol less stable than the wild type even where the assay can only report the floor.

``f``, the floor, is DATA rather than a parameter: each item carries its library's own estimate (``y_floor``, cached per item by
``data.MutationStabilityDataset``), clipped in training to ``[floor_min, floor_max]`` (default -1.0 to 0.2, the range the assay's own reported
floor occupies) because a thin or unlucky library can produce a wild estimate. A library with no floor evidence carries NaN and falls back to
the learned ``global_floor``, which is identifiable precisely from that population.

``c``, the ceiling, is a single learned scalar with no per-library counterpart, because nothing in the data locates a per-library ceiling the
way the pile-up at the bottom locates a floor. It is parameterised as ``floor_max + softplus(raw) + 1e-3`` so it can never reach, let alone cross,
any library's floor whatever the band does - without the margin a learned span can underflow to exactly zero in
float32 and collapse the link to a constant. Passing ``ceiling=None`` leaves the second clamp out entirely and recovers the floor-only link.

``tau_lo`` and ``tau_hi`` set how sharp each knee is. 0.59 kcal/mol fits the measured pile-up at the floor; there is no independently known
value for the top, so ``tau_hi`` starts there too. With ``--learn_link_tau`` both are free, and ``--link_tau_penalty`` adds
``lambda * ((tau_lo - 0.59)^2 + (tau_hi - 0.59)^2)`` to the loss so they are anchored at that value and drift only where the data pays for it.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TAU_DEFAULT = 0.59        # kcal/mol; the knee width that fits the measured pile-up at the floor
FLOOR_MIN_DEFAULT = -1.0  # the cached per-library floor is clipped into this band before use
FLOOR_MAX_DEFAULT = 0.2
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
    """A soft floor at the library's own ``y_floor`` followed by an optional soft ceiling at a learned scalar."""

    def __init__(self, tau: float = TAU_DEFAULT, floor_min: float = FLOOR_MIN_DEFAULT, floor_max: float = FLOOR_MAX_DEFAULT,
                 learn_tau: bool = False, learn_floor_bounds: bool = False,
                 global_floor_init: float = 0.0, learn_global_floor: bool = True,
                 ceiling: float = None, learn_ceiling: bool = True, tau_hi: float = None):
        super().__init__()
        if not floor_max > floor_min:
            raise ValueError(f"floor_max ({floor_max}) must exceed floor_min ({floor_min})")
        if not tau > 0:
            raise ValueError(f"tau ({tau}) must be positive")
        if ceiling is not None and not ceiling > floor_max:
            raise ValueError(f"ceiling ({ceiling}) must exceed floor_max ({floor_max})")
        # tau is kept positive by construction; the band is kept ordered in `bounds()`.
        self.raw_tau = nn.Parameter(torch.tensor(_inv_softplus(float(tau))), requires_grad=learn_tau)
        self.floor_min = nn.Parameter(torch.tensor(float(floor_min)), requires_grad=learn_floor_bounds)
        self.floor_max = nn.Parameter(torch.tensor(float(floor_max)), requires_grad=learn_floor_bounds)
        # The knee for items whose library has no floor of its own. Learned, because it is identifiable: it is the
        # one value that best explains how far the strongly destabilised predictions have to be compressed to match
        # what the assay reported, pooled over every library that cannot supply its own. Initialised at the median
        # of the libraries that CAN (a measured quantity) and clipped to the same band, so it cannot run off to
        # absorb a calibration bias.
        gi = float(global_floor_init) if np.isfinite(global_floor_init) else 0.5 * (float(floor_min) + float(floor_max))
        self.global_floor = nn.Parameter(torch.tensor(gi), requires_grad=learn_global_floor)

        self.has_ceiling = ceiling is not None
        if self.has_ceiling:
            # Parameterised as a positive span above floor_max, so the ceiling can never cross a library's floor
            # however the band moves. tau_hi has no independently known value, so it starts where tau_lo does.
            self.raw_ceiling_span = nn.Parameter(
                torch.tensor(_inv_softplus(float(ceiling) - float(floor_max) - CEILING_MIN_SPAN)),
                requires_grad=learn_ceiling)
            self.raw_tau_hi = nn.Parameter(torch.tensor(_inv_softplus(float(tau if tau_hi is None else tau_hi))),
                                           requires_grad=learn_tau)

    def tau(self) -> torch.Tensor:
        """Softplus keeps it positive for any raw value, and round-trips the configured tau exactly."""
        return F.softplus(self.raw_tau)

    def tau_hi(self) -> torch.Tensor:
        """The ceiling's knee width; equal to ``tau()`` when there is no ceiling."""
        return F.softplus(self.raw_tau_hi) if self.has_ceiling else self.tau()

    def bounds(self):
        """(floor_min, floor_max) with the order enforced, so a learned band cannot invert."""
        lo = self.floor_min
        return lo, torch.maximum(self.floor_max, lo + 1e-3)

    def ceiling(self):
        """The ceiling as a tensor, or None for the floor-only link."""
        if not self.has_ceiling:
            return None
        return self.bounds()[1] + F.softplus(self.raw_ceiling_span) + CEILING_MIN_SPAN

    def tau_penalty(self, weight: float) -> torch.Tensor:
        """``weight * sum (tau - 0.59)^2`` over the temperatures that are actually free; 0 when none are."""
        terms = [t for t in (self.raw_tau, self.raw_tau_hi if self.has_ceiling else None)
                 if t is not None and t.requires_grad]
        if not terms or not weight:
            return torch.zeros((), device=self.raw_tau.device, dtype=self.raw_tau.dtype)
        taus = [F.softplus(t) for t in terms]
        return weight * sum(((t - TAU_DEFAULT) ** 2 for t in taus))

    def effective_floor(self, y_floor) -> torch.Tensor:
        """
        The knee this item bends at, clipped into the band.

        A library that showed evidence of a floor supplies its own (``y_floor``, data). One that did not supplies
        NaN, and those items fall back to the shared ``global_floor``, which is learned from exactly that
        population. So the floor is measured where it can be and estimated where it cannot, and the two are never
        mixed: a library with its own floor contributes no gradient to the global one.
        """
        lo, hi = self.bounds()
        f = torch.as_tensor(y_floor, dtype=lo.dtype, device=lo.device)
        f = torch.where(torch.isfinite(f), f, self.global_floor.expand_as(f))
        return torch.minimum(torch.maximum(f, lo), hi)

    def forward(self, z: torch.Tensor, y_floor) -> torch.Tensor:
        """``h`` on absolute dG."""
        f = self.effective_floor(y_floor)
        u = f + self.tau() * F.softplus((z - f) / self.tau())
        c = self.ceiling()
        if c is None:
            return u
        th = self.tau_hi()
        return c - th * F.softplus((c - u) / th)

    def obs_ddG(self, latent: torch.Tensor, dG_wt: torch.Tensor, y_floor, bg_offset=0.0) -> torch.Tensor:
        """
        The ddG the assay would report: ``h(dG_wt + bg_offset + latent) - dG_wt``.

        ``dG_wt`` is the dG of the library's starting sequence. ``bg_offset`` is the (observed) ddG of a conditional item's background,
        ``ddG_AB - ddG(A|B)``, so a conditional item saturates as the double it came from; it is 0 for singles. NaN wherever ``dG_wt`` is
        unknown, which is how callers find the items the observed scale cannot be computed for.
        """
        return self(dG_wt + bg_offset + latent, y_floor) - dG_wt

    @torch.no_grad()
    def numpy(self):
        """This link as a numpy function of absolute dG (see ``numpy_link``)."""
        lo, hi = self.bounds()
        c = self.ceiling()
        return numpy_link(float(self.tau()), float(lo), float(hi), float(self.global_floor),
                          None if c is None else float(c), float(self.tau_hi()))

    @torch.no_grad()
    def summary(self) -> dict:
        lo, hi = self.bounds()
        out = {'tau': float(self.tau()), 'floor_min': float(lo), 'floor_max': float(hi),
               'global_floor': float(torch.minimum(torch.maximum(self.global_floor, lo), hi))}
        c = self.ceiling()
        if c is not None:
            out.update({'ceiling': float(c), 'tau_hi': float(self.tau_hi())})
        return out
