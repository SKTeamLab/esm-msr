"""A soft floor between latent stability and what the assay reports.

The assay infers dG by comparing a variant's measured K50 against a K50 predicted for its UNFOLDED sequence from composition. A variant that
is unfolded anyway has K50 = its true unfolded K50, so its inferred dG is the error of that prediction for that sequence: a constant of the
library and of the protease. That constant is the library's practical FLOOR. No measurement can be reported below it, so a variant whose true
stability is far beneath it is reported at it, and an additive model of mutation effects sees the difference as "epistasis" that is only the
readout saturating.

The link states that explicitly:

    dG_obs = h(dG_latent),    h(z) = y_floor + tau * log(1 + exp((z - y_floor) / tau))

with ``dG_latent = dG_wt + latent``, where ``latent`` is the model's additive-in-effects prediction (the calibrated LLR sum). ``h`` is
``softplus`` with its knee at the floor: increasing everywhere, asymptotically the identity well above the floor (so latent and observed agree
for variants inside the dynamic range), and asymptotically flat at ``y_floor`` below it. Regression is done on the observed scale (through
``h``); ranking is done on the latent scale and is unaffected by ``h`` because ``h`` is monotone, so the model keeps an unsaturated opinion -
it can say a double is 7 kcal/mol less stable than the wild type even where the assay can only report that it is at the floor.

``tau`` sets how sharp the knee is; 0.59 kcal/mol is the default and is fixed unless ``--learn_link_tau``. At the knee itself
``h(y_floor) = y_floor + tau*ln 2``, about 0.41 kcal/mol above the floor.

``y_floor`` is DATA, not a parameter: each item carries its library's own floor estimate (``y_floor``, cached per item by
``data.MutationStabilityDataset``; see ``preprocess_megascale.library_floors``), so the link adds no per-library parameters. The floor of a
library is measured with a standard error around 0.05 against a between-library spread of about 0.31 kcal/mol, but a thin or unlucky library
can still produce a wild estimate, so in training the cached value is clipped to ``[floor_min, floor_max]`` (default -1.0 to 0.2, the range the
assay's own reported floor occupies). The clip bounds are shared across libraries and can be learned with ``--learn_link_floor_bounds``.

There is deliberately no ceiling term. The upper end of the range ('>5', hyperstable) is handled as censoring rather than saturation (see
``esm_msr.censoring``): those variants have no numeric value to regress onto, so a second soft clamp would be fitting a shape to data that is
not there.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TAU_DEFAULT = 0.59        # kcal/mol; the knee width that fits the measured pile-up at the floor
FLOOR_MIN_DEFAULT = -1.0  # the cached per-library floor is clipped into this band before use
FLOOR_MAX_DEFAULT = 0.2


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20.0 else y


def numpy_link(tau: float, floor_min: float, floor_max: float, global_floor: float = 0.0):
    """The same ``h`` as a numpy function of absolute dG, for metrics computed off the GPU (``FloorLink.numpy()``) and for evaluating a
    trained checkpoint. Takes the item's cached ``y_floor`` and clips it exactly as training did."""
    def h(z, y_floor=0.0):
        z = np.asarray(z, dtype=np.float64)
        f = np.where(np.isfinite(np.asarray(y_floor, dtype=np.float64)),
                     np.asarray(y_floor, dtype=np.float64), global_floor)
        f = np.clip(f, floor_min, floor_max)
        return f + tau * np.logaddexp(0.0, (z - f) / tau)
    return h


def numpy_link_from_state_dict(state_dict, prefix: str = 'link_head.'):
    """The numpy link of a checkpoint trained with ``--link`` (its ``link_head.*`` entries), or None when it has none. Inference returns the
    latent ddG; evaluation code uses this to put predictions on the assay's observed scale."""
    keys = {k: prefix + k for k in ('raw_tau', 'floor_min', 'floor_max', 'global_floor')}
    if not all(v in state_dict for v in keys.values()):
        return None
    f = lambda k: float(torch.as_tensor(state_dict[keys[k]], dtype=torch.float64))
    tau = float(F.softplus(torch.as_tensor(state_dict[keys['raw_tau']], dtype=torch.float64)))
    lo = f('floor_min')
    return numpy_link(tau, lo, max(f('floor_max'), lo + 1e-3),     # the same ordering FloorLink.bounds() enforces
                      f('global_floor'))


class FloorLink(nn.Module):
    """``h(z) = y_floor + tau * softplus((z - y_floor) / tau)``, with ``y_floor`` supplied per item and clipped to a learned-or-fixed band."""

    def __init__(self, tau: float = TAU_DEFAULT, floor_min: float = FLOOR_MIN_DEFAULT, floor_max: float = FLOOR_MAX_DEFAULT,
                 learn_tau: bool = False, learn_floor_bounds: bool = False,
                 global_floor_init: float = 0.0, learn_global_floor: bool = True):
        super().__init__()
        if not floor_max > floor_min:
            raise ValueError(f"floor_max ({floor_max}) must exceed floor_min ({floor_min})")
        if not tau > 0:
            raise ValueError(f"tau ({tau}) must be positive")
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

    def tau(self) -> torch.Tensor:
        """Softplus keeps it positive for any raw value, and round-trips the configured tau exactly."""
        return F.softplus(self.raw_tau)

    def bounds(self):
        """(floor_min, floor_max) with the order enforced, so a learned band cannot invert."""
        lo = self.floor_min
        return lo, torch.maximum(self.floor_max, lo + 1e-3)

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
        tau = self.tau()
        f = self.effective_floor(y_floor)
        return f + tau * F.softplus((z - f) / tau)

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
        return numpy_link(float(self.tau()), float(lo), float(hi), float(self.global_floor))

    @torch.no_grad()
    def summary(self) -> dict:
        lo, hi = self.bounds()
        return {'tau': float(self.tau()), 'floor_min': float(lo), 'floor_max': float(hi),
                'global_floor': float(torch.minimum(torch.maximum(self.global_floor, lo), hi))}
