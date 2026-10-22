"""A floor-censored item's hinge holds exactly when its latent dG is at or below the floor (half unfolded).

With a softplus link the readout only reaches the floor itself as the latent goes to -inf. Bounding the observed
prediction AT the floor therefore never switched off: every floor-censored item kept a penalty and a push toward -inf
(mean hinge loss 0.36 on wt_r8a11 / dc18 validation singles, half of them already below the knee). The bound is now the
knee's reading, h(floor) = floor + tau*ln2.
"""
import math

import pytest
import torch

from esm_msr.link import SaturatingLink
from esm_msr.losses import hinge_regression_loss
from esm_msr.training import reachable_bounds

TAU = 0.59


def _setup(latent):
    link = SaturatingLink(tau=TAU, floor=0.0)
    dG_wt = torch.full_like(latent, 2.0)
    y_floor = torch.zeros_like(latent)                        # --assay_floor 0
    offset = -dG_wt                                           # a single: ddG = dG_meas - dG_wt
    raw = y_floor + offset                                    # what censoring.apply_item_censoring stores
    bound = reachable_bounds(link, raw, y_floor, offset)
    pred = link.obs_ddG(latent, dG_wt, y_floor)
    return link, bound, pred


def test_bound_is_the_knee_reading():
    _, bound, _ = _setup(torch.tensor([0.0]))
    assert float(bound) == pytest.approx(TAU * math.log(2) - 2.0, abs=1e-4)


def test_hinge_is_zero_at_or_below_the_floor_and_positive_above():
    latent_ddG = torch.tensor([-3.0, -2.5, -2.0, -1.5])       # latent dG = 2 + ddG: -1, -0.5, 0, +0.5
    _, bound, pred = _setup(latent_ddG)
    L = hinge_regression_loss(torch.nn.MSELoss(reduction='none'), pred, torch.zeros(4), bound, -torch.ones(4, dtype=torch.long))
    assert L[:3].abs().max() < 1e-6                           # at or below the knee: satisfied
    assert L[3] > 0.01                                        # above it: penalised


def test_the_old_floor_bound_was_never_satisfiable():
    latent_ddG = torch.tensor([-5.0])                         # latent dG -3, deep below the floor
    _, _, pred = _setup(latent_ddG)
    assert float(pred) > -2.0                                 # the reading never reaches the floor (bound 0 - dG_wt)
