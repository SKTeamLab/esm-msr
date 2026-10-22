"""Per-item confidence weights from the assay's own 95% confidence interval (``--ci_weight``).

Every measured row carries ``deltaG_95CI``, the full width of the 95% interval of its fitted dG (``data.MEAS_COLUMNS``;
for a numeric row the fit is the training label itself, ``dG_ML == deltaG``). A label's SD is that width over 3.92. The
weight of an item is its precision relative to the training population,

    w_i = [1 / (sigma_i^2 + tau^2)] / mean_j [1 / (sigma_j^2 + tau^2)],

so the weights average 1 over the items they apply to and ``--ci_weight`` redistributes the loss between items without
changing its overall scale (the learning rate and ``--lambda_*`` keep their meaning). The ranking loss uses a second normalisation of
the same weights, geometric mean 1: each score is offset by ``log w`` centred on the population's mean log-weight
(``item_offsets``) and each choice term scaled by the exponential of that offset, which keeps a list that mixes weighted doubles with
unweighted singles balanced (``losses.py`` says why) and costs a rank term e^(-mean log w) times larger on weighted items.

``tau`` is the variance that is NOT the label's: the model's own residual, which adds to every item alike. The inverse-
variance weights of a regression whose residual is label noise plus misfit are exactly ``1 / (sigma_i^2 + tau^2)``, so
``tau`` is the strength knob. ``tau = 0`` is pure inverse variance; ``tau`` at the model's real residual SD (about 0.4
kcal/mol on this data, against a median label SD of 0.04) makes the weights flat. The default sits in between, where the
weights still discriminate but no item can carry more than about 4x the average (see docs/ci_weighting.md).

Only UNCENSORED items are weighted. A censored item's label is a bound, not a value with an interval (and a range-bounded row's
``deltaG_95CI`` belongs to the unbounded fit the assay declined to report, four times wider than a fitted row's). An item with no
usable interval either (an external benchmark) gets weight 1: neither up- nor down-weighted.
"""
from typing import Dict

import numpy as np
import torch

CI_Z95 = 3.92               # a 95% interval spans +-1.96 SD, so its full width is 3.92 SD
CI_COLUMN = 'deltaG_95CI'   # the data.MEAS_COLUMNS entry the weights are read from
_MIN_VAR = 1e-8             # keeps tau = 0 finite on a degenerate zero-width interval


def label_sigma(ci_width):
    """The SD of a label whose 95% interval is ``ci_width`` wide."""
    return ci_width / CI_Z95


def raw_weight(ci_width, tau: float):
    """Unnormalised inverse variance, ``1 / (sigma^2 + tau^2)``, of a label with a ``ci_width``-wide interval."""
    var = label_sigma(ci_width) ** 2 + tau ** 2
    if isinstance(var, torch.Tensor):
        return 1.0 / var.clamp_min(_MIN_VAR)
    return 1.0 / np.maximum(var, _MIN_VAR)


def _usable(ci_width):
    return np.isfinite(ci_width) & (ci_width >= 0)


def reference_weight(ci_widths, tau: float) -> float:
    """The population mean of ``raw_weight`` over the usable intervals: what the weights are divided by."""
    w = np.asarray(ci_widths, dtype=np.float64)
    w = w[_usable(w)]
    if w.size == 0:
        raise ValueError("--ci_weight: no item of the weighted subsets carries a usable deltaG_95CI, so there is "
                         "nothing to weight (is the cache missing the measurement columns?)")
    return float(np.mean(raw_weight(w, tau)))


def reference_log_weight(ci_widths, tau: float, ref: float) -> float:
    """The population mean of ``log(w)`` for the normalised weights: what the ranking loss centres its score offsets on (negative,
    since the weights average 1 and the log is concave)."""
    w = np.asarray(ci_widths, dtype=np.float64)
    w = w[_usable(w)]
    return float(np.mean(np.log(raw_weight(w, tau) / ref))) if w.size else 0.0


def usable_mask(ci_width: torch.Tensor, applies: torch.Tensor) -> torch.Tensor:
    """Which items get a weight: those that ``applies`` to and whose interval is a usable (finite, non-negative) number."""
    ci = ci_width.float()
    return applies.bool() & torch.isfinite(ci) & (ci >= 0)


def item_weights(ci_width: torch.Tensor, applies: torch.Tensor, tau: float, ref: float) -> torch.Tensor:
    """
    Per-item loss weights for a batch: the normalised precision where the item ``applies`` and has a usable interval,
    1.0 everywhere else. ``ci_width`` is ``deltaG_95CI`` per item (NaN where the row has none).
    """
    usable = usable_mask(ci_width, applies)
    w = raw_weight(torch.where(usable, ci_width.float(), torch.zeros_like(ci_width.float())), tau) / float(ref)
    return torch.where(usable, w, torch.ones_like(w))


def item_offsets(weights: torch.Tensor, usable: torch.Tensor, logmean: float) -> torch.Tensor:
    """
    The score offsets the ranking loss adds: ``log w - logmean`` for a weighted item (``usable``), 0 for every other item. Centring on
    the population's mean log-weight makes the offsets average 0 over the weighted items: a Plackett-Luce list is only invariant to a
    shift of ALL its scores, and the training lists mix subsets (singles at 0, doubles weighted), so an uncentred offset would hand
    every double a systematic score shift against the singles of its list.
    """
    return torch.where(usable, torch.log(weights.clamp_min(_MIN_VAR)) - logmean, torch.zeros_like(weights))


def summarise(ci_widths, tau: float, ref: float) -> Dict[str, float]:
    """How hard the weights bite on a population: the quantiles, the extremes and the effective sample size as a share
    of the sample (1.0 = flat; the lower, the more the loss rests on few items). Logged once at start-up."""
    w = np.asarray(ci_widths, dtype=np.float64)
    w = w[_usable(w)]
    if w.size == 0:
        return {}
    v = raw_weight(w, tau) / ref
    return {'n': float(w.size), 'mean': float(v.mean()), 'p05': float(np.quantile(v, .05)),
            'p50': float(np.quantile(v, .50)), 'p95': float(np.quantile(v, .95)), 'max': float(v.max()),
            'ess_frac': float(v.sum() ** 2 / (v ** 2).sum() / v.size)}
