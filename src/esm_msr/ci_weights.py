"""Per-item label uncertainty, and the loss weights built from it (``--ci_weight``).

Every measured variant has TWO independent estimates of its stability, one per protease (``deltaG_t``, ``deltaG_c``), and the training label
``dG_ML`` is their joint fit. That is enough to estimate the label's error from the data itself:

* **Statistical part.** ``deltaG_95CI`` is the full width of the 95% interval of the fit, so ``s = CI / 3.92``. It is tiny (median 0.035
  kcal/mol) and, measured against the held-out error of a trained model, nearly uninformative: it is the fit's precision, not the assay's error.
* **Disagreement part.** Each protease's reading is ``mu + e``, with an error ``e`` that the fit's interval does not cover (what the assay calls
  composition error: each protease's own model of the unfolded state). The two errors are independent, so their difference ``g`` estimates the
  per-protease error variance, ``E[g^2] = 2 u^2``, and the mean of the two readings has error variance ``u^2 / 2 = g^2 / 4``. That is
  ``kappa = 1/4`` with nothing fitted; calibrated against a trained model's held-out squared error on in-range items it comes out at
  0.263 +- 0.015 (docs/ci_weighting.md section 2). Only the part of ``g`` that is the VARIANT's own counts: a protease offset shared by a whole
  library cancels in ``ddG = dG - dG_wt``, so ``g`` is taken relative to the library's median signed gap over its in-range variants. Left
  uncentred, the gap predicts nothing (library offsets dominate it). ``g`` is independent of the error of the mean (for equal variances), so
  weighting by it does not select on the label's noise - which weighting by the reading would.
* **Depth part.** For a double, the additive expectation ``x = dG_wt + ddG_A + ddG_B`` is known from other measurements, and the disagreement
  grows steeply with how far below the floor it sits (capped squared deviation 0.09 at x ~ 1.5, 1.3 at x ~ -0.75, 3.6-4.5 below -2.5). ``m(x)``,
  the mean squared deviation at that depth (tabulated below, fitted on the training split by analysis_scripts/ci_weighting_fit.py), is a prior
  for the double's own ``g^2``: the two are averaged, an empirical-Bayes estimate with a prior worth one observation. Unlike the reading, ``x``
  comes from OTHER measurements (the two singles), so a weight built on it does not select on this item's own noise.

    L_i = s_i^2 + (1/4) * (g_i^2 [+ m(x_i)] / 2)          label variance, kcal/mol^2

The loss weight is that variance's inverse after adding ``tau^2``, the part of the residual that is the MODEL's (about 0.25 for a trained head on
clean items), normalised to mean 1 over the training items:

    w_i = [1 / (tau^2 + L_i)] / mean_j [1 / (tau^2 + L_j)].

That is the optimal weight of a regression whose residual is label error plus a model error common to all items. The ranking loss takes the
same weights as a score offset ``log w - mean(log w)`` plus a term scale ``w`` (``losses.py``). A censored item (a bound, not a value) and an item
without readings keep weight 1.
"""
from typing import Dict, List

import numpy as np
import torch

CI_Z95 = 3.92               # the full width of a 95% interval, in SDs
GAP_CAP = 3.0               # a deviation beyond this is capped: dG_ML is clamped to [-1, 5], so no label can be wrong by more than a few kcal/mol
KAPPA = 0.25                # variance of the mean of two independent readings, per unit squared disagreement
INRANGE = (1.0, 4.0)        # readings that define a library's protease offset: in range, away from both ends of the assay
MIN_REF_ITEMS = 10          # fewer in-range variants than this and the library's offset is not estimated (taken as 0)
_MIN_VAR = 1e-6

# E[min(|g|, 3)^2 | x] for doubles, g the library-centred protease deviation, x = dG_wt + ddG_A + ddG_B the additive expectation of dG. Fitted on the
# training split (analysis_scripts/ci_weighting_fit.py); piecewise linear between the centres, flat beyond them.
DEPTH_X = (-4.5, -3.5, -2.5, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.5, 3.5)
DEPTH_M = (4.522, 4.245, 3.609, 2.358, 1.347, 0.581, 0.272, 0.154, 0.090, 0.091, 0.156)


def depth_prior(x):
    """The expected squared protease deviation of a double whose additive expectation is ``x`` (NaN where ``x`` is)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x), np.interp(np.clip(np.nan_to_num(x, nan=0.0), DEPTH_X[0], DEPTH_X[-1]), DEPTH_X, DEPTH_M), np.nan)


def signed_gap(dG_t, dG_c):
    return np.asarray(dG_t, dtype=np.float64) - np.asarray(dG_c, dtype=np.float64)


def library_offset(signed_gaps, readings) -> float:
    """A library's protease offset: the median signed gap over its in-range variants (0.0 if it has too few to say)."""
    g, r = np.asarray(signed_gaps, dtype=np.float64), np.asarray(readings, dtype=np.float64)
    ok = np.isfinite(g) & np.isfinite(r) & (r > INRANGE[0]) & (r < INRANGE[1])
    return float(np.median(g[ok])) if ok.sum() >= MIN_REF_ITEMS else 0.0


def label_variance(ci_width, dG_t, dG_c, offset, x=None, prior: bool = True):
    """
    The variance of a label, ``L = s^2 + g^2 / 4`` with the library-centred deviation ``g`` (capped at ``GAP_CAP``). Where the additive
    expectation ``x`` is given (doubles) and ``prior`` is on, ``g^2`` is averaged with the depth prior ``m(x)``. NaN where the readings are
    missing, which the weights treat as "no information": weight 1.
    """
    ci = np.asarray(ci_width, dtype=np.float64)
    g = np.minimum(np.abs(signed_gap(dG_t, dG_c) - offset), GAP_CAP)
    g2 = g * g
    if prior and x is not None:
        m = depth_prior(x)
        g2 = np.where(np.isfinite(m), 0.5 * (g2 + m), g2)
    s2 = np.where(np.isfinite(ci) & (ci >= 0), (ci / CI_Z95) ** 2, 0.0)
    return np.where(np.isfinite(g2), s2 + KAPPA * g2, np.nan)


def stamp_label_variance(items: List[dict], col: Dict[str, int]) -> Dict[str, float]:
    """
    Set ``item['label_var']`` on every item of ONE library (``col`` maps the ``meas`` column names to their index) and return the offset used.
    The library's protease offset is estimated from all its items before any subset filter, so it does not depend on which subsets a run keeps.
    """
    def read(it, name):
        a = it.get('meas')
        return float('nan') if a is None else float(a[col[name]])
    t = np.array([read(i, 'deltaG_t') for i in items])
    c = np.array([read(i, 'deltaG_c') for i in items])
    r = np.array([read(i, 'dG_ML') for i in items])
    ci = np.array([read(i, 'deltaG_95CI') for i in items])
    off = library_offset(signed_gap(t, c), r)
    x = np.array([(i.get('dG_wt', np.nan) + i.get('ddG_additive', np.nan)) if i.get('subset_type') == 'double' else np.nan for i in items])
    lv = label_variance(ci, t, c, off, x)
    for it, v in zip(items, lv):
        it['label_var'] = float(v)
    return {'offset': off}


def reference_weight(label_vars, tau: float) -> float:
    """The mean of ``1 / (tau^2 + L)`` over the usable items: what the weights are divided by."""
    v = np.asarray(label_vars, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        raise ValueError("--ci_weight: no training item carries a label variance, so there is nothing to weight (a cache without the "
                         "protease readings, `meas`?)")
    return float(np.mean(1.0 / np.maximum(tau * tau + v, _MIN_VAR)))


def reference_log_weight(label_vars, tau: float, ref: float) -> float:
    """The mean of ``log w`` over the usable items: what the ranking loss centres its score offsets on (negative, the log being concave)."""
    v = np.asarray(label_vars, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    return float(np.mean(np.log(1.0 / np.maximum(tau * tau + v, _MIN_VAR) / ref))) if v.size else 0.0


def usable_mask(label_var: torch.Tensor, applies: torch.Tensor) -> torch.Tensor:
    """Which items get a weight: those that ``applies`` to (uncensored) and carry a finite, non-negative label variance."""
    v = label_var.float()
    return applies.bool() & torch.isfinite(v) & (v >= 0)


def item_weights(label_var: torch.Tensor, applies: torch.Tensor, tau: float, ref: float) -> torch.Tensor:
    """Per-item loss weights for a batch: ``1 / (tau^2 + L)`` over ``ref`` where usable, 1.0 everywhere else."""
    usable = usable_mask(label_var, applies)
    v = torch.where(usable, label_var.float(), torch.zeros_like(label_var.float()))
    w = 1.0 / torch.clamp(tau * tau + v, min=_MIN_VAR) / float(ref)
    return torch.where(usable, w, torch.ones_like(w))


def item_offsets(weights: torch.Tensor, usable: torch.Tensor, logmean: float) -> torch.Tensor:
    """
    The score offsets the ranking loss adds: ``log w - logmean`` for a weighted item, 0 for every other item. Centring on the population's mean
    log-weight makes the offsets average 0 over the weighted items, so a bound sitting in a list (offset 0) is neither favoured nor penalised
    against them.
    """
    return torch.where(usable, torch.log(weights.clamp_min(_MIN_VAR)) - logmean, torch.zeros_like(weights))


def summarise(label_vars, tau: float, ref: float) -> Dict[str, float]:
    """How hard the weights bite on a population: quantiles, extremes and the effective sample size as a share of the sample (1.0 = flat),
    and the weighting efficiency E[tau^2 + L] * E[1 / (tau^2 + L)] (the most variance that optimal weights can remove: 1.0 = nothing)."""
    v = np.asarray(label_vars, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0:
        return {}
    w = 1.0 / np.maximum(tau * tau + v, _MIN_VAR) / ref
    return {'n': float(v.size), 'mean': float(w.mean()), 'p05': float(np.quantile(w, .05)), 'p50': float(np.quantile(w, .50)),
            'p95': float(np.quantile(w, .95)), 'max': float(w.max()), 'ess_frac': float(w.sum() ** 2 / (w ** 2).sum() / w.size),
            'efficiency': float(np.mean(tau * tau + v) * np.mean(1.0 / (tau * tau + v)))}
