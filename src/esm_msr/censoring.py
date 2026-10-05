"""One place for every kind of censoring the training data can carry.

An item is *censored* when its measurement only bounds the true value:

* ``cens = -1``: the true value is at or below ``cens_bound`` (lower-censored).
* ``cens = +1``: the true value is at or above ``cens_bound`` (upper-censored).
* ``cens = 0``: an ordinary measurement.

Two sources set it, recorded in ``cens_src``:

* ``SRC_RANGE`` (1): the assay reports a variant whose dG is confidently outside its dynamic range as the text
  ``<-1`` (dead: unfolded beyond measurement) or ``>5`` (hyperstable). These carry no usable value, only the bound.
* ``SRC_FLOOR`` (2): ``--censor_floor F``. A numeric measurement at or below F is pinned near the assay floor, so
  its order among its peers is not trusted.

Losses use the same two fields everywhere: the rank losses tie censored members at the bottom or top of their list
(``ListMLELoss.forward_censored``) and the regression losses penalise only a prediction on the wrong side of the bound
(:func:`censored_regression_loss`). Bounds are stored on the item's own ``ddG`` scale, so one number serves WT-context
singles, doubles and conditional (``ddG(A|B)``) items alike.
"""
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

CENS_LO, CENS_NONE, CENS_HI = -1, 0, 1
SRC_NONE, SRC_RANGE, SRC_FLOOR = 0, 1, 2

# the assay's reported dynamic range (kcal/mol); see MutationStabilityDataset.DG_FLOOR / DG_CEILING
DG_RANGE_LOW, DG_RANGE_HIGH = -1.0, 5.0
LOW_TAG, HIGH_TAG = '<-1', '>5'


def range_censoring(dG_raw, dG_wt) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse the raw ``dG_ML`` column.

    ``dG_raw`` is the column as read (numbers and the strings ``<-1`` / ``>5`` / ``-``); ``dG_wt`` is the per-row
    wild-type dG of the row's library (NaN when unknown). Returns ``(cens, ddG)``: ``cens`` is -1 for ``<-1``, +1
    for ``>5`` and 0 otherwise, and ``ddG`` is the numeric ddG for ordinary rows and the *bound* on the ddG scale
    (-1 - dG_wt, or 5 - dG_wt) for censored ones; NaN where it cannot be computed (``-``, unknown dG_wt).
    """
    raw = np.asarray(dG_raw, dtype=object)
    wt = np.asarray(dG_wt, dtype=float)
    as_str = np.array([str(x).strip() for x in raw])
    num = np.array([_to_float(x) for x in raw])
    cens = np.zeros(len(raw), dtype=np.int8)
    cens[as_str == LOW_TAG] = CENS_LO
    cens[as_str == HIGH_TAG] = CENS_HI
    dG = num.copy()
    dG[cens == CENS_LO] = DG_RANGE_LOW
    dG[cens == CENS_HI] = DG_RANGE_HIGH
    return cens, dG - wt


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def apply_item_censoring(data: Iterable[Dict[str, Any]], include_out_of_range: bool,
                         censor_floor: Optional[float] = None) -> Tuple[list, Dict[str, int]]:
    """
    Finalise the censoring fields of one library's items and return ``(kept_items, counts)``.

    * Items from out-of-range rows are dropped unless ``include_out_of_range``.
    * With ``censor_floor``, flip-column items whose numeric measured dG is at or below it become lower-censored.
    * ``cens_bound`` (the item's own ddG scale) is derived as ``dG_bound + (ddG - dG_meas)``: the offset between an
      item's label and the measured dG it came from is constant, and for a conditional item it already carries the
      partner's single.
    """
    kept, counts = [], {'range_lo': 0, 'range_hi': 0, 'floor': 0, 'dropped_range': 0}
    for item in data:
        cens, src = int(item.get('cens', 0)), int(item.get('cens_src', 0))
        if cens != 0 and src == SRC_RANGE and not include_out_of_range:
            counts['dropped_range'] += 1
            continue
        if cens == 0 and censor_floor is not None and item.get('flip_key'):
            dG = item.get('dG_meas', float('nan'))
            if np.isfinite(dG) and dG <= censor_floor:
                item['cens'], item['cens_src'], item['dG_bound'] = CENS_LO, SRC_FLOOR, float(censor_floor)
                cens, src = CENS_LO, SRC_FLOOR
                counts['floor'] += 1
        if cens != 0:
            if src == SRC_RANGE:
                counts['range_lo' if cens < 0 else 'range_hi'] += 1
            dG_meas, dG_bound = item.get('dG_meas', float('nan')), item.get('dG_bound', float('nan'))
            ddG = item.get('ddG', float('nan'))
            item['cens_bound'] = float(dG_bound + (ddG - dG_meas)) if np.isfinite(dG_meas) and np.isfinite(dG_bound) else float('nan')
        else:
            item.setdefault('cens', 0)
            item['cens_bound'] = float('nan')
        kept.append(item)
    return kept, counts


def censored_regression_loss(crit, pred: torch.Tensor, bound: torch.Tensor, cens: torch.Tensor) -> torch.Tensor:
    """
    One-sided regression for censored items, element-wise (same reduction-'none' convention as ``crit``).

    A lower-censored item is penalised only when the prediction is above its bound, an upper-censored one only when
    it is below; a prediction on the right side costs nothing, so no value is imposed that was never measured.
    """
    violated = ((cens < 0) & (pred > bound)) | ((cens > 0) & (pred < bound))
    return crit(pred, bound) * violated.to(pred.dtype)


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(a random positive scores higher than a random negative); ties count half. NaN if either side is empty."""
    p, n = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if len(p) == 0 or len(n) == 0:
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
