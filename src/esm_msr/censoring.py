"""One place for every kind of censoring the training data can carry.

An item is *censored* when its measurement only bounds the true value:

* ``cens = -1``: the true value is at or below ``cens_bound`` (lower-censored).
* ``cens = +1``: the true value is at or above ``cens_bound`` (upper-censored).
* ``cens = 0``: an ordinary measurement.

One source sets it here, recorded in ``cens_src``:

* ``SRC_FLOOR`` (2): the variant has a number, but that number is at or below its LIBRARY'S OWN FLOOR (``y_floor``, the per-library estimate
  cached on each item; see ``data.MutationStabilityDataset._library_floor``). Such a measurement reports the floor rather than the variant: the
  composition model's error for that sequence, not its stability. Its order among its equally-dead peers is noise and its value is not the
  variant's.

``SRC_RANGE`` (1) is reserved for the other half of the same population: the variants the assay declines to give a number to at all, reporting
only ``<-1`` (dead) or ``>5`` (hyperstable). A library's floor and its rate of ``<-1`` reports are the same quantity seen twice, and which of
the two a dead variant lands in is a matter of how far below the floor it fell and how noisy its K50 fit was. Those rows are dropped upstream
(``preprocess_megascale`` keeps only numeric ``ddG_ML``), so nothing sets ``SRC_RANGE`` on this branch; admitting them is a change to the
training population, not just to the loss. The machinery below and in ``losses`` is two-sided throughout so that doing so later needs no
rework here.

Losses use the same two fields everywhere: ``ListMLELoss.forward_censored`` ties censored members at the bottom or top of their list, and
``losses.hinge_regression_loss`` penalises only a prediction on the wrong side of the bound. Bounds are stored on the item's own ``ddG`` scale,
so one number serves WT-context singles, doubles and conditional (``ddG(A|B)``) items alike.
"""
from collections import Counter
from typing import Any, Dict, Iterable, Tuple

import numpy as np

CENS_LO, CENS_NONE, CENS_HI = -1, 0, 1
SRC_NONE, SRC_RANGE, SRC_FLOOR = 0, 1, 2

# the assay's reported dynamic range (kcal/mol); see MutationStabilityDataset.DG_FLOOR / DG_CEILING
DG_RANGE_LOW, DG_RANGE_HIGH = -1.0, 5.0
LOW_TAG, HIGH_TAG = '<-1', '>5'


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def to_numeric(values) -> np.ndarray:
    """A raw dG/ddG column as floats, with the censoring tags and ``-`` becoming NaN."""
    return np.array([_to_float(x) for x in np.asarray(values, dtype=object)], dtype=float)


def apply_item_censoring(data: Iterable[Dict[str, Any]], censor_floor: bool = True) -> Tuple[list, Dict[str, int]]:
    """
    Finalise the censoring fields of one library's items and return ``(items, counts)``.

    With ``censor_floor``, an item whose measured dG (``dG_meas``) is at or below its library's floor (``y_floor``) becomes lower-censored.
    Both must be finite, so a library with no floor estimate simply has no floor censoring.

    ``cens_bound`` (on the item's own ddG scale) is derived as ``dG_bound + (ddG - dG_meas)``: the offset between an item's label and the
    measured dG it came from is constant, and for a conditional item it already carries the partner's single. So a double measured at the
    floor bounds ``ddG_AB`` at ``y_floor - dG_wt``, while the ``ddG(A|B)`` derived from it is bounded ``ddG_B`` higher - each on its own scale,
    from the one floor.
    """
    items, counts = list(data), {'floor': 0, 'floor_by_subset': Counter()}
    for item in items:
        cens = int(item.get('cens', 0))
        if cens == 0 and censor_floor:
            dG, floor = item.get('dG_meas', float('nan')), item.get('y_floor', float('nan'))
            if np.isfinite(dG) and np.isfinite(floor) and dG <= floor:
                item['cens'], item['cens_src'], item['dG_bound'] = CENS_LO, SRC_FLOOR, float(floor)
                cens = CENS_LO
                counts['floor'] += 1
                counts['floor_by_subset'][item.get('subset_type', 'unknown')] += 1
        if cens != 0:
            dG_meas, dG_bound = item.get('dG_meas', float('nan')), item.get('dG_bound', float('nan'))
            ddG = item.get('ddG', float('nan'))
            item['cens_bound'] = float(dG_bound + (ddG - dG_meas)) if np.isfinite(dG_meas) and np.isfinite(dG_bound) else float('nan')
        else:
            item.setdefault('cens', 0)
            item.setdefault('cens_src', 0)
            item['cens_bound'] = float('nan')
    return items, counts


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """P(a random positive scores higher than a random negative); ties count half. NaN if either side is empty."""
    p, n = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if len(p) == 0 or len(n) == 0:
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([p, n]))
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))
