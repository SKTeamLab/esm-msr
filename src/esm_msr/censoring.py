"""One place for every kind of censoring the training data can carry.

An item is *censored* when its measurement only bounds the true value:

* ``cens = -1``: the true value is at or below ``cens_bound`` (lower-censored).
* ``cens = +1``: the true value is at or above ``cens_bound`` (upper-censored).
* ``cens = 0``: an ordinary measurement.

Two sources set it, recorded in ``cens_src``:

* ``SRC_FLOOR`` (2): the variant has a number, but that number is at or below ``--censor_threshold``, a few tenths above the assay's floor.
  Such a measurement reports the floor rather than the variant: the composition model's error for that sequence, not its stability. Its
  order among its equally-dead peers is noise (the two proteases' rank agreement is 0.20 below dG 0) and its value is not the variant's.
  (This used to be the item's own library floor, estimated per library; one threshold replaced it, ``docs/trim_audit.md`` §2.2.)

* ``SRC_GAP`` (3): REMOVED. It censored a variant whose two protease fits disagreed by more than 2 kcal/mol. At the standing threshold it
  added almost nothing (92% of the rows it flags already sit below dG 0.5; what it alone caught was 0.2% of singles and 1.8% of doubles), so
  the rule went. The code is kept as a constant because the validation dumps of older runs carry ``cens_src == 3``.

* ``SRC_RANGE`` (1): the assay declined to fit the variant at all and reported only ``<-1`` (dead) or ``>5``
  (hyperstable). Unlike the threshold this one is set upstream, in ``preprocess_megascale``, because it
  decides which ROWS exist rather than how an existing row is scored.

  Admitting these rows is what keeps the training set from being selected on its own outcome. A double's additive
  expectation is known before the double is measured, and the assay's failure rate rises monotonically with that
  depth - 0.7% of doubles with a shallow expectation go unfitted against 24% of those below -5 kcal/mol. Drop them
  and the surviving deep doubles are the ones that happened to come out less destabilised than their additive
  expectation, so the model sees a population in which deep means rescue. The bound is the tag's own threshold, so
  such a row says only 'at or below -1', which is exactly what the assay established, and the hinge stops pushing
  once the prediction clears it. 86% of the doubles dropped this way carry a usable tag, and that share RISES with
  depth (94% below -5), so the recoverable rows are concentrated where the bias is worst. The remainder report
  ``-`` for dG as well, carry no bound, and stay dropped.

Losses use the same two fields everywhere: ``ListMLELoss.forward_censored`` ties censored members at the bottom or top of their list, and
``losses.hinge_regression_loss`` penalises only a prediction on the wrong side of the bound. Bounds are stored on the item's own ``ddG`` scale,
so one number serves WT-context singles, doubles and conditional (``ddG(A|B)``) items alike.
"""
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

CENS_LO, CENS_NONE, CENS_HI = -1, 0, 1
SRC_NONE, SRC_RANGE, SRC_FLOOR, SRC_GAP = 0, 1, 2, 3

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


def range_tag(dG_raw: Any, ddG_raw: Any) -> Tuple[int, float]:
    """
    The censoring a raw ``(dG_ML, ddG_ML)`` pair declares by its tag, as ``(cens, dG_bound)``.

    Only a row the assay refused to fit (``ddG_ML == '-'``) can be tagged; a numeric ddG is an ordinary
    measurement whatever dG says. ``<-1`` and ``>5`` are the only tags the table uses, and each names its own
    bound. Shared with ``preprocess_megascale`` so one definition decides both which rows are admitted and what
    they are bounded by.
    """
    if str(ddG_raw).strip() != '-':
        return CENS_NONE, float('nan')
    tag = str(dG_raw).strip()
    if tag.startswith('<'):
        return CENS_LO, DG_RANGE_LOW
    if tag.startswith('>'):
        return CENS_HI, DG_RANGE_HIGH
    return CENS_NONE, float('nan')


def apply_item_censoring(data: Iterable[Dict[str, Any]], threshold: Optional[float] = None) -> Tuple[list, Dict[str, int]]:
    """
    Finalise the censoring fields of one library's items and return ``(items, counts)``.

    With ``threshold`` (``--censor_threshold``), an item whose measured dG (``dG_meas``) is at or below it becomes lower-censored,
    bounded by it (``SRC_FLOOR``), in every library. One number serves singles and multi-mutants alike. ``None`` censors nothing here.
    An item that already carries ``SRC_RANGE`` from preprocessing is left as it is.

    ``cens_bound`` (on the item's own ddG scale) is derived as ``dG_bound + (ddG - dG_meas)``: the offset between an item's label and the
    measured dG it came from is constant, so a double measured at the threshold bounds ``ddG_AB`` at ``threshold - dG_wt``.
    """
    items = list(data)
    counts = {'floor': 0, 'range': 0, 'floor_by_subset': Counter(), 'range_by_subset': Counter()}
    for item in items:
        cens = int(item.get('cens', 0))
        subset = item.get('subset_type', 'unknown')
        # SRC_RANGE arrives already set from preprocessing - it is a property of the row, not a decision made here.
        # Counted so the per-library log reports both sources, and left alone: the branch below is gated on
        # cens == 0, so a tagged row keeps the bound the assay stated.
        if cens != 0 and int(item.get('cens_src', 0)) == SRC_RANGE:
            counts['range'] += 1
            counts['range_by_subset'][subset] += 1
        if cens == 0 and threshold is not None:
            dG = item.get('dG_meas', float('nan'))
            if np.isfinite(dG) and dG <= threshold:
                item['cens'], item['cens_src'], item['dG_bound'] = CENS_LO, SRC_FLOOR, float(threshold)
                cens = CENS_LO
                counts['floor'] += 1
                counts['floor_by_subset'][subset] += 1
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
