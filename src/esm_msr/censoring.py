"""One place for every kind of censoring the training data can carry.

An item is *censored* when its measurement only bounds the true value:

* ``cens = -1``: the true value is at or below ``cens_bound`` (lower-censored).
* ``cens = +1``: the true value is at or above ``cens_bound`` (upper-censored).
* ``cens = 0``: an ordinary measurement.

Two sources set it here, recorded in ``cens_src``:

* ``SRC_FLOOR`` (2): the variant has a number, but that number is at or below its LIBRARY'S OWN FLOOR (``y_floor``, the per-library estimate
  cached on each item; see ``data.MutationStabilityDataset._library_floor``). Such a measurement reports the floor rather than the variant: the
  composition model's error for that sequence, not its stability. Its order among its equally-dead peers is noise and its value is not the
  variant's.

* ``SRC_GAP`` (3): the assay fits dG twice, once per protease, and the two estimates of the SAME variant disagree
  by more than ``gap_max``. Each channel compares a measured K50 against a K50 predicted for the unfolded sequence
  from composition, so a folded variant's two channels estimate one physical quantity and agree; a variant that is
  unfolded anyway returns each protease's own composition error, and those are independent. A gap of several
  kcal/mol therefore says the fits do not jointly identify a stability, whatever number the combined fit reported.
  The bound is the variant's OWN reported value: a broken fit still bounds the truth from above, because the
  reported value is at best the floor the variant pinned at.

  This source needs no doubles and no library floor, which is why it carries the censoring where ``SRC_FLOOR``
  cannot: the floor can only be located in libraries that happen to contain enough deeply destabilised doubles.
  Measured on this dataset it is specific but not sensitive - 92% of the numeric rows it flags sit below dG 0.5,
  but it catches only ~15% of the rows down there - so the two sources are a union, not alternatives.

* ``SRC_RANGE`` (1): the assay declined to fit the variant at all and reported only ``<-1`` (dead) or ``>5``
  (hyperstable). Unlike the other two sources this one is set upstream, in ``preprocess_megascale``, because it
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


MULTI_SUBSETS = frozenset({'double', 'cond'})


def threshold_for(item: Dict[str, Any], single: Optional[float], multi: Optional[float]) -> Optional[float]:
    """The censoring threshold that applies to an item: ``multi`` for doubles and the cond items derived from them,
    ``single`` for everything else (singles, native_cond)."""
    return multi if item.get('subset_type') in MULTI_SUBSETS else single


def apply_item_censoring(data: Iterable[Dict[str, Any]], censor_floor: bool = True,
                         gap_max: Optional[float] = 2.0, threshold: Optional[float] = None,
                         threshold_multi: Optional[float] = None, censor_multi: bool = True) -> Tuple[list, Dict[str, int]]:
    """
    Finalise the censoring fields of one library's items and return ``(items, counts)``.

    With ``censor_floor``, an item whose measured dG (``dG_meas``) is at or below its library's floor (``y_floor``) becomes lower-censored;
    both must be finite, so a library with no floor evidence simply has no floor censoring. With ``gap_max``, an item whose two proteases
    disagree by more than that (``protease_gap``) becomes lower-censored against its own reported value. The floor source is checked first
    because its bound is the more robust of the two - the library's central pinning value rather than one unreliable reading. An item that
    already carries ``SRC_RANGE`` from preprocessing is left as it is.

    With ``censor_multi`` False (``--no-censor_doubles``) neither the floor/threshold rule nor the gap rule touches a double or a
    cond item: it keeps its measured value as an ordinary label (to be weighted by its confidence interval instead,
    ``esm_msr.ci_weights``). Singles are censored as before, and a double the assay only BOUNDED (``SRC_RANGE``) stays a bound,
    because it has no value to keep.

    With ``threshold`` / ``threshold_multi`` (``--censor_threshold`` / ``--censor_threshold_double``) the floor rule uses that
    value instead of ``y_floor``: an item measured at or below it is lower-censored, bounded by it, in every library whether or
    not the library has a floor of its own (still ``SRC_FLOOR``).

    ``cens_bound`` (on the item's own ddG scale) is derived as ``dG_bound + (ddG - dG_meas)``: the offset between an item's label and the
    measured dG it came from is constant, and for a conditional item it already carries the partner's single. So a double measured at the
    floor bounds ``ddG_AB`` at ``y_floor - dG_wt``, while the ``ddG(A|B)`` derived from it is bounded ``ddG_B`` higher - each on its own scale,
    from the one floor.
    """
    items = list(data)
    counts = {'floor': 0, 'gap': 0, 'range': 0, 'floor_by_subset': Counter(), 'gap_by_subset': Counter(),
              'range_by_subset': Counter()}
    for item in items:
        cens = int(item.get('cens', 0))
        subset = item.get('subset_type', 'unknown')
        # SRC_RANGE arrives already set from preprocessing - it is a property of the row, not a decision made here.
        # Counted so the per-library log reports all three sources, and left alone: the branches below are gated on
        # cens == 0, so a tagged row keeps the bound the assay stated.
        if cens != 0 and int(item.get('cens_src', 0)) == SRC_RANGE:
            counts['range'] += 1
            counts['range_by_subset'][subset] += 1
        if cens == 0 and not censor_multi and subset in MULTI_SUBSETS:
            # a measured multi-mutant stays an ordinary label whatever its reading or its protease gap says
            item.setdefault('cens', 0)
            item.setdefault('cens_src', 0)
            item['cens_bound'] = float('nan')
            continue
        if cens == 0 and censor_floor:
            dG = item.get('dG_meas', float('nan'))
            thr = threshold_for(item, threshold, threshold_multi)
            floor = float(thr) if thr is not None else item.get('y_floor', float('nan'))
            if np.isfinite(dG) and np.isfinite(floor) and dG <= floor:
                item['cens'], item['cens_src'], item['dG_bound'] = CENS_LO, SRC_FLOOR, float(floor)
                cens = CENS_LO
                counts['floor'] += 1
                counts['floor_by_subset'][subset] += 1
        if cens == 0 and gap_max is not None:
            gap, dG = item.get('protease_gap', float('nan')), item.get('dG_meas', float('nan'))
            if np.isfinite(gap) and gap > gap_max and np.isfinite(dG):
                # Its own reading is the bound: dG_bound = dG_meas makes cens_bound come out as the item's own ddG.
                item['cens'], item['cens_src'], item['dG_bound'] = CENS_LO, SRC_GAP, float(dG)
                cens = CENS_LO
                counts['gap'] += 1
                counts['gap_by_subset'][subset] += 1
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
