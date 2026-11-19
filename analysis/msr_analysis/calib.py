"""OLS kcal/mol recalibration and error metrics across single- and multi-mutant datasets.

Selects each model's dataset-appropriate head on the target ``Frame`` (``comb`` on multi-mutant sets,
``wt`` on singles-only sets for ESM-MSR) so single-mutant calibration parameters are applied to the
matching deployed head on multi-mutant datasets and multi-mutant donors fit ``comb`` rather than ``wt``.
"""
from __future__ import annotations

import numpy as np

from .common import primary_key
from .data import Frame


def fit_scale_bias(pred, y) -> tuple[float, float]:
    """Fit OLS ``y ~ a * pred + b`` on finite pairs, returning ``(a, b)``."""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(y)
    p, t = pred[ok], y[ok]
    a, b = np.polyfit(p, t, 1)
    return float(a), float(b)


def apply(pred, ab: tuple[float, float]) -> np.ndarray:
    """Apply linear calibration ``a * pred + b``."""
    return float(ab[0]) * np.asarray(pred, dtype=float) + float(ab[1])


def rmse(pred, y) -> float:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(y)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))


def mae(pred, y) -> float:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(y)
    if not ok.any():
        return np.nan
    return float(np.mean(np.abs(pred[ok] - y[ok])))


def calibration_key(fam: str, fr: Frame) -> str | None:
    """Return the dataset-appropriate prediction key for ``fam`` on ``fr``.

    - ``'msr'`` / ``'msrp0'`` / ``'ESM-MSR'`` -> ``primary_key(fr, 'msrp0')`` (``msrp0:comb`` on multi-mutant sets,
      ``msrp0:wt`` on singles-only sets).
    - Roster model name (e.g. ``'ThermoMPNN'``, ``'SPURS'``) -> ``external.model_key_for_dataset(fam, fr)``.
    - Raw prediction key (e.g. ``'prosst'``, ``'pmpnn'``) -> ``fam`` if present in ``fr.preds``.
    """
    if fam in ('msr', 'msrp0', 'ESM-MSR'):
        k = primary_key(fr, 'msrp0')
        return k if k in fr.preds else None
    if fam in ('msrrel@0', 'ESM3-small-open'):
        k = primary_key(fr, 'msrrel@0')
        return k if k in fr.preds else None
    from .models import BY_NAME
    if fam in BY_NAME:
        from .external import model_key_for_dataset
        return model_key_for_dataset(fam, fr)
    return fam if fam in fr.preds else None


def fit_frame_scale_bias(fr: Frame, fam: str) -> list[tuple[float, float]] | None:
    """Fit ``(a, b)`` per seed for ``fam`` on its dataset-appropriate head on ``fr``."""
    k = calibration_key(fam, fr)
    if k is None or k not in fr.preds:
        return None
    sp = fr.spine
    preds = np.atleast_2d(fr.preds[k])
    ok = sp['use'].to_numpy() & np.isfinite(sp['y'].to_numpy()) & np.isfinite(preds).all(axis=0)
    if ok.sum() < 3:
        return None
    y = sp['y'].to_numpy(dtype=float)[ok]
    return [fit_scale_bias(p[ok], y) for p in preds]


def calibrated_preds(fr: Frame, fam: str, ab_per_seed: list[tuple[float, float]]) -> np.ndarray | None:
    """Apply donor ``ab_per_seed`` to ``fam``'s dataset-appropriate head on ``fr``, returning ``(n_seeds, n_rows)``."""
    k = calibration_key(fam, fr)
    if k is None or k not in fr.preds:
        return None
    preds = np.atleast_2d(fr.preds[k])
    if len(ab_per_seed) == 1 and preds.shape[0] > 1:
        ab_list = ab_per_seed * preds.shape[0]
    else:
        ab_list = [ab_per_seed[min(i, len(ab_per_seed) - 1)] for i in range(preds.shape[0])]
    return np.vstack([apply(preds[i], ab_list[i]) for i in range(preds.shape[0])])
