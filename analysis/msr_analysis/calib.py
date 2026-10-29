"""Putting ESM-MSR on the kcal/mol scale before an error metric is read.

ESM-MSR is trained on protease-derived stabilities, so its numbers are on the protease scale (and carry the protease
dynamic range), not on the calorimetric ddG scale of the external sets.  A rank metric does not care; RMSE does.  The
recalibration is a two-parameter fit, ``y ~ a * pred + b`` (scale and bias, OLS), on ONE sacrificial dataset; the other
datasets are then scored with those two numbers frozen.  The sacrificial set's own RMSE is not an out-of-sample number and is
starred wherever it is shown (and so is any set that contains it: S669 contains S461).
"""
import numpy as np


def fit_scale_bias(pred, y):
    """OLS y ~ a * pred + b on finite pairs."""
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    ok = np.isfinite(pred) & np.isfinite(y)
    p, t = pred[ok], y[ok]
    a, b = np.polyfit(p, t, 1)
    return float(a), float(b)


def apply(pred, ab):
    return ab[0] * np.asarray(pred, float) + ab[1]


def rmse(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(np.sqrt(np.mean((pred[ok] - y[ok]) ** 2)))


def mae(pred, y):
    pred = np.asarray(pred, float); y = np.asarray(y, float)
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(np.mean(np.abs(pred[ok] - y[ok])))


def weighted_rmse(pred, y, w):
    ok = np.isfinite(pred) & np.isfinite(y) & (w > 0)
    return float(np.sqrt(np.sum(w[ok] * (pred[ok] - y[ok]) ** 2) / np.sum(w[ok])))
