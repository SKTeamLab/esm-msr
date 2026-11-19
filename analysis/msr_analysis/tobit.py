"""Post-hoc Tobit observation link of ``FINAL_p0`` and analytical floor-link curves for saturation analyses."""
from __future__ import annotations

import numpy as np

from esm_msr.val_stats import tobit_link

TOBIT = dict(sigma_y=0.28, mu_f=-0.14, sigma_f=0.5)


def linked(latent, dg_wt, sigma_y: float = 0.28, mu_f: float = -0.14, sigma_f: float = 0.5) -> np.ndarray:
    """Return Tobit-linked ddG ``h(dG_wt + latent) - dG_wt`` for 1D ``(n,)`` or 2D ``(n_seeds, n)`` ``latent``."""
    dg = np.asarray(dg_wt, dtype=float)
    lat = np.asarray(latent, dtype=float)
    h = tobit_link(sigma_y=sigma_y, mu_f=mu_f, sigma_f=sigma_f)
    if lat.ndim == 2:
        return h(dg[None, :] + lat) - dg[None, :]
    return h(dg + lat) - dg


def softplus_link(z, floor: float = 0.0, tau: float = 0.59) -> np.ndarray:
    """Analytical softplus floor link ``f + tau * softplus((z - f) / tau)`` on absolute dG."""
    z = np.asarray(z, dtype=float)
    return floor + tau * np.logaddexp(0.0, (z - floor) / tau)


def inv_softplus_link(y, floor: float = 0.0, tau: float = 0.59, eps: float = 0.05) -> np.ndarray:
    """Clipped inverse of ``softplus_link`` (clipped at ``floor + eps``)."""
    y_clip = np.maximum(np.asarray(y, dtype=float), floor + eps)
    v = (y_clip - floor) / tau
    return floor + tau * np.where(v > 20.0, v, np.log(np.expm1(np.maximum(v, 1e-12))))


def leaky_link(
    z,
    knee: float = 0.0,
    tau: float = 0.59,
    alpha: float = 0.10,
    clamp: float = -1.0,
    width: float = 0.25,
) -> np.ndarray:
    """Analytical two-stage leaky softplus link with hard-bound clamp."""
    z = np.asarray(z, dtype=float)
    u = knee + alpha * (z - knee) + (1.0 - alpha) * tau * np.logaddexp(0.0, (z - knee) / tau)
    return clamp + width * np.logaddexp(0.0, (u - clamp) / width)


def inv_leaky_link(
    y,
    knee: float = 0.0,
    tau: float = 0.59,
    alpha: float = 0.10,
    clamp: float = -1.0,
    width: float = 0.25,
    eps: float = 0.05,
) -> np.ndarray:
    """Inverse of ``leaky_link`` solved via damped Newton steps from the upper linear asymptote."""
    y_clip = np.maximum(np.asarray(y, dtype=float), clamp + eps)
    v = (y_clip - clamp) / width
    u = clamp + width * np.where(v > 20.0, v, np.log(np.expm1(np.maximum(v, 1e-12))))
    d = (u - knee) / alpha
    for _ in range(80):
        s = d / tau
        f = alpha * d + (1.0 - alpha) * tau * np.logaddexp(0.0, s) - (u - knee)
        grad = alpha + (1.0 - alpha) / (1.0 + np.exp(-np.clip(s, -60.0, 60.0)))
        d = d - f / grad
    return knee + d
