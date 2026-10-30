"""The saturating link of the devel (floor-link) ESM-MSR checkpoints, applied offline.

Devel checkpoints write *latent* ddG: the model's additive-in-effects opinion, unsaturated.  What the assay reports is
``h(dG_wt + latent) - dG_wt`` with ``h(z) = f + tau * softplus((z - f) / tau)`` (``src/esm_msr/link.py``).  Ranking within a library is unchanged by ``h`` (it is
monotone), but anything that mixes libraries, reads levels, or differences two predictions (dddG) must go through it.  The parameters
are read from the checkpoints' ``link_head`` (``reference/devel_link_params.json``, extracted once by ``extract_link_params``).  The
per-item floors cached on the cluster (``y_floor.json``) are not collected locally, so every item uses the checkpoint's one learned floor.
"""
import json

import numpy as np

from .registry import REFERENCE

PARAMS = REFERENCE / 'devel_link_params.json'


def extract_link_params(root):
    """Read link_head.* of the three devel checkpoints (CPU, state dict only) and write the json."""
    import glob
    import torch
    want = {1: 'epoch=04-val_rho_combined_avg=0.840.ckpt', 2: 'epoch=07-val_rho_combined_avg=0.839.ckpt', 3: 'epoch=05-val_rho_combined_avg=0.841.ckpt'}
    out = {}
    for s, fn in want.items():
        p = glob.glob(f'{root}/dc_runs/logs/dc8_floorlink_canon_w2a4_m16a16_s{s}/0/{fn}')[0]
        sd = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
        sp = lambda v: float(torch.nn.functional.softplus(torch.as_tensor(v, dtype=torch.float64)))
        out[str(s)] = dict(tau=sp(sd['link_head.raw_tau']), floor_min=float(sd['link_head.floor_min']), floor_max=float(sd['link_head.floor_max']),
                           global_floor=float(sd['link_head.global_floor']), checkpoint=fn)
    PARAMS.write_text(json.dumps(out, indent=1))
    return out


def load_params():
    return json.loads(PARAMS.read_text())


def h(z, tau, floor):
    z = np.asarray(z, float)
    return floor + tau * np.logaddexp(0.0, (z - floor) / tau)


def observed(latent, dg_wt, seed):
    """latent ddG -> ddG on the observed (assay) scale for devel seed ``seed`` (1..3)."""
    p = load_params()[str(seed)]
    f = float(np.clip(p['global_floor'], p['floor_min'], p['floor_max']))
    return h(dg_wt + np.asarray(latent, float), p['tau'], f) - dg_wt


def link_fn(seed):
    """The checkpoint's observed-scale link of devel seed ``seed`` as a numpy function ``link(dG, y_floor) -> dG_obs`` on absolute dG, as ``esm_msr.link.numpy_link_from_state_dict``
    builds it: ``h(z) = f + tau softplus((z - f) / tau)`` with the knee ``f`` the item's library floor clipped into [floor_min, floor_max] (global_floor where it has none)."""
    p = load_params()[str(seed)]
    lo = p['floor_min']
    hi = max(p['floor_max'], lo + 1e-3)

    def link(z, y_floor=None):
        z = np.asarray(z, float)
        if y_floor is None:
            f = np.full(z.shape, p['global_floor'])
        else:
            yf = np.asarray(y_floor, float)
            f = np.where(np.isfinite(yf), np.clip(yf, lo, hi), p['global_floor'])
        return h(z, p['tau'], f)
    return link


def singles_of_doubles(spine):
    """(ia, ib): for every row of ``spine`` that is a double, the row index of each of its two singles of the same library (-1 where a single is not in the table or the row is not a double)."""
    code, mut, nm = spine['code'].astype(str).to_numpy(), spine['mut'].to_numpy(dtype=object), spine['n_mut'].to_numpy()
    where = {(c, m): i for i, (c, m, n) in enumerate(zip(code, mut, nm)) if n == 1}
    ia, ib = np.full(len(spine), -1), np.full(len(spine), -1)
    for i in np.flatnonzero(nm == 2):
        a, b = str(mut[i]).split(':')
        ia[i], ib[i] = where.get((code[i], a), -1), where.get((code[i], b), -1)
    return ia, ib


def observed_epistasis(spine, wt, mt, seed, y_floor):
    """The predicted epistasis of a devel checkpoint on the observed (assay) scale, as ``esm_msr.epi_metrics`` defines the naive one: the observed double minus the observed singles,
    ``obs(MT(AB)) - [obs(WT(A)) + obs(WT(B))]`` with obs(L) = link(dG_wt + L, floor) - dG_wt, the same form as the measured dddG = ddG_AB - ddG_A - ddG_B.  NaN where a single is missing."""
    link = link_fn(seed)
    c = spine['dG_wt'].to_numpy(float)
    fl = np.array([float(y_floor.get(k, np.nan)) for k in spine['code'].astype(str)])
    obs = lambda lat: link(c + lat, fl) - c
    ia, ib = singles_of_doubles(spine)
    om, ow = obs(np.asarray(mt, float)), obs(np.asarray(wt, float))
    out = np.full(len(spine), np.nan)
    ok = (ia >= 0) & (ib >= 0)
    out[ok] = om[ok] - (ow[ia[ok]] + ow[ib[ok]])
    return out
