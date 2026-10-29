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
