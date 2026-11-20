"""Pooled validation statistics: one Spearman over every in-distribution validation item at once.

``val_rho_combined_avg`` and ``val_rho_singles_wt_avg`` are means of per-library correlations (every library counts the same whatever its size and its own
spread). The pooled versions rank all the items of all the validation libraries together, so they also see a model's calibration ACROSS libraries (a library-level
offset or scale) and weight a library by its number of items.

* ``rho_combined_pooled``    the reported two-path average on every measured item (singles, doubles, native-background items): ``stats.compute_metrics``'s rho_combined, pooled.
* ``rho_singles_wt_pooled``  the WT head on the singles: its rho_singles_wt, pooled.

The population is the per-library one: the canonical items (every item the assay fitted, censored or not, only the rows it merely bounded -- cens_src == SRC_RANGE --
held back) with a finite label. The external benchmark sets are never part of it. The statistics read the latent scores (a monotone link leaves ranks alone).
Logged as ``val_<regime prefix><name>``; computed offline from a validation dump by ``scripts/revision_ablations_rescore.py`` with this same function.
"""
import numpy as np

from esm_msr import censoring, routing, stats

NAMES = ('rho_combined_pooled', 'rho_singles_wt_pooled')


def pooled_rho(wt_scores, comb_scores, ground_truths, subset_types, cens_src=None):
    """``{name: value}`` of NAMES over the given items (NaN where undefined)."""
    wt, comb, gt = (np.asarray(x, dtype=np.float64) for x in (wt_scores, comb_scores, ground_truths))
    st = np.asarray(subset_types)
    st = np.where(st == 'native_cond', 'single', st)
    keep = np.isfinite(gt)
    if cens_src is not None:
        keep &= np.asarray(cens_src) != censoring.SRC_RANGE
    measured = keep & np.isin(st, list(routing.MEASURED_SUBSETS))
    single = keep & np.isin(st, list(routing.WT_HEAD_SUBSETS))
    return {'rho_combined_pooled': stats.safe_spearman(comb[measured], gt[measured]),
            'rho_singles_wt_pooled': stats.safe_spearman(wt[single], gt[single])}
