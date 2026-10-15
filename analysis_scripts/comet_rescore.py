"""Recompute a run's validation metrics from its val dumps, mirroring on_validation_epoch_end exactly.

Used to bring runs logged BEFORE the metric definition was fixed onto the current one. Those runs scored only the
uncensored items, which made them incomparable to a baseline that censors nothing and biased them down by range
restriction wherever censoring was common. The definition now in training.py scores the canonical population -
every item the assay actually fitted, censored or not - and holds back only the rows it merely bounded
(SRC_RANGE), whose label is their bound and a library constant. This reproduces that.

A dump from before --censor_range carries no cens_src, so its canonical population is simply all of it.
"""
import json
import numpy as np

from esm_msr import epi_metrics, stats
from esm_msr.censoring import SRC_RANGE
from esm_msr.link import numpy_link

POOLED_4 = ('val_rho_ungrouped_wt', 'val_rho_ungrouped_combined',
            'val_ndcg@k=96_ungrouped_combined', 'val_ndcg@k>0_ungrouped_combined')


def load_dump(path):
    z = np.load(path, allow_pickle=True)
    n = len(z['ground_truth'])
    d = dict(
        loader=np.asarray([str(x) for x in z['loader']]),
        subset_type=[str(x) for x in z['subset_type']],
        mut_key=[tuple(tuple(m) for m in json.loads(k)) for k in z['mut_key']],
        n_mut=np.asarray(z['n_mut']),
        gt=np.asarray(z['ground_truth'], float), dddG=np.asarray(z['dddG'], float),
        wt=np.asarray(z['wt'], float), mt=np.asarray(z['mt'], float), comb=np.asarray(z['comb'], float),
        wt_rev=np.asarray(z['wt_rev'], float),
        mt_fwd=np.asarray(z['mt_fwd'], float) if 'mt_fwd' in z.files else np.full(n, np.nan),
        dG_wt=np.asarray(z['dG_wt'], float), y_floor=np.asarray(z['y_floor'], float),
        cens=np.asarray(z['cens'], int),
        cens_src=(np.asarray(z['cens_src'], int) if 'cens_src' in z.files else np.zeros(n, int)),
    )
    d['link'] = None
    if 'link_summary' in z.files:
        ls = json.loads(str(z['link_summary'].item()))
        d['link'] = numpy_link(ls['tau'], ls['floor_min'], ls['floor_max'], ls.get('global_floor', 0.0))
    return d


def _obs(d, idx=None):
    """validation_step builds obs for exactly the 'wt' and 'comb' heads, where dG_wt is finite."""
    if d['link'] is None:
        return None
    sl = slice(None) if idx is None else idx
    dgw, yf = d['dG_wt'][sl], d['y_floor'][sl]
    ok = np.isfinite(dgw)
    z = np.where(ok, dgw, 0.0)
    return {k: np.where(ok, d['link'](z + d[k][sl], yf) - z, d[k][sl]) for k in ('wt', 'comb')}


def recompute(d, canonical_only=True):
    """
    Returns {metric_name: value} using the same names on_validation_epoch_end logs.

    ``canonical_only`` mirrors the run-time definition: score everything the assay fitted and hold back only the
    SRC_RANGE rows. False scores literally every row in the dump, which is only useful as a diagnostic.
    """
    keep = (d['cens_src'] != SRC_RANGE) if canonical_only else np.ones(len(d['gt']), bool)
    out, per_metric = {}, {}

    for name in dict.fromkeys(d['loader'].tolist()):            # preserves dataloader order
        m = np.where((d['loader'] == name) & keep)[0]
        if not len(m):
            continue
        mm = stats.compute_metrics(d['wt'][m], d['mt'][m], d['comb'][m], d['gt'][m],
                                   [d['subset_type'][i] for i in m], obs=_obs(d, m))
        for metric, val in mm.items():
            if val is not None and np.isfinite(val):
                out[f'val_{metric}/{name}'] = float(val)
                per_metric.setdefault(metric, []).append(float(val))
    for metric, vals in per_metric.items():
        out[f'val_{metric}_avg'] = float(np.nanmean(vals))

    ok = np.isfinite(d['gt']) & keep
    if ok.any():
        out['val_rho_ungrouped_wt'] = float(stats.safe_spearman(d['wt'][ok], d['gt'][ok]))
        out['val_rho_ungrouped_combined'] = float(stats.safe_spearman(d['comb'][ok], d['gt'][ok]))
        out['val_ndcg@k=96_ungrouped_combined'] = float(stats.safe_ndcg_k96(d['comb'][ok], d['gt'][ok]))
        out['val_ndcg@k>0_ungrouped_combined'] = float(stats.safe_ndcg_t0(d['comb'][ok], d['gt'][ok]))

    idx = np.flatnonzero(keep)
    items = {'mut_key': [d['mut_key'][i] for i in idx], 'subset_type': [d['subset_type'][i] for i in idx],
             'ddG': d['gt'][keep], 'dddG': d['dddG'][keep], 'dG_wt': d['dG_wt'][keep],
             'y_floor': d['y_floor'][keep], 'wt': d['wt'][keep], 'mt': d['mt'][keep],
             'comb': d['comb'][keep], 'wt_rev': d['wt_rev'][keep]}
    for name, val in epi_metrics.compute(epi_metrics.Table(items, d['link'])).items():
        if np.isfinite(val):
            out[f'val_{name}'] = float(val)
    return out
