"""Recompute a run's validation metrics from its val dumps, mirroring on_validation_epoch_end exactly.

Censoring belongs in the loss, not the yardstick: stats.compute_metrics and epi_metrics both drop cens!=0 items,
which makes every val_* number incomparable to a run without censoring and biases it down by range restriction
wherever censoring is common. `use_cens=False` recomputes the same metric set over the full population.

Fidelity is checked by running with use_cens=True and diffing against the run's own metrics.csv.
"""
import json
import numpy as np

from esm_msr import epi_metrics, stats
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


def recompute(d, use_cens=True):
    """Returns {metric_name: value} using the same names on_validation_epoch_end logs."""
    cens = d['cens'] if use_cens else np.zeros_like(d['cens'])
    out, per_metric = {}, {}

    for name in dict.fromkeys(d['loader'].tolist()):            # preserves dataloader order
        m = np.where(d['loader'] == name)[0]
        mm = stats.compute_metrics(d['wt'][m], d['mt'][m], d['comb'][m], d['gt'][m],
                                   [d['subset_type'][i] for i in m], cens=cens[m], obs=_obs(d, m))
        for metric, val in mm.items():
            if val is not None and np.isfinite(val):
                out[f'val_{metric}/{name}'] = float(val)
                per_metric.setdefault(metric, []).append(float(val))
    for metric, vals in per_metric.items():
        out[f'val_{metric}_avg'] = float(np.nanmean(vals))

    ok = np.isfinite(d['gt']) & (cens == 0)
    if ok.any():
        out['val_rho_ungrouped_wt'] = float(stats.safe_spearman(d['wt'][ok], d['gt'][ok]))
        out['val_rho_ungrouped_combined'] = float(stats.safe_spearman(d['comb'][ok], d['gt'][ok]))
        out['val_ndcg@k=96_ungrouped_combined'] = float(stats.safe_ndcg_k96(d['comb'][ok], d['gt'][ok]))
        out['val_ndcg@k>0_ungrouped_combined'] = float(stats.safe_ndcg_t0(d['comb'][ok], d['gt'][ok]))

    items = {'mut_key': d['mut_key'], 'subset_type': d['subset_type'], 'cens': cens,
             'ddG': d['gt'], 'dddG': d['dddG'], 'dG_wt': d['dG_wt'], 'y_floor': d['y_floor'],
             'wt': d['wt'], 'mt': d['mt'], 'comb': d['comb'], 'wt_rev': d['wt_rev']}
    for name, val in epi_metrics.compute(epi_metrics.Table(items, d['link'])).items():
        if np.isfinite(val):
            out[f'val_{name}'] = float(val)
    return out
