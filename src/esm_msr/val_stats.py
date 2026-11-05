"""The validation statistics a run logs, in one place: which names, which groups, what the external summary holds.

This file is shared VERBATIM by train-speedup and main-baseline (``cmp`` them), like ``epi_metrics.py`` and ``stats.py``, so a baseline run and a
tip run send the same names to Comet. Nothing here touches training: it only decides what is logged.

What is logged (``<p>`` is '' for the run's main validation regime, 'unmasked_' / 'masked_' for the copy ``--val_unmasked`` / ``--val_masked`` adds):

* ddG, per library (``val_<p><metric>/<library>``): ``rho_combined``, ``rho_doubles_combined``, ``rho_singles_wt``, ``rmse_combined``, ``rmse_singles_wt``.
* their library-equal means (``val_<p><metric>_avg``): the first four of those, and ``ndcg@k=96`` / ``ndcg>0`` for the combined and (``_wt``) the WT head.
* ``val_<p>{nat,des,mutbg}_rho_combined_avg``: the mean of ``rho_combined`` over the natural, de novo designed and mutant-background libraries.
* ``val_<p><metric>_external`` for ``rho_combined``, ``rho_wt``, ``rmse_combined``, ``rmse_singles_wt``: the mean over the external loaders (never part of an average).
* epistasis, pooled over every library (``esm_msr.epi_metrics``): ``val_<p>epi_<metric>_{comb,ctx}`` for ``beyond_lib``, ``beyond_rho``, ``pair_rho``, ``line_rho``,
  ``cell_mag``, ``cell_rank``, ``cell_flipacc``, ``val_<p>epi_naive_rho_comb``, ``val_<p>epi_{err,bias}_xgt1_comb`` (the comb epistasis error and bias on the doubles
  whose additive expectation x = dG_wt + ddG_A + ddG_B is above 1 kcal/mol, where the assay floor is not in the way), and ``val_<p>epi_composite``.

``comb`` is the reported prediction's epistasis, latent: (~MT(AB) + WT(AB))/2 - WT(A) - WT(B). ``ctx`` is the WT-in-context control, latent: (~WT(AB) + WT(AB))/2 -
WT(A) - WT(B); it needs ``--val_cycle_passes``.
"""
import re
from typing import Dict, Iterable

import numpy as np

EPI_RANK_METRICS = ('beyond_lib', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag', 'cell_rank', 'cell_flipacc')
EPI_HEADS = ('comb', 'ctx')
EPI_LOGGED = frozenset({f'epi_{m}_{h}' for m in EPI_RANK_METRICS for h in EPI_HEADS} | {'epi_naive_rho_comb'})
# epi_metrics.compute_clean_diagnostics' names -> the names that are logged ('clean' meant: the additive expectation x is above min_x = 1.0 kcal/mol)
EPI_DIAG_RENAMED = {'epi_clean_err_comb': 'epi_err_xgt1_comb', 'epi_clean_bias_comb': 'epi_bias_xgt1_comb'}
EPI_COMPOSITE = 'epi_composite'

PER_LIBRARY_LOGGED = ('rho_combined', 'rho_doubles_combined', 'rho_singles_wt', 'rmse_combined', 'rmse_singles_wt')
AVG_LOGGED = ('rho_combined', 'rho_singles_wt', 'rho_doubles_combined', 'rmse_combined', 'ndcg@k=96', 'ndcg>0', 'ndcg@k=96_wt', 'ndcg>0_wt')
GROUPS = ('nat', 'des', 'mutbg')
EXTERNAL_LOGGED = ('rho_combined', 'rho_wt', 'rmse_combined', 'rmse_singles_wt')
REGIME_PREFIXES = ('', 'unmasked_', 'masked_')


def library_group(name: str) -> str:
    """'mutbg' (a library measured in a mutant background: '2M7O_I21A', 'v2K43S_2KVV'), 'nat' (a natural domain: '1A32') or 'des' (de novo designed: 'HEEH_...')."""
    if re.match(r'^v2[A-Z0-9_]+_[0-9][A-Za-z0-9]{3}$', name) or re.match(r'^[0-9][A-Za-z0-9]{3}_', name):
        return 'mutbg'
    if re.match(r'^[0-9][A-Za-z0-9]{3}$', name):
        return 'nat'
    return 'des'


def is_mutbg_library(name: str) -> bool:
    return library_group(name) == 'mutbg'


def epi_log_names(res: Dict[str, float], diag: Dict[str, float], prefix: str = '') -> Dict[str, float]:
    """The ``val_*`` epistasis scalars one validation regime sends to the logger: finite values of the kept names only."""
    out = {f'val_{prefix}{k}': v for k, v in res.items() if k in EPI_LOGGED}
    out.update({f'val_{prefix}{EPI_DIAG_RENAMED[k]}': v for k, v in diag.items() if k in EPI_DIAG_RENAMED})
    return {k: float(v) for k, v in out.items() if np.isfinite(v)}


def summarise_external(per_loader: Dict[str, Dict[str, float]], external: Iterable[str], prefix: str = '') -> Dict[str, float]:
    """``val_<p><metric>_external``: the mean over the external loaders of each per-loader metric in ``EXTERNAL_LOGGED``. ``per_loader`` maps loader -> {metric: value}.
    These loaders stay out of every average, group and pooled statistic: they are out of the training distribution."""
    external = set(external)
    out = {}
    for metric in EXTERNAL_LOGGED:
        vals = [v[metric] for name, v in per_loader.items() if name in external and metric in v and np.isfinite(v[metric])]
        if vals:
            out[f'val_{prefix}{metric}_external'] = float(np.mean(vals))
    return out


def logged_summary_names(prefixes: Iterable[str] = REGIME_PREFIXES) -> frozenset:
    """Every summary (non-per-library) name a run can send to the logger: what ``--monitor_metric`` may name."""
    names = set()
    for p in prefixes:
        names |= {f'val_{p}{m}_avg' for m in AVG_LOGGED}
        names |= {f'val_{p}{g}_rho_combined_avg' for g in GROUPS}
        names |= {f'val_{p}{m}_external' for m in EXTERNAL_LOGGED}
        names |= {f'val_{p}{n}' for n in EPI_LOGGED | set(EPI_DIAG_RENAMED.values()) | {EPI_COMPOSITE}}
    return frozenset(names)
