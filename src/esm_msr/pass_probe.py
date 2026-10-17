"""
The validation pass probe (``--val_pass_probe``): what each way of reading a mutation has learned.

A pass scores a mutation X as ``logit(to) - logit(from)`` at X's site. What it can know depends on what it is shown there and at the
partner's site, so for a double AB the readouts form a ladder. Per scored site A:

    view    pass_type  mask           site A     site B            the per-site readout estimates     can carry
    u_wt    wt         none           wt seen    wt                ddG_A          (start of the path) nothing beyond additive
    pair    wt         marginal       masked     masked            ddG_A | pair   (start)             pair and line effects: it knows
                                                                                                      WHICH pair is mutated, not to what
    u_mt    mt         none           mutant     mutant            ddG(A|B)       (end of the path)   everything, with the scored
                                                                                                      residue itself visible
    i_mt    mt         independent    masked     mutant            ddG(A|B)       (end)               everything: the conditional
                                                                                                      masked marginal

Each is read by BOTH adapters (``MSRModel.forward_partitioned(..., adapter=)``), so a view name is ``<adapter>.<view>``: eight views.
On a single the start views read ddG_X on the wild type; ``pair`` and ``i_mt`` see the same masked sequence and differ only in the MT
pass's structure masking. The one rung left out, ``independent`` on the wild type (A masked, B wild type), needs no forward: on a double
it is exactly the sum of its two singles' ``pair`` readouts, i.e. the additive control of the masked family.

A start view sums to about ddG_A + ddG_B and an end view to about ddG(A|B) + ddG(B|A) = 2 ddG_AB - ddG_A - ddG_B, so the TRAPEZOIDS
``tz.<start>+<end> = (wt.<start> + mt.<end>) / 2`` (WT adapter at the start, MT adapter at the end) are the double predictors the
architecture is built around; ``tz.u_wt+u_mt`` is the reported combined prediction when neither pass masks. Every view is also scored
on its own: the metrics are rank based or residualised, so an end view's doubled non-additivity does not matter.

Per head (view or trapezoid), logged as ``val_probe_<head>_<metric>`` with dots in the head name replaced by underscores:

    rho_single / rho_double   mean over libraries of Spearman(head, measured ddG) on uncensored singles / doubles
    beyond_rho .. cell_sigsd  the epistasis levels of ``esm_msr.epi_metrics`` with this head in the 'comb' slot
    nonadd_sd                 SD (latent, kcal/mol-like) of the head's own non-additivity d = L_AB - L_A - L_B over the cells of complete
                              position-pair blocks, around its global mean: how much non-additivity the view predicts at all
    nonadd_pair / _line /     the shares of that variance at each level of the block (pair mean, row and column means, double-centred
    _cell                     cells). Label-free: it says WHERE a view puts its non-additivity, e.g. a pair-masked view puts none in cells.
"""
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from esm_msr import epi_metrics, routing
from esm_msr.stats import safe_spearman

# name -> (adapter, pass_type, mask_strategy)
VIEWS = {
    f'{a}.{v}': (a, pt, m)
    for a in ('wt', 'mt')
    for v, pt, m in (('u_wt', 'wt', None), ('pair', 'wt', 'marginal'), ('u_mt', 'mt', None), ('i_mt', 'mt', 'independent'))
}
START, END = ('u_wt', 'pair'), ('u_mt', 'i_mt')
TRAPEZOIDS = {f'tz.{s}+{e}': (f'wt.{s}', f'mt.{e}') for s in START for e in END}
# every start view with every end view, either adapter on either side (scripts/pass_probe_from_dump.py); the four above are among them
ALL_TRAPEZOIDS = {f'tz.{a}.{s}+{b}.{e}': (f'{a}.{s}', f'{b}.{e}') for a in ('wt', 'mt') for s in START for b in ('wt', 'mt') for e in END}
EPI = ('beyond_rho', 'pair_rho', 'line_rho', 'cell_rank', 'cell_flipacc', 'cell_sigsd')
METRICS = ('rho_single', 'rho_double') + EPI + ('nonadd_sd', 'nonadd_pair', 'nonadd_line', 'nonadd_cell')


def heads(views: Dict[str, np.ndarray], trapezoids: Dict[str, tuple] = TRAPEZOIDS) -> Dict[str, np.ndarray]:
    """The eight views plus the trapezoids (by default the four logged in training), as name -> latent score per item."""
    out = dict(views)
    for name, (s, e) in trapezoids.items():
        if s in views and e in views:
            out[name] = 0.5 * (np.asarray(views[s], float) + np.asarray(views[e], float))
    return out


def run_views(model, batch, rows, known: Optional[Dict[tuple, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """
    Every view of ``VIEWS`` on the batch rows ``rows`` (a long tensor; singles and doubles), as numpy arrays over the WHOLE batch with NaN
    outside ``rows``. ``known`` maps an (adapter, pass_type, mask) triple to scores the caller already has for the whole batch (the run's own
    WT and MT passes), which are reused instead of recomputed. Call under inference mode.
    """
    from esm_msr import utils
    B = int(batch['ddG'].shape[0])
    known = known or {}
    sub = utils.slice_batch_by_index(batch, rows)
    idx = rows.detach().cpu().numpy()
    out = {}
    for name, (adapter, pass_type, mask) in VIEWS.items():
        v = np.full(B, np.nan)
        if (adapter, pass_type, mask) in known:
            v[idx] = np.asarray(known[(adapter, pass_type, mask)], float)[idx]
        else:
            r = model.forward_partitioned(sub, pass_type=pass_type, mask_strategy=mask, adapter=adapter)['pred_calibrated']
            v[idx] = r.detach().float().cpu().numpy()
        out[name] = v
    return out


def _per_library_rho(score, gt, loader, keep) -> float:
    vals = []
    for lib in dict.fromkeys(loader):
        m = keep & (loader == lib) & np.isfinite(score)
        if m.sum() >= 3:
            r = safe_spearman(score[m], gt[m])
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def nonadd_levels(t: 'epi_metrics.Table', h: str = 'comb') -> Dict[str, float]:
    """``nonadd_sd`` and the pair / line / cell shares of a head's own non-additivity on complete blocks (see the module docstring)."""
    out = {'nonadd_sd': np.nan, 'nonadd_pair': np.nan, 'nonadd_line': np.nan, 'nonadd_cell': np.nan}
    if not t.head_ok(h):
        return out
    d = t.L[h] - t.LA[h] - t.LB[h]
    if not np.isfinite(d).any():
        return out
    g = float(np.nanmean(d))
    ss = {'pair': 0.0, 'line': 0.0, 'cell': 0.0}
    n = 0
    for M in t.matrices():
        D = epi_metrics._lay(M, d)
        blk = epi_metrics._complete_block(D)
        if blk is None:
            continue
        E = D[np.ix_(*blk)] - g
        R, C = E.shape
        m = E.mean()
        ss['pair'] += R * C * m ** 2
        ss['line'] += C * float(((E.mean(1) - m) ** 2).sum()) + R * float(((E.mean(0) - m) ** 2).sum())
        ss['cell'] += float((epi_metrics._dc(E) ** 2).sum())
        n += R * C
    if n == 0:
        return out
    tot = sum(ss.values())
    out['nonadd_sd'] = float(np.sqrt(tot / n))
    if out['nonadd_sd'] >= epi_metrics.FLAT:                  # an additive view's d is a constant plus numerical noise: no shares
        for k in ss:
            out[f'nonadd_{k}'] = ss[k] / tot
    return out


def score(items: Dict[str, Sequence], views: Dict[str, np.ndarray], loader: Sequence[str], link: Optional[Callable] = None,
          trapezoids: Dict[str, tuple] = TRAPEZOIDS) -> Dict[str, Dict[str, float]]:
    """
    ``{head: {metric: value}}`` for every view and trapezoid present in ``views``.

    ``items`` holds the pooled validation items the way ``epi_metrics.Table`` takes them (``mut_key``, ``subset_type``, ``cens``, ``ddG``,
    ``dddG``, ``dG_wt``); ``loader`` names each item's library; ``link`` is the run's numpy link on absolute dG, or None.
    """
    st = np.asarray([routing.canonical_subset(s) for s in items['subset_type']])
    gt = np.asarray(items['ddG'], float)
    cens = np.zeros(len(gt), int) if items.get('cens') is None else np.asarray(items['cens'], int)
    loader = np.asarray(loader).astype(str)
    ok = np.isfinite(gt) & (cens == 0)
    res = {}
    for name, sc in heads(views, trapezoids).items():
        sc = np.asarray(sc, float)
        row = {'rho_single': _per_library_rho(sc, gt, loader, ok & (st == 'single')),
               'rho_double': _per_library_rho(sc, gt, loader, ok & (st == 'double'))}
        t = epi_metrics.Table({**items, 'wt': sc, 'mt': sc, 'comb': sc, 'wt_rev': None, 'ctx': None}, link)
        epi = epi_metrics.compute(t, {m: ('comb',) for m in EPI})
        row.update({m: epi.get(f'epi_{m}_comb', np.nan) for m in EPI})
        row.update(nonadd_levels(t))
        res[name] = row
    return res


def log_name(head: str, metric: str) -> str:
    return f"probe_{head.replace('.', '_').replace('+', '_')}_{metric}"
