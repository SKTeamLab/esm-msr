#!/usr/bin/env python
"""p-values for ``01_summary.thermostability_sets``: does ESM-MSR beat the other models on each set?

The figure's points are the seed means (seeds 1-3 of every retrained model) of the nine unified models (ESM-MSR WT head on the single-mutant sets, combined on PTMUL).  Here the same predictions are bootstrapped with many draws and every model is
compared with ESM-MSR **on the same draws** (paired), so the p-value of ``rho(ESM-MSR) - rho(model)`` is the share of draws on the other side of zero:

    p = max(2 min(P(d <= 0), P(d >= 0)), 1 / (B + 1)),   d = rho_ESM-MSR - rho_model over the draws.

Two resampling units, as in ``02_thermostability``: **mutation** (the figure's whiskers; B = ``B_MUT``) and **protein** (whole proteins, the dependence between the mutations of one protein is kept; B = ``B_PROT``).
Models with documented training overlap (the red rings of the figure: GeoDDG-3D on every set) are shown in the pairwise table but left out of the best-competitor test and of the Holm correction.

  thermostability_sets_pvalues.csv   one row per set and model: rho of both, difference, 95% interval, p and Holm-adjusted p (over the models without overlap of the set), for both units
  thermostability_sets_best.csv      one row per set: ESM-MSR against the best other model without overlap.  ``vs best``: the draws of ``rho_ESM-MSR - max over the other models`` (the maximum is taken inside each
                                     draw, so the choice of the runner-up is part of the test); p raw and Holm-adjusted over the sets
  thermostability_sets_pvalues.md    the same two tables

    python analysis/01b_thermostability_pvalues.py [--b-mut 1000] [--b-prot 1000]   # the paper wants 200000 / 100000 [--workers 6] [--only-heads]
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msr_analysis.boot import bootstrap_rho  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.external import EXT_ORDER, fresh_frames, seed_frame  # noqa: E402
from msr_analysis.registry import LABEL, RESULTS  # noqa: E402

OUT = out_dir('01_summary')
CHUNK = 5000
_FR = {}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[o])
    out = np.empty(len(p))
    out[o] = np.maximum.accumulate(adj)
    return out


def p_two_sided(d):
    d = d[np.isfinite(d)]
    return float(max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / (len(d) + 1)))


def _job(a):
    ds, kind, B, seed = a
    fr = _FR[ds]
    pt, rp = bootstrap_rho({ds: fr}, list(fr.preds), kind=kind, B=B, seed=seed, per_protein=False)
    return ds, kind, {m: v for m, v in rp[ds].items()}, {m: (v[0], v[2]) for m, v in pt[ds].items()}


def run(sets, b_mut, b_prot, workers, S):
    for ds, fr in fresh_frames(S[S.dataset.isin(sets)]).items():
        _FR[ds] = seed_frame(ds, fr)[0]
    jobs = []
    for ds in sets:
        for kind, B in (('mutation', b_mut), ('protein', b_prot)):
            for i in range(int(np.ceil(B / CHUNK))):
                jobs.append((ds, kind, min(CHUNK, B - i * CHUNK), 1000 * (1 + (kind == 'protein')) + i))
    print(f'{len(jobs)} chunks, B = {b_mut:,} (mutation) and {b_prot:,} (protein) per set', flush=True)
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    reps, pts = {}, {}
    for ds, kind, rp, pt in res:
        for m, v in rp.items():
            reps.setdefault((ds, kind, m), []).append(v)
        pts[ds] = pt
    return {k: np.concatenate(v) for k, v in reps.items()}, pts


_HEADS = {}
HEAD_SETS = ('ptmul', 'ptmuld')


def _head_job(a):
    ds, kind, B, seed, ns = a
    fr = _HEADS[(ds, ns)]
    pt, rp = bootstrap_rho({ds: fr}, list(fr.preds), kind=kind, B=B, seed=seed, per_protein=False)
    return ds, kind, ns, {m: v for m, v in rp[ds].items()}, {m: v[0] for m, v in pt[ds].items()}


def msr_heads(b_mut, b_prot, workers, datasets=HEAD_SETS):
    """ESM-MSR Dual-view against its WT head on the multi-mutant sets, paired, on the mean of seeds 1-3 of FINAL_p0 (seeds resampled with the draws).
    PTMUL-D is the doubles-only slice of PTMUL: its frame holds the PTMUL run's three seeds aligned on the variant id."""
    S = pd.read_csv(RESULTS / '02_thermostability' / 'thermostability_rho.csv')
    for ds in datasets:
        fr = fresh_frames(S[S.dataset == ds])[ds]
        assert fr.preds['msrp0:comb'].shape[0] == 3, 'FINAL_p0 seeds 1-3 expected in the frame (refresh the frame cache)'
        _HEADS[(ds, 3)] = Frame(ds, fr.spine, {'comb': fr.preds['msrp0:comb'], 'wt': fr.preds['msrp0:wt']})
    jobs = [(ds, kind, min(CHUNK, B - i * CHUNK), 7000 + 100 * ns + i, ns) for ds in datasets for ns in (3,) for kind, B in (('mutation', b_mut), ('protein', b_prot)) for i in range(int(np.ceil(B / CHUNK)))]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_head_job, jobs, chunksize=1)
    reps, pts = {}, {}
    for ds, kind, ns, rp, pt in res:
        for m, v in rp.items():
            reps.setdefault((ds, kind, ns, m), []).append(v)
        pts[(ds, ns)] = pt
    rows = []
    for ds in datasets:
        for ns in (3,):
            for kind in ('mutation', 'protein'):
                d = np.concatenate(reps[(ds, kind, ns, 'comb')]) - np.concatenate(reps[(ds, kind, ns, 'wt')])
                d = d[np.isfinite(d)]
                lo, hi = np.quantile(d, [0.025, 0.975])
                pt = pts[(ds, ns)]
                rows.append(dict(dataset=ds, seeds='seeds 1-3 (mean)', unit=kind, rho_combined=pt['comb'], rho_wt=pt['wt'], delta=pt['comb'] - pt['wt'], lo=lo, hi=hi,
                                 p_one_sided=float(max(np.mean(d <= 0), 1.0 / (len(d) + 1))), p_two_sided=p_two_sided(d), B=len(d)))
    H = pd.DataFrame(rows)
    H.to_csv(OUT / 'thermostability_sets_msr_heads.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(H.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b-mut', type=int, default=1000)          # default size (paper: 200000)
    ap.add_argument('--b-prot', type=int, default=1000)         # default size (paper: 100000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--only-heads', action='store_true', help='only the ESM-MSR combined vs WT head test on PTMUL and PTMUL-D')
    a = ap.parse_args()
    S = pd.read_csv(RESULTS / '01_summary' / 'thermostability_sets.csv')
    if a.only_heads:
        msr_heads(a.b_mut, a.b_prot, a.workers)
        return
    sets = list(S.sort_values('order').drop_duplicates('dataset').dataset)
    leak = {(r.dataset, r.model): bool(r.leak_flag) for r in S.itertuples()}
    reps, pts = run(sets, a.b_mut, a.b_prot, a.workers, S)
    rows, best = [], []
    for ds in sets:
        others = [m for m in EXT_ORDER if m != 'ESM-MSR' and m in pts[ds]]
        clean = [m for m in others if not leak.get((ds, m), False)]
        msr = pts[ds]['ESM-MSR'][0]
        for kind in ('mutation', 'protein'):
            e = reps[(ds, kind, 'ESM-MSR')]
            ps = {}
            for m in others:
                d = e - reps[(ds, kind, m)]
                lo, hi = np.nanquantile(d, [0.025, 0.975])
                ps[m] = p_two_sided(d)
                rows.append(dict(dataset=ds, unit=kind, model=m, excluded_training_overlap=leak.get((ds, m), False), rho_msr=msr, rho_model=pts[ds][m][0], n_model=pts[ds][m][1],
                                 delta=msr - pts[ds][m][0], lo=lo, hi=hi, p=ps[m], B=len(d)))
            adj = dict(zip(clean, holm([ps[m] for m in clean])))
            for r in rows:
                if r['dataset'] == ds and r['unit'] == kind:
                    r['p_holm'] = adj.get(r['model'], np.nan)
            mx = np.max(np.vstack([reps[(ds, kind, m)] for m in clean]), axis=0)
            d = e - mx
            top = max(clean, key=lambda m: pts[ds][m][0])
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            best.append(dict(dataset=ds, unit=kind, rho_msr=msr, best_model=top, rho_best=pts[ds][top][0], delta_point=msr - pts[ds][top][0], delta_vs_max_mean=float(np.nanmean(d)), lo=lo, hi=hi,
                             p_vs_best=p_two_sided(d), p_vs_runner_up=ps[top], msr_is_top=bool(msr > pts[ds][top][0]), B=len(d)))
    P, Bt = pd.DataFrame(rows), pd.DataFrame(best)
    Bt['p_vs_best_holm_sets'] = np.nan
    for kind in ('mutation', 'protein'):
        m = Bt.unit == kind
        Bt.loc[m, 'p_vs_best_holm_sets'] = holm(Bt.loc[m, 'p_vs_best'])
    P.to_csv(OUT / 'thermostability_sets_pvalues.csv', index=False, float_format='%.6g')
    Bt.to_csv(OUT / 'thermostability_sets_best.csv', index=False, float_format='%.6g')
    fmt = lambda p: '<%.0e' % (1.0 / (a.b_mut + 1)) if p <= 1.0 / (a.b_mut + 1) + 1e-12 else f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}'
    lines = ['**ESM-MSR against the best other model without training overlap** (paired bootstrap, seeds 1-3 averaged)', '',
             '| set | ESM-MSR ρ | best other (ρ) | Δρ | p, mutation resampling | p, protein resampling | p vs best, Holm over sets (mutation / protein) |', '|---|---|---|---|---|---|---|']
    for ds in sets:
        m_, p_ = (Bt[(Bt.dataset == ds) & (Bt.unit == u)].iloc[0] for u in ('mutation', 'protein'))
        lines.append(f'| {LABEL.get(ds, ds)} | {m_.rho_msr:.3f} | {m_.best_model} ({m_.rho_best:.3f}) | {m_.delta_point:+.3f} | {fmt(m_.p_vs_best)} | {fmt(p_.p_vs_best)} | {fmt(m_.p_vs_best_holm_sets)} / {fmt(p_.p_vs_best_holm_sets)} |')
    lines += ['', '**Pairwise: ESM-MSR against every model** (Δρ = ESM-MSR − model; p two-sided; Holm over the models without overlap of the set; GeoDDG-3D has training overlap and is not corrected)', '',
              '| set | model | ρ model | Δρ | 95% interval (mutation) | p (mutation) | Holm | p (protein) | Holm |', '|---|---|---|---|---|---|---|---|---|']
    for ds in sets:
        for m in [m for m in EXT_ORDER if m != 'ESM-MSR']:
            a_ = P[(P.dataset == ds) & (P.model == m) & (P.unit == 'mutation')]
            b_ = P[(P.dataset == ds) & (P.model == m) & (P.unit == 'protein')]
            if not len(a_):
                continue
            a_, b_ = a_.iloc[0], b_.iloc[0]
            lines.append(f'| {LABEL.get(ds, ds)} | {m}{" *" if a_.excluded_training_overlap else ""} | {a_.rho_model:.3f} | {a_.delta:+.3f} | [{a_.lo:+.3f}, {a_.hi:+.3f}] | {fmt(a_.p)} | '
                         f'{"–" if a_.excluded_training_overlap else fmt(a_.p_holm)} | {fmt(b_.p)} | {"–" if b_.excluded_training_overlap else fmt(b_.p_holm)} |')
    (OUT / 'thermostability_sets_pvalues.md').write_text('\n'.join(lines) + '\n')
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(Bt.round(4).to_string(index=False))
    msr_heads(a.b_mut, a.b_prot, a.workers)


if __name__ == '__main__':
    main()
