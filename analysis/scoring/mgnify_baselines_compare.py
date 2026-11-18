"""ThermoMPNN(-D) and SPURS(-multi) (capped retrains, 3 seeds each) against ESM-MSR FINAL_p0 on the MGnify scaffolds and Rocklin designs, on the
rows EVERY listed model has scored (the baselines write 20k-row chunks, so a partial run covers the scaffolds of its finished chunks only: the
comparison is always on the common rows, never on each model's own coverage).

Matched like for like.  The baselines learn the saturation (the floor of the assay) implicitly, ESM-MSR needs the Tobit link for it, so ESM-MSR
is scored on the observed scale, h(dG_wt + latent) - dG_wt (esm_msr.val_stats.tobit_link, FINAL_p0 constants 0.28 / -0.14 / 0.5, applied post hoc).
And the heads pair by what they are:
  WT-only models      ESM-MSR WT head (wt_lora_pred, linked)   vs  SPURS (SPURS_single_additive)      and  ThermoMPNN (pred_single_additive)
  combined models     ESM-MSR combined (combined_pred, linked) vs  SPURS-multi (SPURS_multi)           and  ThermoMPNN-D (pred_combined)
The latent ESM-MSR heads are listed as reference rows (point values only).  Metric: pooled Spearman vs ddG_ML per seed, reported as the seed mean
(+- sd over seeds); in the in-range regime (in_range_1_5) a 95% interval from resampling whole scaffolds, the same draws for every model, and the
paired difference of each baseline to its matched ESM-MSR head.  Epistasis (doubles with both singles measured and uncensored, wild-type backbones,
as analysis/scoring/mgnify_epistasis_beyond.py): the combined model's double minus an additive composition of singles, beyond rho and naive rho
(point values); ESM-MSR is shown with the Megascale tables' readout (WT-head singles) and with its own combined singles, linked and latent;
SPURS-multi and ThermoMPNN-D use their own additive composition (SPURS and ThermoMPNN, which add their singles, have none).

  python analysis/scoring/mgnify_baselines_compare.py final_mgnify [--boot 100] [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
sys.path.insert(0, HERE)
from esm_msr import epi_metrics as E  # noqa: E402
from esm_msr.val_stats import tobit_link  # noqa: E402
from mgnify_epistasis_beyond import parse, stats as epi_stats  # noqa: E402
from mgnify_labels import default_table, legacy_cens  # noqa: E402

SEEDS = (1, 2, 3)
LINK = tobit_link(0.28, -0.14, 0.5)
ESM_WT, ESM_COMB = 'ESM-MSR WT head (linked)', 'ESM-MSR combined (linked)'
# baseline -> (seed-dir template, prediction column, own-additive column or None, matched ESM-MSR head)
BASELINES = {
    'SPURS': ('spurs_unfrozen_s{}', 'SPURS_single_additive', None, ESM_WT),
    'ThermoMPNN': ('thermompnn_s{}', 'pred_single_additive', None, ESM_WT),
    'SPURS-multi': ('spurs_unfrozen_s{}', 'SPURS_multi', 'SPURS_multi_additive', ESM_COMB),
    'ThermoMPNN-D': ('thermompnn_s{}', 'pred_combined', 'pred_single_additive', ESM_COMB),
    # zero-shot models: one run, no seeds (the template has no {}); they have no WT/combined split, so they pair with the WT head.  Run them with
    # --models SPURS,ThermoMPNN,<zero-shot model> so the common rows are the zero-shot model's own coverage (it may still be running).
    'ProteinMPNN': ('proteinmpnn', 'mpnn_score', None, ESM_WT),
    'ProSST': ('prosst', 'prosst', None, ESM_WT),
    'ESM3 (zero-shot)': ('esm3_zeroshot', 'wt_lora_pred', None, ESM_WT),
}
TRAINED = 'SPURS,ThermoMPNN,SPURS-multi,ThermoMPNN-D'
REFERENCE = ['ESM-MSR WT head (latent)', 'ESM-MSR combined (latent)']       # point values only
LAB = ['uid', 'source', 'code', 'code_wt', 'n_mut', 'mut_type', 'ddG_ML', 'dddG_ML', 'dG_wt', 'cens', 'in_range_1_5', 'backbone_mutation', 'has_all_singles']


def labels():
    d = pd.read_csv(default_table(), usecols=LAB, low_memory=False)
    d['cens'] = legacy_cens(d['cens'])
    return d.set_index('uid')


def load_esm(root, cols):
    out = {}
    for s in SEEDS:
        fr = [pd.read_csv(os.path.join(root, 'FINAL_p0', f's{s}', sub, 'predictions.csv.gz'), usecols=['uid'] + cols, low_memory=False) for sub in ('all', 'rdenovo')]
        out[s] = pd.concat(fr, ignore_index=True).set_index('uid')
    return out


_CACHE = {}


def load_chunks(root, tpl, s):
    d = os.path.join(root, 'baselines', tpl.format(s))
    if d not in _CACHE:
        fs = glob.glob(os.path.join(d, 'mgnify', '*', 'chunk_*.csv.gz'))
        if fs:
            _CACHE[d] = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True).drop_duplicates('id').set_index('id')
        else:                                                   # the ESM3 runner writes shards keyed by uid
            fs = sorted(glob.glob(os.path.join(d, 'mgnify', 'shards', 'shard_*.csv')))
            _CACHE[d] = pd.concat([pd.read_csv(f, usecols=['uid', 'wt_lora_pred'], low_memory=False) for f in fs], ignore_index=True).drop_duplicates('uid').set_index('uid') if fs else None
    return _CACHE[d]


def sp(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 5 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--models', default=TRAINED, help='comma list of BASELINES keys to compare (default: the four trained baselines)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'baselines_vs_msr')
    os.makedirs(out, exist_ok=True)
    lab = labels()
    esm = load_esm(a.root, ['wt_lora_pred', 'combined_pred', 'combined_pred_additive'])
    raw = {}                                         # name -> seed -> Series over uid (baselines) ; own-additive likewise
    add = {}
    for m, (tpl, col, acol, _) in {k: v for k, v in BASELINES.items() if k in a.models.split(',')}.items():
        for s in SEEDS:
            d = load_chunks(a.root, tpl, s)
            if d is None:
                continue
            raw.setdefault(m, {})[s] = d[col].astype(float)
            if acol:
                add.setdefault(m, {})[s] = d[acol].astype(float)
    for m in [m for m, v in raw.items() if len(v) < len(SEEDS)]:           # a model joins the comparison once all its seeds have chunks (else the common rows collapse to its first chunk)
        print(f'{m}: only seeds {sorted(raw[m])} have output so far, left out')
        raw.pop(m); add.pop(m, None)
    have = {m: sorted(v) for m, v in raw.items()}
    print('baselines with data (seeds):', have)
    ids = esm[1].index
    for v in raw.values():
        for x in v.values():
            ids = ids.intersection(x.index[x.notna()])
    L = lab.loc[ids].copy()
    L = L[L['ddG_ML'].notna() & L['dG_wt'].notna()]
    ids = L.index
    L['code_wt'] = L['code_wt'].fillna(L['code'])
    print(f'common rows with a label: {len(L)} over {L.code_wt.nunique()} scaffolds ({L.source.value_counts().to_dict()})')
    dg = L['dG_wt'].to_numpy(float)
    link = lambda v: LINK(dg + v) - dg
    P = {}
    for s in SEEDS:
        wt, cb = esm[s]['wt_lora_pred'].loc[ids].to_numpy(float), esm[s]['combined_pred'].loc[ids].to_numpy(float)
        P[('ESM-MSR WT head (latent)', s)], P[('ESM-MSR combined (latent)', s)] = wt, cb
        P[(ESM_WT, s)], P[(ESM_COMB, s)] = link(wt), link(cb)
    for m, v in raw.items():
        for s, x in v.items():
            P[(m, s)] = x.loc[ids].to_numpy(float)
    have.update({n: list(SEEDS) for n in (ESM_WT, ESM_COMB, *REFERENCE)})
    names = [ESM_WT, *[m for m in BASELINES if BASELINES[m][3] == ESM_WT and m in raw], ESM_COMB, *[m for m in BASELINES if BASELINES[m][3] == ESM_COMB and m in raw], *REFERENCE]
    ref_of = {m: BASELINES[m][3] for m in BASELINES}
    y = L['ddG_ML'].to_numpy(float)
    inr = L['in_range_1_5'].astype(bool).to_numpy()
    nm, src = L['n_mut'].to_numpy(), L['source'].to_numpy()
    cls = pd.read_csv(os.environ.get('MGNIFY_CLASSES', os.path.expanduser('~/playground/esm-msr-devel/leak_audit/out/mgnify_homology_classes.csv')),
                      usecols=['code_wt', 'cls']).set_index('code_wt')['cls']
    hc = L['code_wt'].map(cls).to_numpy()
    strata = {'all': np.ones(len(L), bool), 'MGnify': src == 'mgnify', 'Rocklin designs': src == 'rocklin_denovo',
              'MGnify, no/low homology': (src == 'mgnify') & np.isin(hc, ['none', 'low']),
              'MGnify, strong homology': (src == 'mgnify') & np.isin(hc, ['H', 'sameprot'])}
    subsets = {'all rows': nm >= 1, 'singles': nm == 1, 'doubles': nm == 2, 'triples': nm == 3}
    codes, inv = np.unique(L['code_wt'].to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    rng = np.random.default_rng(0)
    draws = [np.concatenate([order[bounds[i]:bounds[i + 1]] for i in rng.integers(0, len(codes), len(codes))]) for _ in range(a.boot)]
    boot_names = [n for n in names if n not in REFERENCE]
    rows = []
    for sn, sm in strata.items():
        for sub, mm in subsets.items():
            for reg, rm in (('in_range', inr), ('all rows', np.ones(len(L), bool))):
                mask = sm & mm & rm
                if mask.sum() < 50:
                    continue

                def seedmean(idx, which):
                    return {n: np.nanmean([sp(P[(n, s)][idx], y[idx]) for s in have[n]]) for n in which}
                idx0 = np.flatnonzero(mask)
                pt = seedmean(idx0, names)
                do_boot = a.boot and reg == 'in_range'          # intervals for the in-range regime only (the all-rows regime is a point value)
                if do_boot:
                    bs = pd.DataFrame([seedmean(dr[mask[dr]], boot_names) if mask[dr].sum() > 50 else {n: np.nan for n in boot_names} for dr in draws])
                for n in names:
                    sds = np.std([sp(P[(n, s)][idx0], y[idx0]) for s in have[n]]) if len(have[n]) > 1 else np.nan
                    r = dict(stratum=sn, subset=sub, regime=reg, model=n, n_rows=int(mask.sum()), n_scaffolds=int(len(np.unique(inv[idx0]))),
                             n_seeds=len(have[n]), spearman=pt[n], seed_sd=sds)
                    if do_boot and n in boot_names:
                        r['lo'], r['hi'] = np.nanpercentile(bs[n], [2.5, 97.5])
                        if n in ref_of:
                            r['matched_msr'] = ref_of[n]
                            r['diff_vs_msr'] = pt[n] - pt[ref_of[n]]
                            r['diff_lo'], r['diff_hi'] = np.nanpercentile(bs[n] - bs[ref_of[n]], [2.5, 97.5])
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, 'pooled_spearman.csv'), index=False, float_format='%.4f')

    # epistasis: doubles with both singles measured + uncensored, WT backbones
    bbm = L['backbone_mutation'].map(lambda v: str(v).lower() not in ('nan', 'none', '', 'false', '0')).to_numpy()
    use = (L['cens'].to_numpy() == 0) & ~bbm & np.isin(nm, [1, 2])
    dd = L['dddG_ML'].to_numpy(float)
    erows = []
    for g, gm in (('MGnify', src == 'mgnify'), ('Rocklin designs', src == 'rocklin_denovo'), ('Together', np.ones(len(L), bool))):
        idx = np.flatnonzero(use & gm)
        keys = [parse(m, c) for m, c in zip(L['mut_type'].to_numpy()[idx], L['code_wt'].astype(str).to_numpy()[idx])]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        base = dict(mut_key=keys, subset_type=st, ddG=y[idx], dddG=dd[idx], dG_wt=dg[idx])
        nd = int((st == 'double').sum())

        def own_add(cb, ad):                                    # a single's own prediction is its additive value
            return np.where(nm[idx] == 1, cb, ad)

        def run(label, per_seed, link_fn):
            v = np.array([epi_stats(E.Table(dict(base, wt=wt[idx] if False else wt, mt=cb, comb=cb), link_fn, heads=('epi', 'add'))) for wt, cb in per_seed], float)
            erows.append(dict(group=g, model=label, n_seeds=len(per_seed), n_doubles=nd, beyond_rho=v[:, 0].mean(), naive_rho=v[:, 1].mean()))
        for scale, lk in (('latent', None), ('linked', LINK)):
            run(f'ESM-MSR combined, tables readout (WT-head singles), {scale}',
                [(esm[s]['wt_lora_pred'].loc[ids].to_numpy(float)[idx], esm[s]['combined_pred'].loc[ids].to_numpy(float)[idx]) for s in SEEDS], lk)
            run(f'ESM-MSR combined, own-additive singles, {scale}',
                [(own_add(esm[s]['combined_pred'].loc[ids].to_numpy(float)[idx], esm[s]['combined_pred_additive'].loc[ids].to_numpy(float)[idx]),
                  esm[s]['combined_pred'].loc[ids].to_numpy(float)[idx]) for s in SEEDS], lk)
        for m in ('SPURS-multi', 'ThermoMPNN-D'):
            if m in add:
                run(f'{m} (own additive)', [(own_add(P[(m, s)][idx], add[m][s].loc[ids].to_numpy(float)[idx]), P[(m, s)][idx]) for s in have[m]], None)
    eres = pd.DataFrame(erows)
    eres.to_csv(os.path.join(out, 'epistasis.csv'), index=False, float_format='%.4f')
    pd.DataFrame([dict(model=n, seeds=str(have[n])) for n in names]).to_csv(os.path.join(out, 'coverage.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        for sub in ('all rows', 'singles', 'doubles', 'triples'):
            h = res[(res.regime == 'in_range') & (res.subset == sub)]
            if len(h):
                print(f'\n== in-range, {sub}')
                print(h[[c for c in ['stratum', 'model', 'n_rows', 'n_scaffolds', 'spearman', 'lo', 'hi', 'diff_vs_msr', 'diff_lo', 'diff_hi'] if c in h]].round(3).to_string(index=False))
        print('\n== epistasis'); print(eres.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
