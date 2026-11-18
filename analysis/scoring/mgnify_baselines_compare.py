"""ThermoMPNN(-D) and SPURS(-multi) (capped retrains, 3 seeds each) against ESM-MSR FINAL_p0 on the MGnify scaffolds and Rocklin designs, on the
rows EVERY listed model has scored (the baselines write 20k-row chunks, so a partial run covers the scaffolds of its finished chunks only: the
comparison is always on the common rows, never on each model's own coverage).

Models (positive = stabilizing, all raw regression outputs, no link):
  ESM-MSR        combined_pred of final_mgnify/FINAL_p0/s{1,2,3} (latent)
  SPURS          SPURS_single_additive      SPURS-multi   SPURS_multi        (spurs_unfrozen_s{1,2,3}: unfrozen adapter, capped split)
  ThermoMPNN     pred_single_additive       ThermoMPNN-D  pred_combined      (thermompnn_s{1,2,3}: capped single + corrected epistatic)
Metric: pooled Spearman vs ddG_ML per seed, reported as the seed mean (+- sd over seeds), with a 95% interval from resampling whole scaffolds
(the same draws for every model, so the ESM-MSR difference is paired).  Regimes: in_range (in_range_1_5) and all rows with a label.  Epistasis (doubles
with both singles measured and uncensored, wild-type backbones, as analysis/scoring/mgnify_epistasis_beyond.py): each model's double minus its own
additive composition of its singles, beyond rho and naive rho (point values; SPURS and ThermoMPNN, which add their singles, have none).

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
from mgnify_epistasis_beyond import parse, stats as epi_stats  # noqa: E402
from mgnify_labels import default_table  # noqa: E402

SEEDS = (1, 2, 3)
# model -> (family dir, seed-dir pattern, column, additive column or None)
MODELS = {
    'ESM-MSR': ('esm', 'combined_pred', 'combined_pred_additive'),
    'SPURS': ('spurs_unfrozen_s{}', 'SPURS_single_additive', None),
    'SPURS-multi': ('spurs_unfrozen_s{}', 'SPURS_multi', 'SPURS_multi_additive'),
    'ThermoMPNN': ('thermompnn_s{}', 'pred_single_additive', None),
    'ThermoMPNN-D': ('thermompnn_s{}', 'pred_combined', 'pred_single_additive'),
}
LAB = ['uid', 'source', 'code', 'code_wt', 'n_mut', 'mut_type', 'ddG_ML', 'dddG_ML', 'dG_wt', 'cens', 'in_range_1_5', 'backbone_mutation', 'has_all_singles']


def labels():
    from mgnify_labels import legacy_cens
    d = pd.read_csv(default_table(), usecols=LAB, low_memory=False)
    d['cens'] = legacy_cens(d['cens'])
    return d.set_index('uid')


def load_esm(root, cols):
    out = {}
    for s in SEEDS:
        fr = []
        for sub in ('all', 'rdenovo'):
            fr.append(pd.read_csv(os.path.join(root, 'FINAL_p0', f's{s}', sub, 'predictions.csv.gz'), usecols=['uid'] + cols, low_memory=False))
        out[s] = pd.concat(fr, ignore_index=True).set_index('uid')
    return out


def load_chunks(root, tpl, s):
    fs = glob.glob(os.path.join(root, 'baselines', tpl.format(s), 'mgnify', '*', 'chunk_*.csv.gz'))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    return d.drop_duplicates('id').set_index('id')


def sp(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if len(a) > 5 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'baselines_vs_msr')
    os.makedirs(out, exist_ok=True)
    lab = labels()
    esm = load_esm(a.root, ['combined_pred', 'combined_pred_additive', 'wt_lora_pred'])
    preds, add = {}, {}                              # model -> seed -> Series over uid
    for m, (tpl, col, acol) in MODELS.items():
        for s in SEEDS:
            if m == 'ESM-MSR':
                d = esm[s]
            else:
                d = load_chunks(a.root, tpl, s)
                if d is None:
                    continue
            preds.setdefault(m, {})[s] = d[col].astype(float)
            if acol:
                add.setdefault(m, {})[s] = d[acol].astype(float)
    add['ESM-MSR (tables: WT head)'] = {s: esm[s]['wt_lora_pred'].astype(float) for s in SEEDS}      # the Megascale tables' readout, comb(AB) - WT_A - WT_B
    have = {m: sorted(v) for m, v in preds.items()}
    have['ESM-MSR (tables: WT head)'] = list(SEEDS)
    print('models with data (seeds):', have)
    ids = None
    for m, v in preds.items():
        for s, x in v.items():
            u = x.index[x.notna()]
            ids = u if ids is None else ids.intersection(u)
    L = lab.loc[ids].copy()
    L = L[L['ddG_ML'].notna()]
    ids = L.index
    L['code_wt'] = L['code_wt'].fillna(L['code'])
    print(f'common rows with a label: {len(L)} over {L.code_wt.nunique()} scaffolds ({L.source.value_counts().to_dict()})')
    P = {(m, s): v.loc[ids].to_numpy() for m, d in preds.items() for s, v in d.items()}
    y = L['ddG_ML'].to_numpy(float)
    inr = L['in_range_1_5'].astype(bool).to_numpy()
    nm = L['n_mut'].to_numpy()
    src = L['source'].to_numpy()
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
    names = list(preds)
    rows = []
    for sn, sm in strata.items():
        for sub, mm in subsets.items():
            for reg, rm in (('in_range', inr), ('all rows', np.ones(len(L), bool))):
                mask = sm & mm & rm
                if mask.sum() < 50:
                    continue
                def seedmean(idx, m):
                    return {n: np.nanmean([sp(P[(n, s)][idx], y[idx]) for s in have[n]]) for n in names}
                idx0 = np.flatnonzero(mask)
                pt = seedmean(idx0, None)
                do_boot = a.boot and reg == 'in_range'          # intervals for the in-range regime only (the all-rows regime is a point value)
                if do_boot:
                    bs = []
                    for dr in draws:
                        k = dr[mask[dr]]
                        bs.append(seedmean(k, None) if len(k) > 50 else {n: np.nan for n in names})
                    bs = pd.DataFrame(bs)
                for n in names:
                    sds = np.std([sp(P[(n, s)][idx0], y[idx0]) for s in have[n]]) if len(have[n]) > 1 else np.nan
                    r = dict(stratum=sn, subset=sub, regime=reg, model=n, n_rows=int(mask.sum()), n_scaffolds=int(len(np.unique(inv[idx0]))),
                             n_seeds=len(have[n]), spearman=pt[n], seed_sd=sds)
                    if do_boot:
                        r['lo'], r['hi'] = np.nanpercentile(bs[n], [2.5, 97.5])
                        if n != 'ESM-MSR':
                            dlt = bs[n] - bs['ESM-MSR']
                            r['diff_vs_msr'] = pt[n] - pt['ESM-MSR']
                            r['diff_lo'], r['diff_hi'] = np.nanpercentile(dlt, [2.5, 97.5])
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, 'pooled_spearman.csv'), index=False, float_format='%.4f')

    # epistasis: doubles with both singles measured + uncensored, WT backbones; model double minus its own additive composition of its singles
    bbm = L['backbone_mutation'].map(lambda v: str(v).lower() not in ('nan', 'none', '', 'false', '0')).to_numpy()
    use = (L['cens'].to_numpy() == 0) & ~bbm & np.isin(nm, [1, 2])
    erows = []
    for g, gm in (('MGnify', src == 'mgnify'), ('Rocklin designs', src == 'rocklin_denovo'), ('Together', np.ones(len(L), bool))):
        idx = np.flatnonzero(use & gm)
        keys = [parse(m, c) for m, c in zip(L['mut_type'].to_numpy()[idx], L['code_wt'].astype(str).to_numpy()[idx])]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        base = dict(mut_key=keys, subset_type=st, ddG=y[idx], dddG=L['dddG_ML'].to_numpy(float)[idx], dG_wt=L['dG_wt'].to_numpy(float)[idx])
        for n in list(add):
            if n not in add:
                continue
            pn = 'ESM-MSR' if n.startswith('ESM-MSR') else n
            vals = []
            for s in have[n]:
                cb = P[(pn, s)][idx]
                ad = add[n][s].loc[ids].to_numpy(float)[idx]
                if 'WT head' not in n:
                    ad = np.where(nm[idx] == 1, cb, ad)        # a single's own prediction is its additive value (the stored column is blank for singles on some models)
                t = E.Table(dict(base, wt=ad, mt=P[(pn, s)][idx], comb=P[(pn, s)][idx]), None, heads=('epi', 'add'))
                vals.append(epi_stats(t))
            v = np.array(vals, float)
            erows.append(dict(group=g, model=n, n_seeds=len(have[n]), n_doubles=int((st == 'double').sum()), beyond_rho=v[:, 0].mean(), naive_rho=v[:, 1].mean()))
    eres = pd.DataFrame(erows)
    eres.to_csv(os.path.join(out, 'epistasis.csv'), index=False, float_format='%.4f')
    cov = pd.DataFrame([dict(model=n, seeds=str(have[n])) for n in names])
    cov.to_csv(os.path.join(out, 'coverage.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        h = res[(res.regime == 'in_range') & (res.subset == 'all rows')]
        print(h[[c for c in ['stratum', 'model', 'n_rows', 'n_scaffolds', 'n_seeds', 'spearman', 'lo', 'hi', 'diff_vs_msr', 'diff_lo', 'diff_hi'] if c in h]].round(3).to_string(index=False))
        print(eres.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
