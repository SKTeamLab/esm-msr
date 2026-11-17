"""Score a finished MGnify run (esm_msr_testing.py --mgnify) by homology of each scaffold to the capped training set.

Needs ``mgnify_homology_classes.csv`` from ``analysis/leak_audit/20_mgnify_vs_capped_train.py`` (classes none / low / h / H / sameprot;
sequence evidence only, coverage over the MGnify scaffold).  Strata: all, clean (= none + low), none, low, h, strong (= H + sameprot).
Every interval resamples whole scaffolds (the unit that shares a homology class and a backbone).  Headline metrics are pooled Spearman of
combined_pred (latent and Tobit-linked, the link applied post hoc as in mgnify_link.py) against ddG_ML.  Epistasis, latent only:
  * measured: doubles with both constituent singles assayed and uncensored (dddG_ML; the Megascale tables' rows) vs the model's predicted
    dddG, headline ``epi_msr`` = comb(AB) - WT_A - WT_B (the tables' readout; ``combined_dddg_pred`` / ``mt_lora_dddg_pred`` kept for reference);
  * ranking: within-scaffold Spearman on in-range doubles, combined head vs the same head's additive composition of its singles.

  python analysis/scoring/mgnify_by_homology.py final_mgnify/FINAL_p0/s1/heldout [more run dirs ...] [--boot 100]

Writes ``stats_by_homology.csv`` and ``epistasis_by_homology.csv`` into each run dir.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src'))
from esm_msr.val_stats import tobit_link  # noqa: E402
sys.path.insert(0, HERE)
from mgnify_labels import apply_v2_labels  # noqa: E402

def _find_classes():
    if os.environ.get('LEAK_OUT_DIR'):
        return os.path.join(os.environ['LEAK_OUT_DIR'], 'mgnify_homology_classes.csv')
    d = HERE
    while d != os.path.dirname(d):  # LEAK_ROOT = first ancestor holding leak_audit/out
        f = os.path.join(d, 'leak_audit', 'out', 'mgnify_homology_classes.csv')
        if os.path.exists(f):
            return f
        d = os.path.dirname(d)
    return 'mgnify_homology_classes.csv'


CLASSES = _find_classes()
STRATA = {'all': None, 'clean(none+low)': ['none', 'low'], 'none': ['none'], 'low': ['low'], 'h': ['h'], 'strong(H+same)': ['H', 'sameprot']}


def sp(x, y):
    if len(x) < 5:
        return np.nan
    a, b = rankdata(x), rankdata(y)
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(d, stat, B, rng):
    """95% interval of ``stat(frame)`` resampling scaffolds; ``d`` has a code_wt column."""
    if B <= 0 or len(d) < 5:
        return np.nan, np.nan
    codes, inv = np.unique(d['code_wt'].to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(codes) + 1))
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(codes), len(codes))
        idx = np.concatenate([order[bounds[i]:bounds[i + 1]] for i in pick])
        vals.append(stat(d.iloc[idx]))
    v = np.array(vals, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--classes', default=CLASSES)
    ap.add_argument('--hparams', default=None, help="checkpoint hparams.yaml for the Tobit constants (default FINAL_p0: 0.28/-0.14/0.5)")
    ap.add_argument('--boot', type=int, default=100)
    ap.add_argument('--v1_labels', action='store_true', help='keep the labels stored in the run (v1 table) instead of the v2 table')
    a = ap.parse_args()
    c = dict(tobit_sigma_y=0.28, tobit_mu_f=-0.14, tobit_sigma_f=0.5)
    if a.hparams:
        hp = yaml.safe_load(open(a.hparams))
        c = {k: float(hp[k]) for k in c}
    h = tobit_link(c['tobit_sigma_y'], c['tobit_mu_f'], c['tobit_sigma_f'])
    cls = pd.read_csv(a.classes, usecols=['code_wt', 'cls']).set_index('code_wt')['cls']
    rng = np.random.default_rng(0)
    for rd in a.run_dirs:
        p = os.path.join(rd, 'predictions.csv.gz')
        if not os.path.exists(p):  # unfinished run: concatenate the shards
            import glob
            d = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(rd, 'shards', '*.csv')))], ignore_index=True)
        else:
            d = pd.read_csv(p, low_memory=False)
        if not a.v1_labels:
            d = apply_v2_labels(d)
        d['cls'] = d['code_wt'].map(cls)
        miss = d['cls'].isna().sum()
        if miss:
            print(f'{rd}: {miss} rows without a homology class (dropped)')
            d = d[d['cls'].notna()]
        d['in_range'] = d['in_range_1_5'].astype(bool)
        dg = d['dG_wt'].astype(float)
        d['combined_linked'] = h(dg + d['combined_pred']) - dg
        # Megascale-table epistasis rows: both constituent singles measured and uncensored (analysis/msr_analysis/common.epi_rows); readout
        # comb(AB) - WT_A - WT_B ('epi_msr', the tables' definition)
        sing = d[d['n_mut'] == 1].set_index(['code_wt', 'mut_type'])['cens']
        def _both(r):
            return all(sing.get((r.code_wt, x), np.nan) == 0 for x in r.mut_type.split(':'))
        dd2 = d[d['n_mut'] == 2]
        d['singles_uncens'] = False
        d.loc[dd2.index[[_both(r) for r in dd2.itertuples()]], 'singles_uncens'] = True
        d['epi_msr'] = d['combined_pred'] - d['wt_lora_pred_additive']
        d = d[d['ddG_ML'].notna() & d['combined_pred'].notna()]
        rows = []
        for sn, keep in STRATA.items():
            s_all = d if keep is None else d[d['cls'].isin(keep)]
            for reg, rm in (('in_range', s_all['in_range']), ('all_rows', pd.Series(True, index=s_all.index))):
                for sub, mm in (('all', s_all['n_mut'] >= 1), ('singles', s_all['n_mut'] == 1), ('doubles', s_all['n_mut'] == 2)):
                    s = s_all[rm & mm]
                    if len(s) < 10:
                        continue
                    for form, col in (('latent', 'combined_pred'), ('linked', 'combined_linked')):
                        stat = lambda f, col=col: sp(f[col].to_numpy(), f['ddG_ML'].to_numpy())
                        lo, hi = boot_ci(s, stat, a.boot, rng)
                        rows.append(dict(stratum=sn, regime=reg, subset=sub, form=form, n_rows=len(s), n_scaffolds=s['code_wt'].nunique(),
                                         spearman=stat(s), lo=lo, hi=hi))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(rd, 'stats_by_homology.csv'), index=False, float_format='%.4f')
        # --- epistasis ---
        dbl = d[d['n_mut'] == 2]
        er = []
        for sn, keep in STRATA.items():
            s_all = dbl if keep is None else dbl[dbl['cls'].isin(keep)]
            m = s_all[s_all['has_all_singles'].fillna(False).astype(bool) & s_all['dddG_ML'].notna() & s_all['singles_uncens']]
            for head in ('epi_msr', 'combined_dddg_pred', 'mt_lora_dddg_pred'):
                s = m[m[head].notna()]
                if len(s) >= 10:
                    stat = lambda f, head=head: sp(f[head].to_numpy(), f['dddG_ML'].to_numpy())
                    lo, hi = boot_ci(s, stat, a.boot, rng)
                    er.append(dict(stratum=sn, test=f'measured dddG vs {head}', n_rows=len(s), n_scaffolds=s['code_wt'].nunique(), value=stat(s), lo=lo, hi=hi))
            r = s_all[s_all['in_range']]
            g = r.groupby('code_wt')
            keep_codes = g.size()[g.size() >= 10].index
            r = r[r['code_wt'].isin(keep_codes)]
            if r['code_wt'].nunique() >= 5:
                per = pd.DataFrame({k: r.groupby('code_wt').apply(lambda f, k=k: sp(f[k].to_numpy(), f['ddG_ML'].to_numpy()))
                                    for k in ('combined_pred', 'combined_pred_additive', 'wt_lora_pred')})
                per = per.dropna()
                for name, v in (('combined', per['combined_pred']), ('additive composition', per['combined_pred_additive']),
                                ('combined - additive', per['combined_pred'] - per['combined_pred_additive']), ('wt head', per['wt_lora_pred'])):
                    bs = [v.sample(len(v), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(max(a.boot, 1))]
                    er.append(dict(stratum=sn, test=f'within-scaffold doubles rho: {name}', n_rows=int(len(r)), n_scaffolds=len(v), value=float(v.mean()),
                                   lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
        pd.DataFrame(er).to_csv(os.path.join(rd, 'epistasis_by_homology.csv'), index=False, float_format='%.4f')
        print(f'== {rd}  (Tobit {c})')
        pd.set_option('display.width', 200)
        print(out[(out.regime == 'in_range') & out.subset.isin(['all', 'doubles'])].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        print(pd.DataFrame(er).to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
