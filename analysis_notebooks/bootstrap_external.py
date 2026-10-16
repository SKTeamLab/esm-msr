#!/usr/bin/env python
"""
Bootstrap significance tests for ESM-MSR vs. the next-best model on the external benchmarks.

Regenerates the main Spearman figure of compare_supervised_external_summarize.ipynb from the
prediction files outside this repo (external_loader.PRED_ROOT), using ONE replicate per model,
and adds paired bootstrap tests of the difference in Spearman rho between ESM-MSR and the best
competing model. SPURS (training overlap with Ssym) is excluded from every comparison and from
the figure. ESM-MSR sigma=1.25 is not a candidate; it is dropped as an ESM-MSR variant and is not
a competitor.

Hypotheses are grouped into families, each Bonferroni-corrected on its own:
  stability   the 10 external thermostability sets (panel A); ESM-MSR sigma=1 only; N = 10
  dms         the 6 single-protein DMS sets (Faure DLG4/GRB2 abundance+binding, Kung myoglobin,
              Nutschel EstA); sigma in {0.5, 1}; N = 6 datasets x 2 sigma = 12 (N = 18 reported too,
              for readers who count three sigma values)
  domainome   Beltran domainome (aggregate of 522 domains); sigma in {0.5, 1}; N = 2
Megascale validation/test are plotted but not tested (in-distribution; validation picked checkpoints).

Resampling units:
  protein   whole proteins (the `code` column) with replacement -- the clustered bootstrap. The
            stability sets share proteins, so they are resampled JOINTLY (one draw over the union
            of PDB ids applied to every set). Undefined for the single-protein DMS sets (n/a).
  mutation  individual mutations (within a domain for per-domain-averaged sets); treats mutations
            as independent.

Per dataset and tested sigma: p = one-sided bootstrap P(rho_ESM-MSR - rho_next-best <= 0), where
next-best is the best non-ESM-MSR column by observed rho. "Overall" tests per family: macro-average
(equal weight per dataset) of the difference to the per-dataset next-best, and to each method
present in every dataset of the family (Holm-adjusted). Every replicate resamples all models on
the same rows, so comparisons are paired.

Spearman under resampling is computed with count weights (a replicate row appearing k times has
weight k; ties, including a row tied with its own copies, share the average rank), which equals
scipy's spearmanr on the expanded sample but needs no re-sorting per replicate.

    python bootstrap_external.py --out bootstrap_external_results --n-boot 10000
"""
import argparse
import hashlib
import os
import pickle
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import external_loader as L  # noqa: E402

HIGHLIGHT = 'ESM-MSR_1'          # plain ESM-MSR (sigma=1), replicate 1
SIGMA_HALF = 'ESM-MSR (σ=0.5)_1'
EXCLUDED = ['SPURS', 'SPURS-multi']  # training overlap with Ssym (and Megascale): never a competitor
UNTESTED = ['hyperopt_splits-val', 'hyperopt_splits-test']  # plotted, no test

# (dataset key, label) in the order of the notebook's two Spearman panels
PANEL_A = [('ssym', 'Ssym'), ('s461', 'S461'), ('s783', 'S783'), ('k2369', 'K2369'), ('s8754', 'S8754'),
           ('s2648', 'S2648'), ('q3421', 'Q3421'), ('ptmuld', 'PTMUL-D'), ('PTMUL', 'PTMUL'), ('s571', 'S571')]
PANEL_B = [('hyperopt_splits-val', 'Megascale Validation\n(per-domain avg.)'),
           ('hyperopt_splits-test', 'Megascale Test\n(per-domain avg.)'),
           ('DLG4_HUMAN_Faure_2021_abundance_domain', 'Faure DLG4 Abundance'),
           ('DLG4_HUMAN_Faure_2021_binding_domain', 'Faure DLG4 Binding'),
           ('GRB2_HUMAN_Faure_2021_abundance_domain', 'Faure GRB2 Abundance'),
           ('GRB2_HUMAN_Faure_2021_binding_domain', 'Faure GRB2 Binding'),
           ('MYO_HUMAN_Kung_2025_display', 'Küng Myoglobin Display'),
           ('ESTA_BACSU_Nutschel_2020_dTm', 'Nutschel EstA ΔT_50'),
           ('domainome', 'Beltran Domainome Abundance\n(per-domain avg.)')]
LABELS = dict(PANEL_A + PANEL_B)
LOAD = ['s669', 'ssym', 'q3421', 'k3822', 's571', 's783', 's8754', 's2648', 'PTMUL'] + [
    k for k, _ in PANEL_B]

# column order of the notebook's df_rho (fixes legend order and colours), replicate 1 only
MODEL_ORDER = ['SPURS', 'SPURS-multi', 'ProteinMPNN_additive', 'ProteinMPNN', 'ThermoMPNN_1', 'ThermoMPNN-D_1',
               'MutateEverything_additive_1', 'MutateEverything_1', 'ESM3-small-open_additive', 'ESM3-small-open',
               'ESM-MSR (σ=0.5)_additive_1', 'ESM-MSR (σ=0.5)_1', 'ESM-MSR_additive_1',
               'ESM-MSR (σ=1.25)_additive_1', 'ESM-MSR_1', 'ESM-MSR (σ=1.25)_1',
               'ESM3-small_additive', 'ESM3-medium_additive', 'ESM3-large_additive',
               'ESM3-small', 'ESM3-medium', 'ESM3-large',
               'Rosetta Cartesian DDG_additive_1', 'Rosetta Cartesian DDG_1']
COLORS = ['#d35400', '#34495e', '#3498db', '#2ecc71', '#888888', '#555555', '#333333', '#000000',
          '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#d35400']


DMS_KEYS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
            'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
            'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm']
# n_tests: Bonferroni N for the per-dataset tests of the family; n_tests_alt: N counting 3 sigma values for dms
FAMILIES = {
    'stability': dict(keys=[k for k, _ in PANEL_A], variants=[HIGHLIGHT], n_tests=10, n_tests_alt=10, carry=None,
                      title='External thermostability, σ=1 only: Bonferroni N=10'),
    'dms': dict(keys=DMS_KEYS, variants=[HIGHLIGHT, SIGMA_HALF], n_tests=12, n_tests_alt=18, carry=None,
                title='Single-protein DMS, σ∈{0.5, 1}: Bonferroni N=6×2=12'),
    'domainome': dict(keys=['domainome'], variants=[HIGHLIGHT, SIGMA_HALF], n_tests=2, n_tests_alt=2, carry=3,
                      title='Domainome (522 domains): N=2 (σ)'),
}
VARIANT_NAME = {HIGHLIGHT: 'σ=1', SIGMA_HALF: 'σ=0.5'}


def is_msr(col):
    return 'MSR' in col


# ----------------------------------------------------------------------------------------------
# weighted Spearman engine
# ----------------------------------------------------------------------------------------------
def _tie_struct(v, g):
    """
    Tie-group structure of values `v` within groups `g` (dense ints, one group per row set).
    Returns (tg, T, tg_group, first_tg): tie-group id per row (ascending by (group, value)), their
    number, the group of each tie group, and the first tie group of each group.
    """
    order = np.lexsort((v, g))
    vs, gs = v[order], g[order]
    new = np.ones(len(v), bool)
    new[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    tg_sorted = np.cumsum(new) - 1
    tg = np.empty(len(v), np.int64)
    tg[order] = tg_sorted
    tg_group = gs[new]
    first_tg = np.searchsorted(tg_group, np.arange(g.max() + 1))
    return tg, int(tg_sorted[-1]) + 1, tg_group, first_tg


def _ranks(struct, w):
    """Weighted average ranks (within group) of every row, for row weights w."""
    tg, T, tg_group, first_tg = struct
    tw = np.bincount(tg, weights=w, minlength=T)
    cum = np.cumsum(tw) - tw
    rank_tg = cum - cum[first_tg][tg_group] + (tw + 1) / 2
    return rank_tg[tg]


def _weighted_pearson_by_group(rx, ry, w, gl, n_groups):
    """Per-group weighted Pearson of two rank vectors; NaN for groups with <2 weight or no variance."""
    with np.errstate(invalid='ignore', divide='ignore'):
        sw = np.bincount(gl, weights=w, minlength=n_groups)
        mx = np.bincount(gl, weights=w * rx, minlength=n_groups) / sw
        my = np.bincount(gl, weights=w * ry, minlength=n_groups) / sw
        dx, dy = rx - mx[gl], ry - my[gl]
        sxy = np.bincount(gl, weights=w * dx * dy, minlength=n_groups)
        sxx = np.bincount(gl, weights=w * dx * dx, minlength=n_groups)
        syy = np.bincount(gl, weights=w * dy * dy, minlength=n_groups)
        rho = sxy / np.sqrt(sxx * syy)
    rho[(sw < 2) | ~(sxx > 1e-9) | ~(syy > 1e-9)] = np.nan
    return rho


class SpearmanBootstrap:
    """
    Spearman of several prediction columns against one ground truth on a shared set of rows,
    under integer row weights. grouped=True gives the mean of per-`code` Spearman (the notebook's
    custom_rho(grouped=True)); otherwise a single pooled Spearman.
    """

    def __init__(self, df, models, grouped):
        df = df.dropna(subset=['ground_truth'])
        df = df.iloc[np.argsort(df['code'].values, kind='stable')]
        self.codes, labels = pd.factorize(df['code'].astype(str), sort=True)
        self.code_labels = np.asarray(labels)
        self.n = len(df)
        self.n_proteins = int(self.codes.max()) + 1
        self.grouped = grouped
        self.group = self.codes if grouped else np.zeros(self.n, np.int64)
        self.n_groups = int(self.group.max()) + 1
        self.gsize = np.bincount(self.group, minlength=self.n_groups)
        self.gstart = np.cumsum(self.gsize) - self.gsize
        self.models = [m for m in models if df[m].notna().any()]
        y = df['ground_truth'].to_numpy(float)
        self.rowsets = {}
        self.model_set = {}
        for m in self.models:
            p = df[m].to_numpy(float)
            valid = ~np.isnan(p)
            key = hashlib.md5(np.packbits(valid).tobytes()).hexdigest()
            if key not in self.rowsets:
                rows = np.flatnonzero(valid)
                gmap, gl = np.unique(self.group[rows], return_inverse=True)
                self.rowsets[key] = dict(rows=rows, gmap=gmap, gl=gl, ys=_tie_struct(y[rows], gl), preds={})
            rs = self.rowsets[key]
            rs['preds'][m] = _tie_struct(p[rs['rows']], rs['gl'])
            self.model_set[m] = key

    def rho(self, w, models=None, mult=None):
        """{model: statistic} for row weights w (len n); mult = per-group multiplicity (grouped only)."""
        models = self.models if models is None else models
        out = {}
        for key, rs in self.rowsets.items():
            todo = [m for m in models if self.model_set.get(m) == key]
            if not todo:
                continue
            wr = w[rs['rows']].astype(float)
            rx = _ranks(rs['ys'], wr)
            ng = len(rs['gmap'])
            for m in todo:
                ry = _ranks(rs['preds'][m], wr)
                r = _weighted_pearson_by_group(rx, ry, wr, rs['gl'], ng)
                if self.grouped:
                    mg = np.ones(ng) if mult is None else mult[rs['gmap']].astype(float)
                    ok = np.isfinite(r) & (mg > 0)
                    out[m] = float((r[ok] * mg[ok]).sum() / mg[ok].sum()) if ok.any() else np.nan
                else:
                    out[m] = float(r[0])
        return out

    def observed(self, models=None):
        return self.rho(np.ones(self.n), models)

    def draw(self, rng, unit):
        """Row weights and group multiplicities for one bootstrap replicate."""
        if unit == 'mutation':
            u = rng.random(self.n)
            idx = self.gstart[self.group] + (u * self.gsize[self.group]).astype(np.int64)
            return np.bincount(idx, minlength=self.n), None
        k = rng.multinomial(self.n_proteins, np.full(self.n_proteins, 1.0 / self.n_proteins))
        return k[self.codes], (k if self.grouped else None)

    def bootstrap(self, models, unit, n_boot, rng):
        res = {m: np.empty(n_boot) for m in models}
        for b in range(n_boot):
            w, mult = self.draw(rng, unit)
            r = self.rho(w, models, mult)
            for m in models:
                res[m][b] = r[m]
        return res


def one_sided_p(diff):
    """P(diff <= 0) with the +1 correction, ignoring undefined replicates."""
    diff = diff[np.isfinite(diff)]
    return (1 + np.sum(diff <= 0)) / (1 + len(diff))


def two_sided_p(diff):
    diff = diff[np.isfinite(diff)]
    lo, hi = (1 + np.sum(diff <= 0)) / (1 + len(diff)), (1 + np.sum(diff >= 0)) / (1 + len(diff))
    return min(1.0, 2 * min(lo, hi))


def rank_competitors(obs):
    """Non-ESM-MSR, non-excluded columns with a defined rho, best first."""
    return sorted([m for m in obs if not is_msr(m) and m not in EXCLUDED and np.isfinite(obs[m])],
                  key=lambda m: -obs[m])


def run_group(dfs, fam, keys, unit, joint, n_boot, seed):
    """
    Observed rho for every model plus, for each dataset in `keys`, paired bootstrap draws of the tested
    ESM-MSR variants and the competitors. joint=True resamples one set of proteins (union of PDB ids)
    and applies it to every dataset, so datasets that share proteins stay correlated.
    """
    cfg = FAMILIES[fam]
    engines, rows, boot_models = {}, {}, {}
    for k in keys:
        df = dfs[k]
        models = [c for c in MODEL_ORDER if c in df.columns and c not in EXCLUDED]
        eng = SpearmanBootstrap(df, models, k in L.GROUPED)
        obs = eng.observed()
        comps = rank_competitors(obs)
        carry = comps if cfg['carry'] is None else comps[:cfg['carry']]
        engines[k], boot_models[k] = eng, cfg['variants'] + carry
        status = 'n/a: single protein' if (unit == 'protein' and eng.n_proteins < 2) else 'ok'
        rows[k] = dict(key=k, fam=fam, unit=unit, n_mut=eng.n, n_proteins=eng.n_proteins, obs=obs,
                       competitors=comps, carried=carry, status=status)
    run = [k for k in keys if rows[k]['status'] == 'ok']
    for k in run:
        rows[k]['boot'] = {m: np.empty(n_boot) for m in boot_models[k]}
    if joint:
        glob = np.unique(np.concatenate([engines[k].code_labels for k in run]))
        gidx = {k: np.searchsorted(glob, engines[k].code_labels)[engines[k].codes] for k in run}
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(f'{fam}|{unit}|{"+".join(keys)}'.encode())]))
    for b in range(n_boot):
        if joint:
            kg = rng.multinomial(len(glob), np.full(len(glob), 1.0 / len(glob)))
        for k in run:
            w, mult = (kg[gidx[k]], None) if joint else engines[k].draw(rng, unit)
            r = engines[k].rho(w, boot_models[k], mult)
            for m, v in r.items():
                rows[k]['boot'][m][b] = v
    return rows


def _task(args):
    return run_group(*args)


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------
def pretty(col):
    return col[:-2] if col.endswith('_1') else col


def pct(a):
    return tuple(np.nanpercentile(a, [2.5, 97.5]))


def summarize(rows, alpha=0.05):
    """Per dataset x tested sigma. rows: {key: row}."""
    recs = []
    for fam, cfg in FAMILIES.items():
        for k in cfg['keys']:
            r = rows[k]
            comp = r['competitors'][0]
            for v in cfg['variants']:
                rec = dict(family=fam, dataset=LABELS[k].replace('\n', ' '), key=k, esm_msr_variant=VARIANT_NAME[v],
                           n_mutations=r['n_mut'], n_proteins=r['n_proteins'], esm_msr_rho=r['obs'][v],
                           next_best_model=pretty(comp), next_best_rho=r['obs'][comp],
                           rank=1 + sum(r['obs'][c] > r['obs'][v] for c in r['competitors']),
                           bonferroni_N=cfg['n_tests'], status=r['status'])
                rec['delta_rho'] = rec['esm_msr_rho'] - rec['next_best_rho']
                if r['status'] == 'ok':
                    b = r['boot']
                    d = b[v] - b[comp]
                    rec.update(esm_msr_ci=pct(b[v]), next_best_ci=pct(b[comp]), delta_ci=pct(d),
                               p_one_sided=one_sided_p(d), p_two_sided=two_sided_p(d))
                    rec['p_bonferroni'] = min(1.0, rec['p_one_sided'] * cfg['n_tests'])
                    rec['p_bonferroni_alt_N'] = min(1.0, rec['p_one_sided'] * cfg['n_tests_alt'])
                    k3 = r['carried'][:3]
                    rec['p_vs_top3_bonferroni'] = min(1.0, max(one_sided_p(b[v] - b[c]) for c in k3) * cfg['n_tests'])
                    rec['significant'] = bool(rec['p_bonferroni'] < alpha and rec['delta_rho'] > 0)
                recs.append(rec)
    return pd.DataFrame(recs)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def overall(rows, alpha=0.05):
    """
    Family-level tests: macro-average (equal weight per dataset) of rho(ESM-MSR) - rho(comparator).
    vs 'best other' = each dataset's own next-best; vs each method present in every dataset of the family
    (Holm-adjusted across methods). Both are then multiplied by the number of sigma values tested.
    """
    recs = []
    for fam in ('stability', 'dms'):
        cfg = FAMILIES[fam]
        rws = [rows[k] for k in cfg['keys']]
        for v in cfg['variants']:
            base = dict(family=fam, esm_msr_variant=VARIANT_NAME[v], n_datasets=len(rws), sigma_N=len(cfg['variants']),
                        sigma_N_alt=3 if fam == 'dms' else 1)
            if any(r['status'] != 'ok' for r in rws):
                recs.append(dict(base, comparator='best other (per dataset)', status='n/a: single-protein datasets'))
                continue
            d_best = np.mean([r['boot'][v] - r['boot'][r['competitors'][0]] for r in rws], axis=0)
            obs_best = np.mean([r['obs'][v] - r['obs'][r['competitors'][0]] for r in rws])
            p = one_sided_p(d_best)
            wins = sum(r['obs'][v] > r['obs'][r['competitors'][0]] for r in rws)
            recs.append(dict(base, comparator='best other (per dataset)', status='ok', mean_delta_rho=obs_best,
                             delta_ci=pct(d_best), datasets_won=f'{wins}/{len(rws)}', p_one_sided=p,
                             p_adj=min(1.0, p * base['sigma_N']), p_adj_alt_N=min(1.0, p * base['sigma_N_alt'])))
            common = [c for c in rws[0]['carried'] if all(np.isfinite(r['obs'].get(c, np.nan)) and c in r['boot'] for r in rws)]
            ps = np.array([one_sided_p(np.mean([r['boot'][v] - r['boot'][c] for r in rws], axis=0)) for c in common])
            adj = holm(ps) if len(ps) else ps
            for c, pp, aa in zip(common, ps, adj):
                dc = np.mean([r['boot'][v] - r['boot'][c] for r in rws], axis=0)
                wins = sum(r['obs'][v] > r['obs'][c] for r in rws)
                recs.append(dict(base, comparator=pretty(c), status='ok',
                                 mean_delta_rho=np.mean([r['obs'][v] - r['obs'][c] for r in rws]), delta_ci=pct(dc),
                                 datasets_won=f'{wins}/{len(rws)}', p_one_sided=pp, p_holm=aa,
                                 p_adj=min(1.0, aa * base['sigma_N']), p_adj_alt_N=min(1.0, aa * base['sigma_N_alt'])))
    return pd.DataFrame(recs)


def fmt_ci(x, ci):
    return f'{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]' if isinstance(ci, tuple) else f'{x:.3f}'


def fmt_p(p):
    return '' if pd.isna(p) else (f'{p:.1e}' if p < 1e-3 else f'{p:.3f}')


def markdown_tables(tab, ov, unit, n_boot):
    out = [f'## {unit.capitalize()}-level bootstrap', '',
           f'B = {n_boot}; 95% percentile CIs; one-sided paired test ESM-MSR > next-best non-SPURS model; '
           f'Bonferroni within each family (N in the family header); * = adjusted p < 0.05.', '']
    for fam, cfg in FAMILIES.items():
        out += [f"### {cfg['title']}", '',
                '| Dataset | σ | n mut. | n prot. | ESM-MSR ρ [95% CI] | rank | Next-best | ρ [95% CI] | Δρ [95% CI] | p | p (Bonf.) | p (Bonf., N alt) | p vs top-3 (Bonf.) | sig. |',
                '|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|:--:|']
        for _, r in tab[tab.family == fam].iterrows():
            if r['status'] != 'ok':
                out.append(f"| {r['dataset']} | {r['esm_msr_variant']} | {r['n_mutations']} | {r['n_proteins']} | "
                           f"{r['esm_msr_rho']:.3f} | {r['rank']} | {r['next_best_model']} | {r['next_best_rho']:.3f} | "
                           f"{r['delta_rho']:+.3f} | n/a | n/a | n/a | n/a | {r['status']} |")
                continue
            out.append(
                f"| {r['dataset']} | {r['esm_msr_variant']} | {r['n_mutations']} | {r['n_proteins']} | "
                f"{fmt_ci(r['esm_msr_rho'], r['esm_msr_ci'])} | {r['rank']} | {r['next_best_model']} | "
                f"{fmt_ci(r['next_best_rho'], r['next_best_ci'])} | {r['delta_rho']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                f"{fmt_p(r['p_one_sided'])} | {fmt_p(r['p_bonferroni'])} | {fmt_p(r['p_bonferroni_alt_N'])} | "
                f"{fmt_p(r['p_vs_top3_bonferroni'])} | {'*' if r['significant'] else ''} |")
        out.append('')
    out += ['### Overall (macro-average of Δρ across datasets in the family)', '',
            'Per family and σ: one test vs each dataset\'s own next-best model, plus one per method present in every dataset '
            '(Holm across methods). Adjusted p additionally multiplies by the number of σ values tested '
            '(`p adj`; `p adj (σ=3)` counts three σ values for the DMS family).', '',
            '| Family | σ | Comparator | datasets won | mean Δρ [95% CI] | p | p adj | p adj (σ=3) |', '|---|---|---|---:|---|---:|---:|---:|']
    for _, r in ov.iterrows():
        if r['status'] != 'ok':
            out.append(f"| {r['family']} | {r['esm_msr_variant']} | {r['comparator']} | | {r['status']} | | | |")
        else:
            out.append(f"| {r['family']} | {r['esm_msr_variant']} | {r['comparator']} | {r['datasets_won']} | "
                       f"{r['mean_delta_rho']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                       f"{fmt_p(r['p_one_sided'])} | {fmt_p(r['p_adj'])} | {fmt_p(r['p_adj_alt_N'])} |")
    return '\n'.join(out) + '\n'


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------
def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''


def make_figure(rho_df, rows, tab, unit, n_boot, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import analysis_utils_msr as A

    fam_of = {k: f for f, c in FAMILIES.items() for k in c['keys']}
    tab = tab.set_index(['key', 'esm_msr_variant'])
    fig, axes = plt.subplots(2, 1, figsize=(18, 17))
    panels = [(PANEL_A, dict(ylim=(-0.05, 0.8), legend_loc='lower left')),
              (PANEL_B, dict(ylim=(0, 1), legend_loc='lower left'))]
    for ax, (panel, kw) in zip(axes, panels):
        keys = [k for k, _ in panel]
        data = rho_df.loc[[LABELS[k] for k in keys]]
        A.visualize_model_performance(data, title=None, ylabel="Spearman's Rho", highlighted_model='ESM-MSR',
                                      group_spread=0.07, colors=COLORS[1:], additive_offset=0.02, ax=ax, **kw)
        bases = set(map(A._base_name, data.columns))
        lines = {ln.get_label(): ln for ln in ax.lines if ln.get_label() in bases}
        ticklabels = []
        for i, k in enumerate(keys):
            fam = fam_of.get(k)
            if fam is None:
                ticklabels.append(f'{LABELS[k]}\nnot tested')
                continue
            r, cfg = rows[k], FAMILIES[fam]
            ptxt = []
            if r['status'] == 'ok':
                comp = r['competitors'][0]
                ln = lines[A._base_name(comp)]
                y = r['obs'][comp]
                lo, hi = np.nanpercentile(r['boot'][comp], [2.5, 97.5])
                ax.errorbar([ln.get_xdata()[i]], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt='none', capsize=6,
                            elinewidth=2.2, capthick=2.2, zorder=6, ecolor=ln.get_markeredgecolor())
            for j, v in enumerate(cfg['variants']):
                t = tab.loc[(k, VARIANT_NAME[v])]
                name = A._base_name(v)
                ln = lines[name]
                color = 'black' if v == HIGHLIGHT else ln.get_markeredgecolor()
                if t['status'] == 'ok':
                    y = r['obs'][v]
                    lo, hi = np.nanpercentile(r['boot'][v], [2.5, 97.5])
                    ax.errorbar([ln.get_xdata()[i]], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt='none',
                                capsize=6, elinewidth=2.2, capthick=2.2, zorder=7, ecolor=color)
                    s = stars(t['p_bonferroni']) if t['delta_rho'] > 0 else ''
                    if s:
                        ax.annotate(s, (ln.get_xdata()[i], 0.985 - 0.065 * j), xycoords=('data', 'axes fraction'),
                                    ha='center', va='top', fontsize=22, fontweight='bold', color=color)
                    ptxt.append(f"p_adj{'' if len(cfg['variants']) == 1 else ' ' + VARIANT_NAME[v]}={t['p_bonferroni']:.3g}")
            ticklabels.append(f"{LABELS[k]}\n" + ('p n/a (1 protein)' if r['status'] != 'ok' else '\n'.join(ptxt)))
        ax.set_xticklabels(ticklabels, rotation=30, ha='right', fontsize=11)
        # family labels / separators
        spans, start = [], 0
        for f in dict.fromkeys(fam_of.get(k, 'untested') for k in keys):
            n = sum(fam_of.get(k, 'untested') == f for k in keys)
            spans.append((f, start, start + n))
            start += n
        for f, a, b in spans:
            title = 'Megascale: not tested (in-distribution)' if f == 'untested' else FAMILIES[f]['title']
            ax.text((a + b - 1) / 2, 1.012, title, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                    fontsize=11, style='italic')
            if a > 0:
                ax.axvline(a - 0.5, color='#555555', lw=1.4, ls=':', zorder=2)
    axes[1].get_legend().remove()
    what = 'proteins' if unit == 'protein' else 'mutations'
    fig.suptitle(f'Spearman ρ on external benchmarks, one replicate per model; SPURS excluded (training overlap).\n'
                 f'Error bars: 95% bootstrap CI over {what} (B={n_boot}): ESM-MSR σ=1 (black), σ=0.5 (grey, where tested), next-best model.\n'
                 f'Stars: ESM-MSR better than next-best, one-sided paired bootstrap, Bonferroni within family '
                 f'(* p<0.05, ** p<0.01, *** p<0.001; σ=1 top row, σ=0.5 second row).' + (
                     '\nProtein-level bootstrap is undefined for single-protein datasets.' if unit == 'protein' else ''),
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path + '.png', dpi=200)
    fig.savefig(path + '.pdf')
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def load_results(args):
    cache = os.path.join(args.cache_dir, 'results.pkl')
    if os.path.exists(cache) and not args.reload:
        return pickle.load(open(cache, 'rb'))
    results = L.load_external(LOAD, root=args.pred_root)
    os.makedirs(args.cache_dir, exist_ok=True)
    pickle.dump(results, open(cache, 'wb'))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-root', default=L.PRED_ROOT, help='directory holding one predictions folder per dataset')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bootstrap_external_results'))
    ap.add_argument('--cache-dir', default=None, help='merged-prediction cache (default: <out>/_cache)')
    ap.add_argument('--reload', action='store_true', help='ignore the cache and re-read the prediction files')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--units', nargs='+', default=['protein', 'mutation'], choices=['protein', 'mutation'])
    args = ap.parse_args()
    args.cache_dir = args.cache_dir or os.path.join(args.out, '_cache')
    os.makedirs(args.out, exist_ok=True)
    warnings.filterwarnings('ignore')

    t0 = time.time()
    results = load_results(args)
    print(f'loaded in {time.time() - t0:.0f}s', flush=True)

    # tasks, biggest first: (dfs, family, keys, unit, joint, n_boot, seed)
    tasks = []
    for unit in args.units:
        for fam, cfg in FAMILIES.items():
            if fam == 'stability' and unit == 'protein':
                tasks.append(({k: results[k] for k in cfg['keys']}, fam, cfg['keys'], unit, True, args.n_boot, args.seed))
            else:
                tasks += [({k: results[k]}, fam, [k], unit, False, args.n_boot, args.seed) for k in cfg['keys']]
    tasks.sort(key=lambda t: -sum(len(d) for d in t[0].values()))
    t1 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        outs = list(ex.map(_task, tasks))
    print(f'bootstrap done in {time.time() - t1:.0f}s', flush=True)
    by_unit = {u: {} for u in args.units}
    for t, o in zip(tasks, outs):
        by_unit[t[3]].update(o)

    # observed rho (identical for both units), incl. the untested Megascale sets
    obs_rows = {}
    for k, _ in PANEL_A + PANEL_B:
        df = results[k]
        models = [c for c in MODEL_ORDER if c in df.columns and c not in EXCLUDED]
        obs_rows[k] = SpearmanBootstrap(df, models, k in L.GROUPED).observed()
    rho_df = pd.DataFrame({LABELS[k]: pd.Series(v) for k, v in obs_rows.items()}).T
    rho_df = rho_df[[c for c in MODEL_ORDER if c in rho_df.columns]]
    rho_df.to_csv(os.path.join(args.out, 'rho_all_models_replicate1.csv'))

    for unit, rows in by_unit.items():
        base = os.path.join(args.out, f'external_spearman_bootstrap_{unit}')
        tab, ov = summarize(rows), overall(rows)
        tab.to_csv(base + '_table.csv', index=False)
        ov.to_csv(base + '_overall.csv', index=False)
        with open(base + '_table.md', 'w') as f:
            f.write(markdown_tables(tab, ov, unit, args.n_boot))
        np.savez_compressed(os.path.join(args.cache_dir, f'boot_{unit}.npz'), **{
            f'{k}|{m}': v for k, r in rows.items() if r['status'] == 'ok' for m, v in r['boot'].items()})
        make_figure(rho_df, rows, tab, unit, args.n_boot, base)
        print(f'wrote {base}.png/.pdf and tables', flush=True)
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
