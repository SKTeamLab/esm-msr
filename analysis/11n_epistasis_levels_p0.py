#!/usr/bin/env python
"""Supplementary epistasis-levels figure (figure 5a of ``11b_publication_epistasis_levels``) with ESM-MSR drawn as p0 and its own ctx control only, and SPURS as the retrain with the trained adapter.

Same suite, levels, bootstrap (domains of the Tsuboyama capped test doubles, seeds resampled, ``--B`` draws) and models as ``11d_epistasis_levels_bootstrap.py`` / ``11b``, except:

  ESM-MSR p0 (FINAL_p0 seeds 1-3)   epistasis = 1/2(MT_AB + WT_AB) - WT_A - WT_B = comb(AB) - WT(A) - WT(B)
  ESM-MSR p0 ctx ctrl                1/2(~WT_AB + WT_AB) - WT_A - WT_B: the p0 WT adapter read on the mutated sequence (``final_ctx/predictions`` from the testing queue's ``ctx`` tier,
                                     ``inference_scripts/esm_msr_ctx_testing.py``); left out, with a note, until those runs exist
  SPURS multi (trained adapter)      the retrain with the trained adapter (``retrained/spurs_unfrozen``), multi minus single, three seeds

The ESM-MSR baseline / devel / linked rows and the frozen-adapter SPURS are not drawn.  Output ``results/00_supplementary_figs/fig5a_epistasis_levels_p0_test.{png,pdf}`` and
``fig5_epistasis_levels_p0_bootstrap_test.csv``.

    python analysis/11n_epistasis_levels_p0.py [--B 100] [--procs 7]
"""
import argparse
import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msr_analysis.common import out_dir
from msr_analysis.data import _align_id, load_frame
from msr_analysis.final_runs import load_final_frame
from msr_analysis.registry import BASE, ROOT

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m                              # the worker pool pickles d11.draw by module name
    spec.loader.exec_module(m)
    return m


d11 = load('d11', '11d_epistasis_levels_bootstrap.py')
b11 = load('b11', '11b_publication_epistasis_levels.py')
OUT = out_dir('00_supplementary_figs')
UNF = BASE / 'retrained' / 'spurs_unfrozen' / 'predictions'
CTX = ROOT / 'final_ctx' / 'predictions'
SPLIT = 'test'
P0_EPI, P0_CTX = 'ESM-MSR p0 · 1/2(MT_AB+WT_AB) − WT_A − WT_B', 'ESM-MSR p0 · 1/2(~WT_AB+WT_AB) − WT_A − WT_B'
P1_EPI, P1M_EPI = 'ESM-MSR p1 (independent masking) · 1/2(MT_AB+WT_AB) − WT_A − WT_B', 'ESM-MSR p1 (masked marginal MT) · 1/2(MT_AB+WT_AB) − WT_A − WT_B'


def ctx_rev(sp, seed):
    """~WT of FINAL_p0 seed ``seed`` for every row of the spine (NaN when the ctx run has not produced the file)."""
    f = CTX / f'{d11.DS[SPLIT]}'.replace('tsuboyama_capped_test', 'splits_oct06_capped-test') / 'final_checkpoints' / 'FINAL_p0' / f's{seed}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv'
    if not f.exists():
        return None
    c = pd.read_csv(f, usecols=['code', 'mut_type', 'mt_lora_pred'], low_memory=False)
    c['code'], c['mut_type'] = c['code'].astype(str), c['mut_type'].astype(str)
    c = c.drop_duplicates(['code', 'mut_type'])
    return pd.DataFrame({'code': sp['code'].astype(str), 'mut_type': sp['mut']}).merge(c, on=['code', 'mut_type'], how='left')['mt_lora_pred'].to_numpy(float)


def build():
    fr = load_frame(d11.DS[SPLIT])
    sp, p = fr.spine, fr.preds
    ids = sp['id'].to_numpy(dtype=object)
    f = load_final_frame(d11.DS[SPLIT], families=['FINAL_p0'])
    assert (f.spine['id'].to_numpy() == sp['id'].to_numpy()).all()
    S, M = [], []
    for sd in (1, 2, 3):
        d = pd.read_csv(UNF / f'seed{sd}' / 'tsuboyama_test.csv', usecols=['uid', 'SPURS_single_additive', 'SPURS_multi'], low_memory=False)
        S.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_single_additive'].to_numpy()))
        M.append(_align_id(ids, d['uid'].astype(str).to_numpy(), d['SPURS_multi'].to_numpy()))
    p['spursun'], p['spursunm'] = np.vstack(S), np.vstack(M)
    d11.TABLES.clear()
    have_ctx = True
    for seed in (1, 2, 3):
        rev = ctx_rev(sp, seed)
        have_ctx &= rev is not None
        heads = ('epi', 'add') if rev is None else ('epi', 'ctx', 'add')
        d11.TABLES[('p0', seed)] = d11.make_table(sp, f.preds['FINAL_p0:wt'][seed - 1], f.preds['FINAL_p0:comb'][seed - 1], heads, wt_rev=rev, link=None)
    models = [(P0_EPI, 'p0', 'epi')] + ([(P0_CTX, 'p0', 'ctx')] if have_ctx else [])
    print('p0 ctx control:', 'included' if have_ctx else 'NOT available yet (final_ctx not complete): drawn without it')
    plain = [('ME stage 2', 'me2', 'me1'), ('ThermoMPNN-D', 'tmpnnd', 'tmpnn'), ('SPURS multi', 'spursunm', 'spursun'), ('ESM3-small-open', 'msrrel@0:comb', 'msrrel@0:wt'),
             ('ProteinMPNN', 'pmpnn', 'pmpnn+'), ('GeoDDG-3D', 'geoddg', 'geoddg+'), ('Rosetta', 'rosetta', 'rosetta+')]
    for name, direct, add in plain:
        ns = p[direct].shape[0]
        for s in range(ns):
            d11.TABLES[(name, s + 1)] = d11.make_table(sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], ('epi', 'add'))
        models.append((name, name, 'epi'))
    c_exact = np.array([float(__import__('json').load(open(d11.REFERENCE / 'dG_wt.json')).get(k, np.nan)) for k in sp['code'].astype(str)])
    wt_true = np.where((sp['n_mut'] == 1).to_numpy(), sp['y'].to_numpy(float), sp['ddG_additive_ML'].to_numpy(float))
    d11.TABLES[('neg_dg_add', 1)] = d11.make_table(sp, wt_true, -c_exact, ('epi', 'add'), link=None)
    models.append(('−dG_add', 'neg_dg_add', 'epi'))
    return models


def build_masking():
    """ESM-MSR p0 (unmasked), its ctx control, and the masking ablations: FINAL_p1 (independent single-site masking) and FINAL_p1_marg (independent-masked WT pass, mutant pass masked marginal), three seeds each."""
    fr = load_frame(d11.DS[SPLIT])
    sp = fr.spine
    f = load_final_frame(d11.DS[SPLIT], families=['FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg'])
    assert (f.spine['id'].to_numpy() == sp['id'].to_numpy()).all()
    d11.TABLES.clear()
    have_ctx = True
    for seed in (1, 2, 3):
        rev = ctx_rev(sp, seed)
        have_ctx &= rev is not None
        d11.TABLES[('p0', seed)] = d11.make_table(sp, f.preds['FINAL_p0:wt'][seed - 1], f.preds['FINAL_p0:comb'][seed - 1], ('epi', 'add') if rev is None else ('epi', 'ctx', 'add'), wt_rev=rev, link=None)
        d11.TABLES[('p1', seed)] = d11.make_table(sp, f.preds['FINAL_p1:wt'][seed - 1], f.preds['FINAL_p1:comb'][seed - 1], ('epi', 'add'), link=None)
        d11.TABLES[('p1m', seed)] = d11.make_table(sp, f.preds['FINAL_p1_marg:wt'][seed - 1], f.preds['FINAL_p1_marg:comb'][seed - 1], ('epi', 'add'), link=None)
    models = [(P0_EPI, 'p0', 'epi')] + ([(P0_CTX, 'p0', 'ctx')] if have_ctx else []) + [(P1_EPI, 'p1', 'epi'), (P1M_EPI, 'p1m', 'epi')]
    c_exact = np.array([float(__import__('json').load(open(d11.REFERENCE / 'dG_wt.json')).get(k, np.nan)) for k in sp['code'].astype(str)])
    wt_true = np.where((sp['n_mut'] == 1).to_numpy(), sp['y'].to_numpy(float), sp['ddG_additive_ML'].to_numpy(float))
    d11.TABLES[('neg_dg_add', 1)] = d11.make_table(sp, wt_true, -c_exact, ('epi', 'add'), link=None)
    models.append(('−dG_add', 'neg_dg_add', 'epi'))
    return models


STEMS = {'p0': 'p0', 'masking': 'masking'}


def bootstrap(B, procs, variant='p0'):
    d11.REF = 'p0'
    models = build() if variant == 'p0' else build_masking()
    t0 = d11.TABLES[('p0', 1)]
    assert all(np.array_equal(t.lib, t0.lib) for t in d11.TABLES.values()), 'tables disagree on the doubles'
    libs = np.unique(t0.lib)
    point = np.array([np.nanmean(np.array([d11.metrics_of(t, head) for (k, s), t in d11.TABLES.items() if k == key], float), axis=0) for name, key, head in models])
    rng = np.random.default_rng(11)
    counts = rng.multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=B)
    seed_ws = rng.multinomial(3, np.full(3, 1.0 / 3.0), size=B).astype(float) / 3.0
    with mp.get_context('fork').Pool(procs) as pool:
        reps = np.array(pool.map(d11.draw, [(c, sw, models) for c, sw in zip(counts, seed_ws)], chunksize=4))
    rows = []
    for i, (name, key, head) in enumerate(models):
        for j, m in enumerate(d11.NAMES):
            lo, hi = np.nanpercentile(reps[:, i, j], [2.5, 97.5])
            rows.append(dict(split=SPLIT, model=name, metric=m, value=point[i, j], lo=lo, hi=hi, n_domains=len(libs), n_doubles=t0.n, B=B))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f'fig5_epistasis_levels_{STEMS[variant]}_bootstrap_test.csv', index=False, float_format='%.5f')
    np.savez(OUT / f'fig5_epistasis_levels_{STEMS[variant]}_reps_test.npz', reps=reps, models=np.array([m[0] for m in models]), metrics=np.array(d11.NAMES), B=B)         # every draw, for the paired tests
    return T


def paired_tests(metrics=('cell_rank', 'cell_mag_pooled', 'line_rho', 'pair_rho', 'beyond_rho', 'naive_rho'), variant='p0'):
    """One-sided paired domain-bootstrap test of ESM-MSR p0 against every other model, level by level: the draws of ``rho(p0) - rho(other)`` (the same domain and seed resamples for all models);
    p = max(P(delta <= 0), 1 / (B + 1)) is the evidence that p0 is better; Holm over the comparisons of one level.  Needs ``reps`` from ``bootstrap``."""
    z = np.load(OUT / f'fig5_epistasis_levels_{STEMS[variant]}_reps_test.npz')
    reps, models, names, B = z['reps'], list(z['models']), list(z['metrics']), int(z['B'])
    i0 = models.index(P0_EPI)
    rows = []
    for m in metrics:
        j = names.index(m)
        grp = []
        for i, name in enumerate(models):
            if i == i0 or name == '−dG_add':
                continue
            d = reps[:, i0, j] - reps[:, i, j]
            d = d[np.isfinite(d)]
            grp.append(dict(metric=m, other=name, delta=float(np.nanmean(reps[:, i0, j]) - np.nanmean(reps[:, i, j])), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                            p_one=float(max((d <= 0).mean(), 1.0 / (len(d) + 1))), p_two=float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / (len(d) + 1))), B=len(d)))
        order = np.argsort([g['p_one'] for g in grp])
        adj = np.maximum.accumulate(np.minimum(1.0, (len(grp) - np.arange(len(grp))) * np.array([grp[k]['p_one'] for k in order])))
        for rank, k in enumerate(order):
            grp[k]['p_one_holm'] = float(adj[rank])
        rows += grp
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f'fig5_epistasis_levels_{STEMS[variant]}_paired_test.csv', index=False, float_format='%.5f')
    return res


def draw(T):
    models = [(P0_EPI, 'retrained', '#000000', 'ESM-MSR p0'), (P0_CTX, 'retrained', '#b2babb', 'ESM-MSR p0 ctx ctrl'),
              ('ME stage 2', 'retrained', '#1b8a4b', 'ME stage 2'), ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'), ('SPURS multi', 'retrained', '#d35400', 'SPURS multi'),
              ('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open'), ('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN'),
              ('GeoDDG-3D', 'supervised', '#117a65', 'GeoDDG-3D'), ('Rosetta', 'supervised', '#1abc9c', 'Rosetta')]
    b11.OUT = OUT
    foot = [f.replace('ESM-MSR = 1/2', 'ESM-MSR p0 = 1/2').replace('ESM-MSR ctx ctrl = ', 'ESM-MSR p0 ctx ctrl = ').replace('SPURS multi,', 'SPURS multi (trained adapter),') for f in b11.FOOT]
    b11.draw_version(T, SPLIT, levels=b11.LEVELS, foot=foot, suffix='_p0', models=models)
    print('wrote', OUT / 'fig5a_epistasis_levels_bars_test_p0.png')


def draw_masking(T):
    """Supplementary: ESM-MSR p0 against its independent-masked (FINAL_p1) and masked-marginal (FINAL_p1_marg) ablations, level by level."""
    models = [(P0_EPI, 'retrained', '#000000', 'ESM-MSR p0\n(unmasked)'), (P0_CTX, 'retrained', '#b2babb', 'ESM-MSR p0\nctx ctrl'),
              (P1_EPI, 'retrained', '#6c3483', 'ESM-MSR p1\n(independent masking)'), (P1M_EPI, 'retrained', '#b07cc6', 'ESM-MSR p1\n(masked marginal)')]
    b11.OUT = OUT
    foot = [f.replace('ESM-MSR = 1/2', 'ESM-MSR p0 = 1/2').replace('ESM-MSR ctx ctrl = ', 'ESM-MSR p0 ctx ctrl = ') for f in b11.FOOT[:1]]
    foot = [foot[0].split(';')[0] + '.',
            'ESM-MSR p0: unmasked WT and MT passes; p1: trained and tested with independent single-site masking (block_mask_wt/mt = 1); p1 masked marginal: independent-masked WT pass, mutant pass with all mutated positions masked at once (block_marginal_mt).',
            'Each model is read under the masking it was trained with; all three seeds (FINAL_p0, FINAL_p1, FINAL_p1_marg s1-s3). Intervals: 95% bootstrap over domains.'] + b11.FOOT[2:]
    b11.draw_version(T, SPLIT, levels=b11.LEVELS, foot=foot, suffix='_masking', models=models, title_desc='ESM-MSR masking ablations, epistasis level by level')
    print('wrote', OUT / 'fig5a_epistasis_levels_bars_test_masking.png')


def draw_main(T):
    """The main-text version, panel e: black line per model = the same level on the doubles with additive dG > 0 (no interval), no title or footnote."""
    models = [(P0_EPI, 'retrained', '#000000', 'ESM-MSR p0'), (P0_CTX, 'retrained', '#b2babb', 'ESM-MSR p0 ctx ctrl'),
              ('ME stage 2', 'retrained', '#1b8a4b', 'ME stage 2'), ('ThermoMPNN-D', 'retrained', '#1f5f8b', 'ThermoMPNN-D'), ('SPURS multi', 'retrained', '#d35400', 'SPURS multi'),
              ('ESM3-small-open', 'zeroshot', '#7f8c8d', 'ESM3-small-open'), ('ProteinMPNN', 'zeroshot', '#34495e', 'ProteinMPNN'),
              ('GeoDDG-3D', 'supervised', '#117a65', 'GeoDDG-3D'), ('Rosetta', 'supervised', '#1abc9c', 'Rosetta')]
    b11.OUT = out_dir('00_main_text_figs')
    b11.draw_version(T, SPLIT, levels=b11.LEVELS, suffix='', models=models, overlay_sub0=True, label='e', stem='epistasis_levels_e')
    print('wrote', b11.OUT / 'epistasis_levels_e.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=7)
    ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--paired-only', action='store_true', help='only the paired tests from the saved draws')
    ap.add_argument('--variant', choices=list(STEMS), default='p0', help='p0: the epistasis-levels figure of p0 and the other models; masking: p0 against its masking ablations')
    a = ap.parse_args()
    if a.variant == 'masking':
        T = pd.read_csv(OUT / 'fig5_epistasis_levels_masking_bootstrap_test.csv') if (a.figures_only or a.paired_only) else bootstrap(a.B, a.procs, 'masking')
        if not a.paired_only:
            draw_masking(T)
        with pd.option_context('display.width', 220, 'display.max_columns', 20):
            print(paired_tests(variant='masking').round(4).to_string(index=False))
        return
    if a.paired_only:
        with pd.option_context('display.width', 220, 'display.max_columns', 20):
            print(paired_tests().round(4).to_string(index=False))
        return
    T = pd.read_csv(OUT / 'fig5_epistasis_levels_p0_bootstrap_test.csv') if a.figures_only else bootstrap(a.B, a.procs)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(T[T.metric.isin(['naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank'])].pivot(index='model', columns='metric', values='value').round(3).to_string())
    draw(T)
    draw_main(T)
    if not a.figures_only:
        with pd.option_context('display.width', 220, 'display.max_columns', 20):
            print(paired_tests().round(4).to_string(index=False))


if __name__ == '__main__':
    main()
