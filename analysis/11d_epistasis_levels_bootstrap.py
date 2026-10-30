#!/usr/bin/env python
"""The epistasis hierarchy of every model on the capped Tsuboyama doubles, recomputed with ``esm_msr.epi_metrics`` (the suite behind BENCHMARKS tables 2.0-2.2) and bootstrapped over
domains: naive rho, saturation-subtracted ("beyond") rho, pair rho, line rho and cell rank, for the test and the validation doubles.

Models and the predicted epistasis dddG' the suite reads (always the double prediction of one model against the additive reading of another or of another head):
  ESM-MSR devel, three seeds   epi  = MT(AB) - WT(AB)            the checkpoint's own link and per-library floors, as ``analysis_scripts/rescore_epistasis.py``
                               ctx  = ~WT(AB) - WT(AB)           the WT adapter read on the mutated sequence (the cluster's ctx run; ``reference/ctx_wt_rev_devel.csv.gz``)
  ME stage 2, ThermoMPNN-D, SPURS multi (three seeds each)       direct double - the same seed's sum of singles
  ESM3-small-open (sigma = 0 control)   independent-masking mutant pass - masked-marginal WT pass
  ProteinMPNN, GeoDDG-3D, Rosetta      direct double - sum of the model's singles
Rows are the measured doubles of ``cens == 0`` libraries with both singles measured (12,455 test, 8,494 validation), as the tables.  The point value of a multi-seed model is the
seed mean.  The interval resamples the *domains* (the libraries that have doubles: 13 in test, 9 in validation) with replacement, the same draw for every model and seed, recomputes
the suite on the resample (a repeated domain is a new domain: its pairs and residues are kept apart) and takes the seed mean of each draw: a 95% percentile interval.  The point
values reproduce the document's rows (checked below).

    python 11d_epistasis_levels_bootstrap.py [--B 300] [--ctx DIR]    # --ctx rebuilds reference/ctx_wt_rev_devel.csv.gz from outputs/ctx_run/predictions on the cluster
Writes ``results/11_publication/fig5_epistasis_levels_bootstrap_{test,val}.csv``.
"""
import argparse
import copy
import glob
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src'))
from esm_msr import epi_metrics as E  # noqa: E402
from esm_msr.link import numpy_link_from_state_dict  # noqa: E402

from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import add_esm3_masked, load_frame  # noqa: E402
from msr_analysis.registry import REFERENCE, ROOT  # noqa: E402

OUT = out_dir('11_publication')
DS = {'test': 'tsuboyama_capped_test', 'val': 'tsuboyama_capped_val'}
CKPT = {1: 'epoch=04-val_rho_combined_avg=0.840.ckpt', 2: 'epoch=07-val_rho_combined_avg=0.839.ckpt', 3: 'epoch=05-val_rho_combined_avg=0.841.ckpt'}
METRICS = {'naive_rho': None, 'beyond_rho': None, 'pair_rho': None, 'line_rho': None, 'cell_rank': None}      # head list filled per table
# model -> (frame key of the direct double, frame key of the additive reading) for the models without heads of their own
PLAIN = [('ME stage 2', 'me2', 'me1'), ('ThermoMPNN-D', 'tmpnnd', 'tmpnn'), ('SPURS multi', 'spursm', 'spurs'), ('ESM3-small-open', 'esm3ind', 'esm3mm'),
         ('ProteinMPNN', 'pmpnn', 'pmpnn+'), ('GeoDDG-3D', 'geoddg', 'geoddg+'), ('Rosetta', 'rosetta', 'rosetta+')]
DOC = {   # BENCHMARKS table 2.2, test: naive, beyond, pair, line, cell rank (the check)
    ('test', 'ME stage 2'): (0.405, 0.223, 0.124, 0.225, 0.112), ('test', 'ThermoMPNN-D'): (0.077, 0.071, -0.003, 0.197, 0.016),
    ('test', 'SPURS multi'): (0.308, 0.278, 0.399, 0.270, 0.041), ('test', 'ProteinMPNN'): (0.314, 0.273, 0.473, 0.125, 0.064),
    ('test', 'GeoDDG-3D'): (0.055, 0.046, 0.083, 0.087, 0.054), ('test', 'Rosetta'): (0.381, 0.207, 0.258, 0.195, 0.066),
    ('test', 'ESM-MSR devel · MT − WT'): (0.464, 0.487, 0.606, 0.382, 0.209)}
NAMES = ['naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_rank']
_MUT = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')
TABLES = {}          # (model, seed) -> Table, shared with the fork workers


def parse_mut(m, code):
    out = []
    for part in str(m).split(':'):
        g = _MUT.match(part.strip())
        if not g:
            return None
        out.append((code, int(g.group(2)), g.group(3)))
    return tuple(out)


def extract_ctx(ctx_dir):
    """The reverse leg of the cluster's ctx run (``mt_lora_pred`` of those files is ~WT) for the three devel seeds, as one small file."""
    rows = []
    for split in ('test', 'val'):
        for seed in (1, 2, 3):
            c = pd.read_csv(glob.glob(f'{ctx_dir}/{split}/s{seed}/*.csv')[0], usecols=['code', 'mut_type', 'mt_lora_pred'], low_memory=False)
            c['code'], c['mut_type'] = c['code'].astype(str), c['mut_type'].astype(str)
            rows.append(c.drop_duplicates(['code', 'mut_type']).rename(columns={'mt_lora_pred': 'wt_rev'}).assign(split=split, seed=seed))
    pd.concat(rows).to_csv(REFERENCE / 'ctx_wt_rev_devel.csv.gz', index=False, float_format='%.6g')


def link_of(seed):
    p = glob.glob(str(ROOT / f'dc_runs/logs/dc8_floorlink_canon_w2a4_m16a16_s{seed}/0/{CKPT[seed]}'))[0]
    sd = torch.load(p, map_location='cpu', weights_only=False)
    return numpy_link_from_state_dict(sd.get('state_dict', sd))


def make_table(sp, wt, mt, heads, wt_rev=None, link=None, y_floor=None):
    use = sp['use'].to_numpy()
    idx = np.flatnonzero(use)
    code, mut = sp['code'].astype(str).to_numpy()[idx], sp['mut'].to_numpy()[idx]
    keys = [parse_mut(m, c) for m, c in zip(mut, code)]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    items = dict(mut_key=keys, subset_type=st, ddG=sp['y'].to_numpy(float)[idx], dddG=sp['dddG_ML'].to_numpy(float)[idx], dG_wt=sp['dG_wt'].to_numpy(float)[idx],
                 wt=wt[idx], mt=mt[idx], comb=mt[idx])
    if wt_rev is not None:
        items['wt_rev'] = wt_rev[idx]
    arr = {k: (v if k in ('mut_key', 'subset_type') else np.asarray(v, float)) for k, v in items.items()}
    if y_floor is not None:
        arr['y_floor'] = np.array([float(y_floor.get(c, np.nan)) for c in code])
    return E.Table(arr, link, heads=heads)


def build(split, ctx_rev):
    fr = load_frame(DS[split])
    add_esm3_masked(fr, DS[split])
    sp, p = fr.spine, fr.preds
    dg = json.load(open(REFERENCE / 'dG_wt.json'))
    y_floor = json.load(open(REFERENCE / 'y_floor.json'))
    TABLES.clear()
    models = []
    for seed in (1, 2, 3):                                  # ESM-MSR devel: two heads of one table per seed
        c = ctx_rev[(ctx_rev.split == split) & (ctx_rev.seed == seed)]
        rev = pd.DataFrame({'code': sp['code'].astype(str), 'mut_type': sp['mut']}).merge(c[['code', 'mut_type', 'wt_rev']], on=['code', 'mut_type'], how='left')['wt_rev'].to_numpy(float)
        TABLES[('devel', seed)] = make_table(sp, p['msrdev:wt'][seed - 1], p['msrdev:mt'][seed - 1], ('epi', 'ctx', 'add'), wt_rev=rev, link=link_of(seed), y_floor=y_floor)
    models += [('ESM-MSR devel · MT − WT', 'devel', 'epi'), ('ESM-MSR devel · ~WT − WT', 'devel', 'ctx')]
    for name, direct, add in PLAIN:
        if direct not in p or add not in p:
            print('skip', name)
            continue
        ns = p[direct].shape[0]
        for s in range(ns):
            TABLES[(name, s + 1)] = make_table(sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], ('epi', 'add'))
        models.append((name, name, 'epi'))
    return models


def metrics_of(t, head):
    r = E.compute(t, {m: (head,) for m in NAMES})
    return [r.get(f'epi_{m}_{head}', np.nan) for m in NAMES]


def resampled(t, idx, rep):
    r = copy.copy(t)
    n0 = t.n
    for name, v in vars(t).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    r.pair = t.pair[idx] + rep * (t.pair.max() + 1)
    r.row = t.row[idx] + rep * (max(t.row.max(), t.col.max()) + 1)
    r.col = t.col[idx] + rep * (max(t.row.max(), t.col.max()) + 1)
    r.lib = t.lib[idx] + rep * (t.lib.max() + 1)
    r._mats = None
    return r


def draw(args):
    """One bootstrap replicate: the seed-mean metrics of every model on the same domain resample."""
    counts, models = args
    t0 = TABLES[('devel', 1)]
    libs = np.unique(t0.lib)
    by = {l: np.flatnonzero(t0.lib == l) for l in libs}
    idx_l, rep_l, k = [], [], 0
    for l, c in zip(libs, counts):
        for _ in range(int(c)):
            idx_l.append(by[l]); rep_l.append(np.full(len(by[l]), k)); k += 1
    idx, rep = np.concatenate(idx_l), np.concatenate(rep_l)
    out = []
    cache = {}
    for name, key, head in models:
        vals = []
        for (k_, s), t in TABLES.items():
            if k_ != key:
                continue
            ck = (k_, s)
            if ck not in cache:
                cache[ck] = resampled(t, idx, rep)
            vals.append(metrics_of(cache[ck], head))
        out.append(np.nanmean(np.array(vals, float), axis=0))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=300)
    ap.add_argument('--ctx', default=None)
    ap.add_argument('--procs', type=int, default=7)
    a = ap.parse_args()
    if a.ctx:
        extract_ctx(a.ctx)
    ctx_rev = pd.read_csv(REFERENCE / 'ctx_wt_rev_devel.csv.gz', dtype={'code': str, 'mut_type': str})
    for split in ('test', 'val'):
        models = build(split, ctx_rev)
        t0 = TABLES[('devel', 1)]
        assert all(np.array_equal(t.lib, t0.lib) for t in TABLES.values()), 'tables disagree on the doubles'
        libs = np.unique(t0.lib)
        point = []
        for name, key, head in models:
            v = np.array([metrics_of(t, head) for (k_, s), t in TABLES.items() if k_ == key], float)
            point.append(np.nanmean(v, axis=0))
            if (split, name) in DOC:
                print(f'  {split} {name:26s} suite {np.round(point[-1], 3)}  document {DOC[(split, name)]}')
        point = np.array(point)
        rng = np.random.default_rng(11)
        counts = rng.multinomial(len(libs), np.full(len(libs), 1.0 / len(libs)), size=a.B)
        with mp.get_context('fork').Pool(a.procs) as pool:
            reps = np.array(pool.map(draw, [(c, models) for c in counts], chunksize=4))          # B x models x metrics
        rows = []
        for i, (name, key, head) in enumerate(models):
            for j, m in enumerate(NAMES):
                lo, hi = np.nanpercentile(reps[:, i, j], [2.5, 97.5])
                rows.append(dict(split=split, model=name, metric=m, value=point[i, j], lo=lo, hi=hi, n_domains=len(libs), n_doubles=t0.n, B=a.B))
        pd.DataFrame(rows).to_csv(OUT / f'fig5_epistasis_levels_bootstrap_{split}.csv', index=False, float_format='%.5f')
        print(split, 'done:', len(libs), 'domains,', t0.n, 'doubles')


if __name__ == '__main__':
    main()
