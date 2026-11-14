#!/usr/bin/env python
"""Ablation table of the FINAL_* families: every family x seed x readout on every benchmark, and candidate table layouts.

Inputs (no GPU, no checkpoints): the testing queue's prediction files, located by ``msr_analysis.final_runs``
(``MSR_ROOT`` or ``MSR_FINAL_BENCH`` / ``MSR_FINAL_PGYM_WT`` / ``MSR_FINAL_PGYM_COMB``), the label spines in
``baselines/public/_tasks``, ProteinGym categories in ``baselines/pgym_per_dms/pgym_all_models_per_dms.csv`` and the
committed homology classes in ``reference/leak_audit_structure_homology.csv``.

Scoring follows the rest of ``analysis/`` (01_core_rho, 05_proteingym, 06_epistasis): measured rows only (cens == 0);
pooled Spearman per external / DMS set; mean of per-library rho on Tsuboyama (per wild-type code) and per-domain rho on
Domainome; ProteinGym = mean over assays per category, PG avg = mean of the 5 category means; "low-id" drops the
MegaScale-homologous (P/H) Domainome domains and ProteinGym Stability assays.  Epistasis = comb(AB) - WT(A) - WT(B)
against measured dddG over the doubles with both singles measured (``common.epi_rows``).

Readouts (``head``): ``wt`` = WT head (on a multi-mutant row: the WT head's own prediction of that row), ``comb`` =
1/2(MT + WT), ``mt`` = MT head, ``wt+`` = sum of the WT head's single-mutant predictions (additive), ``primary`` = wt on a
singles-only column and comb on a column with multi-mutants (the deployed readout).

Outputs in ``results/16_ablation/``:
  cells_long.csv     one row per family x seed x head x column (point rho, n rows, n proteins)
  summary.csv        seed mean / SD / n_seeds per family x head x column
  delta_s1.csv       paired bootstrap of (family - FINAL_p0) at seed 1 (the only seed FINAL_p0 has so far) on the
                     headline columns, B draws (protein level; site level for the single-protein DMS; assays for PG)
  table_main_delta.{md,csv,tex}   delta vs FINAL_p0 under the deployed readout, seed + sampling noise
  table_wt_delta.{md,csv,tex}     the same under the WT head alone (WT-relevant ablations only, no epistasis columns)
  layout_*.md        candidate tables (see ``LAYOUTS``)
  informativeness.csv / head_choice.csv   per column: family spread vs seed noise; comb - wt and whether the heads agree
  coverage.csv       which family / seed / dataset predictions were found

Usage:  python 16_final_ablation_table.py [--B 200] [--sigma 1.0] [--skip-boot]
"""
import argparse
import copy
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from esm_msr import epi_metrics as E  # noqa: E402
from msr_analysis.boot import DatasetBoot, bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import epi_rows, out_dir, subframe  # noqa: E402
from msr_analysis.data import Frame  # noqa: E402
from msr_analysis.final_runs import FAMILIES, REFERENCE as REF, SEEDS, load_final_frame, pgym_per_dms  # noqa: E402
from msr_analysis.registry import DMS7, EXT12, LABEL, PER_PROTEIN, PGYM, REFERENCE  # noqa: E402

OUT = out_dir('16_ablation')
EXT_SINGLE = [d for d in EXT12 if d not in ('ptmul', 'ptmuld')]
EXT_MULTI = ['ptmul', 'ptmuld']
DMS_MULTI = [d for d in DMS7 if d != 'ESTA_BACSU_Nutschel_2020_dTm']
PG_CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']
PG_DROP = {'BRCA1_HUMAN_Findlay_2018', 'BRCA2_HUMAN_Erwood_2022_HEK293T', 'POLG_CXB3N_Mattenberger_2021', 'SCN5A_HUMAN_Glazer_2019'}
HEADS = ['wt', 'comb', 'mt', 'wt+']


def homologous(dataset):
    sh = pd.read_csv(REFERENCE / 'leak_audit_structure_homology.csv')
    sh = sh[sh['dataset'] == dataset]
    return set(sh.loc[sh['class_with_struct'].isin(['P', 'H']), 'item_id'].astype(str))


# ----------------------------------------------------------------------------------------------
# columns: (column id, frame builder) -- a column is one rho per family x seed x head
# ----------------------------------------------------------------------------------------------
def columns_for(ds, fr):
    """[(column, subframe, per_protein)] for one dataset frame."""
    sp = fr.spine
    pp = ds in PER_PROTEIN
    nm = sp['n_mut'].to_numpy()
    use = sp['use'].to_numpy()
    cols = [(ds, fr, pp)]
    if (nm[use] > 1).any() and (nm[use] == 1).any():
        cols.append((f'{ds}|singles', subframe(fr, use & (nm == 1)), pp))
        cols.append((f'{ds}|multi', subframe(fr, use & (nm > 1)), pp))
    if 'dddG_ML' in sp and ds.startswith('tsuboyama'):
        m = epi_rows(fr) & sp['wt_backbone'].to_numpy()
        # dddG rho of the predicted epistasis (pooled, "naive"); keys renamed so 'epi' is scored as a readout
        efr = subframe(fr, m, y=sp['dddG_ML'].to_numpy(dtype=float))
        cols.append((f'{ds}|dddG', efr, False))
    if ds == 'domainome':
        hom = homologous('domainome')
        cols.append((f'{ds}|lowid', subframe(fr, use & ~sp['group'].astype(str).isin(hom).to_numpy()), True))
    return cols


def score_columns(frames):
    """Point rho for every family x seed x head on every column."""
    rows = []
    for ds, fr in frames.items():
        for col, f, pp in columns_for(ds, fr):
            keys = list(f.preds)
            b = DatasetBoot(f, keys, per_protein=pp)
            for k in b.keys:
                fam, head = k.rsplit(':', 1)
                if col.endswith('|dddG') != (head == 'epi'):
                    continue                  # epi is scored against dddG only, the ddG readouts against ddG only
                _, per_seed, n, ng = b.point(k)
                for s, r in zip(fr.notes['seeds'][fam], per_seed):
                    rows.append(dict(family=fam, seed=s, head=('epi' if head == 'epi' else head), column=col, dataset=ds,
                                     rho=float(r), n=n, n_proteins=ng))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# ProteinGym
# ----------------------------------------------------------------------------------------------
def pg_table():
    a = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
    a = a[~a.index.isin(PG_DROP)]
    cat = a['category']
    stab_hom = homologous('proteingym') & set(cat.index[cat == 'Stability'])
    per = {}
    for fam in FAMILIES:
        for s in SEEDS:
            for mode in ('wt', 'comb'):
                r = pgym_per_dms(fam, s, mode, dms_ids=list(cat.index))
                if r is None or r.notna().sum() < len(cat) - 1:    # incomplete run: leave the cell out
                    continue
                per[(fam, s, mode)] = r
    return cat, stab_hom, per


def pg_summaries(cat, stab_hom, per):
    rows = []
    for (fam, s, mode), r in per.items():
        means = {c: float(np.nanmean(r[cat == c])) for c in PG_CATS}
        lo = r.drop(index=list(stab_hom))
        cl = cat.drop(index=list(stab_hom))
        means_lo = {c: float(np.nanmean(lo[cl == c])) for c in PG_CATS}
        vals = {f'pg|{c}': v for c, v in means.items()}
        vals['pg|avg'] = float(np.mean(list(means.values())))
        vals['pg|Stability_lowid'] = means_lo['Stability']
        vals['pg|avg_lowid'] = float(np.mean(list(means_lo.values())))
        for col, v in vals.items():
            rows.append(dict(family=fam, seed=s, head=mode, column=col, dataset='proteingym', rho=v, n=int(r.notna().sum()),
                             n_proteins=np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# primary readout + composites
# ----------------------------------------------------------------------------------------------
MULTI_COLS = set(EXT_MULTI) | set(DMS_MULTI) | {'tsuboyama_capped_val', 'tsuboyama_capped_test'}


def is_multi_col(c):
    """True for a column whose scored rows include multi-mutants (the combined readout is the deployed one)."""
    return c in MULTI_COLS or c.endswith('|multi')


def add_primary(cells):
    """primary = wt on singles-only columns, comb where multi-mutants are scored (incl. |multi; |dddG is epi)."""
    is_multi = is_multi_col
    p = cells[((cells['head'] == 'comb') & cells['column'].map(is_multi))
              | ((cells['head'] == 'wt') & ~cells['column'].map(is_multi) & ~cells['column'].str.endswith('|dddG'))
              | (cells['head'] == 'epi')].copy()
    p['head'] = 'primary'
    return pd.concat([cells, p], ignore_index=True)


COMPOSITES = {
    'ext_singles_mean': EXT_SINGLE,
    'ext_multi_mean': EXT_MULTI,
    'dms_mean': DMS7,
    'dms_multi_mean': [f'{d}|multi' for d in DMS_MULTI],
}


def add_composites(cells):
    out = []
    for name, cols in COMPOSITES.items():
        sub = cells[cells['column'].isin(cols)]
        g = sub.groupby(['family', 'seed', 'head'])
        agg = g['rho'].agg(['mean', 'count']).reset_index()
        agg = agg[agg['count'] == len(cols)]
        for r in agg.itertuples():
            out.append(dict(family=r.family, seed=r.seed, head=r.head, column=name, dataset='composite', rho=r.mean, n=np.nan,
                            n_proteins=np.nan))
    return pd.concat([cells, pd.DataFrame(out)], ignore_index=True)


def summarize(cells):
    g = cells.groupby(['family', 'head', 'column'])['rho']
    s = g.agg(mean='mean', sd='std', n_seeds='count').reset_index()
    seeds = cells.groupby(['family', 'head', 'column'])['seed'].apply(lambda x: ','.join(map(str, sorted(x)))).rename('seeds').reset_index()
    return s.merge(seeds, on=['family', 'head', 'column'])


# ----------------------------------------------------------------------------------------------
# paired bootstrap at seed 1 against the reference
# ----------------------------------------------------------------------------------------------
def seed1(fr):
    """The frame restricted to seed 1 of every family (FINAL_p0 has seed 1 only)."""
    f = Frame(fr.name, fr.spine, {}, notes=fr.notes)
    for k, a in fr.preds.items():
        fam = k.rsplit(':', 1)[0]
        if 1 in fr.notes['seeds'].get(fam, []):
            f.preds[k] = a[[fr.notes['seeds'][fam].index(1)]]
    return f


def boot_deltas(frames, B, pg=None):
    """Paired (family - REF) at seed 1 on the headline columns, primary readout and wt / comb separately."""
    groups = [  # (name, kind, {column: (frame, per_protein)})
        ('external', 'protein', {d: frames[d] for d in EXT12 if d in frames}),
        ('dms', 'site', {d: frames[d] for d in DMS7 if d in frames}),
        ('tsu_test', 'protein', {'tsuboyama_capped_test': frames['tsuboyama_capped_test']}),
        ('domainome', 'protein', {'domainome': frames['domainome']}),
    ]
    tsu = frames['tsuboyama_capped_test']
    m = epi_rows(tsu) & tsu.spine['wt_backbone'].to_numpy()
    efr = subframe(tsu, m, y=tsu.spine['dddG_ML'].to_numpy(dtype=float), keys=[k for k in tsu.preds if k.endswith(':epi')])
    efr.notes = tsu.notes
    groups.append(('tsu_dddG', 'protein', {'tsuboyama_capped_test|dddG': efr}))
    dom = frames['domainome']
    hom = homologous('domainome')
    lfr = subframe(dom, dom.spine['use'].to_numpy() & ~dom.spine['group'].astype(str).isin(hom).to_numpy())
    lfr.notes = dom.notes
    groups.append(('domainome_lowid', 'protein', {'domainome|lowid': lfr}))
    nm = tsu.spine['n_mut'].to_numpy()
    mfr = subframe(tsu, tsu.spine['use'].to_numpy() & (nm > 1))
    mfr.notes = tsu.notes
    groups.append(('tsu_test_multi', 'protein', {'tsuboyama_capped_test|multi': mfr}))
    sfr = subframe(tsu, tsu.spine['use'].to_numpy() & (nm == 1))
    sfr.notes = tsu.notes
    groups.append(('tsu_test_singles', 'protein', {'tsuboyama_capped_test|singles': sfr}))
    reps_all = {}
    for name, kind, fs in groups:
        fs = {c: seed1(f) for c, f in fs.items()}
        keys = sorted({k for f in fs.values() for k in f.preds if k.rsplit(':', 1)[1] in ('wt', 'comb', 'epi')})
        pp = {c: (c.split('|')[0] in PER_PROTEIN and not c.endswith('|dddG')) for c in fs}
        # bootstrap_rho decides per-protein from the dataset name; pass explicit frames one dataset name each
        point, reps = bootstrap_rho(fs, keys, kind=kind, B=B, seed=16, per_protein=None if name in ('external', 'dms') else
                                    next(iter(pp.values())))
        for c in reps:
            reps_all[c] = reps[c]
        print(f'  boot {name:16s} B={B} {len(keys)} keys', flush=True)
    # the deployed readout per column, then composites from paired replicate means
    for c, d in list(reps_all.items()):
        h = 'epi' if c.endswith('|dddG') else ('comb' if is_multi_col(c) else 'wt')
        for k in list(d):
            fam, head = k.rsplit(':', 1)
            if head == h:
                d[f'{fam}:primary'] = d[k]
    comp = {'ext_singles_mean': EXT_SINGLE, 'ext_multi_mean': EXT_MULTI, 'dms_mean': DMS7}
    for nm, cols in comp.items():
        ks = set.intersection(*[set(reps_all[c]) for c in cols if c in reps_all])
        reps_all[nm] = {k: np.nanmean([reps_all[c][k] for c in cols], axis=0) for k in ks}
    rows = []
    for col, d in reps_all.items():
        for k, r in d.items():
            fam, head = k.rsplit(':', 1)
            ref = d.get(f'{REF}:{head}')
            if fam == REF or ref is None:
                continue
            diff = r - ref
            lo, hi = ci(diff)
            p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
            rows.append(dict(family=fam, head=head, column=col, delta_boot_mean=float(np.nanmean(diff)), lo=lo, hi=hi,
                             p=max(p, 1 / (len(diff) + 1)), B=len(diff)))
    if pg is not None:
        rows += pg_boot(*pg, B=B)
    return pd.DataFrame(rows)


def pg_boot(cat, stab_hom, per, B):
    """Assays resampled within category, one draw shared by every family (seed 1)."""
    rng = np.random.default_rng(16)
    idx = {c: np.flatnonzero((cat == c).to_numpy()) for c in PG_CATS}
    draws = {c: rng.choice(v, (B, len(v))) for c, v in idx.items()}
    rows = []
    for mode in ('wt', 'comb'):
        ref = per.get((REF, 1, mode))
        if ref is None:
            continue
        for fam in FAMILIES:
            r = per.get((fam, 1, mode))
            if fam == REF or r is None:
                continue
            d = (r - ref).to_numpy()
            reps = np.mean([np.nanmean(d[draws[c]], axis=1) for c in PG_CATS], axis=0)
            lo, hi = ci(reps)
            p = 2 * min(np.mean(reps <= 0), np.mean(reps >= 0))
            for h in ([mode, 'primary'] if mode == 'wt' else [mode]):
                rows.append(dict(family=fam, head=h, column='pg|avg', delta_boot_mean=float(reps.mean()), lo=lo, hi=hi,
                                 p=max(p, 1 / (B + 1)), B=B))
    return rows


# ----------------------------------------------------------------------------------------------
# epistasis suite (esm_msr.epi_metrics, as 11d_epistasis_levels_bootstrap / BENCHMARKS section 2)
# ----------------------------------------------------------------------------------------------
EPI_METRICS = ['naive_rho', 'beyond_rho', 'cell_mag', 'cell_rank']
EPI_DS = ['tsuboyama_capped_test', 'tsuboyama_capped_val']
_MUT = re.compile(r'^([A-Za-z])(-?\d+)([A-Za-z])$')
EPI_TABLES = {}       # (family, seed) -> E.Table, shared with fork workers


def _parse_mut(m, code):
    out = []
    for part in str(m).split(':'):
        g = _MUT.match(part.strip())
        if not g:
            return None
        out.append((code, int(g.group(2)), g.group(3)))
    return tuple(out)


def epi_tables(fr):
    """One ``epi_metrics.Table`` per family x seed: predicted epistasis = comb(AB) - WT(A) - WT(B) (the 'epi' head is fed comb, as in
    11d), on the measured doubles whose singles are both measured (12,455 test / 8,494 val).  Latent scale, no link."""
    sp = fr.spine
    idx = np.flatnonzero(sp['use'].to_numpy())
    keys = [_parse_mut(m, c) for m, c in zip(sp['mut'].to_numpy()[idx], sp['code'].astype(str).to_numpy()[idx])]
    st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
    base = dict(mut_key=keys, subset_type=st, ddG=sp['y'].to_numpy(float)[idx], dddG=sp['dddG_ML'].to_numpy(float)[idx],
                dG_wt=sp['dG_wt'].to_numpy(float)[idx])
    out = {}
    for fam, seeds in fr.notes['seeds'].items():
        if f'{fam}:wt' not in fr.preds or f'{fam}:comb' not in fr.preds:
            continue
        for j, sd in enumerate(seeds):
            wt, comb = fr.preds[f'{fam}:wt'][j][idx], fr.preds[f'{fam}:comb'][j][idx]
            out[(fam, sd)] = E.Table(dict(base, wt=wt, mt=comb, comb=comb), None, heads=('epi', 'add'))
    return out


def _epi_values(t):
    r = E.compute(t, {m: ('epi',) for m in EPI_METRICS})
    return {m: float(r.get(f'epi_{m}_epi', np.nan)) for m in EPI_METRICS}


def epi_cells(frames):
    rows = []
    for ds in EPI_DS:
        if ds not in frames:
            continue
        for (fam, sd), t in epi_tables(frames[ds]).items():
            for m, v in _epi_values(t).items():
                rows.append(dict(family=fam, seed=sd, head='epi', column=f'{ds}|{m}', dataset=ds, rho=v, n=t.n, n_proteins=len(np.unique(t.lib))))
    return pd.DataFrame(rows)


def _resampled(t, idx, rep):
    """The table on a domain resample; a repeated domain is a new domain (its pairs and residues kept apart), as 11d."""
    r = copy.copy(t)
    n0 = t.n
    for name, v in vars(t).items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n0:
            setattr(r, name, v[idx])
        elif isinstance(v, dict):
            setattr(r, name, {k: (a[idx] if isinstance(a, np.ndarray) and a.ndim == 1 and len(a) == n0 else a) for k, a in v.items()})
    r.n = len(idx)
    off_rc = max(t.row.max(), t.col.max()) + 1
    r.pair = t.pair[idx] + rep * (t.pair.max() + 1)
    r.row, r.col = t.row[idx] + rep * off_rc, t.col[idx] + rep * off_rc
    r.lib = t.lib[idx] + rep * (t.lib.max() + 1)
    r._mats = None
    return r


def _epi_draw(counts):
    t0 = next(iter(EPI_TABLES.values()))
    libs = np.unique(t0.lib)
    by = {l: np.flatnonzero(t0.lib == l) for l in libs}
    idx_l, rep_l, k = [], [], 0
    for l, c in zip(libs, counts):
        for _ in range(int(c)):
            idx_l.append(by[l]); rep_l.append(np.full(len(by[l]), k)); k += 1
    idx, rep = np.concatenate(idx_l), np.concatenate(rep_l)
    return {fam: _epi_values(_resampled(t, idx, rep)) for (fam, _), t in EPI_TABLES.items()}


def epi_boot(fr, B, procs=8):
    """Paired domain bootstrap of (family - REF) at seed 1 on the Tsuboyama test doubles; same draw for every family."""
    EPI_TABLES.clear()
    EPI_TABLES.update({k: t for k, t in epi_tables(fr).items() if k[1] == 1})
    t0 = next(iter(EPI_TABLES.values()))
    assert all(np.array_equal(t.lib, t0.lib) for t in EPI_TABLES.values()), 'tables disagree on the doubles'
    nl = len(np.unique(t0.lib))
    rng = np.random.default_rng(16)
    draws = [rng.multinomial(nl, np.full(nl, 1.0 / nl)) for _ in range(B)]
    with mp.get_context('fork').Pool(procs) as pool:
        reps = pool.map(_epi_draw, draws)
    rows = []
    for m in EPI_METRICS:
        for fam, _ in EPI_TABLES:
            if fam == REF:
                continue
            diff = np.array([r[fam][m] - r[REF][m] for r in reps])
            diff = diff[np.isfinite(diff)]
            lo, hi = ci(diff)
            p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
            for h in ('epi', 'primary'):
                rows.append(dict(family=fam, head=h, column=f'tsuboyama_capped_test|{m}', delta_boot_mean=float(diff.mean()),
                                 lo=lo, hi=hi, p=max(p, 1 / (len(diff) + 1)), B=len(diff)))
    print(f'  boot epistasis suite B={B} {len(EPI_TABLES)} tables, {nl} domains', flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# candidate layouts
# ----------------------------------------------------------------------------------------------
COLLABEL = {
    'tsuboyama_capped_test': 'Tsu test', 'tsuboyama_capped_test|singles': 'Tsu test singles',
    'tsuboyama_capped_test|multi': 'Tsu test doubles', 'tsuboyama_capped_test|dddG': 'Tsu test ΔΔΔG',
    'ext_singles_mean': 'Ext. singles (10)', 'ext_multi_mean': 'PTMUL/-D', 'dms_mean': 'DMS (7)',
    'dms_multi_mean': 'DMS multis (6)', 'domainome': 'Domainome', 'domainome|lowid': 'Domainome low-id',
    'pg|avg': 'PG avg', 'pg|avg_lowid': 'PG avg low-id', 'pg|Stability': 'PG Stab.', 'pg|Stability_lowid': 'PG Stab. low-id',
}
for _d in EXT12 + DMS7:
    COLLABEL.setdefault(_d, LABEL.get(_d, _d))
    COLLABEL.setdefault(f'{_d}|multi', LABEL.get(_d, _d) + ' multi')
    COLLABEL.setdefault(f'{_d}|singles', LABEL.get(_d, _d) + ' singles')

HEADLINE = ['tsuboyama_capped_test', 'tsuboyama_capped_test|dddG', 'ext_singles_mean', 'ext_multi_mean', 'dms_mean',
            'domainome|lowid', 'pg|avg']

LAYOUTS = {
    # A: one compact table, deployed readout only, group means
    'A_compact_primary': dict(heads=['primary'], cols=HEADLINE),
    # B: same columns, wt and comb side by side for every column (is the head choice informative?)
    'B_compact_wt_vs_comb': dict(heads=['wt', 'comb'], cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|singles',
                                 'tsuboyama_capped_test|multi', 'ext_singles_mean', 'ext_multi_mean', 'dms_mean',
                                 'dms_multi_mean', 'domainome|lowid', 'pg|avg']),
    # C: in-distribution vs out-of-distribution split into two tables
    'C1_in_distribution': dict(heads=['primary'], cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|singles',
                               'tsuboyama_capped_test|multi', 'tsuboyama_capped_test|dddG']),
    'C2_out_of_distribution': dict(heads=['primary'], cols=['ext_singles_mean', 'ext_multi_mean', 'dms_mean', 'domainome',
                                   'domainome|lowid', 'pg|avg', 'pg|avg_lowid', 'pg|Stability_lowid']),
    # D: every dataset, deployed readout (the detailed supplementary table)
    'D_full_primary': dict(heads=['primary'], cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|dddG'] + EXT12 + DMS7
                           + ['domainome', 'domainome|lowid', 'pg|avg']),
    # E: multi-mutant sets only, every readout (where the comb/wt+ question lives)
    'E_multi_readouts': dict(heads=['wt', 'wt+', 'comb'], cols=['tsuboyama_capped_test|multi', 'ptmul', 'ptmuld']
                             + [f'{d}|multi' for d in DMS_MULTI]),
}


def fmt_cell(r, delta=None, ref_row=None):
    if r is None or not np.isfinite(r['mean']):
        return '—'
    s = f"{r['mean']:.3f}"
    if r['n_seeds'] > 1 and np.isfinite(r['sd']):
        s += f" ± {r['sd']:.3f}"
    if delta is not None:
        mark = '**' if (delta['lo'] > 0 or delta['hi'] < 0) else ''
        s += f" ({mark}{delta['delta_boot_mean']:+.3f}{mark})"
    return s


def render(name, spec, summ, deltas, mega_own):
    S = summ.set_index(['family', 'head', 'column'])
    D = deltas.set_index(['family', 'head', 'column']) if len(deltas) else None
    heads, cols = spec['heads'], spec['cols']
    hdr = ['Family', 'Change', 'Seeds'] + [f'{COLLABEL.get(c, c)}' + (f' [{h}]' if len(heads) > 1 else '') for c in cols for h in heads]
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for fam, meta in FAMILIES.items():
        cells = []
        seeds = set()
        for c in cols:
            for h in heads:
                hh = 'epi' if (c.endswith('|dddG') and h in ('primary', 'comb')) else h
                key = (fam, hh if c.endswith('|dddG') else h, c)
                r = S.loc[key] if key in S.index else None
                if fam == 'FINAL_p0_mega' and c.startswith('tsuboyama') and mega_own is not None:
                    mk = ('FINAL_p0_mega', key[1], c)
                    r = mega_own.loc[mk] if mk in mega_own.index else None
                    cells.append((fmt_cell(r) + '†') if r is not None else '—')
                    if r is not None:
                        seeds.add(r['seeds'])
                    continue
                dh = key[1]
                dl = D.loc[(fam, dh, c)] if (D is not None and dh and (fam, dh, c) in D.index) else None
                cells.append(fmt_cell(r, dl))
                if r is not None:
                    seeds.add(r['seeds'])
        sd = '/'.join(sorted(seeds)) if seeds else '—'
        lines.append(f"| {meta['short']} | {meta['change']} | {sd} | " + ' | '.join(cells) + ' |')
    note = ('Cells: seed mean ± seed SD (Spearman ρ). In parentheses: paired bootstrap Δ vs p0 at seed 1 (bold: 95% CI excludes 0). '
            '† FINAL_p0_mega on its own mega_splits Tsuboyama split (not comparable). ΔΔΔG = ρ of comb(AB) − WT(A) − WT(B) vs measured ΔΔΔG.')
    text = f'### {name}\n\n' + '\n'.join(lines) + f'\n\n{note}\n'
    (OUT / f'layout_{name}.md').write_text(text)
    return text


def informativeness(summ):
    """Which columns separate the families, and does the readout matter?

    Per column and head, over the capped-split families with >= 2 seeds (FINAL_p0 has one seed and FINAL_p0_mega another
    split, so both are left out of the spread):  ``spread`` = SD of the family means, ``noise`` = RMS seed SD,
    ``snr`` = spread / (noise / sqrt(n_seeds)) -- above ~2 the column can rank the ablations at all.
    Per column with both heads: mean comb - wt over families, and the Spearman correlation of the family means under the
    two heads (1 = the head choice does not change the ablation story)."""
    from scipy.stats import spearmanr
    fams = [f for f in FAMILIES if f not in (REF, 'FINAL_p0_mega') and '@' not in f]
    d = summ[summ['family'].isin(fams) & (summ['n_seeds'] >= 2)]
    rows = []
    for (col, head), g in d.groupby(['column', 'head']):
        if len(g) < 4:
            continue
        noise = float(np.sqrt(np.mean(g['sd'] ** 2)))
        spread = float(g['mean'].std())
        n = float(g['n_seeds'].mean())
        best = g.loc[g['mean'].idxmax(), 'family']
        rows.append(dict(column=col, head=head, n_families=len(g), spread=spread, noise=noise,
                         snr=spread / (noise / np.sqrt(n)) if noise > 0 else np.nan, best=FAMILIES[best]['short'],
                         range=float(g['mean'].max() - g['mean'].min())))
    inf = pd.DataFrame(rows)
    hv = []
    for col, g in d[d['head'].isin(['wt', 'comb'])].groupby('column'):
        w = g[g['head'] == 'wt'].set_index('family')['mean']
        c = g[g['head'] == 'comb'].set_index('family')['mean']
        common = w.index.intersection(c.index)
        if len(common) < 4:
            continue
        hv.append(dict(column=col, comb_minus_wt=float((c[common] - w[common]).mean()),
                       comb_minus_wt_ref=float(summ.query("family == @REF and column == @col and head == 'comb'")['mean'].sum()
                                               - summ.query("family == @REF and column == @col and head == 'wt'")['mean'].sum()),
                       rank_agreement=float(spearmanr(w[common], c[common])[0])))
    return inf, pd.DataFrame(hv)


def write_informativeness(summ):
    inf, hv = informativeness(summ)
    inf.sort_values(['head', 'snr'], ascending=[True, False]).to_csv(OUT / 'informativeness.csv', index=False, float_format='%.4f')
    hv.to_csv(OUT / 'head_choice.csv', index=False, float_format='%.4f')
    return inf, hv


# ----------------------------------------------------------------------------------------------
# the ablation tables: deltas vs FINAL_p0, deployed readout (`main`) and WT head alone (`wt`)
# ----------------------------------------------------------------------------------------------
# The single-protein DMS sets are left out of both tables (ProteinGym and Domainome cover out-of-distribution
# fitness); they stay in summary.csv and layout_D / layout_E.
# FINAL_p0_mega belongs with the retrained-methods comparison, not here (it stays in the CSVs and layout_*).
def pooled_seed_sd(summ, col, head='primary'):
    """RMS seed SD of a column (one readout) over the capped-split ablation families with >= 2 seeds: the seed noise of
    one run, used for a family whose own SD is not estimable (a single-seed run)."""
    d = summ[(summ['head'] == head) & (summ['column'] == col) & (summ['n_seeds'] >= 2)
             & ~summ['family'].isin([REF, 'FINAL_p0_mega']) & ~summ['family'].str.contains('@')]
    return float(np.sqrt(np.mean(d['sd'] ** 2))) if len(d) else np.nan


NOISE_NOTE = ('Bold: |Δ| > {z:g} SE, where SE combines seed noise (family SD, or the pooled seed SD of the column for a '
              'single-seed run) and the paired seed-1 bootstrap over proteins / sites / assays. ')
SEED_NOTE = ('‡: fewer seeds than the rest of the row (the cell\'s seeds are in {csv}). —: not yet run.')
EPI_NOTE = ('Epistasis columns: comb(AB) − WT(A) − WT(B) against measured ΔΔΔG on the 12,455 Tsuboyama test doubles with both '
            'singles measured (esm_msr.epi_metrics; beyond ρ = Spearman above the global saturation curve, cell mag. = '
            'double-centred residuals within complete blocks of each position pair), sampling noise by a paired bootstrap over '
            'the 13 test domains. ')

# The two tables.  `main`: the deployed readout (WT head on singles-only sets, 1/2(MT + WT) where multi-mutants are scored).
# `wt`: the WT head alone on every row (wt_lora_pred, what a WT-only deployment outputs: on a multi-mutant row its own
# prediction of that row). Rows whose change reaches the MT head only are left out of `wt`: their WT heads move only through
# joint training, by less than FINAL_p0's seeds differ from each other (matched-seed WT r 0.995-0.999 vs 0.992 between p0
# seeds on the Tsuboyama test singles). The epistasis columns are left out of `wt`: the WT head is additive by construction.
MT_ONLY = {'FINAL_p0_pw0', 'FINAL_p0_r2a4', 'FINAL_p1_marg'}
TABLES = {
    'main': dict(head='primary', exclude={'FINAL_p0_mega'},
                 cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|multi', 'tsuboyama_capped_test|beyond_rho',
                       'tsuboyama_capped_test|cell_mag', 'ext_singles_mean', 'ext_multi_mean', 'domainome|lowid', 'pg|avg'],
                 note='p0 row: Spearman ρ (seed mean; seeds listed). Other rows: Δρ = family seed mean − p0, deployed readout '
                      '(WT head on singles-only sets, 1/2(MT + WT) where multi-mutants are scored). ' + NOISE_NOTE + EPI_NOTE),
    'wt': dict(head='wt', exclude={'FINAL_p0_mega'} | MT_ONLY,
               cols=['tsuboyama_capped_test', 'tsuboyama_capped_test|singles', 'tsuboyama_capped_test|multi',
                     'ext_singles_mean', 'ext_multi_mean', 'domainome|lowid', 'pg|avg'],
               note='WT head alone on every row (wt_lora_pred). p0 row: Spearman ρ (seed mean; seeds listed). Other rows: '
                    'Δρ = family seed mean − p0 under the same readout. Left out: ablations that reach the MT head only '
                    '(no pair loss, MT LoRA rank 2, p1 MT marginal), whose WT heads differ from p0 by less than p0\'s own '
                    'seeds do, and the epistasis columns (the WT head is additive by construction). ' + NOISE_NOTE),
}
COL_LABEL = {'tsuboyama_capped_test': 'Tsuboyama test', 'tsuboyama_capped_test|multi': 'Tsuboyama doubles',
             'tsuboyama_capped_test|singles': 'Tsuboyama singles',
             'ext_singles_mean': 'External singles (10)', 'ext_multi_mean': 'PTMUL / PTMUL-D', 'dms_mean': 'DMS (7)',
             'domainome|lowid': 'Domainome low-id', 'pg|avg': 'ProteinGym avg',
             'tsuboyama_capped_test|beyond_rho': 'Epistasis beyond ρ', 'tsuboyama_capped_test|cell_mag': 'Epistasis cell mag.'}
GROUP_LABEL = {'masking': 'Masking', 'loss': 'Loss', 'architecture': 'Architecture', 'data': 'Training data',
               'cross-mask': 'Cross-masking (evaluation only)'}


def delta_table(summ, deltas, name, z=2.0):
    """Delta = family seed mean - FINAL_p0 seed mean under the table's readout.  Its standard error combines
      seed noise:      sqrt(sd_f^2 / n_f + sd_ref^2 / n_ref), each SD the family's own, or the pooled seed SD of the
                       column when a family has one seed, and
      sampling noise:  the paired seed-1 bootstrap (delta_s1.csv; 95% CI width / 3.92) of the same column and readout.
    A delta is flagged (bold) when |delta| > z * SE.  Writes table_<name>_delta.{csv,md,tex}."""
    spec = TABLES[name]
    head, cols = spec['head'], spec['cols']
    csv_name = f'table_{name}_delta.csv'
    # the epistasis columns hold the 'epi' readout under every head name
    S = summ[(summ['head'] == head) | ((summ['head'] == 'primary') & summ['column'].str.endswith(('|beyond_rho', '|cell_mag')))]
    S = S[(S['head'] == head) | ~S.duplicated(['family', 'column'], keep=False)].set_index(['family', 'column'])
    D = deltas[deltas['head'] == head].set_index(['family', 'column']) if len(deltas) else None
    if D is not None and head != 'primary':
        D = pd.concat([D, deltas[deltas['head'] == 'primary'].set_index(['family', 'column'])]).loc[lambda x: ~x.index.duplicated()]
    rows = []
    for fam in FAMILIES:
        if fam in spec['exclude']:
            continue
        for col in cols:
            if (fam, col) not in S.index or (REF, col) not in S.index:
                continue
            f, r = S.loc[(fam, col)], S.loc[(REF, col)]
            pooled = pooled_seed_sd(summ, col, head)
            sd_f = f['sd'] if f['n_seeds'] > 1 and np.isfinite(f['sd']) else pooled
            sd_r = r['sd'] if r['n_seeds'] > 1 and np.isfinite(r['sd']) else pooled
            se_seed = float(np.sqrt(sd_f ** 2 / f['n_seeds'] + sd_r ** 2 / r['n_seeds']))
            se_boot = np.nan
            if D is not None and (fam, col) in D.index:
                b = D.loc[(fam, col)]
                se_boot = float((b['hi'] - b['lo']) / 3.92)
            se = float(np.sqrt(se_seed ** 2 + (se_boot if np.isfinite(se_boot) else 0.0) ** 2))
            delta = float(f['mean'] - r['mean']) if fam != REF else np.nan
            rows.append(dict(family=fam, column=col, head=head, value=float(f['mean']), n_seeds=int(f['n_seeds']),
                             seeds=f['seeds'], delta=delta, se_seed=se_seed, se_boot=se_boot, se=se,
                             flag=bool(fam != REF and abs(delta) > z * se)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / csv_name, index=False, float_format='%.5f')

    def cell(fam, col, tex=False):
        r = T[(T['family'] == fam) & (T['column'] == col)] if len(T) else T
        if r.empty:
            return '—' if not tex else '--'
        r = r.iloc[0]
        few = r['n_seeds'] < T.loc[T['family'] == fam, 'n_seeds'].max()        # fewer seeds than the rest of the row
        mark = ('‡' if not tex else '$^\\ddagger$') if few else ''
        if fam == REF:
            return f"{r['value']:.3f}{mark}"
        s = '0.000' if round(r['delta'], 3) == 0 else f"{r['delta']:+.3f}".replace('-', '−' if not tex else '$-$')
        if r['flag']:
            s = f'**{s}**' if not tex else f'\\textbf{{{s}}}'
        return s + mark

    def seeds(fam):
        x = T.loc[T['family'] == fam, 'seeds'].dropna() if len(T) else []
        return x.iloc[0] if len(x) else '—'

    hdr = ['Ablation', 'Change from p0', 'Seeds'] + [COL_LABEL[c] for c in cols]
    md = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    tex = ['\\begin{tabular}{ll' + 'r' * len(cols) + '}', '\\toprule',
           'Ablation & Change & ' + ' & '.join(COL_LABEL[c] for c in cols) + ' \\\\', '\\midrule']
    md.append(f"| **{FAMILIES[REF]['short']}** | {FAMILIES[REF]['change']} | {seeds(REF)} | "
              + ' | '.join(cell(REF, c) for c in cols) + ' |')
    tex.append('p0 (reference) & -- & ' + ' & '.join(cell(REF, c, True) for c in cols) + ' \\\\')
    for grp, glabel in GROUP_LABEL.items():
        fams = [f for f, m in FAMILIES.items() if m['group'] == grp and f not in spec['exclude']]
        if not fams:
            continue
        md.append(f"| *{glabel}* |" + ' |' * (len(hdr) - 1))
        tex.append('\\midrule\n\\multicolumn{' + str(len(cols) + 2) + '}{l}{\\textit{' + glabel + '}} \\\\')
        for fam in fams:
            short, change = FAMILIES[fam]['short'], FAMILIES[fam]['change'].split(' (')[0]
            md.append(f"| {short} | {change} | {seeds(fam)} | " + ' | '.join(cell(fam, c) for c in cols) + ' |')
            change_tex = change.replace('_', '\\_').replace('->', '$\\to$')
            tex.append(f"{short} & {change_tex} & " + ' & '.join(cell(fam, c, True) for c in cols) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    note = spec['note'].format(z=z) + SEED_NOTE.format(csv=csv_name)
    (OUT / f'table_{name}_delta.md').write_text('\n'.join(md) + '\n\n' + note + '\n')
    (OUT / f'table_{name}_delta.tex').write_text('\n'.join(tex) + '\n% ' + note + '\n')
    return '\n'.join(md) + '\n\n' + note


def main_delta_table(summ, deltas, z=2.0):
    return '\n\n'.join(delta_table(summ, deltas, name, z) for name in TABLES)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--B', type=int, default=200, help='bootstrap draws (draft default 200; paper 1000-2000)')
    ap.add_argument('--sigma', default='1.0')
    ap.add_argument('--skip-boot', action='store_true')
    ap.add_argument('--render-only', action='store_true', help='re-draw the layouts from the CSVs already in results/16_ablation')
    args = ap.parse_args()
    if args.render_only:
        summ = pd.read_csv(OUT / 'summary.csv', dtype={'seeds': str})
        mega_own = summarize(pd.read_csv(OUT / 'cells_long_mega_own_split.csv')).set_index(['family', 'head', 'column'])
        deltas = pd.read_csv(OUT / 'delta_s1.csv') if (OUT / 'delta_s1.csv').exists() else pd.DataFrame()
        write_informativeness(summ)
        print(main_delta_table(summ, deltas))
        for name, spec in LAYOUTS.items():
            print(render(name, spec, summ, deltas, mega_own))
        return

    datasets = EXT12 + DMS7 + ['domainome', 'tsuboyama_capped_val', 'tsuboyama_capped_test']
    frames, cov = {}, []
    for ds in datasets:
        fr = load_final_frame(ds, sigma=args.sigma)
        frames[ds] = fr
        for fam, ss in fr.notes['seeds'].items():
            cov.append(dict(dataset=ds, family=fam, seeds=','.join(map(str, ss))))
        print(f'loaded {ds:42s} {len(fr.notes["seeds"])} families', flush=True)
    pd.DataFrame(cov).to_csv(OUT / 'coverage.csv', index=False)

    cells = score_columns(frames)
    cells = pd.concat([cells, epi_cells(frames)], ignore_index=True)
    # FINAL_p0_mega on its own Tsuboyama split
    mega_frames = {ds: load_final_frame(ds, sigma=args.sigma, mega_own_split=True) for ds in ('tsuboyama_capped_val', 'tsuboyama_capped_test')}
    mega_cells = score_columns({k: v for k, v in mega_frames.items() if v is not None})
    mega_cells = add_primary(mega_cells)

    pg = pg_table()
    cells = pd.concat([cells, pg_summaries(*pg)], ignore_index=True)
    cells = add_composites(add_primary(cells))
    cells.to_csv(OUT / 'cells_long.csv', index=False, float_format='%.5f')
    mega_cells.to_csv(OUT / 'cells_long_mega_own_split.csv', index=False, float_format='%.5f')
    summ = summarize(cells)
    summ.to_csv(OUT / 'summary.csv', index=False, float_format='%.5f')
    mega_own = summarize(mega_cells).set_index(['family', 'head', 'column'])
    write_informativeness(summ)

    deltas = pd.DataFrame()
    if not args.skip_boot:
        deltas = boot_deltas(frames, args.B, pg=pg)
        deltas = pd.concat([deltas, epi_boot(frames['tsuboyama_capped_test'], args.B)], ignore_index=True)
        deltas.to_csv(OUT / 'delta_s1.csv', index=False, float_format='%.5f')
    elif (OUT / 'delta_s1.csv').exists():
        deltas = pd.read_csv(OUT / 'delta_s1.csv')

    print(main_delta_table(summ, deltas))
    for name, spec in LAYOUTS.items():
        print(render(name, spec, summ, deltas, mega_own))


if __name__ == '__main__':
    main()
