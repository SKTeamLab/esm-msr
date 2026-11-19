#!/usr/bin/env python
"""ProteinGym (213 DMS) & Domainome: unified main models + full-model PGym evaluation (seeds 1-3 averaged).

Evaluates the mean over seeds 1-3 of every retrained model across all 213 non-dropped ProteinGym assays, leaving out every assay (all categories) and every Domainome domain that is homologous to the training set under the split's own four-part
rule (MMseqs2 E <= 1e-3, identity > 0.40, Foldseek E <= 1e-3, Foldseek TM > 0.70 against the capped training libraries; ``msr_analysis.heldout``, ``leak_audit/21_tsuboyama_rule.py``).
``pg_summary_all213.csv`` / ``pg_all_models_summary_all213.csv`` keep every assay for reference.

- Unified roster (``EXT_ORDER``, for ``pg_categories`` (drawn by ``01_summary.py`` from ``pg_summary.csv``) and ``pg_per_dms_scatter``):
  - Singles-only assays use singles models (ESM-MSR WT head, MutateEverything st1,
    ThermoMPNN single-additive, SPURS single-additive);
  - Multi-mutant assays use multi models (ESM-MSR p0 combined when finished else WT fallback,
    MutateEverything st2, ThermoMPNN-D on doubles / ThermoMPNN additive on singles & >2 via
    corrected run, seeds 1-3 (seeds 2 and 3 run on gpu4 on Nov 19), SPURS multi);
  - ``pg_categories`` appends the held-out Domainome domains (seeds 1-3 singles, 1,000-draw domain bootstrap, seeds resampled).
- Full ProteinGym roster (all models with PGym data, for ``pg_table_heatmap`` and ``pg_overview``).

Bootstrap: resamples DMS assays with replacement within each category (B = 1000, paired across models),
and resamples the held-out Domainome domains with replacement (B = 1000) for the appended Domainome column.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from msr_analysis import style  # noqa: E402
from msr_analysis.boot import bootstrap_rho, ci  # noqa: E402
from msr_analysis.common import out_dir  # noqa: E402
from msr_analysis.data import Frame, load_frame  # noqa: E402
from msr_analysis.external import EXT_ORDER, register_ext_styles, seed_frame  # noqa: E402
from msr_analysis.final_runs import load_final_frame  # noqa: E402
from msr_analysis.registry import C, CACHE, MODELS, MSR, PGYM, PUBLIC, REFERENCE, RETRAINED, ROOT  # noqa: E402

OUT = out_dir('05_proteingym')
CATS = ['Activity', 'Binding', 'Expression', 'OrganismalFitness', 'Stability']
CAT_LABEL = {
    'Activity': 'Activity\n(43)',
    'Binding': 'Binding\n(13)',
    'Expression': 'Expression\n(18)',
    'OrganismalFitness': 'Org. fitness\n(73)',
    'Stability': 'Stability*\n(40/66)',
    'PG avg': 'PG avg*',
    'DMS mean': 'DMS mean*\n(187/213)',
    'Domainome': 'Domainome*\n(held-out)',
}
OV_TITLES = {
    'Activity': 'Activity (43 assays)',
    'Binding': 'Binding (13 assays)',
    'Expression': 'Expression (18 assays)',
    'OrganismalFitness': 'Org. fitness (73 assays)',
    'Stability': 'Stability* (40/66 assays)',
}
DROP = {
    'BRCA1_HUMAN_Findlay_2018',
    'BRCA2_HUMAN_Erwood_2022_HEK293T',
    'POLG_CXB3N_Mattenberger_2021',
    'SCN5A_HUMAN_Glazer_2019',
}
B = 1000             # default size (the cache below is keyed on B, so the first run after a change recomputes); paper 2000
SIGMA_RHO = Path(__file__).resolve().parent / 'proteingym' / 'p0_sigma' / 'results' / 'per_assay_rho.csv'      # FINAL_p0 seeds 1-3 (mean), WT head, per-assay rho at sigma 0, 0.5, 1 (proteingym/p0_sigma/01_per_assay_rho.py)
SIGMA_MODELS = ('ESM-MSR σ=1', 'ESM-MSR σ=0.5')
ESM3_S0_COMB = ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'      # ESM3-small-open as the sigma-0 read of the ESM-MSR-small checkpoint (adapters off, unmasked), with the combined prediction: the zero-shot reference on the multi-mutant assays

ALL_PG_MODELS = [
    'pg:msrp0:comb',
    'pg:msrp0:wt',
    'pg:msrdev:comb',
    'pg:msrdev:wt',
    'pg:msr:comb',
    'pg:msr:wt',
    'pg:msrrel:comb',
    'pg:msrrel:wt',
    'pg:me1',
    'pg:me2',
    'pg:tmpnnd',
    'pg:tmpnn',
    'pg:spurs',
    'pg:spursm',
    'pg:esm3dg',
    'pg:geoddg',
    'pg:rosetta',
    'pg:prosst',
    'pg:esm3_pub',
    'pg:msrrel@0:wt',
    'pg:pmpnn',
    'pg:pmpnn_pub',
    'pg:spurs_pub',
    'pg:spursm_pub',
    'pg:esmtherm',
]
MEGA_PG = ['pg:esmtherm', 'pg:spursm_pub', 'pg:spurs_pub']

PG_MODEL_STYLES = {
    'pg:msrp0:comb':  dict(label='ESM-MSR (p0) combined',            color=C['msr_p0'], marker='D', fill=True,  group='msr'),
    'pg:msrp0:wt':    dict(label='ESM-MSR (p0) WT head',             color=C['msr_p0'], marker='o', fill=False, group='msr'),
    'pg:msrdev:comb': dict(label='ESM-MSR (devel) combined',         color='#000000',   marker='D', fill=True,  group='msr'),
    'pg:msrdev:wt':   dict(label='ESM-MSR (devel) WT head',          color='#000000',   marker='o', fill=False, group='msr'),
    'pg:msr:comb':    dict(label='ESM-MSR (baseline) combined',      color='#555555',   marker='D', fill=True,  group='msr'),
    'pg:msr:wt':      dict(label='ESM-MSR (baseline) WT head',       color='#555555',   marker='o', fill=False, group='msr'),
    'pg:msrrel:comb': dict(label='ESM-MSR (released) combined',      color='#7f8c8d',   marker='D', fill=True,  group='msr'),
    'pg:msrrel:wt':   dict(label='ESM-MSR (released) WT head',       color='#7f8c8d',   marker='o', fill=False, group='msr'),
    'pg:me1':         dict(label='MutateEverything st1',             color=C['me1'],    marker='h', fill=True,  group='retrain'),
    'pg:me2':         dict(label='MutateEverything st2',             color=C['me2'],    marker='H', fill=True,  group='retrain'),
    'pg:tmpnnd':      dict(label='ThermoMPNN + ThermoMPNN-D',        color=C['tmd'],    marker='P', fill=True,  group='retrain'),
    'pg:tmpnn':       dict(label='ThermoMPNN',                       color=C['tm'],     marker='p', fill=True,  group='retrain'),
    'pg:spurs':       dict(label='SPURS',                            color=C['spurs'],  marker='s', fill=True,  group='retrain'),
    'pg:spursm':      dict(label='SPURS multi',                      color=C['spursm'], marker='s', fill=False, group='retrain'),
    'pg:esm3dg':      dict(label='ESM3ΔG',                           color='#cb4335',   marker='*', fill=True,  group='public'),
    'pg:geoddg':      dict(label='GeoDDG-3D',                        color='#117a65',   marker='^', fill=True,  group='public'),
    'pg:rosetta':     dict(label='Rosetta cart. ΔΔG (singles)',      color=C['ros'],    marker='+', fill=True,  group='public'),
    'pg:prosst':      dict(label='ProSST (zero-shot)',               color=C['prosst'], marker='X', fill=True,  group='zeroshot'),
    'pg:esm3_pub':    dict(label='ESM3-open (PG reported)',          color='#e74c3c',   marker='o', fill=True,  group='zeroshot'),
    'pg:msrrel@0:wt': dict(label='ESM3-small-open, σ = 0 (zero-shot)', color=C['zero'], marker='^', fill=False, group='zeroshot'),
    'pg:pmpnn':       dict(label='ProteinMPNN (zero-shot)',          color=C['pmpnn'],  marker='d', fill=True,  group='zeroshot'),
    'pg:pmpnn_pub':   dict(label='ProteinMPNN (PG reported)',        color=C['pmpnn'],  marker='d', fill=False, group='zeroshot'),
    'pg:spurs_pub':   dict(label='SPURS (public ckpt)*',             color='#a04000',   marker='s', fill=True,  group='public'),
    'pg:spursm_pub':  dict(label='SPURS multi (public ckpt)*',       color='#a04000',   marker='s', fill=False, group='public'),
    'pg:esmtherm':    dict(label='ESMtherm*',                        color='#7d6608',   marker='<', fill=True,  group='public'),
}

# Documented training / homolog overlap on Stability* (per BENCHMARKS table 1.4.2 & leak_audit_pg_per_dms.csv):
# ESMtherm*, SPURS (public ckpt)*, SPURS multi (public ckpt)* directly trained on unsplit MegaScale libraries
# (33-38 of 40 remaining Stability assays); ESM3dG has homolog overlap H.02 / h.33 against Stability.
PG_STAB_LEAK = {'pg:esmtherm', 'pg:spurs_pub', 'pg:spursm_pub', 'pg:esm3dg'}


SEEDS = (1, 2, 3)


def _p0_wt_candidates(seed=1):
    cands = []
    if os.environ.get('MSR_PGYM_P0_WT') and seed == 1:
        cands.append(Path(os.environ['MSR_PGYM_P0_WT']).expanduser())
    cands.extend([
        ROOT / 'final_pgym_wt' / 'FINAL_p0' / f's{seed}',
        ROOT / 'p0_fir_pgym_wt' / f's{seed}',
    ])
    return cands


def _p0_comb_candidates(seed=1):
    cands = []
    if os.environ.get('MSR_PGYM_P0_COMB') and seed == 1:
        cands.append(Path(os.environ['MSR_PGYM_P0_COMB']).expanduser())
    cands.extend([
        ROOT / 'final_pgym_comb' / 'FINAL_p0' / f's{seed}',
        ROOT / 'final_pgym' / 'FINAL_p0' / f's{seed}',
        ROOT / 'p0_fir_pgym_comb' / f's{seed}',
        ROOT / 'p0_fir_pgym' / f's{seed}',
        MSR / 'p0' / f'seed{seed}' / 'sigma1' / 'proteingym',
        ROOT / 'final_pgym_wt' / 'FINAL_p0' / f's{seed}',
        ROOT / 'p0_fir_pgym_wt' / f's{seed}',
    ])
    return cands


def _find_p0_comb_csv(dms, seed=1):
    for d in _p0_comb_candidates(seed):
        p = d / f'{dms}.csv'
        if not p.exists():
            continue
        hdr = set(pd.read_csv(p, nrows=0).columns)
        if 'combined_pred' in hdr or ('wt_lora_pred' in hdr and 'mt_lora_pred' in hdr):
            return p
    return None


def _find_p0_wt_csv(dms, seed=1):
    for d in _p0_wt_candidates(seed) + _p0_comb_candidates(seed):
        p = d / f'{dms}.csv'
        if not p.exists():
            continue
        hdr = set(pd.read_csv(p, nrows=0).columns)
        if 'wt_lora_pred' in hdr:
            return p
    raise FileNotFoundError(f'No p0 WT prediction CSV found for {dms} (seed {seed})')


def _seed_mean(vals):
    """Mean over the seeds of a per-assay rho (the seeds with a finite value; NaN when there is none)."""
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def homologous_domainome_domains(rule='tsu'):
    """Domainome domains homologous to the training set under the split's four-part rule (``msr_analysis.heldout``)."""
    from msr_analysis.heldout import excluded
    return excluded('domainome')


def tsuboyama_rule_assays():
    """ProteinGym assays (all categories) homologous to the training set under the split's four-part rule (``msr_analysis.heldout``)."""
    from msr_analysis.heldout import excluded
    return excluded('proteingym')


def _score_all_pgym_models(dms_ids, refresh=False):
    """Compute (or load cached) per-DMS Spearman rhos across all 213 assays: the mean over seeds 1-3 of the per-assay rho for every retrained model."""
    cache = CACHE / 'pg_all_models.csv'
    req_cols = set(EXT_ORDER) | (set(ALL_PG_MODELS) - {'pg:msrp0:comb'}) | {'has_multi', 'has_double', 'msr_p0_head', 'esm3_0_head'}
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0)
        if set(dms_ids) <= set(df.index) and req_cols <= set(df.columns):
            sub = df.reindex(dms_ids)
            mm_ids = sub.index[sub['has_multi'].astype(bool)]
            comb_match = all(
                (_find_p0_comb_csv(dms) is not None) == (sub.loc[dms, 'msr_p0_head'] == 'combined')
                for dms in mm_ids
            )
            if comb_match:
                return sub

    a = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
    pub = pd.read_csv(PGYM / 'pgym_public_per_dms.csv').set_index('DMS_id')
    ro = pd.read_csv(PGYM / 'pgym_rosetta_singles_per_dms.csv').set_index('DMS_id')
    pm = pd.read_csv(PGYM / 'proteinmpnn_020_unmasked_fast_ensemble3_per_dms.csv').set_index('DMS_id')
    tmpd_ref = pd.read_csv(REFERENCE / 'pgym_thermompnn_corrected_s1_per_dms.csv').set_index('DMS_id')

    def _sd(*parts, tail='proteingym'):
        """The directory of each seed: ``parts`` with ``seed{k}`` after the last one."""
        return {k: RETRAINED.joinpath(*parts, f'seed{k}', tail) for k in SEEDS}
    tmp_sa_dir = _sd('thermompnn', 'predictions', 'single_additive')
    tmp_d_dir = _sd('thermompnn', 'predictions', 'epistatic_corrected')              # per-variant files of the corrected ThermoMPNN-D run, seeds 1-3 (collected Nov 19); ``tmpd_ref`` (seed 1 table) is only the fallback
    me1_dir = _sd('mutate_everything', 'predictions', 'stage1')
    me2_dir = _sd('mutate_everything', 'predictions', 'stage2')
    sp_dir = _sd('spurs', 'predictions')
    msrdev_dir = {k: MSR / 'devel_dc8_floorlink_w2a4_m16a16' / f'seed{k}' / 'sigma1' / 'proteingym' for k in SEEDS}
    msrbase_dir = {k: MSR / 'baseline_w2a4_m16a16' / f'seed{k}' / 'sigma1' / 'proteingym' for k in SEEDS}
    esm3dg_dir = PUBLIC / 'esm3dg' / 'proteingym'
    esmtherm_dir = PUBLIC / 'esmtherm' / 'proteingym'
    geo_dir = PUBLIC / 'geoddg' / 'faspr' / 'proteingym'

    def _align_me2(dm2, dt_spine):
        m_ren = dm2['mut_info'].isin(set(dt_spine['mut_type_renumbered'])).mean()
        m_mut = dm2['mut_info'].isin(set(dt_spine['mutant'])).mean()
        kcol = 'mut_type_renumbered' if m_ren >= m_mut else 'mutant'
        s_map = dm2.drop_duplicates('mut_info').set_index('mut_info')['ddg_pred']
        v2 = -dt_spine[kcol].map(s_map).to_numpy(dtype=float)
        return float(spearmanr(v2, dt_spine['DMS_score'], nan_policy='omit')[0])

    def _score_rowid_dir(folder, dms, y_full):
        dp = pd.read_csv(folder / f'{dms}.csv', usecols=['row_id', 'pred'])
        rid = dp['row_id'].to_numpy(dtype=int)
        assert rid.min() >= 0 and rid.max() < len(y_full), f'{folder.name}/{dms}: row_id out of bounds'
        return float(spearmanr(dp['pred'], y_full[rid], nan_policy='omit')[0])

    def _rho_seeds(dirs, dms, pred, y='DMS_score'):
        """Per-seed rho of column ``pred`` of every seed's file of ``dms``; the seed mean (seeds without the file are skipped)."""
        out = []
        for k in SEEDS:
            f = dirs[k] / f'{dms}.csv'
            if f.exists():
                d = pd.read_csv(f, usecols=[pred, y])
                out.append(float(spearmanr(d[pred], d[y], nan_policy='omit')[0]))
        return _seed_mean(out)

    rows = {}
    for dms in dms_ids:
        dt = pd.read_csv(
            tmp_sa_dir[1] / f'{dms}.csv',
            usecols=['mut_type_renumbered', 'mutant', 'n_mut', 'pred_single_additive', 'DMS_score'],
        )
        y_full = dt['DMS_score'].to_numpy(dtype=float)
        has_mm = bool((dt['n_mut'] > 1).any())
        has_dbl = bool((dt['n_mut'] == 2).any())

        # ESM-MSR p0 (WT head + combined when available): the seed mean of the per-assay rho (seeds 1-3)
        rho_p0_wt = _seed_mean([float(spearmanr(*pd.read_csv(_find_p0_wt_csv(dms, k), usecols=['wt_lora_pred', 'DMS_score']).to_numpy().T, nan_policy='omit')[0]) for k in SEEDS])
        rc = []
        for k in SEEDS:
            p0_comb_path = _find_p0_comb_csv(dms, k)
            if p0_comb_path is None:
                continue
            hdr = set(pd.read_csv(p0_comb_path, nrows=0).columns)
            use = [c for c in ('combined_pred', 'wt_lora_pred', 'mt_lora_pred', 'DMS_score') if c in hdr]
            dp0_c = pd.read_csv(p0_comb_path, usecols=use)
            comb_vals = (
                dp0_c['combined_pred'].to_numpy(dtype=float)
                if 'combined_pred' in dp0_c
                else 0.5 * (dp0_c['wt_lora_pred'].to_numpy(dtype=float) + dp0_c['mt_lora_pred'].to_numpy(dtype=float))
            )
            rc.append(float(spearmanr(comb_vals, dp0_c['DMS_score'], nan_policy='omit')[0]))
        rho_p0_comb = _seed_mean(rc)

        if has_mm and np.isfinite(rho_p0_comb):
            rho_msr_uni = rho_p0_comb
            p0_head = 'combined'
        elif has_mm:
            rho_msr_uni = rho_p0_wt
            p0_head = 'wt_fallback'
        else:
            rho_msr_uni = rho_p0_wt
            p0_head = 'wt'

        # ThermoMPNN (additive) & ThermoMPNN + ThermoMPNN-D (corrected)
        # ThermoMPNN additive and ThermoMPNN + ThermoMPNN-D (composite: the epistatic network on the doubles, the single network elsewhere): the seed mean of the three retrains.
        # The corrected ThermoMPNN-D ProteinGym files of seeds 1-3 are in ``tmp_d_dir`` (seeds 2 and 3 were run on gpu4 on Nov 19); the seed-1 table ``tmpd_ref`` is the fallback for a seed without them.
        rho_tmp_sa = _rho_seeds(tmp_sa_dir, dms, 'pred_single_additive')
        d_seeds = []
        for k in SEEDS:
            f = tmp_d_dir[k] / f'{dms}.csv'
            if f.exists():
                dtd = pd.read_csv(f, usecols=['n_mut', 'pred_single_additive', 'pred_combined', 'DMS_score'])
                comb_t = np.where(dtd['n_mut'] == 2, dtd['pred_combined'], dtd['pred_single_additive'])
                d_seeds.append(float(spearmanr(comb_t, dtd['DMS_score'], nan_policy='omit')[0]))
        rho_tmp_d = _seed_mean(d_seeds) if d_seeds else float(tmpd_ref.loc[dms, 'rho_tmpnnd'])
        rho_tmp_uni = rho_tmp_d if has_dbl else rho_tmp_sa

        # MutateEverything st1 and st2 (all 213 assays)
        rho_me1 = _rho_seeds(me1_dir, dms, 'pred_combined')
        rho_me2 = _seed_mean([_align_me2(pd.read_csv(me2_dir[k] / f'{dms}.csv', usecols=['mut_info', 'ddg_pred']), dt) for k in SEEDS if (me2_dir[k] / f'{dms}.csv').exists()])
        rho_me_uni = rho_me2 if has_mm else rho_me1

        # SPURS single & multi (retrained; seed mean)
        rho_sp_sa = _rho_seeds(sp_dir, dms, 'SPURS_single_additive')
        rho_sp_m = _rho_seeds(sp_dir, dms, 'SPURS_multi')
        rho_sp_uni = rho_sp_m if has_mm else rho_sp_sa

        # ESM-MSR (devel) and (baseline), seed mean (kept for the full-roster heat map only)
        rho_dev_comb, rho_dev_wt = (_rho_seeds(msrdev_dir, dms, c) for c in ('combined_pred', 'wt_lora_pred'))
        rho_bas_comb, rho_bas_wt = (_rho_seeds(msrbase_dir, dms, c) for c in ('combined_pred', 'wt_lora_pred'))

        # Public row_id-indexed models
        rho_geo = _score_rowid_dir(geo_dir, dms, y_full)
        rho_e3dg = _score_rowid_dir(esm3dg_dir, dms, y_full)
        rho_etherm = _score_rowid_dir(esmtherm_dir, dms, y_full)

        rho_prosst = float(pub.loc[dms, 'ProSST-2048 rho_full'])
        rho_pmpnn = float(pm.loc[dms, 'mpnn_score_s0'])
        rho_e3_0_wt = float(a.loc[dms, 'ESM-MSR WT eps=0'])
        e3_file = ESM3_S0_COMB / f'{dms}.csv'
        if has_mm and e3_file.exists():                  # ESM3-small-open follows the same rule as the models it is compared with: combined on multi-mutant assays, WT head on the others
            de3 = pd.read_csv(e3_file, usecols=['combined_pred', 'DMS_score'])
            rho_e3_0, e3_head = float(spearmanr(de3['combined_pred'], de3['DMS_score'], nan_policy='omit')[0]), 'combined'
        else:
            rho_e3_0, e3_head = rho_e3_0_wt, 'wt'
        rho_ros = float(ro.loc[dms, 'rho']) if dms in ro.index else np.nan

        rows[dms] = {
            'has_multi': has_mm,
            'has_double': has_dbl,
            'msr_p0_head': p0_head,
            'esm3_0_head': e3_head,
            # Unified 9 models (for pg_categories / pg_per_dms_scatter)
            'ESM-MSR': rho_msr_uni,
            'MutateEverything': rho_me_uni,
            'ThermoMPNN': rho_tmp_uni,
            'SPURS': rho_sp_uni,
            'ESM3-small-open': rho_e3_0,
            'ProteinMPNN': rho_pmpnn,
            'ProSST': rho_prosst,
            'GeoDDG-3D': rho_geo,
            'Rosetta': rho_ros,
            # Full PGym model roster (for pg_table_heatmap & pg_overview)
            'pg:msrp0:comb': rho_p0_comb,
            'pg:msrp0:wt': rho_p0_wt,
            'pg:msrdev:comb': rho_dev_comb,
            'pg:msrdev:wt': rho_dev_wt,
            'pg:msr:comb': rho_bas_comb,
            'pg:msr:wt': rho_bas_wt,
            'pg:msrrel:comb': float(a.loc[dms, 'ESM-MSR full']),
            'pg:msrrel:wt': float(a.loc[dms, 'ESM-MSR WT eps=1.0']),
            'pg:me1': rho_me1,
            'pg:me2': rho_me2,
            'pg:tmpnn': rho_tmp_sa,
            'pg:tmpnnd': rho_tmp_d,
            'pg:spurs': rho_sp_sa,
            'pg:spursm': rho_sp_m,
            'pg:esm3dg': rho_e3dg,
            'pg:geoddg': rho_geo,
            'pg:rosetta': rho_ros,
            'pg:prosst': rho_prosst,
            'pg:esm3_pub': float(pub.loc[dms, 'ESM3 rho_full']),
            'pg:msrrel@0:wt': rho_e3_0_wt,
            'pg:pmpnn': rho_pmpnn,
            'pg:pmpnn_pub': float(pub.loc[dms, 'ProteinMPNN rho_full']),
            'pg:spurs_pub': float(a.loc[dms, 'SPURS single-additive']),
            'pg:spursm_pub': float(a.loc[dms, 'SPURS multi']),
            'pg:esmtherm': rho_etherm,
        }

    df = pd.DataFrame.from_dict(rows, orient='index')
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def _domainome_summary(refresh=False, rule='tsu'):
    """Mean per-domain Spearman rho and 95% domain-resampling bootstrap (B = 1000) across the held-out domains."""
    cache = CACHE / 'pg_domainome_tsu_boot.pkl'
    dom_hom = homologous_domainome_domains(rule)
    n_keep = 522 - len(dom_hom)
    if cache.exists() and not refresh:
        with open(cache, 'rb') as fh:
            cached = pickle.load(fh)
        if (
            isinstance(cached, dict)
            and cached.get('B') == B
            and cached.get('n_domains') == n_keep
            and set(cached.get('summary', {})) >= (set(EXT_ORDER) - {'Rosetta'}) | {'ESM-MSR σ=0.5'}
        ):
            return cached['summary']

    raw_dom = load_frame('domainome', heldout=False)               # all 522 domains: the final-run frame below has every row; the canonical four-part exclusion is applied through `keep`
    keep = ~raw_dom.spine['group'].astype(str).isin(dom_hom).to_numpy()
    dom_458 = Frame(
        'domainome',
        raw_dom.spine.loc[keep].reset_index(drop=True),
        {k: v[:, keep] for k, v in raw_dom.preds.items()},
    )
    dom_fr, _ = seed_frame('domainome', dom_458)
    s05 = load_final_frame('domainome', families=['FINAL_p0'], sigma='0.5', seeds=(1, 2, 3), heldout=False)             # FINAL_p0 seeds 1-3 at LoRA scale 0.5, WT head (the sigma = 1 reading is 'ESM-MSR': singles, so WT head)
    assert (s05.spine['id'].to_numpy() == raw_dom.spine['id'].to_numpy()).all()
    dom_fr.preds['ESM-MSR σ=0.5'] = s05.preds['FINAL_p0:wt'][:, keep]
    pt, rp = bootstrap_rho({'domainome': dom_fr}, list(dom_fr.preds), kind='protein', B=B, seed=30, per_protein=True)
    out = {}
    for m, (rho, _ps, _n, ng) in pt['domainome'].items():
        assert int(ng) == n_keep, f'Expected {n_keep} Domainome domains for {m}, got {ng}'
        lo, hi = ci(rp['domainome'][m])
        out[m] = (float(rho), float(lo), float(hi), int(ng))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(cache, 'wb') as fh:
        pickle.dump(dict(B=B, n_domains=n_keep, summary=out), fh, protocol=4)
    return out


def summarize_cols(T, models, idx_by_cat):
    """Value and NaN-safe bootstrap draws across the 5 categories, PG avg, and DMS mean."""
    res = {}
    for m in models:
        v = T[m]
        point = {c: float(np.nanmean(v[T.category == c])) for c in CATS}
        point['PG avg'] = float(np.mean([point[c] for c in CATS]))
        point['DMS mean'] = float(np.nanmean(v))
        reps = {}
        counts_cat = {}
        for c in CATS:
            vals = v[T.category == c].to_numpy(dtype=float)
            counts_cat[c] = int(np.isfinite(vals).sum())
            reps[c] = np.nanmean(vals[idx_by_cat[c]], axis=1)
        reps['PG avg'] = np.mean([reps[c] for c in CATS], axis=0)
        tot = sum(counts_cat.values())
        reps['DMS mean'] = sum(reps[c] * counts_cat[c] for c in CATS) / tot
        res[m] = (point, reps)
    return res


def summary_rows(T, models, cols):
    rng = np.random.default_rng(2)
    idx_by_cat = {
        c: rng.integers(0, int((T.category == c).sum()), (B, int((T.category == c).sum())))
        for c in CATS
    }
    S = summarize_cols(T, models, idx_by_cat)
    rows = []
    for m, (pt, reps) in S.items():
        for c in cols:
            lo, hi = np.nanquantile(reps[c], [0.025, 0.975])
            n_dms = int(np.isfinite(T.loc[T.category == c, m]).sum()) if c in CATS else int(np.isfinite(T[m]).sum())
            rows.append(dict(model=m, column=c, value=pt[c], lo=lo, hi=hi, n_dms=n_dms))
    return pd.DataFrame(rows), S


def main(refresh=False):
    register_ext_styles()
    MODELS['Rosetta']['label'] = 'Rosetta (singles)'
    for k, spec in PG_MODEL_STYLES.items():
        MODELS[k] = dict(spec)

    a = pd.read_csv(PGYM / 'pgym_all_models_per_dms.csv').set_index('DMS_id')
    a = a[~a.index.isin(DROP)]
    dms_ids = list(a.index)

    scores = _score_all_pgym_models(dms_ids, refresh=refresh)

    n_mm = int(scores['has_multi'].sum())
    n_p0_comb = int((scores['has_multi'] & (scores['msr_p0_head'] == 'combined')).sum())
    n_tmp_dbl = int(scores['has_double'].sum())
    print(f'Multi-mutant scoring status: ESM-MSR p0 combined on {n_p0_comb}/{n_mm} multi-mutant assays '
          f'(WT fallback on {n_mm - n_p0_comb}); ThermoMPNN-D on {n_tmp_dbl}/{n_mm} double-containing assays.')

    # Active full-PGym roster (include pg:msrp0:comb once multi-mutant combined predictions exist on all multi-mutant assays)
    if n_p0_comb == n_mm and n_mm > 0:
        # On singles-only assays, combined == WT head if combined pass wasn't run separately there
        s_mask = ~scores['has_multi'] & scores['pg:msrp0:comb'].isna()
        scores.loc[s_mask, 'pg:msrp0:comb'] = scores.loc[s_mask, 'pg:msrp0:wt']
    full_models = [m for m in ALL_PG_MODELS if int(scores[m].isna().sum()) <= 1]

    # Save unified + full-model per-DMS tables
    T = pd.DataFrame(index=dms_ids)
    T.index.name = 'DMS_id'
    T['category'] = a['category']
    for m in EXT_ORDER:
        T[m] = scores[m]
    T.to_csv(OUT / 'pg_per_dms.csv', float_format='%.4f')

    T_all = T[['category']].copy()
    for m in full_models:
        T_all[m] = scores[m]
    T_all.to_csv(OUT / 'pg_all_models_per_dms.csv', float_format='%.4f')

    non_ros = [m for m in EXT_ORDER if m != 'Rosetta']
    assert int(T[non_ros].isna().sum().sum()) == 0, f'Unexpected NaNs: {T[non_ros].isna().sum().to_dict()}'

    stab_hom = tsuboyama_rule_assays() & set(T.index)                    # every assay homologous to the training set (all categories), left out of every table below

    pg_cols = CATS + ['PG avg', 'DMS mean']
    dom = _domainome_summary(refresh=refresh)

    R_uni_213, _ = summary_rows(T, EXT_ORDER, pg_cols)
    R_uni_213.to_csv(OUT / 'pg_summary_all213.csv', index=False, float_format='%.4f')

    Tf = T[~T.index.isin(stab_hom)]
    R_uni_pg, _ = summary_rows(Tf, EXT_ORDER, pg_cols)
    dom_rows = pd.DataFrame([
        dict(model=m, column='Domainome', value=v[0], lo=v[1], hi=v[2], n_dms=v[3])
        for m, v in dom.items()
        if m in EXT_ORDER
    ])
    R_uni = pd.concat([R_uni_pg, dom_rows], ignore_index=True)
    R_uni.to_csv(OUT / 'pg_summary.csv', index=False, float_format='%.4f')

    # ESM-MSR p0 at sigma 1 (the deployed readout: combined on the multi-mutant assays, WT head elsewhere) and at sigma 0.5 (WT head on every assay: the only readout run at that scale), on the figure's assays and draws
    A = pd.read_csv(SIGMA_RHO).set_index('assay')
    Ts = Tf[['category']].copy()
    Ts['ESM-MSR σ=1'] = Tf['ESM-MSR']
    Ts['ESM-MSR σ=0.5'] = A['rho_dep_0.5'].reindex(Ts.index)               # combined on the multi-mutant assays whose sigma-0.5 combined run exists, the WT head on the others
    assert int(Ts[list(SIGMA_MODELS)].isna().sum().sum()) == 0
    R_sig, _ = summary_rows(Ts, list(SIGMA_MODELS), pg_cols)
    dom_sig = pd.DataFrame([dict(model='ESM-MSR σ=1', column='Domainome', value=dom['ESM-MSR'][0], lo=dom['ESM-MSR'][1], hi=dom['ESM-MSR'][2], n_dms=dom['ESM-MSR'][3]),
                            dict(model='ESM-MSR σ=0.5', column='Domainome', value=dom['ESM-MSR σ=0.5'][0], lo=dom['ESM-MSR σ=0.5'][1], hi=dom['ESM-MSR σ=0.5'][2], n_dms=dom['ESM-MSR σ=0.5'][3])])
    pd.concat([R_sig, dom_sig], ignore_index=True).to_csv(OUT / 'pg_sigma_summary.csv', index=False, float_format='%.4f')
    mm = A[(A.source == 'ProteinGym') & A.has_multi.astype(bool)]
    k05 = int((mm['dep_head_0.5'] == 'combined').sum())
    (OUT / 'pg_sigma_note.txt').write_text(f'{k05} of {len(mm)}\n')

    # the same tables with every protein homologous to the training set under the split's own rule left out (all categories; Domainome likewise): pg_summary_tsu.csv, pg_sigma_summary_tsu.csv
    tsu_assays = tsuboyama_rule_assays() & set(T.index)
    Tt = T[~T.index.isin(tsu_assays)]
    dom_t = _domainome_summary(refresh=refresh, rule='tsu')
    R_t, _ = summary_rows(Tt, EXT_ORDER, pg_cols)
    pd.concat([R_t, pd.DataFrame([dict(model=m, column='Domainome', value=v[0], lo=v[1], hi=v[2], n_dms=v[3]) for m, v in dom_t.items() if m in EXT_ORDER])], ignore_index=True).to_csv(OUT / 'pg_summary_tsu.csv', index=False, float_format='%.4f')
    Tst = Tt[['category']].copy()
    Tst['ESM-MSR σ=1'] = Tt['ESM-MSR']
    Tst['ESM-MSR σ=0.5'] = A['rho_dep_0.5'].reindex(Tst.index)
    R_st, _ = summary_rows(Tst, list(SIGMA_MODELS), pg_cols)
    pd.concat([R_st, pd.DataFrame([dict(model='ESM-MSR σ=1', column='Domainome', value=dom_t['ESM-MSR'][0], lo=dom_t['ESM-MSR'][1], hi=dom_t['ESM-MSR'][2], n_dms=dom_t['ESM-MSR'][3]),
                                   dict(model='ESM-MSR σ=0.5', column='Domainome', value=dom_t['ESM-MSR σ=0.5'][0], lo=dom_t['ESM-MSR σ=0.5'][1], hi=dom_t['ESM-MSR σ=0.5'][2], n_dms=dom_t['ESM-MSR σ=0.5'][3])])],
              ignore_index=True).to_csv(OUT / 'pg_sigma_summary_tsu.csv', index=False, float_format='%.4f')
    cnt = T.loc[sorted(tsu_assays)].category.value_counts().to_dict()
    (OUT / 'pg_tsu_note.txt').write_text(f'{len(tsu_assays)} of {len(T)} ProteinGym assays ({", ".join(f"{k} {v}" for k, v in cnt.items())}) and {522 - dom_t["ESM-MSR"][3]} of 522 Domainome domains\n')
    print('Tsuboyama-rule exclusions:', (OUT / 'pg_tsu_note.txt').read_text().strip())

    Tf_all = T_all[~T_all.index.isin(stab_hom)]
    R_full_213, _ = summary_rows(T_all, full_models, pg_cols)
    R_full_213.to_csv(OUT / 'pg_all_models_summary_all213.csv', index=False, float_format='%.4f')
    R_full, _ = summary_rows(Tf_all, full_models, pg_cols)
    R_full.to_csv(OUT / 'pg_all_models_summary.csv', index=False, float_format='%.4f')

    print(f'All: {len(T)} assays; held-out under the four-part rule: {len(Tf)} assays '
          f'({len(stab_hom)} homologous assays removed; remaining Stability={(Tf.category == "Stability").sum()})')
    print(
        R_uni[R_uni.column.isin(['PG avg', 'Stability', 'DMS mean', 'Domainome'])]
        .pivot(index='model', columns='column', values='value')[['Stability', 'PG avg', 'DMS mean', 'Domainome']]
        .reindex(EXT_ORDER)
        .round(3)
        .to_string()
    )

    p0_desc = 'p0 combined on multi-mutants' if n_p0_comb == n_mm else f'p0 WT head ({n_p0_comb}/{n_mm} multi-mutants combined)'
    stab_note = (
        f'* Proteins homologous to the training set under the split\'s four-part rule (MMseqs2 E <= 1e-3, identity > 0.40, Foldseek E <= 1e-3, TM-score > 0.70) are left out: {len(stab_hom)} of {len(T)} ProteinGym assays and '
        f'{len(homologous_domainome_domains())} of 522 Domainome domains.\n'
        f'Seeds 1-3 averaged Singles models on singles assays / Domainome; multi on PGym multi-mutants '
        f'(ME st1/st2; ThermoMPNN additive / -D on doubles; SPURS single/multi). '
        'Rosetta: PGym singles only (no Domainome).'
    )

    # ---- figure 1 (pg_categories) moved to 01_summary.py (it also carries the MGnify column); the footnote travels as a text file ----
    (OUT / 'pg_categories_note.txt').write_text(stab_note + '\n')

    # ---- figure 2: ranked heatmap table across ALL models with PGym data ----------------------------
    Z = R_full.pivot(index='model', columns='column', values='value')[pg_cols]
    Z = Z.loc[Z['PG avg'].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9.8, 0.32 * len(Z) + 2.0))
    rk = Z.rank(pct=True)
    ax.imshow(rk.to_numpy(), cmap='YlGnBu', vmin=0, vmax=1.15, aspect='auto')
    for i in range(len(Z)):
        for j, _c in enumerate(pg_cols):
            ax.text(
                j, i, f'{Z.iloc[i, j]:.3f}', ha='center', va='center', fontsize=7.6,
                color='white' if rk.iloc[i, j] > 0.75 else 'black',
            )
    ax.set_xticks(range(len(pg_cols)))
    ax.set_xticklabels([CAT_LABEL[c].replace('\n', ' ') for c in pg_cols], rotation=28, ha='right', fontsize=8.8)
    ax.set_yticks(range(len(Z)))
    ax.set_yticklabels([MODELS[m]['label'] for m in Z.index], fontsize=8.4)
    ax.set_title(
        'ProteinGym summary across all models with PGym data (seeds 1-3 averaged; homologous proteins excluded): colour = rank in column',
        loc='left', fontsize=9.3,
    )
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    fig.text(
        0.01, 0.008,
        '* Stability* keeps 40/66 assays (26 P+H removed); ThermoMPNN + ThermoMPNN-D uses the corrected run (seeds 1-3). '
        'Models marked * in label (ESMtherm*, SPURS public ckpts*) trained directly on unsplit MegaScale libraries.',
        fontsize=7.1, va='bottom', ha='left', color='#444444',
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    style.save(fig, OUT / 'pg_table_heatmap')

    # ---- figure 3: tsuboyama_overview-style horizontal per-category forest plot (all PGym models) ---
    P_full = {(r.column, r.model): (r.value, r.lo, r.hi) for r in R_full.itertuples()}
    mega_keys = sorted([k for k in full_models if k in MEGA_PG], key=lambda k: P_full[('PG avg', k)][0])
    held_keys = sorted([k for k in full_models if k not in MEGA_PG], key=lambda k: P_full[('PG avg', k)][0])
    ov_keys = mega_keys + held_keys

    fig, axes = plt.subplots(1, len(CATS), figsize=(17.2, 0.31 * len(ov_keys) + 1.8), sharey=True)
    for ax, cat in zip(axes, CATS):
        panel_vals = [P_full[(cat, k)] for k in ov_keys if (cat, k) in P_full]
        min_lo = min(lo for _v, lo, _hi in panel_vals)
        max_hi = max(hi for _v, _lo, hi in panel_vals)
        x_lo = float(np.floor((min_lo - 0.03) * 20) / 20)
        span = max(0.25, max_hi - x_lo)
        x_hi = max_hi + 0.23 * span

        for i, k in enumerate(ov_keys):
            v, lo, hi = P_full[(cat, k)]
            kw = style.marker_kwargs(k, size=6.0)
            ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.4, alpha=0.45, solid_capstyle='butt')
            ax.plot(v, i, **kw)
            if cat == 'Stability' and k in PG_STAB_LEAK:
                ax.plot(v, i, marker='o', ms=10.5, mfc='none', mec='#c0392b', mew=1.0, ls='none')
            ax.text(hi + 0.022 * span, i, f'{v:.3f}', va='center', fontsize=7.3, color='#444')
        ax.axhspan(-0.5, len(mega_keys) - 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
        ax.set_title(OV_TITLES[cat], loc='left', fontsize=9.8)
        ax.grid(axis='x', color=style.GRID, lw=0.6)
        ax.set_xlabel('mean per-assay ρ', fontsize=8.8)
        ax.set_xlim(x_lo, x_hi)

    axes[0].set_yticks(range(len(ov_keys)))
    axes[0].set_yticklabels([MODELS[k]['label'] for k in ov_keys], fontsize=8.3)
    axes[0].set_ylim(-0.6, len(ov_keys) - 0.4)
    for t, k in zip(axes[0].get_yticklabels(), ov_keys):
        t.set_color({'msr': 'k', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555'}[MODELS[k]['group']])
        if MODELS[k]['group'] == 'msr':
            t.set_fontweight('bold')
    fig.suptitle(
        'ProteinGym per-category overview (seeds 1-3 averaged; sorted by PG avg*): thick bar = 95% assay bootstrap; red ring = Stability training/homolog overlap\n'
        '* Stability excludes 26 MegaScale-homologous assays (40/66 kept); grey rows at bottom: checkpoints trained on unsplit MegaScale libraries',
        fontsize=8.8, y=1.01,
    )
    fig.tight_layout()
    style.save(fig, OUT / 'pg_overview')

    # ---- figure 4: per-assay scatter (ESM-MSR p0 seed mean vs ProSST & vs ESM3-small-open sigma=0) -----
    colors = dict(zip(CATS, ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']))
    cat_leg = {
        'Activity': 'Activity (43)',
        'Binding': 'Binding (13)',
        'Expression': 'Expression (18)',
        'OrganismalFitness': 'Org. fitness (73)',
        'Stability': 'Stability* (40/66)',
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, (xm, xl) in zip(axes, [('ProSST', 'ProSST (zero-shot)'), ('ESM3-small-open', 'ESM3-small-open (σ = 0)')]):
        for c in CATS:
            s = Tf[Tf.category == c]
            ax.scatter(s[xm], s['ESM-MSR'], s=18, color=colors[c], alpha=0.78, lw=0, label=cat_leg[c])
        lim = [-0.2, 1.0]
        ax.plot(lim, lim, color='#999', lw=0.8, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'ρ, {xl}')
        ax.set_ylabel('ρ, ESM-MSR (p0, seeds 1-3)')
        ax.text(0.03, 0.97, f'ESM-MSR higher on {int((Tf["ESM-MSR"] > Tf[xm]).sum())} of {len(Tf)} assays', transform=ax.transAxes, va='top', fontsize=8.5)
    axes[0].legend(fontsize=7.8, loc='lower right')
    fig.suptitle('ProteinGym per-assay ρ (187 assays; homologous Stability proteins excluded)', fontsize=10, y=1.01)
    fig.tight_layout()
    style.save(fig, OUT / 'pg_per_dms_scatter')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true', help='Recompute the per-DMS (seed-mean) & Domainome caches')
    args = ap.parse_args()
    main(refresh=args.refresh)
