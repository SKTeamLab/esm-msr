"""Shared paths, loaders and plot style for the p0 sigma analysis of ProteinGym (sigma = 1, 0.5 from FINAL_p0 seeds 1-3, sigma = 0 from the unmasked ESM3-small-open read, a single deterministic run)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))                          # analysis/
from msr_analysis.registry import ROOT  # noqa: E402

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
LEAK_TABLE = HERE.parent.parent / 'reference' / 'leak_audit_pg_per_dms.csv'    # per-assay overlap of the ProteinGym assays with the ESM-MSR training inventory (leak_audit/17_proteingym.py)
SIGMAS = [0.0, 0.5, 1.0]
SEEDS = (1, 2, 3)                                                       # FINAL_p0 is a triplicate; the dicts below are the seed-1 folders, ``at_seed`` gives the others (sigma 0 is one deterministic run: unchanged)


def at_seed(path, seed):
    return path.with_name(f's{seed}') if path.name == 's1' else path
PG = {1.0: ROOT / 'final_pgym_wt' / 'FINAL_p0' / 's1',                # FINAL_p0 (s1 here, at_seed for s2, s3), WT head, one file per assay
      0.5: ROOT / 'final_pgym_wt_sigma0.5' / 'FINAL_p0' / 's1',
      0.0: ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only'}   # ESM3-small-open unmasked (adapter scaled to 0)
FALLBACK = {1.0: ROOT / 'final_pgym_comb' / 'FINAL_p0' / 's1',       # same checkpoint and read: its wt_lora_pred equals the WT-only run's to the last digit (checked on 6 assays; used for BRCA2 at sigma 1)
            0.0: ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'}   # an older ESM-MSR-small run at sigma 0: tracks the released sigma-0 read closely (per-assay rho differs by 0.002 on average, 0.02 at most, over 214 assays) but is not identical; used only where the released read is missing (BRCA2, POLG_CXB3N, SCN5A)
COMB = {1.0: ROOT / 'final_pgym_comb' / 'FINAL_p0' / 's1',            # combined (1/2 (MT + WT)) and WT head, every assay
        0.5: ROOT / 'final_pgym_comb_sigma0.5' / 'FINAL_p0' / 's1',   # queued 2026-11-17 (tier pgym_comb_s05): the 69 multi-mutant assays only; assays missing here read the WT head
        0.0: ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'}   # the March ESM-MSR-small checkpoint at sigma 0 (adapters off, unmasked; pgym_results/FINDINGS.md section 14): the zero-shot combined read, with that checkpoint's calibration heads
DMS_P0 = ROOT / 'final_bench' / 'predictions'
DMS_S0 = ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0'
DMS_SETS = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']     # the DMS-benchmark sets that are not in ProteinGym

# the look of the earlier sigma analysis (sigma_tradeoff/scripts/viz.py)
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8982', '#e4e3de'
from msr_analysis.models import CATEGORY_COLOR as CAT_COLOR  # noqa: E402   one colour per ProteinGym category in every figure
CAT_LABEL = {'Stability': 'Stability', 'Expression': 'Expression / abundance', 'Activity': 'Activity', 'Binding': 'Binding', 'OrganismalFitness': 'Organismal fitness'}
CATS = ['Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness']
AXIS_COLOR = {'ABUND': '#2a78d6', 'ACT': '#eb6834', 'SPEC': '#8a6f3b', 'FIT': '#e87ba4', 'BIND': '#1baf7a'}
AXIS_NAME = {'ABUND': 'abundance / expression', 'ACT': 'activity / function', 'BIND': 'binding', 'FIT': 'growth / fitness', 'SPEC': 'drug resistance / Hsp90 dep.'}
RC = {'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
      'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True, 'font.size': 9.5, 'legend.frameon': False, 'lines.linewidth': 2, 'lines.markersize': 6,
      'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'}


def fallback_sigmas(name):
    """The sigmas of a ProteinGym assay that are read from ``FALLBACK`` because the primary prediction file does not exist (empty for the DMS-benchmark sets)."""
    if name in DMS_SETS:
        return []
    return [s for s in SIGMAS if s in FALLBACK and not (PG[s] / f'{name}.csv').exists() and (FALLBACK[s] / f'{name}.csv').exists()]


def train_overlap():
    """The ProteinGym assays that are homologous to the training set under the split's four-part rule (MMseqs2 E <= 1e-3, identity > 0.40, Foldseek E <= 1e-3, Foldseek TM > 0.70 against the capped
    training libraries; ``msr_analysis.heldout`` / ``leak_audit/21_tsuboyama_rule.py``).  Every figure of this folder leaves them out."""
    from msr_analysis.heldout import excluded
    return excluded('proteingym')


def per_assay(source='ProteinGym'):
    """``results/per_assay_rho.csv`` without the assays that overlap the training data (the DMS-benchmark rows are kept when ``source`` is None), with the column ``leaked`` of the per-assay leak test."""
    t = pd.read_csv(OUT / 'per_assay_rho.csv')
    t = t[~t.assay.isin(train_overlap())]
    lk = OUT / 'leak_per_assay.csv'                                        # 04_leak_per_assay.py: TM > 0.7 or sequence identity > 0.4 to a training protein
    t['leaked'] = t.assay.map(pd.read_csv(lk).set_index('assay').leaked).astype('boolean').fillna(False).astype(bool) if lk.exists() else False
    return t if source is None else t[t.source == source].copy()


def per_assay_all(source='ProteinGym'):
    """Every assay of ``results/per_assay_rho.csv`` (also the ones homologous to the training set), with ``leaked`` = the assay is excluded by the split's four-part rule (``train_overlap``): the figures that draw
    these assays black and leave them out of their statistics (``03_fig_property_summary.draw_paired(mark_leaks=True, stats_without_leaks=True)``) read it."""
    t = pd.read_csv(OUT / 'per_assay_rho.csv')
    t['leaked'] = t.assay.isin(train_overlap())
    return t if source is None else t[t.source == source].copy()


def rho(pred, y):
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(spearmanr(pred[ok], y[ok])[0]), int(ok.sum())


def load_assay(name, seed=1):
    """DataFrame (key, y, s0.0, s0.5, s1.0) of the variants of one assay present in the predictions of all three sigmas (FINAL_p0 seed ``seed``), or None if one is missing.  ProteinGym assays are keyed by ``mutant``,
    the DMS-benchmark sets by ``id`` (label column ``ddG_ML``, the repo's ground truth)."""
    if name in DMS_SETS:
        files = {1.0: DMS_P0 / name / 'final_checkpoints' / 'FINAL_p0' / f's{seed}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv',
                 0.5: DMS_P0 / name / 'final_checkpoints' / 'FINAL_p0' / f's{seed}' / 'best.ckpt_sigma0.5_unmasked_predictions.csv', 0.0: DMS_S0 / f'{name}.csv'}
        key, lab = 'id', 'ddG_ML'
    else:
        files = {s: at_seed(PG[s] if (PG[s] / f'{name}.csv').exists() or s not in FALLBACK else FALLBACK[s], seed) / f'{name}.csv' for s in SIGMAS}
        key, lab = 'mutant', 'DMS_score'
    if not all(f.exists() for f in files.values()):
        return None
    out = None
    for s, f in files.items():
        d = pd.read_csv(f, usecols=[key, lab, 'wt_lora_pred']).drop_duplicates(key).rename(columns={'wt_lora_pred': f's{s}', lab: 'y'}).set_index(key)
        out = d if out is None else out.join(d[[f's{s}']], how='inner')
    return out.reset_index(drop=True)[['y'] + [f's{s}' for s in SIGMAS]]


def load_assay_dep(name, seed=1):
    """The deployed readout of one ProteinGym assay at every sigma: ``(DataFrame, has_multi, heads)``.  The frame has ``y`` and ``d<sigma>`` (the combined prediction on an assay with multi-mutants, the WT head
    on a singles-only assay; a multi-mutant assay without a combined file at that sigma reads the WT head) plus ``w<sigma>`` (the WT head), over the variants present in every file used.  ``heads[sigma]`` is
    'combined', 'wt' (singles-only assay) or 'wt_fallback'.  None when the WT predictions of a sigma are missing."""
    d = load_assay_frame(name, seed)
    if d is None:
        return None
    multi = d['mutant'].astype(str).str.contains(':').any()
    heads = {}
    for s_ in SIGMAS:
        f = at_seed(COMB[s_], seed) / f'{name}.csv'
        comb = None
        if multi and f.exists() and 'combined_pred' in pd.read_csv(f, nrows=0).columns:
            comb = pd.read_csv(f, usecols=['mutant', 'combined_pred']).drop_duplicates('mutant').set_index('mutant')['combined_pred'].rename(f'c{s_}')
        if comb is not None:
            d = d.join(comb, on='mutant', how='inner')
            d[f'd{s_}'] = d[f'c{s_}']
            heads[s_] = 'combined'
        else:
            d[f'd{s_}'] = d[f'w{s_}']
            heads[s_] = 'wt_fallback' if multi else 'wt'
    return d, bool(multi), heads


def load_assay_frame(name, seed=1):
    """``mutant``, ``y`` and the WT head ``w<sigma>`` of a ProteinGym assay over the variants present at all three sigmas (``load_assay`` with the mutant column kept); None if a file is missing."""
    files = {s_: at_seed(PG[s_] if (PG[s_] / f'{name}.csv').exists() or s_ not in FALLBACK else FALLBACK[s_], seed) / f'{name}.csv' for s_ in SIGMAS}
    if not all(f.exists() for f in files.values()):
        return None
    out = None
    for s_, f in files.items():
        d = pd.read_csv(f, usecols=['mutant', 'DMS_score', 'wt_lora_pred']).drop_duplicates('mutant').rename(columns={'wt_lora_pred': f'w{s_}', 'DMS_score': 'y'}).set_index('mutant')
        out = d if out is None else out.join(d[[f'w{s_}']], how='inner')
    return out.reset_index()

