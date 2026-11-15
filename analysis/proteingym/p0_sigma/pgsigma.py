"""Shared paths, loaders and plot style for the p0 sigma analysis of ProteinGym (sigma = 1, 0.5 from FINAL_p0 seed 1, sigma = 0 from the unmasked ESM3-small-open read)."""
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
PG = {1.0: ROOT / 'final_pgym_wt' / 'FINAL_p0' / 's1',                # FINAL_p0 seed 1, WT head, one file per assay
      0.5: ROOT / 'final_pgym_wt_sigma0.5' / 'FINAL_p0' / 's1',
      0.0: ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only'}   # ESM3-small-open unmasked (adapter scaled to 0)
FALLBACK = {1.0: ROOT / 'final_pgym_comb' / 'FINAL_p0' / 's1',       # same checkpoint and read: its wt_lora_pred equals the WT-only run's to the last digit (checked on 6 assays; used for BRCA2 at sigma 1)
            0.0: ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'}   # an older ESM-MSR-small run at sigma 0: tracks the released sigma-0 read closely (per-assay rho differs by 0.002 on average, 0.02 at most, over 214 assays) but is not identical; used only where the released read is missing (BRCA2, POLG_CXB3N, SCN5A)
DMS_P0 = ROOT / 'final_bench' / 'predictions'
DMS_S0 = ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0'
DMS_SETS = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']     # the DMS-benchmark sets that are not in ProteinGym

# the look of the earlier sigma analysis (sigma_tradeoff/scripts/viz.py)
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8982', '#e4e3de'
CAT_COLOR = {'Stability': '#6250d6', 'Expression': '#2a78d6', 'Activity': '#eb6834', 'Binding': '#1baf7a', 'OrganismalFitness': '#e87ba4'}
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
    """The ProteinGym assays that overlap the ESM-MSR training data (exact or constituent overlap with the capped MegaScale inventory, ``sameprot`` in the leak audit): 23 Tsuboyama stability assays.
    Every figure of this folder leaves them out."""
    t = pd.read_csv(LEAK_TABLE)
    return set(t.loc[t.sameprot == 1, 'DMS'])


def per_assay(source='ProteinGym'):
    """``results/per_assay_rho.csv`` without the assays that overlap the training data (the DMS-benchmark rows are kept when ``source`` is None)."""
    t = pd.read_csv(OUT / 'per_assay_rho.csv')
    t = t[~t.assay.isin(train_overlap())]
    return t if source is None else t[t.source == source].copy()


def rho(pred, y):
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(spearmanr(pred[ok], y[ok])[0]), int(ok.sum())


def load_assay(name):
    """DataFrame (key, y, s0.0, s0.5, s1.0) of the variants of one assay present in the predictions of all three sigmas, or None if one is missing.  ProteinGym assays are keyed by ``mutant``,
    the DMS-benchmark sets by ``id`` (label column ``ddG_ML``, the repo's ground truth)."""
    if name in DMS_SETS:
        files = {1.0: DMS_P0 / name / 'final_checkpoints' / 'FINAL_p0' / 's1' / 'best.ckpt_sigma1.0_unmasked_predictions.csv',
                 0.5: DMS_P0 / name / 'final_checkpoints' / 'FINAL_p0' / 's1' / 'best.ckpt_sigma0.5_unmasked_predictions.csv', 0.0: DMS_S0 / f'{name}.csv'}
        key, lab = 'id', 'ddG_ML'
    else:
        files = {s: (PG[s] if (PG[s] / f'{name}.csv').exists() or s not in FALLBACK else FALLBACK[s]) / f'{name}.csv' for s in SIGMAS}
        key, lab = 'mutant', 'DMS_score'
    if not all(f.exists() for f in files.values()):
        return None
    out = None
    for s, f in files.items():
        d = pd.read_csv(f, usecols=[key, lab, 'wt_lora_pred']).drop_duplicates(key).rename(columns={'wt_lora_pred': f's{s}', lab: 'y'}).set_index(key)
        out = d if out is None else out.join(d[[f's{s}']], how='inner')
    return out.reset_index(drop=True)[['y'] + [f's{s}' for s in SIGMAS]]
