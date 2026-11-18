"""Shared definitions of the ablation-versus-p0 analysis: the families, the set lists, the look of the figures.

Every FINAL_* family differs from FINAL_p0 (the ESM-MSR recipe) in one hparam group (``msr_analysis.final_runs.FAMILIES``).  Each is compared with p0 under the same readout (WT head, combined) on
the same variants; the comparisons are paired.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ANALYSIS / 'combined_vs_wt'))
sys.path.insert(0, str(ANALYSIS / 'proteingym' / 'p0_sigma'))
import cvw as C  # noqa: E402,F401
from msr_analysis.final_runs import FAMILIES  # noqa: E402,F401

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

REF = 'FINAL_p0'
ABL = ['FINAL_p1', 'FINAL_p1_marg', 'FINAL_p0_pw0', 'FINAL_p0_norank', 'FINAL_p0_rcthr0', 'FINAL_p0_noseqhwm', 'FINAL_p0_r2a4', 'FINAL_p0_mega']
HEADS = ['wt', 'comb']
LABEL = {'FINAL_p1': 'p1: masked WT and MT passes', 'FINAL_p1_marg': 'p1 + MT marginal', 'FINAL_p0_pw0': 'pw0: no pair loss', 'FINAL_p0_norank': 'no rank loss',
         'FINAL_p0_rcthr0': 'rank censor threshold 0', 'FINAL_p0_noseqhwm': 'no sequence head', 'FINAL_p0_r2a4': 'MT LoRA rank 2', 'FINAL_p0_mega': 'full MegaScale (not homology-restricted)'}
SHORT = {'FINAL_p1': 'p1', 'FINAL_p1_marg': 'p1 marg.', 'FINAL_p0_pw0': 'pw0', 'FINAL_p0_norank': 'no rank', 'FINAL_p0_rcthr0': 'rank thr 0', 'FINAL_p0_noseqhwm': 'no seq. head',
         'FINAL_p0_r2a4': 'MT r2', 'FINAL_p0_mega': 'full Mega'}
GROUP = {'FINAL_p1': 'masking', 'FINAL_p1_marg': 'masking', 'FINAL_p0_pw0': 'loss', 'FINAL_p0_norank': 'loss', 'FINAL_p0_rcthr0': 'loss', 'FINAL_p0_noseqhwm': 'architecture',
         'FINAL_p0_r2a4': 'architecture', 'FINAL_p0_mega': 'data'}
GROUP_COLOR = {'masking': '#c0392b', 'loss': '#2f7fc1', 'architecture': '#2e8b57', 'data': '#8e44ad'}
COLOR = {f: GROUP_COLOR[GROUP[f]] for f in ABL}
MARKER = {'FINAL_p1': 'D', 'FINAL_p1_marg': 'd', 'FINAL_p0_pw0': 's', 'FINAL_p0_norank': 'v', 'FINAL_p0_rcthr0': '^', 'FINAL_p0_noseqhwm': 'o', 'FINAL_p0_r2a4': 'P', 'FINAL_p0_mega': 'X'}


# ---- the columns of the figures ----------------------------------------------------------------------
import pandas as pd  # noqa: E402

# (column group, [(group, set, subset, short label, head override)]); 'wt' / 'comb' heads of the cell, except the epistasis column (head 'epi', combined only)
def column_spec():
    ext = [('ext_singles', C.LABEL[d], 'all', C.LABEL[d], None) for d in C.EXT_SINGLE] + [('ext_singles', 'ext_singles_composite', 'all', 'composite', None)]
    mul = [('ext_multi', C.LABEL[d], 'all', C.LABEL[d], None) for d in C.EXT_MULTI] + [('ext_multi', 'ext_multi_composite', 'all', 'composite', None)]
    tsu = [('tsuboyama', C.LABEL['tsuboyama_capped_test'], 'single mutants', 'test singles', None), ('tsuboyama', C.LABEL['tsuboyama_capped_test'], 'doubles', 'test doubles', None),
           ('tsuboyama', C.LABEL['tsuboyama_capped_test'], 'epistasis (dddG)', 'test epistasis', 'epi'), ('tsuboyama', C.LABEL['tsuboyama_capped_val'], 'single mutants', 'val singles', None),
           ('tsuboyama', C.LABEL['tsuboyama_capped_val'], 'doubles', 'val doubles', None)]
    dom = [('domainome', 'Domainome', 'all', 'Domainome', None)]
    dms = [('dms', C.LABEL[d], 'all', C.LABEL[d], None) for d in C.DMS] + [('dms', 'dms_composite', 'all', 'composite', None)]
    pg = [('proteingym', c, 'all', {'OrganismalFitness': 'Fitness'}.get(c, C.CAT_LABEL[c].split(' /')[0]), None) for c in C.CATS] + [('proteingym', 'PG avg', 'all', 'average', None)]
    return [('Thermostability,\nsingle-mutant sets', ext), ('Thermostability,\nmulti-mutant sets', mul), ('Homology-Restricted\nMegaScale', tsu), ('Domainome', dom), ('DMS benchmark', dms),
            ('ProteinGym\n(assay mean)', pg)]


def load_all():
    a = pd.read_csv(OUT / 'paired_vs_p0.csv')
    b = pd.read_csv(OUT / 'paired_vs_p0_pg.csv')
    return pd.concat([a, b.drop(columns=['p_holm'])], ignore_index=True)


def cell(D, spec, fam, head):
    """The row of D for one column spec, family and head (None when absent)."""
    g, s, sub, _, ho = spec
    r = D[(D.group == g) & (D.set == s) & (D.subset == sub) & (D.family == fam) & (D['head'] == (ho or head))]
    return r.iloc[0] if len(r) else None
