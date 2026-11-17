#!/usr/bin/env python
"""Task tables for the mega_splits Tsuboyama sets (``_tasks/tsuboyama_mega_{val,test}.csv.gz``), in the layout of the capped ones.

mega_splits has no harness task table; its labels are the columns of the testing harness's prediction file for FINAL_p0_mega
(the same rows and labels for every seed, checked here). Added as in the capped tables: ``truth`` = ``ddG_ML``, ``group`` =
``code``, ``ref_seq`` = the backbone sequence of the library (each row's ``aa_seq`` with its mutations reverted; checked to be one
sequence per library), ``backbone_mutation`` = ``mut_structure`` on mutant-backbone libraries, ``dataset`` and ``row_id``.

    python tsuboyama/mega_splits/00_build_tasks.py [--check-capped]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from msr_analysis.registry import P0, TASKS  # noqa: E402

FOLDER = {'tsuboyama_mega_val': 'mega_splits-val', 'tsuboyama_mega_test': 'mega_splits-test'}
PRED_COLS = ('wt_lora_pred', 'mt_lora_pred', 'combined_pred', 'combined_dddg_pred', 'epi_pred', 'mt_lora_dddg_pred', 'wt_lora_pred_additive',
             'wt_lora_dddg_pred', 'mt_lora_pred_additive', 'combined_pred_additive', 'mut_type_renumbered', 'mut_type_pdb')


def ref_seqs(t):
    """{code: backbone sequence}: every row's aa_seq with its own mutations reverted; one sequence per code or an error."""
    out = {}
    for code, g in t.groupby('code'):
        seqs = set()
        for seq, mt in zip(g['aa_seq'].astype(str), g['mut_type'].astype(str)):
            s = list(seq)
            for m in mt.split(':'):
                wt, pos = m[0], int(m[1:-1])
                if s[pos - 1] != m[-1]:
                    raise SystemExit(f'{code} {mt}: position {pos} is {s[pos - 1]}, not the mutant {m[-1]}')
                s[pos - 1] = wt
            seqs.add(''.join(s))
        if len(seqs) != 1:
            raise SystemExit(f'{code}: {len(seqs)} backbone sequences')
        out[code] = seqs.pop()
    return out


def build(ds):
    files = [P0 / FOLDER[ds] / 'final_checkpoints' / 'FINAL_p0_mega' / f's{s}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv' for s in (1, 2, 3)]
    tabs = [pd.read_csv(f, low_memory=False) for f in files]
    t = tabs[0].drop(columns=[c for c in PRED_COLS if c in tabs[0]])
    for o in tabs[1:]:
        o = o[t.columns]
        same = (o['id'].equals(t['id']) and np.allclose(o['ddG_ML'], t['ddG_ML'], equal_nan=True) and o['cens'].equals(t['cens']))
        if not same:
            raise SystemExit(f'{ds}: the seeds disagree on rows or labels')
    t['ref_seq'] = t['code'].map(ref_seqs(t))
    t['backbone_mutation'] = t['mut_structure'].where(t['mut_structure'] != '-')
    t['group'] = t['code']
    t['truth'] = t['ddG_ML']
    t['seq_mismatch'] = 0
    t['dataset'] = FOLDER[ds]
    t['row_id'] = np.arange(len(t))
    dst = TASKS / f'{ds}.csv.gz'
    t.to_csv(dst, index=False)
    print(f'{ds}: {len(t):,} rows ({(t.cens == 0).sum():,} measured), {t.code_wt.nunique()} libraries, {t.code.nunique()} backbones -> {dst}')


def check_capped():
    t = pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', low_memory=False)
    r = t['code'].map(ref_seqs(t))
    print(f'capped test: rebuilt ref_seq equals the harness ref_seq on {(r == t.ref_seq).mean():.4%} of rows')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-capped', action='store_true', help='also check the ref_seq rebuild against the capped test table')
    a = ap.parse_args()
    if a.check_capped:
        check_capped()
    for ds in FOLDER:
        build(ds)
