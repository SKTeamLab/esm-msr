#!/usr/bin/env python
"""Build canonical task spine tables for ``mega_splits`` Tsuboyama sets (`_tasks/tsuboyama_mega_{val,test}.csv.gz`).

Reverts mutations via ``msr_analysis.structures.parse_mutations`` and writes atomically via ``msr_analysis.paths``.

Usage: ``python -m audits.homology_and_splits.active_builders.build_mega_tasks [--check-capped]``
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from msr_analysis.paths import FINAL_BENCH, TASKS, atomic_to_csv, runner_folder
from msr_analysis.structures import parse_mutations

SETS = ('tsuboyama_mega_val', 'tsuboyama_mega_test')
PRED_COLS = (
    'wt_lora_pred', 'mt_lora_pred', 'combined_pred', 'combined_dddg_pred', 'epi_pred', 'mt_lora_dddg_pred',
    'wt_lora_pred_additive', 'wt_lora_dddg_pred', 'mt_lora_pred_additive', 'combined_pred_additive',
    'mut_type_renumbered', 'mut_type_pdb',
)


def ref_seqs(t: pd.DataFrame) -> dict[str, str]:
    """Reconstruct unique backbone reference sequence per ``code`` by reverting each row's ``mut_type``."""
    out = {}
    for code, g in t.groupby('code'):
        seqs = set()
        for seq, mt in zip(g['aa_seq'].astype(str), g['mut_type'].astype(str)):
            s = list(seq)
            toks = parse_mutations(mt, sep=':')
            if toks is None:
                raise SystemExit(f'{code}: malformed mut_type {mt!r}')
            for wt, pos, mut_aa in toks:
                if s[pos - 1] != mut_aa:
                    raise SystemExit(f'{code} {mt}: position {pos} is {s[pos - 1]}, not mutant {mut_aa}')
                s[pos - 1] = wt
            seqs.add(''.join(s))
        if len(seqs) != 1:
            raise SystemExit(f'{code}: {len(seqs)} backbone sequences')
        out[str(code)] = seqs.pop()
    return out


def build(ds: str) -> None:
    folder = runner_folder(ds)
    files = [
        FINAL_BENCH / folder / 'final_checkpoints' / 'FINAL_p0_mega' / f's{s}' / 'best.ckpt_sigma1.0_unmasked_predictions.csv'
        for s in (1, 2, 3)
    ]
    tabs = [pd.read_csv(f, low_memory=False) for f in files]
    t = tabs[0].drop(columns=[c for c in PRED_COLS if c in tabs[0]])
    for o in tabs[1:]:
        o = o[t.columns]
        same = o['id'].equals(t['id']) and np.allclose(o['ddG_ML'], t['ddG_ML'], equal_nan=True) and o['cens'].equals(t['cens'])
        if not same:
            raise SystemExit(f'{ds}: seeds disagree on rows or labels')
    t['ref_seq'] = t['code'].map(ref_seqs(t))
    t['backbone_mutation'] = t['mut_structure'].where(t['mut_structure'] != '-')
    t['group'] = t['code']
    t['truth'] = t['ddG_ML']
    t['seq_mismatch'] = 0
    t['dataset'] = folder
    t['row_id'] = np.arange(len(t))
    dst = TASKS / f'{ds}.csv.gz'
    atomic_to_csv(t, dst, index=False, compression='gzip')
    print(f'{ds}: {len(t):,} rows ({(t.cens == 0).sum():,} measured), {t.code_wt.nunique()} libraries -> {dst}')


def check_capped() -> None:
    t = pd.read_csv(TASKS / 'tsuboyama_capped_test.csv.gz', low_memory=False)
    r = t['code'].map(ref_seqs(t))
    print(f'capped test: rebuilt ref_seq matches harness ref_seq on {(r == t.ref_seq).mean():.4%} of rows')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check-capped', action='store_true')
    a = ap.parse_args(argv)
    if a.check_capped:
        check_capped()
    for ds in SETS:
        build(ds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
