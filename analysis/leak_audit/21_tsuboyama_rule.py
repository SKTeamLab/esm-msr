#!/usr/bin/env python
"""Which benchmark proteins are homologous to ESM-MSR's training set under the rule that built the Homology-Restricted MegaScale split.

The split joins two proteins when any of these holds (``docs/SPLITTING_AND_HOMOLOGY_AUDIT.md``, ``09_split_audit.py``): MMseqs2 E <= 1e-3, Foldseek E <= 1e-3, BLOSUM62 local identity > 0.40, or TM-score > 0.70
(normalised by the shorter chain).  This script applies that rule, with no coverage requirement, to the proteins of the summary figure against the 172 capped-train scaffolds:

  ProteinGym assays, Domainome domains   ``reference/leak_audit_structure_homology.csv`` (min_seq_evalue, max_seq_fident, min_struct_evalue, max_tm_max), plus the same-protein / strong-homolog class
                                         of the audit (``class_with_struct`` P or H)
  MGnify scaffolds                       every MMseqs2 hit (E <= 10) of ``20_mgnify_vs_capped_train.py`` (sequence only: no structure search was run on them)
  Rocklin designs                        the same MMseqs2 search, run here for the 1,000 design WTs (sequence only)

Output ``reference/tsuboyama_rule_flags.csv``: dataset (proteingym / domainome / mgnify / rocklin_denovo), item_id (DMS id / domain id / code_wt), homologous (rule met), min_evalue, max_fident,
min_struct_evalue, max_tm, reason.  ``$LEAK_OUT_DIR`` (default ``~/playground/esm-msr-devel/leak_audit/out``) holds ``mgnify_vs_capped/`` from step 20.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import MMSEQS_BIN, OUT_DIR, repo_data  # noqa: E402

REF = HERE.parent / 'reference'
E_MAX, IDENT, TM = 1e-3, 0.40, 0.70
WD = Path(OUT_DIR) / 'mgnify_vs_capped'
TABLE = repo_data('mgnify', 'mgnify_mapped.csv')
COLS = ['q', 't', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits', 'qcov']


def from_hits(h):
    g = h.groupby('q')
    return pd.DataFrame({'min_evalue': g['evalue'].min(), 'max_fident': g['fident'].max()})


def flag(d, struct=False):
    r = (d['min_evalue'] <= E_MAX).fillna(False)
    i = (d['max_fident'] > IDENT).fillna(False)
    reason = np.where(r, 'E', '')
    reason = np.where(i, np.char.add(reason.astype(str), 'I'), reason)
    if struct:
        s = (d['min_struct_evalue'] <= E_MAX).fillna(False)
        t = (d['max_tm'] > TM).fillna(False)
        reason = np.where(s, np.char.add(reason.astype(str), 'S'), reason)
        reason = np.where(t, np.char.add(reason.astype(str), 'T'), reason)
        return r | i | s | t, reason
    return r | i, reason


def main():
    out = []
    sh = pd.read_csv(REF / 'leak_audit_structure_homology.csv')
    for ds in ('proteingym', 'domainome'):
        d = sh[sh['dataset'] == ds].copy()
        d['item_id'] = d['item_id'].astype(str)
        d = d.rename(columns={'min_seq_evalue': 'min_evalue', 'max_seq_fident': 'max_fident', 'max_tm_max': 'max_tm'})
        h, reason = flag(d, struct=True)
        audit = d['class_with_struct'].isin(['P', 'H']).to_numpy()
        d['homologous'] = (h | audit).to_numpy()
        d['reason'] = [(r + ('A' if a else '')) for r, a in zip(reason, audit)]
        out.append(d.assign(dataset=ds)[['dataset', 'item_id', 'homologous', 'min_evalue', 'max_fident', 'min_struct_evalue', 'max_tm', 'reason']])

    mg = pd.read_csv(TABLE, usecols=['code_wt', 'source', 'wt_seq']).drop_duplicates('code_wt')
    m8 = WD / 'mgnify_vs_capped_train.m8'
    h = pd.read_csv(m8, sep='\t', names=COLS)
    ev = from_hits(h)
    m = mg[mg['source'] == 'mgnify'].set_index('code_wt')
    d = ev.reindex(m.index)
    d['homologous'], d['reason'] = flag(d)
    out.append(d.reset_index().rename(columns={'code_wt': 'item_id'}).assign(dataset='mgnify', min_struct_evalue=np.nan, max_tm=np.nan)
               [['dataset', 'item_id', 'homologous', 'min_evalue', 'max_fident', 'min_struct_evalue', 'max_tm', 'reason']])

    rd = mg[mg['source'] == 'rocklin_denovo']
    q_fa, r_m8 = WD / 'rocklin_designs.fa', WD / 'rocklin_vs_capped_train.m8'
    if not r_m8.exists():
        with open(q_fa, 'w') as fh:
            for r in rd.itertuples():
                fh.write(f'>{r.code_wt}\n{r.wt_seq}\n')
        subprocess.run([MMSEQS_BIN, 'easy-search', str(q_fa), str(WD / 'capped_train.fa'), str(r_m8), str(WD / 'tmp_rocklin'), '-s', '7.5', '-e', '10', '--max-seqs', '500', '--threads', '4',
                        '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue,bits,qcov'], check=True)
    hr = pd.read_csv(r_m8, sep='\t', names=COLS)
    d = from_hits(hr).reindex(rd['code_wt'])
    d['homologous'], d['reason'] = flag(d)
    out.append(d.reset_index().rename(columns={'code_wt': 'item_id'}).assign(dataset='rocklin_denovo', min_struct_evalue=np.nan, max_tm=np.nan)
               [['dataset', 'item_id', 'homologous', 'min_evalue', 'max_fident', 'min_struct_evalue', 'max_tm', 'reason']])

    res = pd.concat(out, ignore_index=True)
    res.to_csv(REF / 'tsuboyama_rule_flags.csv', index=False, float_format='%.6g')
    print(res.groupby('dataset')['homologous'].agg(['sum', 'size']).rename(columns={'sum': 'homologous', 'size': 'items'}).to_string())
    print(res[res.homologous].groupby(['dataset', 'reason']).size().to_string())


if __name__ == '__main__':
    main()
