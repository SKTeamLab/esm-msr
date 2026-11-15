#!/usr/bin/env python
"""Per-assay leak test of the ProteinGym assays against the ESM-MSR training data.

For every ProteinGym assay: the highest structure similarity (TM-score, normalised by the assay protein, Foldseek) and the highest sequence identity (MMseqs2 ``fident``) between its protein and any protein
of the ESM-MSR training inventory (the capped MegaScale train libraries, expanded to their proteins as in ``leak_audit/17_proteingym.py``).  An assay is **leaked** when TM > 0.7 or sequence identity > 0.4.
The hits are the audit's own tables (``leak_audit/data/foldseek_hits_dom_pg.pkl`` and ``mmseqs_pairs.csv``); ``seq_cov`` is the share of the assay protein covered by the alignment of the best-identity
hit and ``aln_len`` / ``seq_evalue`` its alignment length and E-value, so that an identity over a short stretch can be told from a homologous domain.  Writes ``results/leak_per_assay.csv`` (assay,
category, max_tm, max_seqid, seq_cov, aln_len, seq_evalue, leaked, leaked_tm, leaked_seqid, exact_overlap, leaked_signif; ``leaked_signif`` keeps the identity
hits only when their E-value is at most 1e-3) and prints how many assays are flagged per category; ``exact_overlap`` marks the 23 assays whose variants are in the training data (``pgsigma.train_overlap``).

    python analysis/proteingym/p0_sigma/04_leak_per_assay.py
"""
import collections
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pgsigma as P
sys.path.insert(0, str(P.HERE.parent.parent / 'leak_audit'))
from lib import WORK_DATA_DIR, expand_lib_pids  # noqa: E402

TM_CUT, ID_CUT, SIG_E = 0.7, 0.4, 1e-3          # SIG_E: E-value an identity hit needs to count as significant (the stricter ``leaked_signif`` flag)


def main():
    sp = pickle.load(open(Path(WORK_DATA_DIR) / 'capped_split_sets.pkl', 'rb'))
    lib = pd.read_pickle(Path(WORK_DATA_DIR) / 'megascale_libraries.pkl')
    inv = expand_lib_pids(lib, set(sp['train']))
    prots = pd.read_pickle(Path(WORK_DATA_DIR) / 'proteins.pkl')
    L = dict(zip(prots['pid'], prots['seq'].str.len()))
    mm = pd.read_csv(Path(WORK_DATA_DIR) / 'mmseqs_pairs.csv')
    adj = collections.defaultdict(list)
    for a, b, f, al, e in zip(mm['p'], mm['q'], mm['fident'], mm['alnlen'], mm['evalue']):
        adj[a].append((b, float(f), int(al), float(e)))
        adj[b].append((a, float(f), int(al), float(e)))
    fs = pd.read_pickle(Path(WORK_DATA_DIR) / 'foldseek_hits_dom_pg.pkl')
    fs = fs[fs['t_pid'].isin(inv)]
    qtm = fs.groupby('q_pid')['qtmscore'].max()
    ref = pd.read_csv(P.ROOT / 'ProteinGym' / 'DMS_substitutions.csv')
    overlap = P.train_overlap()
    rows = []
    for dms, cat in zip(ref.DMS_id, ref.coarse_selection_type):
        pid = 'pg:' + dms
        hits = [(f, al, e) for q, f, al, e in adj.get(pid, ()) if q in inv]
        best = max(hits, default=(np.nan, np.nan, np.nan))
        tm = float(qtm.get(pid, np.nan))
        rows.append(dict(assay=dms, category=cat, max_tm=tm, max_seqid=best[0], seq_cov=best[1] / L[pid] if hits else np.nan, aln_len=best[1], seq_evalue=best[2],
                         leaked_tm=bool(tm > TM_CUT), leaked_seqid=bool(best[0] > ID_CUT) if hits else False, exact_overlap=dms in overlap))
    t = pd.DataFrame(rows)
    t['leaked'] = t.leaked_tm | t.leaked_seqid
    t['leaked_signif'] = t.leaked_tm | (t.leaked_seqid & (t.seq_evalue <= SIG_E))
    t.to_csv(P.OUT / 'leak_per_assay.csv', index=False, float_format='%.4f')
    u = t[~t.exact_overlap]
    print(f'{len(t)} assays; {int(t.exact_overlap.sum())} with exact variant overlap (left out of the figures); of the other {len(u)}: leaked (TM > {TM_CUT} or seq id > {ID_CUT}) '
          f'{int(u.leaked.sum())}  (TM only {int((u.leaked_tm & ~u.leaked_seqid).sum())}, seq id only {int((u.leaked_seqid & ~u.leaked_tm).sum())}, both {int((u.leaked_tm & u.leaked_seqid).sum())})')
    print(f'  with significant identity hits only (E <= {SIG_E:g}): {int(u.leaked_signif.sum())}')
    print(u.groupby('category').agg(n=('leaked', 'size'), leaked=('leaked', 'sum'), tm=('leaked_tm', 'sum'), seqid=('leaked_seqid', 'sum')).to_string())
    print('assays with no structure hit / no sequence hit to the training proteins:', int(t.max_tm.isna().sum()), '/', int(t.max_seqid.isna().sum()))
    print(u[u.leaked].sort_values('max_seqid', ascending=False)[['assay', 'category', 'max_tm', 'max_seqid', 'aln_len', 'seq_evalue']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
