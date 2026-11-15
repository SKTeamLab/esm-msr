"""Wild-type reference scores for the ProteinGym Activity assays, read from the RAW DMS files.

ProteinGym's curated files (``DMS_score``) carry no wild-type row, but the raw files (``<root>/ProteinGym/substitutions_raw_DMS``) often do.  For each of the 43 Activity-category
assays this finds, on the CURATED score scale (the raw phenotype column mapped to ``DMS_score`` by a fit over the shared single mutants; it is the identity, or a sign flip for
CCDB / RPC1 / S22A1):

  explicit WT rows      mutant == 'WT' / 'wt' / '_wt' (GFP_AEQVI: the numAAMutations == 0 row)            -> wt_explicit
  synonymous rows       amino-acid identity 'A12A' (synonymous codons; median, and 1.4826 x MAD = noise)   -> wt_syn, syn_sd
  null rows             stop codons ('X12*'; for CASP3/7 position-only rows 'G3')                          -> null_stop, null_sd (1.4826 x MAD)

``wt_value`` = the explicit WT row where there is one, else the synonymous median (``wt_source`` says which); assays with neither are tier C: ``wt_value`` is filled from
``data/wt_manual.csv`` (values taken from the original papers, with the quote and URL) when that file has the assay, else left empty.

A position-specific WT (AICDA: the score is an enrichment within each position's own library, so WT differs per position) is left empty here and stored in
``data/aicda_wt_by_position.csv`` (the WT-residue rows of MaveDB urn:mavedb:00000106-c-1, which ProteinGym dropped); the tier is then ``C-position``.

``syn_distinct`` counts the distinct synonymous values: where it is far below ``n_syn`` the synonymous rows are pinned (to 0 on log-ratio scales built that way), so their spread says nothing
about the measurement noise (``02_activity_screening.py`` then falls back to another noise source).

  python analysis/pg_activity/01_wt_reference.py          # needs the raw DMS files; writes analysis/pg_activity/data/wt_reference.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from msr_analysis.registry import PUBLIC, ROOT  # noqa: E402

DATA = Path(__file__).resolve().parent / 'data'

RAW = ROOT / 'ProteinGym' / 'substitutions_raw_DMS'
STD = re.compile(r'^[A-Z]\d+[A-Z]$')
IDENT = re.compile(r'^([A-Z])\d+\1$')
POSONLY = re.compile(r'^[A-Z]\d+$')
WT_NAMES = {'WT', '_WT', 'WILDTYPE', 'WILD-TYPE'}


def mad_sd(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 2 else np.nan


def one(dms, row):
    fn = RAW / f'{dms}.csv'
    if not fn.exists():
        fn = RAW / str(row.raw_DMS_filename)
    df = pd.read_csv(fn, low_memory=False)
    mc = row.raw_DMS_mutant_column if isinstance(row.raw_DMS_mutant_column, str) and row.raw_DMS_mutant_column in df.columns else 'mutant'
    pc = str(row.raw_DMS_phenotype_name)
    out = dict(DMS_id=dms, raw_file=fn.name, mutant_col=mc, phenotype_col=pc)
    if pc not in df.columns or mc not in df.columns:
        out['note'] = 'phenotype or mutant column not in raw file'
        return out
    m = df[mc].astype(str)
    s = pd.to_numeric(df[pc], errors='coerce')
    cur = pd.read_csv(PUBLIC / 'proteingym_published' / f'{dms}.csv', usecols=['mutant', 'DMS_score']).drop_duplicates('mutant').set_index('mutant')['DMS_score']
    sing = m.str.match(STD)
    agg = pd.DataFrame({'raw': s[sing].values, 'mut': m[sing].values}).groupby('mut')['raw'].mean().to_frame().join(cur, how='inner').dropna()
    slope, icpt = np.polyfit(agg['raw'], agg['DMS_score'], 1)
    out.update(map_slope=slope, map_icpt=icpt, map_r=float(np.corrcoef(agg['raw'], agg['DMS_score'])[0, 1]), map_n=len(agg))
    to_cur = lambda mask: (slope * s[mask].dropna() + icpt)                       # noqa: E731
    ident = m.str.match(IDENT)
    expl = m.str.upper().isin(WT_NAMES)
    if 'numAAMutations' in df.columns:                                          # GFP_AEQVI: the WT row has no mutant name
        expl = expl | (df['numAAMutations'] == 0)
    stop = (m.str.endswith('*') & ~m.str.contains(':')) | (m.str.match(POSONLY) & (dms.startswith('CASP')))
    e, i, st = to_cur(expl), to_cur(ident), to_cur(stop)
    out.update(n_explicit=int(len(e)), wt_explicit=float(e.mean()) if len(e) else np.nan,
               n_syn=int(len(i)), wt_syn=float(i.median()) if len(i) else np.nan, syn_sd=mad_sd(i), syn_distinct=int(len(set(np.round(i.to_numpy(), 6)))),
               n_null=int(len(st)), null_stop=float(st.median()) if len(st) else np.nan, null_sd=mad_sd(st))
    return out


def main():
    ref = pd.read_csv(ROOT / 'ProteinGym' / 'DMS_substitutions.csv')
    ref = ref[ref.coarse_selection_type == 'Activity'].set_index('DMS_id')
    rows = [one(d, r) for d, r in ref.iterrows()]
    t = pd.DataFrame(rows)
    t['wt_value'] = t['wt_explicit'].where(t['n_explicit'] > 0, t['wt_syn'])
    t['wt_source'] = np.where(t['n_explicit'] > 0, 'explicit WT row', np.where(t['n_syn'] > 0, 'synonymous rows', ''))
    man = DATA / 'wt_manual.csv'
    if man.exists():
        mn = pd.read_csv(man).set_index('DMS_id')
        for d in mn.index:
            k = t['DMS_id'] == d
            if t.loc[k, 'wt_source'].eq('').all():
                t.loc[k, 'wt_value'] = mn.loc[d, 'wt_value']
                t.loc[k, 'wt_source'] = 'paper: ' + str(mn.loc[d, 'source'])
                for c in ('null_value', 'noise', 'confidence', 'evidence'):
                    if c in mn.columns:
                        t.loc[k, 'manual_' + c] = mn.loc[d, c]
    t['tier'] = np.where(t['n_explicit'] > 0, 'A', np.where(t['n_syn'] > 0, 'B', np.where(t['wt_value'].notna(), 'C-paper', np.where(t['wt_source'].str.startswith('paper'), 'C-position', 'C-open'))))
    t.to_csv(DATA / 'wt_reference.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 250, 'display.max_rows', 100)
    print(t[['DMS_id', 'tier', 'wt_source', 'wt_value', 'n_explicit', 'n_syn', 'syn_distinct', 'syn_sd', 'null_stop']].round(3).to_string(index=False))
    print(t['tier'].value_counts().to_dict())


if __name__ == '__main__':
    main()
