"""Rosetta cartesian_ddg: which benchmark rows were used to tune the energy function it runs on.

Our runs (Rosetta 2019 release; ``cartesian_ddg -ddg:legacy true -ddg:iterations 3 -ddg:bbnbrs 1`` on
cartesian-FastRelaxed structures) use ``-corrections:beta_nov16_cart``, not the ref2015 default. Rosetta has no
per-variant training, but two of its global parameters were fit against experimental stability data:

  * the 20 per-amino-acid reference energies of beta_nov16 were refit on a weighted objective that includes the
    Spearman correlation on "ProTherm DDGs following PMID 21287615" (Kellogg, Leaver-Fay & Baker 2011, 1,210
    mutations) and on the protease resistance of the Rocklin et al. 2017 designs (PMID 28706065)
    (https://docs.rosettacommons.org/docs/latest/Updates-beta-nov16);
  * the REU -> kcal/mol factor 1/2.94 applied downstream was a least-squares fit on the same Kellogg set
    (Park et al. 2016 JCTC 12:6201, SI "Mutational ddG calculation"); it changes no rank metric.

So the audit codes Rosetta ``V`` (data used for calibration / model selection), never ``M`` or ``P``: about 20 global
offsets cannot encode any one variant. This step counts, per dataset, the rows that were in that tuning data:

  * 12 external sets: rows whose every substitution is in the Kellogg set (same PDB + chain + PDB residue number +
    wild type + mutant, or the same UniProt position through SIFTS for alternate PDB entries); for PTMUL / PTMUL-D
    that is the constituent-substitution reading. ``n_sameprot`` = same PDB or same UniProt accession.
  * Tsuboyama capped val / test: rows on Rocklin-2017 design libraries (``{HHH,EHEE,HEEH,EEHEE}_rd{1-4}_*``), whose
    whole-design stability scores entered the fit (protein level; no mutation data from them was used).
  * ProteinGym: rows of assays in the same >=95%-identity family (08_families.py) as a Kellogg protein or a
    Rocklin-2017 design (protein level).

Inputs: ``data/kellogg2011_1210.csv`` (tracked; Kortemme-lab curated copy of the Kellogg set,
https://github.com/Kortemme-Lab/ddg/blob/master/input/csv/kellogg.csv), the SIFTS table
``$LEAK_DATA_DIR/raw/pdb_chain_uniprot.tsv.gz`` (https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/), and
``data/preprocessed/*_mapped.csv``. Writes ``rosetta_overlap.csv``.
"""
import collections
import gzip
import os
import pickle
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR, data_file, load_testsets

os.makedirs(OUT_DIR, exist_ok=True)
EXTERNAL = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
ROCKLIN2017 = re.compile(r'^(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_')


def norm_num(x):
    x = str(x).strip()
    if re.fullmatch(r'-?\d+\.0', x):
        x = x[:-2]
    return x.upper()


kel = pd.read_csv(data_file('kellogg2011_1210.csv'), dtype=str)
kel['pdb'] = kel['pdb'].str.upper()
kel['resnum'] = kel['resnum'].map(norm_num)

# ---------------------------------------------------------------- external-set rows (PDB numbering)
test_rows = {}
for t in EXTERNAL:
    d = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'preprocessed', f'{t}_mapped.csv'), dtype=str)
    rows = []
    for r in d.itertuples(index=False):
        if t in ('ptmul', 'ptmuld') and isinstance(r.mut_info, str) and r.mut_info:
            muts = []
            for m in r.mut_info.split(':'):
                mm = re.fullmatch(r'([A-Z])(-?\d+[A-Z]?)([A-Z])', m.strip())
                muts.append((mm.group(1), norm_num(mm.group(2)), mm.group(3)))
        else:
            muts = [(r.wild_type, norm_num(r.position), r.mutation)]
        rows.append((str(r.code).upper(), str(r.chain), muts))
    test_rows[t] = rows

# ---------------------------------------------------------------- SIFTS PDB -> UniProt segments
need = set(kel['pdb'].str.lower()) | {c.lower() for rows in test_rows.values() for c, _, _ in rows}
seg = collections.defaultdict(list)
with gzip.open(data_file('raw', 'pdb_chain_uniprot.tsv.gz'), 'rt') as f:
    for line in f:
        if line.startswith('#') or line.startswith('PDB\t'):
            continue
        p = line.rstrip('\n').split('\t')
        if p[0] in need:
            seg[(p[0].upper(), p[1])].append(p[2:])


def to_uniprot(pdb, chain, num):
    """PDB residue number -> (UniProt accession, position) by the constant offset within a SIFTS segment."""
    m = re.fullmatch(r'(-?\d+)([A-Z]?)', num)
    if not m or m.group(2):
        return None
    n = int(m.group(1))
    for acc, _, _, pb, pe, sb, se in seg.get((pdb, chain), []):
        sb, se = int(sb), int(se)
        if re.fullmatch(r'-?\d+', pb):
            off = int(pb) - sb
        elif re.fullmatch(r'-?\d+', pe):
            off = int(pe) - se
        else:
            continue
        if sb <= n - off <= se:
            return acc, n - off
    return None


k_pdb = set(zip(kel['pdb'], kel['chain'], kel['resnum'], kel['wt'], kel['mt']))
k_uni = set()
for r in kel.itertuples(index=False):
    u = to_uniprot(r.pdb, r.chain, r.resnum)
    if u:
        k_uni.add((u[0], u[1], r.wt, r.mt))
k_pdbs = set(kel['pdb'])
k_accs = {s[0] for (p, c), ss in seg.items() if p in k_pdbs for s in ss}

out = []
for t, rows in test_rows.items():
    tuned = sameprot = 0
    for code, chain, muts in rows:
        hit = []
        for wt, num, mt in muts:
            u = to_uniprot(code, chain, num)
            hit.append((code, chain, num, wt, mt) in k_pdb or (u is not None and (u[0], u[1], wt, mt) in k_uni))
        tuned += all(hit)
        sameprot += code in k_pdbs or bool({s[0] for s in seg.get((code, chain), [])} & k_accs)
    out.append(dict(dataset=t, n=len(rows), n_tuned=tuned, n_sameprot=sameprot, basis='Kellogg 2011 mutation (PDB or UniProt position)'))

# ---------------------------------------------------------------- Tsuboyama capped val / test and ProteinGym
tests = load_testsets()
for t in ('tsu_capped_val', 'tsu_capped_test'):
    d = tests[t]
    n_design = int(d['pid'].str[4:].str.match(ROCKLIN2017).sum())
    out.append(dict(dataset=t, n=len(d), n_tuned=n_design, n_sameprot=n_design, basis='Rocklin 2017 design library (protein level)'))

fm = pickle.load(open(os.path.join(WORK_DATA_DIR, 'family_maps.pkl'), 'rb'))
pid_fam = {p: v[0] for p, v in fm['maps'].items()}
prot = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'proteins.pkl'))
fam_of = dict(zip(prot['pid'], prot['family']))
kel_chains = set(zip(kel['pdb'], kel['chain']))
tuned_fams = set()
for pid in prot['pid']:
    src, _, name = str(pid).partition(':')
    if src in EXTERNAL and tuple(name.rsplit('_', 1)) in kel_chains:
        tuned_fams.add(fam_of[pid])
    elif src == 'lib' and ROCKLIN2017.match(name):
        tuned_fams.add(fam_of[pid])
pg = tests['proteingym']
pg_hit = pg['pid'].map(fam_of).isin(tuned_fams)
print(f'ProteinGym: {pg.loc[pg_hit, "pid"].nunique()} assays in a Kellogg / Rocklin-2017 family')
out.append(dict(dataset='proteingym', n=len(pg), n_tuned=int(pg_hit.sum()), n_sameprot=int(pg_hit.sum()),
                basis='same family as a Kellogg protein or Rocklin 2017 design (protein level)'))

res = pd.DataFrame(out)
res['frac_tuned'] = res['n_tuned'] / res['n']
res.to_csv(os.path.join(OUT_DIR, 'rosetta_overlap.csv'), index=False)
print(res.to_string(index=False))
