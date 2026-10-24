"""Build the test-side variant inventory: every row reported in BENCHMARKS_OCT2026.md.

Output: data/testsets/*.pkl, each a DataFrame with columns
    dataset, pid, seq, mutation, n_sub, label
where ``mutation`` is the canonical 'WT<pos><MT>' string in 1-based coordinates of ``seq``.
"""
import os, sys, pickle, re, glob
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from lib import parse_mut_str, check_against_seq, hash_seq

ROOT = '/home/sareeves/playground/esm-msr-devel'
OUT = os.path.join(ROOT, 'leak_audit', 'data', 'testsets')
os.makedirs(OUT, exist_ok=True)
REPO = os.path.join(ROOT, 'repo')

def finish(rows, dataset, path):
    d = pd.DataFrame(rows, columns=['dataset', 'pid', 'seq', 'mutation', 'n_sub', 'label', 'row_id'])
    d['dataset'] = dataset
    d.to_pickle(os.path.join(OUT, dataset + '.pkl'))
    n_bad = 0
    print(f'{dataset:24s} rows={len(d):7d} proteins={d.pid.nunique():5d} '
          f'subs={d.n_sub.value_counts().to_dict()}')
    return d

# ---------------------------------------------------------------- Tsuboyama capped
for split in ['val', 'test']:
    f = os.path.join(ROOT, 'remote_data', f'splits_oct06_capped-{split}.csv.gz')
    d = pd.read_csv(f, low_memory=False)
    rows = []
    bad = 0
    for pid, seq, mt, lab, rid in zip(d['code_wt'], d['ref_seq'], d['mut_type'], d['ddG_ML'], d['id']):
        muts = parse_mut_str(mt)
        if muts is None:
            bad += 1; continue
        ok, _ = check_against_seq(str(seq), muts)
        if not ok: bad += 1
        rows.append((f'tsu_capped_{split}', 'tsu:' + str(pid), str(seq),
                     ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
                     len(muts), lab, rid))
    print('capped', split, 'unparsed/mismatch', bad, 'of', len(d))
    finish(rows, f'tsu_capped_{split}', None)

# ---------------------------------------------------------------- Tsuboyama hyperopt
msv = pd.read_pickle(os.path.join(ROOT, 'leak_audit', 'data', 'megascale_variants.pkl'))
libs = pd.read_pickle(os.path.join(ROOT, 'leak_audit', 'data', 'megascale_libraries.pkl'))
sp = pickle.load(open(os.path.join(REPO, 'data', 'hyperopt_splits.pkl'), 'rb'))
for split, key in [('val', 'val'), ('test', 'test_internal')]:
    codes = [x.replace('.pdb', '') for x in sp[key]]
    codes = [c.replace('|', '_') for c in codes]
    sub = msv[msv['code'].isin(codes)]
    rows = [(f'tsu_hyperopt_{split}', 'tsu:' + c, libs.loc[c, 'wt_seq'], mut, n, lab, c)
            for c, mut, n, lab in zip(sub['code'], sub['mutation'], sub['n_mut'], sub['ddG_ML'])
            if isinstance(mut, str)]
    finish(rows, f'tsu_hyperopt_{split}', None)

# ---------------------------------------------------------------- external thermostability sets
def ext_adapter(dataset, fname, mutcol=None, seqcol='wt_seq', poscol=None, wtcol='wild_type',
                mutcol2='mutation', codecol='code', chaincol='chain', labelcol=None, fixed_seq=None):
    """External-set adapter. Mutations are built from position/wild_type/mutation columns
    (``seq_pos`` when present, which is the 1-based index into ``wt_seq``), or parsed from
    ``mutcol`` when that string is already in sequence coordinates (PTMUL)."""
    d = pd.read_csv(os.path.join(REPO, 'data', 'preprocessed', fname), low_memory=False)
    rows, bad, why = [], 0, {}
    for i, r in d.iterrows():
        seq = fixed_seq if fixed_seq is not None else str(r[seqcol])
        if mutcol is not None and isinstance(r.get(mutcol), str) and ':' in str(r.get(mutcol)):
            muts = parse_mut_str(r[mutcol])
        elif poscol is not None:
            p = r[poscol]
            if pd.isna(p):
                p = r.get('position', np.nan)
            muts = None if pd.isna(p) else [(str(r[wtcol]), int(p), str(r[mutcol2]))]
        elif mutcol is not None and isinstance(r.get(mutcol), str):
            muts = parse_mut_str(r[mutcol])
        else:
            muts = None
        if not muts:
            bad += 1; why['parse'] = why.get('parse', 0) + 1; continue
        ok, _ = check_against_seq(seq, muts)
        if not ok:
            bad += 1; why['seq'] = why.get('seq', 0) + 1; continue
        lab = r[labelcol] if labelcol else np.nan
        rows.append((dataset, f'{dataset}:{r[codecol]}_{r[chaincol]}', seq,
                     ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
                     len(muts), lab, str(i)))
    if bad: print('   ', dataset, 'skipped', bad, why)
    return finish(rows, dataset, None)

ext = {
    's571':  ('s571_mapped.csv',  dict(poscol='seq_pos')),
    's783':  ('s783_mapped.csv',  dict(poscol='seq_pos')),
    's2648': ('s2648_mapped.csv', dict(poscol='seq_pos')),
    's8754': ('s8754_mapped.csv', dict(poscol='seq_pos')),
    's669':  ('s669_mapped.csv',  dict(poscol='seq_pos')),
    's461':  ('s461_mapped.csv',  dict(poscol='seq_pos')),
    'ssym':  ('ssym_mapped.csv',  dict(poscol='seq_pos')),
    'q3421': ('q3421_mapped.csv', dict(poscol='seq_pos')),
    'k3822': ('k3822_mapped.csv', dict(poscol='seq_pos')),
    'k2369': ('k2369_mapped.csv', dict(poscol='seq_pos')),
    'ptmul': ('ptmul_mapped.csv', dict(mutcol='mut_info_seq_pos', poscol=None)),
    'ptmuld':('ptmuld_mapped.csv',dict(mutcol='mut_info_seq_pos', poscol=None)),
}
for name, (f, kw) in ext.items():
    ext_adapter(name, f, **kw)

# ---------------------------------------------------------------- DMS assays
dms = {
    'DLG4 abund.': ('DLG4_HUMAN_Faure_2021_abundance_domain.csv', 'ddG_ML'),
    'DLG4 bind.':  ('DLG4_HUMAN_Faure_2021_binding_domain.csv', 'ddG_ML'),
    'GRB2 abund.': ('GRB2_HUMAN_Faure_2021_abundance_domain.csv', 'ddG_ML'),
    'GRB2 bind.':  ('GRB2_HUMAN_Faure_2021_binding_domain.csv', 'ddG_ML'),
    'MYO':         ('MYO_HUMAN_Kung_2025_display.csv', 'ddG_ML'),
    'ESTA':        ('ESTA_BACSU_Nutschel_2020_dTm.csv', 'ddG_ML'),
    'GB1':         ('GB1_Wu_2016_binding_domain.csv', 'ddG_ML'),
}
GB1_SEQ = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
for name, (f, lab) in dms.items():
    ext_adapter(name, f, mutcol='mut_type', labelcol=lab,
                fixed_seq=GB1_SEQ if name == 'GB1' else None)

# ---------------------------------------------------------------- Domainome
dom_f = os.path.join('/home/sareeves', 'software', 'SPURS_mod', 'data', 'dataset', 'Domainome',
                     'Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv')
dd = pd.read_csv(dom_f, low_memory=False)
rows, bad, nbadseq = [], 0, 0
for dom, v, umut, seq, fit in zip(dd['domain_ID'], dd['variant_ID'], dd['uniprot_ID_mutation'],
                                  dd['pdb_seq'], dd['aPCA_normalized_fitness']):
    # variant_ID = '<domain_ID>_<WT><pos><MT>' in 1-based domain (pdb_seq) numbering
    suffix = str(v)[len(str(dom))+1:] if str(v).startswith(str(dom)) else str(v).split('_')[-1]
    muts = parse_mut_str(suffix)
    if not muts:
        bad += 1; continue
    ok, _ = check_against_seq(str(seq), muts)
    if not ok:
        nbadseq += 1; continue
    rows.append(('domainome', 'dom:' + str(dom), str(seq),
                 ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])), len(muts), fit, umut))
print('domainome skipped unparsed', bad, 'seq-inconsistent', nbadseq)
finish(rows, 'domainome', None)

# ---------------------------------------------------------------- ProteinGym
pgm = pd.read_csv(os.path.join(ROOT, 'ProteinGym', 'DMS_substitutions.csv'))
pgdir = os.path.join(ROOT, 'ProteinGym', 'DMS_ProteinGym_substitutions', 'DMS_ProteinGym_substitutions')
rows, badtot = [], 0
for _, r in pgm.iterrows():
    f = os.path.join(pgdir, r['DMS_filename'])
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    for i, rr in d.iterrows():
        muts = parse_mut_str(rr['mutant'])
        if not muts:
            badtot += 1; continue
        ok, _ = check_against_seq(str(r['target_seq']), muts)
        if not ok:
            badtot += 1; continue
        rows.append(('pg:' + str(r['DMS_id']), 'pg:' + str(r['DMS_id']), str(r['target_seq']),
                     ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
                     len(muts), rr['DMS_score'], f"{r['DMS_id']}:{i}"))
print('proteingym skipped', badtot)
finish(rows, 'proteingym', None)
print('done')
