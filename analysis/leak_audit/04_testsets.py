"""Build the test-side variant inventory: every row reported in BENCHMARKS_OCT2026.md.

Output: $LEAK_DATA_DIR/testsets/*.pkl, each a DataFrame with columns
    dataset, pid, seq, mutation, n_sub, label, row_id
where ``mutation`` is the canonical 'WT<pos><MT>' string in 1-based coordinates of ``seq``.
For Tsuboyama capped splits, ``pid`` uses ``tsu:<code>`` (preserving mutant-background suffixes
such as ``tsu:2MKX_R27A``) so every ``pid`` maps to a unique wild-type/background sequence.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import LEAK_ROOT, REPO_ROOT, SOFTWARE_DIR, WORK_DATA_DIR, check_against_seq, data_file, norm_wt_name, parse_mut_str

OUT = os.path.join(WORK_DATA_DIR, 'testsets')
os.makedirs(OUT, exist_ok=True)


def finish(rows, dataset):
    d = pd.DataFrame(rows, columns=['dataset', 'pid', 'seq', 'mutation', 'n_sub', 'label', 'row_id'])
    d['dataset'] = dataset
    d.to_pickle(os.path.join(OUT, dataset + '.pkl'))
    print(f'{dataset:24s} rows={len(d):7d} proteins={d.pid.nunique():5d} subs={d.n_sub.value_counts().to_dict()}')
    return d


# ---------------------------------------------------------------- Tsuboyama capped
for split in ['val', 'test']:
    f = os.path.join(LEAK_ROOT, 'remote_data', f'splits_oct06_capped-{split}.csv.gz')
    d = pd.read_csv(f, low_memory=False)
    code_col = d['code'] if 'code' in d.columns else d['code_wt']
    rows, bad = [], 0
    for code, seq, mt, lab, rid in zip(code_col, d['ref_seq'], d['mut_type'], d['ddG_ML'], d['id']):
        muts = parse_mut_str(mt)
        if muts is None:
            bad += 1
            continue
        ok, _ = check_against_seq(str(seq), muts)
        if not ok:
            bad += 1
        rows.append((
            f'tsu_capped_{split}',
            'tsu:' + str(code),
            str(seq),
            ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
            len(muts),
            lab,
            rid,
        ))
    print('capped', split, 'unparsed/mismatch', bad, 'of', len(d))
    finish(rows, f'tsu_capped_{split}')

# ---------------------------------------------------------------- Tsuboyama hyperopt
msv = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_variants.pkl'))
libs = pd.read_pickle(os.path.join(WORK_DATA_DIR, 'megascale_libraries.pkl'))
sp = pickle.load(open(os.path.join(REPO_ROOT, 'data', 'hyperopt_splits.pkl'), 'rb'))
for split, key in [('val', 'val'), ('test', 'test_internal')]:
    codes = {norm_wt_name(x)[1] for x in sp[key]}
    sub = msv[msv['code_wt'].isin(codes)]
    rows = [
        (f'tsu_hyperopt_{split}', 'tsu:' + c, libs.loc[c, 'wt_seq'], mut, n, lab, c)
        for c, mut, n, lab in zip(sub['code'], sub['mutation'], sub['n_mut'], sub['ddG_ML'])
        if isinstance(mut, str) and c in libs.index
    ]
    finish(rows, f'tsu_hyperopt_{split}')


# ---------------------------------------------------------------- external thermostability sets
def ext_adapter(dataset, fname, mutcol=None, seqcol='wt_seq', poscol=None, wtcol='wild_type',
                mutcol2='mutation', codecol='code', chaincol='chain', labelcol=None, fixed_seq=None):
    """External-set adapter. Mutations are built from position/wild_type/mutation columns
    (``seq_pos`` when present, which is the 1-based index into ``wt_seq``), or parsed from
    ``mutcol`` when that string is already in sequence coordinates (PTMUL)."""
    d = pd.read_csv(os.path.join(REPO_ROOT, 'data', 'preprocessed', fname), low_memory=False)
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
            bad += 1
            why['parse'] = why.get('parse', 0) + 1
            continue
        ok, _ = check_against_seq(seq, muts)
        if not ok:
            bad += 1
            why['seq'] = why.get('seq', 0) + 1
            continue
        lab = r[labelcol] if labelcol else np.nan
        rows.append((
            dataset,
            f'{dataset}:{r[codecol]}_{r[chaincol]}',
            seq,
            ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
            len(muts),
            lab,
            str(i),
        ))
    if bad:
        print('   ', dataset, 'skipped', bad, why)
    return finish(rows, dataset)


ext = {
    's571': ('s571_mapped.csv', dict(poscol='seq_pos')),
    's783': ('s783_mapped.csv', dict(poscol='seq_pos')),
    's2648': ('s2648_mapped.csv', dict(poscol='seq_pos')),
    's8754': ('s8754_mapped.csv', dict(poscol='seq_pos')),
    's669': ('s669_mapped.csv', dict(poscol='seq_pos')),
    's461': ('s461_mapped.csv', dict(poscol='seq_pos')),
    'ssym': ('ssym_mapped.csv', dict(poscol='seq_pos')),
    'q3421': ('q3421_mapped.csv', dict(poscol='seq_pos')),
    'k3822': ('k3822_mapped.csv', dict(poscol='seq_pos')),
    'k2369': ('k2369_mapped.csv', dict(poscol='seq_pos')),
    'ptmul': ('ptmul_mapped.csv', dict(mutcol='mut_info_seq_pos', poscol=None)),
    'ptmuld': ('ptmuld_mapped.csv', dict(mutcol='mut_info_seq_pos', poscol=None)),
}
for name, (f, kw) in ext.items():
    ext_adapter(name, f, **kw)

# ---------------------------------------------------------------- DMS assays
dms = {
    'DLG4 abund.': ('DLG4_HUMAN_Faure_2021_abundance_domain.csv', 'ddG_ML'),
    'DLG4 bind.': ('DLG4_HUMAN_Faure_2021_binding_domain.csv', 'ddG_ML'),
    'GRB2 abund.': ('GRB2_HUMAN_Faure_2021_abundance_domain.csv', 'ddG_ML'),
    'GRB2 bind.': ('GRB2_HUMAN_Faure_2021_binding_domain.csv', 'ddG_ML'),
    'MYO': ('MYO_HUMAN_Kung_2025_display.csv', 'ddG_ML'),
    'ESTA': ('ESTA_BACSU_Nutschel_2020_dTm.csv', 'ddG_ML'),
    'GB1': ('GB1_Wu_2016_binding_domain.csv', 'ddG_ML'),
}
GB1_SEQ = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
for name, (f, lab) in dms.items():
    ext_adapter(name, f, mutcol='mut_type', labelcol=lab, fixed_seq=GB1_SEQ if name == 'GB1' else None)

# ---------------------------------------------------------------- Domainome
# Beltran et al. (2025) Supplementary Table 5 (Domainome 1.0 variant table). Looked up in $LEAK_DATA_DIR/raw first; the
# only other local copy is the one shipped inside the SPURS repository's data folder.
DOM_TABLE = 'Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv'
dom_f = os.environ.get('DOMAINOME_CSV') or next(
    (c for c in (data_file('raw', DOM_TABLE),
                 os.path.join(SOFTWARE_DIR, 'SPURS_mod', 'data', 'dataset', 'Domainome', DOM_TABLE),
                 os.path.join(SOFTWARE_DIR, 'SPURS', 'data', 'dataset', 'Domainome', DOM_TABLE)) if os.path.exists(c)),
    data_file('raw', DOM_TABLE))
dd = pd.read_csv(dom_f, low_memory=False)
rows, bad, nbadseq = [], 0, 0
for dom, v, umut, seq, fit in zip(
    dd['domain_ID'], dd['variant_ID'], dd['uniprot_ID_mutation'], dd['pdb_seq'], dd['aPCA_normalized_fitness']
):
    suffix = str(v)[len(str(dom)) + 1:] if str(v).startswith(str(dom)) else str(v).split('_')[-1]
    muts = parse_mut_str(suffix)
    if not muts:
        bad += 1
        continue
    ok, _ = check_against_seq(str(seq), muts)
    if not ok:
        nbadseq += 1
        continue
    rows.append((
        'domainome', 'dom:' + str(dom), str(seq),
        ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
        len(muts), fit, umut,
    ))
print('domainome skipped unparsed', bad, 'seq-inconsistent', nbadseq)
finish(rows, 'domainome')

# ---------------------------------------------------------------- ProteinGym
pg_root = os.environ.get('PROTEINGYM_DIR', os.path.join(LEAK_ROOT, 'ProteinGym'))
pgm = pd.read_csv(os.path.join(pg_root, 'DMS_substitutions.csv'))
pgdir = os.path.join(pg_root, 'DMS_ProteinGym_substitutions', 'DMS_ProteinGym_substitutions')
rows, badtot = [], 0
for _, r in pgm.iterrows():
    f = os.path.join(pgdir, r['DMS_filename'])
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    for i, rr in d.iterrows():
        muts = parse_mut_str(rr['mutant'])
        if not muts:
            badtot += 1
            continue
        ok, _ = check_against_seq(str(r['target_seq']), muts)
        if not ok:
            badtot += 1
            continue
        rows.append((
            'pg:' + str(r['DMS_id']),
            'pg:' + str(r['DMS_id']),
            str(r['target_seq']),
            ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1])),
            len(muts),
            rr['DMS_score'],
            f"{r['DMS_id']}:{i}",
        ))
print('proteingym skipped', badtot)
finish(rows, 'proteingym')
