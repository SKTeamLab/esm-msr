"""Assemble the per-(model, dataset) leak matrix with the audit notation.

One cell per (model, dataset) pair reported in ``docs/benchmarks/all_by_all_long.csv`` (plus ``EXTRA_CELLS``, pairs
printed in BENCHMARKS_OCT2026.md whose rho is not in the long CSV). Counts come from:
  10_overlap_engine.py   exact / constituent / same-protein rows for the Megascale ESM-MSR inventories
  11_homology_classes.py P / H / h rows for every inventory with sequences in the protein corpus
  12_literature_exact.py verified exact / constituent rows for DDGemb, DDMut, GeoDDG-3D
  14_esm3dg_homology.py  P / H / h rows against ESM3dG's MGnify training sequences
  16_rosetta.py          rows in the data used to tune Rosetta's beta_nov16 energy function (V)

The retrained transfer-learning baselines (MutateEverything, ThermoMPNN / ThermoMPNN-D, SPURS / SPURS multi) are not
audited here: they were trained on the same capped-train scaffolds as ESM-MSR (BENCHMARKS_OCT2026.md section 1.6), so
their protein-level codes are ESM-MSR's by construction.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR

os.makedirs(OUT_DIR, exist_ok=True)
long_csv = os.environ.get('LONG_CSV', os.path.join(REPO_ROOT, 'docs', 'benchmarks', 'all_by_all_long.csv'))
long = pd.read_csv(long_csv)[['model', 'dataset', 'rho']]
# ESM3dG's ProteinGym aggregates are printed in BENCHMARKS section 1.4 but absent from the long CSV.
EXTRA_CELLS = [('ESM3dG', 'PG avg'), ('ESM3dG', 'PG Stability'), ('ESM3dG', 'PG DMS mean')]
have = set(zip(long['model'], long['dataset']))
long = pd.concat([long, pd.DataFrame([dict(model=m, dataset=d) for m, d in EXTRA_CELLS if (m, d) not in have])], ignore_index=True)

ov = pd.read_csv(os.path.join(OUT_DIR, 'inventory_overlaps.csv'))
hom = pd.concat([pd.read_csv(os.path.join(OUT_DIR, 'homology_classes.csv')),
                 pd.read_csv(os.path.join(OUT_DIR, 'esm3dg_homology.csv'))], ignore_index=True)
cex = pd.read_csv(os.path.join(OUT_DIR, 'code_inventory_exact.csv'))
ros = pd.read_csv(os.path.join(OUT_DIR, 'rosetta_overlap.csv')).set_index('dataset')

ESM_MSR = ['ESM-MSR (released)', 'ESM-MSR (devel)']
MODEL_INV = {f'{m}{s}': 'esm_msr_capped' for m in ESM_MSR for s in ('', ' WT', ' MT', ' σ=0.25', ' σ=0.5', ' σ=0.75')}
MODEL_INV.update({
    'ESM-MSR-small (March, crystal)': 'esm_msr_small_hyperopt',
    'ESMtherm': 'esmtherm',
    'IFUM (WT backbone)': 'ifum',
    'IFUM (native, ESMFold)': 'ifum',
    'DDGemb': 'ddgemb_s2450',
    'DDMut': 'ddmut',
    'DDMut [partial]': 'ddmut',
    'GeoDDG-3D': 'geoddg_s8754',
    'ESM3dG': 'esm3dg_mgnify',
    'Rosetta': 'rosetta_beta_nov16',
    'ProSST': 'corpus',
    'ProteinMPNN': 'corpus',
    'ProteinMPNN (published)': 'corpus',
    'ESM3-small zero-shot': 'corpus',
    'ESM3-medium zero-shot': 'corpus',
    'ESM3-large zero-shot': 'corpus',
    'ESM3-open (published)': 'corpus',
    'ESM3-small-open (σ=0, MSR readout)': 'corpus',
    'ESM3-small-open (σ=0, MSR readout) WT': 'corpus',
    'ESM3-small-open (σ=0, MSR readout) MT': 'corpus',
})
MEGASCALE_INV = {'esm_msr_capped', 'esm_msr_small_hyperopt'}
# Validation / model-selection split of each in-house inventory (V on the whole dataset)
VAL_SPLIT = {'esm_msr_capped': 'tsu_capped_val'}
# Megascale-trained third-party inventories with an exact-sequence variant join in 10_overlap_engine.py:
# EsmTherm's train sequences are public (verified M, validation-only rows V); IFUM's are reconstructed from its
# domain split (identical-variant ceiling, reported M?; rows in its merged val/test block V?)
UNVERIFIED_M = {'ifum'}

DS = {
    **{k: k for k in ('s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul',
                      'ptmuld', 'tsu_capped_val', 'tsu_capped_test', 'domainome', 'domainome_low_id')},
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
    'MYO_HUMAN_Kung_2025_display': 'MYO',
    'ESTA_BACSU_Nutschel_2020_dTm': 'ESTA',
    'GB1_Wu_2016_binding_domain': 'GB1',
    'PG avg': 'proteingym',
    'PG Stability': 'proteingym',
    'PG DMS mean': 'proteingym',
    'tsu_hyperopt_val': None,
    'tsu_hyperopt_test': None,
}


def pick(df, inv, ds):
    r = df[(df['inventory'] == inv) & (df['dataset'] == ds)]
    return r.iloc[0] if len(r) else None


_N = {}


def nrows(ds):
    if ds not in _N:
        p = os.path.join(WORK_DATA_DIR, 'testsets', ds + '.pkl')
        _N[ds] = len(pd.read_pickle(p)) if os.path.exists(p) else None
    return _N[ds]


rows = []
for r in long.itertuples(index=False):
    inv = MODEL_INV.get(r.model)
    if inv is None or r.dataset not in DS:
        continue  # retrained baselines and rows outside the audited datasets
    ds = DS[r.dataset]
    rec = dict(model=r.model, dataset=r.dataset, rho=r.rho, inventory=inv, n=None, exact=0, constituent=0, P=0, H=0,
               h=0, D=0, V=0, bad=0, note='')
    if ds is None:
        rec.update(code='n.a.', note='legacy hyperopt split: out of the audit scope')
        rows.append(rec)
        continue
    n = rec['n'] = nrows(ds)
    hh = pick(hom, inv, ds)
    if inv in MEGASCALE_INV:
        e = pick(ov, inv, ds)
        if e is not None:
            rec.update(exact=int(e['exact']), constituent=int(e['constituent']), P=int(e['sameprot']))
        if hh is not None:
            rec['H'], rec['h'] = int(hh['n_H']), int(hh['n_h'])
            rec['P'] = max(rec['P'], int(hh['n_sameprot']) - rec['exact'] - rec['constituent'])
    elif inv in ('esmtherm', 'ifum'):
        e = pick(ov, inv, ds)
        if hh is not None:
            rec.update(P=int(hh['n_sameprot']), H=int(hh['n_H']), h=int(hh['n_h']))
        if e is not None:
            rec['exact'], rec['V'] = int(e['exact']), int(e['val'])
            rec['P'] = max(0, rec['P'] - rec['exact'])
            if inv == 'ifum' and rec['V']:
                rec['note'] = "V?: rows of IFUM's merged validation / test domains"
    elif inv == 'esm3dg_mgnify':
        if hh is not None:
            rec.update(P=int(hh['n_sameprot']), H=int(hh['n_H']), h=int(hh['n_h']))
    elif inv in ('ddgemb_s2450', 'ddmut', 'geoddg_s8754'):
        if hh is not None:
            rec.update(P=int(hh['n_sameprot']), H=int(hh['n_H']), h=int(hh['n_h']))
        c = pick(cex, inv, ds)
        if c is not None:
            rec['exact'] = int(c['exact'])
            rec['constituent'] = int(c['constituent']) if pd.notna(c.get('constituent')) else 0
        rec['P'] = max(0, rec['P'] - rec['exact'] - rec['constituent'])
        if inv == 'geoddg_s8754' and ds == 's8754':
            rec.update(exact=n, P=0, bad=1, note='the test set IS the training set')
        if (inv, ds) in {('ddgemb_s2450', 's2648'), ('ddmut', 's2648'), ('ddmut', 's8754')}:
            rec['D'] = 1  # training set documented as derived from this dataset
    elif inv == 'rosetta_beta_nov16':
        if ds in ros.index:
            rec['V'] = int(ros.loc[ds, 'n_tuned'])
            rec['note'] = 'beta_nov16 tuning data: ' + ros.loc[ds, 'basis']
        else:
            rec['note'] = 'beta_nov16 tuning overlap not computed for this dataset'
    if VAL_SPLIT.get(inv) == ds:
        rec['V'] = n
    rows.append(rec)

mx = pd.DataFrame(rows)


def f(x, n):
    v = x / n
    if v == 0:
        return '.000'
    if v < 0.001:
        return '<.001'
    if v < 0.01:
        return f'{v:.3f}'.lstrip('0')
    if v >= 0.995:
        return '1'
    return f'{v:.2f}'.lstrip('0')


def code_for(r):
    if r.get('code') == 'n.a.':
        return 'n.a.'
    n = r['n'] or 0
    parts = []
    mk = 'M?' if r['inventory'] in UNVERIFIED_M else 'M'
    parts += [k + f(r[c], n) for k, c in ((mk, 'exact'), ('m', 'constituent'), ('P', 'P')) if r[c] > 0]
    parts += [k + f(r[k], n) for k in ('H', 'h') if r[k] > 0]
    if r['V']:
        vk = 'V?' if r['inventory'] in UNVERIFIED_M else 'V'
        parts.append(vk if r['V'] >= n else vk + f(r['V'], n))
    if r['D']:
        parts.append('D')
    if r['bad']:
        parts.append('#')
    return ' '.join(parts) if parts else '–'


SUP = {'M': 'ᴹ', 'm': 'ᵐ', 'P': 'ᴾ', 'H': 'ᴴ', 'h': 'ʰ', 'V': 'ⱽ'}


def worst(r):
    """Superscript for the numeric tables: the most severe class reaching 0.5 % of rows."""
    if r.get('code') == 'n.a.':
        return ''
    n = r['n'] or 1
    if r['bad']:
        return '#'
    order = [('M', r['exact']), ('m', r['constituent']), ('P', r['P']), ('H', r['H']), ('h', r['h']), ('V', r['V'])]
    for k, v in order:
        if v / n >= 0.005:
            return SUP[k]
    return 'ᴰ' if r['D'] else ''


mx['code'] = mx.apply(code_for, axis=1)
mx['sup'] = mx.apply(worst, axis=1)
mx.to_csv(os.path.join(OUT_DIR, 'leak_matrix.csv'), index=False)
print('saved leak_matrix.csv:', len(mx), 'cells')
