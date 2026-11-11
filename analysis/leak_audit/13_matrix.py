"""Assemble the per-(model, dataset) leak matrix with the audit notation.

Uses ``–`` (no supervised overlap) in place of the former ``C`` pretraining-corpus flag for
zero-shot / unsupervised models (ProSST, ProteinMPNN, ESM3 zero-shot, ESM3 σ=0 readout), and maps
benchmark SPURS and ThermoMPNN-D rows to their Homology-Restricted MegaScale retrained inventories.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import OUT_DIR, REPO_ROOT, WORK_DATA_DIR

os.makedirs(OUT_DIR, exist_ok=True)
long_csv = os.environ.get('LONG_CSV', os.path.join(REPO_ROOT, 'docs', 'benchmarks', 'all_by_all_long.csv'))
long = pd.read_csv(long_csv)
for _pg_ds, _pg_rho in [('PG avg', 0.380), ('PG Stability', 0.788), ('PG DMS mean', 0.441)]:
    if not ((long['model'] == 'ESM3dG') & (long['dataset'] == _pg_ds)).any():
        long = pd.concat([long, pd.DataFrame([dict(model='ESM3dG', dataset=_pg_ds, rho=_pg_rho)])], ignore_index=True)
ov = pd.read_csv(os.path.join(OUT_DIR, 'inventory_overlaps.csv'))
hom = pd.read_csv(os.path.join(OUT_DIR, 'homology_classes.csv'))
cex = pd.read_csv(os.path.join(OUT_DIR, 'code_inventory_exact.csv'))

MODEL_INV = {
    'ESM-MSR (released)': 'esm_msr_capped',
    'ESM-MSR (released) WT': 'esm_msr_capped',
    'ESM-MSR (released) MT': 'esm_msr_capped',
    'ESM-MSR (released) σ=0.25': 'esm_msr_capped',
    'ESM-MSR (released) σ=0.5': 'esm_msr_capped',
    'ESM-MSR (released) σ=0.75': 'esm_msr_capped',
    'ESM-MSR (devel)': 'esm_msr_capped',
    'ESM-MSR (devel) WT': 'esm_msr_capped',
    'ESM-MSR (devel) MT': 'esm_msr_capped',
    'ESM-MSR (devel) σ=0.25': 'esm_msr_capped',
    'ESM-MSR (devel) σ=0.5': 'esm_msr_capped',
    'ESM-MSR (devel) σ=0.75': 'esm_msr_capped',
    'ME stage1': 'me_stage1',
    'ME stage1+': 'me_stage1',
    'ME stage2': 'me_stage2',
    'ME stage2+': 'me_stage2',
    'TMPNN': 'tmpnn',
    'TMPNN+D': 'tmpnnd',
    'SPURS': 'spurs_retrain_single',
    'SPURS multi': 'spurs_retrain_multi',
    'SPURS (retrained)': 'spurs_retrain_single',
    'SPURS multi (retrained)': 'spurs_retrain_multi',
    'SPURS (public checkpoint)': 'spurs_single',
    'SPURS multi (public checkpoint)': 'spurs_multi',
    'ESMtherm': 'esmtherm',
    'DDGemb': 'ddgemb_s2450',
    'DDMut': 'ddmut_s9028',
    'DDMut [partial]': 'ddmut_s9028',
    'GeoDDG-3D': 'geoddg_s8754',
    'IFUM (WT backbone)': 'ifum',
    'IFUM (native, ESMFold)': 'ifum',
    'ESM3dG': 'esm3dg_mgnify',
    'ESM-MSR-small (March, crystal)': 'esm_msr_small_hyperopt',
    'ProSST': 'corpus',
    'ProteinMPNN': 'corpus',
    'ProteinMPNN (published)': 'corpus',
    'Rosetta': 'none',
    'ESM3-small zero-shot': 'corpus',
    'ESM3-medium zero-shot': 'corpus',
    'ESM3-large zero-shot': 'corpus',
    'ESM3-open (published)': 'corpus',
    'ESM3-small-open (σ=0, MSR readout)': 'corpus',
    'ESM3-small-open (σ=0, MSR readout) WT': 'corpus',
    'ESM3-small-open (σ=0, MSR readout) MT': 'corpus',
}

DS = {
    's571': 's571',
    's783': 's783',
    's2648': 's2648',
    's8754': 's8754',
    's669': 's669',
    's461': 's461',
    'ssym': 'ssym',
    'q3421': 'q3421',
    'k3822': 'k3822',
    'k2369': 'k2369',
    'ptmul': 'ptmul',
    'ptmuld': 'ptmuld',
    'tsu_capped_val': 'tsu_capped_val',
    'tsu_capped_test': 'tsu_capped_test',
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
    'MYO_HUMAN_Kung_2025_display': 'MYO',
    'ESTA_BACSU_Nutschel_2020_dTm': 'ESTA',
    'GB1_Wu_2016_binding_domain': 'GB1',
    'domainome': 'domainome',
    'PG avg': 'proteingym',
    'PG Stability': 'proteingym',
    'PG DMS mean': 'proteingym',
    'tsu_hyperopt_val': 'HYPEROPT',
    'tsu_hyperopt_test': 'HYPEROPT',
}


def eng(inv, ds):
    r = ov[(ov['inventory'] == inv) & (ov['dataset'] == ds)]
    return r.iloc[0] if len(r) else None


def hm(inv, ds):
    r = hom[(hom['inventory'] == inv) & (hom['dataset'] == ds)]
    return r.iloc[0] if len(r) else None


def cx(inv, ds):
    r = cex[(cex['inventory'] == inv) & (cex['dataset'] == ds)]
    return r.iloc[0] if len(r) else None


_N_CACHE = {}


def nrows(ds):
    if ds not in _N_CACHE:
        p = os.path.join(WORK_DATA_DIR, 'testsets', ds + '.pkl')
        _N_CACHE[ds] = len(pd.read_pickle(p)) if os.path.exists(p) else None
    return _N_CACHE[ds]


rows = []
for _, r in long.iterrows():
    model = r['model']
    dn = r['dataset']
    inv = MODEL_INV.get(model)
    ds = DS.get(dn)
    if not inv or not ds:
        continue
    rec = dict(
        model=model, dataset=dn, rho=r['rho'], inventory=inv, n=None,
        exact=0, constituent=0, P=0, H=0, h=0, C=0.0, D=0, U=0, V=0, bad=0, note='',
    )
    if ds == 'HYPEROPT':
        rec['note'] = 'legacy hyperopt split — out of scope'
        rec['code'] = '–'
        rows.append(rec)
        continue

    n = nrows(ds)
    rec['n'] = n
    if inv in (
        'esm_msr_capped', 'me_stage1', 'me_stage2', 'tmpnn', 'tmpnnd',
        'spurs_retrain_single', 'spurs_retrain_multi', 'spurs_single', 'spurs_multi',
        'esm3dg_released', 'esm_msr_small_hyperopt',
    ):
        e = eng(inv, ds)
        hh = hm(inv, ds)
        if e is not None:
            rec['exact'] = int(e['exact'])
            rec['constituent'] = int(e['constituent'])
            rec['P'] = int(e['sameprot'])
        if hh is not None:
            rec['H'] = int(hh['n_H'])
            rec['h'] = int(hh['n_h'])
            rec['P'] = max(rec['P'], int(hh['n_sameprot']) - rec['exact'] - rec['constituent'])
    elif inv in ('esmtherm', 'ifum', 'esm3dg_mgnify'):
        hh = hm(inv, ds)
        if hh is not None:
            rec['P'] = int(hh['n_sameprot'])
            rec['H'] = int(hh['n_H'])
            rec['h'] = int(hh['n_h'])
    elif inv in ('ddgemb_s2450', 'ddmut_s9028', 'geoddg_s8754'):
        hh = hm(inv, ds)
        if hh is not None:
            rec['P'] = int(hh['n_sameprot'])
            rec['H'] = int(hh['n_H'])
            rec['h'] = int(hh['n_h'])
        c = cx(inv, ds)
        if c is not None:
            rec['exact'] = int(c['exact'])
            rec['constituent'] = int(c['constituent']) if pd.notna(c.get('constituent')) else 0
        if inv == 'ddmut_s9028':
            # DDMut dispatches multi-mutant rows to its SM1242 multi-mutant branch
            c2 = cx('ddmut_sm1242', ds)
            hh2 = hm('ddmut_sm1242', ds)
            if c2 is not None and int(c2['exact']) > 0:
                add_ex = int(c2['exact'])
                rec['exact'] += add_ex
                rec['constituent'] = max(0, rec['constituent'] - add_ex)
            if hh2 is not None and ds in ('ptmul', 'ptmuld'):
                rec['P'] = max(rec['P'], int(hh2['n_sameprot']))
                rec['H'] = max(rec['H'], int(hh2['n_H']))
                rec['h'] = max(rec['h'], int(hh2['n_h']))
        rec['P'] = max(0, rec['P'] - rec['exact'] - rec['constituent'])
        if inv == 'geoddg_s8754' and ds == 's8754':
            rec['exact'] = n
            rec['P'] = 0
            rec['bad'] = 1
            rec['note'] = 'the test set IS the training set'
        if inv == 'ddgemb_s2450' and ds == 's2648':
            rec['D'] = 1
        if inv == 'ddmut_s9028' and ds in ('s2648', 's8754'):
            rec['D'] = 1
    elif inv in ('corpus', 'none'):
        # Pretraining-corpus-only / physics-based models carry no supervised ΔΔG training overlap ('–')
        pass

    if ds == 'tsu_capped_val' and inv not in ('corpus', 'none'):
        rec['V'] = 1
    rows.append(rec)

mx = pd.DataFrame(rows)


def f(x, n):
    if not n:
        return ''
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
    if r.get('code') == '–':
        return '–'
    n = r['n'] or 0
    parts = []
    if r['inventory'] in ('esmtherm', 'ifum') and r['P'] > 0:
        m_est = int(r['P'] * (0.8 if r['inventory'] == 'esmtherm' else 1.0))
        parts.append('P' + f(r['P'], n))
        parts.append('M?' + f(m_est, n))
    else:
        if r['exact'] > 0:
            parts.append('M' + f(r['exact'], n))
        if r['constituent'] > 0:
            parts.append('m' + f(r['constituent'], n))
        if r['P'] > 0:
            parts.append('P' + f(r['P'], n))
    if r['H'] > 0:
        parts.append('H' + f(r['H'], n))
    if r['h'] > 0:
        parts.append('h' + f(r['h'], n))
    if r['V']:
        parts.append('V')
    if r['D']:
        parts.append('D')
    if r['bad']:
        parts.append('#')
    return ' '.join(parts) if parts else '–'


def worst(r):
    if r.get('code') == '–':
        return ''
    n = r['n'] or 1
    if r['bad']:
        return '#'
    if r['inventory'] in ('esmtherm', 'ifum'):
        if r['P'] / n >= 0.005:
            return 'ᴾ'
        for k, v in [('H', r['H']), ('h', r['h'])]:
            if v / n >= 0.005:
                return {'H': 'ᴴ', 'h': 'ʰ'}[k]
        return ''
    for k, v in [('M', r['exact']), ('m', r['constituent']), ('P', r['P']), ('H', r['H']), ('h', r['h'])]:
        if v / n >= 0.005:
            return {'M': 'ᴹ', 'm': 'ᵐ', 'P': 'ᴾ', 'H': 'ᴴ', 'h': 'ʰ'}[k]
    if r['D']:
        return 'ᴰ'
    return ''


mx['code'] = mx.apply(code_for, axis=1)
mx['sup'] = mx.apply(worst, axis=1)
mx.to_csv(os.path.join(OUT_DIR, 'leak_matrix.csv'), index=False)
print('saved leak_matrix.csv:', len(mx), 'cells')
