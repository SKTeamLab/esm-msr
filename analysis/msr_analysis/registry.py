"""Where the data is, which datasets and models exist, and how each model is drawn.

Everything outside the repository is found through one root directory (default
``~/playground/esm-msr-devel``; override with ``MSR_ROOT``) laid out as described in
``msr_predictions/README.md`` and ``baselines/{retrained,public}/README.md``:

    <root>/msr_predictions/<spec>[/seed<k>]/sigma<s>/<dataset>.csv     ESM-MSR
    <root>/baselines/retrained/{mutate_everything,thermompnn,spurs}/predictions/...
    <root>/baselines/public/<model>/<dataset>.csv                      third-party harness layout
    <root>/baselines/public/_tasks/<dataset>.csv.gz                    the label "spine" of every dataset
    <root>/baselines/pgym_per_dms/*.csv                                per-DMS ProteinGym rho tables
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('MSR_ROOT', '~/playground/esm-msr-devel')).expanduser()
MSR = ROOT / 'msr_predictions'
P0 = ROOT / 'final_bench' / 'predictions'        # FINAL_p0 (seed 1 only): <P0>/<folder>/final_checkpoints/FINAL_p0/s<seed>/
BASE = ROOT / 'baselines'
RETRAINED = BASE / 'retrained'
MEGA_RAW = RETRAINED / 'mega_splits' / 'tracked'     # the cluster's mega_splits retrain runs (outputs/tracked/*mega*), as collected
PUBLIC = BASE / 'public'
TASKS = PUBLIC / '_tasks'
PGYM = BASE / 'pgym_per_dms'

HERE = Path(__file__).resolve().parent.parent          # analysis/
CACHE = HERE / 'cache'
REFERENCE = HERE / 'reference'
RESULTS = HERE / 'results'
if os.environ.get('MSR_TSU_SPLIT') == 'mega':
    RESULTS = RESULTS / '03_tsuboyama' / 'mega_splits'

# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
EXT12 = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
DMS3 = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']
TSU_CAPPED = ['tsuboyama_capped_val', 'tsuboyama_capped_test']
TSU_MEGA = ['tsuboyama_mega_val', 'tsuboyama_mega_test']
# MSR_TSU_SPLIT=mega re-points every Tsuboyama analysis at the full-MegaScale split (`mega_splits`): TSU becomes the mega pair, the
# models are the mega_splits retrains (ESM-MSR = FINAL_p0_mega under the `msrp0` keys) and RESULTS moves to
# results/03_tsuboyama/mega_splits/ (analysis/tsuboyama/mega_splits/run_all.sh).
TSU_SPLIT = os.environ.get('MSR_TSU_SPLIT', 'capped')
if TSU_SPLIT not in ('capped', 'mega'):
    raise ValueError(f'MSR_TSU_SPLIT must be capped or mega, not {TSU_SPLIT!r}')
TSU = TSU_MEGA if TSU_SPLIT == 'mega' else TSU_CAPPED
ALL_DATASETS = EXT12 + DMS3 + ['domainome'] + TSU

LABEL = {
    's571': 'S571', 's783': 'S783', 's2648': 'S2648', 's8754': 'S8754', 's669': 'S669', 's461': 'S461',
    'ssym': 'Ssym', 'q3421': 'Q3421', 'k3822': 'K3822', 'k2369': 'K2369', 'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.', 'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
    'MYO_HUMAN_Kung_2025_display': 'MYO', 'domainome': 'Domainome',
    'tsuboyama_capped_val': 'Tsuboyama val', 'tsuboyama_capped_test': 'Tsuboyama test',
    'tsuboyama_mega_val': 'Tsuboyama val (mega_splits)', 'tsuboyama_mega_test': 'Tsuboyama test (mega_splits)',
}
# per-protein-mean datasets (the metric is the mean over proteins of a per-protein statistic)
PER_PROTEIN = {'domainome', *TSU_CAPPED, *TSU_MEGA}
# the spine's `id` is not unique in some datasets; the column that names a protein for clustering
TSU_CODE = 'code_wt'

# ----------------------------------------------------------------------------------------------
# models.  A model key is "<family>[@sigma]:<head>"; the family says where the file is.
# ----------------------------------------------------------------------------------------------
# display: label, colour, marker, fill, group
#   group: 'msr' ESM-MSR; 'retrain' retrained on Homology-Restricted MegaScale; 'public' held-out or leaky third party;
#          'zeroshot' no training on stability data
C = dict(msr='#7f8c8d', msr_dev='#000000', msr_p0='#6c3483', me1='#2ecc71', me2='#1b8a4b', tm='#3498db', tmd='#1f5f8b',
         spurs='#d35400', spursm='#e59866', zero='#888888', pmpnn='#34495e', prosst='#8e44ad',
         esm3='#e74c3c', ros='#1abc9c', other='#95a5a6')

MODELS = {
    'msr': dict(label='ESM-MSR (baseline; WT head, combined for multi-mutant sets)', color=C['msr'], marker='o', fill=True, group='msr'),
    # --- ESM-MSR -------------------------------------------------------------------------
    'msr:wt':      dict(label='ESM-MSR (baseline) WT head', color=C['msr'], marker='o', fill=False, group='msr'),
    'msr:comb':    dict(label='ESM-MSR (baseline) combined', color=C['msr'], marker='D', fill=True, group='msr'),
    'msrdev:wt':   dict(label='ESM-MSR WT head', color=C['msr_dev'], marker='o', fill=False, group='msr'),
    'msrdev:comb': dict(label='ESM-MSR combined', color=C['msr_dev'], marker='D', fill=True, group='msr'),
    'msrp0:wt':    dict(label='ESM-MSR (p0) WT head', color=C['msr_p0'], marker='o', fill=False, group='msr'),
    'msrp0:comb':  dict(label='ESM-MSR (p0) combined', color=C['msr_p0'], marker='D', fill=True, group='msr'),
    'msrrel:wt':   dict(label='ESM-MSR (released) WT head', color=C['msr_dev'], marker='o', fill=False, group='msr'),
    'msrrel:comb': dict(label='ESM-MSR (released) combined', color=C['msr_dev'], marker='D', fill=True, group='msr'),
    'msrsmall:comb': dict(label='ESM-MSR-small (March)', color='#7f8c8d', marker='D', fill=False, group='msr'),
    'msr:epi': dict(label='ESM-MSR (baseline) epistasis (1/2(MT_AB+WT_AB) − WT_A − WT_B)', color=C['msr'], marker='D', fill=True, group='msr'),
    'msrp0:epi': dict(label='ESM-MSR (p0) epistasis (1/2(MT_AB+WT_AB) − WT_A − WT_B)', color=C['msr_p0'], marker='D', fill=True, group='msr'),
    'me2:epi': dict(label='MutateEverything st2 (vs st1 singles)', color=C['me2'], marker='H', fill=True, group='retrain'),
    'tmpnnd:epi': dict(label='ThermoMPNN-D (vs ThermoMPNN)', color=C['tmd'], marker='P', fill=True, group='retrain'),
    'spursm:epi': dict(label='SPURS multi (vs SPURS singles)', color=C['spursm'], marker='s', fill=False, group='retrain'),
    # --- retrained on Homology-Restricted MegaScale ----------------------------------------------------
    'me1':     dict(label='MutateEverything st1', color=C['me1'], marker='h', fill=True, group='retrain'),
    'me2':     dict(label='MutateEverything st2', color=C['me2'], marker='H', fill=True, group='retrain'),
    'me1+':    dict(label='MutateEverything st1, additive', color=C['me1'], marker='h', fill=False, group='retrain'),
    'me2+':    dict(label='MutateEverything st2, additive', color=C['me2'], marker='H', fill=False, group='retrain'),
    'tmpnn':   dict(label='ThermoMPNN', color=C['tm'], marker='p', fill=True, group='retrain'),
    'tmpnnd':  dict(label='ThermoMPNN + ThermoMPNN-D', color=C['tmd'], marker='P', fill=True, group='retrain'),
    'spurs':   dict(label='SPURS', color=C['spurs'], marker='s', fill=True, group='retrain'),
    'spursm':  dict(label='SPURS multi', color=C['spursm'], marker='s', fill=False, group='retrain'),
    # --- mega_splits only: other SPURS / ThermoMPNN versions (data._load_mega) -------------------------
    'spursu':    dict(label='SPURS, trained adapter (retrain)', color='#a04000', marker='s', fill=True, group='retrain'),
    'spursum':   dict(label='SPURS multi, trained adapter (retrain)', color='#dc7633', marker='s', fill=False, group='retrain'),
    'spursum:epi': dict(label='SPURS multi, trained adapter (vs its singles)', color='#dc7633', marker='s', fill=False, group='retrain'),
    'spurspub':  dict(label='SPURS, published checkpoint', color='#6e2c00', marker='*', fill=True, group='public'),
    'spurspubm': dict(label='SPURS multi, published checkpoint', color='#b9770e', marker='*', fill=False, group='public'),
    'spurspubm:epi': dict(label='SPURS multi, published (vs its singles)', color='#b9770e', marker='*', fill=False, group='public'),
    'tmpnnpub':  dict(label='ThermoMPNN, published (3 ensemble members)', color='#154360', marker='p', fill=False, group='public'),
    'tmpnndpub': dict(label='ThermoMPNN-D, published (3 ensemble members)', color='#5dade2', marker='P', fill=False, group='public'),
    'tmpnndpub:epi': dict(label='ThermoMPNN-D, published (vs ThermoMPNN)', color='#5dade2', marker='P', fill=False, group='public'),
    # --- third party ------------------------------------------------------------------------
    'esmtherm':  dict(label='ESMtherm*', color='#7d6608', marker='<', fill=True, group='public'),
    'ddgemb':    dict(label='DDGemb', color='#b7950b', marker='>', fill=True, group='public'),
    'ddmut':     dict(label='DDMut', color='#d4ac0d', marker='>', fill=False, group='public'),
    'ifum_wt':   dict(label='IFUM, WT backbone*', color='#6c3483', marker='v', fill=True, group='public'),
    'ifum_esmfold': dict(label='IFUM, ESMFold*', color='#a569bd', marker='v', fill=False, group='public'),
    'geoddg':    dict(label='GeoDDG-3D', color='#117a65', marker='^', fill=True, group='public'),
    'esm3dg':    dict(label='ESM3ΔG', color='#cb4335', marker='*', fill=True, group='public'),
    'rosetta':   dict(label='Rosetta cart. ΔΔG', color=C['ros'], marker='+', fill=True, group='public'),
    # --- zero-shot ------------------------------------------------------------------------------
    'prosst':    dict(label='ProSST (zero-shot)', color=C['prosst'], marker='X', fill=True, group='zeroshot'),
    'pmpnn':     dict(label='ProteinMPNN (zero-shot)', color=C['pmpnn'], marker='d', fill=True, group='zeroshot'),
    'esm3s':     dict(label='ESM3-small (Forge)', color='#f1948a', marker='o', fill=True, group='zeroshot'),
    'esm3m':     dict(label='ESM3-medium (Forge)', color='#e74c3c', marker='o', fill=True, group='zeroshot'),
    'esm3l':     dict(label='ESM3-large (Forge)', color='#922b21', marker='o', fill=True, group='zeroshot'),
    'msrrel@0:comb': dict(label='ESM3-small-open, σ = 0 (zero-shot)', color=C['zero'], marker='^', fill=True, group='zeroshot'),
}
for _s in ('0.25', '0.5', '0.75'):
    for _fam, _lab in (('msrdev', ''), ('msrrel', ' (released)')):
        for _h, _hl in (('wt', 'WT head'), ('comb', 'combined')):
            MODELS[f'{_fam}@{_s}:{_h}'] = dict(label=f'ESM-MSR{_lab} σ = {_s} {_hl}', color=C['msr_dev'],
                                               marker='o' if _h == 'wt' else 'D', fill=False, group='msr')
MODELS['msrrel@0:wt'] = dict(label='ESM3-small-open, σ = 0 WT head', color=C['zero'], marker='^', fill=False, group='zeroshot')

# sets of keys used by the figure scripts -----------------------------------------------------------
# Models whose training data include Megascale (Tsuboyama) libraries: not held out on the Tsuboyama sets, and drawn last, greyed, with an asterisk
# (ESM3ΔG is not in the list: it is trained on the MGnify stability set and Megascale is its held-out test, BENCHMARKS section 6.3)
MEGASCALE_TRAINED = ['esmtherm', 'ifum_wt', 'ifum_esmfold']
MEGASCALE_NOTE = '* trained on Megascale (Tsuboyama) data: not held out'

RETRAINS = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
PUBLIC_MODELS = ['esmtherm', 'ddgemb', 'ddmut', 'ifum_wt', 'ifum_esmfold', 'geoddg', 'esm3dg']
ZEROSHOT = ['prosst', 'pmpnn', 'esm3s', 'esm3m', 'esm3l', 'msrrel@0:comb']


def label(key):
    return MODELS[key]['label'] if key in MODELS else key


def is_msr(key):
    return key.split(':')[0].split('@')[0].startswith('msr')
