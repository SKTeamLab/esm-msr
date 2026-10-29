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
BASE = ROOT / 'baselines'
RETRAINED = BASE / 'retrained'
PUBLIC = BASE / 'public'
TASKS = PUBLIC / '_tasks'
PGYM = BASE / 'pgym_per_dms'

HERE = Path(__file__).resolve().parent.parent          # analysis/
CACHE = HERE / 'cache'
REFERENCE = HERE / 'reference'
RESULTS = HERE / 'results'

# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
EXT12 = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
DMS7 = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
        'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
        'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']
TSU = ['tsuboyama_capped_val', 'tsuboyama_capped_test']
ALL_DATASETS = EXT12 + DMS7 + ['domainome'] + TSU

LABEL = {
    's571': 'S571', 's783': 'S783', 's2648': 'S2648', 's8754': 'S8754', 's669': 'S669', 's461': 'S461',
    'ssym': 'Ssym', 'q3421': 'Q3421', 'k3822': 'K3822', 'k2369': 'K2369', 'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D',
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.', 'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.', 'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
    'MYO_HUMAN_Kung_2025_display': 'MYO', 'ESTA_BACSU_Nutschel_2020_dTm': 'ESTA dTm',
    'GB1_Wu_2016_binding_domain': 'GB1', 'domainome': 'Domainome',
    'tsuboyama_capped_val': 'Tsuboyama val', 'tsuboyama_capped_test': 'Tsuboyama test',
}
# per-protein-mean datasets (the metric is the mean over proteins of a per-protein statistic)
PER_PROTEIN = {'domainome', 'tsuboyama_capped_val', 'tsuboyama_capped_test'}
# the spine's `id` is not unique in some datasets; the column that names a protein for clustering
TSU_CODE = 'code_wt'

# ----------------------------------------------------------------------------------------------
# models.  A model key is "<family>[@sigma]:<head>"; the family says where the file is.
# ----------------------------------------------------------------------------------------------
# display: label, colour, marker, fill, group
#   group: 'msr' ESM-MSR; 'retrain' retrained on the capped split; 'public' held-out or leaky third party;
#          'zeroshot' no training on stability data
C = dict(msr='#000000', msr_dev='#555555', me1='#2ecc71', me2='#1b8a4b', tm='#3498db', tmd='#1f5f8b',
         spurs='#d35400', spursm='#e59866', zero='#888888', pmpnn='#34495e', prosst='#8e44ad',
         esm3='#e74c3c', ros='#1abc9c', other='#95a5a6')

MODELS = {
    'msr': dict(label='ESM-MSR (WT head; combined for multi-mutant sets)', color=C['msr'], marker='o', fill=True, group='msr'),
    # --- ESM-MSR -------------------------------------------------------------------------
    'msr:wt':      dict(label='ESM-MSR WT head', color=C['msr'], marker='o', fill=False, group='msr'),
    'msr:comb':    dict(label='ESM-MSR combined', color=C['msr'], marker='D', fill=True, group='msr'),
    'msrdev:wt':   dict(label='ESM-MSR (devel) WT head', color=C['msr_dev'], marker='o', fill=False, group='msr'),
    'msrdev:comb': dict(label='ESM-MSR (devel) combined', color=C['msr_dev'], marker='D', fill=True, group='msr'),
    'msrrel:wt':   dict(label='ESM-MSR (released) WT head', color=C['msr_dev'], marker='o', fill=False, group='msr'),
    'msrrel:comb': dict(label='ESM-MSR (released) combined', color=C['msr_dev'], marker='D', fill=True, group='msr'),
    'msrsmall:comb': dict(label='ESM-MSR-small (March)', color='#7f8c8d', marker='D', fill=False, group='msr'),
    'msr:epi': dict(label='ESM-MSR epistasis head (MT − WT)', color=C['msr'], marker='D', fill=True, group='msr'),
    'msr:cdddg': dict(label='ESM-MSR combined, vs additive', color='#555555', marker='D', fill=False, group='msr'),
    'me2:epi': dict(label='MutateEverything st2 (vs st1 singles)', color=C['me2'], marker='H', fill=True, group='retrain'),
    'tmpnnd:epi': dict(label='ThermoMPNN-D (vs ThermoMPNN)', color=C['tmd'], marker='P', fill=True, group='retrain'),
    'spursm:epi': dict(label='SPURS multi (vs SPURS singles)', color=C['spursm'], marker='s', fill=False, group='retrain'),
    # --- retrained on the capped split ----------------------------------------------------
    'me1':     dict(label='MutateEverything st1', color=C['me1'], marker='h', fill=True, group='retrain'),
    'me2':     dict(label='MutateEverything st2', color=C['me2'], marker='H', fill=True, group='retrain'),
    'me1+':    dict(label='MutateEverything st1, additive', color=C['me1'], marker='h', fill=False, group='retrain'),
    'me2+':    dict(label='MutateEverything st2, additive', color=C['me2'], marker='H', fill=False, group='retrain'),
    'tmpnn':   dict(label='ThermoMPNN', color=C['tm'], marker='p', fill=True, group='retrain'),
    'tmpnnd':  dict(label='ThermoMPNN + ThermoMPNN-D', color=C['tmd'], marker='P', fill=True, group='retrain'),
    'spurs':   dict(label='SPURS', color=C['spurs'], marker='s', fill=True, group='retrain'),
    'spursm':  dict(label='SPURS multi', color=C['spursm'], marker='s', fill=False, group='retrain'),
    # --- third party ------------------------------------------------------------------------
    'esmtherm':  dict(label='ESMtherm*', color='#7d6608', marker='<', fill=True, group='public'),
    'ddgemb':    dict(label='DDGemb', color='#b7950b', marker='>', fill=True, group='public'),
    'ddmut':     dict(label='DDMut', color='#d4ac0d', marker='>', fill=False, group='public'),
    'ifum_wt':   dict(label='IFUM, WT backbone*', color='#6c3483', marker='v', fill=True, group='public'),
    'ifum_esmfold': dict(label='IFUM, ESMFold*', color='#a569bd', marker='v', fill=False, group='public'),
    'geoddg':    dict(label='GeoDDG-3D', color='#117a65', marker='^', fill=True, group='public'),
    'esm3dg':    dict(label='ESM3ΔG*', color='#cb4335', marker='*', fill=True, group='public'),
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
    for _fam, _lab in (('msrdev', 'devel'), ('msrrel', 'released')):
        for _h, _hl in (('wt', 'WT head'), ('comb', 'combined')):
            MODELS[f'{_fam}@{_s}:{_h}'] = dict(label=f'ESM-MSR ({_lab}) σ = {_s} {_hl}', color=C['msr_dev'],
                                               marker='o' if _h == 'wt' else 'D', fill=False, group='msr')
MODELS['msrrel@0:wt'] = dict(label='ESM3-small-open, σ = 0 WT head', color=C['zero'], marker='^', fill=False, group='zeroshot')

# sets of keys used by the figure scripts -----------------------------------------------------------
# Models whose training data include Megascale (Tsuboyama) libraries: not held out on the Tsuboyama sets, and drawn last, greyed, with an asterisk
# (BENCHMARKS section 6.3 lists ESM3ΔG as MGnify-trained with Megascale held out; it is flagged here on the project owner's instruction)
MEGASCALE_TRAINED = ['esmtherm', 'esm3dg', 'ifum_wt', 'ifum_esmfold']
MEGASCALE_NOTE = '* trained on Megascale (Tsuboyama) data: not held out'

RETRAINS = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
PUBLIC_MODELS = ['esmtherm', 'ddgemb', 'ddmut', 'ifum_wt', 'ifum_esmfold', 'geoddg', 'esm3dg']
ZEROSHOT = ['prosst', 'pmpnn', 'esm3s', 'esm3m', 'esm3l', 'msrrel@0:comb']


def label(key):
    return MODELS[key]['label'] if key in MODELS else key


def is_msr(key):
    return key.split(':')[0].split('@')[0].startswith('msr')
