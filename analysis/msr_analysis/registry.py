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
P0 = ROOT / 'final_bench' / 'predictions'        # FINAL_p0 (seeds 1-3): <P0>/<folder>/final_checkpoints/FINAL_p0/s<seed>/
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
# models.  A key is "<model>[@sigma]:<version>" (``msr_analysis.data``); the roster (``msr_analysis.models``) says which model it belongs to and how that model is drawn:
# the same colour, marker, order and version names in every figure.  ESM-MSR is FINAL_p0 and is never labelled p0; no other ESM-MSR version is registered.
# ----------------------------------------------------------------------------------------------
from . import models as _R  # noqa: E402

# group: 'msr' ESM-MSR; 'retrain' the other transfer models; 'public' supervised and Megascale-trained third-party models; 'zeroshot' no training on stability data
_GROUP = {'zeroshot': 'zeroshot', 'transfer': 'retrain', 'supervised': 'public', 'megascale': 'public'}
MODELS = {}


def _reg(key, label, name, fill=None, color=None):
    m = _R.BY_NAME[name]
    MODELS[key] = dict(label=label, color=color or m.color, marker=m.marker, fill=(key not in _R.SINGLES_KEYS) if fill is None else fill, group='msr' if name == 'ESM-MSR' else _GROUP[m.group], model=name, ls=m.ls)


def _version(name, i, key=None):
    return _R.version_label(name, i)


# --- ESM-MSR and ESM3-small-open (the sigma = 0 reading of the same checkpoint): WT-view = the WT head, Dual-view = the combined head --------------------------
for _name, _tag in (('ESM-MSR', 'msrp0'), ('ESM3-small-open', 'msrrel@0')):
    _reg(f'{_tag}:wt', _version(_name, 0), _name)
    _reg(f'{_tag}:comb', _version(_name, 1), _name)
    _reg(f'{_tag}:epi', f'{_name} epistasis (comb_AB − WT_A − WT_B)', _name, fill=True)
    _reg(f'{_tag}:wt+', f'{_name} WT-view, sum of the singles', _name, fill=False)
# --- the other transfer models: the singles version is hollow, the doubles version filled -------------------------------------------------------------------------
for _key, _name, _i in (('me1', 'MutateEverything', 0), ('me2', 'MutateEverything', 1), ('tmpnn', 'ThermoMPNN', 0), ('tmpnnd', 'ThermoMPNN', 1), ('spurs', 'SPURS', 0), ('spursm', 'SPURS', 1)):
    _reg(_key, _version(_name, _i), _name)
_reg('me1+', 'MutateEverything Stage 1, additive', 'MutateEverything', fill=False)
_reg('me2+', 'MutateEverything Stage 2, additive', 'MutateEverything', fill=False)
_reg('me2:epi', 'MutateEverything Stage 2 − Stage 1', 'MutateEverything', fill=True)
_reg('tmpnnd:epi', 'ThermoMPNN-D − ThermoMPNN', 'ThermoMPNN', fill=True)
_reg('spursm:epi', 'SPURS-multi − SPURS', 'SPURS', fill=True)
# --- mega_splits only (``data._load_mega``): the SPURS adapter trained (the paper's configuration), the published checkpoints; a lighter shade of the model's colour ----------------------
_reg('spursu', 'SPURS, trained adapter (retrain)', 'SPURS')
_reg('spursum', 'SPURS-multi, trained adapter (retrain)', 'SPURS')
_reg('spursum:epi', 'SPURS-multi, trained adapter (vs its singles)', 'SPURS', fill=True)
_reg('spurspub', 'SPURS, published checkpoint*', 'SPURS published*')
_reg('spurspubm', 'SPURS-multi, published checkpoint*', 'SPURS published*')
_reg('spurspubm:epi', 'SPURS-multi, published (vs its singles)*', 'SPURS published*', fill=True)
_reg('tmpnnpub', 'ThermoMPNN, published (3 ensemble members)', 'ThermoMPNN', color=_R.tint(_R.color('ThermoMPNN'), 0.4))
_reg('tmpnndpub', 'ThermoMPNN-D, published (3 ensemble members)', 'ThermoMPNN', color=_R.tint(_R.color('ThermoMPNN'), 0.4))
_reg('tmpnndpub:epi', 'ThermoMPNN-D, published (vs ThermoMPNN)', 'ThermoMPNN', fill=True, color=_R.tint(_R.color('ThermoMPNN'), 0.4))
# --- one version: supervised, not trained on Megascale; zero-shot; trained on Megascale ------------------------------------------------------------------------------------
_reg('geoddg', 'GeoDDG-3D', 'GeoDDG-3D')
_reg('geoddg+', 'GeoDDG-3D, sum of the singles', 'GeoDDG-3D', fill=False)
_reg('rosetta', 'Rosetta cart. ΔΔG', 'Rosetta')
_reg('rosetta+', 'Rosetta cart. ΔΔG, sum of the singles', 'Rosetta', fill=False)
_reg('esm3dg', 'ESM3ΔG', 'ESM3ΔG')
_reg('prosst', 'ProSST', 'ProSST')
_reg('pmpnn', 'ProteinMPNN', 'ProteinMPNN')
_reg('pmpnn+', 'ProteinMPNN, sum of the singles', 'ProteinMPNN', fill=False)
_reg('esmtherm', 'ESMtherm*', 'ESMtherm')
_reg('ifum_wt', 'IFUM, WT backbone*', 'IFUM')
_reg('ifum_esmfold', 'IFUM, ESMFold*', 'IFUM', fill=False, color=_R.tint(_R.color('IFUM'), 0.35))
_reg('ddgemb', 'DDGemb', 'DDGemb')
_reg('ddmut', 'DDMut', 'DDMut', fill=False)

# sets of keys used by the figure scripts -----------------------------------------------------------
# Models whose training data include Megascale (Tsuboyama) libraries: not held out on the Tsuboyama sets, and drawn last, greyed, with an asterisk
# (ESM3ΔG is not in the list: it is trained on the MGnify stability set and Megascale is its held-out test, BENCHMARKS section 6.3)
MEGASCALE_TRAINED = ['esmtherm', 'ifum_wt', 'ifum_esmfold']
MEGASCALE_NOTE = '* trained on Megascale (Tsuboyama) data: not held out'

RETRAINS = ['me1', 'me2', 'tmpnn', 'tmpnnd', 'spurs', 'spursm']
PUBLIC_MODELS = ['esmtherm', 'ddgemb', 'ddmut', 'ifum_wt', 'ifum_esmfold', 'geoddg', 'esm3dg']
ZEROSHOT = ['prosst', 'pmpnn', 'msrrel@0:wt', 'msrrel@0:comb']


def label(key):
    return MODELS[key]['label'] if key in MODELS else key


def is_msr(key):
    return key.split(':')[0].startswith('msrp0')
