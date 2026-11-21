"""Dataset groups, path re-exports, and the immutable prediction-key style registry.

All paths originate in ``msr_analysis.paths`` and all visual identities derive from ``msr_analysis.models.ROSTER``.
Split selection is parameterized explicitly through ``tsu_datasets(split)`` and ``results_dir(name, split)``
with zero import-time mutation from environment variables.
"""
from __future__ import annotations

from types import MappingProxyType

from . import models as _R
from .models import StyleSpec
from .paths import (
    ABLATION_RESULTS_DIR,
    ANALYSIS_ROOT as HERE,
    BASE,
    CACHE,
    CANONICAL_RESULT_TOPS,
    CORE_RHO_BOOTSTRAP_CSV,
    DATA_DIR,
    DMS3_RESULTS_DIR,
    DOMAINOME_RESULTS_DIR,
    FINAL_BENCH,
    FOR_REVIEWERS_DIR,
    INFERENCE_RESOURCES_DIR,
    MAIN_TEXT_FIGS_DIR,
    MANIFEST_OUTPUTS_CSV,
    MEGA_RAW,
    MEGA_SPLITS_DIR,
    MEGASCALE_DIR,
    MGNIFY_RESULTS_DIR,
    MISC_RESULTS_DIR,
    MSR,
    MSR_ROOT as ROOT,
    P0,
    PGYM,
    PROTEINGYM_RESULTS_DIR,
    PUBLIC,
    REFERENCE,
    REPO_ROOT,
    RESULT_DIRS,
    RESULTS,
    RETRAINED,
    REVIEWER_REFERENCE_DIR,
    SCREENING_OVERVIEW_DIR,
    SEQUENCE_LENGTH_DIR,
    SPLIT_CONSTRUCTION_DIR,
    SUMMARY_RESULTS_DIR,
    SUPPLEMENTARY_FIGS_DIR,
    TABLE_FIGURE_MAP_CSV,
    TASKS,
    THERMOSTABILITY_DIR,
    VALIDATION_VS_BENCHMARKS_CSV,
    results_dir,
    strip_fig_prefix,
)

# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
EXT12 = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
DMS3 = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']
DMS_AUX = [
    'DLG4_HUMAN_Faure_2021_abundance_domain',
    'GRB2_HUMAN_Faure_2021_abundance_domain',
    'ESTA_BACSU_Nutschel_2020_dTm',
    'GB1_Wu_2016_binding_domain',
]
TSU_CAPPED = ['tsuboyama_capped_val', 'tsuboyama_capped_test']
TSU_MEGA = ['tsuboyama_mega_val', 'tsuboyama_mega_test']
TSU_SPLIT = 'capped'
TSU = list(TSU_CAPPED)
MAIN_BENCHMARK_DATASETS = EXT12 + ['domainome'] + TSU_CAPPED
ALL_DATASETS = EXT12 + DMS3 + ['domainome'] + TSU_CAPPED


def tsu_datasets(split: str = 'capped') -> list[str]:
    """Return ``[val_ds, test_ds]`` for Tsuboyama ``split`` ('capped' or 'mega')."""
    if split == 'capped':
        return list(TSU_CAPPED)
    if split == 'mega':
        return list(TSU_MEGA)
    raise ValueError(f"split must be 'capped' or 'mega', not {split!r}")


def main_benchmark_datasets(split: str = 'capped') -> list[str]:
    """Return the main non-DMS3 benchmark dataset list (``EXT12 + ['domainome'] + tsu_datasets(split)``)."""
    return EXT12 + ['domainome'] + tsu_datasets(split)


def all_datasets(split: str = 'capped') -> list[str]:
    """Return the full benchmark dataset list (including ``DMS3``) for ``split`` ('capped' or 'mega')."""
    return EXT12 + DMS3 + ['domainome'] + tsu_datasets(split)


LABEL = MappingProxyType({
    's571': 'S571', 's783': 'S783', 's2648': 'S2648', 's8754': 'S8754', 's669': 'S669', 's461': 'S461',
    'ssym': 'Ssym', 'q3421': 'Q3421', 'k3822': 'K3822', 'k2369': 'K2369', 'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D',
    'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.',
    'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.',
    'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.',
    'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.',
    'ESTA_BACSU_Nutschel_2020_dTm': 'ESTA dTm',
    'GB1_Wu_2016_binding_domain': 'GB1 bind.',
    'MYO_HUMAN_Kung_2025_display': 'MYO',
    'domainome': 'Domainome',
    'tsuboyama_capped_val': 'Tsuboyama val', 'tsuboyama_capped_test': 'Tsuboyama test',
    'tsuboyama_mega_val': 'Tsuboyama val (mega_splits)', 'tsuboyama_mega_test': 'Tsuboyama test (mega_splits)',
})
PER_PROTEIN = frozenset({'domainome', *TSU_CAPPED, *TSU_MEGA})

# ----------------------------------------------------------------------------------------------
# immutable model & prediction-key style registry
# ----------------------------------------------------------------------------------------------
_GROUP = {'zeroshot': 'zeroshot', 'transfer': 'retrain', 'supervised': 'public', 'megascale': 'public'}
_MODELS_MUT: dict[str, StyleSpec] = {}


def _reg(key: str, lbl: str, name: str, fill: bool | None = None, color: str | None = None):
    m = _R.BY_NAME[name]
    is_fill = (key not in _R.SINGLES_KEYS) if fill is None else bool(fill)
    grp = 'msr' if name in ('ESM-MSR', 'ESM-MSR σ=0.5') else _GROUP[m.group]
    _MODELS_MUT[key] = StyleSpec(
        label=lbl,
        color=color or m.color,
        marker=m.marker,
        fill=is_fill,
        group=grp,
        model=name,
        ls=m.ls,
        lw=m.lw,
    )


# --- ESM-MSR (sigma=1 and sigma=0.5) and ESM3-small-open (sigma=0) ---
for _name, _tag in (('ESM-MSR', 'msrp0'), ('ESM-MSR σ=0.5', 'msrp0@0.5'), ('ESM3-small-open', 'msrrel@0')):
    _disp = _R.BY_NAME[_name].label
    _reg(f'{_tag}:wt', _R.version_label(_name, 0), _name)
    _reg(f'{_tag}:comb', _R.version_label(_name, 1), _name)
    _reg(f'{_tag}:epi', f'{_disp} epistasis (comb_AB − WT_A − WT_B)', _name, fill=True)
    _reg(f'{_tag}:wt+', f'{_disp} WT-view, sum of the singles', _name, fill=False)

# --- Transfer baselines: singles version hollow, doubles version filled ---
for _key, _name, _i in (
    ('me1', 'MutateEverything', 0), ('me2', 'MutateEverything', 1),
    ('tmpnn', 'ThermoMPNN', 0), ('tmpnnd', 'ThermoMPNN', 1),
    ('spurs', 'SPURS', 0), ('spursm', 'SPURS', 1),
):
    _reg(_key, _R.version_label(_name, _i), _name)
_reg('me1+', 'MutateEverything Stage 1, additive', 'MutateEverything', fill=False)
_reg('me2+', 'MutateEverything Stage 2, additive', 'MutateEverything', fill=False)
_reg('me2:epi', 'MutateEverything Stage 2 − Stage 1', 'MutateEverything', fill=True)
_reg('tmpnnd:epi', 'ThermoMPNN-D − ThermoMPNN', 'ThermoMPNN', fill=True)
_reg('spursm:epi', 'SPURS-multi − SPURS', 'SPURS', fill=True)

# --- mega_splits variants (SPURS trained adapter, published checkpoints) ---
_reg('spursu', 'SPURS, trained adapter (retrain)', 'SPURS')
_reg('spursum', 'SPURS-multi, trained adapter (retrain)', 'SPURS')
_reg('spursum:epi', 'SPURS-multi, trained adapter (vs its singles)', 'SPURS', fill=True)
_reg('spurspub', 'SPURS, published checkpoint*', 'SPURS published*')
_reg('spurspubm', 'SPURS-multi, published checkpoint*', 'SPURS published*')
_reg('spurspubm:epi', 'SPURS-multi, published (vs its singles)*', 'SPURS published*', fill=True)
_reg('tmpnnpub', 'ThermoMPNN, published (3 ensemble members)', 'ThermoMPNN', color=_R.tint(_R.color('ThermoMPNN'), 0.4))
_reg('tmpnndpub', 'ThermoMPNN-D, published (3 ensemble members)', 'ThermoMPNN', color=_R.tint(_R.color('ThermoMPNN'), 0.4))
_reg('tmpnndpub:epi', 'ThermoMPNN-D, published (vs ThermoMPNN)', 'ThermoMPNN', fill=True, color=_R.tint(_R.color('ThermoMPNN'), 0.4))

# --- Supervised, zero-shot, and Megascale-trained models (including all + and :epi keys) ---
_reg('geoddg', 'GeoDDG-3D', 'GeoDDG-3D')
_reg('geoddg+', 'GeoDDG-3D, sum of the singles', 'GeoDDG-3D', fill=False)
_reg('geoddg:epi', 'GeoDDG-3D, direct − sum of singles', 'GeoDDG-3D', fill=True)

_reg('rosetta', 'Rosetta cart. ΔΔG', 'Rosetta')
_reg('rosetta+', 'Rosetta cart. ΔΔG, sum of the singles', 'Rosetta', fill=False)
_reg('rosetta:epi', 'Rosetta cart. ΔΔG, direct − sum of singles', 'Rosetta', fill=True)

_reg('esm3dg', 'ESM3ΔG', 'ESM3ΔG')
_reg('esm3dg+', 'ESM3ΔG, sum of the singles', 'ESM3ΔG', fill=False)
_reg('esm3dg:epi', 'ESM3ΔG, direct − sum of singles', 'ESM3ΔG', fill=True)

_reg('prosst', 'ProSST', 'ProSST')

_reg('pmpnn', 'ProteinMPNN', 'ProteinMPNN')
_reg('pmpnn+', 'ProteinMPNN, sum of the singles', 'ProteinMPNN', fill=False)
_reg('pmpnn:epi', 'ProteinMPNN, direct − sum of singles', 'ProteinMPNN', fill=True)

_reg('esmtherm', 'ESMtherm*', 'ESMtherm')
_reg('esmtherm+', 'ESMtherm*, sum of the singles', 'ESMtherm', fill=False)
_reg('esmtherm:epi', 'ESMtherm*, direct − sum of singles', 'ESMtherm', fill=True)

_reg('ifum_wt', 'IFUM, WT backbone*', 'IFUM')
_reg('ifum_wt+', 'IFUM, WT backbone*, sum of the singles', 'IFUM', fill=False)
_reg('ifum_wt:epi', 'IFUM, WT backbone*, direct − sum of singles', 'IFUM', fill=True)
_reg('ifum_esmfold', 'IFUM, ESMFold*', 'IFUM', fill=False, color=_R.tint(_R.color('IFUM'), 0.35))
_reg('ifum_esmfold+', 'IFUM, ESMFold*, sum of the singles', 'IFUM', fill=False, color=_R.tint(_R.color('IFUM'), 0.35))
_reg('ifum_esmfold:epi', 'IFUM, ESMFold*, direct − sum of singles', 'IFUM', fill=True, color=_R.tint(_R.color('IFUM'), 0.35))

_reg('ddgemb', 'DDGemb', 'DDGemb')
_reg('ddgemb+', 'DDGemb, sum of the singles', 'DDGemb', fill=False)
_reg('ddgemb:epi', 'DDGemb, direct − sum of singles', 'DDGemb', fill=True)

_reg('ddmut', 'DDMut', 'DDMut', fill=False)
_reg('ddmut+', 'DDMut, sum of the singles', 'DDMut', fill=False)
_reg('ddmut:epi', 'DDMut, direct − sum of singles', 'DDMut', fill=True)

# --- Canonical roster display names (so style_of('ESM-MSR') works without mutating MODELS) ---
for _m in _R.ROSTER:
    _reg(_m.name, _m.label, _m.name, fill=True)

MODELS = MappingProxyType(_MODELS_MUT)

MEGASCALE_TRAINED = ['esmtherm', 'ifum_wt', 'ifum_esmfold']
MEGASCALE_NOTE = '* trained on Megascale (Tsuboyama) data: not held out'

ZEROSHOT_KEYS = ('prosst', 'pmpnn', 'msrrel@0:wt', 'msrrel@0:comb')
ZEROSHOT_MODELS = list(_R.ZEROSHOT_MODELS)
ZEROSHOT = ZEROSHOT_MODELS


def label(key: str) -> str:
    if key in MODELS:
        return MODELS[key].label
    if key in _R.ABLATION_STYLES:
        return _R.ABLATION_STYLES[key].label
    return key


def is_msr(key: str) -> bool:
    return key.split(':')[0].startswith('msrp0') or key in ('ESM-MSR', 'ESM-MSR σ=0.5')
