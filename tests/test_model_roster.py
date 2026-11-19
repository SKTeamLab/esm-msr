"""analysis/msr_analysis/models.py: one roster (order, colour, version keys) that registry, external and style all derive from."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis'))
from msr_analysis import models as M  # noqa: E402
from msr_analysis import registry, style  # noqa: E402
from msr_analysis.external import EXT_MODELS, EXT_ORDER  # noqa: E402


def test_order_is_zero_shot_then_transfer_then_supervised_then_megascale():
    assert M.ORDER[:10] == ['ProSST', 'ProteinMPNN', 'ESM3-small-open', 'ESM-MSR', 'ESM-MSR σ=0.5', 'MutateEverything', 'ThermoMPNN', 'SPURS', 'GeoDDG-3D', 'Rosetta']
    assert M.ORDER[10] == 'ESM3ΔG'                                  # ESM3ΔG ends the supervised models
    assert [m.group for m in M.ROSTER] == sorted((m.group for m in M.ROSTER), key=M.GROUP_ORDER_INDEX.get)
    assert EXT_ORDER == M.UNIFIED == [n for n in M.ORDER if n not in ('ESM-MSR σ=0.5', 'ESMtherm', 'IFUM', 'SPURS published*', 'DDGemb', 'DDMut')]


def test_main_models_have_distinct_colours_and_markers():
    main = [m for m in M.ROSTER if m.group != 'megascale' and not m.sigma]
    assert len({m.color for m in main}) == len(main)
    assert len({m.marker for m in main}) == len(main)


def test_registry_and_external_take_their_style_from_the_roster():
    for key, spec in registry.MODELS.items():
        m = M.BY_NAME[spec['model']]
        assert spec['color'] == m.color or key.startswith(('ifum_esmfold', 'tmpnnpub', 'tmpnndpub'))          # a lighter shade of the model's colour only for the second IFUM version and the published ThermoMPNN members
        assert spec['marker'] == m.marker
    for name, spec in EXT_MODELS.items():
        assert spec['color'] == M.BY_NAME[name].color and spec['marker'] == M.BY_NAME[name].marker


def test_only_the_final_esm_msr_is_registered_and_it_is_never_called_p0():
    assert 'msrp0:wt' in registry.MODELS and 'msrp0:comb' in registry.MODELS
    assert not [k for k in registry.MODELS if k.startswith(('msr:', 'msrdev', 'msrrel:', 'msrsmall'))]
    assert not [s['label'] for s in registry.MODELS.values() if 'p0' in s['label'] or 'devel' in s['label'] or 'baseline' in s['label']]


def test_singles_version_is_hollow_and_doubles_version_filled():
    assert style.marker_kwargs('msrp0:wt')['mfc'] == 'none' and style.marker_kwargs('msrp0:comb')['mfc'] != 'none'
    assert style.marker_kwargs('me1')['mfc'] == 'none' and style.marker_kwargs('me2')['mfc'] != 'none'
    assert style.marker_kwargs('geoddg')['mfc'] != 'none'                   # one version: filled


def test_every_variant_is_scored_by_the_version_that_fits_it():
    esm = M.BY_NAME['ESM-MSR']
    assert (esm.single, esm.multi) == ('msrp0:wt', 'msrp0:comb') and esm.views == ('WT-view', 'Dual-view')
    assert (M.BY_NAME['MutateEverything'].single, M.BY_NAME['MutateEverything'].multi) == ('me1', 'me2')
    assert (M.BY_NAME['ThermoMPNN'].single, M.BY_NAME['ThermoMPNN'].multi) == ('tmpnn', 'tmpnnd')
    assert (M.BY_NAME['SPURS'].single, M.BY_NAME['SPURS'].multi) == ('spurs', 'spursm')
    assert (M.BY_NAME['ESM3-small-open'].single, M.BY_NAME['ESM3-small-open'].multi) == ('msrrel@0:wt', 'msrrel@0:comb')
    assert M.version_label('ESM-MSR', 1) == 'ESM-MSR Dual-view' and M.version_label('ThermoMPNN', 1) == 'ThermoMPNN-D'


def test_unknown_model_has_no_grey_default():
    with pytest.raises(KeyError):
        style.style_of('some:new:model')


def test_bar_columns():
    c = M.column('ESM-MSR', 'all')
    assert (c['single'], c['double'], c['cat']) == ('msrp0:wt', 'msrp0:comb', 'transfer')
    assert M.column('ProSST', 'pair')['double'] is None                      # additive only
    assert M.column('ESM3ΔG', 'pair') is None and M.column('ESM3ΔG', 'epi') is None
    assert M.ordered(['SPURS', 'ProSST', 'ESM3ΔG', 'ESM-MSR']) == ['ProSST', 'ESM-MSR', 'SPURS', 'ESM3ΔG']
