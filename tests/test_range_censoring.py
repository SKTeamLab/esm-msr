"""Admitting the rows the assay bounded rather than fitted.

Dropping them selects the double set on its own outcome: the assay's fit-failure rate rises with additive depth, so
the deep doubles that survive are the ones that came out better than their additive expectation. These tests pin
the three things that has to get right - which rows are admitted, what number they carry, and that the bound
survives into `cens_bound` unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from esm_msr.censoring import (CENS_HI, CENS_LO, CENS_NONE, DG_RANGE_HIGH, DG_RANGE_LOW, SRC_FLOOR, SRC_GAP,
                               SRC_NONE, SRC_RANGE, apply_item_censoring, range_tag)
from esm_msr.preprocess_megascale import _collapse_replicates, _fill_range_values, _tag_range_rows


def _raw():
    """One row of each kind the table contains."""
    return pd.DataFrame({
        'WT_name': ['1ABC.pdb'] * 5,
        'mut_type': ['A1G', 'A2G', 'A3G', 'A4G', 'A5G'],
        'ddG_ML': ['-1.5', '-', '-', '-', '0.3'],
        'dG_ML': ['1.0', '<-1', '>5', '-', '2.0'],
    })


@pytest.mark.parametrize('dG,ddG,expect', [
    ('<-1', '-', (CENS_LO, DG_RANGE_LOW)),
    ('>5', '-', (CENS_HI, DG_RANGE_HIGH)),
    ('-', '-', (CENS_NONE, None)),
    ('<-1', '0.4', (CENS_NONE, None)),      # a fitted ddG is a measurement whatever dG says
    ('2.0', '0.3', (CENS_NONE, None)),
])
def test_range_tag(dG, ddG, expect):
    cens, bound = range_tag(dG, ddG)
    assert cens == expect[0]
    assert (bound == expect[1]) if expect[1] is not None else np.isnan(bound)


def test_admits_bounded_rows_and_drops_the_unbounded_one():
    out = _tag_range_rows(_raw(), admit=True)
    assert sorted(out.mut_type) == ['A1G', 'A2G', 'A3G', 'A5G']      # A4G reports '-' for dG too
    by = dict(zip(out.mut_type, out.cens))
    assert by == {'A1G': CENS_NONE, 'A2G': CENS_LO, 'A3G': CENS_HI, 'A5G': CENS_NONE}
    assert dict(zip(out.mut_type, out.cens_src))['A2G'] == SRC_RANGE
    assert dict(zip(out.mut_type, out.cens_src))['A1G'] == SRC_NONE


def test_admit_false_reproduces_the_old_filter_exactly():
    """The previous behaviour was `df.loc[df.ddG_ML != '-']`; off must be bit-identical to that."""
    raw = _raw()
    assert list(_tag_range_rows(raw, admit=False).mut_type) == list(raw.loc[raw.ddG_ML != '-'].mut_type)


def test_fill_puts_the_bound_on_the_items_own_ddG_scale():
    out = _fill_range_values(_tag_range_rows(_raw(), admit=True), {'1ABC': 2.5})
    row = out.set_index('mut_type')
    assert row.loc['A2G', 'dG_ML'] == DG_RANGE_LOW
    assert row.loc['A2G', 'ddG_ML'] == pytest.approx(DG_RANGE_LOW - 2.5)
    assert row.loc['A3G', 'ddG_ML'] == pytest.approx(DG_RANGE_HIGH - 2.5)
    assert row.loc['A1G', 'ddG_ML'] == '-1.5'                        # untouched, still the raw measurement


def test_fill_leaves_nan_when_the_library_has_no_wild_type_dG():
    """Those rows are then dropped by the caller's notna filter; on the real table every such library consists
    entirely of tags, so there is no reference to place them against."""
    out = _fill_range_values(_tag_range_rows(_raw(), admit=True), {}).set_index('mut_type')
    assert np.isnan(pd.to_numeric(out.loc['A2G', 'ddG_ML']))
    assert np.isnan(pd.to_numeric(out.loc['A3G', 'ddG_ML']))


def test_collapse_replicates_keeps_the_declared_censoring():
    df = _fill_range_values(_tag_range_rows(_raw(), admit=True), {'1ABC': 2.5})
    for c in ('mut_structure', 'code_wt', 'code', 'chain', 'uid', 'aa_seq'):
        df[c] = 'x'
    df['uid'] = df['mut_type']
    for c in ('dG_ML', 'ddG_ML'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = _collapse_replicates(df).set_index('mut_type')
    assert out.loc['A2G', 'cens'] == CENS_LO
    assert out.loc['A2G', 'cens_src'] == SRC_RANGE
    assert out.loc['A2G', 'dG_bound'] == DG_RANGE_LOW


def _item(**kw):
    base = dict(subset_type='double', cens=0, cens_src=SRC_NONE, dG_bound=float('nan'),
                dG_meas=float('nan'), y_floor=float('nan'), protease_gap=float('nan'), ddG=float('nan'))
    base.update(kw)
    return base


def test_a_range_item_keeps_its_bound_as_its_own_label():
    """dG_bound == dG_meas, so cens_bound comes out as the item's ddG."""
    it = _item(cens=CENS_LO, cens_src=SRC_RANGE, dG_bound=DG_RANGE_LOW, dG_meas=DG_RANGE_LOW, ddG=-3.5)
    (out,), counts = apply_item_censoring([it], threshold=0.5)
    assert out['cens'] == CENS_LO and out['cens_src'] == SRC_RANGE
    assert out['cens_bound'] == pytest.approx(-3.5)
    assert counts['range'] == 1 and counts['floor'] == 0


def test_the_threshold_never_overwrites_an_assay_bound():
    """The threshold would fire on this item (dG -1 is below 0.5); the assay's own statement must win."""
    it = _item(cens=CENS_LO, cens_src=SRC_RANGE, dG_bound=DG_RANGE_LOW, dG_meas=DG_RANGE_LOW, ddG=-3.5,
               y_floor=0.1, protease_gap=9.0)
    (out,), counts = apply_item_censoring([it], threshold=0.5)
    assert out['cens_src'] == SRC_RANGE
    assert out['dG_bound'] == DG_RANGE_LOW
    assert counts['floor'] == 0


def test_an_upper_bound_is_carried_as_upper_censoring():
    it = _item(cens=CENS_HI, cens_src=SRC_RANGE, dG_bound=DG_RANGE_HIGH, dG_meas=DG_RANGE_HIGH, ddG=2.5)
    (out,), counts = apply_item_censoring([it], threshold=0.5)
    assert out['cens'] == CENS_HI and out['cens_bound'] == pytest.approx(2.5)
    assert counts['range'] == 1


def test_ordinary_items_still_reach_the_threshold():
    below = _item(dG_meas=-0.5, ddG=-3.0)
    above = _item(dG_meas=2.0, ddG=-1.0)
    items, counts = apply_item_censoring([below, above], threshold=0.5)
    assert items[0]['cens_src'] == SRC_FLOOR and items[1]['cens'] == 0
    assert counts['range'] == 0 and counts['floor'] == 1


def test_a_fitted_replicate_beats_a_bounded_one():
    """Averaging the two would blend a real dG with the synthetic bound, corrupting the measurement."""
    df = pd.DataFrame({
        'WT_name': ['1ABC.pdb'] * 3, 'mut_type': ['A1G', 'A1G', 'A2G'],
        'ddG_ML': ['-1.5', '-', '-'], 'dG_ML': ['1.0', '<-1', '<-1'],
    })
    df = _fill_range_values(_tag_range_rows(df, admit=True), {'1ABC': 2.5})
    for c in ('mut_structure', 'code_wt', 'code', 'chain', 'aa_seq'):
        df[c] = 'x'
    df['uid'] = df['mut_type']
    for c in ('dG_ML', 'ddG_ML'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = _collapse_replicates(df).set_index('mut_type')
    assert len(out) == 2
    assert out.loc['A1G', 'cens'] == CENS_NONE          # one replicate was fitted, so the variant is measured
    assert out.loc['A1G', 'dG_ML'] == pytest.approx(1.0)   # NOT the mean of 1.0 and -1.0
    assert out.loc['A2G', 'cens'] == CENS_LO            # every replicate was bounded


def test_a_library_with_no_measured_rows_is_dropped():
    """Every range item of a library shares one label (its bound is a library constant), so a library made only of
    bounds has no label variance - nothing to learn from and nothing to score against."""
    from esm_msr.preprocess_megascale import _drop_all_bounded_libraries
    df = pd.DataFrame({'code': ['A', 'A', 'B', 'B'], 'cens': [CENS_NONE, CENS_LO, CENS_LO, CENS_LO]})
    out = _drop_all_bounded_libraries(df)
    assert sorted(out.code.unique()) == ['A']
    assert len(out) == 2                       # A keeps both its measured and its bounded row


def test_wt_dG_is_the_exact_reference_the_labels_were_built_against():
    """ddG_ML is defined against ONE designated wt replicate, not the mean of them: a wt row whose own ddG_ML is 0
    carries the reference. Averaging the replicates misses it by a per-library constant, which biases the whole
    library's knee. dG_ML - ddG_ML recovers it exactly from any measured row."""
    from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
    df = pd.DataFrame({
        'WT_name': ['1ABC.pdb'] * 6,
        'mut_type': ['wt', 'wt', 'wt', 'A1G', 'A2G', 'A3G'],
        # the designated replicate is the 4.0 one (its own ddG_ML is 0); the others are scored against it
        'dG_ML': ['4.6', '4.0', '3.4', '3.0', '2.0', '1.0'],
        'ddG_ML': ['0.6', '0.0', '-0.6', '-1.0', '-2.0', '-3.0'],
    })
    got = MegaScaleDatasetPreprocessor._wt_dG_by_code(df)
    assert got['1ABC'] == pytest.approx(4.0)        # not the mean of the wt rows, which is also 4.0 here...
    # ...so bias the replicates to make the mean wrong while the reference is unchanged
    df.loc[0, ['dG_ML', 'ddG_ML']] = ['4.9', '0.9']
    got = MegaScaleDatasetPreprocessor._wt_dG_by_code(df)
    assert got['1ABC'] == pytest.approx(4.0)        # the reference, from the variants
    assert np.mean([4.9, 4.0, 3.4]) == pytest.approx(4.1)   # what the old mean-of-wt-rows would have given


def test_wt_dG_makes_dG_wt_plus_ddG_equal_the_measurement():
    """The identity the link depends on: its knee and the measurement must sit on one axis."""
    from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor
    df = pd.DataFrame({
        'WT_name': ['1ABC.pdb'] * 4, 'mut_type': ['wt', 'A1G', 'A2G', 'A3G'],
        'dG_ML': ['4.9', '3.0', '2.0', '1.0'], 'ddG_ML': ['0.9', '-1.0', '-2.0', '-3.0'],
    })
    ref = MegaScaleDatasetPreprocessor._wt_dG_by_code(df)['1ABC']
    for dg, ddg in ((3.0, -1.0), (2.0, -2.0), (1.0, -3.0)):
        assert ref + ddg == pytest.approx(dg, abs=1e-9)


def test_the_batch_sampler_is_seeded_and_reproducible():
    """Without rng_seed the sampler seeded itself from system entropy, so --seed controlled only the model init:
    the capped subsets and the batch composition differed every run, and two configurations could not be paired
    on a seed."""
    import inspect
    from esm_msr.data import ProteinCyclingBatchSampler, create_consolidated_dataloader
    assert 'rng_seed' in inspect.signature(create_consolidated_dataloader).parameters
    src = inspect.getsource(create_consolidated_dataloader)
    assert 'rng_seed=rng_seed' in src, "the factory must pass the seed through to the sampler"

    import random
    a, b, c = random.Random(1), random.Random(1), random.Random(2)
    xs = list(range(50))
    pa, pb, pc = xs[:], xs[:], xs[:]
    a.shuffle(pa); b.shuffle(pb); c.shuffle(pc)
    assert pa == pb, "the same seed must give the same order"
    assert pa != pc, "different seeds must give different orders"
