"""``native_cond`` belongs to the MT head (routing.py), but the released recipe treats those measurements as
ordinary singles so the WT head trains on them. The flags choose between the two routings.

A mutant-background library's measurements are conditional effects ddG(X | background): the sequence carries the
background mutation, while the structure is the native one, so the WT head (which models wild-type-context singles)
has no business being regressed onto them. ``--native_cond_for_mt`` labels them ``native_cond``, includes the subset
(overriding its 0.0 cap), and the training loop keeps them out of the WT losses and in the MT/combined loss.
"""
import sys

from esm_msr import config, routing


def parse(*argv):
    old = sys.argv
    try:
        sys.argv = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x',
                    '--experiment_name', 'x'] + list(argv)
        return config.parse_arguments()
    finally:
        sys.argv = old


def test_the_released_default_treats_them_as_singles():
    a = parse()
    assert a.native_cond_for_wt is True
    assert a.native_cond_for_mt is False


def test_for_mt_alone_means_both_heads_see_them():
    a = parse('--native_cond_for_mt')
    assert a.native_cond_for_wt is True and a.native_cond_for_mt is True
    assert a.incl_native_cond is True
    assert a.subset_caps['native_cond'] is None


def test_mt_only_and_neither():
    mt = parse('--no-native_cond_for_wt', '--native_cond_for_mt')
    assert mt.native_cond_for_wt is False and mt.native_cond_for_mt is True
    assert mt.incl_native_cond is True and mt.subset_caps['native_cond'] is None
    neither = parse('--no-native_cond_for_wt')
    assert neither.native_cond_for_wt is False and neither.native_cond_for_mt is False
    assert neither.incl_native_cond is False and neither.subset_caps['native_cond'] == 0.0


def test_the_label_condition_covers_all_four_combos():
    # labelled native_cond iff for_mt, or for_wt is off (the neither case is then excluded by the include flag)
    assert parse().native_cond_for_mt is False and parse().native_cond_for_wt is True
    for wt, mt in ((True, False), (True, True), (False, True), (False, False)):
        a = parse(*(['--native_cond_for_wt'] if wt else ['--no-native_cond_for_wt']),
                  *(['--native_cond_for_mt'] if mt else []))
        label = a.native_cond_for_mt or not a.native_cond_for_wt
        assert label == (mt or not wt), (wt, mt)


def test_the_old_spelling_is_an_alias():
    assert parse('--no-native_cond_as_single').native_cond_for_wt is False
    assert parse('--native_cond_as_single').native_cond_for_wt is True


def test_routing_puts_native_cond_on_the_mt_head():
    assert routing.head_for('native_cond') == 'mt'
    assert routing.head_for('single') == 'wt'
    assert routing.head_for('double') == 'ensemble'
