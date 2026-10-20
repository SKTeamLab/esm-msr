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


def test_for_mt_labels_includes_uncaps_and_takes_precedence():
    a = parse('--native_cond_for_mt')
    assert a.native_cond_for_wt is False
    assert a.incl_native_cond is True
    assert a.subset_caps['native_cond'] is None


def test_the_old_spelling_is_an_alias():
    assert parse('--no-native_cond_as_single').native_cond_for_wt is False
    assert parse('--native_cond_as_single').native_cond_for_wt is True


def test_routing_puts_native_cond_on_the_mt_head():
    assert routing.head_for('native_cond') == 'mt'
    assert routing.head_for('single') == 'wt'
    assert routing.head_for('double') == 'ensemble'
