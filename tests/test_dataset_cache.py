"""Training reads a prebuilt cache and its manifest: the dataset keeps the subsets a run asks for, relabels mutant-background libraries,
censors at one threshold, and refuses (loudly) to run without a cache."""
import argparse
import pickle

import pytest

from esm_msr import censoring
from esm_msr import blocks
from esm_msr.data import MutationStabilityDataset
from esm_msr.preprocess_megascale import read_manifest, setup_dataloaders, write_manifest


def _item(code, subset, ddG=-1.0, dG_meas=2.0, cens=0, cens_src=0):
    return {'pdb': code, 'subset_type': subset, 'ddG': ddG, 'dG_meas': dG_meas, 'cens': cens, 'cens_src': cens_src,
            'dG_bound': float('nan'), 'dG_wt': 3.0}


def _write(tmp_path, code, items, version='c6', score='ddG_ML'):
    with open(tmp_path / f"{code}_{score}_{version}.pkl", 'wb') as f:
        pickle.dump({'items': items, 'floor_stats': {}}, f)


def _ds(tmp_path, code='1ABC', **kw):
    return MutationStabilityDataset(code, str(tmp_path), cache_version='c6', **kw)


def test_the_requested_subsets_are_kept_and_cond_is_never_served(tmp_path):
    _write(tmp_path, '1ABC', [_item('1ABC', 'single'), _item('1ABC', 'double'), _item('1ABC', 'cond'), _item('1ABC', 'single')])
    assert [i['subset_type'] for i in _ds(tmp_path, subsets=('single', 'double'))] == ['single', 'double', 'single']
    assert [i['subset_type'] for i in _ds(tmp_path, subsets=('single',))] == ['single', 'single']
    assert [i['subset_type'] for i in _ds(tmp_path, subsets=('double',))] == ['double']


def test_a_mutant_background_library_is_relabelled_only_on_request(tmp_path):
    _write(tmp_path, '1A0N_L7S', [_item('1A0N_L7S', 'single'), _item('1A0N_L7S', 'single')])
    plain = _ds(tmp_path, '1A0N_L7S', subsets=('single',))
    assert [i['subset_type'] for i in plain] == ['single', 'single']
    relabelled = _ds(tmp_path, '1A0N_L7S', subsets=('single', 'native_cond'), label_native_cond=True)
    assert [i['subset_type'] for i in relabelled] == ['native_cond', 'native_cond']
    # ... and an ordinary library is untouched by the same flag
    _write(tmp_path, '1ABC', [_item('1ABC', 'single')])
    assert [i['subset_type'] for i in _ds(tmp_path, subsets=('single',), label_native_cond=True)] == ['single']
    # excluded entirely when native_cond is not among the subsets
    assert len(_ds(tmp_path, '1A0N_L7S', subsets=('single',), label_native_cond=True)) == 0


def test_range_censored_rows_are_dropped_unless_admitted(tmp_path):
    rows = [_item('1ABC', 'double', cens=-1, cens_src=censoring.SRC_RANGE), _item('1ABC', 'double')]
    _write(tmp_path, '1ABC', rows)
    assert len(_ds(tmp_path, censor_range=True)) == 2
    assert len(_ds(tmp_path, censor_range=False)) == 1


def test_one_threshold_censors(tmp_path):
    _write(tmp_path, '1ABC', [_item('1ABC', 'single', ddG=-3.0, dG_meas=0.2), _item('1ABC', 'double', ddG=-1.0, dG_meas=2.0)])
    ds = _ds(tmp_path, censor_threshold=0.5)
    assert [i['cens'] for i in ds] == [censoring.CENS_LO, 0]
    assert list(ds.cens_arr) == [-1, 0]
    assert [i['cens'] for i in _ds(tmp_path, censor_threshold=None)] == [0, 0]


def test_a_missing_cache_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_cache.py'):
        _ds(tmp_path, '9ZZZ')


def test_the_manifest_round_trips_and_is_required(tmp_path):
    with pytest.raises(FileNotFoundError, match='manifest'):
        read_manifest(str(tmp_path), 'c6')
    write_manifest(str(tmp_path), 'c6', {'train': ['A', 'B'], 'val': ['C']}, ['ptmuld'], '/x/splits.pkl', note='t')
    m = read_manifest(str(tmp_path), 'c6')
    assert m['train'] == ['A', 'B'] and m['val'] == ['C'] and m['benchmarks'] == ['ptmuld'] and m['split_file'] == 'splits.pkl'






def _args(tmp_path, **kw):
    a = dict(cache_path=str(tmp_path), cache_version='c6', num_workers=0, exclude_mutbg=False, rank_censor_threshold=0.5, seed=1,
             block_positions=8, block_pair_weight=1.0, block_domain_steps=False)
    a.update(kw)
    return argparse.Namespace(**a)


def _library(tmp_path, code, n=8):
    _write(tmp_path, code, [_item(code, 'single') for _ in range(n)])


@pytest.fixture
def captured(monkeypatch):
    """The block loader needs real sequences; what these tests look at is which datasets reach it."""
    seen = {}
    monkeypatch.setattr(blocks, 'build_block_loader', lambda datasets, names, *a, **k: (seen.update(datasets=list(datasets), names=list(names), kwargs=k), 'block loader')[1])
    return seen


def test_setup_hands_the_block_loader_the_training_libraries_in_manifest_order_and_skips_empty_ones(tmp_path, captured):
    for code in ('1AAA', '1BBB', '1CCC'):
        _library(tmp_path, code)
    _write(tmp_path, '1DDD', [])                                      # nothing in it
    write_manifest(str(tmp_path), 'c6', {'train': ['1BBB', '1AAA', '1DDD'], 'val': ['1CCC']}, [], 'splits.pkl')
    train, val, bench, tn, vn, bn = setup_dataloaders(_args(tmp_path))
    assert vn == ['1CCC'] and bn == [] and tn == ['cycled_train'] and train == ['block loader']
    assert captured['names'] == ['1BBB', '1AAA'] and [len(d) for d in captured['datasets']] == [8, 8]    # 1DDD was empty and skipped
    assert captured['kwargs']['singles'] is True and captured['kwargs']['pairs'] is True
    setup_dataloaders(_args(tmp_path, block_pair_weight=0.0))
    assert captured['kwargs']['pairs'] is False                        # a pair weight of 0 runs no pair blocks


def test_a_mutant_background_library_is_trained_on_as_ordinary_singles(tmp_path, captured):
    _write(tmp_path, '1A0N_L7S', [_item('1A0N_L7S', 'single') for _ in range(8)])
    _library(tmp_path, '1AAA')
    write_manifest(str(tmp_path), 'c6', {'train': ['1A0N_L7S', '1AAA'], 'val': ['1A0N_L7S']}, [], 'splits.pkl')
    train, val, _, tn, vn, _ = setup_dataloaders(_args(tmp_path))
    assert sorted({i['subset_type'] for d in captured['datasets'] for i in d}) == ['single']               # no 'native_cond' label: both heads see singles
    assert [len(d) for d in captured['datasets']] == [8, 8] and vn == ['1A0N_L7S'] and len(val[0].dataset) == 8


def test_exclude_mutbg_leaves_the_mutant_background_libraries_out_of_training_only(tmp_path, captured):
    _write(tmp_path, '1A0N_L7S', [_item('1A0N_L7S', 'single') for _ in range(8)])
    _library(tmp_path, '1AAA')
    write_manifest(str(tmp_path), 'c6', {'train': ['1A0N_L7S', '1AAA'], 'val': ['1A0N_L7S']}, [], 'splits.pkl')
    train, val, _, tn, vn, _ = setup_dataloaders(_args(tmp_path, exclude_mutbg=True))
    assert [len(d) for d in captured['datasets']] == [8]                                                   # only 1AAA
    assert vn == ['1A0N_L7S'] and len(val[0].dataset) == 8                                                 # validation still scores the mutant-background library


