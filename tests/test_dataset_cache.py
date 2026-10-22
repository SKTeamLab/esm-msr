"""Training reads a prebuilt cache and its manifest: the dataset keeps the subsets a run asks for, relabels mutant-background libraries,
stamps the link's one floor, censors at one threshold, and refuses (loudly) to run without a cache."""
import argparse
import pickle

import pytest

from esm_msr import censoring
from esm_msr.data import MutationStabilityDataset, ProteinCyclingBatchSampler
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


def test_one_floor_is_stamped_and_one_threshold_censors(tmp_path):
    _write(tmp_path, '1ABC', [_item('1ABC', 'single', ddG=-3.0, dG_meas=0.2), _item('1ABC', 'double', ddG=-1.0, dG_meas=2.0)])
    ds = _ds(tmp_path, link_floor=0.25, censor_threshold=0.5)
    assert [i['y_floor'] for i in ds] == [0.25, 0.25]
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


def test_the_sampler_keeps_a_batch_inside_one_library_and_is_seeded(tmp_path):
    for code in ('1AAA', '1BBB'):
        _write(tmp_path, code, [_item(code, 'single') for _ in range(8)])
    ds = [_ds(tmp_path, c) for c in ('1AAA', '1BBB')]

    def batches(seed):
        sam = ProteinCyclingBatchSampler(ds, 4, ['1AAA', '1BBB'], rng_seed=seed)
        return list(sam), sam

    a, sam = batches(1)
    assert len(a) == 4 and len(sam) == 4
    for b in a:
        assert len({i // 8 for i in b}) == 1                 # one library per batch
    assert a == batches(1)[0] and a != batches(2)[0]


def test_the_sampler_drops_a_remainder_smaller_than_a_batch(tmp_path):
    _write(tmp_path, '1AAA', [_item('1AAA', 'single') for _ in range(10)])
    assert len(list(ProteinCyclingBatchSampler([_ds(tmp_path, '1AAA')], 4, ['1AAA'], rng_seed=1))) == 2


def _args(tmp_path, **kw):
    a = dict(cache_path=str(tmp_path), cache_version='c6', batch_size=4, num_workers=0, subsets=['single', 'double'],
             native_cond_for_wt=True, native_cond_for_mt=False, censor_range=False, link_floor=0.0, censor_threshold=0.5, seed=1)
    a.update(kw)
    return argparse.Namespace(**a)


def _library(tmp_path, code, n=8):
    _write(tmp_path, code, [_item(code, 'single') for _ in range(n)])


def test_setup_builds_loaders_in_manifest_order_and_skips_empty_libraries(tmp_path):
    for code in ('1AAA', '1BBB', '1CCC'):
        _library(tmp_path, code)
    _write(tmp_path, '1DDD', [_item('1DDD', 'double')])             # only a double: empty under --subsets single
    write_manifest(str(tmp_path), 'c6', {'train': ['1BBB', '1AAA', '1DDD'], 'val': ['1CCC']}, [], 'splits.pkl')
    train, val, bench, tn, vn, bn = setup_dataloaders(_args(tmp_path, subsets=['single']))
    assert vn == ['1CCC'] and bn == [] and tn == ['cycled_train']
    assert [len(d) for d in train[0].dataset.datasets] == [8, 8]    # 1BBB then 1AAA; 1DDD was empty and skipped


def test_native_cond_for_mt_adds_the_subset_to_training_only(tmp_path):
    _write(tmp_path, '1A0N_L7S', [_item('1A0N_L7S', 'single') for _ in range(8)])
    _library(tmp_path, '1AAA')
    write_manifest(str(tmp_path), 'c6', {'train': ['1A0N_L7S', '1AAA'], 'val': ['1A0N_L7S']}, [], 'splits.pkl')
    train, val, *_ = setup_dataloaders(_args(tmp_path, native_cond_for_mt=True))
    assert sorted({i['subset_type'] for d in train[0].dataset.datasets for i in d}) == ['native_cond', 'single']
    assert len(val[0].dataset) == 8                                  # validation keeps native_cond (scored as singles)


def test_a_split_file_that_is_not_the_manifests_is_refused(tmp_path):
    _library(tmp_path, '1AAA')
    write_manifest(str(tmp_path), 'c6', {'train': ['1AAA'], 'val': ['1AAA']}, [], 'splits_a.pkl')
    setup_dataloaders(_args(tmp_path, requested_split_file='/some/where/splits_a.pkl'))
    with pytest.raises(ValueError, match='manifest'):
        setup_dataloaders(_args(tmp_path, requested_split_file='/some/where/splits_b.pkl'))
