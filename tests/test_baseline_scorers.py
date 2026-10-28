"""The scoring cores of the baseline scripts, with fake models (no SPURS / ThermoMPNN install, no GPU)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASELINES = Path(__file__).resolve().parent.parent / 'inference_scripts' / 'baselines'
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES.parent))


def _load_spurs_module(monkeypatch):
    pytest.importorskip('omegaconf')
    """Import baselines/spurs.py with the `spurs` package replaced by stubs; its own esm/torchtext mocking stays in place."""
    stub = types.ModuleType('spurs')
    sub = {}
    for name in ('spurs.utils', 'spurs.models', 'spurs.models.stability', 'spurs.models.stability.spurs', 'spurs.models.stability.spurs_multi',
                 'spurs.datamodules', 'spurs.datamodules.datasets', 'spurs.datamodules.datasets.data_utils', 'spurs.inference'):
        sub[name] = types.ModuleType(name)
    sub['spurs.utils'].seed_everything = lambda s: None
    sub['spurs.models.stability.spurs'].SPURS = object
    sub['spurs.models.stability.spurs_multi'].SPURSMulti = object
    sub['spurs.datamodules.datasets.data_utils'].Alphabet = object
    inf = sub['spurs.inference']
    inf.parse_pdb = lambda path, code, chain, cfg, overwrite_seq=None: {'seq': 'ACDEFGHIKL'}
    inf.get_SPURS_from_hub = lambda: (None, {})
    inf.get_SPURS_multi_from_hub = lambda: (None, {})
    inf.parse_pdb_for_mutation = lambda lists: (lists, __import__('torch').zeros(len(lists), 1))
    for k, v in {'spurs': stub, **sub}.items():
        monkeypatch.setitem(sys.modules, k, v)
    sys.modules.pop('spurs_baseline_under_test', None)
    import importlib.util
    spec = importlib.util.spec_from_file_location('spurs_baseline_under_test', BASELINES / 'spurs.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeSingle:
    """ddG matrix (L x 20): destabilizing by (aa index) so additive sums are easy to predict."""
    def __call__(self, pdb_data, return_logist=False):
        import torch
        L = len(pdb_data['seq'])
        m = torch.tile(torch.arange(20, dtype=torch.float32), (L, 1)) * 0.1
        for i, a in enumerate(pdb_data['seq']):
            m[i, 'ACDEFGHIKLMNPQRSTVWY'.index(a)] = 0.0         # wild type scores exactly 0
        return m


class _FakeMulti:
    """One value per requested mutation list: 0.5 per mutation, so a single is 0.5 and a double 1.0."""
    def __call__(self, pdb_data):
        import torch
        return torch.tensor([0.5 * len(m) for m in pdb_data['mut_ids']])


def _spurs(mod, models, mutants, tracker):
    obj = mod.Spurs.__new__(mod.Spurs)
    obj.args, obj.tracker, obj.mutants, obj.models, obj.chunk = None, tracker, mutants, models, 4096
    obj.single = _FakeSingle() if models in ('single', 'both') else None
    obj.multi = _FakeMulti() if models in ('multi', 'both') else None
    obj.cfg_single, obj.cfg_multi = {}, {}
    return obj


def test_spurs_readouts_models_and_mutants(monkeypatch, tmp_path):
    mod = _load_spurs_module(monkeypatch)
    import resource_tracker as rt
    t = rt.Tracker(tmp_path, 'test', interval=0.05)
    muts = ['A1C', 'C2D:E4F', 'A1C:C2D']                  # sequence is ACDEFGHIKL; A1C -> index of C = 1 -> 0.1
    both = _spurs(mod, 'both', 'both', t).score('P', {'seq': 'ACDEFGHIKL'}, muts)
    assert np.allclose(both['SPURS_single_additive'], [-0.1, -0.6, -0.3])
    assert both['SPURS_single'][0] == both['SPURS_single_additive'][0] and np.isnan(both['SPURS_single'][1])
    assert np.allclose(both['SPURS_multi'], [-0.5, -1.0, -1.0]) and np.allclose(both['SPURS_multi_additive'], [-0.5, -1.0, -1.0])
    only_multi = _spurs(mod, 'multi', 'both', t).score('P', {'seq': 'ACDEFGHIKL'}, muts)
    assert np.isnan(only_multi['SPURS_single_additive']).all() and np.allclose(only_multi['SPURS_multi'], both['SPURS_multi'])
    only_single = _spurs(mod, 'single', 'both', t).score('P', {'seq': 'ACDEFGHIKL'}, muts)
    assert np.isnan(only_single['SPURS_multi']).all() and np.allclose(only_single['SPURS_single_additive'], both['SPURS_single_additive'])
    singles = _spurs(mod, 'both', 'single', t).score('P', {'seq': 'ACDEFGHIKL'}, muts)
    assert not np.isnan(singles['SPURS_multi'][0]) and np.isnan(singles['SPURS_multi'][1:]).all()
    multis = _spurs(mod, 'both', 'multi', t).score('P', {'seq': 'ACDEFGHIKL'}, muts)
    assert np.isnan(multis['SPURS_multi'][0]) and not np.isnan(multis['SPURS_multi'][1:]).any()
    t.close()
    df = rt.load_scopes(t.dir)
    assert (df.level == 'protein').sum() == 5 and {'t_single_forward_s', 't_multi_targets_s'} <= set(df.columns)


def test_spurs_score_table_keeps_row_alignment(monkeypatch, tmp_path):
    mod = _load_spurs_module(monkeypatch)
    import resource_tracker as rt
    t = rt.Tracker(tmp_path, 'test', interval=0.05)
    obj = _spurs(mod, 'both', 'both', t)
    obj.parse = lambda p, c, ch, overwrite_seq=None: {'seq': 'ACDEFGHIKL'}
    df = pd.DataFrame({'id': ['a', 'b', 'c'], 'code': ['X', 'Y', 'X'], 'chain': 'A', 'pdb_file': ['x.pdb', 'y.pdb', 'x.pdb'],
                       'mut_type': ['A1C', 'A1D', 'C2D:E4F'], 'ddG': [1.0, 2.0, 3.0]}).set_index('id')
    out = obj.score_table(df, 'demo')
    assert list(out.index) == ['a', 'b', 'c'] and np.allclose(out['SPURS_multi'], [-0.5, -0.5, -1.0])
    assert out.loc['c', 'SPURS_multi_epi'] == pytest.approx(0.0)
    t.close()


def _load_mpnn_module(monkeypatch):
    stub = types.ModuleType('protein_mpnn_utils')
    for n in ('tied_featurize', 'parse_PDB', 'StructureDatasetPDB', 'ProteinMPNN', 'gather_nodes', 'cat_neighbors_nodes'):
        setattr(stub, n, object)
    monkeypatch.setitem(sys.modules, 'protein_mpnn_utils', stub)
    import importlib.util
    spec = importlib.util.spec_from_file_location('mpnn_baseline_under_test', BASELINES / 'proteinmpnn.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mpnn_score_columns_and_mutants_filter(monkeypatch, tmp_path):
    import torch
    mod = _load_mpnn_module(monkeypatch)
    import resource_tracker as rt
    calls = []

    def fake_fast(model, muts, X, S, mask, residue_idx, chain_enc, alphabet, max_tokens, seed, n_samples, coord_noise):
        calls.append(list(muts))
        w = np.arange(1.0, len(muts) + 1)
        return w, w + 10.0

    monkeypatch.setattr(mod, 'score_unmasked_fast', fake_fast)
    t = rt.Tracker(tmp_path, 'test', interval=0.05)
    obj = mod.Mpnn.__new__(mod.Mpnn)
    obj.args = types.SimpleNamespace(mutants='both', max_tokens=32768, n_samples=1, coord_noise=0.0, seed=0)
    obj.tracker, obj.model, obj.device = t, None, torch.device('cpu')
    feats = (None, torch.zeros(1, 12, dtype=torch.long), None, None, None)
    r = obj.score('P', feats, ['A1C', 'A1C:C2D', 'D3E'])
    assert np.allclose(r['mpnn_score_wt_seq'], [1, 2, 3]) and np.allclose(r['mpnn_score'], [6, 7, 8])
    assert np.allclose(r['mpnn_score_epistasis'], [5, 5, 5]) and np.allclose(r['mpnn_score_additive'], [1, 2, 3])
    obj.args.mutants = 'multi'
    r = obj.score('P', feats, ['A1C', 'A1C:C2D', 'D3E'])
    assert calls[-1] == ['A1C:C2D'] and np.isnan(r['mpnn_score'][[0, 2]]).all() and r['mpnn_score'][1] == 6
    obj.args.mutants = 'single'
    r = obj.score('P', feats, ['A1C:C2D'])                   # nothing requested: no call, all NaN
    assert len(calls) == 2 and np.isnan(r['mpnn_score']).all()
    t.close()
    df = rt.load_scopes(t.dir)
    assert (df.level == 'protein').sum() == 2 and (df.seq_len == 12).all()


def _load_tmpnn_module(monkeypatch):
    pytest.importorskip('omegaconf')
    def stub(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m
    stub('thermompnn')
    stub('thermompnn.datasets')
    stub('thermompnn.datasets.v2_datasets', MegaScaleDatasetv2=object,
         tied_featurize_mut=lambda items, side_chains=False: _featurize(items))
    stub('thermompnn.protein_mpnn_utils', parse_PDB=lambda *a, **k: None)
    stub('thermompnn.train_thermompnn', parse_cfg=lambda c: c)
    import importlib.util
    spec = importlib.util.spec_from_file_location('tmpnn_baseline_under_test', BASELINES / 'thermompnn.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _featurize(items):
    import torch
    # tied_featurize_mut returns twelve things (X, S, mask, lengths, chain_M, chain_enc, res_idx, mut_pos, mut_wt, mut_mt,
    # mut_ddG, atom_mask); the fake model reads only mut_pos
    return (None,) * 7 + (torch.tensor([it['value'] for it in items], dtype=torch.float32),) + (None,) * 4


class _FakeNet:
    def eval(self):
        return self

    def cuda(self):
        return self

    def parameters(self):
        import torch
        return iter([torch.zeros(1)])

    def __call__(self, X, S, mask, chain_M, res_idx, chain_enc, mut_pos, *rest):
        return mut_pos * 2.0, None


def test_tmpnn_predict_batches_within_proteins_and_restores_order(monkeypatch, tmp_path):
    mod = _load_tmpnn_module(monkeypatch)
    import resource_tracker as rt
    rng = np.random.default_rng(0)
    codes = rng.choice(list('ABC'), 40)
    items = [{'value': float(i), 'mutation': types.SimpleNamespace(pdb=c), 'seq': 'A' * (30 + 20 * 'ABC'.index(c))}
             for i, c in enumerate(codes)]
    t = rt.Tracker(tmp_path, 'test', interval=0.05)
    p = mod.predict(_FakeNet(), items, batch_size=7, tracker=t, groups=codes, tag={'model': 'single'})
    assert np.allclose(p, np.arange(40) * 2.0)                      # every item scored once, in item order
    t.close()
    df = rt.load_scopes(t.dir)
    assert sorted(df.name) == ['A', 'B', 'C'] and df.n_rows.sum() == 40 and (df.model == 'single').all()
    assert {'t_featurize_s', 't_forward_s', 't_d2h_s'} <= set(df.columns)
    mixed = mod.predict(_FakeNet(), items, batch_size=7, tracker=rt.NullTracker(), groups=codes, mixed_protein_batches=True)
    assert np.allclose(mixed, p)                                    # batching changes nothing about a prediction


def test_tmpnn_combine_and_wanted(monkeypatch):
    mod = _load_tmpnn_module(monkeypatch)
    df = pd.DataFrame({'code': ['A', 'A', 'B', 'C', 'B'], 'x': range(5)})
    assert mod._limit(df, 2)['code'].tolist() == ['A', 'A', 'B', 'B'] and len(mod._limit(df, 0)) == 5
    n_mut = np.array([1, 2, 3])
    add = np.array([1.0, 2.0, 3.0]); epi = np.array([np.nan, 5.0, np.nan])
    pa, comb = mod._combine(add, epi, n_mut, {'single': 1, 'epistatic': 1})
    assert np.allclose(pa, [-1, -2, -3]) and np.allclose(comb, [-1, -5, -3])           # doubles take the epistatic network
    pa, comb = mod._combine(add, np.full(3, np.nan), n_mut, {'single': 1})
    assert np.allclose(comb, [-1, -2, -3])                                              # single network only: additive everywhere
    pa, comb = mod._combine(np.full(3, np.nan), epi, n_mut, {'epistatic': 1})
    assert np.isnan(comb[0]) and comb[1] == -5 and np.isnan(comb[2])                    # epistatic only: doubles only
    pa, comb = mod._combine(np.array([1.0, np.nan, 3.0]), epi, n_mut, {'single': 1, 'epistatic': 1})
    assert np.isnan(comb[1])                                                            # an unscorable row stays NaN
    assert mod._wanted([1, 2, 3], 'single').tolist() == [True, False, False]
    assert mod._wanted([1, 2, 3], 'multi').tolist() == [False, True, True]


def _load_me_module(monkeypatch, tmp_path):
    """baselines/mutate_everything.py against a throwaway 'MutateEverything checkout' and stubbed upstream modules."""
    import paths
    me = tmp_path / 'ME'
    root = me / 'data'
    names = {'oct06_capped_allbl/mutations': [f'oct06_capped_allbl_{n}' for n in ('val', 'valfull', 'test', 'testcomp')],
             'external_test_feb/mutations': ['s669', 'ssym', 'q3421', 'k3822', 'ptmul'],
             'additional_test_march/mutations': ['s571', 's783', 's2648', 's8754'], 's461/mutations': ['s461'],
             'domainome_oct06/mutations': ['domainome'], 'pgym_oct06/mutations': ['PG_A', 'PG_B']}
    for sub, stems in names.items():
        (root / sub).mkdir(parents=True)
        for stem in stems:
            pd.DataFrame({'pdb_id': 'P', 'mut_info': ['A1C', 'A1C:C2D', 'D3E', 'E4F:F5G:G6H'], 'ddg': 0.0}).to_csv(root / sub / f'{stem}.csv', index=False)
    (root / 'dms_oct06' / 'mutations').mkdir(parents=True)
    import importlib
    monkeypatch.setattr(paths, 'MUTATE_EVERYTHING_DIR', me)
    monkeypatch.chdir(tmp_path)
    parser = types.SimpleNamespace(set_defaults=lambda **k: None)
    for name, attrs in {'misc': {}, 'engine_train': {'mem_inputs_to_device': None}, 'main_train': {'get_args_parser': lambda: parser},
                        'data': {'SeqDetDatatset': object, 'protein_collate_fn': None, 'get_dense_mut_infos': None},
                        'modeling': {}, 'modeling.mutate_everything': {'MutateEverything': object}}.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
    import importlib.util
    spec = importlib.util.spec_from_file_location('me_baseline_under_test', BASELINES / 'mutate_everything.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for a in mod.DMS:
        pd.DataFrame({'pdb_id': 'P', 'mut_info': ['A1C']}).to_csv(root / 'dms_oct06' / 'mutations' / f'{a}.csv', index=False)
    return mod


def test_me_set_resolution_and_mutant_filter(monkeypatch, tmp_path):
    mod = _load_me_module(monkeypatch, tmp_path)
    allsets = mod.resolve_sets('all', 'oct06_capped_allbl')
    assert len(allsets) == 3 + 10 + 7 + 1 + 2                       # tsuboyama, external (incl. s461), DMS, domainome, 2 ProteinGym
    assert [p.stem for p in mod.resolve_sets('tsuboyama', 'oct06_capped_allbl')] == ['oct06_capped_allbl_val', 'oct06_capped_allbl_test', 'oct06_capped_allbl_testcomp']
    assert [p.stem for p in mod.resolve_sets('s669,val', 'oct06_capped_allbl')] == ['s669', 'oct06_capped_allbl_val']
    assert [p.stem for p in mod.resolve_sets('PG_B', 'oct06_capped_allbl')] == ['PG_B']              # one ProteinGym DMS by id
    with pytest.raises(SystemExit):
        mod.resolve_sets('nope', 'oct06_capped_allbl')
    (tmp_path / 'ME' / 'data' / 's461' / 'mutations' / 's461.csv').unlink()
    with pytest.raises(SystemExit):
        mod.resolve_sets('s461', 'oct06_capped_allbl')
    src = tmp_path / 'ME' / 'data' / 'external_test_feb' / 'mutations' / 's669.csv'
    assert mod.filter_mutants(src, 'both', tmp_path / 'out') == str(src)
    single = pd.read_csv(mod.filter_mutants(src, 'single', tmp_path / 'out'))
    multi = pd.read_csv(mod.filter_mutants(src, 'multi', tmp_path / 'out'))
    assert single.mut_info.tolist() == ['A1C', 'D3E'] and multi.mut_info.tolist() == ['A1C:C2D', 'E4F:F5G:G6H']
    two = pd.DataFrame({'pdb_id': ['P', 'P', 'Q', 'R'], 'mut_info': ['A1C', 'A1C:C2D', 'D3E', 'E4F']})
    src2 = tmp_path / 'two.csv'
    two.to_csv(src2, index=False)
    first = pd.read_csv(mod.filter_mutants(src2, 'both', tmp_path / 'out', max_proteins=2))          # --max_proteins keeps the first N proteins
    assert first.pdb_id.tolist() == ['P', 'P', 'Q']
    assert pd.read_csv(mod.filter_mutants(src2, 'single', tmp_path / 'out', max_proteins=2)).mut_info.tolist() == ['A1C', 'D3E']
