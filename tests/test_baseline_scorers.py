"""The scoring cores of the baseline scripts, with fake models (no SPURS / ThermoMPNN install, no GPU)."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASELINES = Path(__file__).resolve().parent.parent / 'inference_scripts' / 'baselines'
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES.parent))


def _fixed(batch_size):
    """A fixed-mode Batching (no sizer, so no GPU)."""
    import batching
    return batching.Batching(types.SimpleNamespace(batch_mode='fixed', batch_size=batch_size, auto_batch_headroom=0.9, batch_table=None), model='t')


class _LadderSizer:
    """Stands in for the AutoBatchSizer: the same 1, 2, 4, ... ladder (clamped to what is left), no GPU."""
    def run(self, n_items, chunk_fn, label=''):
        ladder, i, b = [], 0, 1
        while i < n_items:
            c = min(b, n_items - i)
            chunk_fn(i, i + c)
            ladder.append(c)
            i += c
            b *= 2
        return types.SimpleNamespace(max_batch=max(ladder), ladder=ladder, ooms=0)


def _auto_stub():
    b = _fixed(None)
    b.mode, b._sizer = 'auto', _LadderSizer()
    return b


def _load_spurs_module(monkeypatch):
    pytest.importorskip('omegaconf')
    """Import baselines/spurs_inference.py with the `spurs` package replaced by stubs; its own esm/torchtext mocking stays in place."""
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
    spec = importlib.util.spec_from_file_location('spurs_baseline_under_test', BASELINES / 'spurs_inference.py')
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
    obj.args, obj.tracker, obj.mutants, obj.models = None, tracker, mutants, models
    obj.batching = _fixed(4096)
    obj.single = _FakeSingle() if models in ('single', 'both') else None
    obj.multi = _FakeMulti() if models in ('multi', 'both') else None
    obj.cfg_single, obj.cfg_multi = {}, {}
    obj.skipped = []
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
    spec = importlib.util.spec_from_file_location('mpnn_baseline_under_test', BASELINES / 'proteinmpnn_inference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mpnn_score_columns_and_mutants_filter(monkeypatch, tmp_path):
    import torch
    mod = _load_mpnn_module(monkeypatch)
    import resource_tracker as rt
    calls = []

    def fake_fast(model, muts, X, S, mask, residue_idx, chain_enc, alphabet, max_tokens, seed, n_samples, coord_noise, batching=None, info_out=None):
        if info_out is not None:
            info_out.append({'batch_mode': 'fixed', 'batch_max': 1, 'n_chunks': 1, 'batch_ooms': 0})
        calls.append(list(muts))
        w = np.arange(1.0, len(muts) + 1)
        return w, w + 10.0

    monkeypatch.setattr(mod, 'score_unmasked_fast', fake_fast)
    t = rt.Tracker(tmp_path, 'test', interval=0.05)
    obj = mod.Mpnn.__new__(mod.Mpnn)
    obj.args = types.SimpleNamespace(mutants='both', max_tokens=32768, n_samples=1, coord_noise=0.0, seed=0)
    obj.batching = _fixed(None)
    obj.tracker, obj.model, obj.device = t, None, torch.device('cpu')
    obj.skipped = []
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
    spec = importlib.util.spec_from_file_location('tmpnn_baseline_under_test', BASELINES / 'thermompnn_inference.py')
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
    p = mod.predict(_FakeNet(), items, _fixed(7), tracker=t, groups=codes, tag={'model': 'single'})
    assert np.allclose(p, np.arange(40) * 2.0)                      # every item scored once, in item order
    t.close()
    df = rt.load_scopes(t.dir)
    assert sorted(df.name) == ['A', 'B', 'C'] and df.n_rows.sum() == 40 and (df.model == 'single').all()
    assert {'t_featurize_s', 't_forward_s', 't_d2h_s'} <= set(df.columns)
    for other in (_fixed(1), _fixed(100), _auto_stub()):             # batching changes nothing about a prediction
        assert np.allclose(mod.predict(_FakeNet(), items, other, tracker=rt.NullTracker(), groups=codes), p)
    assert (df.batch_mode == 'fixed').all() and (df.batch_max <= 7).all() and df.n_chunks.sum() >= 40 // 7


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
    """baselines/mutate_everything_inference.py against a throwaway 'MutateEverything checkout' and stubbed upstream modules."""
    import paths
    me = tmp_path / 'ME'
    root = me / 'data'
    names = {'oct06_capped_allbl/mutations': [f'oct06_capped_allbl_{n}' for n in ('val', 'valfull', 'test', 'testcomp')],
             'external_test_feb/mutations': ['s669', 'ssym', 'q3421', 'k3822', 'ptmul'],
             'additional_test_march/mutations': ['s571', 's783', 's2648', 's8754'], 's461/mutations': ['s461'],
             'domainome_oct06/mutations': ['domainome'], 'pgym_oct06/mutations': ['PG_A', 'PG_B']}
    for sub, stems in names.items():
        (root / sub).mkdir(parents=True)
        for extra in ('fasta', 'msa'):
            (root / Path(sub).parent / extra / 'P').mkdir(parents=True, exist_ok=True)
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
    spec = importlib.util.spec_from_file_location('me_baseline_under_test', BASELINES / 'mutate_everything_inference.py')
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
    # the copy keeps upstream's <set>/{mutations,fasta,msa} layout, because the dataset finds fasta/msa relative to the CSV
    copy = Path(mod.filter_mutants(src, 'single', tmp_path / 'out'))
    assert (copy.parent.parent / 'fasta' / 'P').is_dir() and (copy.parent.parent / 'msa' / 'P').is_dir()
    two = pd.DataFrame({'pdb_id': ['P', 'P', 'Q', 'R'], 'mut_info': ['A1C', 'A1C:C2D', 'D3E', 'E4F']})
    src2 = tmp_path / 'two.csv'
    two.to_csv(src2, index=False)
    first = pd.read_csv(mod.filter_mutants(src2, 'both', tmp_path / 'out', max_proteins=2))          # --max_proteins keeps the first N proteins
    assert first.pdb_id.tolist() == ['P', 'P', 'Q']
    assert pd.read_csv(mod.filter_mutants(src2, 'single', tmp_path / 'out', max_proteins=2)).mut_info.tolist() == ['A1C', 'D3E']


def test_no_baseline_script_shadows_a_package_it_imports():
    """The scripts run with their own directory first on sys.path, so a script named like an upstream package
    (thermompnn.py, spurs.py) would import itself instead of the package. The fake-module tests above cannot see that."""
    import ast
    scripts = sorted(BASELINES.glob('*.py')) + sorted((BASELINES / 'prepare').glob('*.py'))
    stems = {p.stem for p in scripts}
    local_helpers = {'paths', 'benchmark_rows', 'dms_source', 'batching', 'calibration'}
    imported = set()
    for p in scripts:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split('.')[0])
    clash = (stems & imported) - local_helpers
    assert not clash, f'a script is named like a module that scripts import: {sorted(clash)}'
    assert not (BASELINES / 'thermompnn.py').exists() and not (BASELINES / 'spurs.py').exists()


def test_batching_policy_fixed_and_auto():
    import batching
    seen = []
    info = _fixed(None).run(10, lambda a, b: seen.append((a, b)), default_chunk=4)       # the method's own default
    assert seen == [(0, 4), (4, 8), (8, 10)] and info == {'batch_mode': 'fixed', 'batch_max': 4, 'n_items': 10, 'n_chunks': 3, 'batch_ooms': 0}
    seen.clear()
    _fixed(3).run(7, lambda a, b: seen.append((a, b)), default_chunk=lambda: 100)         # --batch_size wins over the default
    assert seen == [(0, 3), (3, 6), (6, 7)]
    seen.clear()
    info = _auto_stub().run(10, lambda a, b: seen.append((a, b)))
    assert [b - a for a, b in seen] == [1, 2, 4, 3] and info['batch_mode'] == 'auto' and info['batch_max'] == 4 and info['n_chunks'] == 4
    assert _fixed(None).run(0, lambda a, b: 1 / 0)['n_chunks'] == 0
    assert batching.merge_info([info, info])['n_chunks'] == 8 and batching.merge_info([]) == {}


def test_spurs_multi_runner_is_batch_invariant(monkeypatch):
    mod = _load_spurs_module(monkeypatch)
    lists = [['A1C'], ['A1D', 'C2D'], ['A1E'], ['A1F'], ['A1G', 'C2E'], ['A1H'], ['A1I']]

    def run(b):
        vals, info = mod.run_multi(_FakeMulti(), {'seq': 'ACDEFGHIKL'}, lists, b)
        return vals, info
    ref, info = run(_fixed(100))
    assert np.allclose(ref, [0.5 * len(m) for m in lists]) and info['n_chunks'] == 2           # one call per mutation count
    for other in (_fixed(1), _fixed(2), _auto_stub()):
        v, i = run(other)
        assert np.allclose(v, ref) and i['batch_max'] >= 1


def test_auto_mode_builds_the_esm_msr_sizer_by_path():
    import batching
    b = batching.Batching(types.SimpleNamespace(batch_mode='auto', batch_size=8, auto_batch_headroom=0.85, batch_table=None), model='t')
    assert type(b.sizer).__name__ == 'AutoBatchSizer' and b.sizer.max_batch == 8 and b.sizer.headroom == 0.85
    assert b.meta() == {'batch_mode': 'auto', 'batch_size_arg': 8, 'auto_batch_headroom': 0.85, 'batch_table': None}


def test_calibration_helpers(tmp_path):
    import calibration
    pdb = tmp_path / 'x.pdb'
    pdb.write_text(''.join(f'ATOM  {i:5d}  CA  {r} A{i:4d}    0.0 0.0 0.0\n' for i, r in enumerate(['ALA', 'GLY', 'MSE', 'LEU', 'LEU'], 1))
                   + 'ATOM      9  CA  VAL B   1    0.0 0.0 0.0\n')
    assert calibration.pdb_chain_seq(pdb, 'A') == 'AGXLL' and calibration.pdb_chain_seq(pdb, 'B') == 'V'
    seq = 'ACDEFGHIKLMNPQRSTVWY' * 3                                                     # 60 residues
    v = calibration.mixed_variants(seq, cap=400, seed=1)
    assert v == calibration.mixed_variants(seq, cap=400, seed=1) and len(v) == 400          # deterministic, capped
    singles = [m for m in v if ':' not in m]
    doubles = [m for m in v if ':' in m]
    assert len(singles) == len(doubles) == 200
    for m in v:                                                                            # every mutation is valid for the sequence
        parts = [(x[0], int(x[1:-1]), x[-1]) for x in m.split(':')]
        assert all(seq[p - 1] == w and t != w for w, p, t in parts)
        assert [p for _, p, _ in parts] == sorted(p for _, p, _ in parts) and len({p for _, p, _ in parts}) == len(parts)
    assert [m for m in calibration.mixed_variants('AXA', cap=10) if ':' not in m] and all('2' not in m[1:-1] for m in calibration.mixed_variants('AXA', cap=10))
    short = calibration.mixed_variants('ACDEF', cap=400, seed=0)                          # fewer singles than cap // 2: repeated, so the list is long enough
    assert len([m for m in short if ':' not in m]) == 200 and len({m for m in short if ':' not in m}) == 95
    structs = [{'code': c, 'seq': 'A' * n} for c, n in (('a', 30), ('b', 90), ('c', 60))]
    assert [s['code'] for s in calibration.longest(structs, 2)] == ['b', 'c']
    seen = []

    def run_one(struct, muts):
        seen.append((struct['code'], len(muts)))
        return {'batch_max': 1000 // len(struct['seq']), 'batch_ooms': 0}
    sizes, detail = calibration.calibrate({'d1': structs, 'd2': structs[:1]}, run_one, k=2, cap=50, log=lambda m: None)
    assert sizes == {'d1': 11, 'd2': 33}                                                   # the smallest batch of the k longest wins
    assert [d['code'] for d in detail['d1']] == ['b', 'c'] and not any(d['list_bounded'] for d in detail['d1'])
    # a list the sizer got through bounds the answer: it is made 4x longer until memory is what stops it (or max_cap)
    lens = []

    def run_bounded(struct, muts):
        lens.append(len(muts))
        return {'batch_max': min(len(muts) // 2, 300), 'n_items': len(muts), 'batch_ooms': 0}      # memory stops at 300
    sizes2, detail2 = calibration.calibrate({'d': structs}, run_bounded, k=1, cap=100, max_cap=6400, log=lambda m: None)
    assert lens == [100, 400, 1600] and sizes2 == {'d': 300} and detail2['d'][0]['variants'] == 1600 and not detail2['d'][0]['list_bounded']
    lens.clear()
    sizes3, detail3 = calibration.calibrate({'d': structs}, lambda s_, m_: {'batch_max': len(m_), 'n_items': len(m_)}, k=1, cap=100, max_cap=400, log=lambda m: None)
    assert sizes3 == {'d': 400} and detail3['d'][0]['list_bounded']                           # never memory-bound: flagged
    assert calibration.worst([{'batch_max': 8, 'n_items': 100, 'batch_ooms': 1}, {'batch_max': 3, 'n_items': 90, 'batch_ooms': 2}]) == \
        {'batch_max': 3, 'n_items': 90, 'batch_ooms': 3}                                      # both memory-limited: the smaller
    assert calibration.worst([{'batch_max': 8, 'n_items': 12}, {'batch_max': 30, 'n_items': 500, 'batch_ooms': 0}]) == \
        {'batch_max': 30, 'n_items': 500, 'batch_ooms': 0}                                    # the short group (12 items) does not limit
    assert calibration.list_bounded(calibration.worst([{'batch_max': 8, 'n_items': 12}, {'batch_max': 5, 'n_items': 9}]))   # all bounded
    assert not calibration.list_bounded({'batch_max': 5, 'n_items': 100}) and calibration.list_bounded({'batch_max': 30, 'n_items': 100})
    assert not calibration.list_bounded({'batch_max': 0, 'n_items': 0}) and not calibration.list_bounded({'batch_max': 7})
    calibration.save_entries(tmp_path / 'a.json', {'m1': sizes}, {'m1': detail}, {'gpu': 'G', 'headroom': 0.9})
    calibration.save_entries(tmp_path / 'b.json', {'m2': {'d1': 4}}, {'m2': {}}, {'gpu': 'G', 'headroom': 0.9})
    calibration.merge([tmp_path / 'a.json', tmp_path / 'b.json'], tmp_path / 'table.json')
    t = json.load(open(tmp_path / 'table.json'))
    assert t['m1'] == {'d1': 11, 'd2': 33, 'proteingym': 'auto'} and t['m2']['d1'] == 4 and t['_meta']['gpu'] == 'G'
    calibration.save_entries(tmp_path / 'c.json', {'m3': {}}, {'m3': {}}, {'gpu': 'OTHER', 'headroom': 0.9})
    with pytest.raises(AssertionError):                                                   # tables from different GPUs do not mix
        calibration.merge([tmp_path / 'a.json', tmp_path / 'c.json'], tmp_path / 'bad.json')


def test_table_mode_batching(tmp_path):
    import batching
    table = tmp_path / 't.json'
    table.write_text(json.dumps({'m': {'s571': 5, 'dmsA': 3, 'proteingym': 'auto'}, 'm_epistatic': {'s571': 2}}))
    args = types.SimpleNamespace(batch_mode='table', batch_size=None, auto_batch_headroom=0.9, batch_table=str(table))
    b = batching.Batching(args, model='m')
    assert batching.dataset_key('external_s571') == 's571' and batching.dataset_key('dms_dmsA') == 'dmsA' and batching.dataset_key('x') == 'x'
    seen = []
    b.dataset = 'external_s571'
    info = b.run(12, lambda a, c: seen.append((a, c)))
    assert seen == [(0, 5), (5, 10), (10, 12)] and info['batch_mode'] == 'table' and info['batch_max'] == 5
    seen.clear()
    b.run(5, lambda a, c: seen.append((a, c)), model='m_epistatic')                        # a per-network column
    assert seen == [(0, 2), (2, 4), (4, 5)]
    b.dataset = 'dms_dmsA'
    seen.clear()
    b.run(7, lambda a, c: seen.append((a, c)))
    assert [c - a for a, c in seen] == [3, 3, 1]
    b.dataset = 'proteingym'                                                              # `auto` entries use the dynamic sizer
    b._sizer = _LadderSizer()
    assert b.run(10, lambda a, c: None)['batch_mode'] == 'auto'
    b.dataset = 'not_in_table'
    with pytest.raises(KeyError, match='calibrat'):
        b.run(3, lambda a, c: None)
    over = batching.Batching(types.SimpleNamespace(batch_mode='table', batch_size=4, auto_batch_headroom=0.9, batch_table=str(table)), model='m')
    seen.clear()
    over.run(9, lambda a, c: seen.append((a, c)))                                         # --batch_size overrides the table
    assert [c - a for a, c in seen] == [4, 4, 1]
    with pytest.raises(FileNotFoundError, match='calibrate'):
        batching.Batching(types.SimpleNamespace(batch_mode='table', batch_size=None, auto_batch_headroom=0.9, batch_table=str(tmp_path / 'none.json')), model='m')


def test_spurs_parse_with_overwritten_sequence_needs_no_patched_spurs(monkeypatch):
    import torch
    mod = _load_spurs_module(monkeypatch)
    seen = {}

    def alt_parse_PDB(path, chain):
        return [{'seq': 'ACDE', 'seq_chain_A': 'ACDE', 'resn_list': ['1', '2', '3', '4']}]

    def get_pdb(d, name, model_name, check_assert=True):
        seen.update(d)
        return dict(d)

    class Alphabet:
        def __init__(self, **kw):
            pass

        def featurize(self, items):
            return items[0]

    utils_mod = types.ModuleType('spurs.datamodules.datasets.utils')
    utils_mod.alt_parse_PDB, utils_mod.get_pdb = alt_parse_PDB, get_pdb
    monkeypatch.setitem(sys.modules, 'spurs.datamodules.datasets.utils', utils_mod)
    monkeypatch.setattr(sys.modules['spurs.datamodules.datasets.data_utils'], 'Alphabet', Alphabet)
    out = mod.parse_pdb_with_seq('x.pdb', 'lib', 'A', {'datamodule': {'alphabet': {}}}, overwrite_seq='AGDE', device='cpu')
    assert seen['seq'] == 'AGDE' and seen['seq_chain_A'] == 'AGDE' and isinstance(out['mut_ids'], torch.Tensor)
    with pytest.raises(AssertionError):                                   # a sequence of another length is a bug, not a skip
        mod.parse_pdb_with_seq('x.pdb', 'lib', 'A', {'datamodule': {'alphabet': {}}}, overwrite_seq='AGD', device='cpu')
    # without overwrite_seq it is upstream's function, called with no extra argument
    assert mod.parse_pdb_with_seq('x.pdb', 'lib', 'A', {}, device='cpu') == {'seq': 'ACDEFGHIKL'}


def test_me_predict_survives_a_protein_that_runs_out_of_memory(monkeypatch, tmp_path):
    mod = _load_me_module(monkeypatch, tmp_path)
    import resource_tracker as rt
    batches = [{'pdb_ids': [n], 'seqs': ['A' * 5]} for n in ('P1', 'P2', 'P3')]

    def fake_batch(model, batch, device, args, sc):
        if batch['pdb_ids'][0] == 'P2':
            raise RuntimeError('CUDA out of memory. Tried to allocate 20.00 GiB')
        if batch['pdb_ids'][0] == 'P4':
            raise RuntimeError('some other runtime error')
        return {batch['pdb_ids'][0]: {'mutations': [], 'scores': []}}
    monkeypatch.setattr(mod, '_predict_batch', fake_batch)
    t = rt.Tracker(tmp_path / 'res', 'test', interval=0.05)
    failed = []
    preds = mod.predict(types.SimpleNamespace(eval=lambda: None), iter(batches), None, None, t, failed)
    assert sorted(preds) == ['P1', 'P3'] and failed == ['P2']                       # the others still scored, the failure is listed
    t.close()
    df = rt.load_scopes(t.dir)
    assert df[df.level == 'protein'].set_index('name').loc['P2', 'status'] == 'error:RuntimeError'
    with pytest.raises(RuntimeError, match='some other'):                            # only OOM is tolerated
        mod.predict(types.SimpleNamespace(eval=lambda: None), iter([{'pdb_ids': ['P4'], 'seqs': ['A']}]), None, None, None, [])
    with pytest.raises(RuntimeError, match='out of memory'):                         # and only when the caller collects failures
        mod.predict(types.SimpleNamespace(eval=lambda: None), iter([batches[1]]), None, None, None, None)
