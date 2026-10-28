"""Tests for inference_scripts/resource_tracker.py (CPU only; the CUDA columns are exercised on the cluster)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'inference_scripts'))
import resource_tracker as rt  # noqa: E402


def _tracker(tmp_path, **kw):
    return rt.Tracker(tmp_path, 'test', interval=0.05, **kw)


def test_scope_records_wall_cpu_ram(tmp_path):
    t = _tracker(tmp_path)
    with t.scope('dataset', 'd1'):
        with t.scope('protein', 'p1', n_rows=7, seq_len=70) as sc:
            end = time.perf_counter() + 0.4
            x = 0
            while time.perf_counter() < end:             # one busy core
                x += 1
            buf = np.ones(40_000_000)                    # 320 MB resident
            with sc.phase('forward'):
                time.sleep(0.15)
            time.sleep(0.2)                              # the sampler sees the buffer
            del buf
    t.close()
    df = rt.load_scopes(t.dir)
    p = df[df.level == 'protein'].iloc[0]
    d = df[df.level == 'dataset'].iloc[0]
    assert p['n_rows'] == 7 and p['seq_len'] == 70 and p['dataset'] == 'd1' and p['status'] == 'ok'
    assert p['wall_s'] >= 0.7
    assert 0.3 < p['cpu_cores'] < 1.3
    assert p['ram_rss_peak_gb'] - p['ram_rss_start_gb'] > 0.25          # the 320 MB buffer
    assert 0.13 < p['t_forward_s'] < 0.5
    assert d['ram_rss_peak_gb'] >= p['ram_rss_peak_gb'] - 1e-3          # a child's peak counts for its parent
    assert d['wall_s'] >= p['wall_s']
    assert pd.read_csv(t.dir / 'resources_scopes.csv').shape[0] == 2
    meta = json.load(open(t.dir / 'resources_run.json'))
    assert meta['label'] == 'test' and 'finished' in meta and meta['process_peak_rss_gb'] > 0


def test_error_scope_is_recorded_and_propagates(tmp_path):
    t = _tracker(tmp_path)
    with pytest.raises(ValueError):
        with t.scope('protein', 'bad'):
            raise ValueError('x')
    t.close()
    assert rt.load_scopes(t.dir).iloc[0]['status'] == 'error:ValueError'


def test_disabled_tracker_is_inert(tmp_path):
    class A:
        no_resource_tracking = True
    t = rt.Tracker.from_args(A(), 'x', tmp_path)
    with t.scope('dataset', 'd') as s:
        with s.phase('f'):
            pass
        s.note(a=1)
    t.close()
    assert not any(tmp_path.iterdir())


def test_env_switch_disables(tmp_path, monkeypatch):
    class A:
        no_resource_tracking = False
    monkeypatch.setenv('MSR_TRACK_RESOURCES', '0')
    assert isinstance(rt.Tracker.from_args(A(), 'x', tmp_path), rt.NullTracker)


def test_overhead_per_scope_is_small(tmp_path):
    t = _tracker(tmp_path)
    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        with t.scope('protein', f'p{i}', n_rows=1):
            pass
    per = (time.perf_counter() - t0) / n
    t.close()
    assert per < 2e-3, f'{per * 1e3:.2f} ms per scope'   # per-protein inference takes tens of ms at the very least


def test_argv_secrets_are_redacted(tmp_path):
    t = rt.Tracker(tmp_path, 'test', argv=['x.py', '--hf_token', 'hf_SECRET', '--api_key=abc', '--seed', '3'])
    t.close()
    meta = json.load(open(t.dir / 'resources_run.json'))
    assert 'hf_SECRET' not in json.dumps(meta) and 'abc' not in json.dumps(meta['argv'])
    assert meta['argv'][-2:] == ['--seed', '3']


def test_limits_check_flags_what_the_slot_forbids(monkeypatch):
    monkeypatch.setenv('MSR_FIXED_CPUS', '1')
    monkeypatch.setenv('MSR_FIXED_RAM_GB', '1')
    monkeypatch.setenv('MSR_FIXED_GPUS', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setattr(rt, 'cpu_affinity', lambda: 6)
    monkeypatch.setattr(rt, 'cgroup_memory', lambda: {'limit_gb': 200.0, 'peak_gb': None})
    chk = rt.Limits.from_env().check()
    assert not chk['ok']
    text = ' '.join(chk['problems'])
    assert '2 GPUs' in text and '6 CPUs' in text and '200.0 GB' in text
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(rt, 'cpu_affinity', lambda: 1)
    monkeypatch.setattr(rt, 'cgroup_memory', lambda: {'limit_gb': 1.0, 'peak_gb': None})
    monkeypatch.setattr(rt.torch, 'get_num_threads', lambda: 1)
    assert rt.Limits.from_env().check()['ok']


def test_strict_mode_refuses_a_wrong_slot(tmp_path, monkeypatch):
    monkeypatch.setenv('MSR_FIXED_GPUS', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2')
    with pytest.raises(rt.ResourceLimitExceeded):
        rt.Tracker(tmp_path, 'test', limits=rt.Limits.from_env().__class__(gpus=1, strict=True))


def test_ram_over_limit_raises_in_strict_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, 'cgroup_memory', lambda: {'limit_gb': 0.001, 'peak_gb': None})
    t = rt.Tracker(tmp_path, 'test', interval=0.05, limits=rt.Limits(ram_gb=0.001, strict=True))
    with pytest.raises(rt.ResourceLimitExceeded):
        with t.scope('protein', 'p'):
            time.sleep(0.1)
    t.close()


def test_cgroup_parsing_v1_and_v2(tmp_path, monkeypatch):
    mem = tmp_path / 'memory'
    mem.mkdir()
    (mem / 'memory.limit_in_bytes').write_text(str(64 * 2 ** 30))
    (mem / 'memory.max_usage_in_bytes').write_text(str(3 * 2 ** 30))
    monkeypatch.setattr(rt, '_cgroup_memory_files', lambda: (mem / 'memory.limit_in_bytes', mem / 'memory.max_usage_in_bytes'))
    assert rt.cgroup_memory() == {'limit_gb': 64.0, 'peak_gb': 3.0}
    (mem / 'memory.limit_in_bytes').write_text('9223372036854771712')      # v1 "unlimited"
    assert rt.cgroup_memory()['limit_gb'] is None
    (mem / 'memory.limit_in_bytes').write_text('max')                       # v2 "unlimited"
    assert rt.cgroup_memory()['limit_gb'] is None


def test_pgym_runner_records_one_dms_scope_per_dms(tmp_path):
    import pgym_runner
    pg = tmp_path / 'pg'
    pg.mkdir()
    pd.DataFrame({'DMS_id': ['A', 'B']}).to_csv(pg / 'manifest.csv', index=False)
    for did, n in (('A', 30), ('B', 50)):
        pd.DataFrame({'pdb_file': ['/x.pdb'] * n, 'code': did, 'chain': 'A', 'mut_type_renumbered': [f'A1{i}' for i in range(n)],
                      'mutant': [f'A1{i}' for i in range(n)], 'DMS_score': np.arange(n)}).to_csv(pg / f'{did}.csv', index=False)
    t = _tracker(tmp_path / 'res')
    s = pgym_runner.run_protein_gym(pg, tmp_path / 'out', lambda df: pd.DataFrame({'p': np.arange(len(df)) * 1.0}),
                                    pred_col='p', tracker=t)
    t.close()
    df = rt.load_scopes(t.dir)
    d = df[df.level == 'dms']
    assert sorted(d.name) == ['A', 'B'] and d.set_index('name').loc['B', 'n_rows'] == 50
    assert (d.dataset == 'proteingym').all()
    assert {'ram_peak_gb', 'cpu_cores'} <= set(s.columns) and (s.status == 'done').all()


def test_pgym_runner_without_tracker_is_unchanged(tmp_path):
    import pgym_runner
    pg = tmp_path / 'pg'
    pg.mkdir()
    pd.DataFrame({'DMS_id': ['A']}).to_csv(pg / 'manifest.csv', index=False)
    pd.DataFrame({'pdb_file': ['/x.pdb'] * 5, 'code': 'A', 'chain': 'A', 'mut_type_renumbered': list('abcde'),
                  'mutant': list('abcde'), 'DMS_score': np.arange(5)}).to_csv(pg / 'A.csv', index=False)
    s = pgym_runner.run_protein_gym(pg, tmp_path / 'out', lambda df: pd.DataFrame({'p': np.arange(5) * 1.0}), pred_col='p')
    assert 'ram_peak_gb' not in s.columns and (s.status == 'done').all()


def test_resource_report_sums_inference_rows_only(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analysis_scripts'))
    import resource_report
    t = rt.Tracker(tmp_path, 'demo', interval=0.05)
    with t.stage('model_load'):
        time.sleep(0.05)
    with t.scope('dataset', 'setA'):
        for p, n in (('p1', 10), ('p2', 30)):
            with t.scope('load', f'b{p}'):
                time.sleep(0.02)
            with t.scope('protein', p, n_rows=n, model='single'):          # an extra column named like the report's own
                time.sleep(0.1)
        time.sleep(0.1)                                # data handling outside the inference calls
    t.close()
    summary, prot = resource_report.summarise(t.dir)
    a = summary[summary.dataset == 'setA'].iloc[0]
    assert a.n_proteins == 2 and a.n_rows == 40 and 0.2 <= a.inference_wall_s < 0.3
    assert a.dataset_wall_s > a.inference_wall_s + 0.08 and a.load_wall_s >= 0.04
    assert (summary.dataset == '(stage) model_load').any() and len(prot) == 2 and (prot.network == 'single').all()
    assert resource_report.find_runs([tmp_path]) == [t.dir]


class _FakeCuda:
    """Just enough of torch.cuda to drive the GPU branches without a GPU: an allocator with a resettable peak."""
    def __init__(self):
        self.cur = 0
        self.peak = 0
        self.reserved_peak = 0

    def alloc(self, nbytes):
        self.cur += nbytes
        self.peak = max(self.peak, self.cur)
        self.reserved_peak = max(self.reserved_peak, self.cur + 100)

    def free(self, nbytes):
        self.cur -= nbytes


def test_cuda_peaks_nest_correctly(tmp_path, monkeypatch):
    fake = _FakeCuda()
    c = rt.torch.cuda
    monkeypatch.setattr(rt, '_cuda', lambda: True)
    for name, fn in {'synchronize': lambda: None, 'memory_allocated': lambda: fake.cur,
                     'max_memory_allocated': lambda: fake.peak, 'max_memory_reserved': lambda: fake.reserved_peak,
                     'reset_peak_memory_stats': lambda: (setattr(fake, 'peak', fake.cur), setattr(fake, 'reserved_peak', fake.cur)),
                     'mem_get_info': lambda dev=None: (10 * 2 ** 30 - fake.cur - 2 ** 30, 10 * 2 ** 30),
                     'current_device': lambda: 0, 'set_device': lambda d: None}.items():
        monkeypatch.setattr(c, name, fn, raising=False)
    monkeypatch.setattr(c, 'get_device_properties', lambda d: type('P', (), {'name': 'Fake', 'total_memory': 10 * 2 ** 30, 'major': 8, 'minor': 6})())
    fake.alloc(1 * 2 ** 30)                                       # "weights"
    t = _tracker(tmp_path)
    with t.scope('dataset', 'd'):
        fake.alloc(3 * 2 ** 30)                                   # peak 4 GiB in the parent, before the child
        fake.free(3 * 2 ** 30)
        with t.scope('protein', 'small'):
            fake.alloc(1 * 2 ** 30)
            time.sleep(0.12)
            fake.free(1 * 2 ** 30)
        with t.scope('protein', 'big'):
            fake.alloc(5 * 2 ** 30)
            time.sleep(0.12)
            fake.free(5 * 2 ** 30)
    t.close()
    df = rt.load_scopes(t.dir).set_index('name')
    assert df.loc['small', 'vram_weights_gb'] == 1.0 and df.loc['small', 'vram_alloc_peak_gb'] == 2.0
    assert df.loc['big', 'vram_alloc_peak_gb'] == 6.0
    assert df.loc['d', 'vram_alloc_peak_gb'] == 6.0 and df.loc['d', 'vram_weights_gb'] == 1.0   # parent keeps its own 4 GiB and the child's 6
    assert df.loc['big', 'vram_device_peak_gb'] >= 7.0            # 1 GiB context + weights + 5 GiB, sampled
    # a scope far shorter than the sampler interval still has a device figure (entry and exit samples)
    t2 = _tracker(tmp_path / 'b')
    t2.interval = 3600
    with t2.scope('protein', 'blink'):
        pass
    t2.close()
    assert pd.notna(rt.load_scopes(t2.dir).iloc[0]['vram_device_peak_gb'])
    assert json.load(open(t.dir / 'resources_run.json'))['gpu']['name'] == 'Fake'
