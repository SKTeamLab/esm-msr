"""Per-scope resource tracking for inference runs: wall time, CPU, host RAM, GPU memory and I/O.

Used by the benchmark wrappers of ESM-MSR, MutateEverything and ThermoMPNN(-D) (and by pgym_runner.py, so every
ProteinGym run through the shared loop gets it too). Kept free of esm_msr imports so it runs in any environment; torch,
psutil and pynvml are optional (without CUDA the GPU columns are empty).

    tracker = Tracker.from_args(args, label='esm_msr', default_dir=out_dir, meta={'dtype': 'bf16'})
    with tracker.scope('stage', 'model_load'):
        model = load()
    with tracker.scope('dataset', 's669') as ds:
        for code, grp in groups:
            with tracker.scope('protein', code, n_rows=len(grp), seq_len=L) as sc:
                with sc.phase('forward'):
                    preds = model(...)
    tracker.close()

Levels are free-form; the wrappers use `stage` (model load, anything outside inference), `dataset` (one benchmark
set, including its data handling) and `protein` (only the inference call on one protein or library, so the sum of
the protein rows is the inference cost of the dataset). A scope may nest in another; peaks are kept per scope.

Each scope row (resources_scopes.jsonl, plus .csv at close) records:
  wall_s       perf_counter between enter and exit, with torch.cuda.synchronize at both ends, so queued GPU work is
               charged to the scope that launched it
  cpu_s        user + system CPU seconds of this process (all threads), cpu_cores = cpu_s / wall_s
  ram_*        resident set size of this process: at start, at end, and the peak over the scope (a sampler thread
               reads /proc every `interval` s; scopes shorter than that still get max(start, end), so a transient
               allocation inside a scope shorter than the interval can be missed)
  vram_*       PyTorch's exact peak allocated / reserved bytes for the scope (counters reset at scope entry), the
               allocated bytes at entry (weights and anything resident), and `vram_device_peak_gb`: the sampled peak of
               the device's used memory, which also counts the CUDA context and non-PyTorch allocations
  io_read_mb / io_write_mb   bytes this process read / wrote (from /proc/self/io), page cache hits excluded
  loadavg1     1-minute load average of the node at exit (a crude neighbour signal)
  t_<phase>_s  wall time of each `scope.phase(name)` block inside the scope (CUDA-synchronised)

The cost is two synchronisations and a few /proc reads per scope plus one sampler thread; tests/test_resource_tracker.py
measures it. Disable with --no_resource_tracking or MSR_TRACK_RESOURCES=0.

Fixed resources: a benchmark comparing models should give each the same slot. `Limits` (env MSR_FIXED_CPUS,
MSR_FIXED_RAM_GB, MSR_FIXED_GPUS, set by slurm_fixed_inference.sh) is checked against what the process really has:
visible GPUs, the CPU affinity mask, the cgroup memory limit. Run metadata (resources_run.json) records the check;
with --strict_resources (MSR_STRICT_RESOURCES=1) a violated limit aborts the run, and a process whose RSS exceeds
the RAM limit raises ResourceLimitExceeded at the next scope exit.
"""
import atexit
import contextlib
import datetime
import json
import os
import re
import resource
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - the wrappers always have torch
    torch = None

GB = 2 ** 30
_PAGE = os.sysconf('SC_PAGE_SIZE')
_SECRET = re.compile(r'(token|key|secret|password)', re.I)


class ResourceLimitExceeded(RuntimeError):
    pass


def _cuda():
    return torch is not None and torch.cuda.is_available()


def _rss_bytes():
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * _PAGE
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _cpu_seconds():
    t = os.times()
    return t.user + t.system


def _io_bytes():
    try:
        d = {}
        with open('/proc/self/io') as f:
            for line in f:
                k, v = line.split(':')
                d[k] = int(v)
        return d['read_bytes'], d['write_bytes']
    except (OSError, KeyError, ValueError):
        return None, None


def _loadavg1():
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def _sync():
    if _cuda():
        torch.cuda.synchronize()


def _gb(x):
    return None if x is None else round(x / GB, 4)


# ---------------------------------------------------------------------------------------------------------------
# what the process really has: GPUs, CPUs, cgroup memory
# ---------------------------------------------------------------------------------------------------------------
def _cgroup_memory_files():
    """(limit_file, peak_file) of this process's memory cgroup, v1 or v2, or (None, None)."""
    try:
        lines = open('/proc/self/cgroup').read().splitlines()
    except OSError:
        return None, None
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, ctrl, path = parts
        if 'memory' in ctrl.split(','):                       # cgroup v1
            base = Path('/sys/fs/cgroup/memory') / path.lstrip('/')
            return base / 'memory.limit_in_bytes', base / 'memory.max_usage_in_bytes'
        if ctrl == '':                                         # cgroup v2
            base = Path('/sys/fs/cgroup') / path.lstrip('/')
            return base / 'memory.max', base / 'memory.peak'
    return None, None


def _read_int(path):
    try:
        s = Path(path).read_text().strip()
        return None if s in ('max', '') else int(s)
    except (OSError, ValueError, TypeError):
        return None


def cgroup_memory():
    """{'limit_gb', 'peak_gb'} of the memory cgroup (None where unavailable or unlimited)."""
    lim_f, peak_f = _cgroup_memory_files()
    lim = _read_int(lim_f) if lim_f else None
    if lim is not None and lim >= 2 ** 60:                     # v1 reports "unlimited" as a huge number
        lim = None
    return {'limit_gb': _gb(lim), 'peak_gb': _gb(_read_int(peak_f)) if peak_f else None}


def visible_gpus():
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is not None:
        return len([x for x in cvd.split(',') if x.strip() != ''])
    return torch.cuda.device_count() if torch is not None else 0


def cpu_affinity():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


class Limits:
    """The resource slot every model is benchmarked in. From the environment, so the Slurm launcher is the single
    place the numbers are set; an unset field is not checked."""

    def __init__(self, cpus=None, ram_gb=None, gpus=None, strict=False):
        self.cpus, self.ram_gb, self.gpus, self.strict = cpus, ram_gb, gpus, strict

    @classmethod
    def from_env(cls):
        def f(name, cast):
            v = os.environ.get(name)
            return cast(v) if v not in (None, '') else None
        return cls(f('MSR_FIXED_CPUS', int), f('MSR_FIXED_RAM_GB', float), f('MSR_FIXED_GPUS', int),
                   os.environ.get('MSR_STRICT_RESOURCES', '0') == '1')

    def any(self):
        return any(x is not None for x in (self.cpus, self.ram_gb, self.gpus))

    def check(self):
        """What the process has against what the slot says: {'ok': bool, 'problems': [...], 'have': {...}}."""
        cg = cgroup_memory()
        have = {'gpus': visible_gpus(), 'cpus': cpu_affinity(), 'cgroup_ram_limit_gb': cg['limit_gb'],
                'torch_threads': torch.get_num_threads() if torch is not None else None}
        problems = []
        if self.gpus is not None and have['gpus'] != self.gpus:
            problems.append(f"{have['gpus']} GPUs visible, slot is {self.gpus}")
        if self.cpus is not None and have['cpus'] is not None and have['cpus'] > self.cpus:
            problems.append(f"{have['cpus']} CPUs in the affinity mask, slot is {self.cpus}")
        if self.cpus is not None and have['torch_threads'] is not None and have['torch_threads'] > self.cpus:
            problems.append(f"torch uses {have['torch_threads']} threads, slot is {self.cpus} CPUs")
        if self.ram_gb is not None:
            lim = have['cgroup_ram_limit_gb']
            if lim is None:
                problems.append('no cgroup memory limit in force (RAM cap not enforced by the scheduler)')
            elif lim > self.ram_gb * 1.02:
                problems.append(f'cgroup memory limit {lim} GB exceeds the slot ({self.ram_gb} GB)')
        return {'ok': not problems, 'problems': problems, 'have': have}


# ---------------------------------------------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------------------------------------------
class _Phase:
    def __init__(self, scope, name):
        self.scope, self.name = scope, name

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        dt = time.perf_counter() - self.t0
        key = f't_{self.name}_s'
        self.scope.phases[key] = self.scope.phases.get(key, 0.0) + dt
        return False


class Scope:
    def __init__(self, tracker, level, name, extra):
        self.tracker, self.level, self.name, self.extra = tracker, level, name, dict(extra)
        self.phases = {}
        self.row = None          # the finished row, for callers that want to copy a few columns into their own table

    def phase(self, name):
        return _Phase(self, name)

    def note(self, **kw):
        """Add columns to this scope's row after the fact (e.g. n_rows known only once the data is built)."""
        self.extra.update(kw)

    def __enter__(self):
        t = self.tracker
        _sync()
        self.parent = t._stack[-1] if t._stack else None
        if _cuda():
            if self.parent is not None:     # the child's reset must not erase the parent's peak so far
                self.parent.vram_alloc_peak = max(self.parent.vram_alloc_peak, torch.cuda.max_memory_allocated())
                self.parent.vram_res_peak = max(self.parent.vram_res_peak, torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            self.vram_alloc_start = torch.cuda.memory_allocated()
        else:
            self.vram_alloc_start = None
        self.vram_alloc_peak = self.vram_alloc_start or 0
        self.vram_res_peak = 0
        self.dev_peak = t._device_used() or 0       # sampled at entry, exit and every `interval`, so even a scope shorter than the interval has two samples
        self.rss_start = _rss_bytes()
        self.rss_peak = self.rss_start
        self.io0 = _io_bytes()
        self.cpu0 = _cpu_seconds()
        self.t_start = time.time()
        self.t0 = time.perf_counter()
        with t._lock:
            t._stack.append(self)
        return self

    def __exit__(self, etype, exc, tb):
        t = self.tracker
        _sync()
        wall = time.perf_counter() - self.t0
        cpu = _cpu_seconds() - self.cpu0
        rss_end = _rss_bytes()
        io1 = _io_bytes()
        if _cuda():
            alloc_peak = max(self.vram_alloc_peak, torch.cuda.max_memory_allocated())
            res_peak = max(self.vram_res_peak, torch.cuda.max_memory_reserved())
            if self.parent is not None:
                self.parent.vram_alloc_peak = max(self.parent.vram_alloc_peak, alloc_peak)
                self.parent.vram_res_peak = max(self.parent.vram_res_peak, res_peak)
        else:
            alloc_peak = res_peak = None
        used_end = t._device_used() or 0
        with t._lock:
            t._stack.remove(self)
            rss_peak = max(self.rss_peak, rss_end)
            dev_peak = max(self.dev_peak, used_end)
            if self.parent is not None:
                self.parent.rss_peak = max(self.parent.rss_peak, rss_peak)
                self.parent.dev_peak = max(self.parent.dev_peak, dev_peak)
        dataset = self.name
        s = self.parent
        while s is not None:
            if s.level == 'dataset':
                dataset = s.name
            s = s.parent
        row = {
            'run_id': t.run_id, 'level': self.level, 'dataset': dataset if self.level != 'dataset' else self.name,
            'name': self.name, **self.extra,
            'wall_s': round(wall, 5), 'cpu_s': round(cpu, 4), 'cpu_cores': round(cpu / wall, 3) if wall > 0 else None,
            'ram_rss_start_gb': _gb(self.rss_start), 'ram_rss_end_gb': _gb(rss_end), 'ram_rss_peak_gb': _gb(rss_peak),
            'vram_weights_gb': _gb(self.vram_alloc_start), 'vram_alloc_peak_gb': _gb(alloc_peak),
            'vram_reserved_peak_gb': _gb(res_peak), 'vram_device_peak_gb': _gb(dev_peak) if dev_peak else None,
            'io_read_mb': None if io1[0] is None or self.io0[0] is None else round((io1[0] - self.io0[0]) / 2 ** 20, 2),
            'io_write_mb': None if io1[1] is None or self.io0[1] is None else round((io1[1] - self.io0[1]) / 2 ** 20, 2),
            'loadavg1': _loadavg1(), 't_start': round(self.t_start, 3),
            'status': 'ok' if etype is None else f'error:{etype.__name__}',
            **{k: round(v, 5) for k, v in self.phases.items()},
        }
        self.row = row
        t._emit(row)
        t._check_ram(rss_peak)
        return False


class _NullScope:
    def phase(self, name):
        return contextlib.nullcontext()

    row = None

    def note(self, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class NullTracker:
    """Drop-in for a disabled tracker: every call is a no-op."""
    enabled = False
    run_id = None

    def scope(self, level, name, **extra):
        return _NullScope()

    def stage(self, name, **extra):
        return _NullScope()

    def close(self, **final):
        pass


class Tracker:
    enabled = True

    def __init__(self, out_dir, label, interval=0.5, meta=None, limits=None, argv=None, flush_s=30.0):
        self.label = label
        stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        job = os.environ.get('SLURM_JOB_ID', 'local')
        self.run_id = f'{label}-{job}-{stamp}'
        self.dir = Path(out_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval = float(interval)
        self._stack = []
        self._lock = threading.Lock()
        self._rows = []
        self._pending = []
        self._flush_s = flush_s
        self._last_flush = time.time()
        self._violation = None
        self._closed = False
        self._dev = None
        if _cuda():
            try:
                self._dev = torch.cuda.current_device()
            except Exception:
                self._dev = None
        self.limits = limits or Limits()
        self._jsonl = open(self.dir / 'resources_scopes.jsonl', 'a')
        self._t_open = time.time()
        self.meta = {'run_id': self.run_id, 'label': label, 'started': datetime.datetime.now().isoformat(timespec='seconds'),
                     'host': socket.gethostname(), 'pid': os.getpid(),
                     'argv': self._redact(argv or sys.argv),
                     'slurm': {k: os.environ.get(k) for k in (
                         'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
                         'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')},
                     'python': sys.version.split()[0], **(meta or {})}
        if torch is not None:
            self.meta['torch'] = torch.__version__
            self.meta['torch_threads'] = torch.get_num_threads()
            if _cuda():
                p = torch.cuda.get_device_properties(torch.cuda.current_device())
                self.meta['gpu'] = {'name': p.name, 'total_gb': round(p.total_memory / GB, 2),
                                    'capability': f'{p.major}.{p.minor}'}
        self.meta['cpus_visible'] = cpu_affinity()
        self.meta['cgroup_memory'] = cgroup_memory()
        self.meta['git'] = self._git_commit()
        self.meta['neighbours_at_start'] = self._neighbours()
        if self.limits.any():
            self.meta['limits'] = {'cpus': self.limits.cpus, 'ram_gb': self.limits.ram_gb, 'gpus': self.limits.gpus,
                                   'strict': self.limits.strict, 'check': self.limits.check()}
            if self.limits.strict and not self.meta['limits']['check']['ok']:
                self._write_meta()
                raise ResourceLimitExceeded('fixed-resource check failed: ' + '; '.join(self.meta['limits']['check']['problems']))
        self._write_meta()
        self._stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample, name='resource-sampler', daemon=True)
        self._sampler.start()
        atexit.register(self.close)       # a crashed run still gets its csv and totals (a SIGKILL leaves the jsonl)

    # -- construction from the wrappers' argparse namespace -------------------------------------------------------
    @classmethod
    def from_args(cls, args, label, default_dir, meta=None):
        if getattr(args, 'no_resource_tracking', False) or os.environ.get('MSR_TRACK_RESOURCES', '1') == '0':
            return NullTracker()
        limits = Limits.from_env()
        if getattr(args, 'strict_resources', False):
            limits.strict = True
        out = getattr(args, 'resource_dir', None) or default_dir
        return cls(out, label, interval=getattr(args, 'resource_interval', 0.5), meta=meta, limits=limits)

    # -- internals ------------------------------------------------------------------------------------------------
    @staticmethod
    def _redact(argv):
        """The command line without secret values (--hf_token X, --api_key=X): it ends up in a file that gets shared."""
        out = []
        for i, a in enumerate(argv):
            if '=' in a and _SECRET.search(a.split('=', 1)[0]):
                out.append(a.split('=', 1)[0] + '=<redacted>')
            elif i and _SECRET.search(argv[i - 1]) and not a.startswith('-'):
                out.append('<redacted>')
            else:
                out.append(a)
        return out

    @staticmethod
    def _git_commit():
        try:
            return subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', '--short', 'HEAD'],
                                  capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _neighbours():
        """Other Slurm jobs on this node right now (the sharing the run was subject to), or None without squeue."""
        node = os.environ.get('SLURMD_NODENAME') or socket.gethostname().split('.')[0]
        try:
            out = subprocess.run(['squeue', '-h', '-w', node, '-t', 'R', '-o', '%i|%C|%m|%b'], capture_output=True,
                                 text=True, timeout=10).stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            return None
        me = os.environ.get('SLURM_JOB_ID')
        jobs = [l.split('|') for l in out if l and l.split('|')[0] != me]
        return {'node': node, 'other_jobs': len(jobs), 'other_cpus': sum(int(j[1]) for j in jobs if j[1].isdigit())}

    def _write_meta(self):
        with open(self.dir / 'resources_run.json', 'w') as f:
            json.dump(self.meta, f, indent=1, default=str)

    def _device_used(self):
        """Used memory of the current CUDA device in bytes (the CUDA context and non-PyTorch allocations included), or None."""
        if self._dev is None:
            return None
        try:
            free, total = torch.cuda.mem_get_info(self._dev)
            return total - free
        except Exception:
            return None

    def _sample(self):
        while not self._stop.wait(self.interval):
            rss = _rss_bytes()
            used = self._device_used()
            with self._lock:
                for s in self._stack:
                    if rss > s.rss_peak:
                        s.rss_peak = rss
                    if used is not None and used > s.dev_peak:
                        s.dev_peak = used
            try:
                self._check_ram(rss)
            except ResourceLimitExceeded:
                pass                      # recorded in self._violation; raised on the main thread at the next scope exit

    def _check_ram(self, rss):
        lim = self.limits.ram_gb
        if lim is not None and rss > lim * GB and self._violation is None:
            self._violation = f'process RSS {rss / GB:.1f} GB exceeded the {lim} GB slot'
            self.meta['ram_limit_violation'] = self._violation
        if self._violation and self.limits.strict:
            raise ResourceLimitExceeded(self._violation)

    def _emit(self, row):
        with self._lock:
            self._rows.append(row)
            self._pending.append(row)
        if time.time() - self._last_flush > self._flush_s or row['level'] == 'dataset':
            self.flush()

    def flush(self):
        with self._lock:
            pending, self._pending = self._pending, []
        for r in pending:
            self._jsonl.write(json.dumps(r, default=str) + '\n')
        self._jsonl.flush()
        self._last_flush = time.time()

    # -- public ---------------------------------------------------------------------------------------------------
    def scope(self, level, name, **extra):
        return Scope(self, level, name, extra)

    def stage(self, name, **extra):
        return Scope(self, 'stage', name, extra)

    def close(self, **final):
        """Flush, write resources_scopes.csv, and finish resources_run.json with the run totals."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self.flush()
        self._jsonl.close()
        try:
            import pandas as pd
            pd.DataFrame(self._rows).to_csv(self.dir / 'resources_scopes.csv', index=False)
        except ImportError:
            pass
        self.meta['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        self.meta['total_wall_s'] = round(time.time() - self._t_open, 2)
        self.meta['process_peak_rss_gb'] = _gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        self.meta['cgroup_memory_at_exit'] = cgroup_memory()
        self.meta['neighbours_at_end'] = self._neighbours()
        if _cuda():
            self.meta['process_peak_vram_alloc_gb'] = _gb(torch.cuda.max_memory_allocated())
        self.meta.update(final)
        self._write_meta()


def add_tracking_args(parser):
    """The flags every wrapper shares."""
    g = parser.add_argument_group('resource tracking')
    g.add_argument('--no_resource_tracking', action='store_true',
                   help='Do not record per-dataset / per-protein wall time, CPU, RAM, VRAM and I/O (on by default).')
    g.add_argument('--resource_dir', type=str, default=None,
                   help='Where the tracker writes <run_id>/resources_{run.json,scopes.jsonl,scopes.csv}.')
    g.add_argument('--resource_interval', type=float, default=0.5, help='Sampler period in seconds for peak RAM / device VRAM.')
    g.add_argument('--strict_resources', action='store_true',
                   help='Abort if the process does not have exactly the slot in MSR_FIXED_CPUS / MSR_FIXED_RAM_GB / MSR_FIXED_GPUS.')
    return parser


def load_scopes(run_dir):
    """The scope table of a finished or running run (reads the jsonl, so a killed run is still readable)."""
    import pandas as pd
    p = Path(run_dir) / 'resources_scopes.jsonl'
    return pd.read_json(p, lines=True) if p.exists() and p.stat().st_size else pd.DataFrame()
