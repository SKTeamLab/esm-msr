"""Peak CUDA memory and step time of ``--block_training`` blocks of different sizes, on the real model and the real training libraries.

    PYTHONPATH=src python analysis_scripts/block_memory_probe.py --out probe.jsonl [--libpick longest|median] [--singles 8 16 32 48] \\
        [--pair_chunks 10 20 0] [--reps 4] -- <the training flags of the run being sized: --config ... --cache_path ... --block_training ...>

The training entry point runs unchanged, but its sampler is replaced by a plan: ``reps`` singles blocks of B positions for every B in
``--singles`` (no checkpointing, ``block_row_chunk`` 0, so the number is the cost of B live graphs), then ``reps`` pair blocks for every
chunk in ``--pair_chunks`` (0 = all 40 mutant passes alive at once). Each step's peak is ``torch.cuda.max_memory_allocated`` since the step
began, so the one-off load of the base model (10.7 GB) does not hide it. A step that runs out of memory is recorded as OOM and the plan goes
on. The first block of every group is a warm-up and is left out of the means. Library choice matters (memory scales with its length):
``longest`` is the worst case of the training split, ``median`` the typical one.
"""
import argparse
import gc
import json
import random
import statistics
import sys
import time

import numpy as np
import torch
import lightning.pytorch as pl

from esm_msr import blocks, training

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument('--out', required=True)
ap.add_argument('--libpick', choices=['longest', 'median'], default='longest')
ap.add_argument('--singles', type=int, nargs='*', default=[4, 8, 12, 16, 24, 32, 40, 48, 56, 64])
ap.add_argument('--pair_chunks', type=int, nargs='*', default=[10, 20, 0])
ap.add_argument('--reps', type=int, default=4)
OPT, REST = ap.parse_known_args()
sys.argv = ['training.py'] + [a for a in REST if a != '--']

PLAN, RECORDS, STATE = [], [], {'oom': False}


def pick(libs, pool, how):
    pool = sorted(pool, key=lambda li: libs[li].seq.shape[0])
    return pool[-1] if how == 'longest' else pool[len(pool) // 2]


def make_plan(sampler):
    libs = sampler.libs
    ls = pick(libs, [li for li, l in enumerate(libs) if len(l.single_pos)], OPT.libpick)
    paired = [li for li, l in enumerate(libs) if l.n_pairs]
    lp = pick(libs, paired, OPT.libpick) if paired else None
    rng, plan = random.Random(0), []
    if lp is not None:                                      # the pair blocks first: the answer that decides the most, before any OOM can fragment the pool
        lib = libs[lp]
        q = int(np.argmax(np.isfinite(lib.pair_data['y']).reshape(lib.n_pairs, -1).sum(1)))
        for chunk in OPT.pair_chunks:
            for rep in range(OPT.reps):
                plan.append(dict(kind='pair', B=0, chunk=chunk, rep=rep, L=int(lib.seq.shape[0]), lib=lib.name, desc=('p', lp, q)))
    lib = libs[ls]
    for B in OPT.singles:
        B = min(B, len(lib.single_pos))
        for rep in range(OPT.reps):
            plan.append(dict(kind='single', B=B, chunk=0, rep=rep, L=int(lib.seq.shape[0]), lib=lib.name,
                             desc=('s', ls, tuple(rng.sample(range(len(lib.single_pos)), B)))))
    return plan


def _len(self):
    if not PLAN:
        PLAN.extend(make_plan(self))
    return len(PLAN)


def _iter(self):
    _len(self)
    yield from [p['desc'] for p in PLAN]


blocks.BlockSampler.__len__, blocks.BlockSampler.__iter__ = _len, _iter


class Probe(pl.Callback):
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        p = PLAN[batch_idx]
        module.hparams['block_row_chunk'] = p['chunk']
        STATE['oom'] = False
        torch.cuda.synchronize()
        STATE['base'] = torch.cuda.memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        STATE['t0'] = time.time()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        p = PLAN[batch_idx]
        passes = 2 * p['B'] if p['kind'] == 'single' else 2 + 40
        rec = {k: p[k] for k in ('kind', 'B', 'chunk', 'rep', 'L', 'lib')}
        rec.update(oom=STATE['oom'], sec=round(time.time() - STATE['t0'], 3), base_gb=round(STATE['base'], 3),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3), reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                   passes=passes, total_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
        RECORDS.append(rec)
        print('PROBE ' + json.dumps(rec), flush=True)
        with open(OPT.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')


_step = training.ESM3EpistasisLightningModule.training_step


def guarded(self, batch, batch_idx):
    try:
        return _step(self, batch, batch_idx)
    except torch.cuda.OutOfMemoryError:
        STATE['oom'] = True
    for q in self.parameters():
        q.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    return None


training.ESM3EpistasisLightningModule.training_step = guarded
_init = pl.Trainer.__init__


def init(self, *a, **k):
    k['callbacks'] = list(k.get('callbacks') or []) + [Probe()]
    k['log_every_n_steps'] = 1000000
    _init(self, *a, **k)


pl.Trainer.__init__ = init


def summary():
    print('\n== summary (library: %s)' % OPT.libpick)
    groups = {}
    for r in RECORDS:
        groups.setdefault((r['kind'], r['B'], r['chunk']), []).append(r)
    fit = []
    for (kind, B, chunk), rs in groups.items():
        ok = [r for r in rs if r['rep'] > 0 and not r['oom']]
        oom = any(r['oom'] for r in rs)
        if ok:
            sec, peak = statistics.mean(r['sec'] for r in ok), max(r['peak_gb'] for r in ok)
            print(f"{kind:6s} B={B:3d} chunk={chunk:3d} L={rs[0]['L']:3d}  peak {peak:6.2f} GB  {sec:6.2f} s/step  {rs[0]['passes'] / sec:6.1f} passes/s"
                  + ('  (a rep OOM)' if oom else ''))
            if kind == 'single' and not oom:
                fit.append((B, peak))
        else:
            print(f"{kind:6s} B={B:3d} chunk={chunk:3d} L={rs[0]['L']:3d}  OOM")
    if len(fit) >= 2 and RECORDS:
        x, y = np.array([f[0] for f in fit], float), np.array([f[1] for f in fit], float)
        slope, icpt = np.polyfit(x, y, 1)
        total = RECORDS[0]['total_gb']
        print(f"singles: peak = {icpt:.2f} GB + {slope:.3f} GB x B; the {total:.1f} GB card (1 GB spare) holds B = {int((total - 1.0 - icpt) / slope)}")


try:
    training.main()
except BaseException as e:               # --skip_val leaves the checkpoint callback without its monitor: it raises after the last step
    print('EXC', type(e).__name__, str(e)[:200])
finally:
    summary()
