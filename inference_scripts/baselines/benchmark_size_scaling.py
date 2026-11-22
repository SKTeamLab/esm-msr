"""Controlled protein-size scaling benchmark: minimum VRAM (B=1) and full 1-mut / 2-mut screening cost vs length L.

Runs ONE model per invocation across a shared protein panel (size_scaling_panel.py) in ascending length L,
stopping automatically once the model hits CUDA OOM (or an architectural length limit) at its minimum batch size (B=1).

Models:
  esm_msr_wt, esm_msr_dual, me_st1, me_st2,
  thermompnn_single, thermompnn_d, spurs_single, spurs_multi,
  proteinmpnn, prosst

Outputs under --out/<model>/:
  meta.json       GPU, Slurm slot, model checkpoint, commit
  results.jsonl   one row per tested protein (flushed after every protein)
  results.csv     tabular summary
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import resource_tracker  # noqa: E402

GB = 2 ** 30
ALPHABET20 = 'ACDEFGHIKLMNPQRSTVWY'


def is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ('out of memory' in msg or 'cuda error: out of memory' in msg or 'cublas_status_alloc_failed' in msg)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class DevicePeakSampler:
    """High-frequency (15 ms) driver-level VRAM peak sampler around a GPU scope."""

    def __init__(self, interval: float = 0.015):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        self._dev = torch.cuda.current_device() if torch.cuda.is_available() else None

    def _used(self) -> int:
        if self._dev is None:
            return 0
        try:
            free, total = torch.cuda.mem_get_info(self._dev)
            return total - free
        except Exception:
            return 0

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            u = self._used()
            if u > self.peak_bytes:
                self.peak_bytes = u

    def __enter__(self):
        if self._dev is not None:
            torch.cuda.synchronize()
            self.peak_bytes = self._used()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._dev is not None:
            torch.cuda.synchronize()
            u = self._used()
            if u > self.peak_bytes:
                self.peak_bytes = u
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        return False


def measure_once(fn):
    """Run fn() once with exact CUDA sync, PyTorch peak memory counters, and 15 ms device-level peak polling."""
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats()
    weights_gb = torch.cuda.memory_allocated() / GB
    with DevicePeakSampler() as smp:
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
    alloc_peak_gb = torch.cuda.max_memory_allocated() / GB
    res_peak_gb = torch.cuda.max_memory_reserved() / GB
    dev_peak_gb = smp.peak_bytes / GB
    return out, dict(
        wall_s=round(wall_s, 6),
        vram_weights_gb=round(weights_gb, 4),
        vram_alloc_peak_gb=round(alloc_peak_gb, 4),
        vram_reserved_peak_gb=round(res_peak_gb, 4),
        vram_device_peak_gb=round(dev_peak_gb, 4),
    )


def measure_stable(fn, max_repeats: int = 3, fast_thresh_s: float = 0.5, force_warmup: bool = False):
    """Measure fn(); if force_warmup, discard the first pass (CUDA JIT/cuBLAS init) before timing."""
    if force_warmup:
        _ = fn()
        torch.cuda.synchronize()
    out, m0 = measure_once(fn)
    if m0['wall_s'] >= fast_thresh_s or max_repeats <= 1:
        return out, m0
    times = [m0['wall_s']]
    allocs, resvs, devs = [m0['vram_alloc_peak_gb']], [m0['vram_reserved_peak_gb']], [m0['vram_device_peak_gb']]
    for _ in range(max_repeats - 1):
        out, m = measure_once(fn)
        times.append(m['wall_s'])
        allocs.append(m['vram_alloc_peak_gb'])
        resvs.append(m['vram_reserved_peak_gb'])
        devs.append(m['vram_device_peak_gb'])
    m0['wall_s'] = round(float(np.median(times)), 6)
    m0['vram_alloc_peak_gb'] = round(float(max(allocs)), 4)
    m0['vram_reserved_peak_gb'] = round(float(max(resvs)), 4)
    m0['vram_device_peak_gb'] = round(float(max(devs)), 4)
    return out, m0


def find_opt_batch(run_batch_fn, candidates: list[int], vram_cap_frac: float = 0.88, max_probe_s: float = 6.0):
    """Probe ascending candidate batch sizes until OOM, VRAM cap, or probe time limit; return (best_B, metrics)."""
    total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / GB
    best_b = None
    best_m = None
    for b in candidates:
        try:
            _, m = measure_once(lambda b=b: run_batch_fn(b))
        except Exception as e:
            if is_oom(e):
                cleanup_cuda()
                break
            raise
        best_b, best_m = b, m
        if m['vram_device_peak_gb'] >= total_gb * vram_cap_frac or m['wall_s'] >= max_probe_s:
            break
    return best_b, best_m


def timed_throughput(run_batch_fn, batch_size: int, total_items: int, max_wall_s: float = 10.0, min_batches: int = 2):
    """Measure steady-state throughput (items/s) at batch_size; if total_items runs within max_wall_s, runs all of them."""
    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats()
    n_done = 0
    # Warmup 1 batch if total_items > batch_size
    if total_items > batch_size:
        run_batch_fn(batch_size)
        torch.cuda.synchronize()
    with DevicePeakSampler() as smp:
        t0 = time.perf_counter()
        n_batches = 0
        while n_done < total_items:
            cur = min(batch_size, total_items - n_done)
            run_batch_fn(cur)
            torch.cuda.synchronize()
            n_done += cur
            n_batches += 1
            elapsed = time.perf_counter() - t0
            if n_batches >= min_batches and elapsed >= max_wall_s:
                break
        elapsed = time.perf_counter() - t0
    rate = n_done / elapsed if elapsed > 0 else float('nan')
    return dict(
        measured_items=n_done,
        measured_full=(n_done == total_items),
        elapsed_s=round(elapsed, 6),
        rate_vps=round(rate, 4),
        vram_alloc_peak_gb=round(torch.cuda.max_memory_allocated() / GB, 4),
        vram_device_peak_gb=round(smp.peak_bytes / GB, 4),
    )


def load_seq(fasta_path: str) -> str:
    lines = [ln.strip() for ln in Path(fasta_path).read_text().splitlines() if ln.strip() and not ln.startswith('>')]
    return ''.join(lines)


def make_single_muts(seq: str, limit: int | None = None) -> list[str]:
    out = []
    for i, wt in enumerate(seq):
        wt_c = wt if wt in ALPHABET20 else 'A'
        for mt in ALPHABET20:
            if mt != wt_c:
                out.append(f'{wt_c}{i + 1}{mt}')
                if limit is not None and len(out) >= limit:
                    return out
    return out


def make_double_muts(seq: str, count: int) -> list[str]:
    """Deterministic diverse double mutations spread across the sequence."""
    L = len(seq)
    rng = np.random.default_rng(42)
    out = []
    p1 = rng.integers(0, L - 1, size=count)
    p2 = rng.integers(1, L, size=count)
    same = p1 == p2
    p2[same] = (p1[same] + 1) % L
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    a1 = rng.integers(0, 19, size=count)
    a2 = rng.integers(0, 19, size=count)
    for i in range(count):
        w1 = seq[lo[i]] if seq[lo[i]] in ALPHABET20 else 'A'
        w2 = seq[hi[i]] if seq[hi[i]] in ALPHABET20 else 'A'
        alts1 = [c for c in ALPHABET20 if c != w1]
        alts2 = [c for c in ALPHABET20 if c != w2]
        out.append(f'{w1}{int(lo[i]) + 1}{alts1[a1[i]]}:{w2}{int(hi[i]) + 1}{alts2[a2[i]]}')
    return out


def n_singles(L: int) -> int:
    return 19 * L


def n_doubles(L: int) -> int:
    return (L * (L - 1) // 2) * 361


def parse_muts_list(mut_strs: list[str]) -> list[list[tuple[str, int, str]]]:
    """Convert ['A12C', 'V5L:G8P'] directly into _prepare_sparse_batch's list-of-tuples format."""
    return [[(tok[0], int(tok[1:-1]), tok[-1]) for tok in m.split(':')] for m in mut_strs]


# ==============================================================================================
# Model runners
# ==============================================================================================
def bench_esm_msr(panel: list[dict], mode: str, args) -> list[dict]:
    """mode in ('esm_msr_wt', 'esm_msr_dual', 'esm_msr_indep', 'esm_msr_marg')."""
    from esm_msr.models import MSRModel
    from esm_msr.inference import (
        parse_hparams_to_lora_config,
        resolve_calibration,
        preprocess_structure,
        _prepare_sparse_batch,
        _dtype_autocast,
    )

    if mode == 'esm_msr_indep':
        ckpt = args.esm_msr_indep_ckpt
        mask_strategy = 'independent'
    elif mode == 'esm_msr_marg':
        ckpt = args.esm_msr_marg_ckpt
        mask_strategy = {'wt': 'independent', 'mt': 'marginal'}
    else:
        ckpt = args.esm_msr_ckpt
        mask_strategy = None
    hparams_path = str(Path(ckpt).parent / 'hparams.yaml')
    parsed_hp = parse_hparams_to_lora_config(hparams_path, sigma=1.0)
    lora_cfg = {'wt_config': parsed_hp['wt_config'], 'mt_config': parsed_hp['mt_config']}
    cal = resolve_calibration(parsed_hp['calibration'])
    dev = torch.device('cuda')
    t0 = time.perf_counter()
    model = MSRModel(
        lora_config=lora_cfg,
        shared_scale_init=1,
        shared_bias_init=0,
        inference_mode=True,
        log_likelihood=True,
        use_plddt=True,
        quaternary_mode=False,
        model_dtype=torch.bfloat16,
        adapter_mode=parsed_hp['adapter_mode'],
        lora_mode=parsed_hp['lora_mode'],
        tie_mt_scale=cal['tie_mt_scale'],
        per_mutation_bias=cal['per_mutation_bias'],
        native_cond_mt_input=parsed_hp['native_cond_for_mt'],
        mask_native_cond_struct=parsed_hp['mask_native_cond_struct'],
        fused_antisymmetric=parsed_hp['fused_antisymmetric'],
    )
    model.load_lora_weights(checkpoint_path=ckpt, load_wt_only=(mode == 'esm_msr_wt'))
    model.to(dev)
    model.eval()
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    double_impractical = False
    for idx, item in enumerate(panel):
        L = item['seq_len']
        pdb_path = item['pdb_path']
        row = _base_row(mode, item, load_s)
        try:
            # Measure minimum B=1 VRAM across preprocess_structure + forward pass(es) at batch_size=1
            def _b1():
                seq_toks, coords, plddt, struct_toks, wt_seq = preprocess_structure(
                    model, pdb_path, 'A', dev, backbone_mutation=None
                )
                cached_wt = None
                if mask_strategy is None:
                    with torch.no_grad(), _dtype_autocast(model, dev):
                        out = model._get_esm3_outputs(seq_toks, coords, struct_toks, plddt)
                        cached_wt = {
                            'seq': seq_toks,
                            'logits': model._process_logits(out.sequence_logits.float()),
                            'embeddings': getattr(out, 'embeddings', None),
                        }
                muts_1 = make_single_muts(wt_seq, limit=1)
                sp1 = _prepare_sparse_batch(model, parse_muts_list(muts_1), dev)
                with torch.no_grad(), _dtype_autocast(model, dev):
                    _ = model.score_screening_batch(
                        seq_toks, sp1['mut_pos'], sp1['wt_id'], sp1['mt_id'], sp1['mut_mask'],
                        coords=coords, structure_tokens=struct_toks, plddt=plddt,
                        mask_strategy=mask_strategy, batch_size=1,
                        skip_reverse=(mode == 'esm_msr_wt'), cached_wt_esm3=cached_wt, quiet=True,
                    )
                return seq_toks, coords, plddt, struct_toks, wt_seq, cached_wt

            (seq_toks, coords, plddt, struct_toks, wt_seq, cached_wt), m_b1 = measure_stable(
                _b1, max_repeats=3, fast_thresh_s=0.4, force_warmup=(idx == 0)
            )
            row.update(
                status='ok',
                vram_weights_gb=m_b1['vram_weights_gb'],
                vram_min_alloc_gb=m_b1['vram_alloc_peak_gb'],
                vram_min_reserved_gb=m_b1['vram_reserved_peak_gb'],
                vram_min_device_gb=m_b1['vram_device_peak_gb'],
                wall_b1_s=m_b1['wall_s'],
            )

            # Separate pure WT-pass + structure-encode time (needed for both WT-view and Dual-view)
            def _wt_only_full():
                st, co, pl, stk, ws = preprocess_structure(model, pdb_path, 'A', dev, backbone_mutation=None)
                with torch.no_grad(), _dtype_autocast(model, dev):
                    out = model._get_esm3_outputs(st, co, stk, pl)
                    cwt = {
                        'seq': st,
                        'logits': model._process_logits(out.sequence_logits.float()),
                        'embeddings': getattr(out, 'embeddings', None),
                    }
                all_1 = make_single_muts(ws)
                sp = _prepare_sparse_batch(model, parse_muts_list(all_1), dev)
                with torch.no_grad(), _dtype_autocast(model, dev):
                    _ = model.score_screening_batch(
                        st, sp['mut_pos'], sp['wt_id'], sp['mt_id'], sp['mut_mask'],
                        coords=co, structure_tokens=stk, plddt=pl,
                        mask_strategy=None, batch_size=max(1, len(all_1)),
                        skip_reverse=True, cached_wt_esm3=cwt, quiet=True,
                    )
            if mode == 'esm_msr_wt':
                _, m_wt_full = measure_stable(_wt_only_full, max_repeats=3, fast_thresh_s=0.5)
                row.update(
                    single_capable=True,
                    single_screen_s=m_wt_full['wall_s'],
                    single_screen_method='measured_full',
                    single_batch_opt=1,
                    single_vram_opt_alloc_gb=m_wt_full['vram_alloc_peak_gb'],
                    double_capable=False,
                    double_screen_s=np.nan,
                    double_screen_method='additive_only',
                )
            elif mode in ('esm_msr_indep', 'esm_msr_marg'):
                # Masked state deduplication (Route 2 in models.py):
                #   Full 1-mut (19L variants): L unique WT masked states + L unique MT masked states = 2L forward passes
                #   Full 2-mut (361*L(L-1)/2):
                #     esm_msr_indep: L WT states + 19*L*(L-1) unique MT states (partner mutated, target masked)
                #     esm_msr_marg:  L WT states + L*(L-1)/2 unique MT states (both positions masked -> 361x dedup!)
                _, m_prep = measure_stable(
                    lambda: preprocess_structure(model, pdb_path, 'A', dev, backbone_mutation=None),
                    max_repeats=2, fast_thresh_s=0.3,
                )
                cands = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                from esm.utils.constants import esm3 as C_ESM3
                mask_tok = int(C_ESM3.SEQUENCE_MASK_TOKEN)

                def _run_states(b, active_model):
                    bt = seq_toks.expand(b, -1).clone()
                    pos_idx = (torch.arange(b, device=dev) % L) + 1
                    bt[torch.arange(b, device=dev), pos_idx] = mask_tok
                    with torch.no_grad(), _dtype_autocast(model, dev):
                        out = model._get_esm3_outputs(
                            bt,
                            coords.expand(b, -1, -1, -1) if coords is not None else None,
                            struct_toks.expand(b, -1) if struct_toks is not None else None,
                            plddt.expand(b, -1) if plddt is not None else None,
                            active_model=active_model,
                        )
                        _ = model._process_logits(out.sequence_logits.float())

                b_opt, _ = find_opt_batch(lambda b: _run_states(b, model.peft_mt), cands, vram_cap_frac=0.88, max_probe_s=5.0)
                tp_wt = timed_throughput(lambda b: _run_states(b, model.peft_wt), min(b_opt, L), L, max_wall_s=8.0, min_batches=2)
                tp_mt = timed_throughput(lambda b: _run_states(b, model.peft_mt), b_opt, max(L, b_opt * 2), max_wall_s=8.0, min_batches=2)
                t_wt_L = tp_wt['elapsed_s'] if tp_wt['measured_full'] else (L / tp_wt['rate_vps'])
                t_mt_L = L / tp_mt['rate_vps']
                s_total = m_prep['wall_s'] + t_wt_L + t_mt_L

                n_mt_states_2 = (19 * L * (L - 1)) if mode == 'esm_msr_indep' else (L * (L - 1) // 2)
                row.update(
                    single_capable=True,
                    single_screen_s=round(s_total, 4),
                    single_screen_method='timed_state_rate_opt_batch',
                    single_batch_opt=b_opt,
                    single_rate_vps=round(n_singles(L) / s_total, 4) if s_total > 0 else np.nan,
                    single_vram_opt_alloc_gb=max(tp_wt['vram_alloc_peak_gb'], tp_mt['vram_alloc_peak_gb']),
                    double_capable=True,
                    wt_pass_s=round(t_wt_L, 4),
                    extra_note=f'n_wt_states={L};n_mt_states_1={L};n_mt_states_2={n_mt_states_2};mt_state_rate={tp_mt["rate_vps"]}',
                )
                if double_impractical:
                    row.update(
                        double_screen_s=np.nan,
                        double_screen_method='impractical_gt_24h',
                        double_impractical=True,
                    )
                else:
                    d_total = m_prep['wall_s'] + t_wt_L + (n_mt_states_2 / tp_mt['rate_vps'])
                    hit_cap = bool(d_total >= args.double_cap_s)
                    row.update(
                        double_screen_s=round(d_total, 4),
                        double_screen_method='timed_state_rate_cap_reached' if hit_cap else 'timed_state_rate_opt_batch',
                        double_batch_opt=b_opt,
                        double_rate_vps=round(n_doubles(L) / d_total, 4) if d_total > 0 else np.nan,
                        double_vram_opt_alloc_gb=tp_mt['vram_alloc_peak_gb'],
                        double_impractical=hit_cap,
                    )
                    if hit_cap:
                        double_impractical = True
            else:
                # Dual-view: measure WT pass time separately, then optimal MT batch size & throughput for singles and doubles
                def _wt_pass_only():
                    with torch.no_grad(), _dtype_autocast(model, dev):
                        out = model._get_esm3_outputs(seq_toks, coords, struct_toks, plddt)
                        _ = model._process_logits(out.sequence_logits.float())
                _, m_wt = measure_stable(_wt_pass_only, max_repeats=3, fast_thresh_s=0.3)

                max_cand = min(512, max(1, n_singles(L)))
                cands = [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if b <= max_cand]
                # Pre-build sparse tensors for up to max(cands) single and double mutants
                pool_s = _prepare_sparse_batch(model, parse_muts_list(make_single_muts(wt_seq, limit=max(cands) * 2)), dev)
                pool_d = _prepare_sparse_batch(model, parse_muts_list(make_double_muts(wt_seq, max(cands) * 2)), dev)

                def _run_mt_single(b):
                    with torch.no_grad(), _dtype_autocast(model, dev):
                        _ = model.score_screening_batch(
                            seq_toks, pool_s['mut_pos'][:b], pool_s['wt_id'][:b], pool_s['mt_id'][:b], pool_s['mut_mask'][:b],
                            coords=coords, structure_tokens=struct_toks, plddt=plddt,
                            mask_strategy=None, batch_size=b, skip_reverse=False, cached_wt_esm3=cached_wt, quiet=True,
                        )

                def _run_mt_double(b):
                    with torch.no_grad(), _dtype_autocast(model, dev):
                        _ = model.score_screening_batch(
                            seq_toks, pool_d['mut_pos'][:b], pool_d['wt_id'][:b], pool_d['mt_id'][:b], pool_d['mut_mask'][:b],
                            coords=coords, structure_tokens=struct_toks, plddt=plddt,
                            mask_strategy=None, batch_size=b, skip_reverse=False, cached_wt_esm3=cached_wt, quiet=True,
                        )

                b_opt, _ = find_opt_batch(_run_mt_single, cands, vram_cap_frac=0.88, max_probe_s=5.0)
                tp_s = timed_throughput(_run_mt_single, b_opt, n_singles(L), max_wall_s=10.0, min_batches=2)
                s_total = m_wt['wall_s'] + (tp_s['elapsed_s'] if tp_s['measured_full'] else n_singles(L) / tp_s['rate_vps'])
                row.update(
                    single_capable=True,
                    single_screen_s=round(s_total, 4),
                    single_screen_method='measured_full' if tp_s['measured_full'] else 'timed_rate_opt_batch',
                    single_batch_opt=b_opt,
                    single_rate_vps=tp_s['rate_vps'],
                    single_vram_opt_alloc_gb=tp_s['vram_alloc_peak_gb'],
                    double_capable=True,
                    wt_pass_s=m_wt['wall_s'],
                )
                if double_impractical:
                    row.update(
                        double_screen_s=np.nan,
                        double_screen_method='impractical_gt_24h',
                        double_impractical=True,
                    )
                else:
                    tp_d = timed_throughput(_run_mt_double, b_opt, n_doubles(L), max_wall_s=8.0, min_batches=2)
                    d_total = m_wt['wall_s'] + n_doubles(L) / tp_d['rate_vps']
                    hit_cap = bool(d_total >= args.double_cap_s)
                    row.update(
                        double_screen_s=round(d_total, 4),
                        double_screen_method='timed_rate_opt_batch_cap_reached' if hit_cap else 'timed_rate_opt_batch',
                        double_batch_opt=b_opt,
                        double_rate_vps=tp_d['rate_vps'],
                        double_vram_opt_alloc_gb=tp_d['vram_alloc_peak_gb'],
                        double_impractical=hit_cap,
                    )
                    if hit_cap:
                        double_impractical = True
            # Clear local GPU refs
            del seq_toks, coords, plddt, struct_toks, cached_wt
            cleanup_cuda()
        except Exception as e:
            if is_oom(e):
                row.update(status='oom', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
            rows.append(row)
            _flush_row(args, row)
            cleanup_cuda()
            break
        rows.append(row)
        _flush_row(args, row)
    return rows


def bench_mutate_everything(panel: list[dict], mode: str, args) -> list[dict]:
    """mode in ('me_st1', 'me_st2')."""
    from functools import partial
    import paths as bpaths

    me_dir = str(bpaths.MUTATE_EVERYTHING_DIR)
    os.chdir(me_dir)
    sys.path.insert(0, me_dir)
    import misc
    from data import SeqDetDatatset, protein_collate_fn
    from engine_train import mem_inputs_to_device
    from main_train import get_args_parser
    from modeling.mutate_everything import MutateEverything

    ckpt = args.me_st1_ckpt if mode == 'me_st1' else args.me_st2_ckpt
    p = get_args_parser()
    p.set_defaults(backbone='af', finetune_backbone='models/finetuning_ptm_2.pt', batch_size=1, eval=True, disable_wandb=True)
    me_args = p.parse_args(['--resume', ckpt, '--device', 'cuda'])
    me_args.max_context_length = 2000  # upstream default; raised per-protein below if needed to test true OOM

    t0 = time.perf_counter()
    model = MutateEverything(me_args)
    alphabet = model.backbone.get_alphabet()
    model.to(me_args.device)
    misc.load_model(me_args, model, None, None)
    model.eval()
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    with tempfile.TemporaryDirectory(prefix=f'me_scale_{mode}_') as tmp:
        tmp_root = Path(tmp)
        for idx, item in enumerate(panel):
            L = item['seq_len']
            seq = load_seq(item['fasta_path'])
            row = _base_row(mode, item, load_s)
            # Raise max_context_length beyond L so we measure true GPU VRAM scaling rather than silent cropping at 2000
            me_args.max_context_length = max(2000, L + 16)
            sub_dir = tmp_root / item['name']
            (sub_dir / 'mutations').mkdir(parents=True, exist_ok=True)
            (sub_dir / 'fasta' / item['name']).mkdir(parents=True, exist_ok=True)
            (sub_dir / 'msa' / item['name']).mkdir(parents=True, exist_ok=True)
            # Symlink fasta & a3m in ME's expected directory layout: fasta/<pdb_id>/<pdb_id>.fasta, msa/<pdb_id>/*.a3m
            fa_dst = sub_dir / 'fasta' / item['name'] / f'{item["name"]}.fasta'
            if not fa_dst.exists():
                fa_dst.symlink_to(item['fasta_path'])
            a3m_dst = sub_dir / 'msa' / item['name'] / f'{item["name"]}.a3m'
            if not a3m_dst.exists():
                a3m_dst.symlink_to(item['a3m_path'])
            # Create minimal valid mutations CSV (with wt_seq and fr1/pos1/to1/fr2/pos2/to2 for doubles)
            mut_s = make_single_muts(seq, limit=1)[0]
            mut_rows = [{
                'pdb_id': item['name'], 'mut_info': mut_s, 'ddg': 0.0, 'wt_seq': seq,
                'fr1': mut_s[0], 'pos1': int(mut_s[1:-1]), 'to1': mut_s[-1],
                'fr2': np.nan, 'pos2': np.nan, 'to2': np.nan,
            }]
            if mode == 'me_st2':
                w1, w2 = (seq[0] if seq[0] in ALPHABET20 else 'A'), (seq[1] if seq[1] in ALPHABET20 else 'A')
                m1 = 'A' if w1 != 'A' else 'G'
                m2 = 'A' if w2 != 'A' else 'G'
                mut_rows.append({
                    'pdb_id': item['name'], 'mut_info': f'{w1}1{m1}:{w2}2{m2}', 'ddg': 0.0, 'wt_seq': seq,
                    'fr1': w1, 'pos1': 1, 'to1': m1, 'fr2': w2, 'pos2': 2, 'to2': m2,
                })
            csv_path = sub_dir / 'mutations' / f'{item["name"]}.csv'
            pd.DataFrame(mut_rows).to_csv(csv_path, index=False)

            try:
                t_load0 = time.perf_counter()
                ds = SeqDetDatatset(str(csv_path), me_args, train=False)
                collate_fn = partial(protein_collate_fn, alphabet=alphabet, args=me_args)
                dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)
                batch = next(iter(dl))
                featurize_s = round(time.perf_counter() - t_load0, 4)

                def _fwd():
                    with torch.no_grad():
                        x = mem_inputs_to_device(batch, torch.device('cuda'), me_args)
                        batch['known_mask1'] = batch['ddg_dense'] != 999
                        batch['known_mask2'] = batch['ddg_dense2'] != 999
                        batch['only_eval_single'] = (mode == 'me_st1')
                        batch['eval_list'] = False
                        pred_dict = model(x, batch)
                        # Sync & drop
                        _ = pred_dict['mut1_ddg'].shape
                        return pred_dict

                _, m = measure_stable(_fwd, max_repeats=2, fast_thresh_s=1.0, force_warmup=(idx == 0))
                row.update(
                    status='ok',
                    vram_weights_gb=m['vram_weights_gb'],
                    vram_min_alloc_gb=m['vram_alloc_peak_gb'],
                    vram_min_reserved_gb=m['vram_reserved_peak_gb'],
                    vram_min_device_gb=m['vram_device_peak_gb'],
                    wall_b1_s=m['wall_s'],
                    featurize_s=featurize_s,
                    single_capable=True,
                    single_screen_s=m['wall_s'],
                    single_screen_method='measured_full',
                    single_batch_opt=1,
                    single_vram_opt_alloc_gb=m['vram_alloc_peak_gb'],
                    double_capable=(mode == 'me_st2'),
                    double_screen_s=m['wall_s'] if mode == 'me_st2' else np.nan,
                    double_screen_method='measured_full' if mode == 'me_st2' else 'additive_only',
                    double_batch_opt=1 if mode == 'me_st2' else np.nan,
                    double_vram_opt_alloc_gb=m['vram_alloc_peak_gb'] if mode == 'me_st2' else np.nan,
                )
                del batch, ds, dl
                cleanup_cuda()
            except Exception as e:
                if is_oom(e):
                    row.update(status='oom', error_msg=str(e)[:240])
                    rows.append(row)
                    _flush_row(args, row)
                    cleanup_cuda()
                    break
                row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            rows.append(row)
            _flush_row(args, row)
    return rows


def bench_thermompnn(panel: list[dict], mode: str, args) -> list[dict]:
    """mode in ('thermompnn_single', 'thermompnn_d') using upstream v2_ssm.py fast SSM paths."""
    import paths as bpaths
    from omegaconf import OmegaConf

    tmpnn = str(bpaths.THERMOMPNN_DIR)
    sys.path.insert(0, tmpnn)
    os.chdir(os.path.join(tmpnn, 'thermompnn'))
    from thermompnn.datasets.dataset_utils import Mutation
    from thermompnn.datasets.v2_datasets import tied_featurize_mut
    from thermompnn.model.v2_model import _dist, batched_index_select
    from thermompnn.protein_mpnn_utils import parse_PDB
    from thermompnn.train_thermompnn import parse_cfg
    from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese

    is_d = (mode == 'thermompnn_d')
    cfg_name = 'train_epistatic.yaml' if is_d else 'train_single.yaml'
    ckpt = args.thermompnn_d_ckpt if is_d else args.thermompnn_single_ckpt
    cfg = OmegaConf.merge(
        OmegaConf.create({'platform': {'accel': 'gpu', 'thermompnn_dir': tmpnn}, 'data_loc': {'rosetta_data': '/tmp/'}}),
        OmegaConf.load(os.path.join(tmpnn, 'examples/configs', cfg_name)),
    )
    cfg = parse_cfg(cfg)

    t0 = time.perf_counter()
    cls = TransferModelPLv2Siamese if is_d else TransferModelPLv2
    model_pl = cls.load_from_checkpoint(ckpt, cfg=cfg, map_location='cpu')
    model = model_pl.model.cuda().eval()
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    double_impractical = False
    for idx, item in enumerate(panel):
        L = item['seq_len']
        row = _base_row(mode, item, load_s)
        try:
            pdb = parse_PDB(item['pdb_path'], side_chains=False)[0]
            pdb['mutation'] = Mutation([0], ['A'], ['A'], [0.0], '')
            batch = tied_featurize_mut([pdb])
            X, S, mask, lengths, chain_M, chain_enc, res_idx, _, _, _, mut_ddGs, _ = batch
            X = torch.nan_to_num(X.cuda(), nan=0.0)
            S, mask, chain_M, res_idx, chain_enc = S.cuda(), mask.cuda(), chain_M.cuda(), res_idx.cuda(), chain_enc.cuda()

            if not is_d:
                # Upstream v2_ssm.run_single_ssm: 1 pass gives full [L, 21] single-mutant matrix
                def _single_ssm():
                    with torch.no_grad():
                        all_hid, mpnn_emb, _, _ = model.prot_mpnn(X, S, mask, chain_M, res_idx, chain_enc)
                        all_hid = torch.cat(all_hid[:cfg.model.num_final_layers], -1)
                        all_hid = torch.squeeze(torch.cat([all_hid, mpnn_emb], -1), 0)
                        all_hid = model.light_attention(torch.unsqueeze(all_hid, -1))
                        ddg = model.ddg_out(all_hid)
                        s_sq = torch.squeeze(S)
                        wt_ddg = batched_index_select(ddg, dim=-1, index=s_sq)
                        return ddg - wt_ddg.expand(-1, 21)

                _, m = measure_stable(_single_ssm, max_repeats=3, fast_thresh_s=0.5, force_warmup=(idx == 0))
                row.update(
                    status='ok',
                    vram_weights_gb=m['vram_weights_gb'],
                    vram_min_alloc_gb=m['vram_alloc_peak_gb'],
                    vram_min_reserved_gb=m['vram_reserved_peak_gb'],
                    vram_min_device_gb=m['vram_device_peak_gb'],
                    wall_b1_s=m['wall_s'],
                    single_capable=True,
                    single_screen_s=m['wall_s'],
                    single_screen_method='measured_full',
                    single_batch_opt=1,
                    single_vram_opt_alloc_gb=m['vram_alloc_peak_gb'],
                    double_capable=False,
                    double_screen_s=np.nan,
                    double_screen_method='additive_only',
                )
            else:
                # Upstream v2_ssm.run_epistatic_ssm: 1 WT prot_mpnn pass + batched Siamese MLP head (run_double)
                def _backbone():
                    with torch.no_grad():
                        all_hid, mpnn_emb, _, mpnn_edges = model.prot_mpnn(X, S, mask, chain_M, res_idx, chain_enc)
                        all_hid = torch.cat(all_hid[:cfg.model.num_final_layers], -1)
                        _, E_idx = _dist(X[:, :, 1, :], mask)
                        return all_hid, mpnn_emb, mpnn_edges, E_idx

                def _double_head_chunk(all_hid, mpnn_emb, mpnn_edges_raw, E_idx, pos, mutAA):
                    with torch.no_grad():
                        B = mutAA.shape[0]
                        hid_b = all_hid.expand(B, -1, -1)
                        emb_b = mpnn_emb.expand(B, -1, -1)
                        # Upstream v2_ssm.py calls .repeat(B, ...); we mirror its exact gather + Siamese MLP
                        # (using expand where read-only, avoiding pointless memory copy if gather works, wait:
                        # let's mirror upstream's exact gather ops cleanly!)
                        edges_b = mpnn_edges_raw.expand(B, -1, -1, -1)
                        eidx_b = E_idx.expand(B, -1, -1)

                        mut_embed = torch.stack([model.prot_mpnn.W_s(mutAA[:, m]) for m in range(2)], dim=-1)
                        edges = []
                        for n_cur in (0, 1):
                            mp_edges_tmp = torch.squeeze(batched_index_select(edges_b, 1, pos[:, n_cur:n_cur + 1]), 1)
                            eidx_tmp = torch.squeeze(batched_index_select(eidx_b, 1, pos[:, n_cur:n_cur + 1]), 1)
                            n_oth = 1 - n_cur
                            mp_other = pos[:, n_oth:n_oth + 1].unsqueeze(-1).expand(-1, 1, eidx_tmp.shape[-1])
                            a, _, c = torch.where(eidx_tmp[:, None, :] == mp_other)
                            edge = torch.zeros((B, mp_edges_tmp.shape[-1]), device=E_idx.device, dtype=mp_edges_tmp.dtype)
                            edge[a, :] = mp_edges_tmp[a, c, :]
                            edges.append(edge)
                        mpnn_edges = torch.stack(edges, dim=-1)

                        final_embed = []
                        for i in range(2):
                            cur_pos = pos[:, i:i + 1]
                            g_struct = torch.gather(hid_b, 1, cur_pos.unsqueeze(-1).expand(B, 1, hid_b.size(2))).squeeze(1)
                            g_seq = torch.gather(emb_b, 1, cur_pos.unsqueeze(-1).expand(B, 1, emb_b.size(2))).squeeze(1)
                            if cfg.model.mutant_embedding:
                                g_seq = g_seq - mut_embed[:, :, i]
                            g_emb = torch.cat([g_struct, g_seq], -1)
                            if cfg.model.edges:
                                g_emb = torch.cat([g_emb, mpnn_edges[:, :, i]], -1)
                            final_embed.append(g_emb)
                        final_embed = model.light_attention(torch.stack(final_embed, dim=0))
                        embedAB = torch.cat((final_embed[0], final_embed[1]), dim=-1)
                        embedBA = torch.cat((final_embed[1], final_embed[0]), dim=-1)
                        return (model.ddg_out(embedAB) + model.ddg_out(embedBA)) / 2.0

                # B=1 minimum VRAM across backbone + 1 double mutation
                pos_1 = torch.tensor([[0, min(1, L - 1)]], dtype=torch.long, device='cuda')
                mut_1 = torch.tensor([[1, 2]], dtype=torch.long, device='cuda')

                def _b1_full():
                    bb = _backbone()
                    _ = _double_head_chunk(*bb, pos_1, mut_1)
                    return bb

                bb, m_b1 = measure_stable(_b1_full, max_repeats=3, fast_thresh_s=0.4, force_warmup=(idx == 0))
                _, m_bb = measure_stable(_backbone, max_repeats=3, fast_thresh_s=0.4)

                row.update(
                    status='ok',
                    vram_weights_gb=m_b1['vram_weights_gb'],
                    vram_min_alloc_gb=m_b1['vram_alloc_peak_gb'],
                    vram_min_reserved_gb=m_b1['vram_reserved_peak_gb'],
                    vram_min_device_gb=m_b1['vram_device_peak_gb'],
                    wall_b1_s=m_b1['wall_s'],
                    wt_pass_s=m_bb['wall_s'],
                    single_capable=False,
                    single_screen_s=np.nan,
                    single_screen_method='use_thermompnn_single',
                    double_capable=True,
                )
                if double_impractical:
                    row.update(
                        double_screen_s=np.nan,
                        double_screen_method='impractical_gt_24h',
                        double_impractical=True,
                    )
                else:
                    # Pre-allocate a pool of synthetic double indices on GPU for head throughput measurement
                    cands = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]
                    max_b = max(cands)
                    rng = np.random.default_rng(42)
                    p1 = rng.integers(0, L - 1, size=max_b)
                    p2 = np.minimum(L - 1, p1 + rng.integers(1, max(2, L // 2), size=max_b))
                    pos_pool = torch.from_numpy(np.stack([p1, p2], axis=1)).long().cuda()
                    mut_pool = torch.from_numpy(rng.integers(0, 20, size=(max_b, 2))).long().cuda()

                    b_opt, _ = find_opt_batch(
                        lambda b: _double_head_chunk(*bb, pos_pool[:b], mut_pool[:b]),
                        cands, vram_cap_frac=0.85, max_probe_s=4.0,
                    )
                    tp_d = timed_throughput(
                        lambda b: _double_head_chunk(*bb, pos_pool[:b], mut_pool[:b]),
                        b_opt, n_doubles(L), max_wall_s=6.0, min_batches=3,
                    )
                    d_total = m_bb['wall_s'] + (tp_d['elapsed_s'] if tp_d['measured_full'] else n_doubles(L) / tp_d['rate_vps'])
                    hit_cap = bool(d_total >= args.double_cap_s)
                    base_m = 'measured_full' if tp_d['measured_full'] else 'timed_rate_opt_batch'
                    row.update(
                        double_screen_s=round(d_total, 4),
                        double_screen_method=f'{base_m}_cap_reached' if hit_cap else base_m,
                        double_batch_opt=b_opt,
                        double_rate_vps=tp_d['rate_vps'],
                        double_vram_opt_alloc_gb=tp_d['vram_alloc_peak_gb'],
                        double_impractical=hit_cap,
                    )
                    if hit_cap:
                        double_impractical = True
                    del pos_pool, mut_pool
                del bb
            del X, S, mask, chain_M, res_idx, chain_enc
            cleanup_cuda()
        except Exception as e:
            if is_oom(e):
                row.update(status='oom', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
            rows.append(row)
            _flush_row(args, row)
            cleanup_cuda()
            break
        rows.append(row)
        _flush_row(args, row)
    return rows


def bench_spurs(panel: list[dict], mode: str, args) -> list[dict]:
    """mode in ('spurs_single', 'spurs_multi'), each loaded ALONE to measure true per-model VRAM."""
    # Reuse spurs_inference's exact import mocks & loaders
    import spurs_inference as si

    which = 'single' if mode == 'spurs_single' else 'multi'
    t0 = time.perf_counter()
    m_single, m_multi, cfg_single, _ = si.load_spurs_models(
        args.spurs_single_ckpt, args.spurs_multi_ckpt, models=which
    )
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    double_impractical = False
    for idx, item in enumerate(panel):
        L = item['seq_len']
        seq = load_seq(item['fasta_path'])
        row = _base_row(mode, item, load_s)
        try:
            pdb_data = si.parse_pdb(item['pdb_path'], item['name'], 'A', cfg_single, 'cuda')
            if mode == 'spurs_single':
                def _run_s():
                    with torch.no_grad():
                        return m_single(pdb_data, return_logist=True).float().cpu()

                _, m = measure_stable(_run_s, max_repeats=3, fast_thresh_s=0.5, force_warmup=(idx == 0))
                row.update(
                    status='ok',
                    vram_weights_gb=m['vram_weights_gb'],
                    vram_min_alloc_gb=m['vram_alloc_peak_gb'],
                    vram_min_reserved_gb=m['vram_reserved_peak_gb'],
                    vram_min_device_gb=m['vram_device_peak_gb'],
                    wall_b1_s=m['wall_s'],
                    single_capable=True,
                    single_screen_s=m['wall_s'],
                    single_screen_method='measured_full',
                    single_batch_opt=1,
                    single_vram_opt_alloc_gb=m['vram_alloc_peak_gb'],
                    double_capable=False,
                    double_screen_s=np.nan,
                    double_screen_method='additive_only',
                )
            else:
                # SPURS-multi: B=1 minimum VRAM with 1 double mutation, then opt chunk size for singles & doubles
                d1 = [make_double_muts(seq, 1)[0].split(':')]
                mid1, app1 = si.parse_pdb_for_mutation(d1)
                app1 = app1.cuda()

                def _b1():
                    g = pdb_data.copy()
                    g['mut_ids'] = mid1
                    g['append_tensors'] = app1
                    with torch.no_grad():
                        return m_multi(g).view(-1).float().cpu()

                _, m_b1 = measure_stable(_b1, max_repeats=3, fast_thresh_s=0.5, force_warmup=(idx == 0))

                cands = [1, 16, 256, 1024, 4096, 16384, 65536]
                max_c = max(cands)
                # Build pools of parsed mutation tensors up to max_c
                s_all = [[m] for m in make_single_muts(seq, limit=min(n_singles(L), max_c))]
                d_pool = [m.split(':') for m in make_double_muts(seq, max_c)]

                def _run_chunk_d(b):
                    mid, app = si.parse_pdb_for_mutation(d_pool[:b])
                    g = pdb_data.copy()
                    g['mut_ids'] = mid
                    g['append_tensors'] = app.cuda()
                    with torch.no_grad():
                        _ = m_multi(g).view(-1).float().cpu()

                def _run_chunk_s(b):
                    mid, app = si.parse_pdb_for_mutation(s_all[:b])
                    g = pdb_data.copy()
                    g['mut_ids'] = mid
                    g['append_tensors'] = app.cuda()
                    with torch.no_grad():
                        _ = m_multi(g).view(-1).float().cpu()

                b_opt, _ = find_opt_batch(_run_chunk_d, cands, vram_cap_frac=0.85, max_probe_s=5.0)
                tp_s = timed_throughput(_run_chunk_s, min(b_opt, len(s_all)), n_singles(L), max_wall_s=8.0, min_batches=2)
                s_total = tp_s['elapsed_s'] if tp_s['measured_full'] else n_singles(L) / tp_s['rate_vps']
                row.update(
                    status='ok',
                    vram_weights_gb=m_b1['vram_weights_gb'],
                    vram_min_alloc_gb=m_b1['vram_alloc_peak_gb'],
                    vram_min_reserved_gb=m_b1['vram_reserved_peak_gb'],
                    vram_min_device_gb=m_b1['vram_device_peak_gb'],
                    wall_b1_s=m_b1['wall_s'],
                    single_capable=True,
                    single_screen_s=round(s_total, 4),
                    single_screen_method='measured_full' if tp_s['measured_full'] else 'timed_rate_opt_batch',
                    single_batch_opt=min(b_opt, n_singles(L)),
                    single_rate_vps=tp_s['rate_vps'],
                    single_vram_opt_alloc_gb=tp_s['vram_alloc_peak_gb'],
                    double_capable=True,
                )
                if double_impractical:
                    row.update(
                        double_screen_s=np.nan,
                        double_screen_method='impractical_gt_24h',
                        double_impractical=True,
                    )
                else:
                    tp_d = timed_throughput(_run_chunk_d, b_opt, n_doubles(L), max_wall_s=8.0, min_batches=2)
                    d_total = tp_d['elapsed_s'] if tp_d['measured_full'] else n_doubles(L) / tp_d['rate_vps']
                    hit_cap = bool(d_total >= args.double_cap_s)
                    base_m = 'measured_full' if tp_d['measured_full'] else 'timed_rate_opt_batch'
                    row.update(
                        double_screen_s=round(d_total, 4),
                        double_screen_method=f'{base_m}_cap_reached' if hit_cap else base_m,
                        double_batch_opt=b_opt,
                        double_rate_vps=tp_d['rate_vps'],
                        double_vram_opt_alloc_gb=tp_d['vram_alloc_peak_gb'],
                        double_impractical=hit_cap,
                    )
                    if hit_cap:
                        double_impractical = True
            del pdb_data
            cleanup_cuda()
        except Exception as e:
            if is_oom(e):
                row.update(status='oom', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
            rows.append(row)
            _flush_row(args, row)
            cleanup_cuda()
            break
        rows.append(row)
        _flush_row(args, row)
    return rows


def bench_proteinmpnn(panel: list[dict], args) -> list[dict]:
    """ProteinMPNN zero-shot (score_unmasked_fast): 1 encode + L decoder rows (1-mut) / L + 19L(L-1) rows (2-mut)."""
    import proteinmpnn_inference as pmi
    import paths as bpaths

    dev = torch.device('cuda')
    t0 = time.perf_counter()
    ckpt_path = Path(bpaths.PROTEINMPNN_DIR) / 'vanilla_model_weights' / 'v_48_020.pt'
    checkpoint = torch.load(ckpt_path, map_location=dev)
    model = pmi.ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0,
        k_neighbors=checkpoint['num_edges'],
    ).to(dev)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    double_impractical = False
    for idx, item in enumerate(panel):
        L = item['seq_len']
        row = _base_row('proteinmpnn', item, load_s)
        try:
            X, S_wt, mask, chain_M, residue_idx, chain_encoding_all = pmi.featurize_structure(
                item['name'], item['pdb_path'], 'A', dev
            )
            wt_seq = S_wt[0]
            gen = torch.Generator(device=dev).manual_seed(0)
            noise = torch.randn((L, L), generator=gen, device=dev)

            def _decode_rows(enc, n_rows, with_bg_mut=False):
                p_t = torch.arange(n_rows, device=dev) % L
                vals = noise[p_t].clone()
                vals[torch.arange(n_rows, device=dev), p_t] = 1000.0
                S_chunk = wt_seq.unsqueeze(0).repeat(n_rows, 1)
                if with_bg_mut:
                    q_t = (p_t + 1) % L
                    S_chunk[torch.arange(n_rows, device=dev), q_t] = (wt_seq[q_t] + 1) % 20
                with torch.no_grad():
                    return pmi.mpnn_decode_at(model, enc, S_chunk, vals, p_t).float().cpu()

            def _b1():
                with torch.no_grad():
                    enc = pmi.mpnn_encode(model, X, mask, residue_idx, chain_encoding_all)
                _ = _decode_rows(enc, 1, with_bg_mut=True)
                return enc

            enc, m_b1 = measure_stable(_b1, max_repeats=3, fast_thresh_s=0.4, force_warmup=(idx == 0))
            _, m_enc = measure_stable(lambda: pmi.mpnn_encode(model, X, mask, residue_idx, chain_encoding_all), max_repeats=3, fast_thresh_s=0.3)

            cands = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            b_opt, _ = find_opt_batch(lambda b: _decode_rows(enc, b, with_bg_mut=True), cands, vram_cap_frac=0.85, max_probe_s=4.0)

            # Full 1-mut screen = exactly L unique decoder rows (all 19 muts per pos decoded simultaneously)
            tp_1 = timed_throughput(lambda b: _decode_rows(enc, b, with_bg_mut=False), min(b_opt, L), L, max_wall_s=10.0, min_batches=1)
            s_total = m_enc['wall_s'] + (tp_1['elapsed_s'] if tp_1['measured_full'] else L / tp_1['rate_vps'])

            row.update(
                status='ok',
                vram_weights_gb=m_b1['vram_weights_gb'],
                vram_min_alloc_gb=m_b1['vram_alloc_peak_gb'],
                vram_min_reserved_gb=m_b1['vram_reserved_peak_gb'],
                vram_min_device_gb=m_b1['vram_device_peak_gb'],
                wall_b1_s=m_b1['wall_s'],
                wt_pass_s=m_enc['wall_s'],
                single_capable=True,
                single_screen_s=round(s_total, 4),
                single_screen_method='measured_full' if tp_1['measured_full'] else 'timed_rate_opt_batch',
                single_batch_opt=min(b_opt, L),
                single_rate_vps=round(n_singles(L) / s_total, 4) if s_total > 0 else np.nan,
                single_vram_opt_alloc_gb=tp_1['vram_alloc_peak_gb'],
                double_capable=True,
            )
            # Full 2-mut screen = L + 19 * L * (L - 1) unique decoder rows (score_unmasked_fast deduplication)
            n_dec_rows_2 = L + 19 * L * (L - 1)
            if double_impractical:
                row.update(
                    double_screen_s=np.nan,
                    double_screen_method='impractical_gt_24h',
                    double_impractical=True,
                    extra_note=f'n_dec_rows_1={L};n_dec_rows_2={n_dec_rows_2};impractical_gt_24h',
                )
            else:
                tp_2 = timed_throughput(lambda b: _decode_rows(enc, b, with_bg_mut=True), b_opt, n_dec_rows_2, max_wall_s=8.0, min_batches=2)
                d_total = m_enc['wall_s'] + (tp_2['elapsed_s'] if tp_2['measured_full'] else n_dec_rows_2 / tp_2['rate_vps'])
                hit_cap = bool(d_total >= args.double_cap_s)
                base_m = 'measured_full' if tp_2['measured_full'] else 'timed_rate_opt_batch'
                row.update(
                    double_screen_s=round(d_total, 4),
                    double_screen_method=f'{base_m}_cap_reached' if hit_cap else base_m,
                    double_batch_opt=b_opt,
                    double_rate_vps=round(n_doubles(L) / d_total, 4) if d_total > 0 else np.nan,
                    double_vram_opt_alloc_gb=tp_2['vram_alloc_peak_gb'],
                    double_impractical=hit_cap,
                    extra_note=f'n_dec_rows_1={L};n_dec_rows_2={n_dec_rows_2};dec_row_rate={tp_2["rate_vps"]}',
                )
                if hit_cap:
                    double_impractical = True
            del X, S_wt, mask, chain_M, residue_idx, chain_encoding_all, enc, noise
            cleanup_cuda()
        except Exception as e:
            if is_oom(e):
                row.update(status='oom', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
            rows.append(row)
            _flush_row(args, row)
            cleanup_cuda()
            break
        rows.append(row)
        _flush_row(args, row)
    return rows


def bench_prosst(panel: list[dict], args) -> list[dict]:
    """ProSST-2048 zero-shot: SST structure quantization + 1 WT forward pass (full 19L singles, additive-only)."""
    sys.path.insert(0, args.prosst_dir)
    import prosst.structure.get_sst_seq as gss
    from prosst.structure.get_sst_seq import SSTPredictor
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # Avoid fork deadlock in SSTPredictor
    def seq_map(func, data, workers=1):
        for x in data:
            yield func(x)
    gss.iter_parallel_map = seq_map

    dev = 'cuda'
    t0 = time.perf_counter()
    predictor = SSTPredictor(structure_vocab_size=2048, num_processes=1, num_threads=4, device=dev)
    model = AutoModelForMaskedLM.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained('AI4Protein/ProSST-2048', trust_remote_code=True)
    load_s = round(time.perf_counter() - t0, 3)

    rows = []
    double_impractical = False
    for idx, item in enumerate(panel):
        L = item['seq_len']
        row = _base_row('prosst', item, load_s)
        try:
            def _full_pass():
                t_s0 = time.perf_counter()
                res = predictor.predict_from_pdb([item['pdb_path']])[0]
                aa_seq, st = res['aa_seq'], res['2048_sst_seq']
                torch.cuda.synchronize()
                t_sst = time.perf_counter() - t_s0
                enc = tok([aa_seq], return_tensors='pt')
                ss_ids = torch.tensor([[1, *[i + 3 for i in st], 2]], dtype=torch.long, device=dev)
                with torch.no_grad():
                    lg = model(
                        input_ids=enc['input_ids'].to(dev),
                        attention_mask=enc['attention_mask'].to(dev),
                        ss_input_ids=ss_ids,
                    ).logits
                    _ = torch.log_softmax(lg[:, 1:-1, :], dim=-1)[0].float().cpu()
                return t_sst

            t_sst, m = measure_stable(_full_pass, max_repeats=2, fast_thresh_s=0.8, force_warmup=(idx == 0))
            row.update(
                status='ok',
                vram_weights_gb=m['vram_weights_gb'],
                vram_min_alloc_gb=m['vram_alloc_peak_gb'],
                vram_min_reserved_gb=m['vram_reserved_peak_gb'],
                vram_min_device_gb=m['vram_device_peak_gb'],
                wall_b1_s=m['wall_s'],
                featurize_s=round(t_sst, 4),
                single_capable=True,
                single_screen_s=m['wall_s'],
                single_screen_method='measured_full',
                single_batch_opt=1,
                single_vram_opt_alloc_gb=m['vram_alloc_peak_gb'],
                double_capable=False,
                double_screen_s=np.nan,
                double_screen_method='additive_only',
            )
            cleanup_cuda()
        except Exception as e:
            if is_oom(e):
                row.update(status='oom', error_msg=str(e)[:240])
                rows.append(row)
                _flush_row(args, row)
                cleanup_cuda()
                break
            row.update(status=f'error:{type(e).__name__}', error_msg=str(e)[:240])
            rows.append(row)
            _flush_row(args, row)
            cleanup_cuda()
            break
        rows.append(row)
        _flush_row(args, row)
    return rows


def _base_row(model: str, item: dict, load_s: float) -> dict:
    L = int(item['seq_len'])
    return dict(
        model=model,
        bin=int(item['bin']),
        name=item['name'],
        seq_len=L,
        source=item['source'],
        msa_depth=int(item.get('msa_depth', 0)),
        n_singles_full=n_singles(L),
        n_doubles_full=n_doubles(L),
        model_load_s=load_s,
        status='pending',
    )


def _flush_row(args, row: dict) -> None:
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'results.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')
    print(
        f"[{args.model}] L={row['seq_len']:5d} ({row['name']:18s}) status={row['status']:8s} "
        f"vram_min={row.get('vram_min_alloc_gb')} GB (dev={row.get('vram_min_device_gb')}) "
        f"1mut={row.get('single_screen_s')} s  2mut={row.get('double_screen_s')} s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=[
        'esm_msr_wt', 'esm_msr_dual', 'esm_msr_indep', 'esm_msr_marg', 'me_st1', 'me_st2',
        'thermompnn_single', 'thermompnn_d', 'spurs_single', 'spurs_multi',
        'proteinmpnn', 'prosst',
    ])
    ap.add_argument('--panel', required=True, help='Path to panel.json from size_scaling_panel.py')
    ap.add_argument('--out', required=True, help='Root output directory (writes <out>/<model>/*)')
    ap.add_argument('--max_len', type=int, default=0, help='Optional upper length cap for smoke tests')
    ap.add_argument('--double_cap_s', type=float, default=86400.0,
                    help='Practicality cap for full double-mutant screening (default 86400 s = 24 h); once reached, larger proteins are labelled impractical.')
    ap.add_argument('--esm_msr_ckpt', default='/home/sreeves/esm-msr-devel/runs_final_ablations/FINAL_p0_s1/0/epoch=06-val_unmasked_rho_combined_avg=0.8558.ckpt')
    ap.add_argument('--esm_msr_indep_ckpt', default='/home/sreeves/esm-msr-devel/runs_final_ablations/FINAL_p1_s1/0/epoch=06-val_rho_combined_avg=0.8606.ckpt')
    ap.add_argument('--esm_msr_marg_ckpt', default='/home/sreeves/esm-msr-devel/runs_final_ablations/FINAL_p1_marg_s1/0/epoch=07-val_rho_combined_avg=0.8575.ckpt')
    ap.add_argument('--me_st1_ckpt', default='/home/sreeves/software/MutateEverything-oct06/logs/af_single_oct06_capped_seed1/checkpoint-19.pth')
    ap.add_argument('--me_st2_ckpt', default='/home/sreeves/software/MutateEverything-oct06/logs/af_double_oct06_capped_allbl_seed1/checkpoint-99.pth')
    ap.add_argument('--thermompnn_single_ckpt', default='/home/sreeves/esm-msr-devel/outputs/baselines_capped/thermompnn/single_seed1/single_oct06_seed1_epoch=92_val_ddG_spearman=0.76.ckpt')
    ap.add_argument('--thermompnn_d_ckpt', default='/home/sreeves/esm-msr-devel/outputs/baselines_corrected/thermompnn/epistatic_seed1/epistatic_oct06_seed1_epoch=90_val_ddG_spearman=0.78.ckpt')
    ap.add_argument('--spurs_single_ckpt', default='oct06_leg_seed1.ckpt')
    ap.add_argument('--spurs_multi_ckpt', default='oct06_legmulti_seed1.ckpt')
    ap.add_argument('--prosst_dir', default=os.path.expanduser('~/software/ProSST'))
    args = ap.parse_args()

    # Check fixed-slot resource limits if set via sbatch_fixed_slot.sh
    limits = resource_tracker.Limits.from_env()
    limit_check = limits.check() if limits.any() else None
    if limit_check is not None and limits.strict and not limit_check['ok']:
        raise RuntimeError('Fixed-slot check failed: ' + '; '.join(limit_check['problems']))

    panel = json.loads(Path(args.panel).read_text())
    if args.max_len > 0:
        panel = [p for p in panel if p['seq_len'] <= args.max_len]

    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'results.jsonl'
    if jsonl_path.exists():
        jsonl_path.unlink()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    meta = dict(
        model=args.model,
        started=datetime.datetime.now().isoformat(timespec='seconds'),
        host=socket.gethostname(),
        gpu=dict(name=props.name, total_gb=round(props.total_memory / GB, 2), capability=f'{props.major}.{props.minor}'),
        slurm={k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE')},
        limits_check=limit_check,
        n_panel=len(panel),
    )
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')

    if args.model in ('esm_msr_wt', 'esm_msr_dual', 'esm_msr_indep', 'esm_msr_marg'):
        rows = bench_esm_msr(panel, args.model, args)
    elif args.model in ('me_st1', 'me_st2'):
        rows = bench_mutate_everything(panel, args.model, args)
    elif args.model in ('thermompnn_single', 'thermompnn_d'):
        rows = bench_thermompnn(panel, args.model, args)
    elif args.model in ('spurs_single', 'spurs_multi'):
        rows = bench_spurs(panel, args.model, args)
    elif args.model == 'proteinmpnn':
        rows = bench_proteinmpnn(panel, args)
    elif args.model == 'prosst':
        rows = bench_prosst(panel, args)
    else:
        raise ValueError(args.model)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'results.csv', index=False)
    meta['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
    meta['n_recorded'] = len(rows)
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Done: {out_dir / "results.csv"} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
