"""Reusable ProSST zero-shot scoring primitives supporting unmasked and independent-masking dual-view epistasis.

Upstream ProSST (Li et al., NeurIPS 2024; zero_shot/proteingym_benchmark.py) scores variants from a single unmasked
wild-type forward pass (`wt_unmasked`), forcing all multi-mutants to be strictly additive (epistasis == 0).

Because `ProSSTForMaskedLM` was pre-trained with MLM (`mask_token_id = 24`, `token_dropout = True`) and its structure
quantizer (`prosst/structure/build_graph.py`) uses only sequence-agnostic backbone geometry (N, CA, C, O; zeroed
residue node features), ProSST natively supports fixed-backbone independent masking on both WT and mutant contexts:

  - `wt_unmasked`:      1 unmasked WT pass; additive sum of log p(mt_k | x_WT, s) - log p(wt_k | x_WT, s).
  - `dual_unmasked`:    unmasked WT pass + unmasked MT pass (on x_MT); symmetric average and epistasis.
  - `wt_independent`:   site-wise masked WT pass (each mutated site masked individually in x_WT; <= L passes).
  - `dual_independent`: site-wise masked WT pass + site-wise masked MT pass (each mutated site masked individually
                        in x_MT while all other mutated sites remain substituted). For doubles AB, the symmetric
                        combined score equals the average of both thermodynamic paths (WT->A->AB and WT->B->AB),
                        singles satisfy mt_pred == wt_pred identically, and epistasis is 0.5 * (mt_pred - wt_pred).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

VALID_MODES = ('wt_unmasked', 'dual_unmasked', 'wt_independent', 'dual_independent')


@dataclass
class ProSSTScoreResult:
    wt_pred: np.ndarray
    mt_pred: np.ndarray
    comb_pred: np.ndarray
    epi_pred: np.ndarray
    n_wt_passes: int
    n_mt_passes: int
    elapsed_s: float


def structure_tokens(sst_seq: Sequence[int]) -> torch.Tensor:
    """Convert raw ProSST cluster IDs [0..K-1] to model `ss_input_ids` [1, i+3..., 2]."""
    return torch.tensor([[1, *[int(i) + 3 for i in sst_seq], 2]], dtype=torch.long)


def parse_mutation(mut_str: str) -> List[Tuple[str, int, str]]:
    """Parse colon-separated 1-based mutations (e.g. 'A12G:L40V') -> list of (wt, 0-based pos, mt)."""
    out: List[Tuple[str, int, str]] = []
    for sub in str(mut_str).strip().split(':'):
        sub = sub.strip()
        if not sub:
            continue
        if len(sub) < 3:
            raise ValueError(f'Malformed mutation token: {sub!r} in {mut_str!r}')
        wt, pos_str, mt = sub[0], sub[1:-1], sub[-1]
        pos = int(pos_str) - 1
        out.append((wt, pos, mt))
    if not out:
        raise ValueError(f'Empty mutation string: {mut_str!r}')
    return out


def _resolve_device(model: torch.nn.Module, device: Optional[str | torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _effective_batch_size(seq_len: int, batch_size: int, max_tokens: int) -> int:
    if max_tokens <= 0:
        return max(1, int(batch_size))
    return max(1, min(int(batch_size), max(1, int(max_tokens) // max(1, seq_len + 2))))


@torch.no_grad()
def _eval_unique_states(
    model: torch.nn.Module,
    ss_ids_1: torch.Tensor,
    unique_inputs: List[np.ndarray],
    needed_positions: List[List[int]],
    batch_size: int,
    max_tokens: int,
    device: torch.device,
) -> List[Dict[int, np.ndarray]]:
    """Run batched forward passes over unique token arrays and retain log-probs only at `needed_positions`."""
    n = len(unique_inputs)
    if n == 0:
        return []
    seq_len_with_special = int(unique_inputs[0].shape[0])
    eff_bs = _effective_batch_size(seq_len_with_special - 2, batch_size, max_tokens)
    results: List[Dict[int, np.ndarray]] = [{} for _ in range(n)]
    ss_dev = ss_ids_1.to(device)

    for start in range(0, n, eff_bs):
        end = min(start + eff_bs, n)
        chunk_np = np.stack(unique_inputs[start:end], axis=0)
        input_ids = torch.from_numpy(chunk_np).to(device=device, dtype=torch.long)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long)
        ss_batch = ss_dev.expand(end - start, -1)

        logits = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            ss_input_ids=ss_batch,
        ).logits
        # Strip <cls> and <eos>; shape [B, L, V]
        log_probs = torch.log_softmax(logits[:, 1:-1, :].float(), dim=-1)

        for local_i in range(end - start):
            global_i = start + local_i
            pos_list = needed_positions[global_i]
            if not pos_list:
                continue
            pos_t = torch.tensor(pos_list, dtype=torch.long, device=device)
            gathered = log_probs[local_i, pos_t, :].cpu().numpy()
            results[global_i] = {p: gathered[k] for k, p in enumerate(pos_list)}

    return results


@torch.no_grad()
def score_protein_variants(
    model: torch.nn.Module,
    tokenizer,
    seq: str,
    sst_seq: Sequence[int],
    mut_strings: Sequence[str],
    *,
    mode: str = 'dual_independent',
    batch_size: int = 64,
    max_tokens: int = 16384,
    device: Optional[str | torch.device] = None,
    verify_wt: bool = True,
) -> ProSSTScoreResult:
    """Score a list of mutations on one protein with ProSST under the requested masking/view mode."""
    if mode not in VALID_MODES:
        raise ValueError(f'Unknown mode={mode!r}; expected one of {VALID_MODES}')
    if len(sst_seq) != len(seq):
        raise ValueError(f'SST length ({len(sst_seq)}) != sequence length ({len(seq)})')

    t0 = time.perf_counter()
    dev = _resolve_device(model, device)
    vocab = tokenizer.get_vocab()
    mask_id = getattr(tokenizer, 'mask_token_id', None)
    if mask_id is None:
        mask_id = vocab['<mask>']

    enc = tokenizer([seq], return_tensors='pt')
    wt_ids = enc['input_ids'][0].cpu().numpy().astype(np.int64)
    if wt_ids.shape[0] != len(seq) + 2:
        raise ValueError(f'Expected tokenized length {len(seq) + 2}, got {wt_ids.shape[0]}')
    ss_ids_1 = structure_tokens(sst_seq)

    n_rows = len(mut_strings)
    parsed: List[Optional[List[Tuple[int, int, int]]]] = [None] * n_rows
    for idx, m_str in enumerate(mut_strings):
        muts = parse_mutation(m_str)
        ok = True
        items: List[Tuple[int, int, int]] = []
        for wt_aa, pos, mt_aa in muts:
            if not (0 <= pos < len(seq)) or wt_aa not in vocab or mt_aa not in vocab:
                ok = False
                break
            if verify_wt and seq[pos] != wt_aa:
                raise AssertionError(f'WT mismatch at 1-based pos {pos + 1}: expected {wt_aa}, sequence has {seq[pos]}')
            items.append((pos, int(vocab[wt_aa]), int(vocab[mt_aa])))
        if ok:
            parsed[idx] = items

    wt_pred = np.full(n_rows, np.nan, dtype=np.float64)
    mt_pred = np.full(n_rows, np.nan, dtype=np.float64)

    # Deduplicated evaluation state registry: seq_bytes -> state_idx
    state_map: Dict[bytes, int] = {}
    unique_inputs: List[np.ndarray] = []
    needed_pos_sets: List[set[int]] = []

    def _register(arr: np.ndarray, pos: int) -> int:
        key = arr.tobytes()
        s_idx = state_map.get(key)
        if s_idx is None:
            s_idx = len(unique_inputs)
            state_map[key] = s_idx
            unique_inputs.append(arr.copy())
            needed_pos_sets.append({pos})
        else:
            needed_pos_sets[s_idx].add(pos)
        return s_idx

    # Register WT pass states
    wt_refs: List[Optional[List[Tuple[int, int, int, int]]]] = [None] * n_rows
    for idx, items in enumerate(parsed):
        if items is None:
            continue
        refs = []
        if mode in ('wt_unmasked', 'dual_unmasked'):
            for pos, wt_tid, mt_tid in items:
                s_idx = _register(wt_ids, pos)
                refs.append((s_idx, pos, wt_tid, mt_tid))
        else:  # wt_independent, dual_independent
            for pos, wt_tid, mt_tid in items:
                m_arr = wt_ids.copy()
                m_arr[pos + 1] = mask_id
                s_idx = _register(m_arr, pos)
                refs.append((s_idx, pos, wt_tid, mt_tid))
        wt_refs[idx] = refs

    n_wt_passes = len(unique_inputs)

    # Register MT pass states (reusing identical WT states automatically, e.g. for k=1 singles under dual_independent!)
    mt_refs: List[Optional[List[Tuple[int, int, int, int]]]] = [None] * n_rows
    if mode in ('dual_unmasked', 'dual_independent'):
        for idx, items in enumerate(parsed):
            if items is None:
                continue
            mt_arr = wt_ids.copy()
            for pos, _wt_tid, mt_tid in items:
                mt_arr[pos + 1] = mt_tid
            refs = []
            if mode == 'dual_unmasked':
                for pos, wt_tid, mt_tid in items:
                    s_idx = _register(mt_arr, pos)
                    refs.append((s_idx, pos, wt_tid, mt_tid))
            else:  # dual_independent
                for pos, wt_tid, mt_tid in items:
                    m_arr = mt_arr.copy()
                    m_arr[pos + 1] = mask_id
                    s_idx = _register(m_arr, pos)
                    refs.append((s_idx, pos, wt_tid, mt_tid))
            mt_refs[idx] = refs

    n_mt_passes = len(unique_inputs) - n_wt_passes

    # Execute all unique forward passes in micro-batches
    needed_positions = [sorted(s) for s in needed_pos_sets]
    state_lps = _eval_unique_states(
        model=model,
        ss_ids_1=ss_ids_1,
        unique_inputs=unique_inputs,
        needed_positions=needed_positions,
        batch_size=batch_size,
        max_tokens=max_tokens,
        device=dev,
    )

    # Assemble per-variant scores
    for idx in range(n_rows):
        refs = wt_refs[idx]
        if refs is None:
            continue
        wt_pred[idx] = sum(float(state_lps[s_idx][pos][mt_tid] - state_lps[s_idx][pos][wt_tid]) for s_idx, pos, wt_tid, mt_tid in refs)
        if mode in ('dual_unmasked', 'dual_independent'):
            mrefs = mt_refs[idx]
            assert mrefs is not None
            mt_pred[idx] = sum(float(state_lps[s_idx][pos][mt_tid] - state_lps[s_idx][pos][wt_tid]) for s_idx, pos, wt_tid, mt_tid in mrefs)
        else:
            mt_pred[idx] = wt_pred[idx]

    comb_pred = 0.5 * (wt_pred + mt_pred)
    epi_pred = 0.5 * (mt_pred - wt_pred)

    return ProSSTScoreResult(
        wt_pred=wt_pred,
        mt_pred=mt_pred,
        comb_pred=comb_pred,
        epi_pred=epi_pred,
        n_wt_passes=n_wt_passes,
        n_mt_passes=n_mt_passes,
        elapsed_s=time.perf_counter() - t0,
    )
