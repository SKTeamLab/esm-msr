"""
Block training: a training step is a BLOCK of a library, not a list of variants.

A masked site reads out the whole alphabet at once: one forward of the wild-type sequence with position p masked gives the 20 logits at p,
so ``logit[x] - logit[wt]`` is the LLR of every substitution there, the wild-type residue itself reading exactly 0. So

* a SINGLES block is B positions of one library: B forward passes give a ``[B, 20]`` tensor, regressed and ranked against the observed
  singles (the unobserved cells are masked out), as ONE list of B x 20 entries;
* a PAIR block is one position pair (i, j): 2 wild-type passes (i masked; j masked) and 38 mutant passes, 19 'row' passes
  (substitution x at i, j masked: the conditional readout of every y at j) and 19 'column' passes (y at j, i masked), give every
  19 x 19 double. The wild-type head's cell is ``llr_i[x] + llr_j[y]`` and the mutant head's ``Row[x][y] + Col[y][x]``, the
  thermodynamic cycle of validation (``--val_masking independent``), so ``0.5 (WT + MT)`` is the reported prediction.

An epoch is every library's singles blocks, then every pair block. The labels are rebuilt as dense ``[P, 20]`` / ``[Q, 20, 20]`` tensors
from the cache's per-item records when the datasets are built (seconds; the cache itself is not rebuilt). This module is torch-only and
knows nothing about the model: the sequences a block needs and the assembly of its cells are pure functions, tested on their own.
"""
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from esm.utils.constants import esm3 as C

AAS = 'ACDEFGHIKLMNPQRSTVWY'                                                  # MSRModel.valid_canonical_aas, in this order
CANON_IDS = [C.SEQUENCE_VOCAB.index(a) for a in AAS]
TOKEN_TO_AA = {t: k for k, t in enumerate(CANON_IDS)}
MASK = int(C.SEQUENCE_MASK_TOKEN)

SINGLE_FIELDS = ('y', 'cens', 'bound', 'dG_meas', 'dG_wt', 'cens_src')


@dataclass
class Library:
    """One library's block tensors. ``single_pos`` / ``pairs`` are token positions; ``wt_tok`` maps a position to its wild-type token id."""
    name: str
    seq: torch.Tensor                       # [Lseq] wild-type tokens
    coords: torch.Tensor                    # [1, 1, L, 37, 3], as collate stacks one item
    structure_tokens: torch.Tensor          # [1, L]
    wt_tok: Dict[int, int]
    single_pos: np.ndarray                  # [P] sorted token positions that have at least one observed single
    singles: Dict[str, np.ndarray] = field(default_factory=dict)       # name -> [P, 20]
    pairs: List[Tuple[int, int]] = field(default_factory=list)         # (i, j), i < j
    pair_data: Dict[str, np.ndarray] = field(default_factory=dict)     # name -> [Q, 20, 20]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


def _fill(shape, name):
    if name in ('cens', 'cens_src'):
        return np.zeros(shape, dtype=np.int64)
    return np.full(shape, np.nan, dtype=np.float32)


def build_library(name: str, items: Sequence[dict]) -> 'Library | None':
    """
    The dense label tensors of one library from its (already filtered and censored) per-item records.

    Only ``single`` and ``double`` items with canonical amino acids count; a position whose wild type is not canonical is dropped. A cell
    seen twice keeps the last record. ``y`` is the item's ddG (the censored bound lives in ``bound``). A double's teacher-forcing singles are read
    from ``singles``, not from its ``ddG_A`` / ``ddG_B``, which are both NaN when either single is tagged or missing.
    """
    items = [it for it in items if it.get('subset_type') in ('single', 'double')]
    if not items:
        return None
    ref = items[0]
    wt_tok: Dict[int, int] = {}
    singles: Dict[int, Dict[str, np.ndarray]] = {}
    pairs: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for it in items:
        pos = [int(p) for p in np.asarray(it['mut_pos']).ravel()]
        wts = [int(t) for t in np.asarray(it['wt_id']).ravel()]
        mts = [int(t) for t in np.asarray(it['mt_id']).ravel()]
        if len(pos) not in (1, 2) or any(t not in TOKEN_TO_AA for t in wts + mts):
            continue
        order = np.argsort(pos)
        pos, wts, mts = [pos[k] for k in order], [wts[k] for k in order], [mts[k] for k in order]
        for p, w in zip(pos, wts):
            wt_tok[p] = w
        vals = {'y': it.get('ddG', np.nan), 'cens': int(it.get('cens', 0)), 'bound': it.get('cens_bound', np.nan),
                'dG_meas': it.get('dG_meas', np.nan), 'dG_wt': it.get('dG_wt', np.nan),
                'cens_src': int(it.get('cens_src', 0))}
        if len(pos) == 1:
            d = singles.setdefault(pos[0], {f: _fill(20, f) for f in SINGLE_FIELDS})
            for f in SINGLE_FIELDS:
                d[f][TOKEN_TO_AA[mts[0]]] = vals[f]
        else:
            d = pairs.setdefault((pos[0], pos[1]), {f: _fill((20, 20), f) for f in SINGLE_FIELDS})
            cell = (TOKEN_TO_AA[mts[0]], TOKEN_TO_AA[mts[1]])
            for f in SINGLE_FIELDS:
                d[f][cell] = vals[f]
    if not singles and not pairs:
        return None
    sp = np.array(sorted(singles), dtype=np.int64)
    lib = Library(
        name=name,
        seq=torch.as_tensor(ref['wt_sequence_tokens'], dtype=torch.long),
        coords=torch.as_tensor(ref['coords_orig'], dtype=torch.float32).unsqueeze(0),
        structure_tokens=torch.as_tensor(ref['structure_tokens_orig'], dtype=torch.long).unsqueeze(0),
        wt_tok=wt_tok, single_pos=sp,
        singles={f: np.stack([singles[int(p)][f] for p in sp]) for f in SINGLE_FIELDS} if len(sp) else {},
        pairs=sorted(pairs),
        pair_data={f: np.stack([pairs[k][f] for k in sorted(pairs)]) for f in SINGLE_FIELDS} if pairs else {})
    return lib


# ---------------------------------------------------------------------------------------------------------------- pure block geometry

def single_sequences(seq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``[B, Lseq]``: the wild-type sequence with position ``positions[b]`` masked in row b."""
    out = seq.unsqueeze(0).repeat(int(positions.shape[0]), 1)
    out[torch.arange(out.shape[0]), positions] = MASK
    return out


def pair_sequences(seq: torch.Tensor, i: int, j: int, wt_i: int, wt_j: int):
    """
    The passes of a pair block, as ``(wt_rows [2, L], row_seqs [20, L], col_seqs [20, L], Xa, Ya)``.

    The alphabet is all 20 residues on both axes: the row pass of the wild-type residue at i is the wild type with j masked (its read is the single at j,
    seen by the mutant adapter) and the column pass of the wild type at j likewise, so the matrices are 20 x 20 with the singles on the identity row and column,
    from 40 mutant passes.

    ``wt_rows[0]`` is the wild type with i masked (read at i), ``wt_rows[1]`` with j masked (read at j). Row pass x carries substitution x
    at i and masks j (read at j: the conditional readout of every y given x); column pass y carries y at j and masks i (read at i).
    ``Xa`` / ``Ya`` are the alphabet indices of the residues at i / j (all 20).
    """
    canon = torch.tensor(CANON_IDS, dtype=torch.long, device=seq.device)
    Xa = list(range(20))
    Ya = list(range(20))
    wt_rows = seq.unsqueeze(0).repeat(2, 1)
    wt_rows[0, i] = MASK
    wt_rows[1, j] = MASK
    row_seqs = seq.unsqueeze(0).repeat(len(Xa), 1)
    row_seqs[:, i] = canon[Xa]
    row_seqs[:, j] = MASK
    col_seqs = seq.unsqueeze(0).repeat(len(Ya), 1)
    col_seqs[:, j] = canon[Ya]
    col_seqs[:, i] = MASK
    return wt_rows, row_seqs, col_seqs, Xa, Ya


def unmasked_rows(seqs: torch.Tensor, seq: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``seqs`` ``[R, L]`` with the rows where ``keep`` is True restored to the plain sequence ``seq``: their site is read VISIBLE."""
    return torch.where(keep.unsqueeze(1), seq.unsqueeze(0), seqs)


def mutant_single_sequences(seq: torch.Tensor, positions: torch.Tensor, wt_ids: torch.Tensor):
    """
    The unmasked mutant-adapter inputs of some positions: every one of the 19 substitutions as its own sequence, ``(seqs [n*19, L], read_pos
    [n*19], wt_at [n*19], row [n*19], col [n*19])`` with ``row`` the index into ``positions`` and ``col`` the alphabet index of the substitution.
    The mutant adapter reads such a sequence with the mutation VISIBLE: ``logit[x] - logit[wt]`` at the site, the per-item unmasked readout.
    """
    canon = torch.tensor(CANON_IDS, dtype=torch.long, device=seq.device)
    rows, cols, wt_cols = [], [], []
    for r in range(int(positions.shape[0])):
        wa = TOKEN_TO_AA[int(wt_ids[r])]
        for a in range(20):
            if a != wa:
                rows.append(r); cols.append(a); wt_cols.append(int(wt_ids[r]))
    row = torch.as_tensor(rows, dtype=torch.long, device=seq.device)
    col = torch.as_tensor(cols, dtype=torch.long, device=seq.device)
    seqs = seq.unsqueeze(0).repeat(len(rows), 1)
    pos = positions[row]
    seqs[torch.arange(len(rows), device=seq.device), pos] = canon[col]
    return seqs, pos, torch.as_tensor(wt_cols, dtype=torch.long, device=seq.device), row, col


def pair_unmasked_sequences(seq: torch.Tensor, i: int, j: int, Xa: Sequence[int], Ya: Sequence[int]):
    """
    The unmasked mutant-adapter inputs of a pair block: one sequence per cell, ``x`` at ``i`` and ``y`` at ``j`` (the wild-type residue where a
    cell has no mutation there), each listed TWICE in a row so one call reads both sites: rows ``2c`` read ``i`` and ``2c + 1`` read ``j`` of cell
    ``c = x_index * len(Ya) + y_index``. Returns ``(seqs [2*nx*ny, L], read_pos [2*nx*ny])``.
    """
    canon = torch.tensor(CANON_IDS, dtype=torch.long, device=seq.device)
    nx, ny = len(Xa), len(Ya)
    cells = seq.unsqueeze(0).repeat(nx * ny, 1)
    cells[:, i] = canon[torch.as_tensor(list(Xa), device=seq.device)].repeat_interleave(ny)
    cells[:, j] = canon[torch.as_tensor(list(Ya), device=seq.device)].repeat(nx)
    seqs = cells.repeat_interleave(2, dim=0)
    read = torch.tensor([i, j], dtype=torch.long, device=seq.device).repeat(nx * ny)
    return seqs, read


def assemble_pair(llr_i: torch.Tensor, llr_j: torch.Tensor, row: torch.Tensor, col: torch.Tensor, Xa: Sequence[int], Ya: Sequence[int]):
    """
    The 19 x 19 raw cells of a pair block, ``(wt_raw, mt_raw)``, rows = substitutions at i, columns = substitutions at j.

    ``llr_i`` / ``llr_j`` are the two wild-type readouts ``[20]``; ``row`` ``[19, 20]`` row pass x read at j, ``col`` ``[19, 20]`` column
    pass y read at i. WT = llr_i[x] + llr_j[y] (the two singles); MT = row[x][y] + col[y][x] = ddG(Y|X) + ddG(X|Y).
    """
    x, y = torch.as_tensor(list(Xa), device=llr_i.device), torch.as_tensor(list(Ya), device=llr_i.device)
    wt_raw = llr_i[x].unsqueeze(1) + llr_j[y].unsqueeze(0)
    mt_raw = row[:, y] + col[:, x].t()
    return wt_raw, mt_raw


def pair_marginal_sequences(seq: torch.Tensor, i: int, j: int, wt_rows: torch.Tensor) -> torch.Tensor:
    """
    The 4 MT readout rows for a simultaneous-masked ('marginal') pair block (i, j), ``[4, L]``:
      row 0: both i and j masked, read at i (for double cells x != wi, y != wj)
      row 1: both i and j masked, read at j (identical sequence to row 0; deduplicated by forward_site_llr)
      row 2: only i masked (wt_rows[0]), read at i (for the identity column y == wj, single mutants at i)
      row 3: only j masked (wt_rows[1]), read at j (for the identity row x == wi, single mutants at j)
    """
    both = seq.clone()
    both[i] = MASK
    both[j] = MASK
    return torch.stack([both, both, wt_rows[0], wt_rows[1]], dim=0)


def assemble_pair_marginal(llr_i: torch.Tensor, llr_j: torch.Tensor, llr_m: torch.Tensor, Xa: Sequence[int], Ya: Sequence[int], wi: int, wj: int):
    """
    Assemble ``(wt_raw, mt_raw)`` over ``Xa`` x ``Ya`` when the MT head masks all mutated sites simultaneously (``--block_marginal_mt``).
    ``llr_m`` is ``[4, 20]`` from ``pair_marginal_sequences``: ``llr_m[0]`` / ``llr_m[1]`` read i / j with both sites masked (used on double
    cells ``x != wi, y != wj``); ``llr_m[2]`` / ``llr_m[3]`` read i / j with only that single site masked (used on the identity column ``y == wj``
    and identity row ``x == wi``).
    """
    dev = llr_i.device
    x, y = torch.as_tensor(list(Xa), device=dev), torch.as_tensor(list(Ya), device=dev)
    wt_raw = llr_i[x].unsqueeze(1) + llr_j[y].unsqueeze(0)
    mt_raw = llr_m[0][x].unsqueeze(1) + llr_m[1][y].unsqueeze(0)
    is_wi = (x == wi).unsqueeze(1)
    is_wj = (y == wj).unsqueeze(0)
    single_i = llr_m[2][x].unsqueeze(1).expand_as(mt_raw)
    single_j = llr_m[3][y].unsqueeze(0).expand_as(mt_raw)
    mt_raw = torch.where(is_wj & ~is_wi, single_i, mt_raw)
    mt_raw = torch.where(is_wi & ~is_wj, single_j, mt_raw)
    mt_raw = torch.where(is_wi & is_wj, torch.zeros_like(mt_raw), mt_raw)
    return wt_raw, mt_raw


def pair_labels(b: dict, Xa: Sequence[int], Ya: Sequence[int], wi: int, wj: int):
    """
    The label matrices of a pair block over the alphabet rows ``Xa`` and columns ``Ya`` (alphabet indices; ``wi`` / ``wj`` the wild-type
    residues at i and j).

    The double cells come from the library's doubles; the identity row and column are the SINGLES at j and at i (cell (x, wt_j) is the single x
    at i), and the wild type itself is 0. The library constants dG_wt and the floor are filled into every cell, so no cell hands the link a NaN.
    Returns ``(lab, ident, full)``: ``lab`` the six label fields restricted to ``Xa`` x ``Ya``; ``ident`` marks the identity cells; ``full`` the
    same fields over the whole 20 x 20 alphabet, which the conditional legs index.
    """
    dev = b['y'].device
    X, Y = torch.as_tensor(list(Xa), device=dev), torch.as_tensor(list(Ya), device=dev)
    M = {f: b[f].clone() for f in SINGLE_FIELDS}
    for f in ('y', 'cens', 'bound', 'dG_meas', 'cens_src'):
        M[f][:, wj] = b['si_' + f]
        M[f][wi, :] = b['sj_' + f]
    finite = torch.cat([t.reshape(-1) for t in (b['dG_wt'], b['si_dG_wt'], b['sj_dG_wt'])])
    finite = finite[torch.isfinite(finite)]
    dG = finite.mean() if finite.numel() else torch.tensor(float('nan'), device=dev)
    M['dG_wt'] = torch.full_like(M['dG_wt'], float(dG))
    M['y'][wi, wj], M['cens'][wi, wj], M['bound'][wi, wj], M['dG_meas'][wi, wj] = 0.0, 0, float('nan'), float(dG)
    ident = torch.zeros(20, 20, dtype=torch.bool, device=dev)
    ident[:, wj] = True
    ident[wi, :] = True
    sl = lambda t: t[X][:, Y]
    return {f: sl(M[f]) for f in M}, sl(ident), M


# ---------------------------------------------------------------------------------------------------------------- dataset and sampler

class BlockSampler(Sampler):
    """
    The blocks of one epoch, as descriptors: every library's singles blocks (B random positions of one library each) in random order, then
    every pair block in random order. Seeded by ``(seed, epoch)``, so an epoch is reproducible and ``set_epoch`` is all a resume needs.

    ``domain_steps``: the libraries in random order, each as its one whole-domain singles block followed by all of its pair blocks, and each
    descriptor carries a fourth field saying whether it closes the library's optimizer step.
    """

    def __init__(self, libs: List[Library], positions_per_block: int, seed: int = 0, singles: bool = True, pairs: bool = True, domain_steps: bool = False):
        self.libs, self.B, self.seed, self.epoch = libs, max(1, int(positions_per_block)), int(seed or 0), 0
        self.singles, self.pairs, self.domain_steps = singles, pairs, domain_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def n_single_blocks(self) -> int:
        if self.domain_steps:
            return sum(1 for l in self.libs if len(l.single_pos)) if self.singles else 0
        return sum(-(-len(l.single_pos) // self.B) for l in self.libs) if self.singles else 0

    def n_pair_blocks(self) -> int:
        return sum(l.n_pairs for l in self.libs) if self.pairs else 0

    def n_steps(self) -> int:
        """Optimizer steps per epoch, before --grad_accum_blocks."""
        if not self.domain_steps:
            return len(self)
        return sum(1 for l in self.libs if (self.singles and len(l.single_pos)) or (self.pairs and l.n_pairs))

    def __len__(self) -> int:
        return self.n_single_blocks() + self.n_pair_blocks()

    def _iter_domains(self, rng):
        order = list(range(len(self.libs)))
        rng.shuffle(order)
        for li in order:
            lib = self.libs[li]
            step = [('s', li, tuple(range(len(lib.single_pos))))] if self.singles and len(lib.single_pos) else []
            if self.pairs:
                qs = list(range(lib.n_pairs))
                rng.shuffle(qs)
                step += [('p', li, q) for q in qs]
            for k, desc in enumerate(step):
                yield (*desc, k == len(step) - 1)

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        if self.domain_steps:
            yield from self._iter_domains(rng)
            return
        sing, pair = [], []
        for li, lib in enumerate(self.libs):
            if self.singles:
                idx = list(range(len(lib.single_pos)))
                rng.shuffle(idx)
                sing += [('s', li, tuple(idx[k:k + self.B])) for k in range(0, len(idx), self.B)]
            if self.pairs:
                pair += [('p', li, q) for q in range(lib.n_pairs)]
        rng.shuffle(sing)
        rng.shuffle(pair)
        yield from sing + pair


class BlockDataset(Dataset):
    """``dataset[descriptor]`` is one block's batch dict (tensors only, plus the library name)."""

    def __init__(self, libs: List[Library]):
        self.libs = libs

    def __len__(self) -> int:
        return len(self.libs)

    def __getitem__(self, desc):
        kind, li, sel, *end = desc
        lib = self.libs[li]
        t = lambda a: torch.as_tensor(a)
        batch = {'pdb': lib.name, 'seq': lib.seq, 'coords': lib.coords, 'structure_tokens': lib.structure_tokens}
        if end:                                                 # --block_domain_steps
            batch.update(step_end=end[0], n_lib_pairs=lib.n_pairs)
        if kind == 's':
            idx = np.asarray(sel, dtype=np.int64)
            pos = lib.single_pos[idx]
            batch.update(block='single', positions=t(pos), wt_ids=t(np.array([lib.wt_tok[int(p)] for p in pos], dtype=np.int64)),
                         **{f: t(lib.singles[f][idx]) for f in SINGLE_FIELDS})
        else:
            i, j = lib.pairs[sel]
            row_of = {int(p): r for r, p in enumerate(lib.single_pos)}
            partners = {}
            for tag, p in (('si_', i), ('sj_', j)):             # the measured singles at the pair's two positions (NaN where there are none)
                r = row_of.get(p)
                for f in ('y', 'cens', 'bound', 'dG_meas', 'dG_wt', 'cens_src'):
                    partners[tag + f] = t(lib.singles[f][r]) if r is not None else t(_fill(20, f))
            batch.update(block='pair', positions=t(np.array([i, j], dtype=np.int64)),
                         wt_ids=t(np.array([lib.wt_tok[i], lib.wt_tok[j]], dtype=np.int64)),
                         **{f: t(lib.pair_data[f][sel]) for f in SINGLE_FIELDS}, **partners)
        return batch


def build_block_loader(datasets: Sequence, names: Sequence[str], positions_per_block: int, seed: int, num_workers: int = 0,
                       singles: bool = True, pairs: bool = True, domain_steps: bool = False):
    """The training DataLoader of the block paradigm, from the per-library ``MutationStabilityDataset`` objects."""
    from torch.utils.data import DataLoader
    libs = []
    for ds, name in zip(datasets, names):
        lib = build_library(name, ds.data)
        if lib is not None:
            libs.append(lib)
    sampler = BlockSampler(libs, positions_per_block, seed, singles=singles, pairs=pairs, domain_steps=domain_steps)
    logging.info(f"Block training: {len(libs)} libraries, {sum(len(l.single_pos) for l in libs)} single positions in "
                 f"{sampler.n_single_blocks()} blocks of up to {'a whole domain' if domain_steps else sampler.B}, {sampler.n_pair_blocks()} pair blocks, "
                 f"{sampler.n_steps()} steps per epoch")
    return DataLoader(BlockDataset(libs), sampler=sampler, batch_size=None, num_workers=num_workers, pin_memory=True)
