# VRAM, batch sizes & maximum structure length

Reference for the paper: **how large a protein can be scored at each
batch size**, and what to expect in wall time. All numbers measured on a
single NVIDIA RTX 5090 (32 GB) with `--dtype bf16` (the model's native
precision — see [known_issues.md](known_issues.md) item 1).

## 1. The VRAM model

Memory reserved during inference scales quadratically with structure
length (the residues the model actually forwards; self-attention) and
linearly with batch size. Fitted to
`torch.cuda.max_memory_reserved()` across the benchmark:

```
reserved_GB ≈ 7.14 + 6.21e-06 · L² · b
```

- `L` = **structure length** (residues) — the AlphaFold structure chain
  the model actually forwards, *not* the full target sequence. For
  `af2_subset` DMS the two differ a lot (BRCA2: target 3,418 → structure
  2,832; ZIKV: target 3,423 → structure 504, which is why ZIKV scores
  fine at batch 16). `b` = batch size.
- **Fixed term ≈ 7.14 GB** (model weights + structure-encoder buffers,
  present even before any sequence is loaded).
- The per-residue² coefficient is **0.75× the fp32 value (8.25e-06), not
  half**: the structure encoder's weights are forced to fp32
  (`src/esm_msr/models.py`), so bf16 saves 25%, not 50%, on the L² term.
- **Validity: L ≲ 1,400.** At longer sequences the fit over-predicts
  (it forecasts 37 GB for POLG at L=2,185, b=1; the measured live
  working set is ~26 GiB). Use the measured anchors in §4 for L > 1,400.

Solving for the largest structure that fits a memory budget `B`:

```
L_max(b) = sqrt( (B − 7.14) / (6.21e-06 · b) )
```

## 2. Maximum structure length per batch size (32 GB card)

Two budgets are useful: **26 GB reserved** (the conservative budget
used for the reference run — leaves ~6 GB of headroom for the CUDA
context and allocator fragmentation) and **29.4 GB** (pushing the card
close to its full 31 GB usable).

| batch `b` | L ≤ (26 GB) | L ≤ (29.4 GB) |
|----------:|------------:|--------------:|
| 1         | 1,742       | 1,893         |
| 2         | 1,232       | 1,338         |
| 3         | 1,006       | 1,093         |
| 4         | 871         | 946           |
| 6         | 711         | 772           |
| 8         | 616         | 669           |
| 12        | 503         | 546           |
| 16        | 435         | 473           |
| 24        | 355         | 386           |
| 32        | 308         | 334           |
| 48        | 251         | 273           |
| 64        | 217         | 236           |
| 96        | 177         | 193           |
| 128       | 154         | 167           |
| 192       | 125         | 136           |
| 256       | 108         | 118           |

In words, on a 32 GB card (bf16, 26 GB budget): a **~1,740-residue**
protein fits at batch 1; **~1,230** at batch 2; **~1,000** at batch 3;
**~870** at batch 4; **~710** at batch 6; **~500** at batch 12; **~435**
at batch 16; **~300** at batch 32; **~215** at batch 64; **~110** at
batch 256.

In practice you do not need to bin manually: `--auto_batch_size` probes
per DMS (doubling ladder, then bisection, with an OOM backstop) and
picks the largest fitting batch for each protein individually.

### Batches actually used in the reference run

The reference benchmark binned DMS by structure length (budget 26 GB)
and ran one
fixed batch per bin:

| L range   | DMS | mutants | batch |
|----------:|----:|--------:|------:|
| ≤ 99      | 69  | 139,064 | 64    |
| ≤ 198     | 26  | 258,451 | 48    |
| ≤ 287     | 27  | 761,823 | 32    |
| ≤ 448     | 29  | 815,604 | 12    |
| ≤ 656     | 32  | 242,333 | 6     |
| ≤ 934     | 19  | 206,265 | 3     |
| ≤ 1,390   | 10  | 22,611  | 1     |
| > 1,390   | 3   | 17,772  | 1     |

Every bin was verified to fit with margin before the run (e.g. L=934 at
b=3 → ~23.4 GB reserved).

## 3. Measured throughput (bf16)

Units per second (`u/s`) at b=1 and b=2, by structure length:

| L       | 93    | 198   | 287   | 448   | 656   | 934   | 1,390 |
|---------|-------|-------|-------|-------|-------|-------|-------|
| b = 1   | 12.5  | 13.4  | 12.7  | 12.9  | 11.9  | 11.8  | 8.8   |
| b = 2   | 24.9  | 23.4  | 25.0  | 24.1  | 23.0  | 17.8  | OOM   |

- Throughput keeps rising with batch size for short sequences (b=16
  validation runs hit ~100–240 u/s for L ≤ 250), but for L ≳ 650 the
  memory wall, not the compute, caps the batch.
- **fp32 vs bf16:** Spearman correlation identical to ±0.001 (rank
  statistics are robust to bf16 rounding) at ~2.5× slower wall time.
  bf16 is the production setting.

## 4. The long-structure anchors (L > 1,400)

The quadratic fit is only validated to L ≈ 1,400; these three DMS are
the measured anchors beyond it (all at batch 1 except SCN5A):

| DMS | L | batch | wall time | u/s | memory behavior |
|-----|----:|------:|----------:|----:|-----------------|
| SCN5A_HUMAN_Glazer_2019 | 2,016 | 2 | 54 s | ~4.2 | on-GPU (needs default allocator) |
| POLG_CXB3N_Mattenberger_2021 | 2,185 | 1 | 3,168 s | 4.96 | on-GPU (~26 GiB live; the fit over-predicts ~37 GB) |
| BRCA2_HUMAN_Erwood_2022_HEK293T | 2,832 | 1 | 760–777 s | 0.34 | 32 GB saturated + CPU spill |

Take-aways for the paper:

1. **Up to ~2,200 residues fit on a 32 GB card at batch 1** (POLG
   proves 2,185; the fit's 1,742 is conservative). Batch 2 has been run
   at L=2,016 (SCN5A, ~4.2 u/s).
2. **~2,832 residues exceeds the card**: the run saturates all 32 GB
   and the remainder pages to system RAM. On WSL2 this completes
   (measured ~13 min for 265 mutants, ~14× slower than on-GPU runs);
   on a bare-metal CUDA machine the same allocation is a hard OOM.
3. Structurally long DMS (L ≳ 1,860) should use PyTorch's **default
   allocator**: `expandable_segments:True` stochastically crashes with
   a driver error ("device not ready") in the first forward pass at
   these lengths (see [known_issues.md](known_issues.md) items 2–4).

## 5. CPU spill (running larger than the card)

- **Mechanism:** on WSL2, a CUDA allocation larger than the remaining
  device memory is backed by host RAM instead of failing. The run
  completes at roughly **10–15× slower** speed (BRCA2: 0.34 u/s vs the
  ~5 u/s of an on-GPU run of comparable length).
- **Requirements:** ~54 GB of free system RAM for BRCA2 (the spill is
  in the low tens of GB on top of the 32 GB on-card).
- **How to do it deliberately:** simply run with the default allocator
  and a fixed `--batch_size 1` (see [known_issues.md](known_issues.md)
  item 3 for the exact command). Do *not* use
  `--auto_batch_size` for spill runs — its memory probes assume the
  allocation budget is the card size.
- **When to prefer a bigger card instead:** anything beyond ~3,000
  residues, or if system RAM is scarce. The L² scaling means a 48 GB
  card would take BRCA2-class structures entirely on-GPU.

## 6. WT-only mode (`--skip_reverse`)

With `--skip_reverse` the compute model is different, and the tables
above mostly stop applying:

- **One forward per DMS, always at batch size 1.** The WT sequence is
  forwarded once, its logits are cached, and every mutant is scored by
  gathering per-position local-substitution log-likelihood ratios from
  that cache. No mutant-context forward is ever run.
- **`--batch_size` is a VRAM no-op in this mode.** It only chunks the
  trivial post-forward gather/scoring loop (a few MB per chunk), so any
  value works; the reference WT-only runs used a fixed 16. Do *not* use
  `--auto_batch_size` — its probes are irrelevant here.
- **The VRAM ceiling is the batch-1 column of §2**, applied to a single
  sequence: a DMS that fits at batch 1 in full mode fits in WT-only
  mode, and the long-structure behavior is identical (SCN5A-class DMS
  still want the default allocator; BRCA2 at L=2,832 still saturates
  the card and spills to system RAM on WSL2).
- **Wall time is ~10× shorter than full mode** because the two-pass
  mutant-context forwards are gone: the full 217-DMS benchmark runs in
  ~17 min per seed on the reference machine (vs ~1 day in full mode).
