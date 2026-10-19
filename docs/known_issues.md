# Known issues & workarounds

Practical notes from running the full ProteinGym benchmark on a single
RTX 5090 (32 GB). Each item: symptom → cause → workaround. Nothing here
changes the scoring logic; all of it is operational.

## 1. `--dtype bf16` is required for the benchmark

The model was trained with `bf16-mixed` precision, so bf16 is its native
precision and the default of `--dtype`. fp32 is kept only for
small-protein debugging: on 32 GB it cannot score the structurally long
DMS at any batch size, and on pre-2025 code it additionally crashed on the
MT structure path (fp32 `plddt` input vs bf16 linear). That crash is
fixed in `src/esm_msr/inference.py` by a dtype-aware autocast helper
(`_dtype_autocast`), but bf16 remains the recommended and tested mode.

## 2. SCN5A_HUMAN_Glazer_2019 (2,016 residues) needs the default PyTorch allocator

Symptom: with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the
setting used for the rest of the benchmark), this DMS dies with a CUDA
driver error ("device not ready") or a ceiling OOM. Cause: at
structure length ≳ 1,860 the expandable-segment allocator
stochastically fails the first forward pass with a driver error —
observed for SCN5A, BRCA1 and POLG_CXB3N (item 4), though not every run
(seed1's BRCA1 passed under the same recipe); the default allocator
never reproduced it, and nothing below ~1,400 structure length has
ever failed. Workaround: score the structurally long DMS with the
default allocator:

```bash
PYTORCH_CUDA_ALLOC_CONF= python inference_scripts/esm_msr_testing.py \
    --checkpoint esm-msr/lora_seed1.safetensors --protein_gym \
    --pgym_dir pgym_inputs --pgym_out pgym_results \
    --auto_batch_size --dtype bf16 --lora_epsilon 1.0 \
    --pgym_dms SCN5A_HUMAN_Glazer_2019
```

In the reference run it then fit at batch size 2 and completed in ~54 s.
The same allocator fix applies to BRCA1 (1,863 residues; batch 1, ~5 min
on-GPU, Spearman 0.4764/0.4784/0.4786 for seeds 1/2/3 at σ=1.0) and to
BRCA2 (item 3) —
the longest DMS in the set — which additionally spills into system RAM.

## 3. BRCA2_HUMAN_Erwood_2022_HEK293T (2,832 residues) needs the default allocator + CPU spill

BRCA2 is the longest structure in the set. At batch size 1 its working
set saturates a 32 GB card and the remainder spills into system RAM
(WSL2 backs over-limit CUDA allocations with host RAM). It **is** in the
reference results, but two things are required:

- **The default PyTorch allocator.** With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  the first forward pass at this length dies with
  `RuntimeError: CUDA driver error: device not ready` inside the geometry
  attention (`esm/layers/geom_attention.py`, `distance_term`), before any
  useful work happens — the same stochastic long-structure flake as
  SCN5A (item 2).
- **`--batch_size 1`, fixed** (do not use `--auto_batch_size`, which
  probes with larger batches and hits the same driver error).

```bash
PYTORCH_CUDA_ALLOC_CONF= python inference_scripts/esm_msr_testing.py \
    --checkpoint esm-msr/lora_seed1.safetensors --protein_gym \
    --pgym_dir pgym_inputs --pgym_out pgym_results \
    --batch_size 1 --dtype bf16 --lora_epsilon 1.0 \
    --pgym_dms BRCA2_HUMAN_Erwood_2022_HEK293T
```

Measured on the reference machine (RTX 5090 + ~54 GB system RAM):
~13 min (760–777 s), ~0.34 units/s (≈14× slower than a fully
on-GPU run of comparable length), Spearman 0.4807/0.4891/0.4822 for
seeds 1/2/3 at σ=1.0, 0.4727 (small σ=0.0), 0.4863 (small σ=0.5),
0.4854 (small σ=1.0), 0.4743 (chain σ=1.0). If you do not have ~54 GB
of free system RAM it will hard-OOM — on a non-WSL2 machine the same
allocation fails outright, so a larger GPU (≥48 GB) is the clean fix.

## 4. POLG_CXB3N_Mattenberger_2021 (2,185 residues) fits on-card, barely

The quadratic VRAM model (see [vram_and_batch_sizes.md](vram_and_batch_sizes.md))
over-predicts memory at this length: it forecasts ~37 GB at batch size 1,
but the measured live working set is ~26 GiB, so it **does** fit on the
32 GB card and runs at GPU speed — 3,168 s (~50 min) for its 15,711
mutants, ~4.96 units/s. Spearman: 0.2505/0.2556/0.2554 for seeds 1/2/3
at σ=1.0, 0.1795 (small σ=0.0), 0.2243 (small σ=0.5), 0.2406 (small
σ=1.0), 0.2368 (chain σ=1.0). If you get a hard OOM on it, the
margin is thin — free up a couple of GB (close other GPU workloads) or
exclude it with `--pgym_dms`.

## 5. Runs are resumable, but `summary.csv` accumulates rows

Re-running skips every DMS that already has an output CSV, so an
interrupted run can simply be relaunched with the same command.
However, `summary.csv` is appended to, so after several partial runs it
contains rows from all of them (including stale failures that have since
been fixed). For final statistics, either use only the last run's rows,
or recompute per-DMS Spearman directly from the per-DMS CSVs — the
authoritative prediction column is `combined_pred`
(`combined_dddg_pred` is the ΔΔG-style variant and must not be used for
the headline score). In WT-only runs (`--skip_reverse`) the only
prediction column is `wt_lora_pred`, and that is the authoritative one
for those runs.

## 6. Offline / HuggingFace token

`--hf_token` is safe even with `HF_HUB_OFFLINE=1` set: HF `login` is
best-effort in the test script, so an offline cache still authenticates
from the token. Once the ESM3 base model (`esm3-sm-open-v1`) is in the
HuggingFace cache, benchmarking needs no network access at all.

## 7. Masking strategy naming: `independent` (not `chain`)

The current user-facing name for the residue-independence masking
strategy is `independent`. Older checkpoints (e.g. the legacy
`esm-msr-chain` model) were saved with `mask_strategy: chain`, and
`src/esm_msr/models.py` keeps `chain` as a read-only alias purely so
those old checkpoints still load unchanged. Do not pass the old name in
new runs or configurations.

## 8. Preprocessor `wt_mismatch` counts are harmless

`pgym_preprocess.py` reports a `wt_mismatch` total (57 for the official
release): cases where the DMS file's wild-type letter disagrees with the
structure at that position. The positional mapping is unaffected and the
official release ships with these — no action needed.

## 9. Use the right ProteinGym release

The substitutions release is Zenodo record **15293562** (v1.3,
DOI 10.5281/zenodo.15293562) — 217 DMS. Earlier records (e.g. 11201211)
are not the substitutions release and will not match this preprocessor's
expectations.

## 10. The mega-scale `wt` rows are the only source of a library's wild-type sequence

`MegaScaleDatasetPreprocessor.preprocess` drops every `mut_type == 'wt'`
row, so `self.df` holds mutants only. Taking a per-library sequence from
it (`groupby('code_wt').first()['aa_seq']`) returns a *mutant*: for all
371 libraries that sequence differs from the wild type at one or two
positions. Use `self.aa_seq_wt` (captured before the filter, keyed like
`code_wt`) when the wild-type sequence is needed; it matches the
AlphaFold model exactly for all 363 libraries that have a `wt` row.

This affected `preprocessing/split_tsuboyama.py`, whose homology FASTA
carried mutant sequences. One designed library then measured 40.0%
identity to an external benchmark instead of its true 42.5%, landing
exactly on the 40% cap and escaping exclusion. Fixed 2026-10-06; the
splits in `data/splits_oct06_capped.*` were regenerated from wild-type
sequences. `aa_seq` is not used as wild-type model context anywhere, so
training was not affected.
