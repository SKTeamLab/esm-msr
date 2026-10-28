# Inference resource tracking

What an inference run costs in wall time, CPU, host RAM and GPU memory, at the resolution of one protein, for ESM-MSR,
MutateEverything (ME), ThermoMPNN / ThermoMPNN-D, SPURS and ProteinMPNN, measured in the same resource slot. This replaces the October
Slurm ledger (`GPU4_INFERENCE_LEDGER.md`, `benchmarks/slurm_jobs_oct2026.csv`, `benchmarks/timing_cache_flags.csv`,
`analysis_scripts/runtime_log_audit*.py`, `benchmark_matrix.py ledger`), which recorded job elapsed time and, for some
models, per-dataset time, and nothing about memory or CPU. It is still readable in history:
`git show d1cf240:docs/GPU4_INFERENCE_LEDGER.md` (its §3 has the ESM3ΔG / IFUM fp16 measurements the IFUM entry of
`OUTSTANDING_EXPERIMENTS.md` cites).

## What existed before (audit, 2026-10-28)

| model | wall time | GPU memory | host RAM | CPU |
|---|---|---|---|---|
| ESM-MSR | `time` column of the stats CSVs (per protein for Tsuboyama and Domainome, per dataset for the external sets, per assay for the DMS; `infer_mutants` only); ProteinGym `summary.csv` `time_s` per DMS | ProteinGym only (`peak_alloc_gb`, `peak_reserved_gb`) | none | none |
| ME | none (Slurm elapsed per job) | none | Slurm `MaxRSS` per job | none |
| ThermoMPNN(-D) | none (Slurm elapsed per job); ProteinGym `time_s` per DMS | ProteinGym only | Slurm `MaxRSS` per job | none |

SPURS and ProteinMPNN had the same shape: a `time` column in their stats files and, on ProteinGym, `time_s` and VRAM from the
runner; no RAM or CPU. The only memory record for non-ProteinGym runs was `sacct`'s job-level `MaxRSS`, and the resources were not uniform: ME's
evaluation script asks for 3 GPUs (`torchrun --nproc_per_node=3`, 24 CPUs, 128 GB; the corrected-seed evals ran with 8 CPUs
and 120 GB) while ThermoMPNN ran 1 GPU with 4 CPUs and 32 GB and ESM-MSR 1 GPU with 4 CPUs and 8–24 GB; batch sizes differ per block in ESM-MSR (1, 16, `mem_size × 32`, 32), ThermoMPNN batched across proteins by
token budget, and the job index stopped at 2026-10-24.

## What is recorded now

`inference_scripts/resource_tracker.py` (stdlib; torch and pandas optional). On by default in `esm_msr_testing.py`,
`pgym_runner.py` (so every ProteinGym run through the shared loop), and the run-everything scripts of
`inference_scripts/baselines/` (`thermompnn_inference.py`, `mutate_everything_inference.py`, `spurs_inference.py`, `proteinmpnn_inference.py`; see its README for
reproducing them, the pinned upstream commits and the environments). `--no_resource_tracking` (or `MSR_TRACK_RESOURCES=0`)
turns it off.

Output: `<resource_dir>/<run_id>/resources_run.json` (host, GPU, Slurm environment, the fixed-slot check, neighbours on the
node at start and end, totals), `resources_scopes.jsonl` (one row per scope, flushed as the run goes, so a killed run is
readable) and `resources_scopes.csv`. `run_id` is `<model>-<slurm job>-<timestamp>`, which also separates a rerun from its
predecessor. Default `resource_dir`: ESM-MSR `analysis_notebooks/resources/<ckpt>_sigma<σ>/` (ProteinGym:
`<pgym_out>/resources/`), the baselines `<--out>/resources/`.

Rows come at four levels:

| level | what it wraps | use |
|---|---|---|
| `protein` | **only the inference call(s)** on one protein or library | the inference cost: sum these per dataset |
| `dataset` | one benchmark set, data handling included | what the whole set costs end to end |
| `load` | ME's batch fetch (featurization, MSAs; the loader runs in-process) | cost that is inference-adjacent but not the forward pass |
| `stage` | model loading | one-off cost, excluded from inference |
| `dms` | one ProteinGym DMS through `score_fn`, input handling included | ProteinGym end to end; its `protein` rows are the inference |

Columns of every row: `wall_s` (CUDA-synchronised at both ends), `cpu_s` and `cpu_cores` (CPU seconds of the process, all
threads, and cores busy on average), `ram_rss_start/end/peak_gb` (resident set of the process; peak from a 0.5 s sampler, so
a transient allocation inside a scope shorter than the interval can be missed; scopes also get `max(start, end)`),
`vram_weights_gb` (allocated on entry), `vram_alloc_peak_gb` and `vram_reserved_peak_gb` (PyTorch's exact peaks for the
scope, counters reset on entry and restored to the parent on exit), `vram_device_peak_gb` (sampled used memory of the
device: also counts the CUDA context and non-PyTorch allocations), `io_read_mb` / `io_write_mb`, `loadavg1` (node load at
exit), and `t_<phase>_s` for the phases a wrapper marks. Per-protein extras: `n_rows`, `seq_len` (ThermoMPNN, ME),
`batch_size`, and for ThermoMPNN `model` / `part` (single, epistatic, additive_0/1) and phases `featurize`, `forward`, `d2h`;
for ME phases `h2d`, `count` (computing `n_rows`, excludable), `forward`, `extract`.

Cost of the tracker (`tests/test_resource_tracker.py`, CPU): about 0.3 ms per protein scope with one phase, and no
measurable slowdown of a compute loop with the sampler running; about 480 bytes per row (Domainome's 522 proteins are
250 kB). On a GPU each scope and phase boundary adds a `torch.cuda.synchronize()`, which is nothing at protein granularity
(ThermoMPNN's batches already end in `.cpu()`).

Granularity: all five models are per protein (ThermoMPNN per batch is also possible from the phases; ESM-MSR's `infer_mutants`
is one call per protein, and splitting its WT and MT passes would need phases inside `src/esm_msr/inference.py`, not done;
SPURS marks `single_forward`, `multi_singles` and `multi_targets` phases; ProteinMPNN `score`). Because ME predicts every
single and double of a protein in one pass whatever subset is requested, its cost is per protein, not per variant.

`analysis_scripts/resource_report.py RUN_DIR... [--out summary.csv] [--proteins proteins.csv]` reads runs (a parent
directory is searched) and prints one row per model and dataset: proteins, rows, inference wall, rows/s, CPU core-seconds
and mean cores, peak RAM, peak VRAM, plus `load`/`dataset` time, the GPU, and `slot_ok`, `neighbours_start/end`, so a
measurement taken outside the slot or next to another job is visible.

## Changes to the baseline scripts that affect comparison with October predictions

The scripts moved into `inference_scripts/baselines/` (README there has the full list): ThermoMPNN's `predict()` no longer
batches across proteins (a batch then belongs to one protein; `--mixed_protein_batches` restores the old behaviour; the model
reads items independently, so predictions should agree to float rounding), MutateEverything runs on one GPU in one process,
SPURS and ProteinMPNN score every benchmark from the shared rows (`benchmark_rows.py`) with one scoring function each.

## The fixed slot

All three models get the same resources, one quarter of gpu4 (RTX A4500 × 4, 24 CPUs, 250 GB): **1 GPU, 6 CPUs, 60 GB**.

```bash
MODEL=thermompnn SINGLE=/path/single.ckpt EPISTATIC=/path/epistatic.ckpt OUT=runs/thermompnn \
    inference_scripts/sbatch_fixed_slot.sh inference_scripts/baselines/slurm_inference.sh
# likewise MODEL=mutate_everything CKPT=..., spurs, proteinmpnn, esm_msr CKPT=...  (README in inference_scripts/baselines)
```

It sets `MSR_FIXED_GPUS/CPUS/RAM_GB`, `OMP/MKL/OPENBLAS_NUM_THREADS=6`, `MSR_REPO` and the sbatch options. 60 GB is a quarter of gpu4's 250,604 MiB (4 x 61,440 MiB = 245,760 MiB fits with 4,844 MiB to spare), so four jobs can run at once with 6 CPUs each. The tracker compares what the
process really has (visible GPUs, CPU affinity mask, torch threads, cgroup memory limit) with the slot and writes the result
to `resources_run.json` (`limits.check`). `STRICT=1` aborts the run at start if the check fails and raises if the process RSS
passes the RAM limit. It is on by default (`STRICT=0` to turn off): the smoke runs of 2026-10-28 showed that gpu4's Slurm
cgroups pass the check (6 CPUs in the affinity mask, 1 visible GPU, a 60 GB cgroup memory limit). Slurm's cgroup is the hard cap
on RAM; the tracker only verifies it.

The wrapper cannot stop other users' jobs sharing the node, so it prints the jobs running there. Submit when gpu4 is
otherwise idle: the three models can run at the same time, one per GPU, which loads the node equally for each. The run
records the neighbours at start and end and the node load per scope.

Native choices stay native and are recorded, not forced: ESM-MSR bf16 (the A4500 has native bf16) with its per-block batch
sizes, ME fp32 at batch 1 per protein with the loader in-process (`num_workers=0`), ThermoMPNN fp32 with the token-budget
batcher. A different batch size or precision is a different measurement.

## What to rerun

Everything is a rerun of inference only, on the gpu4 slot, one checkpoint per model (inference cost does not depend on the
seed): ESM-MSR baseline seed 1 (and devel if its cost should be shown), ME stage 2 seed 1 (stages share the architecture),
ThermoMPNN-D seed 1 (single and epistatic networks), SPURS (published models) and ProteinMPNN. Benchmarks: the 12 external sets,
Tsuboyama val and test, the 7 DMS, Domainome, ProteinGym; `baselines/slurm_inference.sh` runs all of them for one model.
Rough size from the October jobs: ESM-MSR about 2 h for everything but ProteinGym and about 32 h for ProteinGym on an A4500;
ThermoMPNN-D about 1-2 h plus about 7 h for ProteinGym; ME about 4 h for a Tsuboyama + external evaluation in October (possibly
sharded over several GPUs, so longer on one), its ProteinGym time on one GPU unmeasured; SPURS and ProteinMPNN took 0.1-6 h for
ProteinGym on RTX 6000s (not comparable). About 3-4 GPU-days if all of ProteinGym is included, about half a day without it
(`--skip_pgym` for SPURS/ProteinMPNN, `--sets` for ME, `esm_msr_testing.py` without `--protein_gym`). A stratified ProteinGym
subset by protein length would give the length dependence for a fraction of that.

Nothing already measured carries over: no earlier run has memory, CPU or per-protein numbers in the slot, and the earlier
timings mixed RTX 6000 and A4500 nodes, shared nodes and different batch policies.

Known risks for the rerun: the old ME ProteinGym job on gpu4 (387353) was OOM-killed at 12 GB after 3.2 h, and an ME eval on
gpu3 peaked at 41 GB; the long proteins at 60 GB may still not fit, and that is then a result (a failed DMS in the log),
not something to hide by raising the cap for ME. ThermoMPNN's ProteinGym job peaked at 12.9 GB.

## Not verified, and not covered

* Smoke runs of every script on gpu4, 2026-10-28 (`--max_proteins 2` on every benchmark plus two ProteinGym DMS, in the slot,
  next to three other jobs): all five tracked correctly. Checked: the slot check passes; ESM-MSR's tracker time equals its old
  `time` column to 1 ms on all 12 external sets; the tracker's peak RSS is at or above `sacct`'s MaxRSS (sacct samples and
  misses short peaks, e.g. ESM-MSR model loading 8.5 vs 6.4 GB). Not smoke-tested: ProteinGym beyond two DMS, the full DMS
  assays (GB1), mutant-background Tsuboyama libraries in SPURS/ProteinMPNN at scale.
* Reading the numbers: `vram_alloc_peak_gb` and `vram_reserved_peak_gb` are exact per scope; `vram_device_peak_gb` is sampled and
  includes the CUDA context and the allocator's cache from earlier scopes, so it can exceed a small scope's own need. ThermoMPNN
  writes one `protein` row per library and network pass (singles, additive_0/1, epistatic), so its `n_proteins` and summed
  `n_rows` count passes. ME's `load` rows use up to ~5 cores (its featurization is multi-threaded), its forward ~1.
* ThermoMPNN predictions under per-protein batching versus the old mixed batching, on one dataset (for example `external`
  with and without `--mixed_protein_batches`): expect agreement to about 1e-6.
* The baseline scripts have run only on the smoke subset above (and against fake models in `tests/test_baseline_scorers.py`);
  none has run a full benchmark. The smoke runs found and fixed: script names shadowing the upstream packages, ME's old torch
  rejecting `expandable_segments`, upstream's Slurm rank detection, ME losing its fasta/msa layout in filtered inputs, ME
  reporting success with every set failed, and `--max_proteins` not limiting the DMS assays.
* ME's Tsuboyama and external sets used to go through `main_train.py --eval --dist_eval`; `mutate_everything_inference.py` writes the same
  `results_<set>.csv` columns, but its predictions on those sets have not been compared with the earlier files.
* SPURS and ProteinMPNN predictions are not byte-comparable with the earlier runs (shared rows, one scorer; see the README).
* The other third-party models (ESMtherm, IFUM, GeoDDG, ESM3ΔG, ProSST, DDGemb, DDMut) are not covered: they run through the
  stability-baselines harness outside this repository, which keeps its own per-dataset `timing.csv`. The tracker module is
  generic, so a wrapper only needs the same `scope` calls around its inference.
