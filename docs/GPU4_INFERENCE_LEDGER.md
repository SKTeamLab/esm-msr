# gpu4 inference ledger (runtime comparison)

What inference ran where in October, so a runtime comparison can use like-for-like numbers later. **gpu4** is the
cluster's RTX A4500 node (4 × 20 GB, Ampere: native bf16); **gpu1–gpu3** carry RTX 6000s (24 GB, Turing: no native
bf16, so bf16 runs about 4× slower there and fp32 is about 1.5× faster than bf16). Generated from `sacct` with
`analysis_scripts/benchmark_matrix.py ledger`; every job, with node, start and elapsed time, is in
[`benchmarks/slurm_jobs_oct2026.csv`](benchmarks/slurm_jobs_oct2026.csv). Smoke tests, scoring-only CPU jobs and
failed attempts are left out of the table below but kept in the CSV.

## 1. What already ran, and where

"jobs gpu4 / other" counts Slurm jobs; "wall-hours" is the sum of job elapsed time (one GPU per job, except
MutateEverything training, which used three). A job's wall time includes model loading and, for the gamut jobs,
every dataset in the job, so per-dataset timing has to come from the finer sources in §3.

| model | benchmark | kind | where | jobs gpu4 / other | wall-hours gpu4 / other |
|---|---|---|---|---|---|
| ESM-MSR capped | gamut (one job per block) | inference | not gpu4 | 0 / 4 (gpu2) | 0.0 / 8.4 |
| ESM-MSR capped | GB1 rerun | inference | **gpu4 only** | 1 / 0 | 0.2 / 0.0 |
| ESM-MSR capped | ProteinGym full | inference | mixed | 1 / 1 (gpu3) | 31.8 / 11.8 |
| ESM-MSR capped | ProteinGym WT-only ε sweep | inference | not gpu4 | 0 / 2 (gpu3) | 0.0 / 0.6 |
| ESM-MSR dc8 | ProteinGym (ε sweep) | inference | mixed | 12 / 12 (gpu3) | 401.1 / 0.3 |
| ESM-MSR dc8 | ProteinGym BRCA1/SCN5A retry | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 2.6 |
| ESM-MSR dc8 | gamut | inference | not gpu4 | 0 / 3 (gpu1, gpu3) | 0.0 / 22.7 |
| ESM-MSR dc8 | ProteinGym MT-pass ablation | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 15.8 (cancelled at 26 of 213 DMS) |
| ESM-MSR-small (March) | 12 external sets, AF2 structures | inference | **gpu4 only** | 1 / 0 | 1.0 / 0.0 |
| MutateEverything | Tsuboyama val/test + externals | inference | mixed | 3 / 16 (gpu1, gpu2, gpu3) | 5.1 / 14.3 |
| MutateEverything | DMS | inference | mixed | 3 / 17 (gpu1, gpu2, gpu3) | 0.1 / 1.2 |
| MutateEverything | Domainome | inference | mixed | 8 / 4 (gpu1, gpu3) | 2.3 / 1.4 |
| MutateEverything | ProteinGym | inference | mixed | 1 / 7 (gpu3) | 5.3 / 31.8 |
| MutateEverything | training (st1+st2) | training | mixed | 1 / 6 (gpu1, gpu2, gpu3) | 6.2 / 43.6 |
| ThermoMPNN / -D | Tsuboyama val/test + externals | inference | mixed | 6 / 8 (gpu1, gpu3) | 0.4 / 0.8 |
| ThermoMPNN / -D | DMS | inference | mixed | 8 / 1 (gpu3) | 4.8 / 0.7 |
| ThermoMPNN / -D | Domainome | inference | mixed | 4 / 2 (gpu1, gpu3) | 1.6 / 0.9 |
| ThermoMPNN / -D | ProteinGym | inference | mixed | 1 / 2 (gpu3) | 12.0 / 10.5 |
| ThermoMPNN / -D | training | training | mixed | 6 / 3 (gpu1, gpu3) | 60.1 / 26.9 |
| SPURS | ProteinGym | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 0.1 |
| ProteinMPNN | ProteinGym | inference | not gpu4 | 0 / 3 (gpu3) | 0.0 / 5.9 |
| ProteinMPNN | ProteinGym TTA pilot (73 DMS) | inference | not gpu4 | 0 / 2 (gpu3) | 0.0 / 6.6 |
| ESMtherm | gamut (22 sets, hyperopt Tsuboyama) | inference | **gpu4 only** | 2 / 0 | 0.2 / 0.0 |
| IFUM (WT backbone) | gamut (22 sets) | inference | **gpu4 only** | 1 / 0 | 5.8 / 0.0 |
| IFUM (ESMFold per variant) | gamut scoring on folded structures | inference | mixed | 8 / 3 (gpu2) | 26.9 / 9.9 |
| IFUM (ESMFold per variant) | ESMFold folding of variant structures | structure prep | mixed | 14 / 8 (gpu2) | 81.8 / 22.5 |
| GeoDDG-3D (FASPR) | gamut (22 sets) | inference | mixed | 2 / 1 (gpu2) | 4.2 / 1.4 |
| GeoDDG-3D (FoldX) | 12 external sets | inference | mixed | 1 / 1 (gpu3) | 0.2 / 0.2 |
| GeoDDG-3D (FoldX) | FoldX structure building (CPU) | structure prep | not gpu4 | 0 / 3 (gpu2, gpu3) | 0.0 / 15.7 |
| ESM3ΔG | gamut (22 sets, sharded) | inference | not gpu4 | 0 / 5 (gpu3) | 0.0 / 14.4 |
| ProSST | gamut (22 sets) | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 0.6 |
| ColabFold AF2 | WT structures for external sets | structure prep | **gpu4 only** | 6 / 0 | 24.5 / 0.0 |
| SPURS | rerun with the mutant-background libraries (capped val+test) | rerun | **gpu4 only** | 1 / 0 | 0.02 / 0.0 |
| SPURS multi | rerun with the mutant-background libraries (capped val+test) | rerun | **gpu4 only** | 1 / 0 | 0.01 / 0.0 |
| ProteinMPNN | rerun with the mutant-background libraries (capped val+test) | rerun | **gpu4 only** | 1 / 0 | 0.00 / 0.0 |
| ESM-MSR | rerun after the calibration-default change (multi-mutant sets) | rerun | gpu4 | 13 queued (1 running) | – |
| ESM-MSR | ctx head (WT + ~WT)/2, capped val+test | inference | gpu4 | 4 queued | – |
| ESM-MSR | ProteinGym bias correction (CPU, offline) | cpu | not gpu4 | 0 / 4 (gpu1) | 0.0 / 1.3 |

The three **rerun** rows are not repeats of the gamut jobs they replace: they score the 6,026 + 7,931 mutant-background rows the
first runs skipped, so they do more work and must not be summed with the originals. `benchmark_matrix.py ledger` keeps them
in their own kind for that reason. The four CPU jobs are the ProteinGym bias correction; three of them stopped on a built-in
guard (that is what the guards are for) and were rerun after the fix.

How to read the mixed rows:

- **ESM-MSR dc8, ProteinGym:** every main run (3 seeds × 4 ε) ran on gpu4; the 8 gpu3 jobs are 2-minute `_rest`
  tails for the large structures, plus the BRCA1/SCN5A allocator retry. Use these as the gpu4 ESM-MSR timing.
- **ESM-MSR capped, ProteinGym:** 383708 ran on gpu4 for 31.8 h and the remaining DMS finished on gpu3 in fp32
  (383932); the per-DMS `summary.csv` has no gpu column for this run, so which DMS ran where has to come from the
  two job logs (`logs/pgym_msr_a4500-383708.out`, `logs/pgym_msr_fp32-383932.out`).
- **ESM-MSR capped and dc8, thermostability gamut:** never on gpu4 (gpu2, gpu1, gpu3).
- **MutateEverything / ThermoMPNN:** seeds were spread over whatever was free; within one benchmark the seeds ran
  on different GPU types. Seed 1 of ThermoMPNN's ProteinGym run (383742) timed out at 12 h on gpu4 and was
  completed elsewhere.
- **Third-party harness models:** ESMtherm and IFUM (WT backbone) ran entirely on gpu4, but IFUM shared the node
  with GeoDDG and ESMFold jobs; ESM3ΔG and ProSST ran entirely on gpu3; GeoDDG mostly on gpu4.
- **DDGemb and DDMut** ran on the authors' web servers (Bologna, UQ; the capped-split DDGemb stream started
  2026-10-21 logs per-job wall time to `closed_servers/ddgemb_state_capped/timings.csv`): their times measure queueing on a remote
  service and are not comparable with anything here.

## 2. gpu4 queue (as of 2026-10-23)

All pinned with `--nodelist=gpu4`; there are four cards, so lanes are chained with `afterany` and only the heads are
eligible at once. Each harness job writes one line per dataset to
`stability-baselines/results/<model>/predictions[_tag]/timing.csv` (rows, wall seconds, ms/row, host, GPU, job).

**Finished, with their wall time** (a clean per-job figure, not alone on the node):

| job | what | wall |
|---|---|---|
| 386515, 386516 | IFUM, ESMFold folding of 64,008 capped-Tsuboyama sequences (two workers) | 2:58 each |
| 386517, 386518 | IFUM native scoring, capped val / test | 1:22 / 1:55 |
| 386519 | SPURS multi, 22-set gamut | 0:07 |
| 386520–386529 | ESM-MSR ε sweep, DMS + Domainome (released ε = 0.75 / 0.5 / 0.25; devel seeds 1–3 at the intermediate ε) | 1:27–1:28 each |
| 386549–386551 | SPURS, SPURS multi and ProteinMPNN rerun with the mutant-background libraries, capped val / test | 0:55, 0:37, 0:12 |
| 386635 | MT-pass ablation, real-pipeline check on 3 DMS (§ the benchmarks doc, caveat 12) | 0:58 |
| 386532, 386533 | ESMtherm CAS9 retry; ProSST capped Tsuboyama | 0:07, 0:01 |

The ε runs took 1.5 h, not the ~6.6 h I had estimated from the gamut's earlier totals: those included the Tsuboyama
blocks, which these runs skip. **Convention:** jobs 386520–386527 ran before the calibration-default change (merged on the
cluster at 12:10 on 2026-10-23) and used one bias per item; 386528 onward use one per mutated position. The older ones are
being rerun (`gpu4_pmb_*`), so a seed mean over mixed jobs is never shown (the table builder reads the convention from each
prediction file). **Running** (as of 14:40): 386364 ESM3ΔG (fp16), 386277 GeoDDG, 386530 (devel seed 3, ε = 0.5) and 386576
(the first rerun, released ε = 1). **Waiting:** the other 12 `gpu4_pmb_*` reruns, the four `gpu4_ctx_*` jobs, and
`gpu4_eps0.25_dev3` (386531).

**A free GPU does not mean a free slot.** The ε jobs asked for 96 G and 8 CPUs but peak at 6.9 G. With 160 G of the
node's 250 G already allocated, the next job could not fit and a card sat idle until the pending jobs were lowered to
24 G. The same applies to every harness job: `run_model.sbatch` asks for 32 G, which is what ESM3ΔG and GeoDDG
actually hold.

Cancelled earlier: `gpu4_ifum_cappg` 386275 (§3.2), `gpu4_esm3dg_cappg` 386274 (replaced by 386364, §3.1),
`gpu4_prosst_cappg` 386278 (ProteinGym part replaced by the published scores).

Not on this cluster: **Rosetta** `cartesian_ddg` ran on the Alliance's Narval (CPU, 8,775 array chunks) in a separate
session, so it has no gpu4 time; its predictions were copied here afterwards. Another user's pending job can take a
freed A4500 first (the scheduler is FIFO with backfill).

## 3. Inference precision: ESM3ΔG and IFUM (measured 2026-10-22, A4500)

ESM3ΔG was the slowest model in the campaign: one full ESM3-small forward (1.4 B params) per variant per ensemble
member, so cost is variants × length × 3, and the remaining ProteinGym work projected to **287 h**. What the
profiling found, on the adapter's own `--dtype` switch (`logs/bench_esm3dg-386362.out`,
`logs/verify_esm3dg-386363.out`; 3-member ensemble, the production config):

| dtype | PIN1, L=39, 400 variants | HIV, L=852, 60 variants | max abs ddG change vs fp32 | ρ vs fp32 | ρ vs the DMS scores |
|---|---|---|---|---|---|
| fp32 (authors' default) | 1.00× | 1.00× | — | — | 0.8248 / 0.1081 |
| tf32 | 2.27× | 1.63× | 0.0011 | 0.99999 | 0.8248 / 0.1064 |
| bf16 | 3.33× | 2.37× | 0.031 | 0.9998 | 0.8248 / 0.1139 |
| **fp16** | **3.26×** | **2.38×** | **0.0046** | **0.99999** | **0.8249 / 0.1066** |

- **Batch size does not change throughput.** 3 → 28 rows per forward measured the same (211.7 vs 215.9 ms/variant at
  L=852), so the model is compute-bound and `--max_tokens` is a memory knob only. It is left at 3000, which keeps
  batch = 1 for the longest structure (L=2832).
- **TF32 matmuls are off by default in torch 2.x**, which is where the easy 1.6–2.3× was hiding.
- **fp16 is the choice, not bf16**: same speed, but 7× closer to fp32 on short proteins (0.0046 vs 0.031) and
  ρ vs fp32 0.99999 against 0.9981 on the long one. bf16 trades mantissa bits for range, and the range was never
  the problem — no non-finite prediction appeared in any setting.
- The Spearman that is actually reported does not move: 0.8248 → 0.8249 on PIN1. On HIV the ρ is ~0.11 for every
  dtype, and the ±0.006 spread there is resampling noise on a near-zero correlation over 60 variants, not a
  precision effect (the predictions themselves agree to 2.6e-4).
- **Precision mix in the ESM3ΔG row:** the external, DMS, Domainome and capped-Tsuboyama numbers, and the first 13
  ProteinGym DMS, were produced in fp32; the rest of ProteinGym is fp16. Per-DMS ρ is unaffected at this level of
  agreement, so the columns are comparable, but the mix is recorded here rather than left implicit.

### 3.1 ESM3ΔG

Two findings that mattered more than precision:

- **12 h of the 287 was being spent on DMS excluded from every comparison** — BRCA1, BRCA2, POLG_CXB3N and SCN5A,
  all > 1,800 residues (§1.4 of the benchmarks doc drops them because at least one model cannot fit them).
- **The cost is extremely skewed.** Half the DMS account for 1.8 % of the total and 90 % for 26 %; the *Stability*
  category — the relevant one for a stability model — is 6 h of the 287, while `SPG1_STRSG_Olson_2014`
  (536,962 variants, L=448) alone is about 84 h. The run was alphabetical, so it was spending its first days on
  large viral fitness assays and would have delivered the stability numbers last. The replacement job runs the 213
  comparable DMS **cheapest-first** (`harness/lists/pgym_esm3dg_bycost.txt`).

### 3.2 IFUM: the same patch, and why it did not pay (2026-10-22)

IFUM looked like the same opportunity and was not. Its adapter loaded `prot_t5_xl_half_uniref50-enc` with
`T5EncoderModel.from_pretrained(T5)` and no `torch_dtype`, which **upcasts** the released fp16 weights to fp32
(HuggingFace ignores the checkpoint's own `torch_dtype` unless asked), and nothing set TF32. Passing the dtype
through is a real fix. It is just not where IFUM's time goes.

Measured (`logs/verify_ifum-386365.out`, `logs/verify_ifum_L448-386366.out`), fp16 vs fp32, `--struct wt`:

| protein | L | fp16 speedup | max abs ddG change | ρ vs fp32 | ρ vs the DMS scores |
|---|---|---|---|---|---|
| PIN1 | 39 | **2.71×** | 0.0019 | 0.999993 | 0.8280 → 0.8280 |
| HIS7_YEAST | 220 | 1.44× | 0.0048 | 1.000000 | 0.5202 → 0.5202 |
| SPG1_STRSG (the costliest DMS) | 448 | 1.28× | 0.0047 | 0.999889 | 0.0994 → 0.1016 |
| HIV | 852 | 1.13× | 0.0053 | 0.999500 | 0.2905 → 0.2864 |

**The speedup decays with length** because only part of IFUM's per-variant cost is ProtT5. The IEFFEUM head does
O(L²) pairwise work — it builds an `L×L` mask per variant — in fp32, and that term dominates as L grows. ESM3ΔG,
one big transformer forward per variant, gains at every length; IFUM does not.

And IFUM's remaining ProteinGym work is almost all long proteins:

| L | DMS remaining | share of remaining cost |
|---|---|---|
| < 100 | 69 | 1 % |
| 100–200 | 25 | 4 % |
| 200–400 | 44 | 23 % |
| 400–600 | 34 | 49 % |
| > 600 | 33 | 23 % |

Cost-weighted mean L = 488, and only 1 % of the cost sits where fp16 helped. Combining the per-length speedups by
cost share gives **≈1.28× overall**: 315 h → ~246 h, still ~10 days of one A4500. **Job 386275 was cancelled**
after 15:51 h (8 of 217 ProteinGym DMS; its capped-Tsuboyama numbers were already finished and are kept). The
`--dtype` patch is kept because it is correct and defaults to fp32, but it does not justify a restart.

The lesson, which cost a wrong call here: measure a speedup at the **cost-weighted** length. A 39-residue test
protein overstated this one by a factor of two. The cheap levers (skipping datasets every comparison excludes, and
ordering cheapest-first) were worth more than precision for both models.

## 4. Where per-dataset times are recorded

| source | granularity | covers |
|---|---|---|
| `esm-msr-devel/outputs/pgym/<model>/<setting>/summary.csv` | per DMS: `time_s`; `peak_alloc_gb`, `peak_reserved_gb`, `gpu`, `host` from 2026-10-17 on | every ProteinGym run (gpu/host/VRAM columns: dc8 runs only) |
| `stability-baselines/results/<model>/predictions*/timing.csv` | per dataset: rows, wall s, ms/row, host, GPU, job | third-party harness runs (ESMtherm, IFUM ESMFold, GeoDDG FoldX, ESM3ΔG, ProSST, and every queued job) |
| `stability-baselines/results/TIMING.md` | per model × dataset, hand-compiled, with concurrency notes | the six original third-party models |
| `esm-msr-devel/analysis_notebooks/stats/<set>/<ckpt>/…_Combined.csv`, column `time` | per dataset (seconds) | ESM-MSR gamut runs |
| Slurm (`slurm_jobs_oct2026.csv`) | per job | everything, including ME / ThermoMPNN, which log no per-dataset time. Complete record (418 jobs, smoke tests and failures included; `kind` separates them) |
| `timing_cache_flags.csv` | per dataset | the 49 ESMtherm datasets whose `wall_s` is a cache hit, not inference (§4a) |

### 4a. What auditing these records found (2026-10-23; `analysis_scripts/runtime_log_audit.py ledger|timing`)

Consistent, checked on all of them:

- **Slurm:** `Elapsed` agrees with `End − Start` for every finished job (0 disagreements).
- **Harness `timing.csv`** (22 files, 742 rows): the summed per-dataset time never exceeds the job's `Elapsed` (45 jobs, 0 violations);
  the host in the file is the node Slurm reports (0 mismatches). `ms_per_row` is recomputed from a `wall_s` rounded to 0.1 s, so it
  is coarse for short datasets (error up to 0.05 s ÷ rows): do not read ms/row below about a second of wall time.

Not what it looks like:

- **ESMtherm's ProteinGym times include cache hits.** The adapter keeps one sequence cache for the whole process, so in a job that
  scores several datasets a sequence seen earlier costs nothing: in job 386276, 49 of 218 ProteinGym datasets show `wall_s` under a
  second (2,350 variants in 0.0 s), 25 of them Tsuboyama-origin DMS whose sequences the capped Tsuboyama block had just scored.
  The job total (4.28 h of loop time in 4.31 h elapsed) is right; the per-dataset figures for those 49 are not inference time.
  They are listed in `timing_cache_flags.csv`. GeoDDG (per structure-and-mutation cache), IFUM (per-group structure and ddG
  caches) and ProSST (one entry) also cache; none shows near-zero rows, and ESM3ΔG has no cross-dataset cache, but a structure
  scored twice in one job would be cheaper the second time.
- **A dataset name that appears twice is either a shard or a rerun, and they must be treated oppositely.** Shards (ESM3ΔG and IFUM
  Domainome, 2 and 6 rows) have different `n_rows` and are **summed**; reruns have the same `n_rows` and the **latest row wins**.
  The reruns are ESMtherm CAS9 (386276 failed in 0.5 s with every prediction NaN; 386532 is the real 396.5 s), and capped val / test for
  ProteinMPNN, SPURS (hybrid readout) and SPURS multi (the earlier row skipped the mutant-background libraries: `n_nan` 6,026 / 7,931).
- **One rerun wrote to the wrong directory.** `gpu4_spurs_bb` (386549) went to `results/spurs/predictions""` (literal quotes): an
  empty `--tag ""` passed through `sbatch --export` arrives as two quote characters. The table builder therefore kept reading the old
  NaN file until this audit. The files were validated (no NaN, wild-type-background rows bit-identical to the old ones, the
  mutant-background libraries at Spearman 0.70–0.87) and installed into `predictions/`; the stray directory is kept as
  `predictions_MISNAMED_empty_tag_job386549_installed_into_predictions`. `run_model.py` now refuses any `--tag` outside
  `[A-Za-z0-9_.-]` (backup `run_model.py.bak_tagguard`).
- **Job-name patterns can silently merge a rerun into the run it replaces.** The three `*_bb` jobs would have matched the generic
  `gpu4_spurs` / `gpu4_mpnn` patterns and been added to the original gamut rows. They now have their own patterns ahead of those, and
  `benchmark_matrix.py ledger` keeps unmatched jobs (as `smoke` or `other`) instead of dropping them, which the earlier text of
  this document claimed it already did.
- **ESM-MSR ProteinGym `summary.csv`** keeps `time_s`, `gpu` and `host` after the bias correction (only `spearman_combined` changed,
  with the old value kept in `spearman_combined_per_item_bias`), so those timings are unaffected.

## 5. What a clean runtime comparison still needs

Production runs shared nodes and mixed GPU types, so their wall times are an upper bound, not a benchmark. For a
like-for-like table, time each model **alone on one A4500** on a fixed subset (for example S669, the Tsuboyama
capped test split, GB1, and ten ProteinGym DMS spanning lengths 50–1,000), excluding model loading and structure
preparation, which are listed separately above (ESMFold folding for IFUM-native: 104 wall-hours; FoldX building
for GeoDDG: 15.7 CPU-node-hours; AF2 for the external WT structures: 24.5 h). Not yet timed on gpu4 in a controlled way: ESM-MSR on the thermostability gamut (the ε = 0 gamut does the same
forward passes as ε = 1, so its times stand in for it). ESM3ΔG, ProSST, SPURS and ProteinMPNN have now run there,
but alongside other jobs on the same node, and ESM3ΔG's ProteinGym half is fp16 while its other datasets are fp32
(§3) — so those wall times are not comparable with each other either.

Two logging gaps remain and are cheap to close in `run_model.py`'s `log_timing`: a **run id** (so a rerun and its predecessor are
distinguishable without reading job ids) and a **cache-hit count** (rows served without computation). Until then, take timings
for a model from datasets that were the first to touch their sequences in the job, and exclude the rows in
`timing_cache_flags.csv`.
