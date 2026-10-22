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
| ESM-MSR dc8 | ProteinGym (ε sweep) | inference | mixed | 12 / 8 (gpu3) | 371.5 / 0.2 (running) |
| ESM-MSR dc8 | ProteinGym BRCA1/SCN5A retry | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 2.6 |
| ESM-MSR dc8 | gamut | inference | not gpu4 | 0 / 3 (gpu1, gpu3) | 0.0 / 22.7 |
| ESM-MSR dc8 | ProteinGym MT-pass ablation | inference | not gpu4 | 0 / 1 (gpu3) | 0.0 / 12.2 (running) |
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

## 2. Queued on gpu4 (2026-10-21)

All pinned with `--nodelist=gpu4`, in four chains so at most four hold an A4500 at a time. Each harness job writes
one line per dataset to `stability-baselines/results/<model>/predictions[_tag]/timing.csv` (rows, wall seconds,
ms/row, host, GPU, Slurm job).

| job | model | datasets | runs after |
|---|---|---|---|
| ~~386274~~ `gpu4_esm3dg_cappg` | ESM3ΔG | capped Tsuboyama val/test (done, fp32) + ProteinGym | **cancelled** 2026-10-22 after 14:53 h, 13 of 217 DMS; replaced by 386364 |
| 386364 `gpu4_esm3dg_pg_fp16` | ESM3ΔG | the 213 comparable ProteinGym DMS, cheapest first, `--dtype fp16` | — |
| 386275 `gpu4_ifum_cappg` | IFUM (WT backbone) | same | — |
| 386276 `gpu4_esmtherm_cappg` | ESMtherm | same | 386275 |
| 386277 `gpu4_geoddg_cappg` | GeoDDG-3D (FASPR) | same | — |
| 386278 `gpu4_prosst_cappg` | ProSST | same | 386277 |
| 386282 `gpu4_msr_eps0_gamut` | ESM3-small-open (ESM-MSR at ε = 0) | 12 external, capped Tsuboyama, 7 DMS, Domainome | — |
| 386280 `gpu4_spurs_gamut` | SPURS (single + multi) | same 22 sets | 386282 |
| 386281 `gpu4_mpnn_gamut` | ProteinMPNN | same 22 sets | 386280 |

They start when the four dc8 ProteinGym jobs (384287/89/91/93) release gpu4, expected late 2026-10-21. Another
user's pending job can take a freed A4500 first (the scheduler is FIFO with backfill).

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
  all > 1,800 residues (§2.4 of the benchmarks doc drops them because at least one model cannot fit them).
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
| Slurm (`slurm_jobs_oct2026.csv`) | per job | everything, including ME / ThermoMPNN, which log no per-dataset time |

## 5. What a clean runtime comparison still needs

Production runs shared nodes and mixed GPU types, so their wall times are an upper bound, not a benchmark. For a
like-for-like table, time each model **alone on one A4500** on a fixed subset (for example S669, the Tsuboyama
capped test split, GB1, and ten ProteinGym DMS spanning lengths 50–1,000), excluding model loading and structure
preparation, which are listed separately above (ESMFold folding for IFUM-native: 104 wall-hours; FoldX building
for GeoDDG: 15.7 CPU-node-hours; AF2 for the external WT structures: 24.5 h). Not yet timed on gpu4 in a controlled way: ESM-MSR on the thermostability gamut (the ε = 0 gamut does the same
forward passes as ε = 1, so its times stand in for it). ESM3ΔG, ProSST, SPURS and ProteinMPNN have now run there,
but alongside other jobs on the same node, and ESM3ΔG's ProteinGym half is fp16 while its other datasets are fp32
(§3) — so those wall times are not comparable with each other either.
