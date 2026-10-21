# Outstanding experiments (benchmarks)

What still has to run, or cannot run yet, for the all-by-all comparison in
[`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) §2. Each entry says what it fills, why it is not done, and what
unblocks it. Last updated 2026-10-21. Where things ran: [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md).
Cluster paths are on `sreeves@subhak-gpu.uwaterloo.ca`.

## 1. Queued — no decision needed

| what | fills (the `q` cells) | jobs (gpu4) |
|---|---|---|
| ESM3ΔG, IFUM (WT backbone), ESMtherm, GeoDDG-3D (FASPR), ProSST on the capped Tsuboyama val/test and all 217 ProteinGym DMS | §2.2 capped columns, §2.4 | 386274, 386275 → 386276, 386277 → 386278 |
| ESM3-small-open zero-shot (ESM-MSR capped at ε = 0) on the 22-set gamut | §2.1–2.3 | 386282 |
| SPURS and ProteinMPNN on the 22-set gamut | §2.1–2.3 | 386280, 386281 (after 386282) |

**When they finish** (run on the cluster; CPU, about a minute):

```bash
cd ~/consolidation && python benchmark_matrix.py collect --out matrix_long.csv && python benchmark_matrix.py tables matrix_long.csv > tables.md
```

then paste the tables into §2, copy `matrix_long.csv` to `docs/benchmarks/all_by_all_long.csv`, and drop the jobs from
the `QUEUED` dict in `analysis_scripts/benchmark_matrix.py` (a filled cell already overrides `q`). Check first:
any DMS that failed is in the job log as `FAILED`, and `pg_summaries` refuses to summarise a model with fewer than the
213 common DMS, printing how many it has. The harness skips datasets whose output already exists, so a failed job
can be resubmitted as is.

## 2. In flight

| what | state (2026-10-21 12:30) | next |
|---|---|---|
| dc8 ProteinGym ε = 0.5 (s3) and ε = 0.25 (s1–s3): 384287/89/91/93 on gpu4, then their `_rest` tails 384288/90/92/94 on gpu3 | 177/213 each; on SPG1 (537k variants) | when all are at 213: add the dc8 ε rows to §4, and write a combined dc8 per-DMS CSV into `outputs/pgym/` (the cross-model `pgym_all_models_per_dms.csv` has no dc8 columns) |
| MT-pass ablation (`--mt_pass_adapter wt`, dc8 s1 ε = 1): 386255 on gpu3 | 24–26/213; about 6 days at RTX 6000 bf16 speed, so the 3-day limit stops it around halfway | **decision:** cancel and resubmit pinned to an A4500 (~33 h, after the queued jobs), or let it run and queue a follow-on job |
| **DDGemb on the capped Tsuboyama split** (web server, started 2026-10-21 on the login node: `DDGEMB_STATE=ddgemb_state_capped python ddgemb_client.py run`, log `closed_servers/ddgemb_capped_run.log`) | 63,924 new requests in 32 jobs of ≤ 2,000 (requests answered by earlier streams are not re-sent); one job at a time, ~14 min each, so ~7–8 h | then `python ddgemb_client.py assemble --datasets splits_oct06_capped-val,splits_oct06_capped-test` writes `results/ddgemb/predictions/splits_oct06_capped-*.csv`, which `benchmark_matrix.py collect` scores |
| DDMut on the hyperopt Tsuboyama doubles (server stream, session `796d0b37`) | 6,861 + 8,321 of ~20,151 | let it finish; DDMut's Tsuboyama cells stay partial (`*`) until then |

## 3. Blocked

| what | fills | why blocked | unblocks it |
|---|---|---|---|
| **ESM-MSR on the four largest ProteinGym DMS** (BRCA1_HUMAN_Findlay_2018, BRCA2_HUMAN_Erwood_2022_HEK293T, POLG_CXB3N_Mattenberger_2021, SCN5A_HUMAN_Glazer_2019; > 1,800 residues) | full 217-DMS ProteinGym for ESM-MSR; today every model is compared on 213 | out of memory on 24 GB (RTX 6000) and 20 GB (A4500) even at batch size 1. dc8 s1 ε = 1 alone got BRCA1 and SCN5A through an allocator retry. Needs the local 32 GB 5090, which the training queue is using (28.8 of 32.6 GB in use on 2026-10-21) | a free 5090. Inputs exist locally (`playground/esm-msr-devel/pgym_inputs`, `LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`); the dc8 checkpoints are on the cluster under `LoRA_models/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}/` (copy with their `MANIFEST.txt`). Runs: capped full (4 DMS); capped WT-only ε ∈ {0, 0.25, 0.5, 0.75, 1} (BRCA2, POLG, SCN5A); dc8 3 seeds × ε ∈ {1, 0.75, 0.5, 0.25} (s1 ε = 1 needs only BRCA2, POLG). Use `--batch_size 1 --dtype bf16` with the default allocator, as the `slurm_pgym.sh` second pass does |
| MutateEverything on BRCA2 and SCN5A (ProteinGym) | ME at 217 DMS | out of memory at the cluster's card sizes | same 5090 window; low priority, since all comparisons use the 213 common DMS anyway |
| DDGemb on ProteinGym | §2.4 | 2.4 M variants ≈ 10 days of server time | a decision that it is worth it; otherwise reported as not run |
| DDMut on capped Tsuboyama, DMS, Domainome, ProteinGym | the DDMut row | REST API at 20–30 s per variant: the capped split alone is weeks | not planned; reported as not run |
| UniMutStab | the whole row | no released weights or inference code (pseudocode only); reimplementation ruled out on 2026-10-14 | weights from the authors |
| IFUM in ESMFold-per-variant mode on capped Tsuboyama and ProteinGym | the IFUM-native row | ESMFold folding is ~3.9 s per variant sequence: ~95 GPU-h for capped, ~2,600 GPU-h for ProteinGym | not planned; the WT-backbone mode is the reported IFUM |
| GeoDDG with FoldX structures beyond the 12 external sets | — | FoldX BuildModel is CPU-bound (~150–250 CPU-h for the externals); FASPR is the stand-in and gives GeoDDG predictions within 3.5e-4 kcal/mol of FoldX on the externals and ρ = 0.998 with the authors' S669 predictions | not needed |
| ESM3 small / medium / large via the Forge API | ESM3 Forge rows (small has no external or DMS numbers; medium and large are missing S461, K2369, PTMUL-D) | paid, rate-limited API; the March runs were never completed | a Forge key and budget. ESM3-small-open (ε = 0, queued) covers the open model locally |
| SPURS and ProteinMPNN on mutant-background Tsuboyama libraries | 5 of 25 val and 7 of 37 test libraries (all singles) | both models read the sequence from the wild-type PDB, which has the wild-type residue at the background position; the adapters return NaN rather than score the wrong sequence | an adapter that substitutes the background residue in the parsed structure; minor, since these rows are all singles |

## 4. Analysis to do once §1 lands (CPU)

- **Epistasis suite for the third-party models** on capped test and valfull (`beyond`, `pair`, `line`, `cell_*`,
  `global_err`). The harness writes `pred` and `pred_additive`, which is all `epi_metrics.Table` needs; adapt
  `esm-msr-devel/baselines/epi_baselines.py` to read `stability-baselines/results/<model>/predictions/splits_oct06_capped-*.csv`.
- **ProteinGym per-category table (§4) for the new models**, and the bootstrap significance tests
  (`analysis_notebooks/bootstrap_external.py`) with the new rows included.
- **A clean runtime table** on one A4500 per model ([`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md) §4).

## 5. Housekeeping

- The third-party harness (`~/stability-baselines/harness`) is not under version control. Three files were added on
  2026-10-21: `adapters/spurs.py`, `adapters/proteinmpnn.py`, `build_pgym_tasks.py` (plus `lists/`). Both new adapters
  reproduce the existing ProteinGym outputs exactly (max |Δ| = 0 on PIN1 and ZIKV).
- `~/esm-msr-devel` is behind `origin/devel-revision`; pull it once no queued job reads it (the dc8 `_rest` tails and
  386282 do). Its staged `data/preprocessed/*_mapped.csv` changes only rewrite structure paths for the cluster: keep
  them out of any commit.
- Awaiting approval to delete: `~/esm-msr-rev` (old work clone) and `~/esm-msr-devel/outputs/pgym_verify_rev/`.
- A GitHub token pasted into a cluster chat on 2026-10-17 is stored in that session's transcript: revoke it.
