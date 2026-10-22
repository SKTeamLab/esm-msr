# Outstanding experiments (benchmarks)

What still has to run, or cannot run yet, for the all-by-all comparison in
[`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) §2. Each entry says what it fills, why it is not done, and what
unblocks it. Last updated 2026-10-22. Where things ran: [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md).
Cluster paths are on `sreeves@subhak-gpu.uwaterloo.ca`.

## 1. Queued — no decision needed

| what | fills (the `q` cells) | jobs (gpu4) |
|---|---|---|
| ESM3ΔG, IFUM, ESMtherm, GeoDDG, ProSST on the **capped Tsuboyama** val/test | §2.2 capped columns | **done** for ESM3ΔG, IFUM, GeoDDG; ESMtherm 386276 and ProSST 386278 still queued behind 386275 / 386277 |
| the same five models on **ProteinGym** | §2.4 | in progress: ESM3ΔG 386364 (fp16, cost-ordered, ~half done), GeoDDG 386277, ESMtherm 386276 started 2026-10-22, ProSST 386278 queued. **IFUM cancelled** — see below |
| ESM3-small-open zero-shot (ESM-MSR capped at ε = 0) on the 22-set gamut | §2.1–2.3 | **done** (386282) |
| SPURS and ProteinMPNN on the 22-set gamut | §2.1–2.3 | **done** (386280, 386281) |

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

| what | state (2026-10-22) | next |
|---|---|---|
| ~~dc8 ProteinGym ε sweep~~ | **complete 2026-10-22**: 3 seeds × 4 ε, all 217 DMS. §4 now carries the paired table; ε = 0.75 is the only setting that beats capped | still to do: write a combined dc8 per-DMS CSV into `outputs/pgym/` (the cross-model `pgym_all_models_per_dms.csv` has no dc8 columns) |
| **ESM3ΔG ProteinGym**, restarted 2026-10-22 as 386364 (`--dtype fp16`, the 213 comparable DMS cheapest-first) after 386274 was cancelled at 13/217 | fp16 is 2.4–3.3× faster with ρ unchanged, and the 4 over-length DMS are no longer run: ~287 h → ~100 h, with the Stability category inside the first ~2 h and 90 % of DMS in ~1 day (`GPU4_INFERENCE_LEDGER.md` §3) | nothing; the earlier fp32 DMS are kept (they agree to ≤ 5e-4) |
| ~~MT-pass ablation (`--mt_pass_adapter wt`, dc8 s1 ε = 1), 386255~~ | **cancelled** 2026-10-21 after 15:46 h at ~26/213 DMS; the partial output is in `outputs/pgym/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s1/sigma1.0_mtpass_wt/` | resubmit on an A4500 (~33 h) if the ablation is still wanted |
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
| **IFUM on ProteinGym** | the IFUM row of §2.4 | ~246 h (~10 days of one A4500) even after the fp16 patch: the speedup is only 1.13–1.28× at the lengths that carry 95 % of the cost, because the IEFFEUM head's O(L²) term runs in fp32 and dominates (`GPU4_INFERENCE_LEDGER.md` §3). Cancelled 2026-10-22 at 8/217 to free the GPU | a decision that it is worth the time, or a fix to the head's precision rather than ProtT5's. **Cheap partial:** the Stability category alone is 66 DMS / ~5.4 h, and everything under 200 residues is 94 DMS / ~12.5 h — either would fill the most meaningful part of the row and then release the node |
| **SPURS (and the other Megascale-trained models) on a homolog-filtered Tsuboyama split** | makes the §2.2 capped column interpretable, where SPURS currently leads at 0.874 | SPURS trains on Megascale, so the capped split is not held out for it. `esm_msr_testing.py --remove_spurs_homologs` exists for exactly this, but the harness adapters do not apply it, so the baselines are scored on the unfiltered split | run the capped tables through that filter and rescore every ‡ model on the surviving libraries. Until then the SPURS, ESM3ΔG, IFUM and ESMtherm Tsuboyama numbers are upper bounds, not comparisons |
| SPURS and ProteinMPNN on mutant-background Tsuboyama libraries | 5 of 25 val and 7 of 37 test libraries (all singles) | both models read the sequence from the wild-type PDB, which has the wild-type residue at the background position; the adapters return NaN rather than score the wrong sequence | an adapter that substitutes the background residue in the parsed structure; minor, since these rows are all singles |

## 4. Analysis to do once §1 lands (CPU)

- **Epistasis suite for the third-party models** on capped test and valfull (`beyond`, `pair`, `line`, `cell_*`,
  `global_err`). The harness writes `pred` and `pred_additive`, which is all `epi_metrics.Table` needs; adapt
  `esm-msr-devel/baselines/epi_baselines.py` to read `stability-baselines/results/<model>/predictions/splits_oct06_capped-*.csv`.
- **ProteinGym per-category table (§4) for the new models**, and the bootstrap significance tests
  (`analysis_notebooks/bootstrap_external.py`) with the new rows included.
- **A clean runtime table** on one A4500 per model ([`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md) §4).

## 5. Housekeeping

- The third-party harness (`~/stability-baselines/harness`) is not under version control. Added 2026-10-21:
  `adapters/spurs.py`, `adapters/proteinmpnn.py`, `build_pgym_tasks.py`, `lists/`. Both new adapters reproduce the
  existing ProteinGym outputs exactly (max |Δ| = 0 on PIN1 and ZIKV). Added 2026-10-22: `--dtype` on
  `adapters/esm3dg.py` (previous version kept as `esm3dg.py.bak`), plus `bench_esm3dg.py` and
  `verify_esm3dg_dtype.py`. **Worth copying into the repo** if the harness is ever to be reproducible from it.
- `~/esm-msr-devel` is behind `origin/devel-revision`; pull it once no queued job reads it (the dc8 `_rest` tails and
  386282 do). Its staged `data/preprocessed/*_mapped.csv` changes only rewrite structure paths for the cluster: keep
  them out of any commit.
- Awaiting approval to delete: `~/esm-msr-rev` (old work clone) and `~/esm-msr-devel/outputs/pgym_verify_rev/`.
- A GitHub token pasted into a cluster chat on 2026-10-17 is stored in that session's transcript: revoke it.
