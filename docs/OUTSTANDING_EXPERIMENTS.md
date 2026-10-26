# Outstanding experiments (benchmarks)

What still has to run, or cannot run yet, for the all-by-all comparison in
[`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) §1. Each entry says what it fills, why it is not done, and what
unblocks it. Last updated 2026-10-26. Where things ran and how long they took:
[`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md). Cluster paths are on `sreeves@subhak-gpu.uwaterloo.ca`.

## 1. Queued or running on gpu4 — no decision needed

Everything below is pinned to gpu4 (4 × A4500). Lanes are chained with `afterany`, so only the heads are eligible at once.

**Finished since 2026-10-22:** IFUM-native on capped Tsuboyama (fold 386515/386516, 3.0 h each; score 386517/386518,
1.4 and 1.9 h), SPURS multi 386519 (7.5 min), ProSST capped 386533, ESMtherm CAS9 retry 386532 (so ESMtherm now has all
213 ProteinGym DMS), and the released-checkpoint σ sweep at 0.75 / 0.5 / 0.25 (386520–386522, 1.5 h each) plus devel
seed 1 at σ = 0.75 (386523). All checked: no failed datasets, and the only NaN predictions are the 7,931 capped-test
(6,026 val) rows on mutant-background libraries, which the SPURS and ProteinMPNN adapters used to skip. They are
patched and rerun (386549–386551, done and validated 2026-10-23).

**Finished since 2026-10-23:** the devel σ sweep on DMS + Domainome is complete — all twelve runs (386523–386531,
~1.5 h each) landed, so the §1.3 devel rows are filled and `benchmark_matrix.py tables` renders them without a `q`
(regenerated 2026-10-24; the §1.3 prose now carries the devel means). The context-head runs first failed on a missing
structure (see the ctx row below) and were rerun successfully as `gpu4_ctx2_*` 386738–386741 (~10 min each). The calibration-default
reruns finished: 386581–386588 on the first pass, and the five that failed (386576–386580, `rel_e1` and `s1_e1`/`s2_e1`/
`s3_e1`/`rel_e0`) rerun as `gpu4_pmb2_*` 386727–386731 (DMS + Domainome, ~54 min each) and `gpu4_pmb3_tsu_*`
386733–386737 (Tsuboyama, ~10 min each). The mtpass ablation (386637) was still running at 177 / 213 DMS on 2026-10-24.

| what | fills (the `q` cells) | jobs | state |
|---|---|---|---|
| ProteinGym ESM-MSR rows after the calibration-default change | §1.4, §3 | **done offline 2026-10-23** by `analysis_scripts/pgym_per_mutation_bias.py` instead of ~250 GPU-hours of reruns; backup `outputs/pgym_pre_per_mutation_bias/`; validated on a real rerun of GCN4 / PIN1 / SAV1 (`outputs/pgym_verify_pmb/`) | the MT-pass ablation (26 DMS, devel s1) was not corrected and is not comparable; a full ProteinGym rerun would settle the remaining ~0.015 old-code WT offset on multi-mutants |
| ESM-MSR reruns after the calibration-default change (Tsuboyama, PTMUL, DMS; σ = 1 for all four checkpoints, σ = 0 released, σ sweeps) | Table 2.3, §1.2, §1.3, the epistasis tables | 13 jobs 386576–386588, then the reruns 386727–386731 and 386733–386737; 8.8 GPU-h completed, plus 2.6 GPU-h lost to the five first-pass failures | **done 2026-10-23.** Five of the thirteen failed on their first pass (386576–386580: `rel_e1`, `s1_e1`, `s2_e1`, `s3_e1`, `rel_e0`) and were rerun on both the DMS + Domainome and Tsuboyama tag sets; all completed. Seed means that mix old and new conventions are no longer hidden by `benchmark_matrix.py` |
| ESM-MSR ctx head, (WT + ~WT)/2, on capped val + test (released and devel seeds 1–3) | the "WT read in context" rows of §2.1 and §2.2 | `gpu4_ctx2_rel` 386738, `gpu4_ctx2_s1` 386739, `gpu4_ctx2_s2` 386740, `gpu4_ctx2_s3` 386741; ~10 min each (launcher `inference_scripts/esm_msr_ctx_testing.py`, outputs in `outputs/ctx_run/`, scored by `analysis_scripts/epi_ctx.py`) | **done 2026-10-23.** The first four attempts (386566–386569) died in 4.5 min on a missing structure, `data/tsuboyama/AlphaFold_model_PDBs/v2R31S:R32S_2N5D.pdb`; the colon-named file does not exist and the adapter now falls back to the `_` name. The reruns all completed. The §2.1/§2.2 rows are not yet written into the doc |
| **ESM-MSR LoRA scale σ ∈ {0.75, 0.5, 0.25}, DMS + Domainome, devel checkpoint**, one lane per seed | §1.3 σ table, devel rows (shown once all 3 seeds finish) | seed 1: 386524, 386525; seed 2: 386526, 386527, 386528; seed 3: 386529–386531 | **done 2026-10-23** — all twelve runs landed at ~1.5 h each, and the §1.3 devel rows are now filled (regenerated 2026-10-24) |
| ESM3ΔG, ProteinGym, 213 DMS cheapest first, fp16 | §1.4 | 386364, running | 202 / 213 at 2 d 5 h against a 5-day limit (2026-10-24 16:10); the remaining work is dominated by the largest DMS (SPG1, 537k variants) |
| GeoDDG (FASPR), capped Tsuboyama + ProteinGym | §1.2 capped cells, §1.4 | 386277 (killed), **386804 running** | **386277 was killed by its own two-day wall clock** at 148 / 213 — a scheduling limit (`--time=2-00:00:00`), not a failure of the code, and the partition allows unlimited time. Resumed 2026-10-24 as 386804 with `--time=4-00:00:00`; the harness logs `exists, skipping` for the 148 datasets already on disk, and it has since carried the model to **181 / 213** (2026-10-24 16:10) |
| DDMut on the old hyperopt Tsuboyama doubles (web server, login node) | DDMut's Tsuboyama cells stay partial (`*`) until done | two client processes | ~6,861 + 8,321 of ~20,151 done |

**Scheduling note (2026-10-23).** A card sat idle while the next σ job waited, because the node was out of *memory*, not
GPUs: the σ jobs requested 96 G (copied from the gamut script) and use about 6.9 G (`MaxRSS` on every completed run). Reduced
to 24 G on the pending jobs with `scontrol update JobId=… MinMemoryNode=24576` (the value is in MB, and a `G` suffix is
rejected), and the script's default lowered. When a GPU looks free and a job is pending on `Resources`, check
`scontrol show node gpu4 | grep AllocTRES` against the job's `--mem` and `--cpus-per-task` before suspecting the GPUs.

**When they finish** (CPU, about a minute, on the cluster):

```bash
cd ~/consolidation
python benchmark_matrix.py collect --out matrix_long.csv
python benchmark_matrix.py tables matrix_long.csv > tables.md
python benchmark_matrix.py epi > epi_tables.md
```

then replace the tables in §1 and §2.1–2.2, copy `matrix_long.csv` to `docs/benchmarks/all_by_all_long.csv`, and
remove the finished jobs from the `QUEUED` dict in `analysis_scripts/benchmark_matrix.py` (a filled cell already
overrides `q`). A model's ProteinGym summary appears only once all 213 common DMS are in; the script prints how many it
has. The harness skips datasets whose output already exists, so any failed job can be resubmitted as is.

## 2. ProteinGym: is every feasible model queued?

| model | status |
|---|---|
| ESM-MSR (released), (devel) × 4 σ, WT-only sweep, σ = 0 control | done on 213 DMS; the four > 1,800-residue DMS are blocked (§3) |
| MutateEverything, ThermoMPNN, SPURS (both models), ProteinMPNN | done |
| ESMtherm | done on all 213 (one DMS needed a rerun with an adaptive batch size) |
| ESM3ΔG | running (fp16), 202 / 213 |
| GeoDDG | running, 181 / 213 |
| ProSST | **done for free**: the authors' published ProSST-2048 scores ship with ProteinGym; its own job was cancelled |
| Rosetta Cartesian ΔΔG | done, but the file covers **single mutants only**, so it is shown in italics and not compared |
| IFUM | **declined**: ~10 days of an A4500 even in fp16, because the O(L²) IEFFEUM head dominates at the lengths that carry the cost |
| DDGemb, DDMut | not feasible: web servers (DDGemb ≈ 10 days of server time for 2.4 M variants; DDMut 20–30 s per variant) |
| UniMutStab | not possible: no released weights |
| ESM3 through Forge (small / medium / large) | not planned: paid, rate-limited API |

**Free and not yet used:** the same ProteinGym archive holds the authors' scores for ~90 other models, including
ESM-IF1, ProteinMPNN, SaProt, ESM-C, ESM3 (open), PoET, VenusREM, ProtSSN and the ESM-1v / ESM2 families. Adding any of
them as a reference row costs nothing but a column name in `analysis_scripts/pgym_public_scores.py`. Only ProSST was
asked for, so nothing else was added.

## 3. Blocked

| what | fills | why blocked | unblocks it |
|---|---|---|---|
| **ESM-MSR on the four largest ProteinGym DMS** (BRCA1_HUMAN_Findlay_2018, BRCA2_HUMAN_Erwood_2022_HEK293T, POLG_CXB3N_Mattenberger_2021, SCN5A_HUMAN_Glazer_2019; > 1,800 residues) | full 217-DMS ProteinGym for ESM-MSR; today every model is compared on 213 | out of memory on 24 GB (RTX 6000) and 20 GB (A4500) even at batch size 1. Devel seed 1 at σ = 1 alone got BRCA1 and SCN5A through an allocator retry. Needs the local 32 GB 5090, which the training queue is using (28.8 of 32.6 GB in use on 2026-10-21) | a free 5090. Inputs exist locally (`playground/esm-msr-devel/pgym_inputs`, `LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`); the devel checkpoints are on the cluster under `LoRA_models/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}/` (copy with their `MANIFEST.txt`). Runs: released full (4 DMS); released WT-only σ ∈ {0, 0.25, 0.5, 0.75, 1} (BRCA2, POLG, SCN5A); devel 3 seeds × σ ∈ {1, 0.75, 0.5, 0.25} (seed 1 at σ = 1 needs only BRCA2, POLG). Use `--batch_size 1 --dtype bf16` with the default allocator, as the `slurm_pgym.sh` second pass does |
| MutateEverything on BRCA2 and SCN5A (ProteinGym) | ME at 217 DMS | out of memory at the cluster's card sizes | same 5090 window; low priority, since all comparisons use the 213 common DMS anyway |
| **SPURS and the other Megascale-trained models on a homolog-filtered Tsuboyama split** | makes the §1.2 capped column interpretable, where SPURS (single model) currently reads 0.864 on test against ESM-MSR (devel) 0.849 | SPURS trains on Megascale, so the capped split is not held out for it. `esm_msr_testing.py --remove_spurs_homologs` exists for exactly this, but the harness adapters do not apply it | rescore every ‡ model on the libraries that survive that filter. Until then the SPURS, ESM3ΔG, IFUM and ESMtherm Tsuboyama numbers are upper bounds, not comparisons |
| IFUM on ProteinGym | the IFUM row of §1.4 | declined 2026-10-22: ~246 h even with the fp16 patch, since the speedup is only 1.13–1.28× at the lengths that carry 95 % of the cost (`GPU4_INFERENCE_LEDGER.md` §3.2). A partial slice (Stability category 66 DMS / ~5.4 h; everything under 200 residues 94 DMS / ~12.5 h) was considered and turned down | a decision to spend the time, or a fix to the IEFFEUM head's precision rather than ProtT5's |
| DDGemb on ProteinGym | §1.4 | 2.4 M variants ≈ 10 days of server time | a decision that it is worth it; otherwise reported as not run |
| DDMut on capped Tsuboyama, DMS, Domainome, ProteinGym | the DDMut row | REST API at 20–30 s per variant: the capped split alone is weeks | not planned |
| UniMutStab | the whole row | no released weights or inference code (pseudocode only); reimplementation ruled out on 2026-10-14 | weights from the authors |
| GeoDDG with FoldX structures beyond the 12 external sets | — | FoldX BuildModel is CPU-bound (~150–250 CPU-h for the externals); FASPR is the stand-in and matches it to 3.5e-4 kcal/mol on the externals | not needed |
| **ESM3 through Forge: the gaps its stored runs leave** | small lacks GB1 and Domainome; medium lacks GRB2 binding, MYO, ESTA, GB1 and Domainome; large lacks GB1 | paid, rate-limited API; the earlier runs were never completed. (S461, K2369 and PTMUL-D are *not* gaps any more: they are scored from their parent sets.) | a Forge key and budget; ESM3-small-open at σ = 0 covers the open model locally |
| ~~SPURS (both models) and ProteinMPNN on mutant-background Tsuboyama libraries~~ | **done 2026-10-23** | adapters patched (parsed sequence overwritten with the corrected `ref_seq`), reruns 386549–386551 on gpu4; validated: no NaN, wild-type-background rows bit-identical to the old files, mutant-background libraries at Spearman 0.70–0.87 (SPURS) and 0.44–0.70 (ProteinMPNN). The SPURS single-model output first landed in a mislabelled directory (empty `--tag` through `--export`) and was installed after validation; `run_model.py` now rejects such tags | – |
| Rosetta on the external sets, DMS and Domainome | the Rosetta row outside Tsuboyama | the Narval run covered the capped Tsuboyama split only; the published comparison (§1.5) has Rosetta on the older row sets | a `cartesian_ddg` array over those sets (CPU) |
| MT-pass ablation (`--mt_pass_adapter wt`, devel seed 1, σ = 1) | §3 ablation bullet | **resubmitted 2026-10-23** as `gpu4_pgym_s1_mtpass_wt2` (386637, gpu4, ~29 GPU-hours): the 26 DMS from the cancelled job 386255 were corrected to the per-mutation bias (manifest in the output directory; real-rerun check on 3 DMS passed) and the other 187 of the 213 common DMS run under the new defaults, resuming from them. Scores the 213, not the four >1,800-residue structures | **running 2026-10-24**, 177 / 213 at 21 h, ~29 GPU-hours estimated; partial result in the meantime: −0.017 ρ on 26 DMS |

## 4. Analysis to do once §1 lands (CPU)

- **Epistasis for the other models that predict a double directly** (SPURS multi, IFUM native, ESM3ΔG, DDMut):
  `analysis_scripts/epi_rosetta.py` is the template. It feeds a `direct` and a `single` prediction dictionary to
  `epi_baselines.assemble` and `score`, so a model needs only `pred` and `pred_additive` in the harness layout.
- **Bootstrap intervals** (`analysis_notebooks/bootstrap_external.py`) with the new rows included, so the rankings in
  §1.3, where the spread is small, can be read with an uncertainty.
- **A clean runtime table** on one A4500 per model ([`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md) §5).
- **Singles-only ProteinGym for the other models**, so that Rosetta's italic row has a comparable neighbour: every
  model's per-DMS predictions are on disk, so this is a re-scoring of the single-mutant rows, no GPU.

## 5. Housekeeping

- **The `pgym_per_mutation_bias.py` aggregate step refuses rather than guesses.** `pmb_apply` (386604) and
  `pmb_finish` (386619) exited 1 with `AssertionError: … cannot reproduce column …`, on `dc8 eps=1.0 n_seeds`
  (2 DMS differ: BRCA2, POLG) and on `ESM-MSR full` (CD19 singles). That guard is deliberate: the rewritten
  per-mutation files were already in place, and the script declined to overwrite a summary column it could not
  rebuild from the old files. Nothing was lost. It is the reason the summaries had to be regenerated by the rerun
  path rather than by rewriting the old CSVs in place, and any future `pmb_apply` should expect the same stop and
  be followed by the reruns (`gpu4_pmb2_*`, `gpu4_pmb3_tsu_*`) as it was here.
- The third-party harness (`~/stability-baselines/harness`) is not under version control. Added 2026-10-21:
  `adapters/spurs.py`, `adapters/proteinmpnn.py`, `build_pgym_tasks.py`, `lists/`. Added 2026-10-22: `--dtype` on
  `adapters/esm3dg.py` and `adapters/ifum.py`, `--readout` on `adapters/spurs.py` (all default to the previous
  behaviour; `*.bak` copies kept), an adaptive batch size in `adapters/esmtherm.py`, `make_fold_manifest_capped.py`,
  `~/consolidation/msr_eps_sweep.sh` (now `--mem=24G`), and the benchmarking scripts. **Worth copying into the repo** if the harness is ever to be reproducible from it.
- `~/esm-msr-devel` sits at `a61367c`, two commits behind `origin/devel-revision` (`7a5b283`). Update it only as step
  (3) of the ε → σ rename below, once 387181–387183 have finished. Its
  staged `data/preprocessed/*_mapped.csv` changes only rewrite structure paths for the cluster: keep them out of any
  commit.
- **The ε → σ artefact rename is prepared but deliberately not applied.** Commit `7a5b283` renamed the LoRA scaling
  factor ε → σ repo-wide, and the analysis scripts now glob `*_sigma{σ}_unmasked_*`; every artefact on the cluster is
  still `*_epsilon{ε}_unmasked_*`, so `benchmark_matrix.py` would find nothing. `baselines/rename_epsilon_to_sigma.sh`
  (dry-run by default, `--apply` to execute) plans **987 files with 0 conflicts**, and normalises both spellings of the
  unit case (`_epsilon1`, 350 files, and `_epsilon1.0`, 72) to `_sigma1_`, which is what `collect_msr(sigma='1')`
  globs. A plain substitution would leave the second group as `_sigma1.0_` and silently drop every capped val/test
  row. Every move is written to `~/esm-msr-devel/rename_to_sigma.manifest.tsv` so the rename can be reversed. Order:
  (1) wait for the three frozen-WT ProteinGym runs, 387181–387183; (2) run the script with `--apply`; (3)
  `git reset --hard origin/devel-revision` on the cluster; (4) regenerate the tables and diff them against the saved
  pre-rename output as the regression check — a rename must not move a single number; (5) update the `_epsilon1_` path
  globs in `dc_runs/*.py`.
- **Trim statistics will still not reach the collector after the rename.** Their filenames carry the mask tag
  (`..._sigma1_wt-unmasked_mt-independent_Combined.csv`) while `collect_msr` globs `*_sigma1_unmasked_{head}.csv`.
  That is a pre-existing naming gap rather than anything to do with ε → σ — the trim rows in §2.0 come from the
  rescorer, not the collector — but it has to be closed before trim can enter §1.
- Awaiting approval to delete: `~/esm-msr-rev` (old work clone) and `~/esm-msr-devel/outputs/pgym_verify_rev/`.
- `~/playground/narval-recovery/` (local machine) holds the Rosetta session's transcripts and the raw Narval outputs.
  The capped-split predictions were copied into the harness layout at `~/stability-baselines/results/rosetta/`.
- A GitHub token pasted into a cluster chat on 2026-10-17 is stored in that session's transcript: revoke it.
