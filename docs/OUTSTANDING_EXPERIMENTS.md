# Outstanding experiments (benchmarks)

What still has to run, or cannot run yet, for the all-by-all comparison in
[`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) §1. Each entry says what it fills, why it is not done, and what
unblocks it. Last updated 2026-10-22 (evening). Where things ran and how long they took:
[`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md). Cluster paths are on `sreeves@subhak-gpu.uwaterloo.ca`.

## 1. Queued or running on gpu4 — no decision needed

Everything below is pinned to gpu4 (4 × A4500). Lanes are chained with `afterany`, so only the heads are eligible at once.

**Finished since 2026-10-22:** IFUM-native on capped Tsuboyama (fold 386515/386516, 3.0 h each; score 386517/386518,
1.4 and 1.9 h), SPURS multi 386519 (7.5 min), ProSST capped 386533, ESMtherm CAS9 retry 386532 (so ESMtherm now has all
213 ProteinGym DMS), and the released-checkpoint ε sweep at 0.75 / 0.5 / 0.25 (386520–386522, 1.5 h each) plus devel
seed 1 at ε = 0.75 (386523). All checked: no failed datasets, and the only NaN predictions are the 7,931 capped-test
rows on mutant-background libraries that SPURS cannot represent.

| what | fills (the `q` cells) | jobs | state |
|---|---|---|---|
| **ESM-MSR LoRA scale ε ∈ {0.75, 0.5, 0.25}, DMS + Domainome, devel checkpoint**, one lane per seed | §1.3 ε table, devel rows (shown once all 3 seeds finish) | seed 1: 386524 (running), 386525; seed 2: 386526 (running), 386527, 386528; seed 3: 386529–386531 | 7 of 12 runs not yet finished; ~1.5 h each, so ≈ 8 more GPU-h across the lanes |
| ESM3ΔG, ProteinGym, 213 DMS cheapest first, fp16 | §1.4 | 386364, running | 192 / 213; the remaining work is dominated by the largest DMS (SPG1, 537k variants) |
| GeoDDG (FASPR), capped Tsuboyama + ProteinGym | §1.2 capped cells, §1.4 | 386277, running | 119 / 213 |
| DDMut on the old hyperopt Tsuboyama doubles (web server, login node) | DDMut's Tsuboyama cells stay partial (`*`) until done | two client processes | ~6,861 + 8,321 of ~20,151 done |

**Scheduling note (2026-10-23).** A card sat idle while the next ε job waited, because the node was out of *memory*, not
GPUs: the ε jobs requested 96 G (copied from the gamut script) and use about 6.9 G (`MaxRSS` on every completed run). Reduced
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
| ESM-MSR (released), (devel) × 4 ε, WT-only sweep, ε = 0 control | done on 213 DMS; the four > 1,800-residue DMS are blocked (§3) |
| MutateEverything, ThermoMPNN, SPURS (both models), ProteinMPNN | done |
| ESMtherm | done on all 213 (one DMS needed a rerun with an adaptive batch size) |
| ESM3ΔG | running (fp16), 192 / 213 |
| GeoDDG | running, 119 / 213 |
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
| **ESM-MSR on the four largest ProteinGym DMS** (BRCA1_HUMAN_Findlay_2018, BRCA2_HUMAN_Erwood_2022_HEK293T, POLG_CXB3N_Mattenberger_2021, SCN5A_HUMAN_Glazer_2019; > 1,800 residues) | full 217-DMS ProteinGym for ESM-MSR; today every model is compared on 213 | out of memory on 24 GB (RTX 6000) and 20 GB (A4500) even at batch size 1. Devel seed 1 at ε = 1 alone got BRCA1 and SCN5A through an allocator retry. Needs the local 32 GB 5090, which the training queue is using (28.8 of 32.6 GB in use on 2026-10-21) | a free 5090. Inputs exist locally (`playground/esm-msr-devel/pgym_inputs`, `LoRA_models/esm-msr-baseline-w2a4-m16a16-capped`); the devel checkpoints are on the cluster under `LoRA_models/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{1,2,3}/` (copy with their `MANIFEST.txt`). Runs: released full (4 DMS); released WT-only ε ∈ {0, 0.25, 0.5, 0.75, 1} (BRCA2, POLG, SCN5A); devel 3 seeds × ε ∈ {1, 0.75, 0.5, 0.25} (seed 1 at ε = 1 needs only BRCA2, POLG). Use `--batch_size 1 --dtype bf16` with the default allocator, as the `slurm_pgym.sh` second pass does |
| MutateEverything on BRCA2 and SCN5A (ProteinGym) | ME at 217 DMS | out of memory at the cluster's card sizes | same 5090 window; low priority, since all comparisons use the 213 common DMS anyway |
| **SPURS and the other Megascale-trained models on a homolog-filtered Tsuboyama split** | makes the §1.2 capped column interpretable, where SPURS (single model) currently reads 0.866 on test against ESM-MSR (devel) 0.849 | SPURS trains on Megascale, so the capped split is not held out for it. `esm_msr_testing.py --remove_spurs_homologs` exists for exactly this, but the harness adapters do not apply it | rescore every ‡ model on the libraries that survive that filter. Until then the SPURS, ESM3ΔG, IFUM and ESMtherm Tsuboyama numbers are upper bounds, not comparisons |
| IFUM on ProteinGym | the IFUM row of §1.4 | declined 2026-10-22: ~246 h even with the fp16 patch, since the speedup is only 1.13–1.28× at the lengths that carry 95 % of the cost (`GPU4_INFERENCE_LEDGER.md` §3.2). A partial slice (Stability category 66 DMS / ~5.4 h; everything under 200 residues 94 DMS / ~12.5 h) was considered and turned down | a decision to spend the time, or a fix to the IEFFEUM head's precision rather than ProtT5's |
| DDGemb on ProteinGym | §1.4 | 2.4 M variants ≈ 10 days of server time | a decision that it is worth it; otherwise reported as not run |
| DDMut on capped Tsuboyama, DMS, Domainome, ProteinGym | the DDMut row | REST API at 20–30 s per variant: the capped split alone is weeks | not planned |
| UniMutStab | the whole row | no released weights or inference code (pseudocode only); reimplementation ruled out on 2026-10-14 | weights from the authors |
| GeoDDG with FoldX structures beyond the 12 external sets | — | FoldX BuildModel is CPU-bound (~150–250 CPU-h for the externals); FASPR is the stand-in and matches it to 3.5e-4 kcal/mol on the externals | not needed |
| **ESM3 through Forge: the gaps its stored runs leave** | small lacks GB1 and Domainome; medium lacks GRB2 binding, MYO, ESTA, GB1 and Domainome; large lacks GB1 | paid, rate-limited API; the earlier runs were never completed. (S461, K2369 and PTMUL-D are *not* gaps any more: they are scored from their parent sets.) | a Forge key and budget; ESM3-small-open at ε = 0 covers the open model locally |
| SPURS and ProteinMPNN on mutant-background Tsuboyama libraries | 5 of 25 val and 7 of 37 test libraries (all singles) | both models read the sequence from the wild-type PDB, which has the wild-type residue at the background position; the adapters return NaN rather than score the wrong sequence | an adapter that substitutes the background residue in the parsed structure; minor, since these rows are all singles |
| Rosetta on the external sets, DMS and Domainome | the Rosetta row outside Tsuboyama | the Narval run covered the capped Tsuboyama split only; the published comparison (§1.5) has Rosetta on the older row sets | a `cartesian_ddg` array over those sets (CPU) |
| MT-pass ablation (`--mt_pass_adapter wt`, devel seed 1, ε = 1) | §3 ablation bullet | cancelled 2026-10-21 after 15:46 h at ~26 of 213 DMS (it needed ~6 days on an RTX 6000); the partial output is in `outputs/pgym/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s1/sigma1.0_mtpass_wt/` | resubmit on an A4500 (~33 h) if the ablation is still wanted |

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

- The third-party harness (`~/stability-baselines/harness`) is not under version control. Added 2026-10-21:
  `adapters/spurs.py`, `adapters/proteinmpnn.py`, `build_pgym_tasks.py`, `lists/`. Added 2026-10-22: `--dtype` on
  `adapters/esm3dg.py` and `adapters/ifum.py`, `--readout` on `adapters/spurs.py` (all default to the previous
  behaviour; `*.bak` copies kept), an adaptive batch size in `adapters/esmtherm.py`, `make_fold_manifest_capped.py`,
  `~/consolidation/msr_eps_sweep.sh` (now `--mem=24G`), and the benchmarking scripts. **Worth copying into the repo** if the harness is ever to be reproducible from it.
- `~/esm-msr-devel` is behind `origin/devel-revision`; pull it once no queued job reads it (the ε-sweep jobs do). Its
  staged `data/preprocessed/*_mapped.csv` changes only rewrite structure paths for the cluster: keep them out of any
  commit.
- Awaiting approval to delete: `~/esm-msr-rev` (old work clone) and `~/esm-msr-devel/outputs/pgym_verify_rev/`.
- `~/playground/narval-recovery/` (local machine) holds the Rosetta session's transcripts and the raw Narval outputs.
  The capped-split predictions were copied into the harness layout at `~/stability-baselines/results/rosetta/`.
- A GitHub token pasted into a cluster chat on 2026-10-17 is stored in that session's transcript: revoke it.
