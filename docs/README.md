# Documentation index

Status key: **CURRENT** (read/trust this), **REFERENCE** (current facts, narrow scope), **ANALYSIS**
(evidence and measurements; conclusions may be superseded but the numbers stand), **HISTORICAL** (kept
because something still points at it, or because it records a decision; do not treat as current).

## Start here

| document | status | what it is |
|---|---|---|
| [`HANDOFF.md`](HANDOFF.md) | CURRENT | **Orientation for a new agent taking over**: which tree is documentation and which is code, the unversioned pin the queue actually runs and its patch list, the commit/branch state at handoff, what is running now, what the last two days established, and the immediate next actions. Read this before STATE.md. |
| [`STATE.md`](STATE.md) | CURRENT | Where the code, queues, logs and metrics live; what is running and queued; the canonical recipe; how to read results and noise; pitfalls; design facts. **Read first after any reset.** |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | CURRENT | Chronological experiment log with measured numbers, the failures/triage record, and the open questions in priority order. |
| [`nonspecific_epistasis.md`](nonspecific_epistasis.md) | CURRENT | The `--nonspecific_term fx` result: best cell-level numbers on record, the learned f(x) curve, caveats, and the experiment that would confirm it. **The main open lead.** |
| [`queue_stacking.md`](queue_stacking.md) | CURRENT | **The project's central rule**: what each win stacks onto what, the two decision gates, why the WT architecture must travel with its weights, and the exact commands to execute a gate (`make_fwt_block.py`). |
| [`queue_takeover_ns.md`](queue_takeover_ns.md) | HISTORICAL (plan executed, then superseded: the ns arms are deferred to the end of the ladder and the sweep was rescheduled to put the floor knee first) | The written, NOT executed plan to take over the GPU queue and run the `fx` arms before `wtsweep`: arms, runner change, pin, sequence, cost, decision rule, rollback. |
| [`frozen_wt_paradigm.md`](frozen_wt_paradigm.md) | CURRENT | The frozen-WT / MT-only / doubles-only paradigm: `--init_wt_from_checkpoint` + `--frozen_wt_mt_only`, which WT head to freeze and why, the acceptance check, comparability, the prepared `fwt` series, and its risks. |
| [`tf_mix.md`](tf_mix.md) | CURRENT | `--tf_mix`: blending the measured additive label with the WT head in teacher forcing — the mechanism, why it is worth tuning, the queued five-point sweep, what to read, and the self-reference risk. |
| [`BENCHMARKS_OCT2026.md`](BENCHMARKS_OCT2026.md) | CURRENT | **Every comparison of ESM-MSR against another model**: the all-by-all table (12 external ΔΔG sets, Tsuboyama, 7 DMS, Domainome, ProteinGym), the epistasis hierarchy against MutateEverything and ThermoMPNN retrained on our split, the dc8 ε sweep, the third-party predictors, and the caveats that bind the numbers. Every table carries the October 2026 **training-data leak notation** (identical mutation / same protein / homolog / corpus) with the full codes and the audit summary in §6. |
| [`LEAKAGE_AUDIT_OCT2026.md`](LEAKAGE_AUDIT_OCT2026.md) | CURRENT | **Which model trained on which reported test data**: every model and dataset in the benchmark tables, the reconstructed training inventories (papers, Zenodo/Zenodo-style archives, local clones and split files), the validation of the split's MMseqs2/Foldseek homology tooling, the exact-mutation / same-protein / homolog overlaps, the notation, the assumptions and the reproducibility recipe. Produces `benchmarks/leak_audit.sqlite` and the `benchmarks/leak_audit_*.csv` tables. |
| [`OUTSTANDING_EXPERIMENTS.md`](OUTSTANDING_EXPERIMENTS.md) | CURRENT | Benchmark runs that are queued, in flight or blocked (with what unblocks each), and the analysis and housekeeping still to do. |
| [`GPU4_INFERENCE_LEDGER.md`](GPU4_INFERENCE_LEDGER.md) | REFERENCE | Which inference ran on the A4500 node (gpu4) and which on RTX 6000s or remote servers, where per-dataset times are recorded, and what a clean runtime comparison still needs. Job list: `benchmarks/slurm_jobs_oct2026.csv`. |
| [`validation_metrics.md`](validation_metrics.md) | CURRENT | Definition of every logged validation metric and how to read it. Referenced from `src/esm_msr/epi_metrics.py`, `training.py`, `inference_scripts/esm_msr_testing.py`. |

## Reference

| document | status | what it is |
|---|---|---|
| [`dual_adapter_design.md`](dual_adapter_design.md) | REFERENCE | The dual WT/MT adapter + link architecture and why it is shaped that way (2026-10-03). |
| [`vram_and_batch_sizes.md`](vram_and_batch_sizes.md) | REFERENCE | Measured maximum structure length per inference batch size on the 32 GB card, throughput, and CPU-spill behaviour. Linked from `README.md`. |
| [`known_issues.md`](known_issues.md) | REFERENCE | Inference-side edge cases (very large DMS, WSL allocator behaviour). Linked from `README.md`. |
| [`epistasis_reliability_audit.md`](epistasis_reliability_audit.md) | REFERENCE | Audit of which epistasis metrics are reliable at the current validation-set size (2026-10-04). Untracked file — treat the working copy as the only copy. |

## Analysis (evidence)

| document | status | what it is |
|---|---|---|
| [`anova_epistasis_report.md`](anova_epistasis_report.md) + [`anova/`](anova/) | ANALYSIS | Variance partition of ΔΔΔG (additive expectation vs pair offsets vs line effects vs the specific combination), lookup control, split-half; the numbers behind the saturation and nonspecific-epistasis argument. Referenced from `src/esm_msr/link.py` and `analysis_notebooks/anova/`. |
| [`conditional_ordering_test.md`](conditional_ordering_test.md) | ANALYSIS | The conditional-ordering estimator and its simulation battery — how much identity-specific signal exists at all (≈0-5%). |
| [`epistasis_metrics_report.html`](epistasis_metrics_report.html) | ANALYSIS | The report version of the metric design and the saturation/nonspecific analysis, with figures. Referenced from `src/esm_msr/config.py` and `epi_metrics.py`. |

## Historical

| document | status | what it is |
|---|---|---|
| [`hparam_review.md`](hparam_review.md) | HISTORICAL | The 2026-10-06 hyperparameter review that produced queue phases A/B. Still referenced by `scripts/queue_plan1.sh` and `queue_plan2.sh`. Its conclusions have been superseded by `EXPERIMENTS.md` §3-5. |
| [`epistasis_training_handoff.md`](epistasis_training_handoff.md) | HISTORICAL | The 2026-10-06 training handoff. **§2 is the canonical command that `scripts/run_arm.sh` embeds**, which is why it is kept; everything else in it is superseded. |
| [`../FINDINGS.md`](../FINDINGS.md) | HISTORICAL | Early findings summary (up to 2026-10-06). Superseded by `EXPERIMENTS.md`; kept for the narrative of the first month. |

## Deleted 2026-10-18 (recoverable with git)

Five documents were superseded and removed in the consolidation. Nothing in the repository references them
any more; their unique content was carried into `STATE.md` / `EXPERIMENTS.md`, and `git show <commit>:<path>`
recovers any of them:

| deleted | why | where its content went |
|---|---|---|
| `handoff_oct08.md` | run-management instructions for the r2 queue, which has finished | queue operation → `STATE.md` §6; the autocast/packing pitfalls → §7; the r2 decisions → `EXPERIMENTS.md` §3 |
| `summary_oct08.md` | "what changed on 2026-10-08" | the variance/saturation/floor findings → `EXPERIMENTS.md` §2; the saturation argument → `nonspecific_epistasis.md` §5 |
| `supervisor_state.md` | context-reset notes for the pre-r2 state | traps → `STATE.md` §7; design facts (censoring, link algebra, interaction loss) → `STATE.md` §8 |
| `training_handoff.md` | canonical command and cost model for the 6-epoch devel arms | failure modes → `STATE.md` §7; the command itself is now `dc_runs/launch_dc17_s1.sh` |
| `overnight_log.md` | the 2026-10-05 supervision diary | the W0/W2/L1 numbers and the VRAM lesson → `EXPERIMENTS.md` §1; the "hyperstable AUCs are not interpretable" correction → `STATE.md` §5 |

## Conventions used across these documents

* Dates are the day the work happened. A statement without a date is current as of the top of `STATE.md`.
* `training_logs/<run>/0/metrics.csv` is the source of every number quoted; `<run>` names are given exactly.
* Runs whose names begin `_invalid_`, `_aborted_`, `_confounded_`, `_truncated_`, `_superseded_` or
  `_cudafault_` **do not count** — see the triage table in `EXPERIMENTS.md` §6.
* "Cell level" means `val_epi_cell_rank_mt` (+ `cell_mag_comb`, `cell_flipacc_mt`); "beyond-additive" means
  `val_epi_beyond_rho_mt` / `_comb`; the guards are `val_rho_wt_valid_avg` and `val_rmse_combined_avg`.
* A difference is only quoted as a result when it exceeds the measured replicate noise in
  `STATE.md` §5, and single-seed differences are labelled as such.
