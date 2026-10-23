# Trim audit: what to delete, hard-code, export, keep, and what to test before shipping

Written 2026-10-22 on `claude/ci-weighted-training-156a6a` (devel-revision `ce80771` + the three CI-weighting commits). **Nothing in `src/` was changed
for this audit.** Evidence labels: **measured** = read from a run's `hparams.yaml`, `metrics.csv`, a launcher or the code; **inferred** = reasoning no run
has checked. Line counts are from the functions and tests named, not from a diff.

## 0. The numbers

* `config.py` defines **160 flags** (163 spellings, three of them aliases). Every run logs **157 keys** to `hparams.yaml`, and so to Comet.
* Across the **56 completed runs of the current lineage** (`dc_runs/logs`, aborted ones excluded), **98 of the 147 behavioural flags never took a second
  value**. Across all 131 non-retired runs with an `hparams.yaml` the picture is the same for the architecture and loss switches: `adapter_mode`,
  `reg_loss`, `target_mode`, `use_dora`, `unfreeze_layernorms`, `last_n_layers`, `incl_structure_encoder`, `dataloading`, `loader_strategy`, `subset_size`,
  `mt_reg_mask` and `alternate_heads` each have one value in every run; `native_cond_for_mt` is **never true** in any run.
* An AST scan of definitions with no reference finds **about 460 lines in `utils.py` (of 928) that nothing reaches** (§3).
* The end state proposed here is **about 55 flags** (§5), roughly 60 logged keys, with the other ~100 either constants or moved out of the training CLI.
* Method: `hparams.yaml` scan of all runs (`training_logs/`, `dc_runs/logs/`); a census of each flag's use sites in `src/`, `tests/`, `analysis_scripts/` and the
  `dc_runs` launchers; an AST scan for definitions with no reference; and the docs and memory notes cited inline. The scripts are in the session scratchpad, not the repo.

## 1. How to read the verdicts

**DELETE** = remove flag and code. **CONST** = remove the flag, keep the value as a constant (and still write it to `hparams.yaml`, because
`make_fwt_block.py`, `gate_capacity.py` and `inference.parse_hparams_to_lora_config` read it back). **EXPORT** = move the role out of the training CLI.
**KEEP** and **CONSULT** are what they say.

A single mechanism makes the deletions cheap to land: a **table-driven `RETIRED_FLAGS`** in `config.py` (flag -> the one old value it may still be passed
with). A retired flag given that value is accepted with a warning and dropped from the namespace; given any other value it is an error. About 60 launchers in
`dc_runs/` pass `--detach_ensemble_input`, `--rank_loss`, `--lambda_*_combined`, `--adapter_mode dual`, `--mt_reg_mask all`, `--double_weight 1.0`,
`--wt_double_target latent` and similar, and the queue files and `mt_frozenwt_run.sh` itself still do. Today that is hand-written shim code per flag; as a table it is
about 25 lines for all of them.

## 2. Verdicts, by group

### 2.1 Delete: never varied, or inert by construction

| flags | evidence | what dies with them |
|---|---|---|
| `use_dora_{wt,mt}`, `unfreeze_layernorms_{wt,mt}` | false in 153/153 runs | the DoRA line in `_create_lora_config`; the layernorm-unfreeze block in `add_loras_to_esm3` |
| `incl_structure_encoder_{wt,mt}`, `last_n_layers_{wt,mt}`, `target_mode_{wt,mt}`, `incl_sequence_head_{wt,mt}` | one value in 153/153 (False / 0 / `expanded` / True). The structure-encoder switch is **inert** (corrected during the implementation: the model DOES instantiate the structure encoder, `ESM3PredictorBase` calls `get_structure_encoder()`, but ESM3's forward consumes the cached structure tokens and coordinates and never calls it, so LoRA on it would receive no gradient) | the training flags and the `struct_enc_arg` special case in `setup_dataloaders`. The regex branches in `_create_lora_config` were KEPT (inference reads old checkpoints' values through them); training passes constants (`training.LORA_CONSTANTS`) |
| `grad_clip_norm` | 0.0 in 56/56; the branch never ran | 3 lines in `training_step` |
| `calib_delay_steps`, `mt_lora_delay_steps` | 0 in 56/56 | the delay terms in both warm-up lambdas and in the `total_warmup_steps` formula (twice: `training_step` and `on_validation_epoch_end`, with a different fallback, 500, from the config's 0) |
| `wt_plateau_min_lr`, `mt_plateau_min_lr` | 1e-7 in all runs. Three decays at 0.3 reach 5.4e-6, so it never binds when `*_freeze_after_decays` is 3, and 12 decays would be needed otherwise | one `max(min_lr, ...)` each |
| `freeze_wt_after_epoch`, `wt_convergence_metric` | 1000 / `rho_wt_valid` in 56/56 (the epoch trigger can never fire in a 10-epoch run) | the epoch-trigger branch in `on_validation_epoch_end` |
| `double_weight`, `cond_weight`, `native_cond_weight` | 1.0 / 0.5 / 1.0 in 56/56; `cond` is excluded in every run | the per-subset weight loop and `is_native_cond_all` tensors at the top of `_compose_losses_streaming_and_backward` |
| `alternate_heads` | false in 63/63 | `alternating_phase`, the `phase` branches (WT forward skipped on an "mt" turn), `tests/test_alternate_heads.py`. A cost idea that the frozen-WT paradigm made moot |
| `wt_trains_doubles` and `wt_double_target` | `false` in 40 of 49 runs, always so since dc17; `linked` in one confounded run (dc5) | the `multi`/`wt_double_linked` target logic and `cens_wt` re-masking in the WT block; once the WT head never sees a double, neither flag means anything |
| `mt_reg_mask`, `combined_mode` | `all` in 110/110; `mean` in 63 of 64 (the other was `sum`, aborted, and `sum` is already removed). `route` has never run | `combine_predictions` becomes `0.5 * (wt + mt)`; the `route` guard that needs `mt_reg_mask doubles`; `tests/test_combined_mode.py` (93 lines) shrinks |
| `lora_mode` | `ensemble` in 110/110 (absent before); `corrector` never run | the `corrector` branch in `_compose...`, the guards in `MSRModel`. Inference keeps its `hparams.get('lora_mode', 'ensemble')` default so old checkpoints load |
| `shared_scale_init`, `shared_bias_init`, `calib_lr_mult` | 0.3 / 0.0 / 20.0 in 56/56 | become module constants |
| `score_column` | `ddG_ML` in 56/56 | the argument; `score_name` stays an internal parameter (benchmarks use `ddG`) |
| `combine_validation` | false in 56/56 | the concat in `create_protein_dataloaders` |
| `max_train_proteins`, `num_sanity_val_steps`, `check_val_every_n_epoch`, `log_every_n_steps` | -1 / 0 / 1 / 25 in 56/56 | Lightning plumbing; hard-code |
| `gpus`, `strategy`, `precision` | one GPU always (`strategy` is applied only when `gpus > 1`); `bf16-mixed` in 56/56. The streaming `manual_backward` loop with a per-library sampler has never run under DDP and has no `DistributedSampler`. `_cast_frozen_linears_bf16` already assumes bf16 | the `16-mixed` GradScaler plugin and the `amp_scale` log; `accelerator="auto", devices=1` |
| `offline_model` | false in 56/56. It only sets `INFRA_PROVIDER=1` (`esm/utils/constants/esm3.py:101`), which is an environment variable, and the launchers already export `HF_HUB_OFFLINE=1` | EXPORT to the launcher env, as `inference.py` does |
| `micro_batch_tokens` | 0 in 54/54 ("disabled") | `_micro_batch`'s token budget, `tests` (7 refs). Keep `--micro_batch_size` as the VRAM cap |
| `dataloading`, `loader_strategy` | `cycle` / `all` in 153/153 | the `pool` branch of `setup_dataloaders`, the `equal`/`min` sampler paths. `PooledDataLoader` is **not** dead: the benchmark loaders use it (`preprocess_megascale.py:692`); check whether a plain `DataLoader(batch_size=3)` is equivalent before removing it (it groups by WT sequence) |
| `load_lora_checkpoint`, `load_wt_only` | null in 56/56; one launcher mention. `--init_wt_from_checkpoint` covers the WT case and the MT path is always fresh | the `load_lora_weights` call in `main()` (the method stays: inference uses it) |
| `early_stopping_metric`, `monitor_mode`, `checkpoint_path`, `save_top_k`, `checkpoint_filename` | `monitor_mode` is `max` everywhere; `early_stopping_metric` equals `monitor_metric` in the standing MT recipe; `checkpoint_path` is used only when `log_dir` is unset; `save_top_k` is 3 or 5 | derive the filename from `monitor_metric`. **The launchers parse `epoch=*-val_rho_wt_valid_avg=*.ckpt`** (`mt_frozenwt_run.sh`, `gate_capacity.py`), so the template must keep that shape |
| `freeze_wt_adapter`, `freeze_link` (as separate flags) | used only through `frozen_wt_mt_only`, which derives both | derive; see §2.6 on a single `--phase` |

### 2.2 Delete: whole features

| feature | evidence | what dies |
|---|---|---|
| **`--ci_weight*`, `--censor_doubles`** (this session's work) | the weighting efficiency on the items kept as values is **1.065** at tau 0.5, so the realised gain is below the seed noise (`docs/ci_weighting.md` §6); the reported interval `deltaG_95CI` is irrelevant to model error; the only test arm has not run and is expected null. You called the offline evidence a null | `ci_weights.py` (154), the offsets/weights in `losses.py`, `ci_item_weights` and the `label_var` plumbing in `data.py`/`preprocess_megascale.py`/`training.py`, `tests/test_ci_weighting.py` (453). **Keep** `docs/ci_weighting.md` and `analysis_scripts/censoring_band_check.py` as the record. Tag the branch tip first so the code can be cherry-picked. The queued `mt_frozenwt_ciw_r16a16_s1` runs from the locked pin `pin_ciw_f666366`, so it is unaffected |
| **`--censor_gap`, `--censor_gap_max`** | at the standing threshold of 0.5 the gap rule adds almost nothing: 92% of the numeric rows it flags already sit below dG 0.5 (`censoring.py` header), and the remaining rule touches **0.2% of singles and 1.8% of doubles** (`ci_weighting.md` §6). It has never been ablated alone (it is on in 54 of 54 runs that carry the flag), so this is measured coverage, not a measured effect | `SRC_GAP`, `protease_gap`, `gap_max` plumbing in `data.py` and `censoring.py` |
| **`--censor_threshold_double`** | the singles' threshold is insensitive (0.3 / 0.5 / 0.6 give best `val_rho_wt_valid` 0.8460 / 0.8459 / 0.8466, seed 2). The double threshold, 0 against 0.5, is **not established**: paired at seed 1 and matched epoch 3, `cell_rank_mt` +0.008 (about one SD) (`ci_weighting.md` §6) | one flag; `threshold_for` collapses to one value. This is your "combine into one value" |
| **`--nonspecific_term fx`**, `ns_knot_{lo,hi,step}`, `ns_lr` | gains are a post-hoc add-on to the readout, in one library, below the floor; on the floor-0 link a leave-one-library-out f buys nothing (`ns-term-evidence-2026-10-20`). `nonspecific_term` is `none` in all 29 runs on the current basis; the two ns launchers (`launch_fwt_ns_s1.sh`, `launch_mt_frozenwt_ns_s1.sh`) have no log, so they never ran; inference cannot read `ns_head.*` | `NonspecificTerm` (`link.py:246`), the ns blocks in `_compose...` and `validation`, `_apply_nonspecific`, the `nonspecific` optimizer group, `tests/test_nonspecific_term.py` (287). About 400 lines |
| **Per-library floors**: `floor_x_max`, `floor_min_n`, `floor_max_sd`, `floor_percentile`, `link_floor_min`, `link_floor_max`, `learn_link_floor_bounds`, `censor_floor` as a switch, `assay_floor` (becomes `--link_floor`) | every run since 2026-10-19 passes `--assay_floor 0.0`. The estimator finds a floor in only 47 of 411 libraries (123 of 170 training libraries have none). At matched config the data knee and the zero knee tie (`wt_floor_data` 0.8391 against `wt_floor0_base` 0.8403, favouring 0). `floor_percentile` is "logged for comparison only" | `floor_evidence`, `library_floor`, `shrink_floors`, `compute_floors`, `_log_floor_summary`, `_stamp_floor`'s three `floor_source` branches, the `dataset_floor` hparam, parts of `test_floor_censoring.py` / `test_assay_floor_override.py` / `test_global_floor_and_threshold.py`. Cached `y_floor`/`floor_stats` stay in the c5 cache, unread |
| **The upper clamp and learnable link**: `link_ceiling`, `learn_link_ceiling`, `link_tau_hi`, `learn_link_tau`, `learn_global_floor`, `link_tau_penalty`, `link_floor_penalty`, `link_floor_prior_sd`, `link_lr` | the 4-parameter link (dc7) matches the floor-only canonical (dc8): 0.1968 / +0.529 against 0.19-0.198 / +0.50-0.54. `learn_global_floor` drifted to -0.37 / -0.50 (dc3) and was pinned. A floor learned under a hinge is not identified: truncation pulls it up, scored on all items its best value is exactly 0 (`floor-fit-truncation-bias`). The knee width is 0.59 (RT) in every run | the ceiling half of `SaturatingLink`, the `link` optimizer group, the penalties and their once-per-step backward, `tests/test_link_ceiling.py` (166). **Keep** the parameter names `raw_tau`, `floor_min`, `floor_max`, `global_floor` (as buffers) and `numpy_link_from_state_dict`, because the frozen-WT arms copy `link_head.*` out of the gold WT checkpoints (`_load_link_weights`) and the rescoring scripts read old ones |
| **The `cond` subset** | `incl_cond` false in 56/56; `cond_weight` 0.5; the branch's own header says no `cond` here | `incl_cond`, `cond` in `MULTI_SUBSETS` / `MT_HEAD_SUBSETS` / `SUBSET_ORDER`, `bg_offset` in the link, the cond emission in `_load_data`. **Do not bump the cache yet**: c5 carries ~111k cond items against ~61k doubles, so a c6 without them would also cut load time, but that is a rebuild (§8, stage 3) |
| **`--lora_mode corrector`, `--combined_mode route`** | see §2.1 | |
| **The retired shims** `--detach_ensemble_input`, `--rank_loss` | already retired | replace by `RETIRED_FLAGS` rows |

### 2.3 Constants

`tf_delink` and `per_mutation_bias` -> **CONST True**, as you proposed. `per_mutation_bias=False` is already a parser error under `--link` + ensemble, and the
`tf_delink=False` path is "the old scale-mismatch behaviour" (default on since 2026-10-19; false in 26 older runs). One refinement: keep the
`per_mutation` argument of `CalibrationHead`, because inference loads checkpoints trained without it.

`link` -> **CONST True** (true in 56/56; 124 uses in `src`, every `if link is not None` branch in `_compose...` and the validation block goes), together with
`--link_floor 0.0` and `--link_tau 0.59` as the two settable numbers.

Plateau machinery -> **three shared flags**, `--plateau_patience 1 --plateau_factor 0.3 --freeze_after_decays 3`, applied to whichever head is training
(today: six flags, with `wt_plateau_factor 0.1` in eight old runs and `mt_freeze_after_decays 0` in ten). `--mt_convergence_metric` stays (it is
`rho_combined` in the current MT recipe and `epi_beyond_rho_mt` in the dc series).

### 2.4 Export: roles that belong to another part of the pipeline

* **Credentials and paths out of `hparams.yaml`.** The Comet key is a flag, so it lands in `hparams.yaml`, in Comet and in every `ps` line; it leaked twice on
  2026-10-22 (`never-print-hparams-or-ps-cmdlines`). `comet_ml` already reads `COMET_API_KEY`. Take it from the environment and pass `ignore=[...]` to
  `save_hyperparameters` for the key, the 7 path flags and the two name lists (`train_dataloader_names`, `val_dataloader_names`: 28 names). This is independent of every deletion above
  and is the single biggest cut to what Comet shows.
* **A model spec instead of `incl_*`.** `inference.parse_hparams_to_lora_config` rebuilds the adapter architecture by reading training flags back out of
  `hparams.yaml`. Make the architecture an explicit `LoraSpec` (rank, alpha, dropout per head; the rest constants) written next to the checkpoint, with the
  old keys as the fallback. That settles the `incl_*` question without Hydra (§6, decision 6).
* **Cache building out of training.** Training starts by constructing `MegaScaleDatasetPreprocessor(raw_data_file, af_model_folder, ...)` and loading the
  ESM3 structure encoder, even when the cache exists, only to supply `dG_wt`, the floor statistics and the splits. With the floors gone, items already carry
  `dG_wt` (cached since c5) and the splits come from `--split_file`, so training can read a cache and fail if it is missing. `raw_data_file`, `af_model_folder`,
  `mut_structures_root`, `regenerate_cache`, `remove_spurs_homologs` then belong to `analysis_scripts/build_cache.py`. About 450 lines of `data.py`
  (`generate_data`, `_load_data`, `_get_encoded_structure`, `_try_load_modeled_context`) leave the training import path. **Larger and riskier than the rest; stage 3 of §8.**
* **`offline_model`** -> launcher env (§2.1).

### 2.5 Keep, rebut, or leave for the consult

| flag | verdict |
|---|---|
| `resume_from_checkpoint`, `comet_experiment_key` | **KEEP (I disagree with "redundant")**. `--init_wt_from_checkpoint` restores the WT adapter and link only. The MT path is fresh by design, so the optimizer, the plateau counters and the MT adapter are only recoverable by a true resume. It has been the only recovery twice: the reference run was killed by the driver mid-epoch 5 on 2026-10-19 and resumed from its epoch-4 checkpoint (`EXPERIMENTS.md` §6f), and a power outage ended `bl_w2a4_m16a16_capped_s2` on 2026-10-21 (that run was aborted, not resumed). The dc12r arms also used it deliberately. The plumbing is ~40 lines and tested (`test_resume_state`). Possible tidy: read the Comet key from the checkpoint directory |
| `adapter_mode` | **KEEP**, with a warning. `fused` is implemented but no test trains it and no run has used it (`dual` in 153/153). See top-5 #2 |
| `censor_range` | **KEEP**, as you said. It is the selection-bias fix (0.7% of shallow doubles fail to fit against 24% below -5 kcal/mol) |
| `tf_mix` | **CONSULT**. 0.75 and 0.5 are within noise of 1.0 (`cell_rank_mt` 0.1883 / 0.1846 against 0.1878); 0.25 was aborted and **0.0 (never teacher-force) never ran**. Hard-code 1.0 only if you do not want that arm |
| `tie_mt_scale` | **CONSULT**. Default True, but the standing frozen-WT recipe passes `--no-tie_mt_scale`; tied 15 runs, untied 7, never a matched pair on one WT head. Pick the recipe's value and delete the other path, or run the pair |
| `native_cond_for_wt` / `native_cond_for_mt` | **KEEP for one arm** (top-5 #1), then decide. `for_wt` (True) is the released behaviour; `for_mt` has never been true in any run |
| `mask_strategy` | **CONSULT**. Unused in this lineage (null in 152/153; `chain` in one released-era model). It is a known trap: the global `marginal` setting blinds the MT pass to the partner, so `cell_rank` is exactly 0 (`marginal-masking-kills-interaction`). Remove the global flag now and bring back the per-pass pair (top-5 #3), or keep it until then |
| `subset_size` (rank list length, 16) | **KEEP** and rename `--rank_list_size`; it has never been varied (top-5 #4) |
| `lambda_{rank,reg}_{wt,mt}` | **KEEP** four. Drop the `*_combined` aliases into `RETIRED_FLAGS` |
| `learning_rate`, `batch_size`, `micro_batch_size`, `num_epochs`, `lr_warmup_steps`, `weight_decay` | **KEEP**. `weight_decay` is 0 everywhere and `learning_rate` / `batch_size` are fixed at 2e-4 / 256 in every run bar one WT arm (1e-4); none was ever swept in this lineage |
| `early_stopping_patience`, `monitor_metric`, `external_val_sets`, `val_dump`, `val_cycle_passes`, `val_wt_only`, `skip_val`, `comet_project_name`, `comet_offline` | **KEEP** |
| `reg_loss`, `huber_delta` | **COLLAPSE** to one flag, `--huber_delta` (0 = MSE). Both are on top-5 #5; delete whichever side loses |

### 2.6 One structural simplification worth a look

Two named paradigms are each assembled from 4-5 primitives:

* WT-only sweep: `--freeze_mt_adapter --val_wt_only --lambda_*_mt 0 --subset_caps double=0`
* Frozen-WT MT-only: `--frozen_wt_mt_only` (+ derived `--freeze_wt_adapter --freeze_link --lambda_*_wt 0 --no-wt_trains_doubles`)

A single `--phase {wt, mt, joint}` could derive all of it (and record it in `hparams.yaml` as the effective settings, as `--frozen_wt_mt_only` already does).
It also turns `freeze_wt_adapter`, `freeze_mt_adapter`, `freeze_link`, `val_wt_only` and the double-exclusion cap into derived values. **`joint` is the open question**
(§6, decision 1): if it goes, so do `_transition_to_mt`, `converged_wt_*.pt`, `has_transitioned` in the checkpoint and `wt_freeze_after_decays`.

## 3. Dead code with no flag

`training_step`, `validation_step`, `on_validation_epoch_start` and `_create_experiment` show as unreferenced only because Lightning and `CometLogger` call them. The rest is real:

| where | what | lines |
|---|---|---|
| `utils.py` | `apply_masks` (463), `make_conditional_batch_doubles` (332), `_double_indices`, `_mask_sequence_rows`, and the helpers reachable only through them: `_generate_pos_mask`, `_generate_random_mask`, `_map_mutations_to_cols_and_validate`. The model does its masking itself (`forward_partitioned`), so these are the leftover of an older design | ~390 |
| `utils.py` | `convert_cif_to_pdb`, `get_residue_accessibility`, `accept_atom`, `register_temp_file`, `_select_lora_params`, `generate_ids` | ~73 |
| `data.py` | `PooledDataLoader.reset_epoch`; `_collate_with_padding_legacy` (called only from the pool path) | ~210 if the class goes |
| `preprocess.py` | `accept_chain`, `special_patches` | 3 |
| `routing.py` | `subset_mask`; `head_for` is test-only | ~13 |
| `epi_metrics.py`, `censoring.py` | `logged_names`, `auroc`: used by tests and analysis scripts only | keep, or move to `analysis_scripts` |

Inference-only modules (`inference.py` 705, `auto_batch.py` 450, `models.score_screening_batch`) are out of scope here.

**Rough total, if everything in §2.1-§2.4 lands: on the order of 2,000 of 11,400 source lines and 1,300-1,500 of 4,800 test lines.** The pieces I could measure: `utils.py` ~460;
the CI-weighting work **+362 / -16 source lines and +453 test lines** (`git diff ce80771..HEAD`; it exists only on this branch); the ns term ~150 source and 287 test; `config.py`
from 649 lines to roughly 300 once ~105 flags go. The remainder (floors, ceiling, `cond`, gap, pool path) is an estimate from function sizes. The dead helpers and the CI-weighting
work touch no running recipe.

## 4. Where I disagree, or refine

1. **`resume_from_checkpoint`** is not redundant with per-head checkpoints (§2.5).
2. **`tf_delink`/`per_mutation_bias`**: yes to hard-coding the training flag; no to deleting the model-side argument (old checkpoints).
3. **`reg_loss`**: keep one number, not the pair, until Huber has been tried.
4. **`incl_*` / Hydra**: the flags are constants (and one is inert), so what is needed is an explicit `LoraSpec`, not a config framework (§6, decision 6).
5. **`grad_clip_norm`**: delete, but it has never been *tried*, only left at 0. Gradient norms are already logged per group, so adding it back is three lines.
6. **`offline_model`** is an environment variable, not a training option.
7. **"`floor`: one global floor, fixed or learnable"**: one global floor, yes. *Learnable* I would drop for now: under a hinge it is not identified (§2.2). If you want it, it needs a censored likelihood, and my offline check of per-item soft censoring found nothing to gain (`ci_weighting.md` §6), so I would not build one yet.
8. **Interpreting "this most recent censoring"**: I read it as the `--ci_weight` / `--censor_doubles` / `--censor_gap` work. If you meant the censoring machinery wholesale, the part with evidence behind it is `censor_range` (keep) and one `censor_threshold` (keep); the floor-estimator and gap parts go regardless.

## 5. Proposed end state (about 55 flags)

* **Run (8):** `experiment_name`, `version`, `seed`, `log_dir`, `resume_from_checkpoint`, `comet_project_name`, `comet_offline`, `comet_experiment_key` (key from the environment)
* **Data paths (4):** `split_file`, `cache_path`, `cache_version`, `benchmark_data_path` (plus `raw_data_file`, `af_model_folder`, `regenerate_cache` until stage 3)
* **Architecture (7):** `adapter_mode`, `lora_{rank,alpha,dropout}_{wt,mt}`
* **Link (2):** `link_floor` (0.0), `link_tau` (0.59)
* **Paradigm (4):** `init_wt_from_checkpoint`, `frozen_wt_mt_only`, `freeze_mt_adapter`, `val_wt_only` (or one `--phase`)
* **Objective (10):** `lambda_{rank,reg}_{wt,mt}`, `rank_list_size`, `huber_delta`, `tf_mix`, `tie_mt_scale`, `native_cond_for_wt`, `native_cond_for_mt`
* **Data and censoring (5):** `subsets` (replaces `subset_caps` and the four `incl_*`; only {unrestricted, 0} was ever used), `censor_threshold`, `censor_range`, `external_val_sets`, `num_workers`
* **Optimisation (6):** `learning_rate`, `weight_decay`, `batch_size`, `micro_batch_size`, `num_epochs`, `lr_warmup_steps`
* **Schedule (6):** `plateau_patience`, `plateau_factor`, `freeze_after_decays`, `mt_convergence_metric`, `early_stopping_patience`, `monitor_metric`
* **Validation (3):** `val_dump`, `val_cycle_passes`, `skip_val`

## 6. Consult list, with my recommendation

1. **Joint WT+MT training (the dc17 recipe): keep or retire?** It decides `_transition_to_mt`, the WT freeze-after-decays and the transition state in checkpoints (~120 lines). *Keep for now.* dc17 is still the best canonical cell-level result on record (`cell_rank_mt` 0.198, `cell_mag_comb` 0.271, against 0.17-0.19 / 0.23-0.27 for the frozen-WT arms, last two epochs), though its WT head differs and the epoch counts differ, so the two are not a controlled comparison.
2. **Link: fixed constants, or one `--learn_link` switch?** *Fixed*, per §2.2 / §4.7.
3. **`tie_mt_scale`: choose the recipe's value, or run the pair?** *Run the pair* only if you will use the other value; otherwise take untied.
4. **`tf_mix`: hard-code 1.0 or keep for a 0.0 arm?** *Keep one more arm*, then hard-code whichever wins.
5. **Cache split (stage 3) and c6 without `cond`?** *Yes, but last*, since it is a rebuild and changes the dataset under every pinned run.
6. **Config mechanism: Hydra, or YAML defaults under argparse?** *YAML defaults*: `--config recipe.yaml` supplies defaults, the command line overrides. Hydra would change the CLI of ~60 launchers and the queue files for no gain at ~55 flags.
7. **`mask_strategy`: delete the global flag now, or keep until the per-pass pair is ported?**

## 7. Top-5 features that have not been properly tried

Ranked by expected information per GPU-hour. Each names the flag, what exists, and what would count as a result (the noise floor from `STATE.md` §5: `cell_rank_mt` SD 0.007-0.009, `flipacc` 0.024-0.030, so a single seed needs about +0.02 / +0.05, and two seeds are the minimum for a claim).

1. **`--native_cond_for_mt`: measured conditionals as MT supervision.** The 15 mutant-background libraries measure ddG(X | background) directly, which is exactly the quantity the MT head is built to estimate, while every other MT label is derived (a double's conditional is `2 ddG_AB - (ddG_A + ddG_B)`). Implemented, routed and unit-tested (`test_native_cond_{routing,input,scoring}`), and **true in zero of 153 runs**. About 17k items on the Oct-5 cache (`hparam_review.md`; recount on c5). Cost: one MT arm. Risk: only 15 libraries, and the background sequence differs from the structure. Read on `cell_rank_mt` / `flipacc` and the validation doubles, not on the native libraries themselves.
2. **`--adapter_mode fused`: one adapter reading both contexts.** The ctx control, which is the WT adapter read on the mutated sequence and has never seen a double, already earns ~3/4 of the MT head's `cell_rank` (0.15-0.16 against 0.19), and the doubles ablation (`dc18_nodbl`) shows the MT increment comes from the doubles, not from the second adapter (`wt-ctx-epistasis-control`). So a single adapter trained on singles *and* doubles in both contexts is the natural minimal model, and if it matches it removes the second adapter, the second calibration head, the freeze/handover and the WT checkpoint plumbing. Exists in `models.py` (`peft_fused`, `calibration_head_fused`); **needs a smoke test first** (no test trains it). Joint training only: a shared adapter cannot be frozen as the "WT half". Cost: one dc17-style arm, ~5 h.
3. **Per-pass masking: WT `marginal`, MT `independent`.** Designed as dc21 and never run. The global setting is structurally broken for epistasis (the MT pass cannot see the partner), so earlier masking results say nothing about this pairing; independent masking in the MT pass is the conditional masked marginal, i.e. the readout of ddG(A | B). Old rescoring found `independent`-trained models within noise of unmasked, and a pair-masked WT view carrying real pair signal (`rho_double` +0.03-0.05 over unmasked WT). Code is `04ab91c` (on `claude/scale-mismatch-improvements-a44582` and `origin/devel-experimental`), deliberately left out of the consolidation, so it has to be ported. Cost: ~2x an MT unit (one forward per mutated position).
4. **MT loss balance and list length.** `lambda_rank_mt` has never been varied in a completed run (1.0 in every MT arm; 0.2 was killed at epoch 0 as `dc16`, and its re-test `dc19` never ran), so whether the rank loss helps the MT head at all is unmeasured. The regression side has one point: `reg10_s1` (`lambda_reg_mt` 10) reads `cell_rank_mt` 0.1926 / `flipacc` 0.857 at 8 epochs against 0.181 / 0.811 for `r16a16_s1` at 4 epochs, which is not a matched comparison. `subset_size` (ListMLE list length) is 16 in 153/153. Arms: `lambda_rank_mt` in {0, 0.25}, `reg10_s2` (already parked in the queue file), and list length 64.
5. **Robust regression (`--reg_loss huber`).** One flag, never run (`mse` in 110/110). The dthr0 comparison found that **48% of the squared-error change between the two models sits in the (0.5, 1.5] reading band** and 55% in the expected-dead band, (the two shares overlap and are shares of a *change* between two models, not of the loss), which suggests, without proving, that deep doubles reading "alive" carry much of the squared error (`ci_weighting.md` §6). Huber at delta ~1 kcal/mol leaves in-range residuals (0.3-0.5) quadratic and caps those. It is the cheap complement to the censoring threshold, and unlike `ci_weight` it needs no label-variance model.

Also never varied, and cheap: the optimiser basics (`learning_rate` 2e-4, `batch_size` 256, `weight_decay` 0, warm-up 500, MT LoRA dropout 0.1), `tf_mix` 0.0, and an MT arm at `lora_alpha_mt` != 16.

## 8. Execution plan (after the consult)

Done in a **separate worktree/branch, merged only when the queue is idle**: the MT queue's default tree is `devel-main-conservative-merge-5e554f`, and a queued arm reads whatever is checked out there (`queued-runs-read-the-live-worktree`).

0. **No-behaviour-change:** `RETIRED_FLAGS` table; delete the unreachable `utils.py` helpers, `alternate_heads`, the structure-encoder/DoRA/layernorm switches, `grad_clip_norm`, the delay terms, the `min_lr` terms, `offline_model`, `gpus`/`strategy`/`precision`. Credentials and paths out of `hparams.yaml`.
1. **Constants that equal the canonical recipe:** `tf_delink`, `per_mutation_bias`, `combined_mode`, `lora_mode`, `mt_reg_mask`, `wt_trains_doubles`/`wt_double_target`, `link`, the subset weights, the plateau trio, `micro_batch_tokens`, `score_column`.
2. **Features:** `ci_weight`/`censor_doubles`, `censor_gap`, `censor_threshold_double`, ns term, per-library floors, ceiling and learnable link, `cond`.
3. **Pipeline exports:** training reads a prebuilt cache; `build_cache.py` owns the raw table; c6 without `cond`; YAML recipes.
4. **Experiments** of §7, each from a clean tree.

**Safety net for stages 0-2:** the technique from `hparam_review.md` §1: run the real loss composition on a synthetic batch under the canonical recipe's configurations before and after, and require bit-identical losses and gradients. Every stage-0/1 change removes a branch that recipe does not take, so identity is the expected result and any difference is a bug. Then the full suite, a 2-step CPU smoke through `main()`, and a launch of each `dc_runs/*.sh` recipe with `--help`-level argument parsing to prove the old command lines still parse. Readers of `hparams.yaml` to check first: `make_fwt_block.py`, `gate_capacity.py`, `arm_parity.py`, `inference.parse_hparams_to_lora_config`, `comet_reupload.py`.

## 9. What landed (2026-10-22, branch `claude/trim-training-2026-10-22`)

You approved the whole plan with the recommendations at §6 taken as given, and asked for it to be implemented and the top-5 queued. This is
what was done, where it departs from the plan, and how it was checked. Nothing is merged.

**Done (stages 0-3).** (The 'about 2,000 source lines' of §3 came out at a net 1,860, of which 553 moved rather than went.) The flag-by-flag outcome is `config.py`'s `RETIRED` table (every removed flag, the one value it may still be given,
and what replaced it). The code (lines): `utils.py` 928 -> 422; `config.py` 649 -> 453 (about 150 of them the `RETIRED` table, which is what keeps 60 launchers parsing); `link.py` 274 -> 165 (a link with nothing to learn); `data.py` 1853 -> 792 (the dataset only reads a cache; the generation code moved verbatim into the new `cache_build.py`, 553 lines that training does not import); `training.py` 1754 -> 1336. Against devel-revision `ce80771` the source diff is -3,487 / +1,627 lines (the additions include the 553 moved lines) and the tests -1,393 / +681. `ci_weights.py` and its 453-line test are gone (tag `archive/ci-weighting-2026-10-22`). Tests: 333 -> 372 passing (tests for removed features
deleted, new ones for the config table, the cache and manifest, per-pass masking, censoring).

| asked for | landed as |
|---|---|
| RETIRED table so ~60 launchers still parse | `config.RETIRED`; harmless old values accepted, behaviour-changing ones are errors that say what replaced them; `--subset_caps` / `--incl_*` -> `--subsets`, `--assay_floor` -> `--link_floor`, `--reg_loss huber` -> `--huber_delta`, `--no-censor_floor` -> `--censor_threshold none`, `--comet_api_key` -> the environment (never recorded) |
| credentials and paths out of hparams | `training.UNLOGGED_ARGS` + the Comet key from `COMET_API_KEY` / `~/.comet.config`. `hparams.yaml` went from 157 keys to ~60 |
| fixed link, floor 0 / tau 0.59 | `SaturatingLink(tau, floor)`; buffers named as the old parameters, so `link_head.*` loads and rescoring reads old checkpoints |
| one censor threshold | `--censor_threshold` (default 0.5, `none` = off); the gap rule and the per-library floor estimator are gone |
| shared plateau flags | `--plateau_patience/--plateau_factor/--freeze_after_decays`; one `_plateau_step(head)` |
| `tie_mt_scale` | taken untied (the standing recipe's value); the model-side argument stays for checkpoints trained tied |
| `tf_mix` | kept (0.0 arm queued) |
| YAML defaults instead of Hydra | `--config configs/*.yaml` (`mt_frozen`, `joint`, `wt_phase`); the command line wins |
| training reads a prebuilt cache; c6 without `cond` | `analysis_scripts/build_cache.py {build,convert}`, a per-cache `manifest_<version>.json` listing the libraries of each split in loader order; c6 written next to c5 (6.7 GB, 122,176 `cond` items dropped) |
| per-pass masking | `--mask_strategy_wt/--mask_strategy_mt` (the global flag is retired), resolved by `models.pass_mask_strategy`, applied in training, validation and the cycle passes |

**Departures from the audit, on purpose.**

1. **No `--phase`.** §2.6 called it "worth a look" and it depends on the joint-training decision; the four paradigm flags stay.
2. **`_create_lora_config` keeps its `target_mode` / `last_n_layers` / `incl_*` regex branches** (only DoRA and layernorm unfreezing were deleted
   from the model side): inference rebuilds an old checkpoint's adapter through them, so training passes constants instead of the model forgetting them.
3. **Items still carry `y_floor`**, stamped with the one floor, and the link methods still take it: a dump written now is read by the same
   `epi_metrics` as an old one (that file must stay byte-identical across branches).
4. **`ESM_MSR_PRECISION`** (environment, not a flag): fp32 for the CPU smoke test only. CPU bf16 autocast fails inside ESM3's geometric
   attention; production is bf16-mixed on GPU.
5. The structure-encoder remark of §2.1 was wrong as first written (corrected above).
6. `make_fwt_block.py` (stacks frozen-WT launchers from a checkpoint's hparams) targets the old flag set and was not updated; the new arms use `configs/`.

**Checked.**

* *Loss parity.* The real `_compose_losses_streaming_and_backward` on a stub backbone, driven by the canonical recipes' own (old) command lines
  through the new parser: frozen-WT MT, joint, WT-only, `tf_mix` 0.5 and 0, threshold 0, `lambda_rank_mt` 0.25, native-cond-for-MT, plus (second pass) `lambda_rank_mt` 0, `lambda_reg_mt` 10 and `--reg_loss huber`. Losses, every
  micro-batch backward value and every gradient are **bit-identical** to the pre-trim code (11 of 11: those plus `lambda_rank_mt` 0, `lambda_reg_mt` 10 and `--reg_loss huber`).
* *Data parity.* The old raw-table + c5 setup and the new manifest + c6 setup, on a 12-train / 3-val subset with the gap rule off: same libraries
  in the same order, same items, same censoring, bounds and floors, and all **411 sampler batch index lists identical** at seed 1. So a trimmed run
  and an old run at the same seed see the same batches; the one difference is the gap rule, which the old runs applied and the new code does not.
* *Config.* Every retired flag with its old value parses and is dropped; the behaviour-changing values error.
* *End to end on CPU* (`analysis_scripts/smoke_cpu.py`): see §10.
* *Launchers.* All 11 queued launchers were dry-run through the pin's parser: each differs from `trim_ctrl_s1` in exactly the one setting it is meant to vary, and
  `trim_ctrl_s1`'s parsed settings equal the standing `mt_frozenwt_run.sh` recipe's on every shared key (the Comet project name is the only difference).

## 10. The queue, and the end-to-end CPU smoke

Fresh Comet project: **`esm-msr-trim-oct22`** (private, workspace `sareeves96`, created through the API; the key is read from `~/.comet.config` and is on no
command line). Locked pin: `dc_runs/pin_trim_f5a00b8`; runner `dc_runs/trim_run.sh RECIPE SEED NAME [flags]`; recipes `configs/`. Appended to
`queue_wt_r2a4_f0.txt` behind `mt_frozenwt_ciw_r16a16_s1`, which finishes on its own old-code pin (the runner pid was not touched). Seed 1, frozen WT = best epoch of
`wt_r2a4_f0_cthr05_s1`, one variable each against `trim_ctrl_s1`:

| order | arm | variable | audit item |
|---|---|---|---|
| 1 | `trim_ctrl_s1` | none: the standing recipe on the trimmed code (no gap rule), the matched control and the check that the trim did not move the recipe | |
| 2 | `trim_ncmt_s1` | `--native_cond_for_mt` | top-5 #1 |
| 2b | `trim_r2a4_s1`, `trim_r4a8_s1` | MT `--lora_rank_mt/--lora_alpha_mt` 2/4 and 4/8 (rsLoRA scale 2.83 and 4.0 against the control's 4.0), added at your request: capacity, with r4a8 holding the update scale fixed. The old runs had r2a4 below r16a16 on every in-distribution target at matched epoch (`cell_rank_mt` -0.014) and slightly above on the external sets (+0.004 mean rho) | MT capacity |
| 3 | `trim_joint_dual_s1` | joint training, dual adapters (the control for #4) | #2 |
| 4 | `trim_joint_fused_s1` | `--adapter_mode fused`: one shared adapter | #2 |
| 5 | `trim_mask_mt_ind_s1` | `--mask_strategy_mt independent` (micro-batch 32: the two graphs of a double do not fit at 64) | #3 |
| 6 | `trim_huber_s1` | `--huber_delta 1.0` | #5 |
| 7-10 | `trim_rank0_s1`, `trim_rank025_s1`, `trim_list64_s1`, `trim_reg10_s1` | `--lambda_rank_mt` 0 / 0.25, `--rank_list_size 64`, `--lambda_reg_mt` 10 | #4 |
| 11 | `trim_tfmix0_s1` | `--tf_mix 0.0` | §2.5 |

Expected: the control within seed noise of `mt_frozenwt_r16a16_s1/s2` and `mt_frozenwt_dthr0_r16a16_s1` (cell_rank_mt 0.18-0.19). A single-seed lead needs about +0.02 on
`cell_rank_mt` or +0.05 on `flipacc`, and a second seed for both arms, before it is worth a sentence. The 13 arms are roughly 45 GPU-hours.

**Caveats the arms carry.** (a) *Fused:* the shared adapter's parameters sit in the `lora_wt` group, so the WT plateau (on `rho_wt_valid`) decays it, the MT plateau
has no group to act on, and after three decays the handover leaves it training at 2.7% of its rate until the epoch cap: a schedule difference from the dual arm, not a
pure architecture change. (b) *Independent masking:* it runs one MT forward per mutated position over the whole micro-batch and keeps both graphs, so it is ~2x the
cost and memory of a double; the frozen WT pass stays unmasked, as it was trained (the full dc21 pairing needs a WT head trained under marginal masking).
(c) Single seed: screening only.

**End-to-end CPU smoke** (`analysis_scripts/smoke_cpu.py`: the real `training.main()` on truncated real libraries from the c6 cache, fp32 because CPU bf16 autocast breaks
inside ESM3's geometric attention; one epoch, a real validation before and after): `dual_joint`, `frozen_wt`, `fused`, `native_cond_for_mt`, `mask_mt_independent`, `mask_wt_marginal_mt_independent`,
`huber`, `wt_only` and `comet_offline` all complete with two validations and a checkpoint (the tool's first success test looked for train-loss columns, which a 6-step run never
logs at Lightning's every-25-steps interval; it now checks the validations and the checkpoint, and the first batch of runs was re-read against that). What it showed: the fused adapter trains (292 nonzero LoRA-B tensors, one calibration head);
with a frozen WT the WT metric is unchanged to the digit and the WT calibration/link are the loaded ones; the zero-shot `rho_combined` of the fused and dual models
is identical, as it must be (LoRA B starts at zero). It found two real defects, both fixed before the pin: the fused model skipped the bf16 cast of the frozen base
(about 2.8 GB more on the card than the dual model), and `--comet_offline` died because comet_ml does not create its offline directory (a latent bug of the old code).
It cannot say whether any recipe is any good.
