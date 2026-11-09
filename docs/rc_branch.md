# `train-speedup-rc`: the training code with the alternatives taken out

Cut from `train-speedup` (9cf016c) on 2026-11-08. The experiments of Nov 7-8 that did not help (structure-mask tokens, the unmasked split-path legs) are not here; they live on the local
branch `wip-struct-cond-nov08` (02d29f9). **There is no compatibility** with the old command lines, the old `hparams.yaml` or the old tests of removed paths: an unknown flag is an error,
and the retired-flag table, the translations and the `DERIVED` constants are gone.

What a run is: block training (domain steps), the dual-adapter model, the Tobit likelihood on the latent dG, ListMLE on stop-gradient calibrated scores, the posterior (or measured)
teacher-forcing label, a cosine schedule with warmup, no weight decay. `configs/recipe.yaml` is the recipe of `tfl_T_p0_fw_r05_tm1`:

    python -m esm_msr.training --config configs/recipe.yaml --experiment_name NAME --seed 1 [--cache_path ...] [changes]
    # MT-masked read, WT unmasked:   --block_mask_mt 1 --block_row_chunk 40

## Flags that went (109 effective hparams -> 49), with the evidence

Evidence = the Comet parameters of the 71 non-archived block-training runs of `esm-msr-oct24-final` (one value in all of them unless a count is given).

| removed | why |
|---|---|
| `block_training` | the only training path: the per-item loop (`_compose_losses_streaming_and_backward`, micro-batches, the partner rank loss, `ProteinCyclingBatchSampler`) is deleted |
| `adapter_mode`, `fused_antisymmetric`, `lora_mode`, `per_mutation_bias`, `tie_mt_scale`, `target_mode_wt/_mt` | dual / ensemble / per-mutation bias / independent scales / `expanded` in all 71 (the model class still accepts them: the inference harness scores old checkpoints) |
| `block_identity`, `block_rank_norm`, `rank_calibrated_stopgrad`, `rank_group_by`, `tf_detach_wt`, `tf_no_label`, `cycle_detach_wt`, `lambda_cycle`, `joint_doubles`, `mt_direct_pred`, `wt_supervise_doubles`, `mt_single_loss_weight` | one value in all runs; `mt_single_loss_weight` was refused by the parser under block training |
| `freeze_after_decays`, `plateau_patience`, `plateau_factor`, `lr_schedule`, `mt_convergence_metric`, `cosine_min_lr_ratio`, `calib_lr_mult`, `grad_clip_norm`, `grad_clip_per_head` | the plateau code is bypassed when `lr_schedule == cosine` (all 71); the rest are constants of `config.py` now (5.0, 0.05, 1.0 per head) |
| `freeze_wt_adapter`, `freeze_mt_adapter`, `frozen_wt_mt_only`, `init_wt_from_checkpoint`, `val_wt_only` | the frozen / WT-first paradigm; `peft_manager.py` is gone |
| `batch_size`, `micro_batch_size`, `min_batch_size`, `drop_last` | per-item loader plumbing (validation uses 64) |
| `reg_nll`, `huber_delta`, `val_tobit_link` | Tobit is the only regression (65 of 71; the 6 others predate it); Huber was a variant of the MSE branch |
| `censor_threshold` -> `rank_censor_threshold` | with Tobit the regression ignores it; it ties ranks in ListMLE and, with `--tf_label measured` and no `--tf_cut`, is the teacher-forcing cut |
| `monitor_metric` | `config.monitor_metric(hp)`: the unmasked regime when `block_mask_mt <= 0.5`, else the masked one |
| `val_masking_strategy`, `val_unmasked`, `val_masked`, `val_cycle_passes` | the main regime is masked; the unmasked copy is scored whenever a head trains on visible sites |
| `y_floor` | stamped 0.0 on every item; only role was the default teacher-forcing cut (`config.ASSAY_FLOOR`) |
| `skip_val`, `val_every_epochs`, `val_dump`, `external_val_sets`, `subsets`, `censor_range`, `mask_strategy_wt/_mt` | constants |
| `link_*`, the retired-flag table | unused since the links were removed |

## Kept on purpose

`lambda_cond_mt`, `cond_partner_min_dg` (inert unless the legs are on; `cond_partner_min_dg` was 1.0 and `lambda_cond_mt` 0 in all 71 runs), `exclude_mutbg`, `incl_sequence_head_wt/_mt`, the LoRA
rank / alpha / dropout of both adapters, `rank_list_size` (it IS used: the reference length every ListMLE list is rescaled to), `tf_label` / `tf_cut` / `tf_mix` / `tf_prior_sd`, the Tobit constants.

## Not touched

`models.py`, `inference.py` and the inference scripts keep their options (fused adapters, tied scales, target modes, mask strategies): they are what scores a checkpoint, including the old
ones. `epi_metrics.py` is byte-identical to `main-baseline`'s.
