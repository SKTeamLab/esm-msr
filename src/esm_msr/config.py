import argparse
import os
import logging
from esm_msr.data import ProteinCyclingBatchSampler
from esm_msr.routing import LEGACY_SUBSET_ALIASES, canonical_subset

class ParseSubsetCaps(argparse.Action):
    """
    Parses 'key=value' pairs into a dictionary.
    Defaults keys to 0.0, except 'single' which defaults to None.
    Retired subset names (see esm_msr.routing) are accepted and translated, so older
    launch scripts keep working; naming both a retired key and its replacement is an error.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        valid_keys = ProteinCyclingBatchSampler.SUBSET_ORDER
        caps = {k: 0.0 for k in valid_keys}
        caps['single'] = None
        seen = {}

        for kv in values:
            if '=' not in kv:
                raise argparse.ArgumentTypeError(
                    f"Invalid subset_cap format: '{kv}'. Expected 'key=value'."
                )

            k, v = kv.split('=', 1)

            if k in LEGACY_SUBSET_ALIASES:
                logging.warning(f"subset_caps key '{k}' is retired; using '{canonical_subset(k)}'.")
                k = canonical_subset(k)
            if k in seen and seen[k] != kv:
                raise argparse.ArgumentTypeError(
                    f"subset_caps names '{k}' more than once (via {seen[k]!r} and {kv!r})."
                )
            seen[k] = kv

            if k not in valid_keys:
                raise argparse.ArgumentTypeError(
                    f"Invalid subset key: '{k}'. Must be one of {valid_keys}."
                )

            if v.lower() == 'none':
                caps[k] = None
            else:
                try:
                    caps[k] = float(v)
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"Value for '{k}' must be a float or 'None', got '{v}'."
                    )

        setattr(namespace, self.dest, caps)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESM3 Dual-LoRA Stability Model")

    # Define the global defaults to be used if the flag is omitted entirely
    default_caps = {k: 0.0 for k in ProteinCyclingBatchSampler.SUBSET_ORDER}
    default_caps['single'] = None
    
    arch_group = parser.add_argument_group("Architecture Configuration")
    arch_group.add_argument('--adapter_mode', type=str, default='dual', choices=['dual', 'fused'])

    lora_group_mt = parser.add_argument_group("MT LoRA Configuration")
    lora_group_mt.add_argument('--lora_rank_mt', type=int, default=6)
    lora_group_mt.add_argument('--lora_alpha_mt', type=int, default=12)
    lora_group_mt.add_argument('--lora_dropout_mt', type=float, default=0.15)
    lora_group_mt.add_argument('--incl_sequence_head_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--last_n_layers_mt', type=int, default=0)
    lora_group_mt.add_argument('--target_mode_mt', type=str, default='expanded')
    lora_group_mt.add_argument('--unfreeze_layernorms_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--use_dora_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--lora_mode', type=str, default='ensemble', choices=['ensemble', 'corrector'])

    lora_group_wt = parser.add_argument_group("WT LoRA Configuration")
    lora_group_wt.add_argument('--lora_rank_wt', type=int, default=6)
    lora_group_wt.add_argument('--lora_alpha_wt', type=int, default=12)
    lora_group_wt.add_argument('--lora_dropout_wt', type=float, default=0.15)
    lora_group_wt.add_argument('--incl_sequence_head_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--last_n_layers_wt', type=int, default=0)
    lora_group_wt.add_argument('--target_mode_wt', type=str, default='expanded')
    lora_group_wt.add_argument('--unfreeze_layernorms_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--use_dora_wt', action=argparse.BooleanOptionalAction, default=False)
    
    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument('--rank_loss', type=str, default='listmle')   
    loss_group.add_argument('--reg_loss', type=str, default='mse')
    loss_group.add_argument('--huber_delta', type=float, default=1.0)
    loss_group.add_argument('--lambda_rank_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_rank_combined', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_combined', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_mt', type=float, default=1.0,
                            help="Regress the MT pass on MT-head subsets (cond, native_cond); see esm_msr.routing. "
                                 "Keep > 0: this is the only term that gives the MT pass an absolute scale, and "
                                 "combined_pred = 0.5*WT + 0.5*MT is uncalibrated without it. Its targets are the "
                                 "noisy derived conditionals, so down-weight with --cond_weight rather than zeroing.")
    loss_group.add_argument('--lambda_rank_mt', type=float, default=1.0,
                            help="Within-column rank loss on the MT pass: for each flip column (same scored position, "
                                 "same partner identity, varying substitution) impose the measured ordering. Invariant "
                                 "to the assay's monotone response and to the dynamic-range floor by construction, so "
                                 "it cannot be satisfied by learning assay saturation - which the regression term can. "
                                 "This is the loss that targets identity-dependent interaction directly.")
    loss_group.add_argument('--flip_group_units', action=argparse.BooleanOptionalAction, default=True,
                            help="Order MT work-unit rows by flip column so each micro-batch holds whole columns "
                                 "rather than fragments of many. Same items and the same number of forwards; only "
                                 "the grouping changes. Without it a column of ~19 is scattered across the batch.")
    loss_group.add_argument('--flip_list_min', type=int, default=4,
                            help="Minimum members for a flip column to contribute to --lambda_rank_mt. Below ~4 the "
                                 "ordering carries little information and the gradient is mostly noise.")
    loss_group.add_argument('--subfloor_rank_only', action=argparse.BooleanOptionalAction, default=True,
                            help="For double-derived items below --min_additive_dG, mark them rank-only instead of "
                                 "dropping them: kept in the rank losses, withheld from the regression losses. The "
                                 "assay reports sub-floor doubles with no flag; roughly half are unidentifiable fits "
                                 "but a third are genuine compensation, so dropping them discards real data. Their "
                                 "ordering is informative while their absolute value is not.")

    loss_group.add_argument('--censor_floor', type=float, default=None,
                            help="Censored ranking for --lambda_rank_mt. Flip-column items whose MEASURED dG is at or "
                                 "below this value are pinned at the assay's dynamic-range floor, so their order among "
                                 "themselves is unknown. They are treated as tied at the bottom: every above-floor item "
                                 "must still rank above them, but their relative order costs nothing (censored "
                                 "Plackett-Luce; the WT head's rank loss is untouched). Needs the v6 cache. Typical "
                                 "values: 0.0, +0.5 (practical floor); -1.0 censors nothing in cache_v6. Default None = off. "
                                 "Unlike --subfloor_rank_only, which flags on the ADDITIVE prediction, this flags on the "
                                 "measured value, so genuine compensators (measured above floor) keep their ordering.")
    loss_group.add_argument('--mt_single_anchor_weight', type=float, default=0.0,
                            help="Per-item weight for also regressing the MT pass on ordinary singles (the zero-background "
                                 "case of the MT task). 0 disables. Requires --lambda_reg_mt > 0.")
    loss_group.add_argument('--mt_single_anchor_frac', type=float, default=1.0,
                            help="Fraction of the batch's singles to anchor each step, resampled per step. Anchored "
                                 "singles are the dominant cost of the MT pass (~48%% of its backbone rows), so 0.25 "
                                 "cuts total training cost by roughly a third. Weights are scaled by 1/frac so the "
                                 "anchor's expected contribution is unchanged and only its variance rises.")
    loss_group.add_argument('--lambda_epi_combined', type=float, default=0.0)
    loss_group.add_argument('--mt_reg_mask', type=str, default='all', choices=['all', 'doubles'])
    loss_group.add_argument('--double_weight', type=float, default=1.0)
    loss_group.add_argument('--reversion_weight', type=float, default=0.5)
    loss_group.add_argument('--cond_weight', type=float, default=0.5,
                            help="Per-item loss weight for derived conditional effects ddG(A|B). Two are emitted per "
                                 "double, and each is a difference of two measurements, so <1 is appropriate.")
    loss_group.add_argument('--native_cond_weight', type=float, default=1.0,
                            help="Per-item loss weight for conditional effects measured directly in a mutant background.")
    loss_group.add_argument('--weight_decay', type=float, default=0)
    loss_group.add_argument('--residual_wd', type=float, default=1e-5)
    loss_group.add_argument('--calib_lr_mult', type=float, default=20.0)
    loss_group.add_argument('--residual_lr_mult', type=float, default=0.1)
    loss_group.add_argument('--detach_ensemble_input', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--detach_calibration', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--detach_regression', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--zero_epistasis_for_singles', action=argparse.BooleanOptionalAction, default=True,
                            help="Hardcode residual epistasis to 0 for single mutations and exclude them from epistasis loss.")

    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=0.3)
    calibration_group.add_argument('--shared_bias_init', type=float, default=None)

    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--subset_size', type=int, default=16)
    rank_group.add_argument('--invert_list_loss', action=argparse.BooleanOptionalAction, default=False)

    mask_group = parser.add_argument_group("Masking Strategy")
    mask_group.add_argument('--mask_mutated_structure', action=argparse.BooleanOptionalAction, default=False,
                            help="CACHE-BUILD option: blank mutated coordinates before the structure encoder runs, "
                                 "baking masking into the cache. Off by default so one unmasked cache serves every "
                                 "masking setting; use --mask_structure to mask at run time instead.")
    mask_group.add_argument('--mask_strategy', type=str, choices=["marginal", "independent"], default=None,
                            help="SEQUENCE masking of the scored position(s). Off by default: unmasked wt-marginal "
                                 "scored best on singles and tied on conditionals, and it costs one forward per variant "
                                 "instead of one per mutation.")
    mask_group.add_argument('--mask_structure', action=argparse.BooleanOptionalAction, default=False,
                            help="STRUCTURE masking: blank coordinates and structure tokens at every position the MT-pass "
                                 "sequence mutates relative to the structure. The WT pass is never masked (its sequence and "
                                 "structure agree). Saved to hparams.yaml and re-applied at inference.")

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=20)
    train_group.add_argument('--min_epochs', type=int, default=None,
                             help="Minimum number of epochs to train before early stopping can trigger.")
    train_group.add_argument('--learning_rate', type=float, default=2e-4)
    train_group.add_argument('--lr_warmup_steps', type=int, default=250)
    train_group.add_argument('--mt_lora_delay_steps', type=int, default=0)
    train_group.add_argument('--calib_delay_steps', type=int, default=0)
    train_group.add_argument('--lr_total_steps', type=int, default=None)
    train_group.add_argument('--batch_size', type=int, default=256)
    train_group.add_argument('--micro_batch_size', type=int, default=16)
    train_group.add_argument('--dedup_backbone', action=argparse.BooleanOptionalAction, default=True,
                             help="Run ESM3 once per unique (sequence, structure) row of a micro-batch. All singles of a "
                                  "protein share their WT input, so the WT pass becomes one forward per protein per batch.")
    train_group.add_argument('--precision', type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed", "64"])
    train_group.add_argument('--gpus', type=int, default=1)
    train_group.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'deepspeed_stage_2', 'deepspeed_stage_3', 'fsdp'])
    train_group.add_argument('--seed', type=int, default=42)
    train_group.add_argument('--offline_model', action=argparse.BooleanOptionalAction, default=False)
    
    train_group.add_argument('--freeze_wt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--freeze_mt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--freeze_wt_after_epoch', type=int, default=1000)
    train_group.add_argument('--freeze_wt_on_convergence', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--wt_convergence_patience', type=int, default=1)
    train_group.add_argument('--wt_convergence_metric', type=str, default='rho_wt_valid')
    train_group.add_argument('--early_stopping_patience', type=int, default=0)
    train_group.add_argument('--early_stopping_metric', type=str, default='val_rho_combined_avg')

    data_group = parser.add_argument_group("Data Handling")
    data_group.add_argument('--benchmark_data_path', type=str, default='./data/preprocessed/')
    data_group.add_argument('--raw_data_file', type=str, required=True)
    data_group.add_argument('--af_model_folder', type=str, required=True)
    data_group.add_argument('--dataloading', type=str, default="cycle", choices=["pool", "cycle"])
    data_group.add_argument('--loader_strategy', type=str, default='all', choices=['equal', 'min', 'all'])
    #data_group.add_argument('--use_subset_restrict', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--split_file', type=str, default=None)
    data_group.add_argument('--score_column', type=str, default='ddG_ML')
    data_group.add_argument('--cache_path', type=str, default='./data_cache')
    data_group.add_argument('--regenerate_cache', action='store_true')
    data_group.add_argument('--num_workers', type=int, default=4)
    data_group.add_argument('--max_train_proteins', type=int, default=-1)
    
    # These inclusion flags will be automatically updated by subset_caps logic
    data_group.add_argument('--incl_singles', action=argparse.BooleanOptionalAction, default=True)
    data_group.add_argument('--incl_doubles', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--incl_cond', action=argparse.BooleanOptionalAction, default=False,
                            help="Conditional effects ddG(A|B) derived from doubles (MT head).")
    data_group.add_argument('--incl_reversions', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--incl_native_cond', action=argparse.BooleanOptionalAction, default=False,
                            help="Measurements from mutant-background libraries, e.g. code '1A0N_L7S' (MT head).")
    data_group.add_argument('--cond_structure', type=str, default='reuse', choices=['reuse', 'mask', 'model'],
                            help="Structure a conditional ddG(A|B) item conditions on: 'reuse' the parent double's "
                                 "(unmodified WT) structure, 'mask' the WT structure masked at the partner site, or "
                                 "'model' a modeled partner structure when one exists. Baked into the cache.")

    data_group.add_argument('--subset_caps', nargs='*', action=ParseSubsetCaps, default=default_caps,
                            help="Caps for data subsets as a fraction of the unrestricted subsets (e.g., double=0.6 cond=None). Defaults to 0 for all except 'single' (None).")
    data_group.add_argument('--mut_structures_root', type=str, default='/home/sareeves/software/esm-msr/data/tsuboyama/FINAL_results/')
    data_group.add_argument('--min_additive_dG', type=float, default=-1.0,
                            help="Drop double-derived items (double, cond) whose additive dG prediction "
                                 "dG(wt)+ddG_A+ddG_B falls at or below this value. The assay's bounded fit reports no dG "
                                 "below -1, so for those doubles the measured value is obliged to be too high and the "
                                 "error surfaces as spurious stabilising epistasis. None disables.")
    data_group.add_argument('--use_plddt', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--remove_spurs_homologs', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--combine_validation', action=argparse.BooleanOptionalAction, default=False)

    log_group = parser.add_argument_group("Checkpointing & Logging")
    log_group.add_argument('--experiment_name', type=str, required=True)
    log_group.add_argument('--version', type=str, default=None)
    log_group.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    log_group.add_argument('--checkpoint_filename', type=str, default='{epoch:02d}-{val_rho_combined_avg:.3f}')
    log_group.add_argument('--monitor_metric', type=str, default='val_rho_combined_avg')
    log_group.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'])
    log_group.add_argument('--save_top_k', type=int, default=5)
    log_group.add_argument('--load_lora_checkpoint', type=str, default=None)
    log_group.add_argument('--load_wt_only', action=argparse.BooleanOptionalAction, default=False)
    log_group.add_argument('--log_dir', type=str, default='./logs')
    log_group.add_argument('--comet_api_key', type=str, default=None)
    log_group.add_argument('--comet_project_name', type=str, default="esm-msr-april2026")
    log_group.add_argument('--log_every_n_steps', type=int, default=25)
    log_group.add_argument('--check_val_every_n_epoch', type=int, default=1)
    log_group.add_argument('--num_sanity_val_steps', type=int, default=0)
    log_group.add_argument('--offline', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--skip_val', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--resume_global_step', type=int, default=0)

    args, remaining_argv = parser.parse_known_args()
    
    # Synchronize incl_x flags based on subset_caps
    subset_flag_map = {
        'single': 'incl_singles',
        'double': 'incl_doubles',
        'cond': 'incl_cond',
        'reversion': 'incl_reversions',
        'native_cond': 'incl_native_cond',
    }

    if args.subset_caps:
        for subset_key, flag_name in subset_flag_map.items():
            cap_val = args.subset_caps.get(subset_key)
            # If None (unrestricted) or > 0, the subset must be included
            is_included = (cap_val is None or cap_val > 0)
            setattr(args, flag_name, is_included)
    
    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
        
    return args