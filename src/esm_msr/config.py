import argparse
import os
import logging
from esm_msr.data import ProteinCyclingBatchSampler


class ParseSubsetCaps(argparse.Action):
    """
    Parses 'key=value' pairs into a dictionary.

    A value of 'None' means unrestricted (the subset sets the baseline count); a float is the fraction of that
    baseline the subset is capped to, and 0.0 excludes it. Keys are the subsets of ``esm_msr.routing``.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        valid_keys = ProteinCyclingBatchSampler.SUBSET_ORDER
        caps = dict(DEFAULT_SUBSET_CAPS)

        for kv in values:
            if '=' not in kv:
                raise argparse.ArgumentTypeError(
                    f"Invalid subset_cap format: '{kv}'. Expected 'key=value'."
                )

            k, v = kv.split('=', 1)

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


# 'single' and 'double' are the measured wild-type-context subsets and are unrestricted.
#
# The two conditional subsets are both held out of TRAINING for now, pending the MT-head routing that would do
# something with them: 'cond' is derived arithmetic (ddG_AB - ddG_B from a double and its singles), and
# 'native_cond' is a measured conditional effect ddG(X | background) that the WT head has no business being
# regressed onto. Note that excluding native_cond is a deliberate departure from the released model, which trained
# on those items because they were simply labelled 'single' there. They are still kept in VALIDATION (where they
# are a direct measurement and count toward rho_combined), so the headline validation metric stays comparable.
DEFAULT_SUBSET_CAPS = {'single': None, 'double': None, 'native_cond': 0.0, 'cond': 0.0}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESM3 Dual-LoRA Stability Model")

    arch_group = parser.add_argument_group("Architecture Configuration")
    arch_group.add_argument('--adapter_mode', type=str, default='dual', choices=['dual', 'fused'])

    lora_group_mt = parser.add_argument_group("MT LoRA Configuration")
    lora_group_mt.add_argument('--lora_rank_mt', type=int, default=6)
    lora_group_mt.add_argument('--lora_alpha_mt', type=int, default=12)
    lora_group_mt.add_argument('--lora_dropout_mt', type=float, default=0.15)
    lora_group_mt.add_argument('--incl_structure_encoder_mt', action=argparse.BooleanOptionalAction, default=False)
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
    lora_group_wt.add_argument('--incl_structure_encoder_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--incl_sequence_head_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--last_n_layers_wt', type=int, default=0)
    lora_group_wt.add_argument('--target_mode_wt', type=str, default='expanded')
    lora_group_wt.add_argument('--unfreeze_layernorms_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--use_dora_wt', action=argparse.BooleanOptionalAction, default=False)

    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument('--rank_loss', type=str, default='listmle', choices=['listmle'])
    loss_group.add_argument('--reg_loss', type=str, default='mse', choices=['mse', 'huber'])
    loss_group.add_argument('--huber_delta', type=float, default=1.0)
    loss_group.add_argument('--lambda_rank_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_rank_combined', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_combined', type=float, default=0.0)
    loss_group.add_argument('--mt_reg_mask', type=str, default='all', choices=['all', 'doubles'])
    loss_group.add_argument('--double_weight', type=float, default=1.0)
    loss_group.add_argument('--cond_weight', type=float, default=0.5)
    loss_group.add_argument('--native_cond_weight', type=float, default=1.0)
    loss_group.add_argument('--weight_decay', type=float, default=0)
    loss_group.add_argument('--residual_wd', type=float, default=1e-5)
    loss_group.add_argument('--calib_lr_mult', type=float, default=20.0)
    loss_group.add_argument('--residual_lr_mult', type=float, default=0.1)
    loss_group.add_argument('--grad_clip_norm', type=float, default=0.0,
                            help="Clip the gradient norm to this before each step; 0 disables clipping.")
    loss_group.add_argument('--detach_ensemble_input', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--detach_regression', action=argparse.BooleanOptionalAction, default=False)

    link_group = parser.add_argument_group("Observed-Scale Link (esm_msr.link)")
    link_group.add_argument('--link', action=argparse.BooleanOptionalAction, default=True,
                            help="Put the regression target on the assay's observed scale through a soft floor at "
                                 "each library's own floor estimate. Ranking stays on the latent scale.")
    link_group.add_argument('--link_tau', type=float, default=0.59,
                            help="Width of the link's knee in kcal/mol.")
    link_group.add_argument('--learn_link_tau', action=argparse.BooleanOptionalAction, default=False)
    link_group.add_argument('--link_floor_min', type=float, default=-1.0,
                            help="Lower clip on a library's cached floor estimate.")
    link_group.add_argument('--link_floor_max', type=float, default=0.2,
                            help="Upper clip on a library's cached floor estimate.")
    link_group.add_argument('--learn_link_floor_bounds', action=argparse.BooleanOptionalAction, default=False)
    link_group.add_argument('--learn_global_floor', action=argparse.BooleanOptionalAction, default=True,
                            help="Learn the shared knee used by libraries that show no floor of their own. It is "
                                 "identifiable from exactly that population and is initialised at the median of "
                                 "the libraries that do have one.")
    link_group.add_argument('--link_ceiling', type=float, default=None,
                            help="Add a second soft clamp at this dG (kcal/mol), the top of the assay's reported "
                                 "range; 5.0 matches the '>5' tag. Omitted, the link is floor-only.")
    link_group.add_argument('--learn_link_ceiling', action=argparse.BooleanOptionalAction, default=True,
                            help="Let --link_ceiling move. It is parameterised as a positive span above "
                                 "--link_floor_max, so it can never cross a library's floor.")
    link_group.add_argument('--link_tau_hi', type=float, default=None,
                            help="Starting knee width for the ceiling. Defaults to --link_tau: nothing locates a "
                                 "value for the top of the range the way the pile-up at the bottom locates one.")
    link_group.add_argument('--link_tau_penalty', type=float, default=0.0,
                            help="Weight on sum (tau - 0.59)^2 over the free temperatures, added to the loss, so "
                                 "--learn_link_tau drifts from 0.59 only where the data pays for it.")
    link_group.add_argument('--link_lr', type=float, default=5e-3,
                            help="Learning rate of the link's parameters (no weight decay, no warm-up, and not cut "
                                 "by the plateau scheduler).")
    link_group.add_argument('--wt_double_target', type=str, default='latent', choices=['latent', 'linked'],
                            help="What the WT head regresses onto for a MULTI-mutant. 'latent': its unlinked LLR "
                                 "sum against the sum of the measured singles, ddG_A + ddG_B. 'linked': its LLR sum "
                                 "through the link against the measured ddG_AB. These are the only two "
                                 "self-consistent pairings - a linked prediction cannot be compared with a sum of "
                                 "observed singles, which is not a quantity the readout can report. Singles always "
                                 "go through the link against their own measurement.")

    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=0.3)
    calibration_group.add_argument('--shared_bias_init', type=float, default=None)

    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--subset_size', type=int, default=16)

    mask_group = parser.add_argument_group("Masking Strategy")
    mask_group.add_argument('--mask_strategy', type=str, choices=["marginal", "independent"], default=None)

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=20)
    train_group.add_argument('--learning_rate', type=float, default=2e-4)
    train_group.add_argument('--lr_warmup_steps', type=int, default=250)
    train_group.add_argument('--mt_lora_delay_steps', type=int, default=0)
    train_group.add_argument('--calib_delay_steps', type=int, default=0)
    train_group.add_argument('--lr_total_steps', type=int, default=None)
    train_group.add_argument('--batch_size', type=int, default=256)
    train_group.add_argument('--micro_batch_size', type=int, default=16,
                             help="Upper bound on the items per forward/backward. The binding constraint is "
                                  "normally --micro_batch_tokens.")
    train_group.add_argument('--micro_batch_tokens', type=int, default=0,
                             help="Optional token budget per forward/backward (micro batch x sequence length), "
                                  "which is what activation memory tracks. 0 (the default) disables it and uses "
                                  "--micro_batch_size alone, which fits at 64 with about 4.4 GiB spare on a 32 GB "
                                  "card. Set it (3700 holds the peak near 23 GB) only to buy margin on a smaller "
                                  "card or a longer-sequence dataset, where it is gentler than cutting the micro "
                                  "batch flat, since short libraries keep the full size.")
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
    train_group.add_argument('--wt_plateau_patience', type=int, default=0,
                             help="Epochs without improvement in --wt_convergence_metric before the WT path's own "
                                  "learning rate is cut by --wt_plateau_factor. 0 disables it. The shared plateau "
                                  "watches val_rho_combined_avg, which the MT path keeps improving long after the "
                                  "WT head has stopped, so without this the WT adapter keeps taking full-rate "
                                  "steps on a metric it is no longer improving.")
    train_group.add_argument('--wt_plateau_factor', type=float, default=0.1)
    train_group.add_argument('--wt_plateau_min_lr', type=float, default=1e-7)
    train_group.add_argument('--wt_freeze_after_decays', type=int, default=0,
                             help="Freeze the WT path after this many --wt_plateau_patience decays, which also "
                                  "fixes the target the MT path teacher-forces against. 0 never freezes.")
    train_group.add_argument('--early_stopping_patience', type=int, default=0)
    train_group.add_argument('--early_stopping_metric', type=str, default='val_rho_combined_avg')

    data_group = parser.add_argument_group("Data Handling")
    data_group.add_argument('--benchmark_data_path', type=str, default='./data/preprocessed/')
    data_group.add_argument('--raw_data_file', type=str, required=True)
    data_group.add_argument('--af_model_folder', type=str, required=True)
    data_group.add_argument('--dataloading', type=str, default="cycle", choices=["pool", "cycle"])
    data_group.add_argument('--loader_strategy', type=str, default='all', choices=['equal', 'min', 'all'])
    data_group.add_argument('--split_file', type=str, default=None)
    data_group.add_argument('--score_column', type=str, default='ddG_ML')
    data_group.add_argument('--cache_path', type=str, default='./data_cache')
    data_group.add_argument('--cache_version', type=str, default=None,
                            help="Pin the cache version this run reads (e.g. c4), instead of whatever "
                                 "data.MutationStabilityDataset.CACHE_VERSION currently is. Launch scripts run "
                                 "from the live worktree, so pinning is what stops a code change handing a queued "
                                 "job a different dataset under the same name.")
    data_group.add_argument('--regenerate_cache', action='store_true')
    data_group.add_argument('--num_workers', type=int, default=4)
    data_group.add_argument('--max_train_proteins', type=int, default=-1)

    # Kept in sync with --subset_caps below: a subset capped to 0 is never loaded in the first place.
    data_group.add_argument('--incl_singles', action=argparse.BooleanOptionalAction, default=True)
    data_group.add_argument('--incl_doubles', action=argparse.BooleanOptionalAction, default=True)
    data_group.add_argument('--incl_cond', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--incl_native_cond', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--native_cond_as_single', action=argparse.BooleanOptionalAction, default=False,
                            help="Label a mutant-background library's measurements 'single' rather than "
                                 "'native_cond', which is what the released model does: they then train the WT "
                                 "head and count as singles everywhere. Overrides the native_cond subset cap, "
                                 "since no item carries that label.")

    data_group.add_argument('--subset_caps', nargs='*', action=ParseSubsetCaps, default=dict(DEFAULT_SUBSET_CAPS),
                            help=f"Caps for data subsets (e.g. double=0.6 cond=0.1). Defaults: {DEFAULT_SUBSET_CAPS}.")
    floor_group = parser.add_argument_group("Assay Floor (data.floor_evidence)")
    floor_group.add_argument('--censor_floor', action=argparse.BooleanOptionalAction, default=True,
                             help="Treat an item measured at or below its library's floor as a bound rather than a "
                                  "value: regression becomes one-sided and the item's rank is tied at the bottom. "
                                  "Only applies to libraries that show evidence of a floor.")
    floor_group.add_argument('--censor_gap', action=argparse.BooleanOptionalAction, default=True,
                             help="Also censor an item whose two proteases disagree about its stability by more "
                                  "than --censor_gap_max, bounded by its own reported value. Needs no doubles and "
                                  "no library floor, so unlike --censor_floor it reaches every library.")
    floor_group.add_argument('--censor_range', action=argparse.BooleanOptionalAction, default=True,
                             help="Admit the rows the assay bounded rather than fitted ('<-1' / '>5') as censored "
                                  "items instead of dropping them. Dropping them selects the double set on its own "
                                  "outcome: fit failure rises from 0.7%% of shallow doubles to 24%% of those below "
                                  "-5 kcal/mol, so the survivors at depth are those that came out better than their "
                                  "additive expectation.")
    floor_group.add_argument('--censor_gap_max', type=float, default=2.0,
                             help="|deltaG_t - deltaG_c| above which the two fits are taken not to identify a "
                                  "stability (kcal/mol). In-range variants sit near 0.31; the flagged population "
                                  "is bimodal, so 2 and 3 select nearly the same rows.")
    floor_group.add_argument('--floor_x_max', type=float, default=-1.5,
                             help="A variant counts as expected-to-be-pinned when its additive expectation "
                                  "dG_wt + ddG_A + ddG_B is below this.")
    floor_group.add_argument('--floor_min_n', type=int, default=20,
                             help="Expected-pinned variants a library needs before its floor is believed.")
    floor_group.add_argument('--floor_max_sd', type=float, default=0.5,
                             help="They must agree to within this SD (kcal/mol) to count as pinned at a common "
                                  "value rather than merely destabilised.")
    floor_group.add_argument('--floor_percentile', type=float, default=5.0,
                             help="Nth_percentile_dG, computed and logged per library for COMPARISON only; it is "
                                  "a quantile of the library's own sensitivity, not of the readout, and is not "
                                  "what the link or the censoring use.")
    data_group.add_argument('--mut_structures_root', type=str, default='/home/sareeves/software/esm-msr/data/tsuboyama/FINAL_results/')
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
    log_group.add_argument('--val_dump', action=argparse.BooleanOptionalAction, default=True,
                           help="Write per-item validation predictions to val_dump_<tag>.npz each epoch.")
    log_group.add_argument('--val_cycle_passes', action=argparse.BooleanOptionalAction, default=False,
                           help="Run the two extra validation passes (WT adapter on the mutated sequence, MT "
                                "adapter on the wild type). Needed for the 'ctx' control head of the epistasis "
                                "metrics; costs one extra forward pass per validation batch.")

    args, remaining_argv = parser.parse_known_args()

    # A subset capped to 0 contributes nothing, so don't pay to load it; one left unrestricted must be loaded.
    subset_flag_map = {'single': 'incl_singles', 'double': 'incl_doubles',
                       'cond': 'incl_cond', 'native_cond': 'incl_native_cond'}
    for subset_key, flag_name in subset_flag_map.items():
        cap_val = args.subset_caps.get(subset_key)
        setattr(args, flag_name, cap_val is None or cap_val > 0)

    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)

    return args
