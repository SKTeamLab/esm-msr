import argparse
import os
import logging
from esm_msr.data import ProteinCyclingBatchSampler

class ParseSubsetCaps(argparse.Action):
    """
    Parses 'key=value' pairs into a dictionary.
    Defaults keys to 0.0, except 'single' which defaults to None.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        valid_keys = ProteinCyclingBatchSampler.SUBSET_ORDER
        caps = {k: 0.0 for k in valid_keys}
        caps['single'] = None

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
    loss_group.add_argument('--rank_loss', type=str, default='listmle')   
    loss_group.add_argument('--reg_loss', type=str, default='mse')
    loss_group.add_argument('--huber_delta', type=float, default=1.0)
    loss_group.add_argument('--lambda_rank_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_rank_combined', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_combined', type=float, default=0.0)
    loss_group.add_argument('--lambda_epi_combined', type=float, default=0.0)
    loss_group.add_argument('--mt_reg_mask', type=str, default='all', choices=['all', 'doubles'])
    loss_group.add_argument('--double_weight', type=float, default=1.0)
    loss_group.add_argument('--reversion_weight', type=float, default=0.5)
    loss_group.add_argument('--mut_ctx_weight', type=float, default=0.5)
    loss_group.add_argument('--weight_decay', type=float, default=0)
    loss_group.add_argument('--residual_wd', type=float, default=1e-5)
    loss_group.add_argument('--calib_lr_mult', type=float, default=20.0)
    loss_group.add_argument('--residual_lr_mult', type=float, default=0.1)
    loss_group.add_argument('--detach_ensemble_input', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--detach_calibration', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--detach_regression', action=argparse.BooleanOptionalAction, default=False)
    loss_group.add_argument('--zero_epistasis_for_singles', action=argparse.BooleanOptionalAction, default=True,
                            help="Hardcode residual epistasis to 0 for single mutations and exclude them from epistasis loss.")
    loss_group.add_argument('--link', choices=['none', 'softclamp'], default='none',
                            help="Monotone saturating link between latent stability and the assay's observed dG (esm_msr.link, from devel). With "
                                 "'softclamp' every regression whose target is a measurement of the item itself is done on the observed scale, "
                                 "observed ddG = h(dG_wt + latent) - dG_wt, with h a soft floor/ceiling shared by all libraries: the WT regression "
                                 "on singles and the combined (teacher-forced ensemble) regression on singles and doubles. The WT regression of a "
                                 "double onto the sum of its singles stays latent (that target is a sum of two measurements, not a measurement of "
                                 "the double). Rank losses are unchanged (h is monotone); items whose library has no measured dG_wt leave the "
                                 "regressions. 'none' (default) is the released objective exactly.")
    loss_group.add_argument('--tf_singles', choices=['measured', 'latent_link', 'predicted'], default='measured',
                            help="What the teacher-forced WT half of the combined prediction is built from (--lora_mode ensemble). 'measured' "
                                 "(released): the measured singles (a double's sum of its two measured singles). 'latent_link' (needs --link): the "
                                 "measured singles taken back to the latent scale through the inverse of the learned link, h^-1(dG_wt + ddG_X) - "
                                 "dG_wt (detached; near-floor singles, whose measurement is compressed, become deeper); the WT regression of a "
                                 "double onto the sum of its singles uses the same latent sum. 'predicted' (needs --link): the model's own latent "
                                 "single predictions (the WT pass, detached), summed per mutation (the WT pass of a double plus one calibration "
                                 "bias per extra mutation); no single label is read for the forcing.")
    loss_group.add_argument('--tf_latent_max_shift', type=float, default=0.5,
                            help="With --tf_singles latent_link: a single is forced at most this far (kcal/mol) below its measurement, "
                                 "max(h^-1(dG_wt + ddG) - dG_wt, ddG - shift). The inverse is unbounded where h is flat (at or below the floor); "
                                 "the lift test supports about 0.4-0.5 kcal/mol of compression for singles measured at dG 0 to 0.5.")
    loss_group.add_argument('--nonspecific_term', choices=['none', 'fx'], default='none',
                            help="Nonspecific epistasis as its own term (needs --link). 'fx': a learned piecewise-linear f(x) added to every "
                                 "multi-mutant's latent before the link, x = dG_wt + the additive latent ddG (the forced WT half, detached); "
                                 "singles never get it, so the link keeps modelling the readout. At validation comb gains f and the MT readout 2f. "
                                 "Kept in checkpoints as ns_head.*; not yet read by inference_scripts.")
    loss_group.add_argument('--ns_knot_lo', type=float, default=-3.0, help="Lowest knot of f (kcal/mol of additive dG); f is constant below.")
    loss_group.add_argument('--ns_knot_hi', type=float, default=5.0, help="Highest knot of f; f is constant above.")
    loss_group.add_argument('--ns_knot_step', type=float, default=0.5, help="Knot spacing of f (kcal/mol).")
    loss_group.add_argument('--ns_lr', type=float, default=5e-3, help="Learning rate of f (no weight decay, no warm-up, never cut by the plateau).")
    loss_group.add_argument('--link_floor_file', default=None,
                            help="JSON of per-library floor offsets (baseline/estimate_floors.py), added to the link's floor knee only. The "
                                 "assay infers dG against a K50 predicted for the UNFOLDED sequence, so an unfolded variant reads the error of "
                                 "that prediction: a constant of the library and protease (median +0.02, SD 0.31 kcal/mol over 96 libraries, "
                                 "standard error 0.05, uncorrelated between the two proteases, unpredictable from dG_wt). One shared floor "
                                 "fitted to that mixture is both too low and too soft. The offsets are data, like dG_wt, and add no "
                                 "parameters; the learned 'lo' becomes the floor of a typical library. Libraries absent from the file get 0. "
                                 "No effect without --link.")
    loss_group.add_argument('--link_lo', type=float, default=-1.0, help="Initial floor of the link (kcal/mol); the assay reports -1.")
    loss_group.add_argument('--link_hi', type=float, default=5.0, help="Initial ceiling of the link (kcal/mol); the assay reports 5.")
    loss_group.add_argument('--link_tau', type=float, default=0.5, help="Initial softness (kcal/mol) of both knees of the link.")
    loss_group.add_argument('--link_learn_bounds', action=argparse.BooleanOptionalAction, default=True,
                            help="Let the link's floor and ceiling move off the initial values (the temperatures always learn).")
    loss_group.add_argument('--link_lr', type=float, default=5e-3,
                            help="Learning rate of the link's four parameters (no weight decay, no warm-up, never cut by the plateau).")

    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=0.3)
    calibration_group.add_argument('--shared_bias_init', type=float, default=None)

    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--subset_size', type=int, default=16)
    rank_group.add_argument('--invert_list_loss', action=argparse.BooleanOptionalAction, default=False)

    mask_group = parser.add_argument_group("Masking Strategy")
    mask_group.add_argument('--premask_coords', action=argparse.BooleanOptionalAction, default=False)
    mask_group.add_argument('--val_cycle_passes', action=argparse.BooleanOptionalAction, default=False,
                            help="VALIDATION ONLY: also run the two non-native passes on every validation item (the WT adapter on the mutated sequence asked "
                                 "about the reverse mutation, the MT adapter on the wild-type sequence), so that the control head ctx = (WT + ~WT)/2 of the "
                                 "epistasis metrics (val_epi_*_ctx, esm_msr.epi_metrics) exists; both legs are written to the validation dump. Does not touch training.")
    mask_group.add_argument('--mask_strategy', type=str, choices=["marginal", "chain"], default=None)

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--num_epochs', type=int, default=20)
    train_group.add_argument('--learning_rate', type=float, default=2e-4)
    train_group.add_argument('--lr_warmup_steps', type=int, default=250)
    train_group.add_argument('--mt_lora_delay_steps', type=int, default=0)
    train_group.add_argument('--calib_delay_steps', type=int, default=0)
    train_group.add_argument('--lr_total_steps', type=int, default=None)
    train_group.add_argument('--batch_size', type=int, default=256)
    train_group.add_argument('--micro_batch_size', type=int, default=16)
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
    train_group.add_argument('--wt_convergence_metric', type=str, default='rho_wt')
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
    data_group.add_argument('--incl_mut_ctx', action=argparse.BooleanOptionalAction, default=False)
    data_group.add_argument('--incl_reversions', action=argparse.BooleanOptionalAction, default=False)
    
    data_group.add_argument('--subset_caps', nargs='*', action=ParseSubsetCaps, default=default_caps, 
                            help="Caps for data subsets (e.g., double=0.6 over_and_back=0.1). Defaults to 0 for all except 'single' (None).")
    data_group.add_argument('--mut_structures_root', type=str, default='/home/sareeves/software/esm-msr/data/tsuboyama/FINAL_results/')
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
        'mut_ctx': 'incl_mut_ctx',
        'reversion': 'incl_reversions'
    }

    if args.subset_caps:
        for subset_key, flag_name in subset_flag_map.items():
            cap_val = args.subset_caps.get(subset_key)
            # If None (unrestricted) or > 0, the subset must be included
            is_included = (cap_val is None or cap_val > 0)
            setattr(args, flag_name, is_included)
    
    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")
    if args.link != 'none' and args.lambda_epi_combined > 0:
        parser.error("--lambda_epi_combined regresses a latent dddG onto the measured (saturated) one; it is not defined with --link")
    if args.nonspecific_term != 'none' and args.link == 'none':
        parser.error("--nonspecific_term needs --link (f acts on the latent scale, before the link)")
    if args.tf_singles != 'measured' and (args.link == 'none' or args.lora_mode != 'ensemble'):
        parser.error(f"--tf_singles {args.tf_singles} needs --link (the forcing is on the latent scale) and --lora_mode ensemble")
    if args.link_floor_file is not None:
        if args.link == 'none':
            parser.error("--link_floor_file offsets the link's floor knee; it needs --link")
        if not os.path.exists(args.link_floor_file):
            parser.error(f"--link_floor_file not found: {args.link_floor_file} (build it with baseline/estimate_floors.py)")
    if args.link != 'none' and (getattr(args, 'incl_mut_ctx', False) or getattr(args, 'incl_reversions', False)):
        parser.error("--link scores an item as h(dG_wt + latent) - dG_wt, which holds for singles and doubles measured from the library's "
                     "starting sequence only; mut_ctx and reversion items are not supported with it")

    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
        
    return args