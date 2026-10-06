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


# Longest possible flip column: one row per substitution at the scored position (20 residues minus the wild type).
MAX_COLUMN_LEN = 19

_ANY = object()
# Flags removed from the training CLI. They are still ACCEPTED, so older launch scripts and the commands in docs/ keep working: a retired
# flag given the value it used to default to (or any value, when it no longer matters) is ignored with a warning; given a value that used to
# change behaviour it is an error, because silently ignoring it would train a different model from the one the command describes.
# name -> (type, value that is harmless to ignore, or _ANY, why it went)
RETIRED_FLAGS = {
    'lora_mode': (str, 'ensemble', "the 'corrector' mode was only ever read by the legacy combined objective, which is gone"),
    'lambda_rank_combined': (float, 0.0, "the legacy combined (teacher-forced ensemble) objective was removed"),
    'lambda_reg_combined': (float, 0.0, "the legacy combined (teacher-forced ensemble) objective was removed"),
    'lambda_epi_combined': (float, 0.0, "the legacy combined (teacher-forced ensemble) objective was removed"),
    'detach_ensemble_input': (bool, _ANY, "it only acted inside the legacy combined objective"),
    'mt_reg_mask': (str, _ANY, "it only acted inside the legacy combined objective"),
    'zero_epistasis_for_singles': (bool, _ANY, "it only acted inside the legacy combined objective"),
    'detach_calibration': (bool, _ANY, "it was never read (--detach_regression was what reached the model)"),
    'detach_regression': (bool, False, "cutting the adapters off from the regression terms also cuts them off from the component losses"),
    'lambda_int_mt': (float, 0.0, "the interaction loss is now the interaction part of the MT regression: use --mt_comp_int (about 1 + 2.8 * old weight)"),
    'double_weight': (float, _ANY, "doubles are no longer a training subset: they enter as 'cond' items (see esm_msr.routing)"),
    'reversion_weight': (float, _ANY, "reversions are no longer a training subset"),
    'huber_delta': (float, _ANY, "regression is plain MSE; saturation and censoring are handled by the link and the hinge"),
    'reg_loss': (str, 'mse', "regression is plain MSE; saturation and censoring are handled by the link and the hinge"),
    'rank_loss': (str, 'listmle', "ListMLE (censored Plackett-Luce) is the only rank loss"),
    'invert_list_loss': (bool, False, "never used"),
    'dedup_backbone': (bool, True, "always on: one backbone forward per unique input row"),
    'flip_group_units': (bool, _ANY, "MT micro-batches are always cut at position-pair boundaries when a flip loss is on"),
    'flip_align_units': (bool, _ANY, "MT micro-batches are always cut at position-pair boundaries when a flip loss is on"),
    'flip_pair_groups': (int, _ANY, "derived: when a --mt_comp_* weight differs from 1, micro_batch_size // 19 columns of a pair travel together"),
    'int_min_rows': (int, 4, "fixed at 4"),
    'int_min_cols': (int, 2, "fixed at 2"),
    'freeze_wt_after_epoch': (int, 1000, "use --wt_early_stop_patience"),
    'freeze_wt_on_convergence': (bool, False, "use --wt_early_stop_patience"),
    'wt_convergence_patience': (int, _ANY, "use --wt_early_stop_patience"),
    'wt_convergence_metric': (str, 'rho_wt_valid', "the WT head's early stop always reads val_rho_wt_valid_avg"),
    'use_plddt': (bool, False, "pLDDT is never passed to ESM3"),
    'residual_wd': (float, _ANY, "never read"),
    'residual_lr_mult': (float, _ANY, "never read"),
    # derived from --subset_caps, never an independent choice
    'incl_singles': (bool, _ANY, "derived from --subset_caps"),
    'incl_doubles': (bool, _ANY, "derived from --subset_caps"),
    'incl_cond': (bool, _ANY, "derived from --subset_caps"),
    'incl_reversions': (bool, _ANY, "derived from --subset_caps"),
    'incl_native_cond': (bool, _ANY, "derived from --subset_caps"),
}

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
    loss_group.add_argument('--lambda_rank_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_reg_wt', type=float, default=0.0)
    loss_group.add_argument('--lambda_mt_cell', '--lambda_reg_mt', dest='lambda_mt_cell', type=float, default=1.0,
                            help="Cell-level regression of the MT pass on the observed scale (formerly --lambda_reg_mt): the single anchors, every cond / "
                                 "native_cond item outside a complete position-pair block, and, when all three --mt_comp_* weights are 1, every cell. It "
                                 "scales the component losses below too. Keep > 0: it is the only term that gives the MT pass an absolute scale. Its "
                                 "targets are the noisy derived conditionals, so down-weight with --cond_weight rather than zeroing.")
    loss_group.add_argument('--lambda_mt_colrank', '--lambda_rank_mt', dest='lambda_mt_colrank', type=float, default=1.0,
                            help="Within-column rank loss on the MT pass (formerly --lambda_rank_mt): for each flip column (same scored position, "
                                 "same partner identity, varying substitution) impose the measured ordering. Invariant to the assay's monotone response "
                                 "and to the dynamic-range floor by construction, so it cannot be satisfied by learning assay saturation. It sees "
                                 "each substitution's own effect and the interaction, and nothing that is constant within a column.")
    loss_group.add_argument('--mt_comp_offset', type=float, default=1.0,
                            help="Weight of the position-pair OFFSET part of the MT regression error, on complete position-pair blocks. The squared "
                                 "error of a block (prediction - measurement, observed scale) is split exactly into offset, substitution effects and "
                                 "interaction; 1 for all three is the plain cell regression. Set any of the three to a value other than 1 and the sampler "
                                 "packs micro_batch_size // 19 columns of one position pair together (so micro_batch_size must be >= 38) and "
                                 "micro-batches are cut at pair boundaries. A complete block needs >= 4 rows and >= 2 columns.")
    loss_group.add_argument('--mt_comp_subst', type=float, default=1.0,
                            help="Weight of the SUBSTITUTION-EFFECT part (the mean error of each scored substitution over the block's partners, plus the "
                                 "same for each partner over the block's substitutions). See --mt_comp_offset.")
    loss_group.add_argument('--mt_comp_int', type=float, default=1.0,
                            help="Weight of the INTERACTION part (the error left after the offset and both substitution effects are removed). "
                                 "0 drops the interaction from the regression altogether. See --mt_comp_offset. Replaces --lambda_int_mt, which "
                                 "was added on top of the regression with a different normalisation: an old weight L is about 1 + 2.8 * L here "
                                 "(30 ~ 85, 300 ~ 850) for batches like those of the I-series.")
    loss_group.add_argument('--flip_list_min', type=int, default=4,
                            help="Minimum members for a flip column to contribute to --lambda_mt_colrank. Below ~4 the "
                                 "ordering carries little information and the gradient is mostly noise.")
    loss_group.add_argument('--subfloor_rank_only', action=argparse.BooleanOptionalAction, default=True,
                            help="For double-derived items below --min_additive_dG, mark them rank-only instead of "
                                 "dropping them: kept in the rank losses, withheld from the regression losses. The "
                                 "assay reports sub-floor doubles with no flag; roughly half are unidentifiable fits "
                                 "but a third are genuine compensation, so dropping them discards real data. Their "
                                 "ordering is informative while their absolute value is not.")

    loss_group.add_argument('--censor_floor', type=float, default=None,
                            help="Censored ranking for --lambda_mt_colrank. Flip-column items whose MEASURED dG is at or "
                                 "below this value are pinned at the assay's dynamic-range floor, so their order among "
                                 "themselves is unknown. They are treated as tied at the bottom: every above-floor item "
                                 "must still rank above them, but their relative order costs nothing (censored "
                                 "Plackett-Luce; the WT head's rank loss is untouched). Needs the v6 cache. Typical "
                                 "values: 0.0, +0.5 (practical floor); -1.0 censors nothing in cache_v6. Default None = off. "
                                 "Unlike --subfloor_rank_only, which flags on the ADDITIVE prediction, this flags on the "
                                 "measured value, so genuine compensators (measured above floor) keep their ordering.")
    loss_group.add_argument('--include_out_of_range', action=argparse.BooleanOptionalAction, default=False,
                            help="Use the variants whose dG the assay reports only as '<-1' (dead, unfolded beyond measurement) or '>5' "
                                 "(hyperstable) as CENSORED items: a dead variant is known to rank below every measured one and a "
                                 ">5 variant above every measured one, even though neither has a usable value. Rank losses tie them at "
                                 "the bottom / top of their list (two-sided censored Plackett-Luce) and regression losses penalise only a "
                                 "prediction on the wrong side of the bound (see --censor_reg_weight). Needs the v7 cache and "
                                 "--rank_loss listmle. Singles feed the WT head, doubles the MT head; validation adds dead/hyperstable AUROCs.")
    loss_group.add_argument('--censor_reg_weight', type=float, default=1.0,
                            help="Weight of the one-sided (hinge) regression term for censored items, relative to ordinary items. "
                                 "0 leaves censored items out of the regression altogether (they still enter the rank losses).")
    loss_group.add_argument('--censor_floor_hinge', action=argparse.BooleanOptionalAction, default=False,
                            help="Also apply the one-sided regression to items made lower-censored by --censor_floor. Off by default: "
                                 "they keep regressing on their measured value, and --censor_floor changes only the rank losses.")
    loss_group.add_argument('--link', choices=['none', 'softclamp'], default='softclamp',
                            help="Monotone saturating link between latent stability and the assay's observed dG (esm_msr.link). With "
                                 "'softclamp', the calibrated prediction is treated as a LATENT additive-in-effects ddG and regression is done on "
                                 "the observed scale, observed ddG = h(dG_wt + latent) - dG_wt, with h a soft floor/ceiling shared by all "
                                 "libraries. The assay's saturation then lives in h instead of being imitated by the adapters, "
                                 "so --min_additive_dG / --subfloor_rank_only (reg_ok) and the one-sided hinge for out-of-range items are no longer "
                                 "needed for regression (reg_ok is ignored while the link is on); ranking is unchanged because h is monotone. Conditional items are "
                                 "scored as the double they came from. Drop --shared_bias_init (the link fixes the zero point) and the legacy "
                                 "--lambda_*_combined terms are not supported with it. 'none' restores the pre-link behaviour exactly (default since 2026-10-05: softclamp).")
    loss_group.add_argument('--link_lo', type=float, default=-1.0, help="Initial floor of the link (kcal/mol); the assay reports -1.")
    loss_group.add_argument('--link_hi', type=float, default=5.0, help="Initial ceiling of the link (kcal/mol); the assay reports 5.")
    loss_group.add_argument('--link_tau', type=float, default=0.5,
                            help="Initial softness (kcal/mol) of the floor and of the ceiling; smaller = sharper knee. Learned.")
    loss_group.add_argument('--link_learn_bounds', action=argparse.BooleanOptionalAction, default=True,
                            help="Let the link's floor and ceiling move off the assay's -1 / 5. The plateau seen by a model trained without "
                                 "out-of-range items sits above -1 because only variants measured above it are kept.")
    loss_group.add_argument('--link_lr', type=float, default=5e-3, help="Learning rate of the link's four parameters (no weight decay).")
    loss_group.add_argument('--mt_single_anchor_weight', type=float, default=0.0,
                            help="Per-item weight for also regressing the MT pass on ordinary singles (the zero-background "
                                 "case of the MT task). 0 disables. Requires --lambda_mt_cell > 0.")
    loss_group.add_argument('--mt_single_anchor_frac', type=float, default=1.0,
                            help="Fraction of the batch's singles to anchor each step, resampled per step. Anchored "
                                 "singles are the dominant cost of the MT pass (~48%% of its backbone rows), so 0.25 "
                                 "cuts total training cost by roughly a third. Weights are scaled by 1/frac so the "
                                 "anchor's expected contribution is unchanged and only its variance rises.")
    loss_group.add_argument('--cond_weight', type=float, default=0.5,
                            help="Per-item REGRESSION weight for derived conditional effects ddG(A|B) (the rank and interaction losses are "
                                 "unweighted). Two are emitted per double; each is a difference of two measurements (with --link, a re-scoring of the one double), so each counts half.")
    loss_group.add_argument('--native_cond_weight', type=float, default=1.0,
                            help="Per-item REGRESSION weight for conditional effects measured directly in a mutant background.")
    loss_group.add_argument('--weight_decay', type=float, default=0)
    loss_group.add_argument('--calib_lr_mult', type=float, default=20.0)

    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=0.3)
    calibration_group.add_argument('--shared_bias_init', type=float, default=None)

    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--wt_list_size', '--subset_size', dest='wt_list_size', type=int, default=16,
                            help="Length of the lists the WT head's rank loss is computed over: the WT block of a batch (all singles of one protein) "
                                 "is cut into consecutive lists of this many. It does not affect the MT head, whose lists are flip columns, and it no "
                                 "longer rounds micro_batch_size. (Formerly --subset_size.)")

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
    train_group.add_argument('--precision', type=str, default="bf16-mixed", choices=["32", "16-mixed", "bf16-mixed", "64"])
    train_group.add_argument('--gpus', type=int, default=1)
    train_group.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'deepspeed_stage_2', 'deepspeed_stage_3', 'fsdp'])
    train_group.add_argument('--seed', type=int, default=42)
    train_group.add_argument('--offline_model', action=argparse.BooleanOptionalAction, default=False)
    
    train_group.add_argument('--freeze_wt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--freeze_mt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--wt_early_stop_patience', type=int, default=0,
                             help="Early stopping of the WT head alone: after this many consecutive validations without a better "
                                  "val_rho_wt_valid_avg (by 1e-4), restore the WT adapter and its calibration head to their best state, freeze them, "
                                  "and keep training the MT head. 0 = the WT head trains for the whole run.")
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
    
    data_group.add_argument('--cond_structure', type=str, default='reuse', choices=['reuse', 'mask', 'model'],
                            help="Structure a conditional ddG(A|B) item conditions on: 'reuse' the parent double's "
                                 "(unmodified WT) structure, 'mask' the WT structure masked at the partner site, or "
                                 "'model' a modeled partner structure when one exists. Baked into the cache.")

    data_group.add_argument('--subset_caps', nargs='*', action=ParseSubsetCaps, default=default_caps,
                            help="Caps for data subsets as a fraction of the unrestricted subsets (e.g., double=0.6 cond=None). Defaults to 0 for all except 'single' (None).")
    data_group.add_argument('--mut_structures_root', type=str, default='/home/sareeves/software/esm-msr/data/tsuboyama/FINAL_results/')
    data_group.add_argument('--min_additive_dG', type=float, default=-1.0,
                            help="Double-derived items (cond) whose additive dG prediction dG(wt)+ddG_A+ddG_B falls at or below this value "
                                 "are withheld from the regression (--subfloor_rank_only, the default: they keep their rank terms) or "
                                 "dropped (--no-subfloor_rank_only). The assay's bounded fit reports no dG below -1, so for those doubles the "
                                 "measured value is obliged to be too high. NO EFFECT with --link: the link absorbs the saturation.")
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
    log_group.add_argument('--comet_project_name', type=str, default="esm-msr-agent-2")
    log_group.add_argument('--log_every_n_steps', type=int, default=25)
    log_group.add_argument('--check_val_every_n_epoch', type=int, default=1)
    log_group.add_argument('--num_sanity_val_steps', type=int, default=0)
    log_group.add_argument('--offline', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--skip_val', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--resume_global_step', type=int, default=0)
    log_group.add_argument('--ckpt_path', type=str, default=None)
    log_group.add_argument('--comet_experiment_key', type=str, default=None)

    retired_group = parser.add_argument_group("Retired flags (accepted so old commands still run; see RETIRED_FLAGS)")
    for name, (kind, _, _) in RETIRED_FLAGS.items():
        if kind is bool:
            retired_group.add_argument(f'--{name}', action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
        else:
            retired_group.add_argument(f'--{name}', type=kind, default=None, help=argparse.SUPPRESS)

    args, remaining_argv = parser.parse_known_args()
    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    for name, (_, harmless, why) in RETIRED_FLAGS.items():
        given = getattr(args, name)
        delattr(args, name)                       # a retired flag never reaches the model's hparams
        if given is None:
            continue
        if harmless is not _ANY and given != harmless:
            parser.error(f"--{name} {given} is no longer supported: {why}.")
        logging.warning(f"--{name} is retired and ignored: {why}.")

    # What the training and validation loaders include follows from --subset_caps (None = every item of the subset, 0 = none).
    for subset_key, flag_name in {'single': 'incl_singles', 'double': 'incl_doubles', 'cond': 'incl_cond',
                                  'reversion': 'incl_reversions', 'native_cond': 'incl_native_cond'}.items():
        cap_val = args.subset_caps.get(subset_key)
        setattr(args, flag_name, cap_val is None or cap_val > 0)
    for retired_subset, flag_name in (('double', 'incl_doubles'), ('reversion', 'incl_reversions')):
        if getattr(args, flag_name):
            parser.error(f"--subset_caps {retired_subset}=...: {retired_subset} items are no longer a training subset. Doubles enter training "
                         f"as their two conditional 'cond' items; reversions were never routed to a head (see esm_msr.routing).")

    # Pair-matrix sampling for the component losses: columns of one position pair travel together, as many as fit in one micro-batch.
    if (args.mt_comp_offset, args.mt_comp_subst, args.mt_comp_int) != (1.0, 1.0, 1.0):
        if args.micro_batch_size < 2 * MAX_COLUMN_LEN:
            parser.error(f"the --mt_comp_* weights need two flip columns of a pair in one micro-batch: micro_batch_size must be at least "
                         f"{2 * MAX_COLUMN_LEN}, got {args.micro_batch_size}.")
        args.flip_pair_groups = args.micro_batch_size // MAX_COLUMN_LEN
    else:
        args.flip_pair_groups = 0

    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
        
    return args