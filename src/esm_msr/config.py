import argparse
import os
import sys
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
    lora_group_mt.add_argument('--lora_rank_mt', type=int, default=16)
    lora_group_mt.add_argument('--lora_alpha_mt', type=int, default=16)
    lora_group_mt.add_argument('--lora_dropout_mt', type=float, default=0.1)
    lora_group_mt.add_argument('--incl_structure_encoder_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--incl_sequence_head_mt', action=argparse.BooleanOptionalAction, default=True)
    lora_group_mt.add_argument('--last_n_layers_mt', type=int, default=0)
    lora_group_mt.add_argument('--target_mode_mt', type=str, default='expanded')
    lora_group_mt.add_argument('--unfreeze_layernorms_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--use_dora_mt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_mt.add_argument('--lora_mode', type=str, default='ensemble', choices=['ensemble', 'corrector'])

    lora_group_wt = parser.add_argument_group("WT LoRA Configuration")
    lora_group_wt.add_argument('--lora_rank_wt', type=int, default=2)
    lora_group_wt.add_argument('--lora_alpha_wt', type=int, default=4)
    lora_group_wt.add_argument('--lora_dropout_wt', type=float, default=0.1)
    lora_group_wt.add_argument('--incl_structure_encoder_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--incl_sequence_head_wt', action=argparse.BooleanOptionalAction, default=True)
    lora_group_wt.add_argument('--last_n_layers_wt', type=int, default=0)
    lora_group_wt.add_argument('--target_mode_wt', type=str, default='expanded')
    lora_group_wt.add_argument('--unfreeze_layernorms_wt', action=argparse.BooleanOptionalAction, default=False)
    lora_group_wt.add_argument('--use_dora_wt', action=argparse.BooleanOptionalAction, default=False)

    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument('--reg_loss', type=str, default='mse', choices=['mse', 'huber'])
    loss_group.add_argument('--huber_delta', type=float, default=1.0)
    loss_group.add_argument('--lambda_rank_wt', type=float, default=1.0)
    loss_group.add_argument('--lambda_rank_mt', '--lambda_rank_combined', dest='lambda_rank_mt', type=float, default=1.0,
                            help="Weight of the MT path's rank loss. Named --lambda_rank_combined until the joint "
                                 "path was retired: it is scored through the combined prediction, but it now trains "
                                 "the MT path only. The old name is still accepted.")
    loss_group.add_argument('--lambda_reg_wt', type=float, default=1.0)
    loss_group.add_argument('--lambda_reg_mt', '--lambda_reg_combined', dest='lambda_reg_mt', type=float, default=1.0,
                            help="Weight of the MT path's regression. Named --lambda_reg_combined until the joint "
                                 "path was retired: it is scored through the combined prediction, but it now trains "
                                 "the MT path only. The old name is still accepted.")
    loss_group.add_argument('--mt_reg_mask', type=str, default='all', choices=['all', 'doubles'])
    loss_group.add_argument('--tf_delink', action=argparse.BooleanOptionalAction, default=True,
                            help="De-link the teacher-forcing label before it is added to a latent prediction. The "
                                 "measured ddG_A + ddG_B is a sum of two SATURATED readings; this inverts each "
                                 "single through the link and sums the results, which is not the same as inverting "
                                 "the sum. Requires --link. A single measured at or below its library floor has no "
                                 "invertible latent value, so it drops out of teacher forcing. DEFAULT ON since "
                                 "2026-10-19: without it the observed-scale label is injected into a latent "
                                 "prediction (a scale mismatch; pass --no-tf_delink for the old behaviour).")
    loss_group.add_argument('--alternate_heads', action=argparse.BooleanOptionalAction, default=False,
                            help="Train one head per batch instead of both: the WT head on singles, the MT head on "
                                 "multi-mutants, alternating by batch. A batch that cannot serve its turn (no "
                                 "doubles, or no singles) takes the other one, so no batch is wasted.")
    loss_group.add_argument('--tf_mix', type=float, default=1.0,
                            help="How much of an item's WT half comes from the measured label rather than from "
                                 "the WT head's own prediction: 1.0 = the label wherever one is usable (the released "
                                 "behaviour), 0.0 = never teacher-force, 0.5 = the arithmetic mean of the two. It "
                                 "applies to SINGLES too (2026-10-19), so 0.0 is a pure non-teacher-forcing arm for "
                                 "the whole stream: with the label the MT head is a total/interaction predictor, "
                                 "with the WT pass it compensates the WT head's error. The label carries the "
                                 "measurement noise of its singles and the floor clipping of either; the WT head is "
                                 "smooth but biased, so this trades variance for bias. An item with no usable label "
                                 "(e.g. a single at its library floor) falls back to the WT head for any mix. Needs "
                                 "ensemble lora_mode and --link.")
    loss_group.add_argument('--combined_mode', type=str, default='mean', choices=['mean', 'route'],
                            help="How the WT and MT heads form the reported prediction. 'mean' is 0.5/0.5 "
                                 "(released). 'sum' was REMOVED 2026-10-19: the MT pass is a sum of per-position "
                                 "LLRs, i.e. a total ddG-like quantity, so it cannot output a pure difference "
                                 "(epistasis) and cannot be zero on singles by construction. 'route' reports WT for "
                                 "singles and MT for multi-mutants, for a standalone double predictor.")
    loss_group.add_argument('--wt_trains_doubles', action=argparse.BooleanOptionalAction, default=False,
                            help="Whether the WT head's own losses see multi-mutants. The default leaves doubles "
                                 "to the MT head alone (they still teacher-force the WT half, so the WT path "
                                 "still shapes a double's prediction - it just takes no gradient from one). Off "
                                 "by default from dc17 on: dc15 put rho_wt_valid at 0.8397 against dc14's 0.8348 "
                                 "with everything else equal, about ten times the 0.0005 between-seed SD, and "
                                 "under --wt_double_target latent a double's WT target was ddG_A + ddG_B anyway, "
                                 "a sum of two single measurements the head also sees on their own.")
    loss_group.add_argument('--double_weight', type=float, default=1.0)
    loss_group.add_argument('--cond_weight', type=float, default=0.5)
    loss_group.add_argument('--native_cond_weight', type=float, default=1.0)
    loss_group.add_argument('--weight_decay', type=float, default=0)
    loss_group.add_argument('--calib_lr_mult', type=float, default=20.0)
    loss_group.add_argument('--grad_clip_norm', type=float, default=0.0,
                            help="Clip the gradient norm to this before each step; 0 disables clipping.")
    # RETIRED flags that launch scripts still pass. They are ACCEPTED so those scripts keep working, warned about, and
    # removed from the namespace below, so they never reach hparams. --detach_ensemble_input is now always on, and
    # --no-detach_ensemble_input is an ERROR rather than a silent no-op, because that value used to train a different
    # model (see the error message below). --rank_loss had one choice, listmle, which is what always runs.
    loss_group.add_argument('--detach_ensemble_input', action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    loss_group.add_argument('--rank_loss', type=str, default=None, help=argparse.SUPPRESS)

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
    link_group.add_argument('--learn_global_floor', action=argparse.BooleanOptionalAction, default=False,
                            help="Learn the shared knee used by libraries that show no floor of their own. It is "
                                 "identifiable from exactly that population and is initialised at the median of "
                                 "the libraries that do have one. Off by default from dc17 on, with "
                                 "--learn_link_tau, --learn_link_floor_bounds and --learn_link_ceiling: the link "
                                 "is then fixed, every one of its numbers comes from the data, and the 'link' "
                                 "optimizer group disappears (configure_optimizers only adds it for parameters "
                                 "that require grad).")
    link_group.add_argument('--link_ceiling', type=float, default=None,
                            help="Add a second soft clamp at this dG (kcal/mol), the top of the assay's reported "
                                 "range; 5.0 matches the '>5' tag. Omitted, the link is floor-only.")
    link_group.add_argument('--learn_link_ceiling', action=argparse.BooleanOptionalAction, default=False,
                            help="Let --link_ceiling move. It is parameterised as a positive span above "
                                 "--link_floor_max, so it can never cross a library's floor.")
    link_group.add_argument('--link_tau_hi', type=float, default=None,
                            help="Starting knee width for the ceiling. Defaults to --link_tau: nothing locates a "
                                 "value for the top of the range the way the pile-up at the bottom locates one.")
    link_group.add_argument('--link_tau_penalty', type=float, default=0.0,
                            help="Weight on sum (tau - 0.59)^2 over the free temperatures, added to the loss, so "
                                 "--learn_link_tau drifts from 0.59 only where the data pays for it.")
    link_group.add_argument('--link_floor_penalty', type=float, default=0.0,
                            help="Weight on ((global_floor - anchor) / --link_floor_prior_sd)^2, added once per step, so a "
                                 "learned global floor (--assay_floor X --learn_global_floor) moves from X only where the "
                                 "data pays for it. The anchor is --assay_floor.")
    link_group.add_argument('--link_floor_prior_sd', type=float, default=0.25,
                            help="Prior SD of the learned global floor around its anchor, kcal/mol (about the "
                                 "between-library spread of practical floors, 0.27).")
    link_group.add_argument('--link_lr', type=float, default=5e-3,
                            help="Learning rate of the link's parameters (no weight decay, no warm-up, and not cut "
                                 "by the plateau scheduler).")
    link_group.add_argument('--nonspecific_term', choices=['none', 'fx'], default='none',
                            help="Nonspecific epistasis as its own term (needs --link). 'fx': a learned "
                                 "piecewise-linear f(x) added to every multi-mutant's latent before the link, "
                                 "x = dG_wt + the additive latent ddG (the forced WT half, detached); singles never "
                                 "get it, so the link keeps modelling the readout. At validation comb gains f and the "
                                 "MT readout 2f, so comb = (WT + MT) / 2 still holds. Kept in checkpoints as "
                                 "ns_head.*; inference_scripts do not read it yet.")
    link_group.add_argument('--ns_knot_lo', type=float, default=-3.0,
                            help="Lowest knot of f (kcal/mol of additive dG); f is constant below.")
    link_group.add_argument('--ns_knot_hi', type=float, default=5.0,
                            help="Highest knot of f (kcal/mol); f is constant above.")
    link_group.add_argument('--ns_knot_step', type=float, default=0.5,
                            help="Knot spacing of f (kcal/mol).")
    link_group.add_argument('--ns_lr', type=float, default=5e-3,
                            help="Learning rate of f (no weight decay, no warm-up, never cut by a plateau).")
    link_group.add_argument('--wt_double_target', type=str, default='latent', choices=['latent', 'linked'],
                            help="What the WT head regresses onto for a MULTI-mutant. 'latent': its unlinked LLR "
                                 "sum against the sum of the measured singles, ddG_A + ddG_B. 'linked': its LLR sum "
                                 "through the link against the measured ddG_AB. These are the only two "
                                 "self-consistent pairings - a linked prediction cannot be compared with a sum of "
                                 "observed singles, which is not a quantity the readout can report. Singles always "
                                 "go through the link against their own measurement.")

    calibration_group = parser.add_argument_group("Calibration Configuration")
    calibration_group.add_argument('--shared_scale_init', type=float, default=0.3)
    calibration_group.add_argument('--shared_bias_init', type=float, default=0.0)
    calibration_group.add_argument('--tie_mt_scale', action=argparse.BooleanOptionalAction, default=True,
                                   help="Tie the MT calibration head's SCALE to the frozen WT head's (the MT head keeps "
                                        "its own bias). The calibrated outputs are the common latent scale; with an "
                                        "independent MT scale the raw-scale rank loss weights the two halves "
                                        "differently from the calibrated combined prediction. Default on; pass "
                                        "--no-tie_mt_scale for the released independent-scale behaviour.")
    calibration_group.add_argument('--per_mutation_bias', action=argparse.BooleanOptionalAction, default=True,
                                   help="Apply the calibration bias once per MUTATED POSITION (bias * n_mut) "
                                        "instead of once per item. The LLR sum being calibrated is itself a sum "
                                        "over the item's positions, so a per-item bias makes a double's "
                                        "prediction sit one bias below the sum of its own singles - measured at "
                                        "0.1065 kcal/mol (L_AB - L_A - L_B = -b). Note the offline ceiling on "
                                        "this: correcting it moved rho_combined by -0.0001 and doubles-only rho "
                                        "by 0.0000, so it is a consistency fix, not an accuracy one.")

    rank_group = parser.add_argument_group("ListMLE Objective Configuration")
    rank_group.add_argument('--subset_size', type=int, default=16)

    mask_group = parser.add_argument_group("Masking Strategy")
    mask_group.add_argument('--mask_strategy', type=str, choices=["marginal", "independent"], default=None)

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument('--comet_experiment_key', type=str, default=None,
                             help="Continue logging into an EXISTING Comet experiment instead of creating a "
                                  "new one. Pair it with --resume_from_checkpoint, or a resumed run's curves "
                                  "land in a second experiment and the two have to be read side by side.")
    train_group.add_argument('--resume_from_checkpoint', type=str, default=None,
                             help="A last.ckpt to resume from: picks up the epoch, optimizer state, both "
                                  "schedulers and the WT/MT freeze state. Note the run must be relaunched "
                                  "with the SAME flags - the checkpoint does not carry them.")
    train_group.add_argument('--num_epochs', type=int, default=20)
    train_group.add_argument('--learning_rate', type=float, default=2e-4)
    train_group.add_argument('--lr_warmup_steps', type=int, default=500)
    train_group.add_argument('--mt_lora_delay_steps', type=int, default=0)
    train_group.add_argument('--calib_delay_steps', type=int, default=0)
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
    train_group.add_argument('--seed', type=int, default=1)
    train_group.add_argument('--offline_model', action=argparse.BooleanOptionalAction, default=False)

    train_group.add_argument('--freeze_wt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--freeze_mt_adapter', action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument('--freeze_link', action=argparse.BooleanOptionalAction, default=False,
                             help="Freeze the link (the WT readout: latent -> observed ddG) for the whole run. It is "
                                  "part of a pre-trained WT model - a frozen WT adapter paired with a link that still "
                                  "moves is not the model the MT head is meant to be scored against - so "
                                  "--frozen_wt_mt_only turns this on.")
    train_group.add_argument('--init_wt_from_checkpoint', type=str, default=None,
                             help="Warm-start the WT model from an existing checkpoint: its WT adapter, its "
                                  "calibration head AND its link (the whole latent -> observed ddG readout) are "
                                  "copied in before training. Accepts a full Lightning checkpoint (what the queue's "
                                  "arms write) or a converged_wt_*.pt file; a checkpoint without link_head keys "
                                  "leaves the link as configured by its flags. Nothing else is loaded - the MT path "
                                  "is always fresh and the nonspecific term stays configured by its own flags (use "
                                  "--resume_from_checkpoint for a true resume). Pair with --freeze_wt_adapter and "
                                  "--freeze_link to keep the loaded model fixed, or use --frozen_wt_mt_only.")
    train_group.add_argument('--frozen_wt_mt_only', action=argparse.BooleanOptionalAction, default=False,
                             help="The 'frozen WT, MT-only' paradigm: requires --init_wt_from_checkpoint, and sets "
                                  "--freeze_wt_adapter, --freeze_link and --no-wt_trains_doubles. The WT model "
                                  "(adapter, calibration head, link) is then fixed and only the MT path (and the "
                                  "nonspecific term) trains. The training stream is whatever the --incl_* flags say: "
                                  "SINGLES ARE KEPT by default since 2026-10-19 (doubles alone are floor-limited and "
                                  "not more informative; pass --no-incl_singles for the old doubles-only stream). "
                                  "Validation always keeps singles and doubles, so rho_wt_valid stays a guard.")
    train_group.add_argument('--freeze_wt_after_epoch', type=int, default=1000)
    train_group.add_argument('--wt_convergence_metric', type=str, default='rho_wt_valid')
    train_group.add_argument('--mt_plateau_patience', type=int, default=1,
                             help="Epochs of no improvement in --mt_convergence_metric before the MT path's own lr "
                                  "is decayed. 0 disables it. Only lora_mt and calib_mt are touched.")
    train_group.add_argument('--mt_convergence_metric', type=str, default='epi_beyond_rho_mt',
                             help="What the MT plateau watches. Default is the MT head's correlation on the part of "
                                  "a double the two singles do not explain - what the MT path exists to learn. It is "
                                  "noisy epoch to epoch, so pair a patience of 1 with a gentle --mt_plateau_factor.")
    train_group.add_argument('--mt_plateau_factor', type=float, default=0.3)
    train_group.add_argument('--mt_plateau_min_lr', type=float, default=1e-7)
    train_group.add_argument('--mt_freeze_after_decays', type=int, default=3,
                             help="Freeze the MT path after this many decays. 0 never freezes it. 3 by default, "
                                  "matching --wt_freeze_after_decays: at --mt_plateau_factor 0.3 the third decay "
                                  "leaves the head at 2.7 percent of its base rate, which is not training. Once "
                                  "both heads are frozen the run stops itself "
                                  "(ESM3EpistasisLightningModule._stop_if_nothing_is_training).")
    train_group.add_argument('--wt_plateau_patience', type=int, default=1,
                             help="Epochs without improvement in --wt_convergence_metric before the WT path's own "
                                  "learning rate is cut by --wt_plateau_factor. 0 disables it, and then nothing "
                                  "decays the WT groups at all: the shared ReduceLROnPlateau that used to, on "
                                  "val_rho_combined_avg, was removed as redundant once each head got its own "
                                  "plateau. That metric kept improving long after the WT head had stopped, so the "
                                  "WT adapter was taking full-rate steps on a metric it was no longer improving.")
    train_group.add_argument('--wt_plateau_factor', type=float, default=0.3,
                             help="Multiplier applied to the WT groups' lr on each decay. 0.3, matching "
                                  "--mt_plateau_factor: with a patience of 1 these metrics are noisy enough that a "
                                  "decay can fire on a wobble, and a gentler step keeps that from being terminal. "
                                  "The dc8 series and dc12 pin 0.1 explicitly and predate this, so they are "
                                  "unaffected; dc14 onward pass 0.3 on both heads.")
    train_group.add_argument('--wt_plateau_min_lr', type=float, default=1e-7)
    train_group.add_argument('--wt_freeze_after_decays', type=int, default=3,
                             help="Freeze the WT path after this many --wt_plateau_patience decays, which also "
                                  "fixes the target the MT path teacher-forces against. 0 never freezes. 3 by "
                                  "default, matching --mt_freeze_after_decays; the dc8 and dc14 series pin 2.")
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
    data_group.add_argument('--native_cond_for_wt', '--native_cond_as_single', dest='native_cond_for_wt',
                            action=argparse.BooleanOptionalAction, default=True,
                            help="Label a mutant-background library's measurements 'single' rather than "
                                 "'native_cond', which is what the released model does: they then train the WT "
                                 "head and count as singles everywhere. Overrides the native_cond subset cap, "
                                 "since no item carries that label. A mutant-background sequence differs from its "
                                 "structure - these are conditional measurements ddG(X | background) - so this is a "
                                 "deliberate reproduction of the released model, not the rigorous routing. The old "
                                 "spelling --native_cond_as_single is an alias.")
    data_group.add_argument('--native_cond_for_mt', action=argparse.BooleanOptionalAction, default=False,
                            help="Let the MT head train on a mutant-background library's singles: label them "
                                 "'native_cond', include the subset (overriding its 0.0 cap), and add them to the "
                                 "MT/combined loss. The MT pass then gets the SAME input the WT pass gets (the "
                                 "library's starting/background sequence), not the fully mutated one, so the item is "
                                 "a forward conditional mutation in a pseudo-WT context whose sequence already "
                                 "differs from the native structure - the MT head's domain, used as a strong "
                                 "regularizer. Independent of --native_cond_for_wt: both can be on (both heads see "
                                 "them), either alone, or neither (excluded).")

    data_group.add_argument('--subset_caps', nargs='*', action=ParseSubsetCaps, default=dict(DEFAULT_SUBSET_CAPS),
                            help=f"Caps for data subsets (e.g. double=0.6 cond=0.1). Defaults: {DEFAULT_SUBSET_CAPS}.")
    floor_group = parser.add_argument_group("Assay Floor (data.floor_evidence)")
    floor_group.add_argument('--censor_floor', action=argparse.BooleanOptionalAction, default=True,
                             help="Treat an item measured at or below its library's floor as a bound rather than a "
                                  "value: regression becomes one-sided and the item's rank is tied at the bottom. "
                                  "Only applies to libraries that show evidence of a floor.")
    floor_group.add_argument('--assay_floor', type=float, default=None,
                             help="Override every library's floor with this one value, for the link AND for "
                                  "censoring (both read item['y_floor']), and assert floor evidence so it "
                                  "reaches the 364 of 411 libraries that have none of their own. Unset, each "
                                  "library uses its own estimate from data.floor_evidence and the rest carry NaN. "
                                  "0.0 is the physical choice - dG = 0 is the unfolded state - and costs the "
                                  "per-library estimates, which are mostly BELOW zero (47 with evidence: median "
                                  "-0.136, min -0.641, 68 percent negative), so it censors about 5 percent more "
                                  "of the training set. Must sit inside [--link_floor_min, --link_floor_max].")
    floor_group.add_argument('--censor_threshold', type=float, default=None,
                             help="Censor every single (and native_cond item) whose measured dG is at or below this "
                                  "value, in every library, bounded by it - instead of censoring at the library's "
                                  "floor. Decouples censoring from the link: dead variants pile up a little ABOVE the "
                                  "knee, so a threshold of a few tenths ties them in the rank loss and stops "
                                  "regressing them as values. The bound is raised to the knee's reading wherever it "
                                  "sits below it (see training.reachable_bounds). A single censored this way also "
                                  "supplies no teacher-forcing label: its doubles take the WT pass. Unset, censoring "
                                  "follows the library floor as before. Needs --censor_floor.")
    floor_group.add_argument('--censor_threshold_double', type=float, default=None,
                             help="The same for multi-mutants (and cond items). Defaults to --censor_threshold. Only "
                                  "the MT/combined loss sees doubles, so this belongs to the MT block.")
    floor_group.add_argument('--censor_gap', action=argparse.BooleanOptionalAction, default=True,
                             help="Also censor an item whose two proteases disagree about its stability by more "
                                  "than --censor_gap_max, bounded by its own reported value. Needs no doubles and "
                                  "no library floor, so unlike --censor_floor it reaches every library.")
    floor_group.add_argument('--censor_range', action=argparse.BooleanOptionalAction, default=False,
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
    log_group.add_argument('--comet_offline', action='store_true',
                           help="Write the Comet experiment to <log_dir>/comet_offline instead of uploading it: for nodes "
                                "without internet (Alliance compute nodes). Upload afterwards from a node that has it: "
                                "COMET_API_KEY=... comet upload <log_dir>/comet_offline/*.zip")
    log_group.add_argument('--log_every_n_steps', type=int, default=25)
    log_group.add_argument('--check_val_every_n_epoch', type=int, default=1)
    log_group.add_argument('--num_sanity_val_steps', type=int, default=0)
    log_group.add_argument('--offline', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--skip_val', type=bool, default=False, action=argparse.BooleanOptionalAction)
    log_group.add_argument('--external_val_sets', type=str, default='ptmuld,s461,ssym',
                           help="Comma-separated validation loaders to score as OUT-OF-DISTRIBUTION benchmarks. They are "
                                "still scored and logged per loader (val_<metric>/<name>) for diagnostics, but held OUT "
                                "of every aggregate: val_*_avg, the pooled global ranking and the epistasis levels. They "
                                "are summarised separately as val_rho_combined_external and "
                                "val_rmse_combined_external (the mean over these loaders). Keeping OOD libraries out of "
                                "the aggregate makes it an in-distribution quantity: comparable across runs and "
                                "consistent with what the training set covers. NOTE this changed on 2026-10-19, so "
                                "aggregates are NOT comparable to runs before it - recompute from the per-library "
                                "columns when an old run must be compared.")
    log_group.add_argument('--val_dump', action=argparse.BooleanOptionalAction, default=True,
                           help="Write per-item validation predictions to val_dump_<tag>.npz each epoch.")
    log_group.add_argument('--val_wt_only', action=argparse.BooleanOptionalAction, default=False,
                           help="Validate the WT head only: no MT forward and no MT half of the cycle passes. For a run "
                                "whose MT adapter is not trained (a WT-only sweep), where every MT prediction is a "
                                "forward through an untrained adapter. The mt and comb columns then report the WT head "
                                "(documented, not silent), the ctx level still comes from --val_cycle_passes, and "
                                "validation costs one forward per batch instead of two (plus one cycle pass instead "
                                "of two).")
    log_group.add_argument('--val_cycle_passes', action=argparse.BooleanOptionalAction, default=True,
                           help="Run the two extra validation passes (WT adapter on the mutated sequence, MT "
                                "adapter on the wild type). Needed for the 'ctx' control head of the epistasis "
                                "metrics; costs one extra forward pass per validation batch.")

    args, remaining_argv = parser.parse_known_args()

    # A subset capped to 0 contributes nothing, so don't pay to load it; one left unrestricted must be loaded.
    # The two spellings are kept in sync in BOTH directions. Without the second direction the flags were silently
    # discarded: `--no-incl_doubles` alone left incl_doubles True, because the cap (None = unrestricted) then won,
    # so the documented flag was a no-op unless the cap happened to agree. Only an EXPLICIT --no-incl_* forces the
    # cap, because incl_cond/incl_native_cond default to False and must not overwrite a requested cap.
    subset_flag_map = {'single': 'incl_singles', 'double': 'incl_doubles',
                       'cond': 'incl_cond', 'native_cond': 'incl_native_cond'}
    passed = set(sys.argv[1:])
    for subset_key, flag_name in subset_flag_map.items():
        cap_val = args.subset_caps.get(subset_key)
        if f'--no-{flag_name}' in passed:
            args.subset_caps[subset_key] = 0.0      # the explicit exclusion wins, and the cap now agrees with it
            cap_val = 0.0
        setattr(args, flag_name, cap_val is None or cap_val > 0)

    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    if args.rank_loss not in (None, 'listmle'):
        parser.error(f"--rank_loss {args.rank_loss}: the only rank loss is listmle, and the flag is retired.")
    # Where every item's floor comes from, recorded in hparams: 'library' (each library's own estimate, NaN where it
    # has none, those fall back to the link's global_floor), 'assay' (--assay_floor, one fixed value on every item) or
    # 'global' (--assay_floor X --learn_global_floor: every item carries NaN, so the knee is the link's global_floor,
    # learned from X - with --link_floor_penalty keeping it near X).
    if args.assay_floor is None:
        args.floor_source = 'library'
    elif args.learn_global_floor:
        args.floor_source = 'global'
        if args.link and args.link_floor_max - args.link_floor_min < 0.2:
            parser.error(f"--learn_global_floor with --assay_floor: the learned floor is clipped to "
                         f"[--link_floor_min, --link_floor_max] = [{args.link_floor_min}, {args.link_floor_max}], which "
                         f"leaves it no room to move; widen the band (e.g. -1.0 1.0)")
        if not args.link_floor_min <= args.assay_floor <= args.link_floor_max:
            parser.error(f"--assay_floor {args.assay_floor} (the learned floor's start and anchor) must sit inside "
                         f"[--link_floor_min, --link_floor_max]")
    else:
        args.floor_source = 'assay'
    if args.floor_source == 'global' and args.censor_floor and args.censor_threshold is None:
        # No item carries a floor in this mode, so floor censoring would silently switch off; keep the population the
        # fixed-floor recipe censors (read at or below --assay_floor) by deriving the threshold from it.
        logging.info(f"learned global floor: censoring at --censor_threshold {args.assay_floor} (= --assay_floor)")
        args.censor_threshold = float(args.assay_floor)
    if args.link_floor_penalty > 0 and args.floor_source != 'global':
        logging.warning("--link_floor_penalty only acts on a learned global floor (--assay_floor X --learn_global_floor)")
    if args.censor_threshold_double is None:
        args.censor_threshold_double = args.censor_threshold
    if (args.censor_threshold is not None or args.censor_threshold_double is not None) and not args.censor_floor:
        parser.error("--censor_threshold / --censor_threshold_double need --censor_floor")
    if args.nonspecific_term != 'none' and not args.link:
        parser.error("--nonspecific_term needs --link (f acts on the latent scale, before the link)")
    if args.link and args.lora_mode == 'ensemble' and not args.per_mutation_bias:
        parser.error("--per_mutation_bias is required with --link and --lora_mode ensemble: the calibrated LLR sum is "
                     "a sum over mutated positions, so a scalar bias makes a double's prediction sit one bias BELOW "
                     "the sum of its two singles (measured 0.1065 kcal/mol on devel-revision), and the teacher-forced "
                     "additive half then disagrees with the shared latent scale. Pass --per_mutation_bias.")
    if not 0.0 <= args.tf_mix <= 1.0:
        parser.error(f"--tf_mix must be in [0, 1] (0 = never teacher-force, 1 = always use the measured additive "
                     f"label where one is usable); got {args.tf_mix}")
    if args.tf_mix < 1.0 and (args.lora_mode != 'ensemble' or not args.link):
        logging.warning(f"--tf_mix {args.tf_mix} only affects ensemble mode with --link (teacher forcing happens "
                        f"nowhere else), so it has no effect on this run")
    if args.detach_ensemble_input is False:
        parser.error(
            "--no-detach_ensemble_input is retired. It let the MT regression and rank terms backpropagate into the "
            "WT adapter, which is why those weights are still named lambda_*_combined. In ensemble mode it could "
            "never have helped: the WT half of the combined prediction is replaced by the measured additive label "
            "wherever one exists (and singles always have one), so the attached path delivered gradient ONLY on "
            "multi-mutants whose additive label was missing or whose singles were measured at the floor - a subset "
            "selected for having the worst single measurements. It also set the WT head two targets that differ by "
            "exactly the epistasis term: its own objective wants ddG_A + ddG_B, the combined objective wants "
            "ddG_AB, and the WT pass can only reconcile them by distorting its per-position terms."
        )
    for retired in ('detach_ensemble_input', 'rank_loss'):
        if getattr(args, retired) is not None:
            logging.warning(f"--{retired} is retired and ignored (detach_ensemble_input is always on; the rank loss is always listmle).")
        delattr(args, retired)                    # a retired flag never reaches the model's hparams

    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)

    # The frozen-WT / MT-only / doubles-only paradigm is a named combination of primitive flags. Deriving it here
    # (rather than branching in the model) means hparams record the EFFECTIVE settings the run used.
    if args.init_wt_from_checkpoint is not None and not os.path.isfile(args.init_wt_from_checkpoint):
        parser.error(f"--init_wt_from_checkpoint: no such file: {args.init_wt_from_checkpoint}")
    # native_cond is a four-way routing decision (routing.py): each head can see it independently.
    #   for_wt only  -> the released behaviour: labelled 'single', the WT loss trains on it, and the combined
    #                   (MT) loss sees it through the ordinary single route.
    #   for_mt       -> labelled 'native_cond', the MT/combined loss trains on it, and the MT PASS input is the
    #                   WT pass's sequence (see forward_partitioned) so it is a forward mutation from the
    #                   background context rather than a second mutation from the true wild type.
    #   for_wt+for_mt-> both heads train on the same items and the same input.
    #   neither      -> excluded.
    if args.native_cond_for_mt:
        args.incl_native_cond = True
        args.subset_caps['native_cond'] = None           # the 0.0 default would exclude exactly what we want
    if not args.native_cond_for_wt and not args.native_cond_for_mt:
        args.incl_native_cond = False
        args.subset_caps['native_cond'] = 0.0
    if args.frozen_wt_mt_only:
        if args.init_wt_from_checkpoint is None:
            parser.error("--frozen_wt_mt_only needs --init_wt_from_checkpoint: the paradigm is a PRE-TRAINED frozen "
                         "WT head, not a randomly initialised one")
        if not args.link:
            parser.error("--frozen_wt_mt_only needs --link: the WT model being frozen includes its link (the "
                         "latent -> observed ddG readout), and the nonspecific term acts before it")
        if not args.freeze_wt_adapter:
            logging.info("--frozen_wt_mt_only: setting --freeze_wt_adapter")
            args.freeze_wt_adapter = True
        if not args.freeze_link:
            logging.info("--frozen_wt_mt_only: setting --freeze_link (the WT readout is the adapter, its calibration "
                         "head AND the link; a moving link would make the frozen WT a different model)")
            args.freeze_link = True
        if args.learn_link_tau or args.learn_global_floor:
            logging.warning("--frozen_wt_mt_only freezes the link, so --learn_link_tau / --learn_global_floor have "
                            "no effect")
        if args.subset_caps.get('double') is not None and args.subset_caps['double'] <= 0:
            parser.error("--frozen_wt_mt_only trains on multi-mutants only, but --subset_caps double=0 excludes "
                         "them: nothing would be left to train on")
        if args.wt_trains_doubles:
            logging.info("--frozen_wt_mt_only: setting --no-wt_trains_doubles")
            args.wt_trains_doubles = False
        # a frozen WT head trains nothing: its losses would be computed every step and never backpropagated.
        if args.lambda_reg_wt > 0 or args.lambda_rank_wt > 0:
            logging.info("--frozen_wt_mt_only: setting --lambda_reg_wt 0 --lambda_rank_wt 0 (the WT path is frozen)")
            args.lambda_reg_wt = 0.0
            args.lambda_rank_wt = 0.0
    elif args.init_wt_from_checkpoint is not None and not args.freeze_wt_adapter:
        logging.warning(f"--init_wt_from_checkpoint given without --freeze_wt_adapter: {args.init_wt_from_checkpoint} "
                        f"is a warm start, and the WT head will keep training from it")

    return args
