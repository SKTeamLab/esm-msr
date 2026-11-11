"""The training command line.

Training is block training (``esm_msr.blocks``), Tobit regression, ListMLE on stop-gradient calibrated scores, one dual-adapter model: what used to be flags for the alternatives is gone, not
hidden. There is no compatibility with the old command lines or checkpoints' ``hparams.yaml``: an unknown flag is an error.

* ``--config recipe.yaml`` - YAML defaults under the command line. Keys are flag names; the command line wins. ``configs/recipe.yaml`` is the recipe of ``tfl_T_p0_fw_r05_tm1``.
* The constants below are the values every run used; they are code, not hyperparameters, and are not logged.
* The validation metric that picks checkpoints is not a flag: ``monitor_metric(hp)`` reads the validation regime the MT head was trained on.
"""
import argparse
import logging
import os

DEFAULT_CACHE_VERSION = 'c6'            # data.MutationStabilityDataset.CACHE_VERSION (a test pins them together)
SUBSETS = ('single', 'double')          # what training reads. Mutant-background libraries are singles to both heads (--exclude_mutbg leaves them out)

# What every run used. Not flags.
ASSAY_FLOOR = 0.0                       # kcal/mol of absolute dG: the unfolded state; the teacher-forcing cut when neither --tf_cut nor --rank_censor_threshold names one
CENSOR_RANGE = True                     # rows the assay only bounded ('<-1', '>5') are admitted as censored items rather than dropped (they would select the double set on its outcome)
GRAD_CLIP_NORM = 1.0                    # global L2 clip, applied to the WT parameters and the MT parameters separately
CALIB_LR_MULT = 5.0                     # the calibration heads learn this much faster than the adapters
COSINE_MIN_LR_RATIO = 0.05              # the cosine schedule ends at this fraction of the peak rate
LORA_TARGET_MODE = 'expanded'           # QKV and FFN of every transformer block, both adapters
VAL_BATCH_SIZE = 64
EXTERNAL_VAL_SETS = ('ptmuld', 's461', 'ssym')    # scored per loader but held out of every aggregate (out-of-distribution benchmarks)


def _threshold(v):
    return None if str(v).lower() in ('none', 'null', '') else float(v)


def _hp_get(hp, key, default=None):
    return hp.get(key, default) if hasattr(hp, 'get') else getattr(hp, key, default)


def monitor_metric(hp) -> str:
    """The validation statistic that picks checkpoints and drives early stopping: rho_combined of the regime the MT head trained on (unmasked when it read its site visible more often than
    masked, the main masked regime otherwise)."""
    p_mt = float(_hp_get(hp, 'block_mask_mt', 1.0))
    return 'val_unmasked_rho_combined_avg' if p_mt <= 0.5 else 'val_rho_combined_avg'


def convergence_metric(hp, head: str) -> str:
    """The validation statistic whose plateau starts ``head``'s ('wt' or 'mt') cosine decay: ``--{head}_convergence_metric`` if set, else the head's default metric
    (``rho_singles_wt`` for WT, ``epi_cell_mag_comb`` for MT) in the validation regime that head trained on (``unmasked_`` when ``block_mask_{head} <= 0.5``, else masked)."""
    if head not in ('wt', 'mt'):
        raise ValueError(f"head must be 'wt' or 'mt', got {head!r}")
    override = _hp_get(hp, f'{head}_convergence_metric', None)
    if override:
        return str(override)
    p = float(_hp_get(hp, f'block_mask_{head}', 1.0))
    prefix = 'unmasked_' if p <= 0.5 else ''
    if head == 'wt':
        return f'val_{prefix}rho_singles_wt_avg'
    return f'val_{prefix}epi_cell_mag_comb'


def parse_arguments(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Train ESM3 Dual-LoRA Stability Model", allow_abbrev=False)
    parser.add_argument('--config', type=str, default=None,
                        help="A YAML file of defaults (keys are flag names without the dashes). The command line wins.")

    run = parser.add_argument_group("Run")
    run.add_argument('--experiment_name', type=str, required=True)
    run.add_argument('--version', type=str, default=None)
    run.add_argument('--seed', type=int, default=1)
    run.add_argument('--log_dir', type=str, default='./logs')
    run.add_argument('--resume_from_checkpoint', type=str, default=None,
                     help="A last.ckpt to resume from: picks up the epoch, optimizer state and schedule counters. The run must be relaunched with the SAME flags - the checkpoint does not carry them.")
    run.add_argument('--comet', action=argparse.BooleanOptionalAction, default=True,
                     help="Log to Comet. The key is read from the environment (COMET_API_KEY) or ~/.comet.config, never from the command line, "
                          "so it cannot land in hparams.yaml or a process listing. No key available: a warning, and no Comet.")
    run.add_argument('--comet_project_name', type=str, default="esm-msr-oct24-final")
    run.add_argument('--comet_experiment_key', type=str, default=None,
                     help="Continue logging into an EXISTING Comet experiment instead of creating a new one. Pair it with "
                          "--resume_from_checkpoint, or a resumed run's curves land in a second experiment.")
    run.add_argument('--comet_offline', action='store_true',
                     help="Write the Comet experiment to <log_dir>/comet_offline instead of uploading it: for nodes without internet. "
                          "Upload afterwards from a node that has it: comet upload <log_dir>/comet_offline/*.zip")

    data = parser.add_argument_group("Data (a prebuilt cache; scripts/build_cache.py makes one)")
    data.add_argument('--cache_path', type=str, default=os.environ.get('ESM_MSR_CACHE_PATH', './data_cache'))
    data.add_argument('--cache_version', type=str, default=None,
                      help=f"The cache version this run reads (default {DEFAULT_CACHE_VERSION}). Pin it: a launcher that does not is handed "
                           "whatever the code's default is by the time it starts.")
    data.add_argument('--rank_censor_threshold', type=_threshold, default=0.5,
                      help="Items whose measured dG is at or below this (kcal/mol, in every library) are tied at the bottom of their ListMLE list: their numbers are real (the Tobit "
                           "likelihood and the posterior teacher-forcing label use them as they are), only their order among themselves is not trusted. 'none' ties nothing. It is also the "
                           "teacher-forcing cut when --tf_label measured and --tf_cut is unset.")
    data.add_argument('--exclude_mutbg', action=argparse.BooleanOptionalAction, default=False,
                      help="Leave the mutant-background libraries (one introduced mutation, then a scan of single mutants: '2M7O_I21A') out of TRAINING. Off (the default): both heads "
                           "train on their measurements as ordinary singles, whose 'wild type' is the background sequence and whose dG_wt is the background's own dG "
                           "(3 training libraries, 3,782 items, 1.7%% of the stream). Validation scores these libraries either way.")
    data.add_argument('--min_double_dg_add', type=_threshold, default=None,
                      help="Omit TRAINING double mutants whose additive stability expectation dG_add = dG_wt + ddG_A + ddG_B is at or below this threshold (kcal/mol; "
                           "e.g. 0.0 keeps only doubles with both singles measured and dG_add > 0). Singles are unaffected, and validation scores all doubles regardless.")
    data.add_argument('--num_workers', type=int, default=4)

    arch = parser.add_argument_group("Architecture")
    arch.add_argument('--lora_rank_mt', type=int, default=16)
    arch.add_argument('--lora_alpha_mt', type=int, default=16)
    arch.add_argument('--lora_dropout_mt', type=float, default=0.1)
    arch.add_argument('--target_mode_mt', type=str, default=LORA_TARGET_MODE,
                      choices=['expanded', 'qkv_outproj', 'baseline', 'all', 'ffn', 'ffn_outproj'],
                      help="LoRA target modules for MT adapter: expanded (QKV+FFN, default), qkv_outproj (attention only: QKV+out_proj, skipping FFN), baseline (QKV only), all (QKV+out_proj+FFN), ffn, or ffn_outproj.")
    arch.add_argument('--lora_rank_wt', type=int, default=2)
    arch.add_argument('--lora_alpha_wt', type=int, default=4)
    arch.add_argument('--lora_dropout_wt', type=float, default=0.1)
    arch.add_argument('--target_mode_wt', type=str, default=LORA_TARGET_MODE,
                      choices=['expanded', 'qkv_outproj', 'baseline', 'all', 'ffn', 'ffn_outproj'],
                      help="LoRA target modules for WT adapter: expanded (QKV+FFN, default), qkv_outproj (attention only: QKV+out_proj, skipping FFN), baseline (QKV only), all (QKV+out_proj+FFN), ffn, or ffn_outproj.")
    arch.add_argument('--incl_sequence_head_wt', action=argparse.BooleanOptionalAction, default=True,
                      help="Include ESM3's sequence LM output head in the WT LoRA targets (default True). Pass --no-incl_sequence_head_wt to freeze.")
    arch.add_argument('--incl_sequence_head_mt', action=argparse.BooleanOptionalAction, default=True,
                      help="Include ESM3's sequence LM output head in the MT LoRA targets (default True). Pass --no-incl_sequence_head_mt to freeze.")

    obj = parser.add_argument_group("Objective")
    obj.add_argument('--lambda_rank_wt', type=float, default=1.0)
    obj.add_argument('--lambda_reg_wt', type=float, default=1.0)
    obj.add_argument('--lambda_rank_mt', type=float, default=1.0,
                     help="Weight of the MT path's rank loss (scored through the combined prediction, trains the MT path only).")
    obj.add_argument('--lambda_reg_mt', type=float, default=1.0,
                     help="Weight of the MT path's regression (scored through the combined prediction, trains the MT path only).")
    obj.add_argument('--rank_list_size', type=int, default=16,
                     help="The reference length of a ListMLE list: every list's loss is rescaled to this many members' worth (loss x rank_list_size / its valid members), so the rank "
                          "term does not depend on how many positions a list holds. It scales the rank term like --lambda_rank_*.")
    obj.add_argument('--tf_mix', type=float, default=1.0,
                     help="How much of an item's WT half comes from the measured label rather than from the WT head's own prediction: 1.0 = the "
                          "label wherever one is usable, 0.0 = never teacher-force, 0.5 = the arithmetic mean of the two. Applies to singles "
                          "too. An item with no usable label falls back to the WT head for any mix.")
    obj.add_argument('--tf_label', type=str, default='measured', choices=['measured', 'posterior'],
                     help="The teacher-forcing label of one measured single. measured: the reading itself above --tf_cut, and below it the WT head's read of that single alone, so its partner keeps its "
                          "measurement. posterior: E[z | reading] under the Tobit likelihood with the WT head's detached read as the prior (losses.tobit_posterior_mean), at every reading, with no cut.")
    obj.add_argument('--tf_cut', type=_threshold, default=None,
                     help="--tf_label measured: a measured single (or a double's two singles) is used only when its absolute dG is ABOVE this. Unset: --rank_censor_threshold, or the assay floor.")
    obj.add_argument('--tf_prior_sd', type=float, default=0.75,
                     help="--tf_label posterior: the width (kcal/mol) of the prior on a single's latent dG around the WT head's read (losses.TF_PRIOR_SD).")
    obj.add_argument('--tobit_sigma_y', type=float, default=0.28, help="The Tobit likelihood (and the posterior label): the assay's noise on a reading, kcal/mol (losses.TOBIT_SIGMA_Y).")
    obj.add_argument('--tobit_mu_f', type=float, default=0.05, help="The mean of the unfolded floor, kcal/mol (losses.TOBIT_MU_F).")
    obj.add_argument('--tobit_sigma_f', type=float, default=0.38, help="The spread of the unfolded floor, kcal/mol (losses.TOBIT_SIGMA_F).")
    obj.add_argument('--lambda_cond_mt', type=float, default=0.0,
                     help="Weight on the MT head's conditional legs trained directly: with masked blocks the row and column passes are the legs; each is scored as the partner single's latent "
                          "plus the calibrated leg against the double's label, with ListMLE lists of the same legs (rank weight lambda_cond_mt * lambda_rank_mt / lambda_reg_mt). Only "
                          "--cond_partner_min_dg applies to it. Unmasked pair blocks have no legs.")
    obj.add_argument('--cond_partner_min_dg', type=float, default=1.0,
                     help="--lambda_cond_mt: the minimum absolute partner stability dG_wt + ddG_partner (kcal/mol) required to supervise a conditional leg.")

    opt = parser.add_argument_group("Optimisation (per-head convergence-gated cosine schedule with linear warmup; no weight decay; gradient clip 1.0 per head)")
    opt.add_argument('--learning_rate', type=float, default=2e-4)
    opt.add_argument('--num_epochs', type=int, default=20)
    opt.add_argument('--lr_warmup_steps', type=int, default=500)
    opt.add_argument('--plateau_patience', type=int, default=2,
                     help="Validations without improvement in a head's convergence metric before that head starts cosine decay (default 2: decay starts after 2 consecutive validations do not "
                          "exceed the previous best; 0 starts cosine decay immediately after warmup).")
    opt.add_argument('--cosine_decay_epochs', type=int, default=5,
                     help="Epochs over which a head's learning rate cosine-decays from peak to COSINE_MIN_LR_RATIO (0.05) once its convergence metric plateaus, before dropping to 0 (freezing). "
                          "The run ends when all trainable parameter groups reach lr 0.")
    opt.add_argument('--wt_convergence_metric', type=str, default=None,
                     help="Validation statistic whose plateau starts the WT head's cosine decay (default: val_unmasked_rho_singles_wt_avg when --block_mask_wt <= 0.5, else val_rho_singles_wt_avg).")
    opt.add_argument('--mt_convergence_metric', type=str, default=None,
                     help="Validation statistic whose plateau starts the MT head's cosine decay (default: val_unmasked_epi_cell_mag_comb when --block_mask_mt <= 0.5, else val_epi_cell_mag_comb).")
    opt.add_argument('--early_stopping_patience', type=int, default=0,
                     help="Stop when the monitored validation statistic has not improved for this many validations. 0 (the default) never early-stops.")

    blk = parser.add_argument_group("Blocks (esm_msr.blocks)")
    blk.add_argument('--block_row_chunk', type=int, default=0,
                     help="Run a block's passes in checkpointed chunks of this many rows (0 = all at once). Only a chunk's activations are alive at a time and the gradient is exact "
                          "(the dropout draw is restored on recompute). Needed (an even number, e.g. 20) when --block_mask_mt makes sites visible.")
    blk.add_argument('--block_positions', type=int, default=8,
                     help="Positions per singles block, and rows (or columns) per rank list of a pair block.")
    blk.add_argument('--block_list_rows', type=int, default=0,
                     help="Positions (rows) per ListMLE list. 0 keeps a singles block as ONE list of B x 20 cells and a pair block's lists at --block_positions rows or columns. "
                          "R >= 1 makes lists of R rows in both (R = 1: each position, or each row or column of a pair block, ranked on its own, 20 cells with the wild type).")
    blk.add_argument('--grad_accum_blocks', type=int, default=1,
                     help="Gradient accumulation, in BLOCKS: K >= 2 adds up the gradients of K consecutive blocks and takes one optimizer step after the K-th. --lr_warmup_steps and the cosine "
                          "length count OPTIMIZER steps; logged train/* losses are the mean over the K blocks; blocks left over at the end of an epoch carry into the next.")
    blk.add_argument('--block_domain_steps', action=argparse.BooleanOptionalAction, default=False,
                     help="One optimizer step is one library: all of its positions as one singles block (--block_positions no longer sizes it) plus all of its pair blocks, their gradients "
                          "summed before the clip. An epoch is one step per library. A pair's gradient is weighted --block_pair_weight / (the library's pairs). The largest training "
                          "domain peaks at 26.9 GB: smaller cards need --block_row_chunk.")
    blk.add_argument('--block_pair_weight', type=float, default=1.0,
                     help="--block_domain_steps: weight of a library's doubles (mean over its pairs) against its singles. Reaches the MT head only. 0 runs no pair blocks, which leaves the "
                          "fully masked MT head without any partner to read: a control, not a recipe.")
    blk.add_argument('--block_mask_wt', type=float, default=1.0,
                     help="Probability that the WT head reads a position with its site MASKED (1, the default, always; 0 = the plain wild-type sequence every time, the unmasked WT readout; "
                          "in between, drawn per position of a singles block and per read of a pair block). An unmasked WT read is the cheap case: every row is the same sequence.")
    blk.add_argument('--block_mask_mt', type=float, default=1.0,
                     help="Probability that the MT head reads a position with its site MASKED (1, the default, always). An unmasked read sees the mutation: each of a position's 19 "
                          "substitutions is its own sequence (19 passes instead of 1), drawn per position of a singles block; a pair block is drawn once per block and, unmasked, reads all "
                          "20 x 20 sequences at both sites (no row / column passes, so no --lambda_cond_mt legs for that block). Expensive: set --block_row_chunk "
                          "(an even number, so a sequence's two reads share a chunk). This probability also decides which validation regime is monitored (monitor_metric).")
    blk.add_argument('--block_marginal_mt', action=argparse.BooleanOptionalAction, default=False,
                     help="Mask all mutated sites of a pair block simultaneously on the MT pass ('marginal' masking: 1 doubly-masked sequence for the 19x19 doubles plus the 2 singly-masked "
                          "sequences for the identity row and column) instead of one site at a time with the partner visible ('independent'). Validation and inference score the MT pass with "
                          "mask_strategy='marginal'.")

    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            parser.error(f"--config {pre_args.config}: unknown keys {unknown}")
        parser.set_defaults(**cfg)

    args = parser.parse_args(argv)

    if args.cache_version is None:
        args.cache_version = DEFAULT_CACHE_VERSION

    # ---- validation
    if not 0.0 <= args.tf_mix <= 1.0:
        parser.error(f"--tf_mix must be in [0, 1] (0 = never teacher-force, 1 = always use the measured additive label where one is usable); got {args.tf_mix}")
    if args.rank_list_size < 1:
        parser.error("--rank_list_size must be >= 1")
    for flag in ('tf_prior_sd', 'tobit_sigma_y', 'tobit_sigma_f'):
        if getattr(args, flag) <= 0:
            parser.error(f"--{flag} must be positive")
    if args.lambda_cond_mt < 0:
        parser.error("--lambda_cond_mt must be >= 0")
    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
    if args.tf_label != 'measured' and args.tf_cut is not None:
        parser.error(f"--tf_label {args.tf_label} has no cut: every single keeps a label, so --tf_cut means nothing")
    if args.block_positions < 1:
        parser.error("--block_positions must be >= 1")
    if args.block_list_rows < 0:
        parser.error("--block_list_rows must be >= 0")
    for f in ('block_mask_wt', 'block_mask_mt'):
        if not 0.0 <= getattr(args, f) <= 1.0:
            parser.error(f"--{f} is a probability in [0, 1]")
    if args.block_marginal_mt and args.block_mask_mt < 1.0:
        parser.error("--block_marginal_mt requires --block_mask_mt 1.0")
    if args.block_marginal_mt and args.lambda_cond_mt > 0:
        parser.error("--block_marginal_mt has no conditional legs (--lambda_cond_mt requires independent row/col passes)")
    if args.block_domain_steps and args.grad_accum_blocks > 1:
        parser.error("--block_domain_steps sets the step itself: leave --grad_accum_blocks at 1")
    if args.block_pair_weight < 0:
        parser.error("--block_pair_weight must be >= 0")
    if args.block_pair_weight != 1.0 and not args.block_domain_steps:
        parser.error("--block_pair_weight needs --block_domain_steps (the clip cancels a weight on a pair block's own step)")
    if args.grad_accum_blocks < 1:
        parser.error("--grad_accum_blocks must be >= 1")
    if args.plateau_patience < 0:
        parser.error("--plateau_patience must be >= 0")
    if args.cosine_decay_epochs < 1:
        parser.error("--cosine_decay_epochs must be >= 1")
    return args
