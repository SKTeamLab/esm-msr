"""The training command line.

About 55 flags (it was 160; ``docs/trim_audit.md`` has the verdict on each). Three mechanisms keep the old launchers and checkpoints working:

* ``RETIRED`` - a table of removed flags. Given the one value they used to carry as a default, a retired flag is accepted and dropped from
  the namespace (so it never reaches ``hparams.yaml``); given any other value it is an error that says what replaced it. A few are
  translated (``--subset_caps`` -> ``--subsets``, ``--assay_floor`` -> ``--link_floor``, ``--comet_api_key`` -> the environment).
* ``--config recipe.yaml`` - YAML defaults under the command line. Keys are flag names; the command line wins.
* ``DERIVED`` - values the model and the scripts that read ``hparams.yaml`` still need (``per_mutation_bias``, ``lora_mode``),
  recorded as constants.
"""
import argparse
import logging
import os

DEFAULT_CACHE_VERSION = 'c6'            # data.MutationStabilityDataset.CACHE_VERSION (a test pins them together)
SUBSETS = ('single', 'double')          # what --subsets may name. `native_cond` is routed by --native_cond_for_wt / --native_cond_for_mt

# What the model and the readers of hparams.yaml (inference.parse_hparams_to_lora_config, make_fwt_block.py, gate_capacity.py) still need
# to find, written as constants: the code behind them is gone, but a checkpoint trained here must say how its calibration is built.
DERIVED = {'lora_mode': 'ensemble', 'per_mutation_bias': True}

_ANY = object()                          # any value is ignored


def _r(kind, ok=None, why=''):
    return {'kind': kind, 'ok': ok, 'why': why}


# flag -> what it used to be and the only value it may still be given. kind: 'bool' (--x / --no-x), 'val' (--x VALUE), 'flag' (--x, no value).
# ok=_ANY: any value is dropped without comment (a path, a number that governed something now fixed).
RETIRED = {
    # the LoRA spec is fixed (inference reads an old checkpoint's own values from its hparams)
    **{f'{k}_{h}': _r('bool', v, 'a constant now') for h in ('wt', 'mt')
       for k, v in (('use_dora', False), ('unfreeze_layernorms', False), ('incl_structure_encoder', False), ('incl_sequence_head', True))},
    **{f'last_n_layers_{h}': _r('val', 0, 'a constant now') for h in ('wt', 'mt')},
    'lora_mode': _r('val', 'ensemble', 'the MT head is always an ensemble partner of the WT head (corrector removed)'),
    'combined_mode': _r('val', 'mean', 'the reported prediction is always the mean of the two heads (route and sum removed)'),
    'mt_reg_mask': _r('val', 'all', 'a constant now'),
    'detach_ensemble_input': _r('bool', True, 'always on'),
    'rank_loss': _r('val', 'listmle', 'the only rank loss'),
    'rank_ci_margin': _r('val', 0.0, 'uncertainty margins removed'),
    'per_mutation_bias': _r('bool', True, 'always on (a scalar bias puts a double one bias below its two singles)'),
    'tf_delink': _r('bool', True, 'always on'),
    'wt_trains_doubles': _r('bool', False, 'the WT head trains on singles only'),
    'wt_double_target': _r('val', 'latent', 'a constant now'),
    'alternate_heads': _r('bool', False, 'removed (never run)'),
    'shared_scale_init': _r('val', 0.3, 'a constant now'),
    'shared_bias_init': _r('val', 0.0, 'a constant now'),
    'calib_lr_mult': _r('val', 20.0, 'a constant now'),
    # per-subset loss weights and the cond subset
    'double_weight': _r('val', 1.0, 'removed'),
    'cond_weight': _r('val', 0.5, 'the cond subset is gone'),
    'native_cond_weight': _r('val', 1.0, 'removed'),
    'incl_cond': _r('bool', False, 'the cond subset is gone'),
    # the link is two set numbers
    'learn_link_tau': _r('bool', False, 'the link is fixed: --link_floor / --link_tau'),
    'learn_global_floor': _r('bool', False, 'the link is fixed: --link_floor / --link_tau'),
    'learn_link_floor_bounds': _r('bool', False, 'the link is fixed'),
    'learn_link_ceiling': _r('bool', False, 'the upper clamp was removed'),
    'link_ceiling': _r('val', None, 'the upper clamp was removed (the 4-parameter link, dc7, matched the floor-only one, dc8)'),
    'link_tau_hi': _r('val', None, 'the upper clamp was removed'),
    'link_tau_penalty': _r('val', _ANY), 'link_floor_penalty': _r('val', _ANY), 'link_floor_prior_sd': _r('val', _ANY),
    'link_lr': _r('val', _ANY), 'link_floor_min': _r('val', _ANY), 'link_floor_max': _r('val', _ANY),
    'nonspecific_term': _r('val', 'none', 'the nonspecific term was removed (docs/trim_audit.md §2.2)'),
    **{k: _r('val', _ANY) for k in ('ns_knot_lo', 'ns_knot_hi', 'ns_knot_step', 'ns_lr')},
    # censoring and floors
    'censor_gap': _r('bool', _ANY, 'the protease-gap rule was removed: this run does not censor on protease disagreement'),
    'censor_gap_max': _r('val', _ANY),
    'censor_floor': _r('bool', True, 'censoring is --censor_threshold (none = no censoring)'),
    **{k: _r('val', _ANY) for k in ('floor_x_max', 'floor_min_n', 'floor_max_sd', 'floor_percentile')},   # the per-library floor estimator
    # optimiser and schedule
    'grad_clip_norm': _r('val', 0.0, 'removed (never used)'),
    'calib_delay_steps': _r('val', 0, 'removed'),
    'mt_lora_delay_steps': _r('val', 0, 'removed'),
    'freeze_wt_after_epoch': _r('val', 1000, 'removed (the plateau budget freezes the WT head)'),
    'wt_convergence_metric': _r('val', 'rho_wt_valid', 'a constant now'),
    'wt_plateau_min_lr': _r('val', 1e-7, 'removed'),
    'mt_plateau_min_lr': _r('val', 1e-7, 'removed'),
    # data and pipeline plumbing
    'micro_batch_tokens': _r('val', 0, 'removed; --micro_batch_size is the cap'),
    'dataloading': _r('val', 'cycle', 'a constant now'),
    'loader_strategy': _r('val', 'all', 'a constant now'),
    'score_column': _r('val', 'ddG_ML', 'a constant now'),
    'combine_validation': _r('bool', False, 'removed'),
    'max_train_proteins': _r('val', -1, 'removed'),
    'offline': _r('bool', False, 'removed'),
    'offline_model': _r('bool', False, 'an environment variable now: export INFRA_PROVIDER=1'),
    'precision': _r('val', 'bf16-mixed', 'a constant now'),
    'gpus': _r('val', 1, 'one GPU'),
    'strategy': _r('val', 'auto', 'one GPU'),
    'num_sanity_val_steps': _r('val', 0, 'a constant now'),
    'check_val_every_n_epoch': _r('val', 1, 'a constant now'),
    'log_every_n_steps': _r('val', 25, 'a constant now'),
    'load_lora_checkpoint': _r('val', None, 'use --init_wt_from_checkpoint'),
    'load_wt_only': _r('bool', False, 'use --init_wt_from_checkpoint'),
    'checkpoint_path': _r('val', _ANY), 'checkpoint_filename': _r('val', _ANY), 'save_top_k': _r('val', _ANY),
    'monitor_mode': _r('val', 'max', 'a constant now'),
    'freeze_link': _r('bool', _ANY),
    # training reads a prebuilt cache; the raw table and the structures belong to analysis_scripts/build_cache.py
    'raw_data_file': _r('val', _ANY), 'af_model_folder': _r('val', _ANY), 'mut_structures_root': _r('val', _ANY),
    'benchmark_data_path': _r('val', _ANY),
    'remove_spurs_homologs': _r('bool', False, 'a cache-build option'),
    'regenerate_cache': _r('flag', None, 'training never builds a cache: use analysis_scripts/build_cache.py'),
    'mask_strategy': _r('val', None, 'per pass now: --mask_strategy_wt / --mask_strategy_mt'),
}
# retired flags that carry a value worth translating or cross-checking; handled in parse_arguments
_TRANSLATED_VAL = ('subset_caps', 'assay_floor', 'comet_api_key', 'split_file', 'censor_threshold_double', 'reg_loss', 'early_stopping_metric',
                   'wt_plateau_patience', 'wt_plateau_factor', 'wt_freeze_after_decays',
                   'mt_plateau_patience', 'mt_plateau_factor', 'mt_freeze_after_decays')
_TRANSLATED_BOOL = ('incl_singles', 'incl_doubles', 'incl_native_cond', 'freeze_wt_adapter')


def _threshold(v):
    return None if str(v).lower() in ('none', 'null', '') else float(v)


def _same(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _add_retired(parser):
    """Register every removed flag as a hidden option so that an old command line still parses (and is then checked)."""
    for name, spec in RETIRED.items():
        if spec['kind'] == 'bool':
            parser.add_argument(f'--{name}', action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
        elif spec['kind'] == 'flag':
            parser.add_argument(f'--{name}', action='store_true', default=None, help=argparse.SUPPRESS)
        else:
            parser.add_argument(f'--{name}', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--subset_caps', nargs='*', default=None, help=argparse.SUPPRESS)
    for name in _TRANSLATED_VAL:
        if name != 'subset_caps':
            parser.add_argument(f'--{name}', type=str, default=None, help=argparse.SUPPRESS)
    for name in _TRANSLATED_BOOL:
        parser.add_argument(f'--{name}', action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)


def _check_retired(args, parser):
    """Validate and strip every table-driven retired flag; returns the names that were silently ignored."""
    ignored = []
    for name, spec in RETIRED.items():
        v = getattr(args, name, None)
        delattr(args, name)
        if v is None or (spec['kind'] == 'flag' and not v):
            continue
        ok = spec['ok']
        if ok is _ANY:
            ignored.append(name)
            if spec['why']:
                logging.warning(f"--{name} is retired: {spec['why']}.")
            continue
        if (spec['kind'] == 'bool' and v == ok) or (spec['kind'] == 'val' and ok is not None and _same(v, ok)) \
                or (spec['kind'] == 'val' and ok is None and str(v).lower() in ('none', 'null')):
            ignored.append(name)
            continue
        if name == 'censor_floor' and v is False:          # --no-censor_floor: no censoring at all
            logging.warning("--no-censor_floor is retired: translated to --censor_threshold none.")
            args.censor_threshold = None
            continue
        what = f"--no-{name}" if spec['kind'] == 'bool' and v is False else (f"--{name} {v}" if spec['kind'] == 'val' else f"--{name}")
        parser.error(f"{what} is retired and its behaviour is gone: {spec['why']}.")
    return ignored


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
                     help="A last.ckpt to resume from: picks up the epoch, optimizer state, the schedule counters and the WT/MT freeze state. "
                          "The run must be relaunched with the SAME flags - the checkpoint does not carry them. It is the only recovery after a "
                          "crash for the MT adapter, which --init_wt_from_checkpoint (the WT side only) cannot restore.")
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

    data = parser.add_argument_group("Data (a prebuilt cache; analysis_scripts/build_cache.py makes one)")
    data.add_argument('--cache_path', type=str, default=os.environ.get('ESM_MSR_CACHE_PATH', './data_cache'))
    data.add_argument('--cache_version', type=str, default=None,
                      help=f"The cache version this run reads (default {DEFAULT_CACHE_VERSION}). Pin it: a launcher that does not is handed "
                           "whatever the code's default is by the time it starts.")
    data.add_argument('--subsets', type=str, default=None,
                      help="The measured subsets the run trains on, comma-separated: single, double (default both). Validation always scores "
                           "both. Mutant-background libraries are routed by --native_cond_for_wt / --native_cond_for_mt instead.")
    data.add_argument('--censor_threshold', type=_threshold, default=0.5,
                      help="Censor every item whose measured dG is at or below this, in every library, bounded by it (kcal/mol): regression "
                           "becomes one-sided and the item's rank is tied at the bottom. 'none' censors nothing. 0.5 sits a little above the "
                           "link's knee, where dead variants pile up; 0.3 / 0.5 / 0.6 are indistinguishable on the WT head.")
    data.add_argument('--censor_range', action=argparse.BooleanOptionalAction, default=False,
                      help="Admit the rows the assay bounded rather than fitted ('<-1' / '>5') as censored items instead of dropping them. "
                           "Dropping them selects the double set on its own outcome: fit failure rises from 0.7%% of shallow doubles to 24%% "
                           "of those below -5 kcal/mol, so the survivors at depth are those that came out better than their additive expectation.")
    data.add_argument('--native_cond_for_wt', '--native_cond_as_single', dest='native_cond_for_wt',
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Keep a mutant-background library's measurements as ordinary 'single' items, which train the WT head: the released "
                           "behaviour, and strictly wrong (they are conditional effects ddG(X | background)). Off, they are labelled "
                           "'native_cond' and the WT head does not train on them.")
    data.add_argument('--native_cond_for_mt', action=argparse.BooleanOptionalAction, default=False,
                      help="Label a mutant-background library's measurements 'native_cond' and route them to the MT/combined loss. This adds NO examples: "
                           "under the default routing (--native_cond_for_wt) they already train as ordinary singles (3 training libraries, 3,782 items, "
                           "1.7%% of the stream). What changes is the MT pass input for them: the library's background sequence (what the WT pass reads) "
                           "instead of background + the scored mutation, so the item is a forward mutation from the background, the MT adapter's mt_fwd "
                           "probe regime. The same swap applies at validation to mutant-background libraries. Independent of --native_cond_for_wt.")
    data.add_argument('--mask_native_cond_struct', action=argparse.BooleanOptionalAction, default=False,
                      help="Mask the VQ-VAE structure token at the background mutation position on native_cond rows (e.g. pos 54 on 2KT8_G54S), "
                           "preventing the frozen structure track from asserting wild-type sidechain packing at the substituted background site.")
    data.add_argument('--external_val_sets', type=str, default='ptmuld,s461,ssym',
                      help="Comma-separated validation loaders scored as OUT-OF-DISTRIBUTION benchmarks: logged per loader (val_<metric>/<name>) "
                           "but held OUT of every aggregate (val_*_avg, the pooled ranking, the epistasis levels), and summarised as "
                           "val_rho_combined_external and val_rmse_combined_external.")
    data.add_argument('--num_workers', type=int, default=4)

    arch = parser.add_argument_group("Architecture")
    arch.add_argument('--adapter_mode', type=str, default='dual', choices=['dual', 'fused'],
                      help="dual: a WT adapter and an MT adapter. fused: ONE adapter and one calibration head read both the wild-type and the "
                           "mutant sequence (its LoRA shape is the MT one: --lora_*_mt).")
    arch.add_argument('--fused_antisymmetric', action=argparse.BooleanOptionalAction, default=False,
                      help="In fused mode, score the MT pass via the negated reverse calibration -f_theta(MT, m -> w), so the combined cycle "
                           "readout 0.5*(WT - rev_MT) cancels per-mutation bias and enforces exact forward/reverse antisymmetry.")
    arch.add_argument('--lora_rank_mt', type=int, default=16)
    arch.add_argument('--lora_alpha_mt', type=int, default=16)
    arch.add_argument('--lora_dropout_mt', type=float, default=0.1)
    arch.add_argument('--target_mode_mt', type=str, default='expanded',
                      choices=['expanded', 'qkv_outproj', 'baseline', 'all', 'ffn'],
                      help="LoRA target modules for MT adapter: expanded (QKV+FFN, default), qkv_outproj (attention only: QKV+out_proj, skipping FFN), baseline (QKV only), all (QKV+out_proj+FFN), or ffn.")
    arch.add_argument('--lora_rank_wt', type=int, default=2)
    arch.add_argument('--lora_alpha_wt', type=int, default=4)
    arch.add_argument('--lora_dropout_wt', type=float, default=0.1)
    arch.add_argument('--target_mode_wt', type=str, default='expanded',
                      choices=['expanded', 'qkv_outproj', 'baseline', 'all', 'ffn'],
                      help="LoRA target modules for WT adapter.")
    arch.add_argument('--tie_mt_scale', action=argparse.BooleanOptionalAction, default=False,
                      help="Tie the MT calibration head's SCALE to the frozen WT head's (the MT head keeps its own bias). The calibrated outputs are then "
                           "the common latent scale; with an independent MT scale the raw-scale rank loss weights the two halves differently from the "
                           "calibrated combined prediction. Default OFF (independent scales, as the released models and every MT run so far); "
                           "pass --tie_mt_scale to share the WT scale. Recorded in hparams.yaml: inference reads it back.")
    arch.add_argument('--link', action=argparse.BooleanOptionalAction, default=True,
                      help="Enable the saturating assay link (default True). Pass --no-link to disable.")
    arch.add_argument('--link_floor', type=float, default=0.0,
                      help="The link's one knee (kcal/mol of absolute dG): 0.0 is the physical floor, the unfolded state. Stamped on every item.")
    arch.add_argument('--link_tau', type=float, default=0.59, help="Width of the link's knee in kcal/mol (RT).")
    for flag, side in (('mask_strategy_wt', 'WT'), ('mask_strategy_mt', 'MT')):
        arch.add_argument(f'--{flag}', type=str, default=None, choices=['none', 'marginal', 'independent'],
                          help=f"Sequence masking of the {side} pass. marginal masks every mutated site at once (the pass knows WHICH sites changed, "
                               "not to what); independent masks one site at a time with the others as they are, the conditional masked marginal "
                               "(on the MT pass, with the partner's mutant residue visible: the readout of ddG(A|B)); unset = unmasked. The same "
                               "strategy on both passes cannot work: marginal on the MT pass hides the partner, so cell_rank is exactly 0.")

    paradigm = parser.add_argument_group("Paradigm")
    paradigm.add_argument('--init_wt_from_checkpoint', type=str, default=None,
                          help="Warm-start the WT model from an existing checkpoint: its WT adapter, its calibration head AND its link are copied in "
                               "before training. Accepts a full Lightning checkpoint or a converged_wt_*.pt file. Nothing else is loaded: the MT path "
                               "is always fresh (use --resume_from_checkpoint for a true resume).")
    paradigm.add_argument('--frozen_wt_mt_only', action=argparse.BooleanOptionalAction, default=False,
                          help="The 'frozen WT, MT-only' paradigm: requires --init_wt_from_checkpoint, freezes the WT adapter (its calibration head "
                               "and link are fixed anyway) and sets --lambda_reg_wt 0 --lambda_rank_wt 0. Only the MT path trains. Validation "
                               "always keeps singles and doubles, so rho_singles_wt stays a guard.")
    paradigm.add_argument('--freeze_mt_adapter', action=argparse.BooleanOptionalAction, default=False,
                          help="Keep the MT adapter frozen for the whole run: a WT-only arm (pair with --val_wt_only, --subsets single and the MT lambdas at 0).")
    paradigm.add_argument('--val_wt_only', action=argparse.BooleanOptionalAction, default=False,
                          help="Validate the WT head only: no MT forward and no MT half of the cycle passes. For a run whose MT adapter is not "
                               "trained, where every MT prediction is a forward through an untrained adapter. The mt and comb columns then report the WT head.")

    obj = parser.add_argument_group("Objective")
    obj.add_argument('--lambda_rank_wt', type=float, default=1.0)
    obj.add_argument('--lambda_reg_wt', type=float, default=1.0)
    obj.add_argument('--lambda_rank_mt', '--lambda_rank_combined', dest='lambda_rank_mt', type=float, default=1.0,
                     help="Weight of the MT path's rank loss (scored through the combined prediction, trains the MT path only).")
    obj.add_argument('--lambda_reg_mt', '--lambda_reg_combined', dest='lambda_reg_mt', type=float, default=1.0,
                     help="Weight of the MT path's regression (scored through the combined prediction, trains the MT path only).")
    obj.add_argument('--rank_list_size', '--subset_size', dest='rank_list_size', type=int, default=16,
                     help="Length of the ListMLE lists: a batch is cut into consecutive lists of this many items.")
    obj.add_argument('--rank_group_by', type=str, default='none', choices=['none', 'partner'],
                     help="How MT ListMLE lists are formed within a micro-batch: 'none' uses random consecutive slices; 'partner' groups doubles "
                          "by shared mutation partner / position pair and adds contrastive epistatic-residual ListMLE lists.")
    obj.add_argument('--huber_delta', type=float, default=0.0,
                     help="0 = squared error (every run so far). >0: Huber loss with this delta (kcal/mol) on the observed-scale regression, "
                          "so a few items that read far from their prediction pull with a bounded gradient.")
    obj.add_argument('--tf_mix', type=float, default=1.0,
                     help="How much of an item's WT half comes from the measured label rather than from the WT head's own prediction: 1.0 = the "
                          "label wherever one is usable, 0.0 = never teacher-force, 0.5 = the arithmetic mean of the two. Applies to singles "
                          "too. An item with no usable label falls back to the WT head for any mix.")

    opt = parser.add_argument_group("Optimisation")
    opt.add_argument('--learning_rate', type=float, default=2e-4)
    opt.add_argument('--weight_decay', type=float, default=0.0)
    opt.add_argument('--batch_size', type=int, default=256)
    opt.add_argument('--micro_batch_size', type=int, default=16, help="Items per forward/backward (the VRAM cap); 64 fits a 32 GB card.")
    opt.add_argument('--num_epochs', type=int, default=20)
    opt.add_argument('--lr_warmup_steps', type=int, default=500)

    sched = parser.add_argument_group("Schedule (each head that is training gets its own plateau)")
    sched.add_argument('--plateau_patience', type=int, default=1,
                       help="Epochs of no improvement in a head's metric (rho_singles_wt for the WT head, --mt_convergence_metric for the MT head) "
                            "before that head's own learning rate is cut by --plateau_factor. 0 disables the decays.")
    sched.add_argument('--plateau_factor', type=float, default=0.3)
    sched.add_argument('--freeze_after_decays', type=int, default=3,
                       help="Freeze a head after this many decays: at factor 0.3 the third leaves it at 2.7 percent of its base rate, which is "
                            "not training. 0 never freezes. Once every head is frozen the run stops itself.")
    sched.add_argument('--mt_convergence_metric', type=str, default='epi_beyond_rho_epi',
                       help="What the MT plateau watches. The default is the correlation of the MT head's epistasis component, ~MT(AB) - WT(AB), "
                            "with the measured one, both beyond the global level.")
    sched.add_argument('--early_stopping_patience', type=int, default=0,
                       help="Stop when --monitor_metric has not improved for this many validations. 0 (the default) never early-stops.")
    sched.add_argument('--monitor_metric', type=str, default='val_epi_beyond_rho_epi',
                       help="Which validation metric picks the checkpoints and drives early stopping; checkpoints are named "
                            "epoch=NN-<metric>=0.xxxx.ckpt.")

    val = parser.add_argument_group("Validation")
    val.add_argument('--val_dump', action=argparse.BooleanOptionalAction, default=True,
                     help="Write per-item validation predictions to val_dump_<tag>.npz each epoch.")
    val.add_argument('--val_cycle_passes', action=argparse.BooleanOptionalAction, default=True,
                     help="Run the two extra validation passes (WT adapter on the mutated sequence, MT adapter on the wild type). Needed for "
                          "the 'ctx' control head of the epistasis metrics; costs one extra forward pass per validation batch.")
    val.add_argument('--skip_val', action=argparse.BooleanOptionalAction, default=False)

    _add_retired(parser)

    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions} - set(RETIRED) - set(_TRANSLATED_VAL) - set(_TRANSLATED_BOOL)
        unknown = sorted(set(cfg) - known)
        if unknown:
            parser.error(f"--config {pre_args.config}: unknown or retired keys {unknown}")
        parser.set_defaults(**cfg)

    args, remaining_argv = parser.parse_known_args(argv)
    if remaining_argv:
        parser.error(f"unrecognized arguments: {' '.join(remaining_argv)}")

    if args.cache_version is None:
        args.cache_version = DEFAULT_CACHE_VERSION

    # ---- retired flags that have a successor
    notes = []
    caps, flags = args.subset_caps, {k: getattr(args, k) for k in ('incl_singles', 'incl_doubles', 'incl_native_cond')}
    del args.subset_caps
    for k in flags:
        delattr(args, k)
    legacy = None
    if caps is not None or any(v is not None for v in flags.values()):
        keep = {'single': True, 'double': True}
        for kv in caps or []:
            if '=' not in kv:
                parser.error(f"--subset_caps: expected key=value, got {kv!r}")
            key, raw = kv.split('=', 1)
            if key not in ('single', 'double', 'cond', 'native_cond'):
                parser.error(f"--subset_caps: unknown subset {key!r}")
            if raw.lower() != 'none' and float(raw) != 0.0:
                parser.error(f"--subset_caps {key}={raw}: fractional caps were removed (a subset is in or out)")
            excluded = raw.lower() != 'none'
            if key == 'cond' and not excluded:
                parser.error("--subset_caps cond=none: the cond subset was removed")
            if key in keep:
                keep[key] = not excluded
        for flag, key in (('incl_singles', 'single'), ('incl_doubles', 'double')):
            if flags[flag] is not None:
                keep[key] = bool(flags[flag])
        legacy = [k for k in SUBSETS if keep[k]]
        notes.append(f"--subset_caps / --incl_* are retired: translated to --subsets {','.join(legacy)} "
                     f"(mutant-background libraries are routed by --native_cond_for_wt / --native_cond_for_mt).")
    if args.subsets is None:
        args.subsets = legacy if legacy is not None else list(SUBSETS)
    else:
        if legacy is not None:
            notes.append("--subsets given together with --subset_caps / --incl_*: --subsets wins.")
        args.subsets = list(args.subsets) if isinstance(args.subsets, (list, tuple)) else [s.strip() for s in str(args.subsets).split(',') if s.strip()]
    bad = [s for s in args.subsets if s not in SUBSETS]
    if bad or not args.subsets:
        parser.error(f"--subsets must be a comma-separated subset of {list(SUBSETS)}, got {args.subsets}")
    args.subsets = [s for s in SUBSETS if s in args.subsets]

    if args.assay_floor is not None:
        args.link_floor = float(args.assay_floor)
        notes.append(f"--assay_floor is retired: translated to --link_floor {args.link_floor}.")
    del args.assay_floor
    if args.comet_api_key:
        os.environ['COMET_API_KEY'] = args.comet_api_key
        notes.append("--comet_api_key is retired: the key was moved into the environment and is not recorded.")
    del args.comet_api_key
    args.requested_split_file = args.split_file          # checked against the cache manifest in setup_dataloaders, then dropped
    del args.split_file
    if args.censor_threshold_double is not None and args.censor_threshold is not None and not _same(args.censor_threshold_double, args.censor_threshold):
        parser.error(f"--censor_threshold_double {args.censor_threshold_double} differs from --censor_threshold {args.censor_threshold}: "
                     f"one threshold serves singles and doubles now")
    del args.censor_threshold_double
    if args.reg_loss is not None and args.reg_loss != 'mse':
        if args.reg_loss != 'huber':
            parser.error(f"--reg_loss {args.reg_loss}: only mse and huber ever existed")
        if args.huber_delta <= 0:
            args.huber_delta = 1.0
        notes.append(f"--reg_loss huber is retired: translated to --huber_delta {args.huber_delta}.")
    del args.reg_loss
    for head in ('wt', 'mt'):
        for new in ('plateau_patience', 'plateau_factor', 'freeze_after_decays'):
            old = f'{head}_{new}'
            v = getattr(args, old)
            if v is not None and not _same(v, getattr(args, new)):
                parser.error(f"--{old} {v} differs from --{new} {getattr(args, new)}: both heads share one {new} now")
            delattr(args, old)
    if args.early_stopping_metric is not None and args.early_stopping_metric != args.monitor_metric:
        parser.error(f"--early_stopping_metric {args.early_stopping_metric} differs from --monitor_metric {args.monitor_metric}: one metric now")
    del args.early_stopping_metric
    explicit_freeze_wt = args.freeze_wt_adapter
    del args.freeze_wt_adapter

    ignored = _check_retired(args, parser)
    for n in notes:
        logging.warning(n)
    if ignored:
        logging.info(f"retired flags accepted and ignored: {', '.join('--' + n for n in ignored)}")

    # ---- validation
    if not 0.0 <= args.tf_mix <= 1.0:
        parser.error(f"--tf_mix must be in [0, 1] (0 = never teacher-force, 1 = always use the measured additive label where one is usable); got {args.tf_mix}")
    if args.huber_delta < 0:
        parser.error("--huber_delta must be >= 0")
    if args.rank_list_size < 1:
        parser.error("--rank_list_size must be >= 1")
    if args.link_tau <= 0:
        parser.error("--link_tau must be positive")
    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
    if args.init_wt_from_checkpoint is not None and not os.path.isfile(args.init_wt_from_checkpoint):
        parser.error(f"--init_wt_from_checkpoint: no such file: {args.init_wt_from_checkpoint}")

    # The frozen-WT / MT-only paradigm is a named combination of primitive settings. Deriving it here (rather than branching in the
    # model) means hparams record the EFFECTIVE settings the run used.
    args.freeze_wt_adapter = False
    if args.frozen_wt_mt_only:
        if args.init_wt_from_checkpoint is None:
            parser.error("--frozen_wt_mt_only needs --init_wt_from_checkpoint: the paradigm is a PRE-TRAINED frozen WT head, not a randomly initialised one")
        if 'double' not in args.subsets:
            parser.error("--frozen_wt_mt_only trains on multi-mutants, but --subsets leaves doubles out: nothing would be left to train on")
        args.freeze_wt_adapter = True
        # a frozen WT head trains nothing: its losses would be computed every step and never backpropagated.
        if args.lambda_reg_wt > 0 or args.lambda_rank_wt > 0:
            logging.info("--frozen_wt_mt_only: setting --lambda_reg_wt 0 --lambda_rank_wt 0 (the WT path is frozen)")
            args.lambda_reg_wt = 0.0
            args.lambda_rank_wt = 0.0
    elif explicit_freeze_wt:
        parser.error("--freeze_wt_adapter without --frozen_wt_mt_only: freezing the WT head alone is no longer a separate mode")
    elif args.init_wt_from_checkpoint is not None:
        logging.warning(f"--init_wt_from_checkpoint given without --frozen_wt_mt_only: {args.init_wt_from_checkpoint} is a warm start, "
                        f"and the WT head will keep training from it")

    for k, v in DERIVED.items():
        setattr(args, k, v)
    return args
