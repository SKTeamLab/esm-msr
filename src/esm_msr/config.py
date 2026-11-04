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
import re

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
    'weight_decay': _r('val', 0.0, 'no weight decay is ever used (AdamW weight_decay is 0)'),
    # the LoRA spec is fixed except incl_sequence_head_{wt,mt}, which can be ablated
    **{f'{k}_{h}': _r('bool', v, 'a constant now') for h in ('wt', 'mt')
       for k, v in (('use_dora', False), ('unfreeze_layernorms', False), ('incl_structure_encoder', False))},
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


# val_[unmasked_]epi_<metric>_epi was the MT head at twice the reported scale and is no longer logged (training.epi_log_names). The rank levels
# that ignore scale are bit-identical to their _comb twin, and the MT head's own cell rank is val_diag_epi_cell_rank_mt_legacy, so a flag that
# still names the old metric is rewritten rather than left to point at a metric that never appears.
_EPI_SAME_AS_COMB = ('beyond_rho', 'beyond_lib', 'beyond_rho_pairavg', 'pair_rho', 'line_rho')
_RETIRED_EPI = re.compile(r'^(val_)?((?:unmasked_)?)epi_(.+)_epi$')


def current_metric_name(name, monitor):
    """The logged equivalent of a retired ``_epi``-head metric name; a name that is not retired comes back unchanged.
    ``monitor`` (a checkpoint / early-stopping monitor, which reads the logger) cannot use a name that is only kept internally, so for it an
    ``_epi`` metric with no exact twin is an error (returns None)."""
    m = _RETIRED_EPI.match(name)
    if not m:
        return name
    val, unmasked, metric = m.groups()
    val = val or ''
    if metric in _EPI_SAME_AS_COMB:
        return f'{val}{unmasked}epi_{metric}_comb'
    if metric == 'cell_rank' and monitor:
        return f'val_{unmasked}diag_epi_cell_rank_mt_legacy'
    return None if monitor else name


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
    data.add_argument('--drop_last', action=argparse.BooleanOptionalAction, default=True,
                      help="Drop per-library tail batches smaller than --batch_size (legacy default True). Pass --no-drop_last to retain "
                           "tail batches of at least --min_batch_size items (~9.2%% of training variants).")
    data.add_argument('--min_batch_size', type=int, default=32,
                      help="Minimum items in a retained per-library tail batch when --no-drop_last is set.")

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
    arch.add_argument('--incl_sequence_head_wt', action=argparse.BooleanOptionalAction, default=True,
                      help="Include ESM3's sequence LM output head in the WT LoRA targets (default True). Pass --no-incl_sequence_head_wt to freeze.")
    arch.add_argument('--incl_sequence_head_mt', action=argparse.BooleanOptionalAction, default=True,
                      help="Include ESM3's sequence LM output head in the MT LoRA targets (default True). Pass --no-incl_sequence_head_mt to freeze.")
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
    arch.add_argument('--link_kind', type=str, default='softplus', choices=['softplus', 'leaky'],
                      help="softplus: h = f + tau*softplus((z-f)/tau), one number f is both the knee and the asymptote (the released link). leaky: the knee "
                           "(--link_floor, --link_tau) and the asymptote (--link_clamp) are different numbers, with slope --link_alpha below the knee: "
                           "every numeric reading is inside the range and the inverse is defined for all of them (esm_msr.link.LeakyFloorLink).")
    arch.add_argument('--link_alpha', type=float, default=0.10, help="--link_kind leaky: the slope below the knee (0 < alpha < 1).")
    arch.add_argument('--link_clamp', type=float, default=-1.0, help="--link_kind leaky: the asymptote, the lowest dG the link can report (the assay's -1).")
    for flag, side in (('mask_strategy_wt', 'WT'), ('mask_strategy_mt', 'MT')):
        arch.add_argument(f'--{flag}', type=str, default=None, choices=['none', 'marginal', 'independent'],
                          help=f"Sequence masking of the {side} pass. marginal masks every mutated site at once (the pass knows WHICH sites changed, "
                               "not to what); independent masks one site at a time with the others as they are, the conditional masked marginal "
                               "(on the MT pass, with the partner's mutant residue visible: the readout of ddG(A|B)); unset = unmasked. The same "
                               "strategy on both passes cannot work: marginal on the MT pass hides the partner, so cell_rank is exactly 0. "
                               "(--block_training reads sites by --block_mask_wt/_mt instead.)")

    paradigm = parser.add_argument_group("Paradigm")
    paradigm.add_argument('--mt_direct_pred', action=argparse.BooleanOptionalAction, default=False,
                          help="Direct prediction via MT head: zero-shot WT adapter is frozen, only MT adapter trains. "
                               "Skips teacher forcing and blends, evaluating MT directly on singles and doubles.")
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
                     help="Length of the ListMLE lists: a batch is cut into consecutive lists of this many items. --block_training cuts no lists of this size (a list is "
                          "--block_list_rows positions); there it is only the reference length of --block_rank_norm ref, i.e. it scales the rank term like --lambda_rank_*.")
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
    obj.add_argument('--rank_calibrated_stopgrad', action=argparse.BooleanOptionalAction, default=False,
                     help="Score ListMLE on stop-gradient calibrated predictions (s.detach()*raw + b.detach()*n_mut) / s_wt.detach(), aligning "
                          "untied MT/WT scales and per-mutation biases across mixed single/double lists without temperature collapse on s. Off: ListMLE scores the raw "
                          "LLR. Either way the rank loss sends no gradient to the calibration heads. --block_training honours it too (a conditional "
                          "leg's list is always calibrated).")
    obj.add_argument('--reg_nll', type=str, default='none', choices=['none', 'tobit'],
                     help="Regression likelihood. none: MSE/Huber on the observed scale through the link, one-sided at --censor_threshold (the default). "
                          "tobit: the noisy-floor Tobit mixture on the LATENT absolute dG (losses.tobit_regression_loss; its noise constants are losses.TOBIT_*); the link is then only the "
                          "forward map used to report the observed scale, and --censor_threshold only ties the floor items' ranks (the regression ignores it). "
                          "'<-1' / '>5' range tags are scored with the survival term.")
    obj.add_argument('--tf_cut', type=_threshold, default=None,
                     help="Teacher forcing uses a measured single (or a double's two singles) only when its absolute dG is ABOVE this. Unset keeps the old "
                          "coupling to --censor_threshold (or the link floor): that coupling made a censor threshold at or below the floor feed "
                          "near-floor readings to the inverse link, which explodes there. Set it (0.5) to choose the teacher-forcing cut on its own.")
    obj.add_argument('--tf_label', type=str, default='inverse', choices=['inverse', 'clamped', 'posterior'],
                     help="The teacher-forcing label of one measured single. inverse: the inverse link above --tf_cut, and below it the WT head's read of that single alone, so its partner keeps its "
                          "measurement (the default, in both training paths). --block_training only: posterior, E[z | reading] under the Tobit likelihood with the WT head's detached read as the prior "
                          "(losses.tobit_posterior_mean), at every reading, with no cut; clamped, the inverse link with the reading raised to 0.25 (training.TF_CLAMP_Y0).")
    obj.add_argument('--tf_no_label', type=str, default='wt', choices=['wt', 'mt'],
                     help="--block_training with --tf_label inverse: what a double missing a measurement is regressed through. wt: 0.5 (WT + MT), the WT head reading the missing single (the default). "
                          "mt: the MT head alone, which is on half the epistasis scale of a labelled double.")
    obj.add_argument('--mt_single_loss_weight', type=float, default=1.0,
                     help="Relative loss weight on single-mutant items vs multi-mutants (1.0) in the MT regression and ranking losses "
                          "(e.g. 0.1 downweights singles 10x while preserving domain-context regularisation).")
    obj.add_argument('--lambda_cond_mt', type=float, default=0.0,
                     help="Weight on zero-extra-pass two-conditional masked supervision (A|B and B|A from mt_out['unsummed_llr'] when "
                          "--mask_strategy_mt independent and partner dG > --cond_partner_min_dg). With --block_training the MT head's row and column "
                          "passes are the conditional legs: each is scored as the partner single's latent plus the calibrated leg against the "
                          "double's label, with ListMLE lists of the same legs (rank weight lambda_cond_mt * lambda_rank_mt / lambda_reg_mt).")
    obj.add_argument('--lambda_cycle', type=float, default=0.0,
                     help="--block_training: weight on the thermodynamic-cycle loss of every pair block, label-free and over all 19 x 19 doubles of the "
                          "position pair, observed or not: the mean squared closing error (path 1 - path 2)^2 with path 1 = WT(x) + MT(Y|X) and path 2 = "
                          "WT(y) + MT(X|Y), on the stop-gradient calibrated scale. It pulls one path onto the other, not their mean, so it constrains the "
                          "unlabelled doubles without moving the reported prediction directly. val_cycle_gap is the same quantity, validated.")
    obj.add_argument('--cycle_detach_wt', action=argparse.BooleanOptionalAction, default=True,
                     help="--lambda_cycle: hold the WT head's single readouts constant in the cycle loss, so only the MT conditionals move toward them "
                          "(default); off lets the cycle loss train the WT head too.")
    obj.add_argument('--cond_partner_min_dg', type=float, default=1.0,
                     help="Minimum absolute partner stability dG_wt + ddG_partner (kcal/mol) required to supervise a conditional leg.")
    obj.add_argument('--latent_censor_hinge', action=argparse.BooleanOptionalAction, default=False,
                     help="Evaluate censored items' one-sided hinge directly on latent stability (dG_wt + pred_cal vs unclamped dG_bound) "
                          "instead of clamping bounds upward to +0.409 via reachable_bounds, while keeping SaturatingLink on uncensored items.")
    obj.add_argument('--wt_supervise_doubles', action=argparse.BooleanOptionalAction, default=False,
                     help="Supervise the WT head on all true single and double ddG labels (passing doubles through the observed-scale link "
                          "and censoring bounds identically to singles). Designed for --mask_strategy_wt marginal, where simultaneous pair "
                          "masking gives the WT head awareness of both mutated sites (pair and line effects).")
    obj.add_argument('--tf_detach_wt', action=argparse.BooleanOptionalAction, default=True,
                     help="The WT half that teacher-forces the MT head carries no gradient to the WT head (the default, and what every run so far did): the "
                          "WT head learns from its own losses, the MT head from the combined one. --no-tf_detach_wt lets the combined loss train the WT "
                          "head on every item as well, which keeps its graph alive until the MT pass has run (one backward through both, so memory "
                          "for two graphs) and is refused with --joint_doubles alternate. --joint_doubles is the narrower switch: only the doubles.")
    obj.add_argument('--joint_doubles', type=str, default='off', choices=['off', 'both', 'alternate'],
                     help="Train the two adapters TOGETHER on the real doubles. off (every run so far): a double's WT half is the measured "
                          "additive label (--tf_mix) or the WT head's detached prediction, so only MT is trained by it and the WT head sees "
                          "singles alone. both: the WT half of a double carries the WT head's LIVE prediction, 0.5*(WT + MT) is regressed on the "
                          "measured double and the gradient reaches both adapters (both graphs are alive at once, so lower --micro_batch_size). "
                          "alternate: the same loss, but each double keeps only ONE adapter's graph - the other half is read under no_grad - "
                          "and which adapter alternates from double to double and micro-batch to micro-batch. The owner's loss is doubled so "
                          "the expected update is 'both''s; peak memory is one graph. --tf_mix applies to the doubles as well: the WT half is "
                          "tf_mix * (measured label) + (1 - tf_mix) * (live WT prediction), so the WT head gets the (1 - tf_mix) share of the "
                          "gradient where a label is usable (and all of it where none is). At the default --tf_mix 1.0 a labelled double "
                          "trains the MT head alone, as under off: pair this flag with --tf_mix < 1. Singles are untouched. The rank term "
                          "stays an MT-side term with the WT half detached.")

    opt = parser.add_argument_group("Optimisation")
    opt.add_argument('--learning_rate', type=float, default=2e-4)
    opt.add_argument('--grad_clip_norm', type=float, default=0.0,
                     help="Global L2 gradient norm clip across all trainable parameters before optim.step() (0 = disabled).")
    opt.add_argument('--grad_clip_per_head', action=argparse.BooleanOptionalAction, default=False,
                     help="Apply --grad_clip_norm to the WT parameters and the MT parameters separately (adapter and calibration head each; anything else is "
                          "its own group), so the MT head's doubles do not shrink the WT head's step.")
    opt.add_argument('--calib_lr_mult', type=float, default=20.0,
                     help="Learning-rate multiplier for calibration heads relative to --learning_rate (default 20.0; 5.0 recommended with cosine).")
    opt.add_argument('--batch_size', type=int, default=256, help="Items per optimizer step of the per-item path (--block_training: a step is one block and ignores this; validation uses 64).")
    opt.add_argument('--micro_batch_size', type=int, default=16, help="Items per forward/backward (the VRAM cap); 64 fits a 32 GB card. Per-item path only: --block_training ignores it.")
    opt.add_argument('--num_epochs', type=int, default=20)
    opt.add_argument('--lr_warmup_steps', type=int, default=500)

    sched = parser.add_argument_group("Schedule (each head that is training gets its own plateau)")
    sched.add_argument('--lr_schedule', type=str, default='plateau', choices=['plateau', 'cosine'],
                       help="Learning-rate schedule: 'plateau' (per-head ReduceLROnPlateau, legacy default) or 'cosine' (linear warmup + "
                            "cosine decay applied to both LoRA and calibration parameter groups).")
    sched.add_argument('--cosine_min_lr_ratio', type=float, default=0.05,
                       help="Minimum LR as a fraction of peak LR at the end of cosine decay (default 0.05).")
    sched.add_argument('--plateau_patience', type=int, default=1,
                       help="Epochs of no improvement in a head's metric (rho_singles_wt for the WT head, --mt_convergence_metric for the MT head) "
                            "before that head's own learning rate is cut by --plateau_factor. 0 disables the decays.")
    sched.add_argument('--plateau_factor', type=float, default=0.3)
    sched.add_argument('--freeze_after_decays', type=int, default=3,
                       help="Freeze a head after this many decays: at factor 0.3 the third leaves it at 2.7 percent of its base rate, which is "
                            "not training. 0 never freezes. Once every head is frozen the run stops itself.")
    sched.add_argument('--mt_convergence_metric', type=str, default='epi_beyond_rho_comb',
                       help="What the MT plateau watches. The default is the correlation of the epistasis component of the reported prediction, 0.5 (~MT(AB) - WT(AB)), "
                            "with the measured one, both beyond the global level.")
    sched.add_argument('--early_stopping_patience', type=int, default=0,
                       help="Stop when --monitor_metric has not improved for this many validations. 0 (the default) never early-stops.")
    sched.add_argument('--monitor_metric', type=str, default='val_epi_beyond_rho_comb',
                       help="Which validation metric picks the checkpoints and drives early stopping; checkpoints are named "
                            "epoch=NN-<metric>=0.xxxx.ckpt.")

    val = parser.add_argument_group("Validation")
    val.add_argument('--val_dump', action=argparse.BooleanOptionalAction, default=True,
                     help="Write per-item validation predictions to val_dump_<tag>.npz each epoch.")
    val.add_argument('--val_cycle_passes', action=argparse.BooleanOptionalAction, default=True,
                     help="Run the two extra validation passes (WT adapter on the mutated sequence, MT adapter on the wild type). Needed for "
                          "the 'ctx' control head of the epistasis metrics; costs one extra forward pass per validation batch.")
    train_g = parser.add_argument_group("Throughput")
    train_g.add_argument('--block_training', action=argparse.BooleanOptionalAction, default=False,
                         help="Train by BLOCKS (esm_msr.blocks): a step is B positions of one library read as a [B, 20] tensor from B masked-site "
                              "passes (every substitution at once, the wild-type residue reading 0), regressed and ranked as ONE list of B x 20 "
                              "cells; or one position pair, whose 19 x 19 doubles come from 2 wild-type and 38 mutant passes, ranked as lists of "
                              "B rows or columns. An epoch is every library's singles blocks, then every pair block. Training masks are "
                              "structural (the site read is masked; a double's mutant pass shows the partner), so --mask_strategy_* and "
                              "--batch_size / --micro_batch_size do not apply to training (--rank_list_size is only the reference "
                              "length of --block_rank_norm ref); validation is "
                              "unchanged. --joint_doubles other than off makes a pair's WT half live. native_cond items are left out.")
    train_g.add_argument('--block_identity', action=argparse.BooleanOptionalAction, default=True,
                         help="--block_training: the wild-type residue is a cell of every ListMLE list, with label 0 and prediction 0 (no mutation, so no "
                              "bias): a singles block holds B x 20 cells with the wild type in its own column, a pair block is the 20 x 20 matrix whose "
                              "identity row and column are the singles (40 mutant passes instead of 38), and a conditional leg's list holds the "
                              "partner single. The identity cells carry no regression. (The rank scale is --rank_calibrated_stopgrad's, not this flag's.)")
    train_g.add_argument('--block_row_chunk', type=int, default=0,
                         help="--block_training: run a block's passes in checkpointed chunks of this many rows (0 = all at once). Only a chunk's "
                              "activations are alive at a time and the gradient is exact (the dropout draw is restored on recompute). Not needed for "
                              "visible-site-free runs: the 40-pass pair block peaks at 16.3 GB un-checkpointed (longest library), a recompute costs "
                              "~1.7x on the pair blocks. Needed (an even number, e.g. 20) when --block_mask_mt makes sites visible.")
    train_g.add_argument('--block_positions', type=int, default=8,
                         help="B of --block_training: positions per singles block, and rows (or columns) per rank list of a pair block.")
    train_g.add_argument('--block_list_rows', type=int, default=0,
                         help="--block_training: positions (rows) per ListMLE list. 0 (default) keeps the lists as they were: a singles block is ONE list "
                              "of B x 20 cells and a pair block's lists hold --block_positions rows or columns. R >= 1 makes lists of R rows in both "
                              "(R = 1: each position, or each row or column of a pair block, ranked on its own, 20 cells with the wild type). "
                              "ListMLE SUMS over the list, so the rank term grows with the list: see --block_rank_norm.")
    train_g.add_argument('--block_rank_norm', choices=['sum', 'ref'], default='sum',
                         help="--block_training: how a list's ListMLE loss is scaled. 'sum' (default) is the loss as is, so a list of B x 20 cells weighs "
                              "B x 20 / 16 times the 16-member lists of the per-item path against the regression, which is a mean. 'ref' rescales every "
                              "list to --rank_list_size members' worth (loss x rank_list_size / its valid members), so the rank:regression balance "
                              "of --lambda_rank_* / --lambda_reg_* does not depend on B or --block_list_rows.")
    train_g.add_argument('--grad_accum_blocks', type=int, default=1,
                         help="--block_training: gradient accumulation, in BLOCKS. K >= 2 adds up the gradients of K consecutive blocks (they stay in .grad, "
                              "which costs the size of the trainable parameters, ~0.1 GB, and no activation memory) and takes one optimizer step after the K-th, "
                              "so a run at small B takes K x fewer, K x smoother steps. The summed gradient is divided by K before the clip; --lr_warmup_steps "
                              "and the cosine length count OPTIMIZER steps (set warmup to a quarter for the same warmup in blocks); logged train/* losses are "
                              "the mean over the K blocks; blocks left over at the end of an epoch carry into the next, and a resume starts a fresh window.")
    train_g.add_argument('--block_domain_steps', action=argparse.BooleanOptionalAction, default=False,
                         help="--block_training: one optimizer step is one library: all of its positions as one singles block (--block_positions no longer "
                              "sizes it) plus all of its pair blocks, their gradients summed before the clip. An epoch is one step per library. A pair's "
                              "gradient is weighted --block_pair_weight / (the library's pairs), so the ratio of doubles to singles survives the clip, and a "
                              "library counts once however many pairs it has. The largest training domain peaks at 26.9 GB: smaller cards need --block_row_chunk.")
    train_g.add_argument('--block_pair_weight', type=float, default=1.0,
                         help="--block_domain_steps: weight of a library's doubles (mean over its pairs) against its singles. Reaches the MT head only. 0 runs no "
                              "pair blocks, which leaves the fully masked MT head without any partner to read: a control, not a recipe.")
    train_g.add_argument('--block_mask_wt', type=float, default=1.0,
                         help="--block_training: probability that the WT head reads a position with its site MASKED (1, the default, always; 0 = the "
                              "plain wild-type sequence every time, the unmasked WT readout; in between, drawn per position of a singles block and per "
                              "read of a pair block). An unmasked WT read is the cheap case: every row is the same sequence, so a block is one forward.")
    train_g.add_argument('--block_mask_mt', type=float, default=1.0,
                         help="--block_training: probability that the MT head reads a position with its site MASKED (1, the default, always). An "
                              "unmasked read sees the mutation: each of a position's 19 substitutions is its own sequence (19 passes instead of 1), "
                              "drawn per position of a singles block; a pair block is drawn once per block and, unmasked, reads all 20 x 20 sequences "
                              "at both sites (no row / column passes, so no --lambda_cond_mt legs for that block). Expensive: set --block_row_chunk "
                              "(an even number, so a sequence's two reads share a chunk).")
    val.add_argument('--val_masking', choices=['independent', 'train'], default='independent',
                     help="How validation reads the passes. 'independent' (default): BOTH adapters read every mutated site masked on its own, with the "
                          "partner's mutant residue visible (the singles' masked sequences, so it comes free): WT(a, b masked) + WT(b, a masked) and "
                          "MT(B | A) + MT(A | B), the thermodynamic cycle, which is the regime block training and the random-mask paradigm train "
                          "on. Every val_* metric, the plateau schedule, early stopping and the checkpoint monitor read this regime, and "
                          "val_cycle_gap is the mean |path 1 - path 2| of the double's two paths around the cycle. 'train': the run's own "
                          "--mask_strategy_wt/_mt (unmasked when unset, and an eval pass of 'random' is unmasked), which is what validation was before "
                          "and what an unmasked-trained run is matched to; no cycle gap.")
    val.add_argument('--val_unmasked', action=argparse.BooleanOptionalAction, default=False,
                     help="Also score validation with NEITHER adapter masked (the plain wild-type sequence for WT, the full mutant sequence for MT: the "
                          "per-item readout), logged as val_unmasked_* (per-library metrics and their _avg, and the epistasis hierarchy as "
                          "val_unmasked_epi_*, with val_unmasked_epi_composite). Not watched unless pointed at: the plateau, early stopping and the checkpoint monitor read the default regime, "
                          "and a run trained on visible sites sets --monitor_metric val_unmasked_epi_composite and --mt_convergence_metric unmasked_epi_composite. "
                          "The matched regime of a run trained on visible sites (--block_mask_wt/_mt < 1, --mask_strategy_* none or random). The MT "
                          "side is one forward per variant, the cost the validation had before it was consolidated.")
    val.add_argument('--val_every_epochs', type=int, default=1,
                     help="Validate every N epochs. --early_stopping_patience and the plateaus count validations, not epochs.")
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
    if args.link_kind == 'leaky':
        if not 0.0 < args.link_alpha < 1.0:
            parser.error("--link_alpha must be in (0, 1)")
        if args.link_clamp >= args.link_floor:
            parser.error(f"--link_clamp {args.link_clamp} must be below the knee --link_floor {args.link_floor}")
    if args.reg_nll == 'tobit' and not args.link:
        parser.error("--reg_nll tobit with --no-link: the observed-scale report would be the latent itself; use the link (it only reports, it is not trained through)")
    if args.mt_single_loss_weight < 0:
        parser.error("--mt_single_loss_weight must be >= 0")
    if args.lambda_cond_mt < 0:
        parser.error("--lambda_cond_mt must be >= 0")
    if args.grad_clip_norm < 0:
        parser.error("--grad_clip_norm must be >= 0")
    if args.calib_lr_mult <= 0:
        parser.error("--calib_lr_mult must be > 0")
    if args.min_batch_size < 1:
        parser.error("--min_batch_size must be >= 1")
    if args.num_workers == -1:
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs available: {num_cpus}")
        args.num_workers = max(1, num_cpus - 1)
    if args.init_wt_from_checkpoint is not None and not os.path.isfile(args.init_wt_from_checkpoint):
        parser.error(f"--init_wt_from_checkpoint: no such file: {args.init_wt_from_checkpoint}")

    for flag, monitor in (('mt_convergence_metric', False), ('monitor_metric', True)):
        old_name = getattr(args, flag)
        new_name = current_metric_name(old_name, monitor)
        if new_name is None:
            parser.error(f"--{flag} {old_name}: the _epi head is no longer logged; use its _comb twin (the reported prediction), "
                         f"or val_diag_epi_cell_rank_mt_legacy for the MT head's cell rank")
        if new_name != old_name:
            logging.info(f"--{flag} {old_name} is retired; using {new_name}")
            setattr(args, flag, new_name)

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
    elif args.mt_direct_pred:
        if args.adapter_mode == 'fused':
            parser.error("--mt_direct_pred requires --adapter_mode dual: in dual mode WT is frozen at zero-shot and MT trains directly. In fused mode the single adapter is shared.")
        args.freeze_wt_adapter = True
        if args.lambda_reg_wt > 0 or args.lambda_rank_wt > 0:
            logging.info("--mt_direct_pred: setting --lambda_reg_wt 0 --lambda_rank_wt 0 (direct MT prediction; WT is frozen)")
            args.lambda_reg_wt = 0.0
            args.lambda_rank_wt = 0.0
        if args.mt_convergence_metric == 'epi_beyond_rho_comb':
            args.mt_convergence_metric = 'rho_combined'
        if args.monitor_metric == 'val_epi_beyond_rho_comb':
            args.monitor_metric = 'val_rho_combined_avg'
    elif explicit_freeze_wt:
        parser.error("--freeze_wt_adapter without --frozen_wt_mt_only: freezing the WT head alone is no longer a separate mode")
    elif args.init_wt_from_checkpoint is not None:
        logging.warning(f"--init_wt_from_checkpoint given without --frozen_wt_mt_only: {args.init_wt_from_checkpoint} is a warm start, "
                        f"and the WT head will keep training from it")

    if not args.tf_detach_wt and args.joint_doubles == 'alternate':
        parser.error("--no-tf_detach_wt keeps the WT graph alive for every item, but --joint_doubles alternate frees it before the MT pass: use --joint_doubles both (or off)")
    if args.joint_doubles != 'off' and not args.block_training:
        clash = [f for f, on in (('--reg_nll tobit', args.reg_nll == 'tobit'), ('--latent_censor_hinge', args.latent_censor_hinge),
                                 ('--wt_supervise_doubles', args.wt_supervise_doubles)) if on]
        if clash:
            parser.error(f"--joint_doubles regresses the live double through the standard observed-scale hinge; it is not defined with {', '.join(clash)}")
    if (args.tf_label != 'inverse' or args.tf_no_label != 'wt') and not args.block_training:
        parser.error("--tf_label clamped / posterior and --tf_no_label mt are defined for --block_training only")
    if args.tf_label != 'inverse' and args.tf_cut is not None:
        parser.error(f"--tf_label {args.tf_label} has no cut: every single keeps a label, so --tf_cut means nothing")
    if args.tf_no_label != 'wt' and args.tf_label != 'inverse':
        parser.error("--tf_no_label mt needs --tf_label inverse: the other labels leave no double without one")
    if args.block_training:
        # Block training regresses on tensors: the observed-scale hinge or, with --reg_nll tobit, the Tobit likelihood on the latent dG. The per-item
        # extras below have no block form.
        clash = [f for f, on in (('--latent_censor_hinge', args.latent_censor_hinge), ('--wt_supervise_doubles', args.wt_supervise_doubles),
                                 ('--mt_single_loss_weight', args.mt_single_loss_weight != 1.0)) if on]
        if clash:
            parser.error(f"--block_training has its own tensor losses; it is not defined with {', '.join(clash)}")
    if args.block_training:
        bad = ('--mt_direct_pred' if args.mt_direct_pred else '--adapter_mode fused' if args.adapter_mode == 'fused'
               else '--rank_group_by partner' if args.rank_group_by == 'partner' else None)
        if bad:
            parser.error(f"--block_training does not support {bad}")
        if args.block_positions < 1:
            parser.error("--block_positions must be >= 1")
        if args.block_list_rows < 0:
            parser.error("--block_list_rows must be >= 0")
        for f in ('block_mask_wt', 'block_mask_mt'):
            if not 0.0 <= getattr(args, f) <= 1.0:
                parser.error(f"--{f} is a probability in [0, 1]")
    if args.block_domain_steps and not args.block_training:
        parser.error("--block_domain_steps needs --block_training")
    if args.block_domain_steps and args.grad_accum_blocks > 1:
        parser.error("--block_domain_steps sets the step itself: leave --grad_accum_blocks at 1")
    if args.block_pair_weight < 0:
        parser.error("--block_pair_weight must be >= 0")
    if args.block_pair_weight != 1.0 and not args.block_domain_steps:
        parser.error("--block_pair_weight needs --block_domain_steps (the clip cancels a weight on a pair block's own step)")
    if args.grad_clip_per_head and args.grad_clip_norm <= 0:
        parser.error("--grad_clip_per_head needs --grad_clip_norm > 0")
    if args.val_every_epochs < 1:
        parser.error("--val_every_epochs must be >= 1")
    if args.grad_accum_blocks < 1:
        parser.error("--grad_accum_blocks must be >= 1")
    if args.grad_accum_blocks > 1 and not args.block_training:
        parser.error("--grad_accum_blocks needs --block_training (the per-item path already accumulates micro-batches inside one batch)")
    if args.lambda_cycle < 0:
        parser.error("--lambda_cycle must be >= 0")
    if args.lambda_cycle > 0 and not args.block_training:
        parser.error("--lambda_cycle is a --block_training loss (it needs a pair block's row and column passes)")
    if args.joint_doubles != 'off':
        why = ('--frozen_wt_mt_only / --mt_direct_pred freeze the WT head, which is half of the joint prediction' if args.freeze_wt_adapter
               else '--freeze_mt_adapter freezes the MT head, which is the other half' if args.freeze_mt_adapter
               else '--adapter_mode fused shares one adapter between the passes, so there are no two graphs to joint or alternate' if args.adapter_mode == 'fused'
               else '--lambda_reg_mt 0 leaves nothing to regress the joint prediction with' if args.lambda_reg_mt <= 0
               else '--subsets leaves doubles out' if 'double' not in args.subsets else None)
        if why:
            parser.error(f"--joint_doubles {args.joint_doubles}: {why}")
        if args.tf_mix >= 1.0:
            logging.warning(f"--joint_doubles {args.joint_doubles} with --tf_mix {args.tf_mix}: a double with a usable additive label is "
                            f"teacher-forced completely, so the WT head is trained only by the doubles that have none. Use --tf_mix < 1.")

    for k, v in DERIVED.items():
        setattr(args, k, v)
    return args
