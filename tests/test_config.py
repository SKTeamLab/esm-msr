"""The command line of the stripped branch: block training only, no compatibility with the old flags, YAML defaults, and a small bound on the number of flags."""
import glob
import os

import pytest
import yaml

from esm_msr import config

BASE = ['--experiment_name', 'x']


def parse(*extra, base=BASE):
    return config.parse_arguments(list(base) + list(extra))


# ---------------------------------------------------------------------------------------------------------------- what is gone

@pytest.mark.parametrize('flag', [
    '--block_training', '--block_identity', '--block_rank_norm', '--censor_threshold', '--cycle_detach_wt', '--drop_last', '--freeze_after_decays', '--fused_antisymmetric',
    '--grad_clip_per_head', '--huber_delta', '--joint_doubles', '--lambda_cycle', '--link_kind', '--link_tau', '--lora_mode', '--monitor_metric', '--mt_direct_pred',
    '--per_mutation_bias', '--plateau_factor', '--lr_schedule', '--rank_calibrated_stopgrad', '--rank_group_by', '--tf_detach_wt',
    '--tf_no_label', '--y_floor', '--adapter_mode', '--reg_nll', '--mt_single_loss_weight', '--mask_strategy_wt', '--mask_strategy_mt', '--batch_size', '--micro_batch_size',
    '--min_batch_size', '--skip_val', '--val_every_epochs', '--val_dump', '--val_unmasked', '--val_masked', '--val_masking_strategy', '--val_cycle_passes', '--val_tobit_link',
    '--external_val_sets', '--subsets', '--censor_range', '--calib_lr_mult', '--grad_clip_norm', '--cosine_min_lr_ratio', '--tie_mt_scale',
    '--freeze_wt_adapter', '--freeze_mt_adapter', '--frozen_wt_mt_only', '--init_wt_from_checkpoint', '--val_wt_only', '--wt_supervise_doubles', '--weight_decay'])
def test_a_removed_flag_is_an_error(flag):
    with pytest.raises(SystemExit):
        parse(flag, '1')
    with pytest.raises(SystemExit):
        parse(flag)


def test_an_unknown_flag_and_a_prefix_are_errors():
    with pytest.raises(SystemExit):
        parse('--no_such_flag')
    with pytest.raises(SystemExit):
        parse('--lambda_rank')                  # abbreviations are off: a prefix must never silently pick a flag


def test_the_comet_key_is_not_a_flag():
    with pytest.raises(SystemExit):
        parse('--comet_api_key', 'secret-key-123')


# ---------------------------------------------------------------------------------------------------------------- what is there

def test_defaults():
    a = parse()
    assert a.rank_censor_threshold == 0.5 and a.tf_label == 'measured' and a.tf_mix == 1.0 and a.rank_list_size == 16
    assert a.block_mask_wt == 1.0 and a.block_mask_mt == 1.0 and a.block_positions == 8 and a.block_list_rows == 0 and a.grad_accum_blocks == 1
    assert a.lambda_cond_mt == 0.0 and a.cond_partner_min_dg == 1.0 and a.exclude_mutbg is False
    assert a.plateau_patience == 2 and a.cosine_decay_epochs == 5 and a.wt_convergence_metric is None and a.mt_convergence_metric is None
    assert a.target_mode_wt == 'expanded' and a.target_mode_mt == 'expanded'
    assert config.convergence_metric(a, 'wt') == 'val_rho_singles_wt_avg'
    assert config.convergence_metric(a, 'mt') == 'val_epi_cell_mag_comb'
    assert a.cache_version == config.DEFAULT_CACHE_VERSION


def test_convergence_metric_follows_head_masking_and_overrides():
    p0 = parse('--block_mask_wt', '0', '--block_mask_mt', '0')
    assert config.convergence_metric(p0, 'wt') == 'val_unmasked_rho_singles_wt_avg'
    assert config.convergence_metric(p0, 'mt') == 'val_unmasked_epi_cell_mag_comb'
    mixed = parse('--block_mask_wt', '0', '--block_mask_mt', '1')
    assert config.convergence_metric(mixed, 'wt') == 'val_unmasked_rho_singles_wt_avg'
    assert config.convergence_metric(mixed, 'mt') == 'val_epi_cell_mag_comb'
    custom = parse('--wt_convergence_metric', 'val_rho_singles_wt_avg', '--mt_convergence_metric', 'val_epi_composite')
    assert config.convergence_metric(custom, 'wt') == 'val_rho_singles_wt_avg'
    assert config.convergence_metric(custom, 'mt') == 'val_epi_composite'


def test_the_rank_censor_threshold_can_be_off_or_negative():
    assert parse('--rank_censor_threshold', 'none').rank_censor_threshold is None
    assert parse('--rank_censor_threshold', '-1.0').rank_censor_threshold == -1.0


def test_the_posterior_label_has_no_cut_and_the_measured_one_may():
    assert parse('--tf_label', 'measured', '--tf_cut', '0.5').tf_cut == 0.5
    with pytest.raises(SystemExit):
        parse('--tf_label', 'posterior', '--tf_cut', '0.5')
    with pytest.raises(SystemExit):
        parse('--tf_label', 'inverse')                            # the old name of 'measured' is gone


def test_ranges_are_validated():
    for bad in (['--tf_mix', '1.5'], ['--tf_mix', '-0.1'], ['--rank_list_size', '0'], ['--tf_prior_sd', '0'], ['--tobit_sigma_y', '-1'], ['--tobit_sigma_f', '0'],
                ['--lambda_cond_mt', '-1'], ['--block_positions', '0'], ['--block_list_rows', '-1'], ['--block_mask_mt', '1.5'], ['--block_mask_wt', '-0.1'],
                ['--block_pair_weight', '-1'], ['--grad_accum_blocks', '0'], ['--plateau_patience', '-1'], ['--cosine_decay_epochs', '0']):
        with pytest.raises(SystemExit):
            parse(*bad)
    assert parse('--tobit_mu_f', '-0.1').tobit_mu_f == -0.1                   # a mean may be anything


def test_domain_steps_set_the_step_themselves():
    with pytest.raises(SystemExit):
        parse('--block_domain_steps', '--grad_accum_blocks', '2')
    with pytest.raises(SystemExit):
        parse('--block_pair_weight', '0.38')                                   # the weight needs the domain step it is a share of
    assert parse('--block_domain_steps', '--block_pair_weight', '0.38').block_pair_weight == 0.38


def test_the_number_of_flags_is_small():
    """The point of the strip. A new flag is allowed, but it has to be a decision: raise this bound on purpose."""
    keys = set(vars(parse()))
    assert len(keys) <= 56, sorted(keys)


# ---------------------------------------------------------------------------------------------------------------- yaml

def test_yaml_supplies_defaults_and_the_command_line_wins(tmp_path):
    cfg = tmp_path / 'r.yaml'
    cfg.write_text('lambda_rank_mt: 0.25\nrank_list_size: 64\nrank_censor_threshold: 0.3\n')
    a = parse('--config', str(cfg))
    assert (a.lambda_rank_mt, a.rank_list_size, a.rank_censor_threshold) == (0.25, 64, 0.3)
    b = parse('--config', str(cfg), '--lambda_rank_mt', '1.0')
    assert b.lambda_rank_mt == 1.0 and b.rank_list_size == 64


def test_yaml_with_an_unknown_or_removed_key_is_refused(tmp_path):
    for body in ('no_such_flag: 1\n', 'censor_threshold: 0.5\n', 'adapter_mode: dual\n'):
        cfg = tmp_path / 'bad.yaml'
        cfg.write_text(body)
        with pytest.raises(SystemExit):
            parse('--config', str(cfg))


def test_every_recipe_in_configs_parses():
    files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'configs', '*.yaml'))
    assert files
    for f in files:
        a = parse('--config', f)
        assert set(yaml.safe_load(open(f))) <= set(vars(a)), f


def test_the_default_cache_version_is_the_datasets():
    from esm_msr.data import MutationStabilityDataset
    assert config.DEFAULT_CACHE_VERSION == MutationStabilityDataset.CACHE_VERSION
