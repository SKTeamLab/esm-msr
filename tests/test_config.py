"""The command line after the trim: about 55 flags, a table of retired ones that old launchers still pass, YAML defaults, and the few
translations (subset caps -> subsets, assay floor -> link floor, the Comet key -> the environment)."""
import os
import sys

import pytest

from esm_msr import config

BASE = ['training.py', '--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']


def parse(*extra, base=BASE):
    argv = sys.argv
    try:
        sys.argv = list(base) + list(extra)
        return config.parse_arguments()
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------------------------------------------- the table

def _old_value(spec):
    return [] if spec['kind'] in ('bool', 'flag') else ['1' if spec['ok'] is config._ANY else str(spec['ok'])]


@pytest.mark.parametrize('name', sorted(config.RETIRED))
def test_every_retired_flag_with_its_old_value_parses_and_is_dropped(name):
    spec = config.RETIRED[name]
    if spec['kind'] == 'flag':
        pytest.skip('a flag whose presence is the behaviour change (--regenerate_cache)')
    if spec['kind'] == 'bool':
        args = (f'--{name}',) if spec['ok'] is config._ANY or spec['ok'] else (f'--no-{name}',)
    else:
        args = (f'--{name}', *_old_value(spec))
        if spec['ok'] is None:
            args = (f'--{name}', 'none')
    a = parse(*args)
    if name not in config.DERIVED:
        assert not hasattr(a, name), name


def test_retired_flags_never_reach_hparams():
    keys = set(vars(parse('--rank_loss', 'listmle', '--detach_ensemble_input', '--no-use_dora_wt')))
    assert not keys & {'rank_loss', 'detach_ensemble_input', 'use_dora_wt'}


@pytest.mark.parametrize('extra', [('--no-detach_ensemble_input',), ('--rank_loss', 'mse'), ('--link_ceiling', '5.0'), ('--nonspecific_term', 'fx'),
                                   ('--regenerate_cache',), ('--lora_mode', 'corrector'), ('--combined_mode', 'route'),
                                   ('--alternate_heads',), ('--wt_trains_doubles',), ('--mask_strategy', 'marginal'),
                                   ('--wt_plateau_factor', '0.1'), ('--mt_plateau_patience', '0'), ('--wt_freeze_after_decays', '2'),
                                   ('--subset_caps', 'double=0.5'), ('--subset_caps', 'cond=none'), ('--use_dora_mt',),
                                   ('--censor_threshold_double', '0.0'), ('--early_stopping_metric', 'val_rho_wt_valid_avg'),
                                   ('--freeze_wt_adapter',), ('--learn_global_floor',), ('--micro_batch_tokens', '3700')])
def test_a_retired_flag_asking_for_the_old_behaviour_is_an_error(extra):
    with pytest.raises(SystemExit):
        parse(*extra)


def test_an_unknown_flag_and_a_prefix_are_errors():
    with pytest.raises(SystemExit):
        parse('--no_such_flag')
    with pytest.raises(SystemExit):
        parse('--lambda_rank')                  # abbreviations are off: a prefix must never silently pick a flag


# ---------------------------------------------------------------------------------------------------------------- translations

def test_subset_caps_become_subsets():
    assert parse().subsets == ['single', 'double']
    assert parse('--subset_caps', 'single=None', 'double=0.0', 'cond=0.0', 'native_cond=0.0').subsets == ['single']
    assert parse('--subset_caps', 'single=None', 'double=None', 'cond=0.0', 'native_cond=0.0').subsets == ['single', 'double']
    assert parse('--no-incl_doubles').subsets == ['single']
    assert parse('--subsets', 'double').subsets == ['double']
    assert parse('--subsets', 'double', '--subset_caps', 'double=0.0').subsets == ['double']      # --subsets wins


def test_bad_subsets_are_refused():
    for bad in ('cond', 'native_cond', 'single,triple', ''):
        with pytest.raises(SystemExit):
            parse('--subsets', bad)


def test_assay_floor_becomes_link_floor_and_threshold_is_one_number():
    a = parse('--assay_floor', '0.0', '--link_floor_min', '0.0', '--link_floor_max', '0.001')
    assert a.link_floor == 0.0 and not hasattr(a, 'assay_floor')
    assert parse('--assay_floor', '0.25').link_floor == 0.25
    assert parse().censor_threshold == 0.5
    assert parse('--censor_threshold', 'none').censor_threshold is None
    assert parse('--censor_threshold', '0.0', '--censor_threshold_double', '0').censor_threshold == 0.0
    assert parse('--no-censor_floor').censor_threshold is None


def test_reg_loss_huber_becomes_a_delta():
    assert parse().huber_delta == 0.0
    assert parse('--reg_loss', 'huber').huber_delta == 1.0
    assert parse('--reg_loss', 'huber', '--huber_delta', '2.0').huber_delta == 2.0
    assert parse('--reg_loss', 'mse').huber_delta == 0.0


def test_the_comet_key_goes_to_the_environment_and_is_not_recorded(monkeypatch):
    monkeypatch.delenv('COMET_API_KEY', raising=False)
    a = parse('--comet_api_key', 'secret-key-123')
    assert os.environ['COMET_API_KEY'] == 'secret-key-123'
    assert 'secret-key-123' not in repr(vars(a)) and not hasattr(a, 'comet_api_key')


def test_the_shared_plateau_accepts_the_old_per_head_flags_when_they_agree():
    a = parse('--wt_plateau_patience', '1', '--wt_plateau_factor', '0.3', '--mt_freeze_after_decays', '3',
              '--mt_plateau_patience', '1', '--mt_plateau_factor', '0.3', '--wt_freeze_after_decays', '3')
    assert (a.plateau_patience, a.plateau_factor, a.freeze_after_decays) == (1, 0.3, 3)
    b = parse('--plateau_factor', '0.1', '--wt_plateau_factor', '0.1')
    assert b.plateau_factor == 0.1


def test_the_old_aliases_still_work():
    assert parse('--lambda_rank_combined', '0.2').lambda_rank_mt == 0.2
    assert parse('--lambda_reg_combined', '3').lambda_reg_mt == 3.0
    assert parse('--subset_size', '32').rank_list_size == 32


# ---------------------------------------------------------------------------------------------------------------- derived

def test_what_inference_reads_back_is_recorded_as_constants():
    a = parse()
    assert a.per_mutation_bias is True and a.tie_mt_scale is False and a.lora_mode == 'ensemble'
    assert parse('--tie_mt_scale').tie_mt_scale is True              # a real flag, default off


def test_frozen_wt_derives_its_pieces(tmp_path):
    ck = tmp_path / 'w.ckpt'
    ck.write_bytes(b'x')
    a = parse('--init_wt_from_checkpoint', str(ck), '--frozen_wt_mt_only')
    assert a.freeze_wt_adapter is True and a.lambda_reg_wt == 0.0 and a.lambda_rank_wt == 0.0
    with pytest.raises(SystemExit):
        parse('--frozen_wt_mt_only')                                             # no checkpoint to freeze
    with pytest.raises(SystemExit):
        parse('--init_wt_from_checkpoint', str(ck), '--frozen_wt_mt_only', '--subsets', 'single')   # nothing left to train on
    assert parse('--init_wt_from_checkpoint', str(ck), '--frozen_wt_mt_only', '--freeze_wt_adapter').freeze_wt_adapter is True


def test_the_number_of_flags_is_small():
    """The point of the trim. A new flag is allowed, but it has to be a decision: raise this bound on purpose."""
    keys = set(vars(parse()))
    assert len(keys) <= 69, sorted(keys)


# ---------------------------------------------------------------------------------------------------------------- yaml

def test_yaml_supplies_defaults_and_the_command_line_wins(tmp_path):
    cfg = tmp_path / 'r.yaml'
    cfg.write_text('lambda_rank_mt: 0.25\nrank_list_size: 64\nsubsets: [single]\ncensor_threshold: 0.3\n')
    a = parse('--config', str(cfg))
    assert (a.lambda_rank_mt, a.rank_list_size, a.subsets, a.censor_threshold) == (0.25, 64, ['single'], 0.3)
    b = parse('--config', str(cfg), '--lambda_rank_mt', '1.0')
    assert b.lambda_rank_mt == 1.0 and b.rank_list_size == 64


def test_yaml_with_an_unknown_or_retired_key_is_refused(tmp_path):
    for body in ('no_such_flag: 1\n', 'gpus: 1\n', 'link_ceiling: 5.0\n'):
        cfg = tmp_path / 'bad.yaml'
        cfg.write_text(body)
        with pytest.raises(SystemExit):
            parse('--config', str(cfg))


def test_the_default_cache_version_is_the_datasets():
    from esm_msr.data import MutationStabilityDataset
    assert config.DEFAULT_CACHE_VERSION == MutationStabilityDataset.CACHE_VERSION
