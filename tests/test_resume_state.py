"""
Resuming one of our own checkpoints has to restore the things that are NOT in the weights.

``on_save_checkpoint`` strips the frozen base out of the state_dict, which is why ``strict_loading`` has to be
False or Lightning refuses to resume at all. And two pieces of state live outside the state_dict: the PEFT
manager's transition flags, and ``_wt_plateau_step``'s own best/patience/decay counters. Lose the latter and a run
preempted mid-decay restarts the WT plateau from scratch and follows a different learning-rate schedule than it
would have.
"""
import types

import pytest

from esm_msr import training

CLS = training.ESM3EpistasisLightningModule


def stub(has_transitioned=False, best=0.83, bad=1, decays=2):
    s = types.SimpleNamespace()
    s.peft_manager = types.SimpleNamespace(has_transitioned=has_transitioned, wt_best_metric=best,
                                           wt_patience_counter=bad, frozen=[], thawed=[])
    s.peft_manager.freeze_wt_components = lambda: s.peft_manager.frozen.append('wt')
    s.peft_manager.freeze_mt_components = lambda: s.peft_manager.frozen.append('mt')
    s.peft_manager.unfreeze_mt_components = lambda: s.peft_manager.thawed.append('mt')
    s._wt_plateau_best, s._wt_plateau_bad, s._wt_decays = best, bad, decays
    s.hparams = types.SimpleNamespace(freeze_wt_adapter=False, freeze_mt_adapter=False)
    return s


def test_the_plateau_counters_round_trip():
    src = stub(best=0.8361, bad=1, decays=2)
    ck = {'state_dict': {'model.peft_wt.lora_A.weight': 1, 'link_head.raw_tau': 2,
                         'model.base_model.transformer.blocks.0.attn.weight': 3}}
    CLS.on_save_checkpoint(src, ck)
    ts = ck['transition_state']
    assert (ts['wt_plateau_best'], ts['wt_plateau_bad'], ts['wt_decays']) == (0.8361, 1, 2)

    dst = stub(best=-1, bad=0, decays=0)
    CLS.on_load_checkpoint(dst, ck)
    assert (dst._wt_plateau_best, dst._wt_plateau_bad, dst._wt_decays) == (0.8361, 1, 2)


def test_the_base_weights_are_stripped_so_a_strict_load_would_fail():
    """The reason strict_loading must be False: our checkpoints are deliberately incomplete."""
    src = stub()
    ck = {'state_dict': {'model.peft_wt.lora_A.weight': 1, 'calibration_head_wt.bias': 2,
                         'link_head.raw_tau': 3, 'model.base_model.transformer.blocks.0.attn.weight': 4}}
    CLS.on_save_checkpoint(src, ck)
    kept = set(ck['state_dict'])
    assert 'model.base_model.transformer.blocks.0.attn.weight' not in kept
    assert kept == {'model.peft_wt.lora_A.weight', 'calibration_head_wt.bias', 'link_head.raw_tau'}


def test_resuming_after_the_transition_refreezes_wt_and_thaws_mt():
    ck = {'state_dict': {'model.peft_wt.lora_A.weight': 1}}
    CLS.on_save_checkpoint(stub(has_transitioned=True), ck)
    dst = stub(has_transitioned=False)
    CLS.on_load_checkpoint(dst, ck)
    assert dst.peft_manager.has_transitioned is True
    assert 'wt' in dst.peft_manager.frozen and 'mt' in dst.peft_manager.thawed


def test_resuming_before_the_transition_leaves_both_paths_alone():
    ck = {'state_dict': {'model.peft_wt.lora_A.weight': 1}}
    CLS.on_save_checkpoint(stub(has_transitioned=False), ck)
    dst = stub(has_transitioned=True)
    CLS.on_load_checkpoint(dst, ck)
    assert dst.peft_manager.has_transitioned is False
    assert dst.peft_manager.frozen == [] and dst.peft_manager.thawed == []


def test_the_resume_flag_exists_and_defaults_to_off():
    from esm_msr import config
    import sys
    argv = sys.argv
    try:
        req = ['--raw_data_file', 'x.csv', '--af_model_folder', 'x', '--experiment_name', 'x']
        sys.argv = ['training.py'] + req
        assert config.parse_arguments().resume_from_checkpoint is None
        sys.argv = ['training.py'] + req + ['--resume_from_checkpoint', '/p/last.ckpt']
        assert config.parse_arguments().resume_from_checkpoint == '/p/last.ckpt'
    finally:
        sys.argv = argv
