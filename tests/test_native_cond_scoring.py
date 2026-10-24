"""--native_cond_for_wt / --native_cond_for_mt route mutant-background items; they must not move the yardstick.

Relabelling a mutant-background library's singles 'native_cond' used to drop that library from rho_singles_wt (and so
from val_rho_singles_wt_avg): the frozen gold WT head read 0.8536 on 20 libraries instead of 0.8510 on 25, with nothing
changed. Validation now scores every run on the released labelling. The MT-pass input swap is a property of a
--native_cond_for_mt model, not of the label, so a run that only excludes the items keeps the ordinary MT input.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_external_val_sets import _bare_module, _loader_output


def _logged_for(subset_label):
    from esm_msr.training import ESM3EpistasisLightningModule
    obj = _bare_module()
    obj.external_val_sets = set()
    obj.val_dataloader_names = ['LIB1', '2KT8_G54S']
    bg = _loader_output('2KT8_G54S', seed=3)
    bg['subset_types'] = [subset_label] * len(bg['ground_truths'])
    obj.validation_step_outputs = {0: [_loader_output('LIB1', seed=0)], 1: [bg]}
    ESM3EpistasisLightningModule.on_validation_epoch_end(obj)
    return obj.logged


def test_relabelled_native_cond_is_scored_like_the_released_singles():
    released, routed = _logged_for('single'), _logged_for('native_cond')
    assert 'val_rho_singles_wt/2KT8_G54S' in routed
    for key in ('val_rho_singles_wt_avg', 'val_rho_combined_avg', 'val_rho_singles_wt/2KT8_G54S'):
        assert routed[key] == pytest.approx(released[key])


def _mt_input(flag):
    from esm_msr.models import MSRModel
    batch = {'subset_type': ['single', 'native_cond'],
             'wt_sequence_tokens': torch.tensor([[1, 2, 3], [1, 2, 3]]),
             'mt_sequence_tokens': torch.tensor([[1, 9, 3], [1, 9, 3]])}
    return MSRModel._mt_pass_batch(SimpleNamespace(native_cond_mt_input=flag), batch)['mt_sequence_tokens']


def test_mt_input_swap_only_in_a_native_cond_for_mt_model():
    assert _mt_input(False).tolist() == [[1, 9, 3], [1, 9, 3]]
    assert _mt_input(True).tolist() == [[1, 9, 3], [1, 2, 3]]
