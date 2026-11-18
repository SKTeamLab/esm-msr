"""``--last_n_layers_{wt,mt}``: LoRA only in the last N of ESM3's 48 transformer blocks."""
import re

import pytest

from esm_msr.models import MSRModel


@pytest.mark.parametrize('n,blocks', [(0, range(48)), (4, range(44, 48)), (48, range(48))])
def test_last_n_layers_restricts_the_lora_targets(n, blocks):
    m = MSRModel.__new__(MSRModel)
    cfg = m._create_lora_config({'lora_rank': 2, 'lora_alpha': 4, 'last_n_layers': n, 'target_mode': 'expanded'})
    rx = re.compile(cfg.target_modules)
    hit = lambda i, leaf: rx.match(f'transformer.blocks.{i}.{leaf}') is not None
    for i in range(48):
        for leaf in ('attn.layernorm_qkv.1', 'ffn.1', 'ffn.3'):
            assert hit(i, leaf) == (i in blocks), (i, leaf)
