"""Run esm_msr_testing.py with the ctx head in place of the MT head: the WT adapter read in context.

ctx = (WT + ~WT)/2, where ~WT is the WT adapter on the MUTATED sequence asked about the reverse mutation, calibrated and
negated (MSRModel.forward_cycle's wt_rev). It is the control head of the epistasis hierarchy
(esm_msr.epi_metrics, head 'ctx'): it never trains on a double, so it is what the backbone plus single-mutant training
already know about interactions, and the MT adapter's contribution is comb (or mt) minus ctx. esm_msr_testing.py does not
write it (its MT pass is the MT adapter), so this launcher runs that script UNCHANGED apart from two things:

  1. MSRModel.forward_batch is wrapped: the original runs with skip_reverse=True (the WT pass only, no MT adapter), then
     ~WT is computed as forward_cycle does. In the written predictions
         wt_lora_pred  = WT                     (unchanged, same cached WT pass)
         mt_lora_pred  = ~WT                    (the reverse leg alone)
         combined_pred = (WT + ~WT)/2           (ctx)
     so a downstream scorer reads them with its usual head columns: 'mt' = ~WT alone, 'comb' = ctx.
  2. Outputs go under OUT (default outputs/ctx_run/) instead of analysis_notebooks/, so no existing prediction or stats
     file can be overwritten. Inputs resolve exactly as in the original.

Usage (same arguments as esm_msr_testing.py, through the pdbfixer/openmm stubs of baselines/run_msr_testing.py):
    python esm_msr_ctx_testing.py --checkpoint <dir/file.ckpt> --split splits_oct06_capped --local_cluster \
        --skip_external --skip_dms --skip_domainome --local_path_to_structures <dir>
Environment: CTX_OUT overrides OUT; CTX_SMOKE=N restricts the Tsuboyama loop to the first N proteins of each half.
"""
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

for _m in ['pdbfixer', 'openmm', 'openmm.app', 'openmm.unit', 'simtk', 'simtk.openmm', 'simtk.openmm.app', 'simtk.unit']:
    sys.modules.setdefault(_m, MagicMock())

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'esm_msr_testing.py'
OUT = Path(os.environ.get('CTX_OUT', str(HERE.parent / 'outputs' / 'ctx_run')))

sys.path.insert(0, str(HERE.parent / 'src'))
from esm_msr import models  # noqa: E402  (the module esm_msr_testing imports is this same object)

_orig_forward_batch = models.MSRModel.forward_batch


def forward_batch_ctx(self, batch_in, cached_wt_esm3=None, skip_reverse=False, mask_strategy=None):
    out = _orig_forward_batch(self, batch_in, cached_wt_esm3=cached_wt_esm3, skip_reverse=True, mask_strategy=mask_strategy)
    # forward_cycle's wt_rev, line for line (older checkouts of forward_cycle also run an MT probe pass, which is not
    # wanted here): the WT adapter on the mutated sequence, asked about the reverse mutation, calibrated and negated
    r = dict(batch_in)
    r['wt_sequence_tokens'] = batch_in['mt_sequence_tokens']
    r['wt_id'], r['mt_id'] = batch_in['mt_id'], batch_in['wt_id']
    rev = -self.forward_partitioned(r, pass_type='wt', mask_strategy=mask_strategy)['pred_calibrated']
    out = dict(out)
    out['mt_lora_pred'] = rev
    out['combined_pred'] = 0.5 * (out['wt_lora_pred'] + rev)
    return out


models.MSRModel.forward_batch = forward_batch_ctx

src = TARGET.read_text()
root = "REPO_ROOT / 'analysis_notebooks'"
assert src.count(root) >= 8, f'expected the output paths of esm_msr_testing.py, found {src.count(root)}'
src = src.replace(root, f'Path({str(OUT)!r})')
if os.environ.get('CTX_SMOKE'):
    loop = "for code in tqdm(data_scaffold['code_wt'].unique()):"
    assert src.count(loop) == 1
    src = src.replace(loop, f"for code in tqdm(data_scaffold['code_wt'].unique()[:{int(os.environ['CTX_SMOKE'])}]):")
print(f'[ctx] outputs under {OUT}; MT pass replaced by the WT adapter read in context', flush=True)

sys.argv = [str(TARGET)] + sys.argv[1:]
exec(compile(src, str(TARGET), 'exec'), {'__name__': '__main__', '__file__': str(TARGET)})
