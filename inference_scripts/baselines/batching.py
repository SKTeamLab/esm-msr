"""One batch-size policy for every baseline that batches variants.

What a "batch" is, per method (the unit the policy sizes is the work that one forward call does for one protein):

  ESM-MSR         variant sequences per forward call (`infer_mutants`, esm_msr/auto_batch.py is its sizer)
  ThermoMPNN(-D)  mutation items (a structure copy with one or two mutations) per forward call
  SPURS(-multi)   variants per SPURS-multi forward call (the structure pass is repeated per call)
  ProteinMPNN     decoder rows per decoder call (one row = one background-order / context combination)
  MutateEverything  none: a forward pass predicts every single and double of one protein, so the batch is one protein

Two modes, the same flags in every script (`add_batch_args`):

  --batch_mode auto   (default) the AutoBatchSizer of src/esm_msr/auto_batch.py: start at 1 for each protein, double while a
                      memory model fit on the measured peaks of the chunks already run predicts the next size stays under
                      `--auto_batch_headroom` of the GPU, bisect for the largest size that fits, fall back on an OOM by
                      halving. Small proteins reach large batches, long ones stay small, and one that does not fit even at 1
                      raises. `--batch_size N` is then a hard cap.
  --batch_mode fixed  `--batch_size N` items per forward call for every protein (a protein that does not fit fails). Without
                      --batch_size, the script's historical default.

The sizer is the same code and the same headroom for every method, measuring driver-level VRAM against the one GPU of the slot,
so the memory a method may use is the same for all of them. A protein's row in the resource table records what was used
(`batch_mode`, `batch_max`, `n_chunks`, `batch_ooms`).
"""
import importlib.util
import sys
from pathlib import Path

_SIZER = None


def _sizer_class():
    """AutoBatchSizer, loaded from src/esm_msr/auto_batch.py by path: the module only needs torch, and this avoids importing
    the rest of esm_msr into environments (SPURS, ThermoMPNN-D) that do not have its dependencies."""
    global _SIZER
    if _SIZER is None:
        path = Path(__file__).resolve().parents[2] / 'src' / 'esm_msr' / 'auto_batch.py'
        spec = importlib.util.spec_from_file_location('esm_msr_auto_batch_by_path', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod        # dataclasses look the module up by name
        spec.loader.exec_module(mod)
        _SIZER = mod.AutoBatchSizer
    return _SIZER


def add_batch_args(parser, default_batch_size=None):
    g = parser.add_argument_group('batch sizing (see baselines/batching.py)')
    g.add_argument('--batch_mode', choices=['auto', 'fixed'], default='auto',
                   help="auto: size each protein's batches dynamically against the GPU; fixed: --batch_size items per call")
    g.add_argument('--batch_size', type=int, default=default_batch_size,
                   help='fixed: items per forward call (default: the method\'s historical one); auto: hard cap on the batch')
    g.add_argument('--auto_batch_headroom', type=float, default=0.90,
                   help='auto: fraction of the GPU a predicted batch may occupy')
    return parser


class Batching:
    """`run(n_items, chunk_fn)` processes items [0, n_items) in chunks of the policy's size and returns a dict of what was used.

    chunk_fn(start, stop) must do the real work for items [start, stop) in one forward call. `default_chunk` is the method's
    historical batch size for the fixed mode when --batch_size is not given (an int, or a callable of nothing)."""

    def __init__(self, args, device='cuda:0', log=print):
        self.mode = args.batch_mode
        self.batch_size = args.batch_size
        self.headroom = args.auto_batch_headroom
        self.sizer = _sizer_class()(device, headroom=self.headroom, max_batch=self.batch_size, log=log) if self.mode == 'auto' else None

    def meta(self):
        return {'batch_mode': self.mode, 'batch_size_arg': self.batch_size,
                'auto_batch_headroom': self.headroom if self.mode == 'auto' else None}

    def run(self, n_items, chunk_fn, label='', default_chunk=None):
        if n_items <= 0:
            return {'batch_mode': self.mode, 'batch_max': 0, 'n_chunks': 0, 'batch_ooms': 0}
        if self.sizer is not None:
            r = self.sizer.run(n_items, chunk_fn, label=label)
            return {'batch_mode': 'auto', 'batch_max': int(r.max_batch), 'n_chunks': len(r.ladder), 'batch_ooms': int(r.ooms)}
        size = self.batch_size or (default_chunk() if callable(default_chunk) else default_chunk) or n_items
        size = max(1, int(size))
        n_chunks = 0
        for i in range(0, n_items, size):
            chunk_fn(i, min(i + size, n_items))
            n_chunks += 1
        return {'batch_mode': 'fixed', 'batch_max': min(size, n_items), 'n_chunks': n_chunks, 'batch_ooms': 0}


def merge_info(infos):
    """The info dicts of several `run` calls (a protein can take several) as one."""
    infos = [i for i in infos if i]
    if not infos:
        return {}
    return {'batch_mode': infos[0]['batch_mode'], 'batch_max': max(i['batch_max'] for i in infos),
            'n_chunks': sum(i['n_chunks'] for i in infos), 'batch_ooms': sum(i['batch_ooms'] for i in infos)}
