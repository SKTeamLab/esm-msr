"""One batch-size policy for every baseline that batches variants.

What a "batch" is, per method (the unit the policy sizes is the work that one forward call does for one protein):

  ESM-MSR         variant sequences per forward call (`infer_mutants`, esm_msr/auto_batch.py is its sizer)
  ThermoMPNN(-D)  mutation items (a structure copy with one or two mutations) per forward call, per network (single / epistatic)
  SPURS(-multi)   variants per SPURS-multi forward call (the structure pass is repeated per call)
  ProteinMPNN     decoder rows per decoder call (one row = one background-order / context combination)
  MutateEverything  none: a forward pass predicts every single and double of one protein, so the batch is one protein

Three modes, the same flags in every script (`add_batch_args`):

  --batch_mode table  (default) one fixed batch size per model and dataset, read from `batch_sizes_gpu4.json`. The table is
                      made once on the target GPU by running the sizer on each dataset's worst-case proteins (calibration.py,
                      `--calibrate_batch_sizes`), so every method gets the largest batch that fits its heaviest protein
                      inside the same VRAM headroom. ProteinGym is `auto` in the table: its proteins differ too much for one
                      size per dataset, so it is sized dynamically per DMS.
  --batch_mode auto   the AutoBatchSizer of src/esm_msr/auto_batch.py for every protein: start at 1, double while a memory
                      model fit on the measured peaks of the chunks already run predicts the next size stays under
                      `--auto_batch_headroom` of the GPU, bisect for the largest size that fits, halve on an OOM; a protein
                      that does not fit at 1 raises. `--batch_size N` is then a hard cap. Used for calibration and ProteinGym.
  --batch_mode fixed  `--batch_size N` items per call for everything (a protein that does not fit fails); without --batch_size,
                      the script's historical default.

`--batch_size N` in table mode overrides the table (as fixed). A protein's row in the resource table records what was used
(`batch_mode`, `batch_max`, `n_chunks`, `batch_ooms`).
"""
import importlib.util
import json
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


def dataset_key(name):
    """The table's name for a dataset, whichever script named it: `external_s571` and `s571` are one, `dms_X` and `X` too."""
    if name is None:
        return None
    for prefix in ('external_', 'dms_'):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def add_batch_args(parser, default_batch_size=None, default_mode='table'):
    g = parser.add_argument_group('batch sizing (see baselines/batching.py)')
    g.add_argument('--batch_mode', choices=['table', 'auto', 'fixed'], default=default_mode,
                   help="table: the calibrated size per model and dataset (ProteinGym: dynamic); auto: size each protein's "
                        "batches dynamically against the GPU; fixed: --batch_size items per call")
    g.add_argument('--batch_size', type=int, default=default_batch_size,
                   help="table/fixed: items per forward call (overrides the table; fixed default: the method's historical one); "
                        "auto: hard cap on the batch")
    g.add_argument('--auto_batch_headroom', type=float, default=0.90,
                   help='auto: fraction of the GPU a predicted batch may occupy')
    g.add_argument('--batch_table', default=str(Path(__file__).resolve().parent / 'batch_sizes_gpu4.json'),
                   help='table mode: the calibrated batch sizes (calibration.py)')
    return parser


class Batching:
    """`run(n_items, chunk_fn)` processes items [0, n_items) in chunks of the policy's size and returns a dict of what was used.

    chunk_fn(start, stop) must do the real work for items [start, stop) in one forward call. `model` names the table's column
    (a script can pass another per call: ThermoMPNN has one per network); the script sets `dataset` as it enters each dataset.
    `default_chunk` is the method's historical batch size for the fixed mode when --batch_size is not given (an int, or a
    callable of nothing)."""

    def __init__(self, args, model, device='cuda:0', log=print):
        self.mode = args.batch_mode
        self.batch_size = args.batch_size
        self.headroom = args.auto_batch_headroom
        self.model, self.dataset, self.device, self.log = model, None, device, log
        self._sizer = None
        self.table = None
        self.table_path = getattr(args, 'batch_table', None)
        if self.mode == 'table' and self.batch_size is None:
            if not Path(self.table_path).exists():
                raise FileNotFoundError(f'no batch-size table at {self.table_path}: run --calibrate_batch_sizes on the target GPU '
                                        f'and calibration.py merge, or use --batch_mode auto|fixed')
            self.table = json.load(open(self.table_path))

    def dataset_scope(self, tracker, name, **extra):
        """A tracker `dataset` scope that also tells the policy which dataset the following `run` calls belong to."""
        self.dataset = name
        return tracker.scope('dataset', name, **extra)

    @property
    def sizer(self):
        """The AutoBatchSizer, built on first use (a table run needs one only for its `auto` entries)."""
        if self._sizer is None:
            self._sizer = _sizer_class()(self.device, headroom=self.headroom,
                                         max_batch=self.batch_size if self.mode == 'auto' else None, log=self.log)
        return self._sizer

    def table_entry(self, model=None):
        model = model or self.model
        key = dataset_key(self.dataset)
        try:
            return self.table[model][key]
        except KeyError:
            raise KeyError(f'no calibrated batch size for {model} / {key} in {self.table_path}; calibrate it '
                           f'(--calibrate_batch_sizes) or use --batch_mode auto|fixed') from None

    def meta(self):
        return {'batch_mode': self.mode, 'batch_size_arg': self.batch_size,
                'auto_batch_headroom': self.headroom if self.mode != 'fixed' else None,
                'batch_table': Path(self.table_path).name if self.table is not None else None}

    def run(self, n_items, chunk_fn, label='', default_chunk=None, model=None):
        if n_items <= 0:
            return {'batch_mode': self.mode, 'batch_max': 0, 'n_chunks': 0, 'batch_ooms': 0}
        mode, size = self.mode, self.batch_size
        if mode == 'table' and size is None:
            entry = self.table_entry(model)
            mode, size = ('auto', None) if entry == 'auto' else ('table', int(entry))
        if mode == 'auto':
            r = self.sizer.run(n_items, chunk_fn, label=label)
            return {'batch_mode': 'auto', 'batch_max': int(r.max_batch), 'n_chunks': len(r.ladder), 'batch_ooms': int(r.ooms)}
        size = size or (default_chunk() if callable(default_chunk) else default_chunk) or n_items
        size = max(1, int(size))
        n_chunks = 0
        for i in range(0, n_items, size):
            chunk_fn(i, min(i + size, n_items))
            n_chunks += 1
        return {'batch_mode': mode, 'batch_max': min(size, n_items), 'n_chunks': n_chunks, 'batch_ooms': 0}


def merge_info(infos):
    """The info dicts of several `run` calls (a protein can take several) as one."""
    infos = [i for i in infos if i]
    if not infos:
        return {}
    return {'batch_mode': infos[0]['batch_mode'], 'batch_max': max(i['batch_max'] for i in infos),
            'n_chunks': sum(i['n_chunks'] for i in infos), 'batch_ooms': sum(i['batch_ooms'] for i in infos)}
