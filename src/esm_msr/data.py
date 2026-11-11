import os
import pickle
import random
import re
import sys
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

for _sub in ('', '.numeric', '.multiarray', '.umath', '._multiarray_umath', '.fromnumeric', '.records', '.memmap'):
    if f'numpy._core{_sub}' not in sys.modules:
        sys.modules[f'numpy._core{_sub}'] = getattr(np.core, _sub[1:]) if _sub else np.core

from esm.utils.constants import esm3 as C

from esm_msr import censoring

STRUCTURES_DIR = os.environ.get(
    'ESM_MSR_STRUCTURES_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'structures'))


def resolve_structure_path(path: str) -> str:
    """The benchmark tables record the absolute path of the machine that built them; fall back to the same file
    name in the repo's data/structures (or $ESM_MSR_STRUCTURES_DIR) when that path does not exist here."""
    if os.path.exists(path):
        return path
    local = os.path.join(STRUCTURES_DIR, os.path.basename(path))
    return local if os.path.exists(local) else path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# A library code carrying a mutation suffix ('1A0N_L7S') was measured in a mutant background: its singles are
# conditional effects, not wild-type-context ones. See MutationStabilityDataset._label_native_cond.
NATIVE_BACKGROUND_CODE_RE = re.compile(r'_[A-Z]\d+[A-Z]$')

# The raw measurement columns carried through to every item, in this order, as one float32 array per item
# (`meas`). Storing the whole row keeps the per-protease values and their confidence intervals available to
# analysis and to future weighting schemes without regenerating the cache; `meas_value` reads one by name.
MEAS_COLUMNS: Tuple[str, ...] = (
    'dG_ML', 'ddG_ML',                                                     # the bounded ML fit used for training
    'deltaG', 'deltaG_95CI_high', 'deltaG_95CI_low', 'deltaG_95CI',        # the unbounded two-protease fit
    'deltaG_t', 'deltaG_t_95CI_high', 'deltaG_t_95CI_low', 'deltaG_t_95CI',   # trypsin alone
    'deltaG_c', 'deltaG_c_95CI_high', 'deltaG_c_95CI_low', 'deltaG_c_95CI',   # chymotrypsin alone
    'log10_K50_t', 'log10_K50_t_95CI', 'log10_K50unfolded_t', 'fitting_error_t',
    'log10_K50_c', 'log10_K50_c_95CI', 'log10_K50unfolded_c', 'fitting_error_c',
    'log10_K50_trypsin_ML', 'log10_K50_chymotrypsin_ML',
)
_MEAS_INDEX = {name: i for i, name in enumerate(MEAS_COLUMNS)}


def meas_value(item: Dict[str, Any], name: str) -> float:
    """One raw measurement column of an item (see ``MEAS_COLUMNS``); NaN when the item has no ``meas`` array."""
    arr = item.get('meas')
    if arr is None:
        return float('nan')
    return float(arr[_MEAS_INDEX[name]])


class MutationStabilityDataset(torch.utils.data.Dataset):
    """
    One protein library (one ``code``) of stability measurements, read from its cache (``cache_build.LibraryCacheBuilder`` writes it).

    The cache holds every item the library yields; this class keeps the ones a run asks for (``subsets``) and finalises them:

    * ``native_cond`` is assigned by :meth:`_label_native_cond` for libraries whose code carries a mutation suffix, i.e. whose
      measurements were all made in a mutant background. See ``esm_msr.routing`` for what each subset means and which head owns it.
    * ``--rank_censor_threshold`` censors the items measured at or below it (``esm_msr.censoring``).

    Every item carries the raw measurement row it came from (``meas``, see ``MEAS_COLUMNS``), the absolute dG that measurement describes
    (``dG_meas``) and its library's starting dG (``dG_wt``), which together let the loss write a prediction on the assay's absolute dG
    and recognise a measurement that is only a bound.
    """

    # The cache version a run reads when it does not pin one. c6 is c5 without the `cond` items (analysis_scripts/build_cache.py convert).
    # A run pins it with --cache_version, so a code change cannot silently hand a queued job a different dataset under the same name.
    CACHE_VERSION = 'c6'

    # The bounded ML fit reports no dG outside this interval: the measured distribution has zero mass beyond
    # either end, for singles and doubles alike.
    DG_FLOOR, DG_CEILING = -1.0, 5.0

    def __init__(
        self,
        dms_name: str,
        path: str,
        score_name: str = 'ddG_ML',
        *,
        subsets: Iterable[str] = ('single', 'double'),
        label_native_cond: bool = False,
        censor_range: bool = True,
        cache_version: Optional[str] = None,
        censor_threshold: Optional[float] = None,
        min_double_dg_add: Optional[float] = None,
    ):
        """
        Args:
            dms_name: The library's ``code`` (e.g. '1A32', or '1A0N_L7S' for a mutant-background library).
            path: Cache directory. The file read is ``<path>/<dms_name>_<score_name>_<cache_version>.pkl``.
            score_name: Target column the cache was built from; 'ddG_ML' for MegaScale, 'ddG' for benchmarks.
            subsets: Which subsets to keep after loading ('single', 'double', 'cond', 'native_cond').
            label_native_cond: Label a mutant-background library's measurements ``native_cond`` instead of leaving them ``single``.
                Off (the released behaviour), they train the WT head as ordinary singles, which is strictly wrong - ddG(X | background)
                is a conditional effect, not a wild-type-context one - but is the comparison worth having. On, the label lets the
                training losses route them per head (``--native_cond_for_wt`` / ``--native_cond_for_mt``). Validation scores them as
                singles either way.
            censor_range: Keep the rows the assay bounded rather than fitted (``<-1`` / ``>5``), which the cache always stores. False
                drops them, restoring the selected-on-outcome population; see ``esm_msr.censoring``.
            censor_threshold: Items measured at or below this dG become lower-censored, bounded by it. None censors nothing here.
            min_double_dg_add: Omit double mutants whose additive stability expectation ``dG_wt + tf_A + tf_B`` is at or below this threshold (kcal/mol). None keeps all doubles.
        """
        self.score_name = score_name
        self.cache_version = cache_version or self.CACHE_VERSION
        self.dms_name = dms_name
        self.include = set(subsets)
        self.censor_threshold = censor_threshold
        self.min_double_dg_add = min_double_dg_add
        self.cache_path = os.path.join(path, f"{self.dms_name}_{self.score_name}_{self.cache_version}.pkl")
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"no cache for library {dms_name!r}: {self.cache_path}. Training reads a prebuilt cache; build or convert one with "
                f"analysis_scripts/build_cache.py (cache version {self.cache_version}).")

        logging.info(f"Loading cached data for {self.dms_name}")
        with open(self.cache_path, 'rb') as f:
            self.data: List[Dict[str, Any]] = pickle.load(f)['items']
        logging.info(f"Cache contains: {dict(Counter(i.get('subset_type', 'unknown') for i in self.data))}")

        stamp_tf_singles(self.data)
        if not censor_range:
            self._drop_range_censored()
        if self.min_double_dg_add is not None:
            self._drop_doubles_below_dg_add(float(self.min_double_dg_add))
        if label_native_cond:
            self._label_native_cond()
        self._filter_dataset()
        self._apply_censoring()
        self._extract_scalars()

    # ------------------------------------------------------------------ post-load filters

    def _drop_doubles_below_dg_add(self, min_dg_add: float) -> None:
        """Omit double mutants whose additive stability expectation ``dG_wt + tf_A + tf_B`` is not strictly above ``min_dg_add``."""
        before = len(self.data)
        kept = []
        for i in self.data:
            if i.get('subset_type') != 'double':
                kept.append(i)
                continue
            dg_wt = float(i.get('dG_wt', float('nan')))
            tfa = float(i.get('tf_A', i.get('ddG_A', float('nan'))))
            tfb = float(i.get('tf_B', i.get('ddG_B', float('nan'))))
            dg_add = dg_wt + tfa + tfb
            if np.isfinite(dg_add) and dg_add > min_dg_add:
                kept.append(i)
        self.data = kept
        if len(self.data) != before:
            logging.info(f"[{self.dms_name}] dropped {before - len(self.data)} doubles with dG_add <= {min_dg_add} (--min_double_dg_add)")

    def _drop_range_censored(self) -> None:
        """
        Remove the rows the assay only bounded, putting the training population back as it was before they were
        admitted. The cache keeps them either way, so one cache serves a run that wants them and one that does not.
        """
        before = len(self.data)
        self.data = [i for i in self.data if int(i.get('cens_src', 0)) != censoring.SRC_RANGE]
        if len(self.data) != before:
            logging.info(f"[{self.dms_name}] dropped {before - len(self.data)} range-censored items "
                         f"(--censor_range off)")

    def _label_native_cond(self) -> None:
        """
        Label items from a mutant-background library as ``native_cond``.

        A library whose code carries a mutation suffix ('1A0N_L7S' = 1A0N measured in the L7S background) has no
        wild-type counterpart: every measurement in it is already a conditional effect ddG(X | L7S). Its singles
        therefore belong to the MT head, and are relabeled here so they toggle independently of ordinary singles.
        The on-disk cache keeps the generic label, so the same cache serves a run that wants them and one that
        does not.
        """
        if not NATIVE_BACKGROUND_CODE_RE.search(self.dms_name):
            return
        n = 0
        for item in self.data:
            if item.get('subset_type') == 'single':
                item['subset_type'] = 'native_cond'
                n += 1
        if n:
            logging.info(f"[{self.dms_name}] Labeled {n} items as native_cond (mutant-background library)")

    def _filter_dataset(self) -> None:
        """Keeps only the subsets this dataset was asked for."""
        before = len(self.data)
        self.data = [item for item in self.data if item.get('subset_type') in self.include]
        logging.info(f"Filtered dataset from {before} to {len(self.data)} items based on allowed types: {sorted(self.include)}")

    def _apply_censoring(self) -> None:
        """Finalise ``cens`` / ``cens_bound`` on every item (see ``esm_msr.censoring``)."""
        self.data, c = censoring.apply_item_censoring(self.data, self.censor_threshold)
        if c['floor'] or c['range']:
            logging.info(f"[{self.dms_name}] censored {c['floor'] + c['range']} of {len(self.data)} items: "
                         f"{c['range']} bounded by the assay {dict(c['range_by_subset'])}, "
                         f"{c['floor']} at or below the threshold {self.censor_threshold} {dict(c['floor_by_subset'])}")
        else:
            logging.info(f"[{self.dms_name}] nothing censored")

    def _extract_scalars(self) -> None:
        """Contiguous scalar arrays for the sampler's and the metrics' bulk lookups."""
        self.ground_truth_arr = np.array([i.get('ddG', np.nan) for i in self.data], dtype=np.float32)
        self.cens_arr = np.array([i.get('cens', 0) for i in self.data], dtype=np.int8)

    # ------------------------------------------------------------------ dataset API

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def stamp_tf_singles(items: List[Dict[str, Any]]) -> None:
    """
    Give every double its two measured singles as ``tf_A`` / ``tf_B``, NaN where one is missing or only a range tag ('<-1', '>5'), so teacher forcing can keep the other.

    The cache's ``ddG_A`` / ``ddG_B`` cannot serve: both are NaN when either single is missing or tagged. Slot 0 is the item's first mutation, as for ``ddG_A``.
    """
    keyed = [it for it in items if 'mut_pos' in it and 'mt_id' in it]
    measured = {}
    for it in keyed:
        if it.get('subset_type') == 'single' and int(it.get('cens_src', 0)) != censoring.SRC_RANGE:
            measured[(int(np.asarray(it['mut_pos']).ravel()[0]), int(np.asarray(it['mt_id']).ravel()[0]))] = float(it.get('ddG', float('nan')))
    for it in keyed:
        if it.get('subset_type') == 'double':
            pos, mts = np.asarray(it['mut_pos']).ravel(), np.asarray(it['mt_id']).ravel()
            it['tf_A'], it['tf_B'] = (measured.get((int(pos[k]), int(mts[k])), float('nan')) for k in (0, 1))


def collate_fn_twopass(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Vectorized collation specifically built for MSRModel's homogeneous microbatch assumption.
    If the batch contains heterogeneous sequences, the torch.stack commands will intentionally 
    raise a RuntimeError, caught here and re-raised as an AssertionError to prevent silent evaluation corruption.
    """
    B = len(batch)
    
    pdb = [item['pdb'] for item in batch]
    mutations = [item['mutations'] for item in batch]
    subset_type = [item.get('subset_type', 'single') for item in batch]
    plddt = [torch.as_tensor(item['plddt'], dtype=torch.float32) for item in batch]

    ddG = torch.tensor([float(item.get('ddG', float('nan'))) for item in batch], dtype=torch.float32)
    dddG = torch.tensor([float(item.get('dddG', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_additive = torch.tensor([float(item.get('ddG_additive', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_A = torch.tensor([float(item.get('ddG_A', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_B = torch.tensor([float(item.get('ddG_B', float('nan'))) for item in batch], dtype=torch.float32)
    tf_A = torch.tensor([float(item.get('tf_A', item.get('ddG_A', float('nan')))) for item in batch], dtype=torch.float32)
    tf_B = torch.tensor([float(item.get('tf_B', item.get('ddG_B', float('nan')))) for item in batch], dtype=torch.float32)
    valid_dddG_mask = torch.tensor([bool(item.get('valid_dddG_mask', False)) for item in batch], dtype=torch.bool)

    # The assay's own scale: the library's starting dG and floor, and what this item's measurement bounds.
    dG_wt = torch.tensor([float(item.get('dG_wt', float('nan'))) for item in batch], dtype=torch.float32)
    dG_meas = torch.tensor([float(item.get('dG_meas', float('nan'))) for item in batch], dtype=torch.float32)
    cens = torch.tensor([int(item.get('cens', 0)) for item in batch], dtype=torch.long)
    cens_src = torch.tensor([int(item.get('cens_src', 0)) for item in batch], dtype=torch.long)
    cens_bound = torch.tensor([float(item.get('cens_bound', float('nan'))) for item in batch], dtype=torch.float32)
    meas = torch.stack([torch.as_tensor(item['meas'] if item.get('meas') is not None else np.full(len(MEAS_COLUMNS), np.nan),
                                        dtype=torch.float32) for item in batch], dim=0)

    # Sequence stacking. MUST be identical lengths across the batch.
    wt_seq_list = [torch.as_tensor(item['wt_sequence_tokens'], dtype=torch.long) for item in batch]
    mt_seq_list = [torch.as_tensor(item['mt_sequence_tokens'], dtype=torch.long) for item in batch]
    
    try:
        wt_seq_stack = torch.stack(wt_seq_list, dim=0)
        mt_seq_stack = torch.stack(mt_seq_list, dim=0)
    except RuntimeError:
        raise AssertionError(
            "Heterogeneous sequence lengths detected inside a single batch. "
            "Your batch sampler is violating the homogeneous microbatch assumption."
        )

    # Structural stacking
    crd_list = [torch.as_tensor(item['coords_orig'], dtype=torch.float32) for item in batch]
    crd_stack = torch.stack(crd_list, dim=0)

    str_list = [torch.as_tensor(item['structure_tokens_orig'], dtype=torch.long) for item in batch]
    str_stack = torch.stack(str_list, dim=0)
    
    plddt_stack = torch.stack(plddt, dim=0)

    try:
        ri_list = [torch.as_tensor(item['residue_index'], dtype=torch.long) for item in batch]
        ri_stack = torch.stack(ri_list, dim=0)
    except (KeyError, Exception) as e:
        logging.warning(f"Failed to stack residue_index: {e}. Setting to None.")
        ri_stack = None       

    # --- Vectorized Mutation Arrays ---
    mut_pos_list = [torch.as_tensor(item['mut_pos'], dtype=torch.long) for item in batch]
    wt_id_list = [torch.as_tensor(item['wt_id'], dtype=torch.long) for item in batch]
    mt_id_list = [torch.tensor(item['mt_id'], dtype=torch.long) for item in batch]

    lengths = [len(m) for m in mut_pos_list]
    max_len = max(lengths) if lengths else 1
    if max_len == 0: 
        max_len = 1

    # Pad to max mutations in the batch with 0 (ignored by mut_mask)
    mut_pos_stack = torch.nn.utils.rnn.pad_sequence(mut_pos_list, batch_first=True, padding_value=0)
    wt_id_stack = torch.nn.utils.rnn.pad_sequence(wt_id_list, batch_first=True, padding_value=0)
    mt_id_stack = torch.nn.utils.rnn.pad_sequence(mt_id_list, batch_first=True, padding_value=0)

    # Boolean validity mask
    mut_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l > 0:
            mut_mask[i, :l] = True

    return {
        'pdb': pdb,
        'mutations': mutations,
        'ddG': ddG,
        'dddG': dddG,
        'ddG_additive': ddG_additive,
        'ddG_A': ddG_A,
        'ddG_B': ddG_B,
        'tf_A': tf_A,
        'tf_B': tf_B,
        'valid_dddG_mask': valid_dddG_mask,
        'dG_wt': dG_wt,
        'dG_meas': dG_meas,
        'cens': cens,
        'cens_src': cens_src,
        'cens_bound': cens_bound,
        'meas': meas,
        'wt_sequence_tokens': wt_seq_stack,
        'mt_sequence_tokens': mt_seq_stack,
        'mut_pos': mut_pos_stack,
        'wt_id': wt_id_stack,
        'mt_id': mt_id_stack,
        'mut_mask': mut_mask,
        'coords': crd_stack,
        'plddt': plddt_stack,                    
        'structure_tokens': str_stack,
        'residue_index': ri_stack,
        'ground_truth': ddG,
        'subset_type': subset_type
    }




class PooledDataLoader:
    """
    Pools samples from multiple dataloaders and yields padded batches without
    repetition within an epoch. 
    
    Guarantees homogeneous WT sequences within each batch to support 
    Vectorized MSRModel execution.
    """

    def __init__(
        self,
        dataloaders: List[Iterable],
        batch_size: int,
        train_list: Optional[List[str]] = None,
        *,
        seq_pad_token_id: int = C.SEQUENCE_PAD_TOKEN,
        structure_pad_token_id: int = C.STRUCTURE_PAD_TOKEN,
        coord_pad_value: float = float("inf"),
        rng_seed: Optional[int] = None
    ):
        self.dataloaders = dataloaders
        self.batch_size = int(batch_size)
        self.train_list = train_list if train_list else [f"dataset_{i}" for i in range(len(dataloaders))]
        self.num_dataloaders = len(dataloaders)

        # IDs/values
        self.seq_pad_token_id = int(seq_pad_token_id)
        self.structure_pad_token_id = int(structure_pad_token_id)
        self.coord_pad_value = float(coord_pad_value)

        # storage
        self.dataset_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pooled_data: List[Dict[str, Any]] = []
        self.all_batches: List[List[Dict[str, Any]]] = []
        self.current_batch_idx = 0
        self.rng = random.Random(rng_seed)

        # Load data from all dataloaders
        for i, dl in enumerate(self.dataloaders):
            dataset_name = self.train_list[i]
            logging.info(f"Loading data from {dataset_name}")
            try:
                dataset = dl.dataset
                for data in dataset:
                    self.dataset_samples[dataset_name].append(data)
            except AttributeError:
                logging.info(f"Iterating through dataloader {i} to get samples")
                for batch in dl:
                    if isinstance(batch, dict):
                        bsz = None
                        for key, val in batch.items():
                            if isinstance(val, (list, tuple)):
                                bsz = len(val)
                                break
                            elif isinstance(val, torch.Tensor) and val.ndim > 0:
                                bsz = val.size(0)
                                break
                        
                        if bsz is None:
                            raise AssertionError("Failed to determine batch size during unbatching. All batch values appear to be scalars.")

                        for j in range(bsz):
                            sample = {
                                k: (v[j] if isinstance(v, (list, tuple)) and len(v) > j else
                                    (v[j] if isinstance(v, torch.Tensor) and v.size(0) > j else v))
                                for k, v in batch.items()
                            }
                            self.dataset_samples[dataset_name].append(sample)
                    elif isinstance(batch, list):
                        self.dataset_samples[dataset_name].extend(batch)
                    else:
                        self.dataset_samples[dataset_name].append(batch)

        for _, samples in self.dataset_samples.items():
            self.pooled_data.extend(samples)

        logging.info(f"Total pooled samples: {len(self.pooled_data)}")
        self._group_data_by_wt_sequence()

    def _group_data_by_wt_sequence(self):
        """
        Groups all pooled data strictly by their wild-type sequence.
        This mathematically guarantees that no batch will ever trigger the 
        heterogeneity assertion in the MSRModel.
        """
        self.grouped_data = defaultdict(list)
        for item in self.pooled_data:
            if "wt_sequence_tokens" not in item:
                logging.warning("Missing 'wt_sequence_tokens' in pooled data. Cannot group for homogeneous batches.")
                seq = item["sequence_tokens_orig"]
            else:
                seq = item["wt_sequence_tokens"]
            if isinstance(seq, torch.Tensor) or isinstance(seq, np.ndarray):
                key = tuple(seq.tolist())
            else:
                key = tuple(seq)
                
            self.grouped_data[key].append(item)
            
        logging.info(f"Grouped data into {len(self.grouped_data)} unique WT sequence clusters.")
        self._build_batches()

    def _build_batches(self):
        """Chunks grouped data into specific batch sizes."""
        self.all_batches = []
        for key, group_items in self.grouped_data.items():
            for i in range(0, len(group_items), self.batch_size):
                self.all_batches.append(group_items[i:i + self.batch_size])
                
        self.batches_per_epoch = len(self.all_batches)
        logging.info(f"Batches per epoch after homogeneous grouping: {self.batches_per_epoch}")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        self.rng.shuffle(self.all_batches)
        self.current_batch_idx = 0
        return self

    def __next__(self):
        if self.current_batch_idx >= self.batches_per_epoch:
            raise StopIteration

        batch_items = self.all_batches[self.current_batch_idx]

        batch = self._collate_with_padding(batch_items)

        self.current_batch_idx += 1
        return batch

    # Collation helpers
    @staticmethod
    def _to_tensor_long(x: Any) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) and x.dtype == torch.long else torch.as_tensor(x, dtype=torch.long)

    @staticmethod
    def _to_tensor_float(x: Any) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) and x.dtype.is_floating_point else torch.as_tensor(x, dtype=torch.float)

    @staticmethod
    def _ensure_2d_structure(st: Union[np.ndarray, torch.Tensor, List[int]]) -> torch.Tensor:
        t = PooledDataLoader._to_tensor_long(st)
        if t.ndim == 1:
            t = t.unsqueeze(0)  
        if t.ndim != 2:
            raise AssertionError(f"structure tokens must be [K,L] or [L]; got shape {tuple(t.shape)}")
        return t

    @staticmethod
    def _ensure_coords_residue_axis(coords: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        t = PooledDataLoader._to_tensor_float(coords)
        if t.ndim not in (3, 4):
            raise AssertionError(f"coords must be 3D or 4D with residue axis present; got {tuple(t.shape)}")
        return t

    @staticmethod
    def _right_pad_last_dim(x: torch.Tensor, Lmax: int, pad_val: Union[int, float]) -> torch.Tensor:
        need = Lmax - x.size(-1)
        if need <= 0:
            return x
        return F.pad(x, (0, need), value=pad_val)

    def _collate_with_padding(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Vectorized collation.
        Strictly enforces homogeneous microbatch requirement of MSRModel if B > 1.
        """
        if not batch:
            return {}

        B = len(batch)

        def get_wt_seq_arr(item): 
            if "wt_sequence_tokens" not in item: 
                raise AssertionError("Missing 'wt_sequence_tokens' in dataset item.")
            return np.asarray(item["wt_sequence_tokens"])
            
        def get_mt_seq_arr(item): 
            if "mt_sequence_tokens" not in item: 
                raise AssertionError("Missing 'mt_sequence_tokens' in dataset item.")
            return np.asarray(item["mt_sequence_tokens"])

        def get_str_arr(item):
            key = "structure_tokens_orig" if "structure_tokens_orig" in item else "structure_tokens"
            return np.asarray(item[key])

        def get_crd_arr(item):
            key = "coords_orig" if "coords_orig" in item else "coords"
            return np.asarray(item[key])

        # --- Sequences & Homogeneity Check ---
        lengths: List[int] = []
        wt_seq_list_1d: List[torch.Tensor] = []
        mt_seq_list_1d: List[torch.Tensor] = []
        
        for it in batch:
            s_wt = self._to_tensor_long(get_wt_seq_arr(it))
            s_mt = self._to_tensor_long(get_mt_seq_arr(it))
            if s_wt.ndim != 1:
                raise AssertionError(f"sequence must be [L]; got shape {tuple(s_wt.shape)}")
            wt_seq_list_1d.append(s_wt)
            mt_seq_list_1d.append(s_mt)
            lengths.append(int(s_wt.size(0)))
            
        Lmax = max(lengths)

        wt_seq_pad = []
        mt_seq_pad = []
        for s_w, s_m in zip(wt_seq_list_1d, mt_seq_list_1d):
            wt_seq_pad.append(self._right_pad_last_dim(s_w, Lmax, self.seq_pad_token_id))
            mt_seq_pad.append(self._right_pad_last_dim(s_m, Lmax, self.seq_pad_token_id))
            
        wt_sequence_tokens = torch.stack(wt_seq_pad, dim=0)  # [B, Lmax] long
        mt_sequence_tokens = torch.stack(mt_seq_pad, dim=0)

        if B > 1:
            for i in range(1, B):
                if not torch.equal(wt_sequence_tokens[0], wt_sequence_tokens[i]):
                    raise AssertionError(
                        f"Heterogeneous WT sequences detected in batch. MSRModel requires homogeneous "
                        f"WT contexts for vectorization. Found lengths {lengths[0]} vs {lengths[i]} or mismatched tokens."
                    )

        # --- Structure tokens ---
        struct_pad = []
        for it in batch:
            st = self._ensure_2d_structure(get_str_arr(it))  # [K, L]
            K, L = st.shape
            if L != len(np.asarray(get_wt_seq_arr(it))):
                raise AssertionError("Structure/sequence residue length mismatch before pad.")
            st_p = self._right_pad_last_dim(st, Lmax, self.structure_pad_token_id)  # [K, Lmax]
            struct_pad.append(st_p)
        structure_tokens = torch.stack(struct_pad, dim=0)  # [B, K, Lmax] long

        # --- pLDDT ---
        plddt_pad = []
        for it in batch:
            if 'plddt' not in it: 
                raise AssertionError("Missing 'plddt' in dataset item.")
            pt = it['plddt']  # [K, L]
            pt = self._ensure_2d_structure(pt)  # [K, L]
            pt_p = self._right_pad_last_dim(pt, Lmax, 0)  # [K, Lmax]
            plddt_pad.append(pt_p)
        plddt = torch.stack(plddt_pad, dim=0)  # [B, K, Lmax] float

        # --- Coordinates ---
        coords_list = []
        for it in batch:
            c = self._ensure_coords_residue_axis(get_crd_arr(it))  # 3D or 4D
            if c.ndim == 3:
                c_p = self._right_pad_last_dim(c, Lmax, self.coord_pad_value)  # [Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))  # [1, Lmax, A1, A2]
            else:
                c_perm = c.permute(0, 2, 3, 1)            # [Kc, A1, A2, L]
                c_p = self._right_pad_last_dim(c_perm, Lmax, self.coord_pad_value)  # [Kc, A1, A2, Lmax]
                c_p = c_p.permute(0, 3, 1, 2)             # [Kc, Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))       # [1, Kc, Lmax, A1, A2]
        coords = torch.cat(coords_list, dim=0).to(torch.float32)

        # --- Basic metadata & labels ---
        pdb = [it.get("pdb", f"unk_{i}") for i, it in enumerate(batch)]
        st = [it.get("subset_type", None) for i, it in enumerate(batch)]

        ddG = torch.tensor([float(it.get('ddG', float('nan'))) for it in batch], dtype=torch.float32)
        dddG = torch.tensor([float(it.get('dddG', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_additive = torch.tensor([float(it.get('ddG_additive', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_A = torch.tensor([float(it.get('ddG_A', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_B = torch.tensor([float(it.get('ddG_B', float('nan'))) for it in batch], dtype=torch.float32)
        tf_A = torch.tensor([float(it.get('tf_A', it.get('ddG_A', float('nan')))) for it in batch], dtype=torch.float32)
        tf_B = torch.tensor([float(it.get('tf_B', it.get('ddG_B', float('nan')))) for it in batch], dtype=torch.float32)
        valid_dddG_mask = torch.tensor([bool(it.get('valid_dddG_mask', False)) for it in batch], dtype=torch.bool)
        dG_wt = torch.tensor([float(it.get('dG_wt', float('nan'))) for it in batch], dtype=torch.float32)
        dG_meas = torch.tensor([float(it.get('dG_meas', float('nan'))) for it in batch], dtype=torch.float32)
        cens = torch.tensor([int(it.get('cens', 0)) for it in batch], dtype=torch.long)
        cens_src = torch.tensor([int(it.get('cens_src', 0)) for it in batch], dtype=torch.long)
        cens_bound = torch.tensor([float(it.get('cens_bound', float('nan'))) for it in batch], dtype=torch.float32)
        meas = torch.stack([torch.as_tensor(it['meas'] if it.get('meas') is not None else np.full(len(MEAS_COLUMNS), np.nan),
                                            dtype=torch.float32) for it in batch], dim=0)

        # --- Vectorized Mutation Arrays ---
        mutations = [it.get("mutations", []) for it in batch]
        
        mut_pos_list = [torch.as_tensor(it['mut_pos'], dtype=torch.long) for it in batch]
        wt_id_list = [torch.as_tensor(it['wt_id'], dtype=torch.long) for it in batch]
        mt_id_list = [torch.tensor(it['mt_id'], dtype=torch.long) for it in batch]

        lengths_muts = [len(m) for m in mut_pos_list]
        max_muts = max(lengths_muts) if lengths_muts else 1
        if max_muts == 0: 
            max_muts = 1

        mut_pos_stack = torch.nn.utils.rnn.pad_sequence(mut_pos_list, batch_first=True, padding_value=0)
        wt_id_stack = torch.nn.utils.rnn.pad_sequence(wt_id_list, batch_first=True, padding_value=0)
        mt_id_stack = torch.nn.utils.rnn.pad_sequence(mt_id_list, batch_first=True, padding_value=0)

        mut_mask = torch.zeros(B, max_muts, dtype=torch.bool)
        for i, l in enumerate(lengths_muts):
            if l > 0:
                mut_mask[i, :l] = True

        # --- Optional residue_index ---
        try:
            ri_list = [torch.as_tensor(it['residue_index'], dtype=torch.long) for it in batch]
            ri_stack = torch.stack(ri_list, dim=0)
        except (KeyError, Exception) as e:
            logging.error(f"Failed to stack residue_index: {e}. Setting to None.")
            ri_stack = None       

        collated = {
            "pdb": pdb,
            "mutations": mutations,
            "wt_sequence_tokens": wt_sequence_tokens,
            "mt_sequence_tokens": mt_sequence_tokens,
            "structure_tokens": structure_tokens,
            "coords": coords,
            "plddt": plddt,
            "residue_index": ri_stack,
            "mut_pos": mut_pos_stack,
            "wt_id": wt_id_stack,
            "mt_id": mt_id_stack,
            "mut_mask": mut_mask,
            "lengths": torch.as_tensor(lengths, dtype=torch.long),
            "ddG": ddG,
            "dddG": dddG,
            "ddG_additive": ddG_additive,
            "ddG_A": ddG_A,
            "ddG_B": ddG_B,
            "tf_A": tf_A,
            "tf_B": tf_B,
            "valid_dddG_mask": valid_dddG_mask,
            "dG_wt": dG_wt,
            "dG_meas": dG_meas,
            "cens": cens,
            "cens_src": cens_src,
            "cens_bound": cens_bound,
            "meas": meas,
            "subset_type": st,
        }
        return collated
