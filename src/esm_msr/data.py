import os
import pickle
import random
import re
import logging
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, ConcatDataset

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
    * every item gets ``y_floor = link_floor`` (the link's one knee), so a dump written by this code is read by the same
      ``epi_metrics`` as an old one.
    * ``--censor_threshold`` censors the items measured at or below it (``esm_msr.censoring``).

    Every item carries the raw measurement row it came from (``meas``, see ``MEAS_COLUMNS``), the absolute dG that measurement describes
    (``dG_meas``) and its library's starting dG (``dG_wt``), which together let the loss put a prediction on the assay's observed scale
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
        link_floor: float = 0.0,
        censor_threshold: Optional[float] = None,
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
            link_floor: The link's knee, stamped on every item as ``y_floor``.
            censor_threshold: Items measured at or below this dG become lower-censored, bounded by it. None censors nothing here.
        """
        self.score_name = score_name
        self.cache_version = cache_version or self.CACHE_VERSION
        self.dms_name = dms_name
        self.include = set(subsets)
        self.link_floor = float(link_floor)
        self.censor_threshold = censor_threshold
        self.cache_path = os.path.join(path, f"{self.dms_name}_{self.score_name}_{self.cache_version}.pkl")
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"no cache for library {dms_name!r}: {self.cache_path}. Training reads a prebuilt cache; build or convert one with "
                f"analysis_scripts/build_cache.py (cache version {self.cache_version}).")

        logging.info(f"Loading cached data for {self.dms_name}")
        with open(self.cache_path, 'rb') as f:
            self.data: List[Dict[str, Any]] = pickle.load(f)['items']
        logging.info(f"Cache contains: {dict(Counter(i.get('subset_type', 'unknown') for i in self.data))}")

        if not censor_range:
            self._drop_range_censored()
        if label_native_cond:
            self._label_native_cond()
        self._filter_dataset()
        self._stamp_floor()
        self._apply_censoring()
        self._extract_scalars()

    # ------------------------------------------------------------------ post-load filters

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

    def _stamp_floor(self) -> None:
        """Give every item the link's one knee, as ``y_floor``."""
        for item in self.data:
            item['y_floor'] = self.link_floor

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
    valid_dddG_mask = torch.tensor([bool(item.get('valid_dddG_mask', False)) for item in batch], dtype=torch.bool)

    # The assay's own scale: the library's starting dG and floor, and what this item's measurement bounds.
    dG_wt = torch.tensor([float(item.get('dG_wt', float('nan'))) for item in batch], dtype=torch.float32)
    y_floor = torch.tensor([float(item.get('y_floor', float('nan'))) for item in batch], dtype=torch.float32)
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
        'valid_dddG_mask': valid_dddG_mask,
        'dG_wt': dG_wt,
        'y_floor': y_floor,
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


class ProteinCyclingBatchSampler(Sampler[List[int]]):
    """
    Consolidated, high-performance BatchSampler for cycling through multiple protein datasets. Yields grouped indices for a
    ConcatDataset, every batch holding the items of ONE library (the homogeneous-microbatch assumption the model's collate relies on).

    Each epoch every library's items are shuffled, cut into full batches (a remainder smaller than ``batch_size`` is dropped, a
    different one each epoch), and the batches of all libraries are shuffled together. Which subsets a run trains on is decided
    earlier, when the dataset is built (``--subsets``); there is deliberately no density-based subsampling or per-subset capping here
    (an earlier version binned the doubles over (additive expectation, dddG) and capped each bin, which reshapes the ddG
    distribution the model is trained on).
    """
    SUBSET_ORDER = ['single', 'double', 'native_cond']      # the order items are laid out in before the shuffle: it fixes the RNG stream

    def __init__(
        self,
        datasets: List[Any],
        batch_size: int,
        train_list: List[str],
        *,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.train_list = train_list
        self.verbose = verbose
        self.rng_seed = rng_seed
        self.sampler = self      # allows Lightning _set_sampler_epoch to discover set_epoch on this batch_sampler

        self._completed_epochs = 0
        self._active_batches: Optional[List[List[int]]] = None
        self._rng = random.Random(rng_seed)

        # 1. Map cumulative offsets for ConcatDataset
        self.cumulative_sizes = ConcatDataset(self.datasets).cumulative_sizes
        self.offsets = [0] + self.cumulative_sizes[:-1]

        # 2. Cache subset classifications ONCE during initialization
        self._cached_buckets: List[Dict[str, List[int]]] = []

        logging.info("Caching dataset subsets for Sampler...")
        for ds_idx, ds in enumerate(self.datasets):
            buckets = {k: [] for k in self.SUBSET_ORDER}
            first_pdb = ds[0].get('pdb') if len(ds) > 0 else None

            for i in range(len(ds)):
                item = ds[i]
                if item.get('pdb') != first_pdb:
                    raise AssertionError(f"PDB ID mismatch in {self.train_list[ds_idx]}: item {i} has {item.get('pdb')}, expected {first_pdb}")

                stype = item.get('subset_type', 'single')
                if stype not in buckets:
                    stype = 'single'
                buckets[stype].append(i)

            self._cached_buckets.append(buckets)

        # 3. Perform a dry-run to calculate exact batch sizes for the DataLoader __len__
        # Matches the exact initial RNG consumption of the continuous run.
        self.num_batches = self._calculate_epoch_batches(dry_run=True)

    def _calculate_epoch_batches(self, dry_run: bool = False) -> int:
        """Core logic to shuffle and chunk datasets."""
        self.all_batches = []   # List of dataset batch lists

        for idx, ds in enumerate(self.datasets):
            protein_name = self.train_list[idx]
            buckets = self._cached_buckets[idx]
            if sum(len(v) for v in buckets.values()) == 0:
                raise AssertionError(f"No samples for {protein_name}.")

            # Flatten and global offset map for ConcatDataset
            flat_indices = []
            for items in buckets.values():
                flat_indices.extend(items)

            self._rng.shuffle(flat_indices)
            offset = self.offsets[idx]

            # Create batches for this dataset
            ds_batches = [
                [i + offset for i in flat_indices[start:start + self.batch_size]]
                for start in range(0, len(flat_indices), self.batch_size)
            ]

            # Filter out undersized batches based on PyTorch default behavior expectations
            ds_batches = [b for b in ds_batches if len(b) == self.batch_size]
            self.all_batches.append(ds_batches)

        final_batches = []
        for ds_batch_list in self.all_batches:
            final_batches.extend(ds_batch_list)
        self._rng.shuffle(final_batches)

        if not dry_run:
            self._active_batches = final_batches

        return len(final_batches)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for resumable and deterministic shuffling across preemptions."""
        if epoch == self._completed_epochs and self._active_batches is not None:
            return

        if self.rng_seed is not None and epoch != self._completed_epochs:
            logging.info(f"ProteinCyclingBatchSampler: fast-forwarding RNG stream from epoch {self._completed_epochs} to {epoch}")
            self._rng = random.Random(self.rng_seed)
            self._calculate_epoch_batches(dry_run=True)
            for _ in range(epoch):
                self._calculate_epoch_batches(dry_run=True)
            self._completed_epochs = epoch

        self._calculate_epoch_batches(dry_run=False)

    def state_dict(self) -> dict:
        return {
            'epoch': self._completed_epochs,
            'batch_size': self.batch_size,
            'rng_seed': self.rng_seed,
        }

    def load_state_dict(self, state: dict) -> None:
        if 'epoch' in state:
            self.set_epoch(state['epoch'])

    def __iter__(self) -> Iterator[List[int]]:
        """Invoked by PyTorch workers at the start of every epoch."""
        if self._active_batches is None:
            self._calculate_epoch_batches(dry_run=False)
        batches = self._active_batches
        self._active_batches = None
        self._completed_epochs += 1
        yield from batches

    def __len__(self) -> int:
        return self.num_batches


def create_consolidated_dataloader(
    dataloaders: List[DataLoader],
    train_list: List[str],
    batch_size: int,
    collate_fn: Callable,
    num_workers: int = 4,
    pin_memory: bool = True,
    rng_seed: Optional[int] = None
) -> DataLoader:
    """A DataLoader over every library's dataset at once, batched by ProteinCyclingBatchSampler so each batch
    holds one library (the homogeneous-microbatch assumption the model's collate relies on)."""
    datasets = [dl.dataset for dl in dataloaders]

    sampler = ProteinCyclingBatchSampler(
        datasets=datasets,
        batch_size=batch_size,
        train_list=train_list,
        # Without this the sampler seeds itself from system entropy, so --seed controlled only the model init: the batch
        # composition differed every run, runs were not reproducible, and two configurations could not be paired on a seed.
        rng_seed=rng_seed,
    )

    concat_dataset = ConcatDataset(datasets)

    return DataLoader(
        concat_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


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
        valid_dddG_mask = torch.tensor([bool(it.get('valid_dddG_mask', False)) for it in batch], dtype=torch.bool)
        dG_wt = torch.tensor([float(it.get('dG_wt', float('nan'))) for it in batch], dtype=torch.float32)
        y_floor = torch.tensor([float(it.get('y_floor', float('nan'))) for it in batch], dtype=torch.float32)
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
            "valid_dddG_mask": valid_dddG_mask,
            "dG_wt": dG_wt,
            "y_floor": y_floor,
            "dG_meas": dG_meas,
            "cens": cens,
            "cens_src": cens_src,
            "cens_bound": cens_bound,
            "meas": meas,
            "subset_type": st,
        }
        return collated
