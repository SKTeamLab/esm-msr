import os
import gc
import math
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
from tqdm import tqdm

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C

from esm_msr import censoring
from esm_msr.utils import custom_end_gap_alignment, determine_diffs

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


def floor_evidence(dG: np.ndarray, x: np.ndarray, dG_wt: float, *, x_max: float = -1.5, min_n: int = 20,
                   max_sd: float = 0.5, band=(-1.0, 0.2)) -> Dict[str, float]:
    """
    Is there evidence that this library has a floor, and if so where is it?

    The floor is a property of the READOUT, not of the variants: the assay infers dG by comparing a variant's K50
    against a K50 predicted for its unfolded sequence from composition, so a variant that is unfolded anyway reports
    the error of that prediction - a constant of the library and the protease. A library only shows us that constant
    if some of its variants are actually pinned at it. Many libraries contain nothing that destabilised; asking them
    where their floor is produces a number that is really a quantile of their own mutational sensitivity (see
    docs/devel_conservative.md for the measurements), and censoring against it discards good measurements.

    So the floor is only accepted when all three hold, on the variants whose ADDITIVE EXPECTATION
    ``x = dG_wt + ddG_A + ddG_B`` is below ``x_max`` - the ones expected far beneath any floor, and therefore
    expected to be pinned:

    1. ``n >= min_n``: enough of them to estimate anything.
    2. ``sd <= max_sd``: they AGREE. Variants predicted to sit at -3 kcal/mol that come back at one common value are
       pinned at that value; variants that come back spread over 2 kcal/mol are not pinned, they are being measured.
       This is the test that distinguishes a floor from a coincidence, and a quantile cannot do it.
    3. the common value lies inside ``band``: the assay's reported floor occupies roughly -1 to 0.2 kcal/mol. A tight
       cluster of deeply destabilised variants reading +1.5 is not a floor, it is compensation, and conditioning on
       it would be worse than having no floor at all.

    Returns ``floor`` (NaN without evidence), ``n``, ``sd``, ``median``, and ``evidence`` (1.0/0.0). Shrinkage toward
    the dataset-wide floor is applied afterwards by :func:`shrink_floors`, which needs every library at once.
    """
    out = {'floor': float('nan'), 'n': 0.0, 'sd': float('nan'), 'median': float('nan'), 'evidence': 0.0}
    if not np.isfinite(dG_wt):
        return out
    deep = np.isfinite(dG) & np.isfinite(x) & (x < x_max)
    out['n'] = float(deep.sum())
    if out['n'] < min_n:
        return out
    v = dG[deep]
    out['sd'], out['median'] = float(np.std(v)), float(np.median(v))
    if out['sd'] > max_sd or not (band[0] <= out['median'] <= band[1]):
        return out
    out['floor'], out['evidence'] = out['median'], 1.0
    return out


def library_floor(df: Any, percentile: float = 5.0, dG_wt: float = float('nan'), name: str = '',
                  **evidence_kw) -> Dict[str, float]:
    """
    One library's floor statistics.

    ``floor`` / ``evidence`` come from :func:`floor_evidence` and are what training uses. ``floor_pctl``, the
    ``percentile`` percentile of the library's numeric dG, is computed and logged for comparison only: it is a
    quantile of the library's own sensitivity rather than of the readout, it is defined even for libraries with no
    floor population at all, and measured across this dataset it is clipped at the top of the assay's band for 44%
    of training libraries and correlates only 0.40 with the conditioned estimate. ``frac_cens_lo`` is the share of
    rows the assay would not give a number to (``<-1``), which truncates the numeric tail the percentile sees.

    Estimated from MEASURED rows only. With ``--censor_range`` the frame also carries the rows the assay merely
    bounded, whose ``dG_ML`` has been set to the bound itself (-1.0); feeding those to the estimator would hand
    it a pile of identical values sitting exactly at the band edge, which is the shape it reads as evidence of a
    floor, and would manufacture a spurious floor at -1.0 for any library with enough of them.
    """
    out = {'y_floor': float('nan'), 'evidence': 0.0, 'floor_pctl': float('nan'), 'floor_cond': float('nan'),
           'n_dG': 0.0, 'n_deep': 0.0, 'sd_deep': float('nan'), 'frac_cens_lo': float('nan'),
           'dG_wt': float(dG_wt), 'percentile': float(percentile)}
    if 'dG_ML' not in df.columns:
        logging.warning(f"[{name}] no dG_ML column; no floor estimate and no floor censoring.")
        return out

    # The share the assay declined to fit downward: from the admitted rows' own censoring where the frame carries
    # it, else from the raw tag that preprocessing has not yet replaced.
    if 'cens' in df.columns:
        cens_col = np.asarray(df['cens'], dtype=float)
        out['frac_cens_lo'] = float((cens_col < 0).mean()) if len(cens_col) else float('nan')
        df = df.loc[df['cens'] == 0]
    else:
        tags_all = np.array([str(x).strip() for x in df['dG_ML'].to_numpy(dtype=object)])
        out['frac_cens_lo'] = float((tags_all == censoring.LOW_TAG).mean()) if len(tags_all) else float('nan')
    if not len(df):
        logging.warning(f"[{name}] no measured rows; no floor estimate and no floor censoring.")
        return out

    dG = censoring.to_numeric(df['dG_ML'].to_numpy(dtype=object))
    finite = np.isfinite(dG)
    out['n_dG'] = float(finite.sum())
    if finite.any():
        out['floor_pctl'] = float(np.percentile(dG[finite], percentile))

    # The additive expectation of every double with both singles measured: x = dG_wt + ddG_A + ddG_B.
    x = np.full(len(dG), np.nan)
    if np.isfinite(dG_wt) and 'mut_type' in df.columns:
        muts = df['mut_type'].astype(str).to_numpy()
        single = {m: d for m, d in zip(muts, dG) if ':' not in m and np.isfinite(d)}
        for i, m in enumerate(muts):
            if m.count(':') != 1:
                continue
            a, b = m.split(':')
            if a in single and b in single:
                x[i] = single[a] + single[b] - dG_wt

    ev = floor_evidence(dG, x, dG_wt, **evidence_kw)
    out.update(y_floor=ev['floor'], evidence=ev['evidence'], floor_cond=ev['median'],
               n_deep=ev['n'], sd_deep=ev['sd'])
    verdict = 'floor accepted' if ev['evidence'] else 'NO floor evidence -> no censoring'
    logging.info(f"[{name}] {verdict}: floor={ev['floor']:.3f} from n={ev['n']:.0f} deep variants "
                 f"(sd={ev['sd']:.3f}, median={ev['median']:.3f}); {percentile:g}th-pctl comparison="
                 f"{out['floor_pctl']:.3f}; dG_wt={dG_wt:.2f}; {out['frac_cens_lo']*100:.1f}% reported '<-1'")
    return out


def shrink_floors(stats: Dict[str, Dict[str, float]], shrink: float = 5.0) -> float:
    """
    Shrink every accepted floor toward the dataset-wide median of the accepted floors, in place, and return that
    median (the fallback floor).

    A library's floor is measured with a standard error around 0.05 against a between-library spread near 0.3, so a
    well-populated library is essentially untouched by ``shrink`` pseudo-counts of the global value while a library
    that only just cleared ``min_n`` is pulled most of the way back. Libraries with no evidence are left with
    ``y_floor`` NaN and ``evidence = 0``: nothing in them is censored, and the link covers their knee with its
    learned ``global_floor`` (``esm_msr.link.FloorLink``) rather than borrowing the global median as if it were
    data. The returned median is what initialises that parameter.
    """
    accepted = [s['y_floor'] for s in stats.values() if s.get('evidence') and np.isfinite(s['y_floor'])]
    glob = float(np.median(accepted)) if accepted else float('nan')
    for s in stats.values():
        if s.get('evidence') and np.isfinite(s['y_floor']) and np.isfinite(glob):
            n = max(s.get('n_deep', 0.0), 0.0)
            s['y_floor'] = float((n * s['y_floor'] + shrink * glob) / (n + shrink))
        else:
            # Left NaN on purpose: the link's learned global_floor covers these, and keeping them out of the data
            # means a library with a measured floor never has its knee moved by one without.
            s['y_floor'] = float('nan')
    logging.info(f"floor shrinkage: {len(accepted)} of {len(stats)} libraries have floor evidence; "
                 f"dataset floor (median of accepted) = {glob:.3f}; the rest get no floor and are neither "
                 f"censored nor given a knee from the data - the link learns one for them")
    return glob


class MutationStabilityDataset(torch.utils.data.Dataset):
    """
    One protein library (one ``code``) of stability measurements, expanded into training items and cached to disk.

    Generation walks the library's rows once and emits, per row:

    * ``single`` - a measured single mutation on its own structure (WT head).
    * ``double`` - a measured double mutant (predicted by the head ensemble).
    * ``cond``   - the conditional effect of one mutation of a double given the other, derived as ddG_AB - ddG_B,
      scored by the MT head. One per ordered pair, so a double with both singles measured yields two.

    ``native_cond`` is assigned after loading by :meth:`_label_native_cond`, for libraries whose code carries a
    mutation suffix, i.e. whose measurements were all made in a mutant background. See ``esm_msr.routing`` for what
    each subset means and which head owns it.

    Every item also carries the raw measurement row it came from (``meas``, see ``MEAS_COLUMNS``), the absolute dG
    that measurement describes (``dG_meas``), its library's starting dG (``dG_wt``) and its library's floor estimate
    (``y_floor``), which together let the loss put a prediction on the assay's observed scale and recognise a
    measurement that is only a bound (``esm_msr.censoring``).

    Positions are 1-based with respect to the structure's sequence, which equals the index into the tokenized
    sequence because the tokenizer prepends BOS.
    """

    # Bump when the emitted items change shape or meaning, so a stale cache is never served.
    # c3: first conservative-branch cache. Reversions are gone; the former 'mut_ctx' subset is emitted as 'cond'
    # (same construction, the name ``routing`` uses); every item carries meas / dG_meas / dG_wt / y_floor /
    # cens / cens_src / dG_bound, and the library's floor statistics ride along in `floor_stats`.
    CACHE_VERSION = 'c4'

    # The bounded ML fit reports no dG outside this interval: the measured distribution has zero mass beyond
    # either end, for singles and doubles alike.
    DG_FLOOR, DG_CEILING = -1.0, 5.0

    def __init__(
        self,
        dms_df: Any,
        tokenizer: Any,
        dms_name: str,
        mut_structs_root: str,
        score_name: str = 'ddG_ML',
        path: Optional[str] = None,
        generate: bool = False,
        incl_destab_bb: bool = True,
        *,
        structure_encoder: Optional[Any] = None,
        incl_singles: bool = True,
        incl_doubles: bool = True,
        incl_cond: bool = True,
        incl_native_cond: bool = True,
        dG_wt: Optional[float] = None,
        floor_stats: Optional[Dict[str, float]] = None,
        censor_floor: bool = True,
        gap_max: Optional[float] = 2.0,
        native_cond_as_single: bool = False,
        censor_range: bool = True,
    ):
        """
        Args:
            dms_df: DataFrame of measurements for this library.
            tokenizer: ESM3 sequence tokenizer.
            dms_name: The library's ``code`` (e.g. '1A32', or '1A0N_L7S' for a mutant-background library).
            mut_structs_root: Root of the modeled mutant structures tree
                (``<root>/<code>/pdb_models/<chain>[<mut>].pdb``).
            score_name: Target column; 'ddG_ML' for MegaScale, 'ddG' for benchmarks.
            path: Cache directory.
            generate: Regenerate instead of reading the cache.
            incl_destab_bb: Include rows whose backbone is itself a mutant.
            structure_encoder: ESM3 structure encoder; required to emit structure tokens.
            incl_singles / incl_doubles / incl_cond / incl_native_cond: which subsets to keep after loading. All
                four are generated and cached regardless, so toggling them never needs a regenerate.
            dG_wt: Measured dG of this library's starting sequence (the wild type, or the background for a
                mutation-suffixed code). The preprocessor reads it from the 'wt' rows, which it filters out of the
                table before this point. Needed to put an item on the assay's observed scale and to compute the
                additive expectation the floor's doubles-based estimate conditions on.
            floor_stats: This library's floor statistics from ``preprocess_megascale.library_floors`` - ``y_floor``
                (the knee the link bends at) and ``evidence`` (whether the library actually shows a floor, and so
                whether anything in it may be censored against one). Computed from the raw table each run rather
                than baked into the cache, so the floor estimator is a free knob that never needs a regenerate.
            censor_floor: Mark items measured at or below their library's floor as lower-censored, so the losses
                treat their value as a bound and their mutual order as unknown (``esm_msr.censoring``).
            gap_max: Also censor an item whose two proteases disagree by more than this (kcal/mol), bounded by its
                own reported value. None disables it. This is the source that reaches libraries with no floor.
            censor_range: Keep the rows the assay bounded rather than fitted (``<-1`` / ``>5``), which the cache
                always stores. False drops them, restoring the selected-on-outcome population; see
                ``esm_msr.censoring``.
            native_cond_as_single: Leave a mutant-background library's measurements labelled ``single`` instead of
                ``native_cond``, which is what the released model does. They then train the WT head and count as
                singles in every metric. Strictly it is wrong - ddG(X | background) is a conditional effect, not a
                wild-type-context one - but it is the released behaviour and the comparison worth having.
        """
        self.score_name = score_name
        self.dms_name = dms_name
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.incl_destab_bb = incl_destab_bb
        self.mut_structs_root = mut_structs_root
        self.floor_stats = dict(floor_stats) if floor_stats else {}
        self.dG_wt = float(dG_wt) if dG_wt is not None and np.isfinite(dG_wt) else float('nan')

        self.include = {
            'single': incl_singles,
            'double': incl_doubles,
            'cond': incl_cond,
            'native_cond': incl_native_cond,
        }

        # Pre-cache the vocabulary mapping for rapid ID lookups
        self.vocab = self.tokenizer.get_vocab()

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        dms_df['ground_truth'] = dms_df['ddG']

        if path is None:
            path = '.'

        self.cache_path = os.path.join(
            path,
            f"{self.dms_name}_{self.score_name}_{self.CACHE_VERSION}.pkl"
        )

        logging.info(f"Dataset Cache Path: {self.cache_path}")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.data: List[Dict[str, Any]] = []
        self._encoded_struct_cache: Dict[Tuple, Tuple] = {}
        self._mutant_struct_cache: Dict[str, Tuple] = {}
        self._parsed_pdb_cache: Dict[str, Tuple] = {}

        if generate or not os.path.exists(self.cache_path):
            logging.info(f"Generating and caching data for {self.dms_name}")
            self.data = self.generate_data(dms_df)
            for item in self.data:
                item['dG_wt'] = self.dG_wt
            self._save_data_to_cache()
        else:
            logging.info(f"Loading cached data for {self.dms_name}")
            self.load_data_from_cache()

        logging.info(f"Cache contains: {dict(Counter(i.get('subset_type', 'unknown') for i in self.data))}")

        # Mutant-background libraries: their singles are conditional measurements, unless we are reproducing the
        # released behaviour of treating them as ordinary singles.
        if not censor_range:
            self._drop_range_censored()
        if not native_cond_as_single:
            self._label_native_cond()
        self._filter_dataset()
        self._stamp_floor()
        self._apply_censoring(censor_floor, gap_max)
        self._extract_scalars()

    def generate_data(self, dms_df: Any) -> List[Dict[str, Any]]:
        """
        Processes raw DMS dataframe into a list of structured data items.
        """
        if self.score_name == 'ddG_ML':
            df = dms_df.loc[dms_df['code'] == self.dms_name].copy()
            if len(df) == 0:
                raise AssertionError(f"No data found for code {self.dms_name}")
            df['mutated_sequence'] = df['aa_seq']
            data = self._load_data(df, is_predicted=True)
        else:
            data = []
            dms_df['code_wt'] = dms_df['code']
            for (code, chain), df_sub in dms_df.groupby(['code', 'chain']):
                df_sub = df_sub.copy()
                df_sub['mutated_sequence'] = df_sub['mut_seq']
                data.extend(self._load_data(df_sub, incl_chain_in_code=True, is_predicted=False, benchmark=True))
        return data
    
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
        allowed = {k for k, on in self.include.items() if on}
        before = len(self.data)
        self.data = [item for item in self.data if item.get('subset_type') in allowed]
        logging.info(f"Filtered dataset from {before} to {len(self.data)} items based on allowed types: {allowed}")

    def _stamp_floor(self) -> None:
        """
        Give every item its library's floor knee, whether that floor is backed by evidence, and its own
        two-protease disagreement.

        ``protease_gap`` is |deltaG_t - deltaG_c| from the cached measurement row: two independent fits of the same
        variant's stability, so a large gap says they do not jointly identify one (``esm_msr.censoring``). It is
        derived here rather than cached in its own right because ``meas`` already carries both channels.
        """
        self.has_floor_evidence = bool(self.floor_stats.get('evidence', 0.0))
        # The invariant, enforced here rather than trusted from upstream: an item carries a floor only if its
        # library actually showed one. Otherwise it carries NaN and the link's learned global_floor covers it. If
        # a finite floor leaked through for a library without evidence, it would silently pin the knee and make
        # --learn_global_floor inert for that library, which is hard to notice from the metrics.
        y = float(self.floor_stats.get('y_floor', float('nan'))) if self.has_floor_evidence else float('nan')
        self.y_floor_used = y        # what the items actually carry, which is what the log must report
        for item in self.data:
            item['y_floor'] = y
            t, c = meas_value(item, 'deltaG_t'), meas_value(item, 'deltaG_c')
            item['protease_gap'] = abs(t - c) if np.isfinite(t) and np.isfinite(c) else float('nan')

    def _apply_censoring(self, censor_floor: bool, gap_max: Optional[float]) -> None:
        """
        Finalise ``cens`` / ``cens_bound`` on every item (see ``esm_msr.censoring``).

        Censoring requires evidence that this library HAS a floor (``floor_evidence``). Without it the knee is only
        the dataset-wide fallback, and comparing measurements against it would discard good data from libraries that
        simply never destabilised anything.
        """
        active = bool(censor_floor and getattr(self, 'has_floor_evidence', False))
        self.data, c = censoring.apply_item_censoring(self.data, active, gap_max)
        if c['floor'] or c['gap'] or c['range']:
            logging.info(f"[{self.dms_name}] censored {c['floor'] + c['gap'] + c['range']} of {len(self.data)} items: "
                         f"{c['range']} bounded by the assay {dict(c['range_by_subset'])}, "
                         f"{c['floor']} at the library floor "
                         f"(y_floor={getattr(self, 'y_floor_used', float('nan')):.3f}, "
                         f"evidence={int(getattr(self, 'has_floor_evidence', False))}) {dict(c['floor_by_subset'])}, "
                         f"{c['gap']} on protease disagreement {dict(c['gap_by_subset'])}")
        else:
            logging.info(f"[{self.dms_name}] nothing censored"
                         f"{' (no floor evidence)' if censor_floor and not active else ''}")

    def _extract_scalars(self) -> None:
        """Contiguous scalar arrays for the sampler's and the metrics' bulk lookups."""
        self.ground_truth_arr = np.array([i.get('ddG', np.nan) for i in self.data], dtype=np.float32)
        self.cens_arr = np.array([i.get('cens', 0) for i in self.data], dtype=np.int8)

    # ------------------------------------------------------------------ cache / dataset API

    def _save_data_to_cache(self) -> None:
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'items': self.data, 'floor_stats': self.floor_stats},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_from_cache(self) -> None:
        with open(self.cache_path, 'rb') as f:
            blob = pickle.load(f)
        self.data = blob['items']
        self.floor_stats = blob.get('floor_stats', {})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def _try_load_modeled_context(
        self, 
        base_code: str, 
        chain: str, 
        partner_wt: str, 
        partner_pos: int, 
        partner_mut: str, 
        expected_seq: str, 
        target_masks: List[int], 
        force_mask: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Attempts to load, encode, and mask a modeled structure for a specific mutation context.
        """
        dir_lvl1 = f"{base_code}"
        dir_lvl2 = "pdb_models"
        fname = f"{chain}[{partner_wt}{partner_pos}{partner_mut}].pdb"
        pdb_path = os.path.join(self.mut_structs_root, dir_lvl1, dir_lvl2, fname)

        mask_tuple = tuple(sorted(set(target_masks)))
        effective_masking = force_mask
        cache_key = (pdb_path, mask_tuple, effective_masking)

        # 1. Check fully encoded cache
        if cache_key in self._mutant_struct_cache:
            cached_seq_tokens, cached_coords, cached_plddt, cached_struct_tokens, cached_residue_index = self._mutant_struct_cache[cache_key]
            cached_seq = ''.join(self.tokenizer.decode(cached_seq_tokens if isinstance(cached_seq_tokens, list) else cached_seq_tokens.tolist()).split(' ')[1:-1])
            if cached_seq == expected_seq: 
                return cached_seq_tokens, cached_coords, cached_plddt, cached_struct_tokens, cached_residue_index
            else:
                logging.error('Cached seq did not match expected, returning None', cached_seq, expected_seq)
                return None

        # 2. Check unmasked parsed cache
        if pdb_path in self._parsed_pdb_cache:
            seq_loaded, coords_unmasked, plddt_unmasked, residue_index_m = self._parsed_pdb_cache[pdb_path]
        else:
            if not os.path.exists(pdb_path): 
                return None
            logging.info(f"DEBUG: PDB I/O LOAD for {fname}")
            mut_chain = ProteinChain.from_pdb(pdb_path, chain, is_predicted=True)
            seq_loaded, coords_unmasked, plddt_unmasked, residue_index_m = mut_chain.sequence, *mut_chain.to_structure_encoder_inputs()
            self._parsed_pdb_cache[pdb_path] = (seq_loaded, coords_unmasked, plddt_unmasked, residue_index_m)

        if seq_loaded != expected_seq: 
            logging.error('Loaded seq did not match expected, returning None', seq_loaded, expected_seq)
            return None

        coords_m, plddt_m = coords_unmasked.clone(), plddt_unmasked.clone()

        try:
            if effective_masking and mask_tuple:
                for pos in mask_tuple:
                    idx = pos - 1
                    if 0 <= idx < coords_m.shape[0]:
                        coords_m[idx, :, :] = float('nan')
                        plddt_m[idx] = 0.0

            if self.structure_encoder:
                _, structure_tokens_m = self.structure_encoder.encode(coords_m, residue_index=residue_index_m)
                structure_tokens_m = F.pad(structure_tokens_m.squeeze(0), (1, 1), value=0)
                structure_tokens_m[0], structure_tokens_m[-1] = C.STRUCTURE_BOS_TOKEN, C.STRUCTURE_EOS_TOKEN
            else:
                structure_tokens_m = torch.Tensor([-1])
        except Exception as e:
            logging.error(f"Structure encoding failed for {pdb_path}: {e}")
            return None

        coords_m = F.pad(coords_m, (0, 0, 0, 0, 1, 1), value=torch.inf)
        plddt_m = F.pad(plddt_m, (1, 1), value=0)
        sequence_tokens_m = self.tokenizer.encode(seq_loaded)

        self._mutant_struct_cache[cache_key] = (sequence_tokens_m, coords_m, plddt_m, structure_tokens_m, residue_index_m)
        return sequence_tokens_m, coords_m, plddt_m, structure_tokens_m, residue_index_m

    def _get_encoded_structure(
        self, 
        protein_chain: ProteinChain, 
        pdb_identifier: str, 
        mask_positions_1idx: List[int], 
        force_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads and pads the structure tokens and coordinates, applying structural masking if requested.
        """
        mask_tuple = tuple(sorted(set(mask_positions_1idx)))
        effective_masking = force_mask
        cache_key = (pdb_identifier, mask_tuple, effective_masking)

        if cache_key in self._encoded_struct_cache:
            return self._encoded_struct_cache[cache_key]

        coords_unpadded, plddt_unpadded, residue_index_unpadded = protein_chain.to_structure_encoder_inputs()

        if effective_masking and mask_tuple:
            for pos in mask_tuple:
                idx = pos - 1
                if 0 <= idx < coords_unpadded.shape[0]:
                    coords_unpadded[idx, :, :] = float('nan') 
                    plddt_unpadded[idx] = 0.0

        if self.structure_encoder:
            _, struct_tokens_unpadded = self.structure_encoder.encode(
                coords_unpadded,
                residue_index=residue_index_unpadded
            )
            struct_tokens_unpadded = struct_tokens_unpadded.squeeze(0)
        else:
            raise AssertionError("Structure encoder is required.")

        struct_tokens_padded = F.pad(struct_tokens_unpadded, (1, 1), value=0)
        if self.structure_encoder:
            struct_tokens_padded[0] = C.STRUCTURE_BOS_TOKEN
            struct_tokens_padded[-1] = C.STRUCTURE_EOS_TOKEN

        coords_padded = F.pad(coords_unpadded, (0, 0, 0, 0, 1, 1), value=torch.inf)
        plddt_padded = F.pad(plddt_unpadded, (1, 1), value=0)

        result = (coords_padded, plddt_padded, struct_tokens_padded, residue_index_unpadded)
        self._encoded_struct_cache[cache_key] = result
        return result

    def _load_data(
        self, 
        df: Any, 
        is_predicted: bool = False, 
        incl_chain_in_code: bool = False, 
        benchmark: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Main logic for parsing DMS DataFrames and extracting mutations and wildtype data.
        """
        self._mutant_struct_cache.clear()
        self._encoded_struct_cache.clear() 
        self._parsed_pdb_cache.clear() 

        data: List[Dict[str, Any]] = []

        if 'mut_structure' not in df.columns:
            df['mut_structure'] = df['pdb_file']
        df['mut_structure'] = df['mut_structure'].fillna(df['pdb_file'])

        # === PRE-PASS: Collect WT Singles ===
        global_wt_single_map = {}
        global_destab_backbones = []
        wt_protein_chain = None
        wt_canonical_seq = None

        for backbone, group in df.groupby('mut_structure'):
            if backbone.endswith('.pdb'):
                chain = group['chain'].head(1).item()
                wt_protein_chain = ProteinChain.from_pdb(backbone, chain, is_predicted=is_predicted)
                wt_canonical_seq = ''.join(list(wt_protein_chain.sequence))
                
                for uid, row in group.iterrows():
                    if 'mut_type' not in df.columns:
                        mut_seq = row['mutated_sequence']
                        offset, _, _ = custom_end_gap_alignment(mut_seq, wt_canonical_seq)
                        muts = determine_diffs(mut_seq[offset:len(wt_canonical_seq)+offset], wt_canonical_seq)
                    else:
                        muts = [
                            (m[0], int(m[1:-1]), m[-1]) 
                            for m in row['mut_type'].split(':') 
                            if len(m) >= 3 and wt_canonical_seq[int(m[1:-1])-1] == m[0]
                        ]
                    
                    if len(muts) == 1:
                        global_wt_single_map[(muts[0][1], muts[0][2])] = float(row['ddG'])
                break
            else:
                global_destab_backbones.append(backbone)

        # === MAIN PASS ===
        for backbone, group in df.groupby('mut_structure'):
            base_code = group['code'].head(1).item()
            chain = group['chain'].head(1).item()
            code = base_code + chain if incl_chain_in_code else base_code

            protein_chain = ProteinChain.from_pdb(
                group['pdb_file'].head(1).item() if not backbone.endswith('.pdb') else backbone, 
                chain, 
                is_predicted=is_predicted
            )
            corrected_seq = list(protein_chain.sequence)

            base_mask_positions = []
            is_destabilized_backbone = not backbone.endswith('.pdb')
            has_modeled_backbone = False
            wt_bb, pos_bb, mt_bb = None, None, None
            
            if is_destabilized_backbone:
                if not self.incl_destab_bb:
                    continue
                wt_bb, pos_bb, mt_bb = backbone[0], int(backbone[1:-1]), backbone[-1]
                corrected_seq[pos_bb-1] = mt_bb
                
                # Check if the modeled backbone exists for the group
                probe = self._try_load_modeled_context(
                    base_code=base_code, chain=chain,
                    partner_wt=wt_bb, partner_pos=pos_bb, partner_mut=mt_bb,
                    expected_seq=''.join(corrected_seq),
                    target_masks=[]
                )
                if probe is not None:
                    has_modeled_backbone = True
                else:
                    base_mask_positions.append(pos_bb)

            corrected_seq = ''.join(corrected_seq)
            single_map, parsed_rows = {}, []
            warned = False

            for uid, row in tqdm(group.iterrows(), leave=False):
                if 'mut_type' not in df.columns:
                    if not warned:
                        logging.warning('Inferring mutations from mutated_sequence column because mut_type column was missing')
                        warned = True
                    mut_seq = row['mutated_sequence']
                    offset, _, _ = custom_end_gap_alignment(mut_seq, protein_chain.sequence)
                    muts = determine_diffs(mut_seq[offset:len(corrected_seq)+offset], corrected_seq)
                else:
                    muts = [
                        (m[0], int(m[1:-1]), m[-1]) 
                        for m in row['mut_type'].split(':') 
                        if len(m) >= 3 and corrected_seq[int(m[1:-1])-1] == m[0]
                    ]

                ddG_val = float(row['ddG'])
                parsed_rows.append((muts, ddG_val, row))
                # Only a MEASURED single defines an additive expectation. A range-censored single's label is its
                # bound, so admitting it here would turn ddG_additive, dddG and the teacher-forced WT target into
                # bounds masquerading as values. Its partners simply get no additive expectation.
                if len(muts) == 1 and int(row.get('cens', 0) or 0) == 0:
                    single_map[(muts[0][1], muts[0][2])] = ddG_val

            for muts, ddG_val, row in parsed_rows:
                dddG_val = (
                    ddG_val - single_map[(muts[0][1], muts[0][2])] - single_map[(muts[1][1], muts[1][2])] 
                    if len(muts) == 2 and (muts[0][1], muts[0][2]) in single_map and (muts[1][1], muts[1][2]) in single_map 
                    else np.nan
                )

                # (A) Singles
                if len(muts) == 1:
                    wt, pos, mt = muts[0]
                    
                    target_masks = base_mask_positions.copy()

                    maybe_modeled = None
                    if has_modeled_backbone:
                        maybe_modeled = self._try_load_modeled_context(
                            base_code, chain, wt_bb, pos_bb, mt_bb, expected_seq=corrected_seq, target_masks=target_masks, force_mask=False
                        )
                        
                    if maybe_modeled is not None:
                        struct_type = 'model'
                        _, c_b, p_b, s_b, r_b = maybe_modeled
                    else:
                        struct_type = 'fake' if len(base_mask_positions) > 0 else 'af'
                        c_b, p_b, s_b, r_b = self._get_encoded_structure(
                            protein_chain, backbone, target_masks, force_mask=len(base_mask_positions) > 0
                        )

                    mt_seq_list = list(corrected_seq)
                    mt_seq_list[pos-1] = mt
                    mt_seq = ''.join(mt_seq_list)
                    wt_seq = corrected_seq

                    data.append(self._create_data_item(
                        mutations=muts, ddG=ddG_val, dddG=dddG_val, code=code, wt_seq=wt_seq, mt_seq=mt_seq, 
                        coords=c_b, plddt=p_b, structure_tokens=s_b, residue_index=r_b, subset_type='single', 
                        ddG_A=np.nan, ddG_B=np.nan, ddG_additive=np.nan, structure_type=struct_type,
                        row=row, ddG_total=ddG_val
                    ))

                # (C) Doubles
                if len(muts) == 2:
                    (wtA, posA, mtA), (wtB, posB, mtB) = muts
                    
                    target_masks = base_mask_positions.copy()
                            
                    maybe_modeled = None
                    if has_modeled_backbone:
                        maybe_modeled = self._try_load_modeled_context(
                            base_code, chain, wt_bb, pos_bb, mt_bb, expected_seq=corrected_seq, target_masks=target_masks, force_mask=False
                        )
                        
                    if maybe_modeled is not None:
                        struct_type = 'model'
                        _, c_b, p_b, s_b, r_b = maybe_modeled
                    else:
                        struct_type = 'fake' if len(base_mask_positions) > 0 else 'af'
                        c_b, p_b, s_b, r_b = self._get_encoded_structure(
                            protein_chain, backbone, target_masks, force_mask=len(base_mask_positions) > 0
                        )

                    if (posA, mtA) in single_map and (posB, mtB) in single_map:
                        ddG_A = single_map[(posA, mtA)]
                        ddG_B = single_map[(posB, mtB)]
                        ddG_additive = ddG_A + ddG_B
                    else:
                        ddG_A, ddG_B, ddG_additive = np.nan, np.nan, np.nan

                    wt_seq = corrected_seq
                    mt_seq_list = list(corrected_seq)
                    mt_seq_list[posA-1] = mtA
                    mt_seq_list[posB-1] = mtB
                    mt_seq = ''.join(mt_seq_list)

                    data.append(self._create_data_item(
                        mutations=muts, ddG=ddG_val, dddG=dddG_val, 
                        ddG_additive=ddG_additive, ddG_A=ddG_A, ddG_B=ddG_B,
                        code=code, wt_seq=wt_seq, mt_seq=mt_seq,
                        coords=c_b, plddt=p_b, structure_tokens=s_b, residue_index=r_b, 
                        subset_type='double', structure_type=struct_type,
                        row=row, ddG_total=ddG_val
                    ))

                # (D) Conditional items: ddG(B|A) and ddG(A|B), derived from the double and its singles.
                #
                # Each conditions on the structure its parent double already used - the unchanged WT backbone -
                # rather than on a modeled or masked structure of the partner mutation alone. The partner's side
                # chain is therefore shown as wild type, which is wrong, but it is also exactly what the MT adapter
                # is given at inference time, so training and inference see the same input. It also costs nothing:
                # loading and re-encoding a modeled partner structure per item ran at about 0.4 s each, which is
                # ~50 hours over the full dataset, for a subset that is not currently trained on at all.
                if len(muts) == 2:
                    (wtA, posA, mtA), (wtB, posB, mtB) = muts
                    if (posA, mtA) in single_map and (posB, mtB) in single_map:
                        ddG_A = single_map[(posA, mtA)]
                        ddG_B = single_map[(posB, mtB)]
                        ddG_additive = ddG_A + ddG_B

                        for (scored, partner, target) in (
                            ((wtB, posB, mtB), (wtA, posA, mtA), ddG_val - ddG_A),   # ddG(B | A)
                            ((wtA, posA, mtA), (wtB, posB, mtB), ddG_val - ddG_B),   # ddG(A | B)
                        ):
                            _, p_pos, p_mt = partner
                            _, s_pos, s_mt = scored
                            ctx_wt_list = list(corrected_seq)
                            ctx_wt_list[p_pos - 1] = p_mt                 # the background: the partner present
                            ctx_mt_list = list(ctx_wt_list)
                            ctx_mt_list[s_pos - 1] = s_mt                 # plus the scored mutation

                            data.append(self._create_data_item(
                                mutations=[scored], ddG=float(target), dddG=dddG_val,
                                ddG_additive=ddG_additive, ddG_A=ddG_A, ddG_B=ddG_B, code=code,
                                wt_seq=''.join(ctx_wt_list), mt_seq=''.join(ctx_mt_list),
                                coords=c_b, plddt=p_b, structure_tokens=s_b, residue_index=r_b,
                                subset_type='cond', structure_type=struct_type,
                                row=row, ddG_total=ddG_val
                            ))
        return data
    
    def _meas_array(self, row: Any) -> np.ndarray:
        """The raw measurement columns of one row as a float32 array in ``MEAS_COLUMNS`` order; NaN where a column is
        absent (benchmarks) or non-numeric (the ``<-1`` / ``>5`` / ``-`` tags)."""
        out = np.full(len(MEAS_COLUMNS), np.nan, dtype=np.float32)
        if row is None:
            return out
        for i, name in enumerate(MEAS_COLUMNS):
            if name in row.index:
                out[i] = censoring.to_numeric([row[name]])[0]
        return out

    def _dG_meas(self, row: Any, ddG_total: float) -> float:
        """
        The measured absolute dG of the state this item's label describes.

        For a single or a double it is that row's own ``dG_ML``. For a ``cond`` item it is its parent double's, which
        is the same row: ddG(B|A) is a statement about the AB state, and it is the AB state that the assay pinned at
        the floor. Falls back to ``dG_wt + ddG_total`` where the table has no ``dG_ML`` (the benchmarks), and is NaN
        when neither is available - which simply means the item can carry no floor censoring.
        """
        if row is not None and 'dG_ML' in row.index:
            dG = censoring.to_numeric([row['dG_ML']])[0]
            if np.isfinite(dG):
                return float(dG)
        if np.isfinite(self.dG_wt) and np.isfinite(ddG_total):
            return float(self.dG_wt + ddG_total)
        return float('nan')

    def _create_data_item(
        self,
        mutations: List[Tuple[str, int, str]],
        ddG: float,
        dddG: float,
        ddG_additive: float,
        ddG_A: float,
        ddG_B: float,
        code: str,
        wt_seq: str,
        mt_seq: str,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        structure_tokens: torch.Tensor,
        residue_index: torch.Tensor,
        subset_type: str,
        structure_type: str,
        row: Any = None,
        ddG_total: float = float('nan'),
    ) -> Dict[str, Any]:
        """
        Constructs the fully vectorized dictionary item for the Collate function.

        ``row`` is the raw measurement row the label came from (carried through as ``meas``); ``ddG_total`` is the
        total ddG of the state that row describes, which for a ``cond`` item is the parent double's ddG_AB rather
        than the item's own conditional label.
        """
        valid_dddG = dddG == dddG and not np.isnan(dddG)

        wt_sequence_tokens = self.tokenizer.encode(wt_seq)
        mt_sequence_tokens = self.tokenizer.encode(mt_seq)

        mut_pos = []
        wt_ids = []
        mt_ids = []

        for (wt_aa, pos, mt_aa) in mutations:
            mut_pos.append(pos) 
            
            w_id = self.vocab.get(wt_aa)
            m_id = self.vocab.get(mt_aa)
            
            if w_id is None or m_id is None:
                raise AssertionError(f"Unknown amino acid token detected: WT={wt_aa}, MT={mt_aa}")
                
            wt_ids.append(w_id)
            mt_ids.append(m_id)

        logging.debug(f'Created data item: {code}, {mutations}, {subset_type}, {structure_type}, ddG={ddG}, dddG={dddG}')
        return {
            'pdb': code,
            'mutations': mutations,
            'meas': self._meas_array(row),
            'dG_meas': self._dG_meas(row, ddG_total),
            # A row the assay bounded rather than fitted arrives already censored (``SRC_RANGE``); everything else
            # starts clean and may be censored later by its library's floor or its protease gap.
            'cens': int(row.get('cens', 0)) if hasattr(row, 'get') else 0,
            'cens_src': int(row.get('cens_src', 0)) if hasattr(row, 'get') else 0,
            'dG_bound': float(row.get('dG_bound', float('nan'))) if hasattr(row, 'get') else float('nan'),
            'wt_sequence_tokens': np.array(wt_sequence_tokens, dtype=np.int64),
            'mt_sequence_tokens': np.array(mt_sequence_tokens, dtype=np.int64),
            'mut_pos': np.array(mut_pos, dtype=np.int64),
            'wt_id': np.array(wt_ids, dtype=np.int64),
            'mt_id': np.array(mt_ids, dtype=np.int64),
            'coords_orig': coords.clone().cpu().numpy(),
            'structure_tokens_orig': structure_tokens.clone().cpu().numpy(),
            'residue_index': residue_index.clone().cpu().numpy() if residue_index is not None else None,
            'plddt': plddt.clone().cpu().numpy(),
            'ddG': float(ddG),
            'dddG': float(dddG) if valid_dddG else np.nan,
            'ddG_additive': float(ddG_additive) if ddG_additive == ddG_additive else np.nan,
            'ddG_A': float(ddG_A) if ddG_A == ddG_A else np.nan,
            'ddG_B': float(ddG_B) if ddG_B == ddG_B else np.nan,
            'valid_dddG_mask': valid_dddG,
            'subset_type': subset_type,
            'structure_type': structure_type
        }

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
    Consolidated, high-performance BatchSampler for balancing and cycling through 
    multiple protein datasets. Yields grouped indices for a ConcatDataset.
    
    Subset categorizations are cached during __init__ so no epoch pays an O(N) stall, and capping works
    strictly on integer indices, which keeps PyTorch's asynchronous worker prefetching intact.

    Capping is per-subset and proportional: a subset mapped to None is unrestricted and sets the baseline count,
    and a subset mapped to a fraction is downsampled to that fraction of the baseline (0.0 excludes it entirely).
    There is deliberately no density-based subsampling: an earlier version binned the doubles two-dimensionally
    over (additive expectation, dddG) and capped each bin, which reshapes the ddG distribution the model is
    trained on - it preferentially discards the common, well-measured, in-range doubles and keeps the sparse bins,
    which are mostly the saturated ones the floor link now models explicitly.
    """
    SUBSET_ORDER = ['single', 'double', 'cond', 'native_cond']

    def __init__(
        self,
        datasets: List[Any],
        batch_size: int,
        train_list: List[str],
        strategy: str = 'all',
        *,
        subset_caps: Optional[Dict[str, Optional[float]]] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        if strategy not in ('min', 'all'):
            raise ValueError(f"strategy must be 'min' or 'all', got '{strategy}'")
            
        self.datasets = datasets
        self.batch_size = batch_size
        self.train_list = train_list
        self.strategy = strategy
        self.verbose = verbose
        self._rng = random.Random(rng_seed)
        
        self.subset_caps: Dict[str, Optional[float]] = {
            'single': None, 'double': None, 'cond': 0.0, 'native_cond': 0.0
        }
        if subset_caps is not None:
            self.subset_caps.update(subset_caps)

        logging.info(f"Using subset cap config: {self.subset_caps}")

        self.unrestricted_keys = [k for k in self.SUBSET_ORDER if self.subset_caps.get(k) is None]
        if not self.unrestricted_keys:
            raise AssertionError(
                "Invalid subset_caps: At least one category (e.g., 'single') must be unrestricted "
                "(mapped to None). Otherwise, the baseline dataset size drops to 0."
            )

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
                # Avoid loading the full tensor dict if possible; assuming fast dictionary lookup
                item = ds[i]
                if item.get('pdb') != first_pdb:
                    raise AssertionError(f"PDB ID mismatch in {self.train_list[ds_idx]}: item {i} has {item.get('pdb')}, expected {first_pdb}")
                
                stype = item.get('subset_type', 'single')
                if stype not in buckets:
                    stype = 'single'
                buckets[stype].append(i)
                
            self._cached_buckets.append(buckets)

        # 3. Perform a dry-run to calculate exact batch sizes for the DataLoader __len__
        self.num_batches = self._calculate_epoch_batches(dry_run=True)

    def _calculate_epoch_batches(self, dry_run: bool = False) -> int:
        """Core logic to balance, shuffle, and chunk datasets."""
        self.all_batches = [] # List of dataset batch lists
        
        for idx, ds in enumerate(self.datasets):
            protein_name = self.train_list[idx]
            buckets = {k: list(v) for k, v in self._cached_buckets[idx].items()} # copy
            
            # Apply subset capping based on unrestricted size
            total_unrestricted = sum(len(buckets[k]) for k in self.unrestricted_keys if k in buckets)
            if total_unrestricted == 0:
                raise AssertionError(f"Total unrestricted samples for {protein_name} dropped to 0.")
            
            for k, items in buckets.items():
                if k not in self.unrestricted_keys:
                    cap_fraction = self.subset_caps.get(k)
                    if cap_fraction is not None and cap_fraction > 0:
                        max_allowed = int(math.ceil(total_unrestricted * cap_fraction))
                        if len(items) > max_allowed:
                            self._rng.shuffle(items)
                            buckets[k] = items[:max_allowed]
                    elif cap_fraction == 0.0 or cap_fraction == 0:
                        buckets[k] = []

            # Flatten and global offset map for ConcatDataset
            flat_indices = []
            for k, items in buckets.items():
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

        # Apply cycling strategy to interleave batches
        final_batches = []
        if self.strategy == 'min':
            min_len = min(len(b) for b in self.all_batches)
            for i in range(min_len):
                for ds_batch_list in self.all_batches:
                    final_batches.append(ds_batch_list[i])
        else: # 'all'
            for ds_batch_list in self.all_batches:
                final_batches.extend(ds_batch_list)
            self._rng.shuffle(final_batches)
            
        if not dry_run:
            self._active_batches = final_batches
            
        return len(final_batches)

    def __iter__(self) -> Iterator[List[int]]:
        """Invoked by PyTorch workers at the start of every epoch."""
        self._calculate_epoch_batches(dry_run=False)
        yield from self._active_batches

    def __len__(self) -> int:
        return self.num_batches


def create_consolidated_dataloader(
    dataloaders: List[DataLoader], 
    train_list: List[str], 
    batch_size: int, 
    collate_fn: Callable, 
    strategy: str = 'all', 
    subset_caps: Optional[Dict[str, Optional[float]]] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """A DataLoader over every library's dataset at once, batched by ProteinCyclingBatchSampler so each batch
    holds one library (the homogeneous-microbatch assumption the model's collate relies on)."""
    datasets = [dl.dataset for dl in dataloaders]
    
    sampler = ProteinCyclingBatchSampler(
        datasets=datasets,
        batch_size=batch_size,
        train_list=train_list,
        strategy=strategy,
        subset_caps=subset_caps,
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
        strategy: str = "all",
        *,
        seq_pad_token_id: int = C.SEQUENCE_PAD_TOKEN,
        structure_pad_token_id: int = C.STRUCTURE_PAD_TOKEN,
        coord_pad_value: float = float("inf"),
        debug_first_batches: int = 0,
        legacy_mode: bool = False
    ):
        self.dataloaders = dataloaders
        self.batch_size = int(batch_size)
        self.train_list = train_list if train_list else [f"dataset_{i}" for i in range(len(dataloaders))]
        self.num_dataloaders = len(dataloaders)
        self.strategy = strategy

        # IDs/values
        self.seq_pad_token_id = int(seq_pad_token_id)
        self.structure_pad_token_id = int(structure_pad_token_id)
        self.coord_pad_value = float(coord_pad_value)

        self.debug_first_batches = int(debug_first_batches)
        self.legacy_mode = legacy_mode

        # storage
        self.dataset_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pooled_data: List[Dict[str, Any]] = []
        self.all_batches: List[List[Dict[str, Any]]] = []
        self.current_batch_idx = 0
        self._batch_count = 0
        self.rng = random.Random()

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

        self._balance_datasets()

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

    def _balance_datasets(self):
        """Equalizes dataset representation if strategy='min' is passed."""
        if not self.strategy or self.strategy == "all":
            logging.info("No balancing strategy selected - using all available data")
            return

        dataset_sizes = {name: len(samples) for name, samples in self.dataset_samples.items()}
        logging.info("Dataset sizes before balancing:")
        for name, size in dataset_sizes.items():
            logging.info(f"  {name}: {size} samples")

        if self.strategy == "min" and dataset_sizes:
            min_size = min(dataset_sizes.values())
            logging.info(f"Balancing datasets by subsampling to {min_size} samples each")
            for name, samples in list(self.dataset_samples.items()):
                if len(samples) > min_size:
                    self.dataset_samples[name] = self.rng.sample(samples, min_size)

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

        if not self.legacy_mode:
            batch = self._collate_with_padding(batch_items)
        else:
            batch = self._collate_with_padding_legacy(batch_items)

        self.current_batch_idx += 1
        return batch

    def shuffle_all(self):
        self.rng.shuffle(self.all_batches)
        gc.collect()
        logging.info("Shuffled all homogeneous batches.")

    def reset_epoch(self):
        self.shuffle_all()

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

        if self._batch_count < self.debug_first_batches:
            logging.debug(f"\n[DEBUG] Batch {self._batch_count} diagnostics:")
            self._batch_count += 1
            i0 = 0
            logging.debug(f"  Lmax: {Lmax} | lengths[0]: {lengths[i0]}")
            logging.debug(f"  wt_sequence_tokens[0].shape: {tuple(wt_sequence_tokens[i0].shape)}")
            logging.debug(f"  structure_tokens[0].shape: {tuple(structure_tokens[i0].shape)}")
            logging.debug(f"  coords.shape: {tuple(coords.shape)}")

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

    def _collate_with_padding_legacy(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate variable-length proteins by right-padding the residue axis (L) only.
        Emits *_orig tensors and indexing helpers expected by your training code.
        """

        prefer_orig_fields = True  
        mutations_are_1_based = False  

        def _positions_from_mutations(muts: List[Tuple[str, int, str]], L_res: int) -> Tuple[int, ...]:
            """Compute 0-based residue columns from a mutation list."""
            pos_cols = []
            for (_, pos, _) in (muts or []):
                p = int(pos)
                if mutations_are_1_based:
                    p = p - 1
                if not (0 <= p < L_res):
                    raise AssertionError(f"Mutation index {pos} -> {p} out of bounds for length {L_res}")
                pos_cols.append(p)
            return tuple(sorted(set(pos_cols)))

        if not batch:
            return {}

        B = len(batch)

        def get_seq_arr(item):
            if prefer_orig_fields and "sequence_tokens_orig" in item:
                return np.asarray(item["sequence_tokens_orig"])
            return np.asarray(item["sequence_tokens"])

        def get_str_arr(item):
            key = "structure_tokens_orig" if (prefer_orig_fields and "structure_tokens_orig" in item) else "structure_tokens"
            return np.asarray(item[key])

        def get_crd_arr(item):
            key = "coords_orig" if (prefer_orig_fields and "coords_orig" in item) else "coords"
            return np.asarray(item[key])

        # --- Determine per-sample residue lengths (L_i) safely ---
        lengths: List[int] = []
        seq_list_1d: List[torch.Tensor] = []
        for it in batch:
            s = self._to_tensor_long(get_seq_arr(it))
            if s.ndim != 1:
                raise AssertionError(f"sequence must be [L]; got shape {tuple(s.shape)}")
            seq_list_1d.append(s)
            lengths.append(int(s.size(0)))
        Lmax = max(lengths)

        # --- Sequences: pad on residue axis to [B, Lmax] ---
        seq_pad = []
        for s in seq_list_1d:
            seq_pad.append(self._right_pad_last_dim(s, Lmax, self.seq_pad_token_id))
        sequence_tokens_orig = torch.stack(seq_pad, dim=0)  # [B, Lmax] long

        # --- Structure tokens ---
        struct_pad = []
        for it in batch:
            st = self._ensure_2d_structure(get_str_arr(it))  # [K, L]
            K, L = st.shape
            if L != len(np.asarray(get_seq_arr(it))):
                raise AssertionError("Structure/sequence residue length mismatch before pad.")
            st_p = self._right_pad_last_dim(st, Lmax, self.structure_pad_token_id)  # [K, Lmax]
            struct_pad.append(st_p)
        structure_tokens_orig = torch.stack(struct_pad, dim=0)  # [B, K, Lmax] long

        # --- pLDDT ---
        plddt_pad = []
        for it in batch:
            pt = it['plddt']  # [K, L]
            pt = self._ensure_2d_structure(pt)  # [K, L]
            K, L = pt.shape
            if L != len(np.asarray(get_seq_arr(it))):
                raise AssertionError("Structure/sequence residue length mismatch before pad.")
            pt_p = self._right_pad_last_dim(pt, Lmax, 0)  # [K, Lmax]
            plddt_pad.append(pt_p)
        plddt = torch.stack(plddt_pad, dim=0)  # [B, K, Lmax] long

        # --- Coordinates ---
        coords_list = []
        for it in batch:
            c = self._ensure_coords_residue_axis(get_crd_arr(it))  # 3D or 4D
            if c.ndim == 3:
                # [L, A1, A2] -> add batch dimension later
                if c.size(0) != len(np.asarray(get_seq_arr(it))):
                    raise AssertionError("Coords/sequence residue length mismatch before pad.")
                c_p = self._right_pad_last_dim(c, Lmax, self.coord_pad_value)  # [Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))  # [1, Lmax, A1, A2]
            else:
                # [Kc, L, A1, A2] -> pad along L (dim=1)
                if c.size(1) != len(np.asarray(get_seq_arr(it))):
                    raise AssertionError("Coords/sequence residue length mismatch before pad.")
                c_perm = c.permute(0, 2, 3, 1)            # [Kc, A1, A2, L]
                c_p = self._right_pad_last_dim(c_perm, Lmax, self.coord_pad_value)  # [Kc, A1, A2, Lmax]
                c_p = c_p.permute(0, 3, 1, 2)             # [Kc, Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))       # [1, Kc, Lmax, A1, A2]
                
        # Stack; result is either [B, Lmax, ...] or [B, Kc, Lmax, ...] depending on inputs
        coords_orig = torch.cat(coords_list, dim=0)

        # --- Basic metadata & labels ---
        pdb = [it.get("pdb", f"unk_{i}") for i, it in enumerate(batch)]
        st = [it.get("subset_type", None) for i, it in enumerate(batch)]

        ddG_list = []
        dddG_list = []
        for it in batch:
            if "ddG" in it:
                ddG_list.append(float(it["ddG"]))
            elif "ground_truth" in it:
                ddG_list.append(float(it["ground_truth"]))
            else:
                ddG_list.append(float("nan"))

            if "dddG" in it:
                dddG_list.append(float(it["dddG"]))
            else:
                dddG_list.append(float("nan"))

        ddG = torch.as_tensor(ddG_list, dtype=torch.float)
        dddG = torch.as_tensor(dddG_list, dtype=torch.float)

        mutations = [it.get("mutations", []) for it in batch]
        positions = [_positions_from_mutations(m, L_res=lengths[i]) for i, m in enumerate(mutations)]

        attention_mask = torch.zeros((B, Lmax), dtype=torch.long)
        position_ids = torch.zeros((B, Lmax), dtype=torch.long)
        for i, Li in enumerate(lengths):
            attention_mask[i, :Li] = 1
            position_ids[i, :Li] = torch.arange(Li, dtype=torch.long)

        # --- Final shape assertions ---
        if structure_tokens_orig.size(-1) != sequence_tokens_orig.size(1):
            raise AssertionError("Structure/sequence L mismatch after pad.")
            
        if coords_orig.ndim == 4:      
            if coords_orig.size(-3) != sequence_tokens_orig.size(1):
                raise AssertionError("Coords/sequence L mismatch after pad.")
        elif coords_orig.ndim == 5:    
            if coords_orig.size(-3) != sequence_tokens_orig.size(1):
                raise AssertionError("Coords/sequence L mismatch after pad.")
        else:
            raise AssertionError("coords_orig must be 3D or 4D after collation.")

        if self._batch_count < self.debug_first_batches:
            logging.debug(f"\n[DEBUG] Batch {self._batch_count} diagnostics:")
            self._batch_count += 1
            i0 = 0
            logging.debug(f"  Lmax: {Lmax} | lengths[0]: {lengths[i0]}")
            logging.debug(f"  sequence_tokens_orig[0].shape: {tuple(sequence_tokens_orig[i0].shape)}")
            logging.debug(f"  structure_tokens_orig[0].shape: {tuple(structure_tokens_orig[i0].shape)}")
            logging.debug(f"  coords_orig.shape: {tuple(coords_orig.shape)}")
            logging.debug(f"  positions[0]: {positions[i0] if positions else '[]'}")

        collated = {
            "pdb": pdb,
            "mutations": mutations,                        
            "positions": positions,                        
            "sequence_tokens_orig": sequence_tokens_orig,  
            "structure_tokens_orig": structure_tokens_orig,
            "coords_orig": coords_orig,                    
            "attention_mask": attention_mask,              
            "position_ids": position_ids,                  
            "lengths": torch.as_tensor(lengths, dtype=torch.long), 
            "ddG": ddG,
            "dddG": dddG,
            "ground_truth": ddG,                           
            "subset_type": st,
            "plddt": plddt
        }
        return collated