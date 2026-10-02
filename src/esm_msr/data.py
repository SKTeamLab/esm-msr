import os
import re
import gc
import math
import pickle
import random
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

from esm_msr.utils import custom_end_gap_alignment, determine_diffs
from esm_msr.routing import DOUBLE_DERIVED_SUBSETS, canonical_subset

# A library code ending in a mutation ('1A0N_L7S') means every measurement in it was
# made in that mutant background.
NATIVE_BACKGROUND_CODE_RE = re.compile(r'_[A-Z][0-9]+[A-Z]$')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MutationStabilityDataset(torch.utils.data.Dataset):
    """
    One protein library (one ``code``) of stability measurements, expanded into the
    training items of each subset and cached to disk.

    Generation walks the library's rows once and emits, per row:

    * ``single``   - a measured single mutation on its own structure (WT head).
    * ``double``   - a measured multi-mutant (predicted by the head ensemble).
    * ``cond``     - the conditional effect of one mutation of a double given the other,
                     derived as ddG_AB - ddG_B, scored by the MT head.
    * ``reversion`` - the same measurement as ``single`` read backwards (unrouted).

    ``cond`` replaces the former ``mut_ctx`` / ``mut_ctx_rev`` pair, which encoded the
    same quantity twice (see ``esm_msr.routing``). ``native_cond`` is assigned at load
    time by :meth:`_label_native_cond` for libraries whose code carries a mutation
    suffix, i.e. whose measurements were all made in a mutant background.

    Positions are 1-based with respect to the structure's sequence, which equals the
    index into the tokenized sequence because the tokenizer prepends BOS.
    """

    # Bump when the emitted items change shape or meaning, so a stale cache is never served.
    CACHE_VERSION = 'v2cond'

    # Dynamic range of the cDNA-display proteolysis dG estimates; dG_ML is clipped to it.
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
        incl_cond: bool = False,
        incl_reversions: bool = False,
        incl_native_cond: bool = False,
        cond_structure: str = 'reuse',
        dG_wt: Optional[float] = None,
        censor_margin: Optional[float] = None,
    ):
        """
        Args:
            dms_df: DataFrame of measurements for this library.
            tokenizer: ESM3 sequence tokenizer.
            dms_name: The library's ``code`` (e.g. '1A32', or '1A0N_L7S' for a
                mutant-background library).
            mut_structs_root: Root of the modeled mutant structures tree
                (``<root>/<code>/pdb_models/<chain>[<mut>].pdb``).
            score_name: Target column; 'ddG_ML' for MegaScale, 'ddG' for benchmarks.
            path: Cache directory.
            generate: Regenerate instead of reading the cache.
            incl_destab_bb: Include rows whose backbone is itself a mutant.
            structure_encoder: ESM3 structure encoder; required to emit structure tokens.
            incl_singles: Keep ``single`` items.
            incl_doubles: Keep ``double`` items.
            incl_cond: Keep ``cond`` items (conditional effects derived from doubles).
            incl_reversions: Keep ``reversion`` items.
            incl_native_cond: Keep ``native_cond`` items (measured in a mutant background).
            cond_structure: Structure a ``cond`` item conditions on.
                'reuse' uses the structure its parent double uses, i.e. the WT backbone
                unchanged: the partner's side chain is shown as wild type, which is wrong
                but is also what the MT adapter sees at inference time.
                'mask' additionally masks the partner site, which is honest about the
                unknown side chain but trains on inputs that inference does not reproduce
                unless masking is used there too.
                'model' prefers a modeled partner structure when one exists on disk,
                falling back to 'mask'.
                Baked into the generated items, so it is part of the cache name.
            dG_wt: Measured dG of this library's starting sequence; needed for censoring.
            censor_margin: Drop double-derived items whose states come within this many
                kcal/mol of the assay's dynamic range. See :meth:`_drop_censored`.
        """
        if cond_structure not in ('reuse', 'mask', 'model'):
            raise AssertionError(f"cond_structure must be 'reuse', 'mask' or 'model', got '{cond_structure}'.")

        self.score_name = score_name
        self.dms_name = dms_name
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.incl_destab_bb = incl_destab_bb
        self.cond_structure = cond_structure
        self.mut_structs_root = mut_structs_root

        self.include = {
            'single': incl_singles,
            'double': incl_doubles,
            'cond': incl_cond,
            'reversion': incl_reversions,
            'native_cond': incl_native_cond,
        }

        # Pre-cache the vocabulary mapping for rapid ID lookups
        self.vocab = self.tokenizer.get_vocab()

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        dms_df['ground_truth'] = dms_df['ddG']

        cond_tag = '' if cond_structure == 'reuse' else f'_{cond_structure}'
        self.cache_path = os.path.join(
            path if path is not None else '.',
            f"{self.dms_name}_{self.score_name}_{self.CACHE_VERSION}{cond_tag}.pkl"
        )

        logging.info(f"Dataset Cache Path: {self.cache_path}")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.data: List[Dict[str, Any]] = []
        self._encoded_struct_cache: Dict[Tuple, Tuple] = {}
        self._mutant_struct_cache: Dict[Tuple, Tuple] = {}
        self._parsed_pdb_cache: Dict[str, Tuple] = {}

        if generate or not os.path.exists(self.cache_path):
            logging.info(f"Generating and caching data for {self.dms_name}")
            self.data = self.generate_data(dms_df)
            self._save_data_to_cache()
        else:
            logging.info(f"Loading cached data for {self.dms_name}")
            self.load_data_from_cache()

        logging.info(f"Cache contains: {dict(Counter(i.get('subset_type', 'unknown') for i in self.data))}")

        # Mutant-background libraries: their singles are conditional measurements.
        self._label_native_cond()
        self._filter_dataset()
        if censor_margin is not None:
            self._drop_censored(dG_wt, censor_margin)
        self._extract_scalars()

    # ------------------------------------------------------------------ generation

    def generate_data(self, dms_df: Any) -> List[Dict[str, Any]]:
        """Expands the raw measurement rows of this library into training items."""
        if self.score_name == 'ddG_ML':
            df = dms_df.loc[dms_df['code'] == self.dms_name].copy()
            if len(df) == 0:
                raise AssertionError(f"No data found for code {self.dms_name}")
            df['mutated_sequence'] = df['aa_seq']
            return self._load_data(df, is_predicted=True)

        data = []
        dms_df['code_wt'] = dms_df['code']
        for (code, chain), df_sub in dms_df.groupby(['code', 'chain']):
            df_sub = df_sub.copy()
            df_sub['mutated_sequence'] = df_sub['mut_seq']
            data.extend(self._load_data(df_sub, incl_chain_in_code=True, is_predicted=False))
        return data

    # ------------------------------------------------------------------ post-load filters

    def _label_native_cond(self) -> None:
        """
        Label items from a mutant-background library as ``native_cond``.

        A library whose code carries a mutation suffix ('1A0N_L7S' = 1A0N measured in the
        L7S background) has no wild-type counterpart: every measurement in it is already a
        conditional effect ddG(X | L7S). Its singles therefore belong to the MT head, and
        are relabeled here so they toggle independently of ordinary singles. The on-disk
        cache keeps the generic label.

        Reversion items are left as ``reversion`` so they stay behind ``incl_reversions``;
        they restate the same measurements with the opposite residue visible.
        """
        n = 0
        for item in self.data:
            code = item.get('pdb') or self.dms_name
            if NATIVE_BACKGROUND_CODE_RE.search(code) and item.get('subset_type') == 'single':
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

    def _drop_censored(self, dG_wt: Optional[float], margin: float) -> None:
        """
        Drop double-derived items (``double``, ``cond``) that touch the assay's limits.

        dG_ML is clipped to [-1, 5] kcal/mol. When a double's additive expectation falls
        past the floor the measured dG_AB is clipped there, so
        dddG = ddG_AB - ddG_A - ddG_B comes out spuriously positive, and ddG(A|B) inherits
        the same error. Measured on the raw Tsuboyama table: of doubles with both singles,
        the ~34% with a state within 0.5 kcal/mol of a limit have mean dddG +1.74 kcal/mol
        and no agreement between the trypsin and chymotrypsin estimates (r = 0.07); the
        rest have mean +0.51 with r = 0.75.

        A state is the dG of the WT, of each single, of the double, and of the double's
        additive estimate.
        """
        if dG_wt is None or not np.isfinite(dG_wt):
            logging.warning(f"[{self.dms_name}] censor_margin set but WT dG unknown; no censoring applied.")
            return
        lo, hi = self.DG_FLOOR + margin, self.DG_CEILING - margin

        def _censored(item: Dict[str, Any]) -> bool:
            if item.get('subset_type') not in DOUBLE_DERIVED_SUBSETS:
                return False
            a, b = item.get('ddG_A', np.nan), item.get('ddG_B', np.nan)
            ddG_AB = item.get('ddG_AB', np.nan)
            states = np.array([0.0, a, b, ddG_AB, a + b], dtype=np.float64) + dG_wt
            states = states[np.isfinite(states)]
            return bool(((states < lo) | (states > hi)).any())

        before = len(self.data)
        self.data = [item for item in self.data if not _censored(item)]
        logging.info(f"[{self.dms_name}] censor_margin={margin}: dropped {before - len(self.data)} of {before} items "
                     f"(dG_wt={dG_wt:.2f}, allowed state range [{lo:.2f}, {hi:.2f}])")

    def _extract_scalars(self) -> None:
        """Contiguous scalar arrays for the sampler's balancing passes."""
        self.ddg_additive_arr = np.array([i.get('ddG_additive', np.nan) for i in self.data], dtype=np.float32)
        self.dddg_arr = np.array([i.get('dddG', np.nan) for i in self.data], dtype=np.float32)
        self.ground_truth_arr = np.array([i.get('ddG', np.nan) for i in self.data], dtype=np.float32)

    # ------------------------------------------------------------------ cache / dataset API

    def _save_data_to_cache(self) -> None:
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_from_cache(self) -> None:
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    # ------------------------------------------------------------------ item construction

    def _parse_mutations(self, row: Any, ref_seq: str, has_mut_type: bool) -> List[Tuple[str, int, str]]:
        """
        Mutations of one row as (from_aa, 1-based position, to_aa), relative to ``ref_seq``.

        Entries whose stated wild-type residue disagrees with ``ref_seq`` are dropped, which
        is how rows that do not apply to this backbone get skipped.
        """
        if not has_mut_type:
            mut_seq = row['mutated_sequence']
            offset, _, _ = custom_end_gap_alignment(mut_seq, ref_seq)
            return determine_diffs(mut_seq[offset:len(ref_seq) + offset], ref_seq)
        return [
            (m[0], int(m[1:-1]), m[-1])
            for m in row['mut_type'].split(':')
            if len(m) >= 3 and ref_seq[int(m[1:-1]) - 1] == m[0]
        ]

    def _structure_for(
        self,
        protein_chain: Any,
        backbone: str,
        base_code: str,
        chain: str,
        base_masks: List[int],
        *,
        modeled_bb: Optional[Tuple[str, int, str]] = None,
        extra_mask: Optional[int] = None,
        prefer_model: Optional[Tuple[str, int, str]] = None,
    ) -> Tuple[str, Tuple]:
        """
        Resolve the structure an item conditions on.

        Tries, in order: a modeled structure for ``prefer_model`` (the partner mutation the
        item conditions on); the library's modeled mutant backbone, masked at
        ``base_masks + extra_mask``; the backbone structure itself with the same masking.

        Returns ``(structure_type, (coords, plddt, structure_tokens, residue_index))`` where
        ``structure_type`` is 'model' (a real modeled structure), 'af' (the unmodified
        predicted WT backbone) or 'fake' (masked at one or more sites).
        """
        masks = list(base_masks)
        if extra_mask is not None and extra_mask not in masks:
            masks.append(extra_mask)

        if prefer_model is not None:
            wt_p, pos_p, mt_p = prefer_model
            seq_expect = list(protein_chain.sequence)
            if modeled_bb is not None:
                seq_expect[modeled_bb[1] - 1] = modeled_bb[2]
            seq_expect[pos_p - 1] = mt_p
            got = self._try_load_modeled_context(
                base_code, chain, wt_p, pos_p, mt_p,
                expected_seq=''.join(seq_expect), target_masks=list(base_masks),
            )
            if got is not None:
                return 'model', got[1:]

        if modeled_bb is not None:
            seq_expect = list(protein_chain.sequence)
            seq_expect[modeled_bb[1] - 1] = modeled_bb[2]
            got = self._try_load_modeled_context(
                base_code, chain, modeled_bb[0], modeled_bb[1], modeled_bb[2],
                expected_seq=''.join(seq_expect), target_masks=masks, force_mask=bool(masks),
            )
            if got is not None:
                return ('fake' if masks else 'model'), got[1:]

        encoded = self._get_encoded_structure(protein_chain, backbone, masks, force_mask=bool(masks))
        return ('fake' if masks else 'af'), encoded

    def _load_data(
        self,
        df: Any,
        is_predicted: bool = False,
        incl_chain_in_code: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Expands one library's measurement rows into items, one backbone group at a time.

        A library may span several backbones: the wild-type PDB, plus "destabilized"
        backbones named by a mutation (e.g. 'L7S'), whose rows were measured in that
        background. Within a group, single-mutation measurements are indexed so that a
        double can look up its two singles and derive the additive expectation, dddG, and
        the conditional effects.
        """
        self._mutant_struct_cache.clear()
        self._encoded_struct_cache.clear()
        self._parsed_pdb_cache.clear()

        data: List[Dict[str, Any]] = []
        has_mut_type = 'mut_type' in df.columns
        if not has_mut_type:
            logging.warning('Inferring mutations from mutated_sequence column because mut_type column was missing')

        if 'mut_structure' not in df.columns:
            df['mut_structure'] = df['pdb_file']
        df['mut_structure'] = df['mut_structure'].fillna(df['pdb_file'])

        for backbone, group in df.groupby('mut_structure'):
            base_code = group['code'].head(1).item()
            chain = group['chain'].head(1).item()
            code = base_code + chain if incl_chain_in_code else base_code

            is_mutant_backbone = not backbone.endswith('.pdb')
            if is_mutant_backbone and not self.incl_destab_bb:
                continue

            protein_chain = ProteinChain.from_pdb(
                group['pdb_file'].head(1).item() if is_mutant_backbone else backbone,
                chain, is_predicted=is_predicted,
            )

            # The sequence this backbone's rows are stated against: the structure's own
            # sequence, with the backbone mutation applied if this is a mutant backbone.
            seq_chars = list(protein_chain.sequence)
            base_masks: List[int] = []
            modeled_bb: Optional[Tuple[str, int, str]] = None
            if is_mutant_backbone:
                wt_bb, pos_bb, mt_bb = backbone[0], int(backbone[1:-1]), backbone[-1]
                seq_chars[pos_bb - 1] = mt_bb
                # Prefer a modeled structure for the backbone mutation; otherwise mask it,
                # since the WT structure shows the wrong side chain there.
                if self._try_load_modeled_context(base_code, chain, wt_bb, pos_bb, mt_bb,
                                                  expected_seq=''.join(seq_chars), target_masks=[]) is not None:
                    modeled_bb = (wt_bb, pos_bb, mt_bb)
                else:
                    base_masks.append(pos_bb)
            ref_seq = ''.join(seq_chars)

            def _seq_with(*muts: Tuple[str, int, str]) -> str:
                chars = list(ref_seq)
                for _, pos, to_aa in muts:
                    chars[pos - 1] = to_aa
                return ''.join(chars)

            parsed_rows = []
            single_ddG: Dict[Tuple[int, str], float] = {}
            for _, row in group.iterrows():
                muts = self._parse_mutations(row, ref_seq, has_mut_type)
                ddG = float(row['ddG'])
                parsed_rows.append((muts, ddG))
                if len(muts) == 1:
                    single_ddG[(muts[0][1], muts[0][2])] = ddG

            # The structure every row of this group shares (no extra masking).
            base_struct_type, base_struct = self._structure_for(
                protein_chain, backbone, base_code, chain, base_masks, modeled_bb=modeled_bb)

            for muts, ddG in tqdm(parsed_rows, desc=f"Expanding {code}", leave=False):
                if len(muts) == 1:
                    data.extend(self._single_items(
                        muts[0], ddG, code, ref_seq, base_code, chain, backbone, protein_chain,
                        base_masks, modeled_bb, base_struct_type, base_struct, _seq_with))
                elif len(muts) == 2:
                    data.extend(self._double_items(
                        muts, ddG, single_ddG, code, ref_seq, base_code, chain, backbone,
                        protein_chain, base_masks, modeled_bb, base_struct_type, base_struct, _seq_with))

        return data

    def _single_items(self, mut, ddG, code, ref_seq, base_code, chain, backbone, protein_chain,
                      base_masks, modeled_bb, base_struct_type, base_struct, _seq_with) -> List[Dict[str, Any]]:
        """The measured single mutation, and the same measurement read as a reversion."""
        wt_aa, pos, mt_aa = mut
        mt_seq = _seq_with(mut)
        items = [self._create_data_item(
            mutations=[mut], ddG=ddG, code=code, wt_seq=ref_seq, mt_seq=mt_seq,
            subset_type='single', structure_type=base_struct_type, structure=base_struct)]

        if self.include['reversion']:
            # The reverse measurement conditions on the mutant sequence, so it wants the
            # mutant's structure: the modeled one if it exists, else the WT masked at `pos`.
            rev_type, rev_struct = self._structure_for(
                protein_chain, backbone, base_code, chain, base_masks, modeled_bb=modeled_bb,
                extra_mask=pos, prefer_model=mut)
            items.append(self._create_data_item(
                mutations=[(mt_aa, pos, wt_aa)], ddG=-ddG, code=code, wt_seq=mt_seq, mt_seq=ref_seq,
                subset_type='reversion', structure_type=rev_type, structure=rev_struct))
        return items

    def _double_items(self, muts, ddG_AB, single_ddG, code, ref_seq, base_code, chain, backbone,
                      protein_chain, base_masks, modeled_bb, base_struct_type, base_struct,
                      _seq_with) -> List[Dict[str, Any]]:
        """
        The measured double, plus one ``cond`` item per ordered pair.

        For a double AB with both singles measured:
        ddG_additive = ddG_A + ddG_B, dddG = ddG_AB - ddG_additive, and the conditional
        effect of A in the B background is ddG(A|B) = ddG_AB - ddG_B.

        A ``cond`` item is scored by the MT pass on the sequence B+A (the "after" state,
        where both mutations are present) at position A, as logit(A) - logit(wtA). It
        conditions on the B background, whose true structure is unknown; which structure
        stands in for it is ``cond_structure``.
        """
        (wtA, posA, mtA), (wtB, posB, mtB) = muts
        ddG_A = single_ddG.get((posA, mtA), np.nan)
        ddG_B = single_ddG.get((posB, mtB), np.nan)
        ddG_additive = ddG_A + ddG_B
        dddG = ddG_AB - ddG_additive if np.isfinite(ddG_additive) else np.nan

        items = [self._create_data_item(
            mutations=list(muts), ddG=ddG_AB, dddG=dddG, ddG_additive=ddG_additive,
            ddG_A=ddG_A, ddG_B=ddG_B, ddG_AB=ddG_AB, code=code,
            wt_seq=ref_seq, mt_seq=_seq_with(*muts),
            subset_type='double', structure_type=base_struct_type, structure=base_struct)]

        if not self.include['cond'] or not np.isfinite(ddG_additive):
            return items

        # Two ordered pairs: (target A given background B) and (target B given background A).
        for (tgt, bg, ddG_bg) in (((wtA, posA, mtA), (wtB, posB, mtB), ddG_B),
                                  ((wtB, posB, mtB), (wtA, posA, mtA), ddG_A)):
            if self.cond_structure == 'reuse':
                s_type, struct = base_struct_type, base_struct
            else:
                s_type, struct = self._structure_for(
                    protein_chain, backbone, base_code, chain, base_masks, modeled_bb=modeled_bb,
                    extra_mask=bg[1],
                    prefer_model=bg if self.cond_structure == 'model' else None)
            items.append(self._create_data_item(
                mutations=[tgt], ddG=float(ddG_AB - ddG_bg), dddG=dddG,
                ddG_additive=ddG_additive, ddG_A=ddG_A, ddG_B=ddG_B, ddG_AB=ddG_AB, code=code,
                wt_seq=_seq_with(bg), mt_seq=_seq_with(tgt, bg),
                subset_type='cond', structure_type=s_type, structure=struct))
        return items

    def _create_data_item(
        self,
        mutations: List[Tuple[str, int, str]],
        ddG: float,
        code: str,
        wt_seq: str,
        mt_seq: str,
        subset_type: str,
        structure_type: str,
        structure: Tuple,
        dddG: float = np.nan,
        ddG_additive: float = np.nan,
        ddG_A: float = np.nan,
        ddG_B: float = np.nan,
        ddG_AB: float = np.nan,
    ) -> Dict[str, Any]:
        """
        Builds one cached item.

        ``wt_seq`` is the "before" state (the WT pass's input) and ``mt_seq`` the "after"
        state (the MT pass's input); ``wt_id``/``mt_id`` are the from/to residues, so for a
        reversion-style item ``mt_id`` holds a wild-type residue. ``ddG`` is always the
        effect of going from before to after.
        """
        coords, plddt, structure_tokens, residue_index = structure

        mut_pos, wt_ids, mt_ids = [], [], []
        for (from_aa, pos, to_aa) in mutations:
            f_id, t_id = self.vocab.get(from_aa), self.vocab.get(to_aa)
            if f_id is None or t_id is None:
                raise AssertionError(f"Unknown amino acid token detected: WT={from_aa}, MT={to_aa}")
            mut_pos.append(pos)
            wt_ids.append(f_id)
            mt_ids.append(t_id)

        def _f(x):
            x = float(x)
            return x if np.isfinite(x) else np.nan

        logging.debug(f'Created data item: {code}, {mutations}, {subset_type}, {structure_type}, ddG={ddG}, dddG={dddG}')
        return {
            'pdb': code,
            'mutations': mutations,
            'wt_sequence_tokens': np.array(self.tokenizer.encode(wt_seq), dtype=np.int64),
            'mt_sequence_tokens': np.array(self.tokenizer.encode(mt_seq), dtype=np.int64),
            'mut_pos': np.array(mut_pos, dtype=np.int64),
            'wt_id': np.array(wt_ids, dtype=np.int64),
            'mt_id': np.array(mt_ids, dtype=np.int64),
            'coords_orig': coords.clone().cpu().numpy(),
            'structure_tokens_orig': structure_tokens.clone().cpu().numpy(),
            'residue_index': residue_index.clone().cpu().numpy() if residue_index is not None else None,
            'plddt': plddt.clone().cpu().numpy(),
            'ddG': float(ddG),
            'dddG': _f(dddG),
            'ddG_additive': _f(ddG_additive),
            'ddG_A': _f(ddG_A),
            'ddG_B': _f(ddG_B),
            'ddG_AB': _f(ddG_AB),
            'valid_dddG_mask': bool(np.isfinite(dddG)),
            'subset_type': subset_type,
            'structure_type': structure_type,
        }

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


# Retired name; kept so external scripts and pickles that reference it still import.
ProteinStructureMutationEpistasisDataset = MutationStabilityDataset


def collate_fn_twopass(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Vectorized collation specifically built for MSRModel's homogeneous microbatch assumption.
    If the batch contains heterogeneous sequences, the torch.stack commands will intentionally 
    raise a RuntimeError, caught here and re-raised as an AssertionError to prevent silent evaluation corruption.
    """
    B = len(batch)
    
    pdb = [item['pdb'] for item in batch]
    mutations = [item['mutations'] for item in batch]
    subset_type = [canonical_subset(item.get('subset_type', 'single')) for item in batch]
    plddt = [torch.as_tensor(item['plddt'], dtype=torch.float32) for item in batch]

    ddG = torch.tensor([float(item.get('ddG', float('nan'))) for item in batch], dtype=torch.float32)
    dddG = torch.tensor([float(item.get('dddG', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_additive = torch.tensor([float(item.get('ddG_additive', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_A = torch.tensor([float(item.get('ddG_A', float('nan'))) for item in batch], dtype=torch.float32)
    ddG_B = torch.tensor([float(item.get('ddG_B', float('nan'))) for item in batch], dtype=torch.float32)
    valid_dddG_mask = torch.tensor([bool(item.get('valid_dddG_mask', False)) for item in batch], dtype=torch.bool)

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
    
    Architectural improvements:
    1. Restores PyTorch asynchronous worker prefetching.
    2. Caches subset categorizations during __init__ to avoid O(N) epoch stalls.
    3. Handles 2D sampling and subset caps strictly via integer indices.
    """
    SUBSET_ORDER = ['single', 'double', 'reversion', 'cond', 'native_cond']

    def __init__(
        self,
        datasets: List[Any],
        batch_size: int,
        train_list: List[str],
        strategy: str = 'all',
        *,
        subset_caps: Optional[Dict[str, Optional[float]]] = None,
        subset_balance_configs: Optional[Dict[str, Dict[str, Any]]] = None,
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
        
        self.subset_balance_configs = subset_balance_configs or {
            'cond': {'bins': 15, 'cap_percentile': 75.0, 'missing_cap_fraction': 0.20},
        }

        self.subset_caps: Dict[str, Optional[float]] = {
            'single': None, 'double': 0.0, 'reversion': 0.0, 'cond': None, 'native_cond': None,
        }
        if subset_caps is not None:
            self.subset_caps.update(subset_caps)

        logging.info(f"Using subset cap config: {self.subset_caps}")
        logging.info(f"Using subset balance config: {self.subset_balance_configs}")

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
                
                stype = canonical_subset(item.get('subset_type', 'single'))
                if stype not in buckets:
                    stype = 'single'
                buckets[stype].append(i)
                
            self._cached_buckets.append(buckets)

        # 3. Perform a dry-run to calculate exact batch sizes for the DataLoader __len__
        self.num_batches = self._calculate_epoch_batches(dry_run=True)

    def _get_ddg_stats(self, indices: List[int], dataset: Any) -> Tuple[int, float, float]:
        if not indices:
            return 0, float('nan'), float('nan')
        # Optimized lookup directly from scalar arrays if available
        if hasattr(dataset, 'ground_truth_arr'):
            vals = dataset.ground_truth_arr[indices]
            valid_mask = np.isfinite(vals)
            vals = vals[valid_mask]
        else:
            vals = [dataset[i].get('ddG') for i in indices]
            vals = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
            
        if not len(vals):
            return len(indices), float('nan'), float('nan')
        return len(indices), float(np.mean(vals)), float(np.std(vals))

    def _balance_subset_2d(self, indices: List[int], dataset: Any, protein_name: str, config: Dict, subset_name: str) -> List[int]:
        if not indices or config is None:
            return indices
            
        bins = config.get('bins', 15)
        cap_percentile = config.get('cap_percentile', 50.0)
        missing_cap_fraction = config.get('missing_cap_fraction', 0.10)
        
        if not hasattr(dataset, 'ddg_additive_arr') or not hasattr(dataset, 'dddg_arr'):
            raise AssertionError(
                f"Dataset for {protein_name} is missing pre-extracted scalar arrays. "
                "Ensure `_extract_scalars()` was called during dataset generation."
            )
            
        idx_arr = np.array(indices, dtype=np.int64)
        val_ddg_add = dataset.ddg_additive_arr[idx_arr]
        val_dddg = dataset.dddg_arr[idx_arr]
        
        valid_mask = np.isfinite(val_ddg_add) & np.isfinite(val_dddg)
        valid_indices = idx_arr[valid_mask].tolist()
        missing_keys_indices = idx_arr[~valid_mask].tolist()
        
        if len(valid_indices) < 3:
            return indices
            
        ddg_add_arr = val_ddg_add[valid_mask]
        dddg_arr = val_dddg[valid_mask]
        
        H, xedges, yedges = np.histogram2d(ddg_add_arr, dddg_arr, bins=bins)
        populated_counts = H[H > 0]
        
        if len(populated_counts) == 0:
            return indices
            
        cap = int(np.percentile(populated_counts, cap_percentile))
        cap = max(1, cap)
        
        x_bins = np.clip(np.digitize(ddg_add_arr, xedges[:-1]) - 1, 0, bins - 1)
        y_bins = np.clip(np.digitize(dddg_arr, yedges[:-1]) - 1, 0, bins - 1)
        
        bin_dict: Dict[Tuple[int, int], List[int]] = {}
        for i, idx in enumerate(valid_indices):
            coord = (x_bins[i], y_bins[i])
            bin_dict.setdefault(coord, []).append(idx)
            
        balanced_indices = []
        for coord, binned_idxs in bin_dict.items():
            if len(binned_idxs) > cap:
                self._rng.shuffle(binned_idxs)
                balanced_indices.extend(binned_idxs[:cap])
            else:
                balanced_indices.extend(binned_idxs)
                
        max_missing = int(len(balanced_indices) * missing_cap_fraction)
        self._rng.shuffle(missing_keys_indices)
        missing_to_add = missing_keys_indices[:max_missing]
        
        logging.info(f"Balanced '{subset_name}' Final Size: {len(indices)} -> {len(balanced_indices) + len(missing_to_add)}")

        if self.verbose:
            c_valid, m_valid, s_valid = self._get_ddg_stats(valid_indices, dataset)
            c_bal, m_bal, s_bal = self._get_ddg_stats(balanced_indices, dataset)
            c_miss, m_miss, s_miss = self._get_ddg_stats(missing_keys_indices, dataset)
            
            logging.info(f"\n[info] 2D Balance Stats for {protein_name} - '{subset_name}':")
            logging.info(f"  -> Pre-subsample (Valid 2D): Count={c_valid}, Mean(ddG)={m_valid:.3f}, Std(ddG)={s_valid:.3f}")
            logging.info(f"  -> Post-subsample (Valid 2D): Count={c_bal}, Mean(ddG)={m_bal:.3f}, Std(ddG)={s_bal:.3f}")
            logging.info(f"  -> Missing/NaN Items: Count={c_miss}, Mean(ddG)={m_miss:.3f}, Std(ddG)={s_miss:.3f}")
            logging.info(f"  -> Missing Items Capped: {len(missing_keys_indices)} -> {len(missing_to_add)} "
                  f"({missing_cap_fraction*100:.1f}% of {len(balanced_indices)} post-subsampled items)")
            logging.info(f"Balanced '{subset_name}' Final Size: {len(indices)} -> {len(balanced_indices) + len(missing_to_add)} "
                  f"(bins={bins}, random cap={cap} items/bin.)\n")
                  
        balanced_indices.extend(missing_to_add)
        return balanced_indices

    def _calculate_epoch_batches(self, dry_run: bool = False) -> int:
        """Core logic to balance, shuffle, and chunk datasets."""
        self.all_batches = [] # List of dataset batch lists
        
        for idx, ds in enumerate(self.datasets):
            protein_name = self.train_list[idx]
            buckets = {k: list(v) for k, v in self._cached_buckets[idx].items()} # copy
            
            # Apply 2D balancing
            if self.subset_balance_configs is not None:
                for subset_name, config in self.subset_balance_configs.items():
                    if subset_name in buckets and len(buckets[subset_name]) > 0:
                        buckets[subset_name] = self._balance_subset_2d(
                            buckets[subset_name], ds, protein_name, config, subset_name
                        )

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
    subset_balance_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Factory function replacing both SubsetRestrictedProteinCyclingDataLoader 
    and ProteinCyclingDataLoader.
    """
    datasets = [dl.dataset for dl in dataloaders]
    
    sampler = ProteinCyclingBatchSampler(
        datasets=datasets,
        batch_size=batch_size,
        train_list=train_list,
        strategy=strategy,
        subset_caps=subset_caps,
        subset_balance_configs=subset_balance_configs
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
        st = [canonical_subset(it.get("subset_type", 'single')) for it in batch]

        ddG = torch.tensor([float(it.get('ddG', float('nan'))) for it in batch], dtype=torch.float32)
        dddG = torch.tensor([float(it.get('dddG', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_additive = torch.tensor([float(it.get('ddG_additive', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_A = torch.tensor([float(it.get('ddG_A', float('nan'))) for it in batch], dtype=torch.float32)
        ddG_B = torch.tensor([float(it.get('ddG_B', float('nan'))) for it in batch], dtype=torch.float32)
        valid_dddG_mask = torch.tensor([bool(it.get('valid_dddG_mask', False)) for it in batch], dtype=torch.bool)

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
        st = [canonical_subset(it.get("subset_type", 'single')) for it in batch]

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