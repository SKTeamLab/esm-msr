"""Building a library's cache: the half of the dataset that needs the raw table, the structures and the ESM3 structure encoder.

Training does not import this. It reads a cache that was built once (MutationStabilityDataset in data.py), so a training run
starts without the raw table, without the structure encoder and without any of the structure-file plumbing below. The code here was
moved out of data.py unchanged; analysis_scripts/build_cache.py is its command line.

One library (one code) of measurements is expanded into items and pickled to <path>/<code>_<score>_<version>.pkl:

* single - a measured single mutation on its own structure (WT head).
* double - a measured double mutant (predicted by the head ensemble).
* cond   - the conditional effect of one mutation of a double given the other, ddG_AB - ddG_B. Built, because cache c5 has them, but
  nothing trains on them any more and the c6 cache leaves them out (build_cache.py convert).

Every item also carries the raw measurement row it came from (meas, see data.MEAS_COLUMNS) and its library's starting dG
(dG_wt). Positions are 1-based with respect to the structure's sequence, which equals the index into the tokenized sequence because
the tokenizer prepends BOS.
"""
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from esm.utils.constants import esm3 as C
from esm.utils.structure.protein_chain import ProteinChain

from esm_msr import censoring
from esm_msr.data import MEAS_COLUMNS, resolve_structure_path
from esm_msr.utils import custom_end_gap_alignment, determine_diffs


class LibraryCacheBuilder:
    """Builds and saves the cache of one library from its rows of the (preprocessed) raw table."""

    # Bump when the emitted items change shape or meaning, so a stale cache is never served. c3: first conservative-branch cache
    # (every item carries meas / dG_meas / dG_wt / y_floor-era fields). c5: dG_wt is the exact per-library reference recovered
    # from dG_ML - ddG_ML rather than the mean of the 'wt' rows. c6 = c5 without the cond items.
    CACHE_VERSION = 'c5'

    def __init__(self, dms_name: str, tokenizer: Any, mut_structs_root: str, path: str, score_name: str = 'ddG_ML',
                 structure_encoder: Optional[Any] = None, incl_destab_bb: bool = True, dG_wt: Optional[float] = None,
                 cache_version: Optional[str] = None, emit_cond: bool = False):
        self.score_name = score_name
        self.cache_version = cache_version or self.CACHE_VERSION
        self.dms_name = dms_name
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.incl_destab_bb = incl_destab_bb
        self.emit_cond = emit_cond            # c5 carried the cond items; c6 does not (nothing trains on them)
        self.mut_structs_root = mut_structs_root
        self.dG_wt = float(dG_wt) if dG_wt is not None and np.isfinite(dG_wt) else float('nan')
        self.vocab = self.tokenizer.get_vocab()
        self.cache_path = os.path.join(path, f"{self.dms_name}_{self.score_name}_{self.cache_version}.pkl")
        self._encoded_struct_cache: Dict[Tuple, Tuple] = {}
        self._mutant_struct_cache: Dict[str, Tuple] = {}
        self._parsed_pdb_cache: Dict[str, Tuple] = {}

    def build(self, dms_df: Any) -> List[Dict[str, Any]]:
        """Generate the library's items from its rows and write the cache file. Returns the items."""
        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        dms_df['ground_truth'] = dms_df['ddG']
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        logging.info(f"Generating and caching data for {self.dms_name}")
        data = self.generate_data(dms_df)
        for item in data:
            item['dG_wt'] = self.dG_wt
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'items': data, 'floor_stats': {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return data

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
                wt_protein_chain = ProteinChain.from_pdb(resolve_structure_path(backbone), chain, is_predicted=is_predicted)
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
                resolve_structure_path(group['pdb_file'].head(1).item() if not backbone.endswith('.pdb') else backbone),
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
                if self.emit_cond and len(muts) == 2:
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
