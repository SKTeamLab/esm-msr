import os
import re
import gc
import math
import pickle
import random
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C

from esm_msr.utils import custom_end_gap_alignment, determine_diffs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ProteinStructureMutationEpistasisDataset(torch.utils.data.Dataset):
    """
    Optimized dataloader for the Vectorized Two-Pass Latent Ensemble architecture.
    Integrates static WT structural evaluation with modeled mutant contexts and reversions.
    """
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
        incl_mut_ctx: bool = False,
        incl_reversions: bool = False
    ):
        """
        Initializes the dataset, loading from cache if available or generating from scratch.

        Args:
            dms_df: DataFrame containing Deep Mutational Scanning (DMS) data.
            tokenizer: Tokenizer for sequence processing.
            dms_name: Identifier for the DMS dataset.
            mut_structs_root: Root directory for mutant structure files.
            score_name: Column name in `dms_df` to use as the target score (default: 'ddG_ML').
            path: Directory path for caching data. Defaults to current directory.
            generate: If True, forces regeneration of data bypassing the cache.
            incl_destab_bb: Whether to include destabilized backbones.
            structure_encoder: Encoder model for processing 3D structures.
            incl_singles: Include single mutations.
            incl_doubles: Include double mutations.
            incl_mut_ctx: Include synthesized mutant-context singles.
            incl_reversions: Include reversion mutations.
        """
        self.score_name = score_name
        self.dms_name = dms_name
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.incl_destab_bb = incl_destab_bb

        self.include_singles = incl_singles
        self.include_doubles = incl_doubles
        self.include_mut_context = incl_mut_ctx
        self.include_reversions = incl_reversions
        self.mut_structs_root = mut_structs_root
        
        # Pre-cache the vocabulary mapping for rapid ID lookups
        self.vocab = self.tokenizer.get_vocab()

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        dms_df['ground_truth'] = dms_df['ddG']

        if path is None:
            path = '.'

        #self.dms_name = re.sub(r'_[A-Za-z]+[0-9]+[A-Za-z]+$', '', self.dms_name)
        self.cache_path = os.path.join(
            path,
            f"{self.dms_name}_{self.score_name}_MAX_Smasked0.pkl" #_StaticWT
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
            self._save_data_to_cache()
        else:
            logging.info(f"Loading cached data for {self.dms_name}")
            self.load_data_from_cache()

        cache_composition = {}
        for item in self.data:
            stype = item.get('subset_type', 'unknown')
            cache_composition[stype] = cache_composition.get(stype, 0) + 1
        logging.info(f"Cache contains: {cache_composition}")

        # remove unwanted subsets
        self._filter_dataset()  
        # to facilitate subsampling      
        self._extract_scalars()

    def generate_data(self, dms_df: Any) -> List[Dict[str, Any]]:
        """
        Processes raw DMS dataframe into a list of structured data items.
        """
        if self.score_name == 'ddG_ML':
            df = dms_df.loc[dms_df['code'] == self.dms_name].copy()
            if len(df) == 0:
                raise AssertionError(f"No data found for code {self.dms_name}")
            info = df.head(1)
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
    
    def _filter_dataset(self) -> None:
        """Filters the in-memory dataset based on subset inclusion flags."""
        allowed_types = set()
        if self.include_singles:
            allowed_types.add('single')
        if self.include_doubles:
            allowed_types.add('double')
        if self.include_mut_context:
            allowed_types.add('mut_ctx')
        if self.include_reversions:
            allowed_types.add('reversion')
            
        original_len = len(self.data)
        self.data = [item for item in self.data if item.get('subset_type') in allowed_types]
        logging.info(f"Filtered dataset from {original_len} to {len(self.data)} items based on allowed types: {allowed_types}")

    def _extract_scalars(self) -> None:
        """Extracts scalar values into contiguous numpy arrays for fast indexing/subsampling."""
        logging.info(f"Pre-extracting scalar arrays for {len(self.data)} items...")
        ddg_add_list = []
        dddg_list = []
        
        for item in self.data:
            ddg_add_list.append(item.get('ddG_additive', float('nan')))
            dddg_list.append(item.get('dddG', float('nan')))
            
        self.ddg_additive_arr = np.array(ddg_add_list, dtype=np.float32)
        self.dddg_arr = np.array(dddg_list, dtype=np.float32)

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
        wt_backbone_id = None
        wt_protein_chain = None
        wt_canonical_seq = None

        for backbone, group in df.groupby('mut_structure'):
            if backbone.endswith('.pdb'):
                chain = group['chain'].head(1).item()
                wt_backbone_id = backbone
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
                if len(muts) == 1: 
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
                        ddG_A=np.nan, ddG_B=np.nan, ddG_additive=np.nan, structure_type=struct_type
                    ))

                    # (B) Reversions
                    rev_mut = [(mt, pos, wt)]
                    rev_ddG = -float(row['ddG'])
                    
                    rev_target_masks = base_mask_positions.copy()
                            
                    maybe = self._try_load_modeled_context(
                        base_code, chain, wt, pos, mt, expected_seq=mt_seq, target_masks=rev_target_masks
                    )
                    
                    if maybe is not None:
                        struct_type = 'model'
                        _, coords_m, plddt_m, structure_tokens_m, residue_index_m = maybe
                    else:
                        struct_type = 'fake'
                        # Fallback explicitly appends target `pos` and asserts force_mask=True
                        fallback_masks = rev_target_masks.copy()
                        if pos not in fallback_masks:
                            fallback_masks.append(pos)
                            
                        maybe_fallback = None
                        if has_modeled_backbone:
                            maybe_fallback = self._try_load_modeled_context(
                                base_code, chain, wt_bb, pos_bb, mt_bb, expected_seq=corrected_seq, target_masks=fallback_masks, force_mask=True
                            )
                            
                        if maybe_fallback is not None:
                            _, coords_m, plddt_m, structure_tokens_m, residue_index_m = maybe_fallback
                        else:
                            coords_m, plddt_m, structure_tokens_m, residue_index_m = self._get_encoded_structure(
                                protein_chain, backbone, fallback_masks, force_mask=True
                            )
                    
                    data.append(self._create_data_item(
                        mutations=rev_mut, ddG=rev_ddG, dddG=np.nan, code=code,
                        wt_seq=mt_seq, mt_seq=wt_seq, 
                        coords=coords_m, plddt=plddt_m, structure_tokens=structure_tokens_m, residue_index=residue_index_m,
                        subset_type='reversion', ddG_A=np.nan, ddG_B=np.nan, ddG_additive=np.nan, structure_type=struct_type
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
                        subset_type='double', structure_type=struct_type
                    ))

                # (D) Synthesized mutant-context singles
                if len(muts) == 2: 
                    (wtA, posA, mtA), (wtB, posB, mtB) = muts
                    if (posA, mtA) in single_map and (posB, mtB) in single_map:
                        ddG_A = single_map[(posA, mtA)]
                        ddG_B = single_map[(posB, mtB)]
                        ddG_additive = ddG_A + ddG_B
                        
                        t_B_given_A = ddG_val - ddG_A
                        t_A_given_B = ddG_val - ddG_B

                        # --- "B|A": context with A present
                        ctx_wt_A_list = list(corrected_seq)
                        ctx_wt_A_list[posA-1] = mtA
                        ctx_wt_A = ''.join(ctx_wt_A_list)

                        ctx_mt_A_list = list(ctx_wt_A_list)
                        ctx_mt_A_list[posB-1] = mtB
                        ctx_mt_A = ''.join(ctx_mt_A_list)

                        ctx_masks = base_mask_positions.copy()

                        maybe = self._try_load_modeled_context(
                            base_code=base_code, chain=chain, 
                            partner_wt=wtA, partner_pos=posA, partner_mut=mtA, 
                            expected_seq=ctx_wt_A, 
                            target_masks=ctx_masks
                        )
                        
                        if maybe is not None:
                            struct_type_A = 'model'
                            _, coords_ctx_A, plddt_ctx_A, struct_ctx_A, residue_index_ctx_A = maybe
                        else:
                            struct_type_A = 'fake'
                            fallback_masks = ctx_masks.copy()
                            if posA not in fallback_masks:
                                fallback_masks.append(posA)
                                
                            maybe_fallback = None
                            if has_modeled_backbone:
                                maybe_fallback = self._try_load_modeled_context(
                                    base_code, chain, wt_bb, pos_bb, mt_bb, expected_seq=corrected_seq, target_masks=fallback_masks, force_mask=True
                                )
                                
                            if maybe_fallback is not None:
                                _, coords_ctx_A, plddt_ctx_A, struct_ctx_A, residue_index_ctx_A = maybe_fallback
                            else:
                                coords_ctx_A, plddt_ctx_A, struct_ctx_A, residue_index_ctx_A = self._get_encoded_structure(
                                    protein_chain, backbone, fallback_masks, force_mask=True
                                )

                        data.append(self._create_data_item(
                            mutations=[(wtB, posB, mtB)], ddG=float(t_B_given_A), dddG=dddG_val, 
                            ddG_additive=ddG_additive, ddG_A=ddG_A, ddG_B=ddG_B, code=code,
                            wt_seq=ctx_wt_A, mt_seq=ctx_mt_A, coords=coords_ctx_A, plddt=plddt_ctx_A, 
                            structure_tokens=struct_ctx_A, residue_index=residue_index_ctx_A, subset_type='mut_ctx', structure_type=struct_type_A
                        ))

                        # --- "A|B": context with B present
                        ctx_wt_B_list = list(corrected_seq)
                        ctx_wt_B_list[posB-1] = mtB
                        ctx_wt_B = ''.join(ctx_wt_B_list)

                        ctx_mt_B_list = list(ctx_wt_B_list)
                        ctx_mt_B_list[posA-1] = mtA
                        ctx_mt_B = ''.join(ctx_mt_B_list)

                        ctx_masks = base_mask_positions.copy()

                        maybe = self._try_load_modeled_context(
                            base_code=base_code, chain=chain, 
                            partner_wt=wtB, partner_pos=posB, partner_mut=mtB, 
                            expected_seq=ctx_wt_B, 
                            target_masks=ctx_masks
                        )
                        
                        if maybe is not None:
                            struct_type_B = 'model'
                            _, coords_ctx_B, plddt_ctx_B, struct_ctx_B, residue_index_ctx_B = maybe
                        else:
                            struct_type_B = 'fake'
                            fallback_masks = ctx_masks.copy()
                            if posB not in fallback_masks:
                                fallback_masks.append(posB)
                                
                            maybe_fallback = None
                            if has_modeled_backbone:
                                maybe_fallback = self._try_load_modeled_context(
                                    base_code, chain, wt_bb, pos_bb, mt_bb, expected_seq=corrected_seq, target_masks=fallback_masks, force_mask=True
                                )
                                
                            if maybe_fallback is not None:
                                _, coords_ctx_B, plddt_ctx_B, struct_ctx_B, residue_index_ctx_B = maybe_fallback
                            else:
                                coords_ctx_B, plddt_ctx_B, struct_ctx_B, residue_index_ctx_B = self._get_encoded_structure(
                                    protein_chain, backbone, fallback_masks, force_mask=True
                                )

                        data.append(self._create_data_item(
                            mutations=[(wtA, posA, mtA)], ddG=float(t_A_given_B), dddG=dddG_val, 
                            ddG_additive=ddG_additive, ddG_A=ddG_A, ddG_B=ddG_B, code=code,
                            wt_seq=ctx_wt_B, mt_seq=ctx_mt_B, coords=coords_ctx_B, plddt=plddt_ctx_B, 
                            structure_tokens=struct_ctx_B, residue_index=residue_index_ctx_B, subset_type='mut_ctx', structure_type=struct_type_B
                        ))
        return data
    
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
        structure_type: str
    ) -> Dict[str, Any]:
        """Constructs the fully vectorized dictionary item for the Collate function."""
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

        logging.info(f'Created data item: {code}, {mutations}, {subset_type}, {structure_type}, ddG={ddG}, dddG={dddG}')      
        return {
            'pdb': code,
            'mutations': mutations,
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


@dataclass
class ProteinSamplerState:
    """Tracks simplified sampling state for a single protein dataset."""
    protein_name: str
    items: List[Dict[str, Any]]
    cursor: int = 0
    batches_allocated: int = 0
    batches_drawn: int = 0
    
    @property
    def remaining_items(self) -> int:
        return len(self.items) - self.cursor
        
    @property
    def total_batches_possible(self) -> int:
        return len(self.items) # Handled externally based on batch size


class SubsetRestrictedProteinCyclingDataLoader:
    """
    Cycles through multiple protein datasets.
    Subsets are pooled globally per-protein and shuffled, with no per-batch constraints.
    """
    
    SUBSET_ORDER = ['single', 'double', 'reversion', 'mut_ctx']
    
    def __init__(
        self,
        dataloaders: List,
        batch_size: int,
        train_list: List[str],
        collate_fn: Callable,
        strategy: str = 'all',
        *,
        subset_caps: Optional[Dict[str, Optional[float]]] = None,
        subset_balance_configs: Optional[Dict[str, Dict[str, Any]]] = None, 
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initializes the cycling dataloader.

        Args:
            dataloaders: List of PyTorch DataLoaders.
            batch_size: Batch size.
            train_list: List of protein names corresponding to the dataloaders.
            collate_fn: Function to collate data items into batches.
            strategy: Strategy for epoch termination ('min' limits to the shortest dataset, 'all' exhausts all).
            subset_caps: Mapping of subset names to fractional caps (e.g., {'mut_ctx': 0.1}). 
                         At least one subset must be unrestricted (value: None).
            subset_balance_configs: 2D balancing configuration per subset type.
            rng_seed: Seed for local random number generator to ensure reproducibility.
            spatial_threshold: Maximum 3D spatial distance for 'over_and_back' subset.
            min_sequence_distance: Minimum sequence separation for 'over_and_back' subset.
            pdb_dir: Root path for AlphaFold models used in computing spatial distances.
            verbose: Enable extensive logging during epoch initialization.
        """
        if strategy not in ('min', 'all'):
            raise ValueError(f"strategy must be 'min' or 'all', got '{strategy}'")
        
        self.dataloaders = list(dataloaders)
        self.batch_size = batch_size
        self.train_list = list(train_list)
        self.collate_fn = collate_fn
        self.strategy = strategy
        self.verbose = verbose

        self.subset_balance_configs = subset_balance_configs
        if self.subset_balance_configs is None:
            self.subset_balance_configs = {
                'double': {'bins': 15, 'cap_percentile': 75.0, 'missing_cap_fraction': 0.20},
            }

        self._rng = random.Random(rng_seed)
        
        self.subset_caps: Dict[str, Optional[float]] = {
            'single': None,
            'double': None,
            'mut_ctx': 0.0,
            'reversion': 0.0
        }
        if subset_caps is not None:
            self.subset_caps.update(subset_caps)
            logging.info(f"Updated subset_caps with user-provided values: {self.subset_caps}")

        # Validate that at least one bucket is unrestricted to serve as the baseline
        self.unrestricted_keys = [k for k in self.SUBSET_ORDER if self.subset_caps.get(k) is None]
        if not self.unrestricted_keys:
            raise AssertionError(
                "Invalid subset_caps: At least one category (e.g., 'single') must be mapped to None "
                "to serve as the unrestricted baseline. Otherwise, the dataset size collapses to 0."
            )
        
        # State
        self.protein_states: List[ProteinSamplerState] = []
        self.total_batches_drawn = 0
        self.target_batches = 0
        self.current_loader_idx = 0
        self.epoch_counter = 0  
        self._initialized = False
        
        # Cache for distance matrices to avoid per-epoch recalculation
        self._dist_matrix_cache: Dict[str, np.ndarray] = {}

    def _get_ddg_stats(self, indices: List[int], dataset: Any) -> Tuple[int, float, float]:
        """Safely extracts ddG stats for a list of dataset indices."""
        if not indices:
            return 0, float('nan'), float('nan')
            
        vals = []
        for i in indices:
            val = dataset[i].get('ddG')
            if val is not None and not math.isnan(val) and not math.isinf(val):
                vals.append(val)
                
        if not vals:
            return len(indices), float('nan'), float('nan')
            
        arr = np.array(vals)
        return len(indices), float(np.mean(arr)), float(np.std(arr))
        
    def _balance_subset_2d(self, indices: List[int], dataset: Any, protein_name: str, config: Dict, subset_name: str) -> List[int]:
        """
        Randomly subsamples dense 2D bins based on percentile thresholds.
        Does not use a rolling window; samples are unseeded and randomly drawn each epoch.
        """
        if not indices or config is None:
            return indices
            
        bins = config.get('bins', 15)
        cap_percentile = config.get('cap_percentile', 50.0)
        missing_cap_fraction = config.get('missing_cap_fraction', 0.10)
        
        if not hasattr(dataset, 'ddg_additive_arr') or not hasattr(dataset, 'dddg_arr'):
            raise AttributeError(
                f"Dataset for {protein_name} is missing pre-extracted scalar arrays. "
                "Ensure `_extract_scalars()` was called during dataset initialization."
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
        
        x_bins = np.digitize(ddg_add_arr, xedges[:-1]) - 1
        y_bins = np.digitize(dddg_arr, yedges[:-1]) - 1
        x_bins = np.clip(x_bins, 0, bins - 1)
        y_bins = np.clip(y_bins, 0, bins - 1)
        
        bin_dict: Dict[Tuple[int, int], List[int]] = {}
        for i, idx in enumerate(valid_indices):
            coord = (x_bins[i], y_bins[i])
            if coord not in bin_dict:
                bin_dict[coord] = []
            bin_dict[coord].append(idx)
            
        balanced_indices = []
        
        for coord, binned_idxs in bin_dict.items():
            if len(binned_idxs) > cap:
                # Random sampling per epoch
                self._rng.shuffle(binned_idxs)
                balanced_indices.extend(binned_idxs[:cap])
            else:
                balanced_indices.extend(binned_idxs)
                
        # Handle the missing/NaN epistasis items with the default cap fraction
        max_missing = int(len(balanced_indices) * missing_cap_fraction)
        self._rng.shuffle(missing_keys_indices)
        missing_to_add = missing_keys_indices[:max_missing]
        
        if self.verbose:
            c_valid, m_valid, s_valid = self._get_ddg_stats(valid_indices, dataset)
            c_bal, m_bal, s_bal = self._get_ddg_stats(balanced_indices, dataset)
            c_miss, m_miss, s_miss = self._get_ddg_stats(missing_keys_indices, dataset)
            
            logging.debug(f"\n[DEBUG] 2D Balance Stats for {protein_name} - '{subset_name}':")
            logging.debug(f"  -> Pre-subsample (Valid 2D): Count={c_valid}, Mean(ddG)={m_valid:.3f}, Std(ddG)={s_valid:.3f}")
            logging.debug(f"  -> Post-subsample (Valid 2D): Count={c_bal}, Mean(ddG)={m_bal:.3f}, Std(ddG)={s_bal:.3f}")
            logging.debug(f"  -> Missing/NaN Items: Count={c_miss}, Mean(ddG)={m_miss:.3f}, Std(ddG)={s_miss:.3f}")
            logging.debug(f"  -> Missing Items Capped: {len(missing_keys_indices)} -> {len(missing_to_add)} "
                  f"({missing_cap_fraction*100:.1f}% of {len(balanced_indices)} post-subsampled items)")
            logging.debug(f"Balanced '{subset_name}' Final Size: {len(indices)} -> {len(balanced_indices) + len(missing_to_add)} "
                  f"(bins={bins}, random cap={cap} items/bin.)\n")
                  
        balanced_indices.extend(missing_to_add)
        return balanced_indices

    def _initialize_epoch(self) -> None:
        """Sets up random states, shuffling, and subset sampling for a new epoch."""
        if self.verbose:
            logging.info(f"\nInitializing epoch {self.epoch_counter} (strategy='{self.strategy}')...")
        
        combined = list(zip(self.dataloaders, self.train_list))
        self._rng.shuffle(combined)
        if combined:
            self.dataloaders, self.train_list = map(list, zip(*combined))
        
        self.protein_states = []
        
        for idx in range(len(self.dataloaders)):
            dl = self.dataloaders[idx]
            ds = dl.dataset
            protein_name = self.train_list[idx] if idx < len(self.train_list) else f"protein_{idx}"
            
            # 1. Sort dataset items into buckets
            buckets: Dict[str, List[int]] = {k: [] for k in self.SUBSET_ORDER if k != 'over_and_back'}
            first_pdb = None
            if len(ds) > 0:
                first_pdb = ds[0].get('pdb')

            for i in range(len(ds)):
                item = ds[i]
                if item.get('pdb') != first_pdb:
                    raise AssertionError(f"PDB ID mismatch in {protein_name}: item {i} has {item.get('pdb')}, expected {first_pdb}")
                subset_type = item.get('subset_type', 'single')
                if subset_type not in buckets:
                    subset_type = 'single'
                buckets[subset_type].append(i)
            
            # 2. Apply 2D Balance to configurations (e.g. doubles)
            if self.subset_balance_configs is not None:
                for subset_name, config in self.subset_balance_configs.items():
                    if subset_name in buckets and len(buckets[subset_name]) > 0:
                        if subset_name != 'over_and_back':
                            buckets[subset_name] = self._balance_subset_2d(
                                buckets[subset_name], ds, protein_name, config, subset_name
                            )
                            
            # 3. Calculate total unrestricted items and apply fraction caps
            total_unrestricted = sum(len(buckets[k]) for k in self.unrestricted_keys if k in buckets)
            
            if total_unrestricted == 0:
                raise AssertionError(f"Total unrestricted samples for {protein_name} dropped to 0. Cannot compute caps.")
            
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
                        
            # 4. Flatten completely and pre-fuse items
            protein_flat_items = []
            counts = {k: 0 for k in self.SUBSET_ORDER}
            
            for k, items in buckets.items():
                for idx in items:
                    protein_flat_items.append(ds[idx])
                    counts[k] += 1
                        
            self._rng.shuffle(protein_flat_items)
            
            state = ProteinSamplerState(
                protein_name=protein_name,
                items=protein_flat_items
            )
            self.protein_states.append(state)
            
            if self.verbose:
                logging.info(f"[{protein_name}] Flat Pool: {len(protein_flat_items)} total items | {counts}")

        # 5. Set Strategy Constraints
        if self.strategy == 'min':
            min_batches = min(len(ps.items) // self.batch_size for ps in self.protein_states)
            self.target_batches = min_batches * len(self.protein_states)
            for ps in self.protein_states:
                ps.batches_allocated = min_batches
            if self.verbose:
                logging.info(f"\nStrategy 'min': All proteins truncated to {min_batches} batches.")
                
        elif self.strategy == 'all':
            self.target_batches = sum(len(ps.items) // self.batch_size for ps in self.protein_states)
            for ps in self.protein_states:
                ps.batches_allocated = len(ps.items) // self.batch_size
            if self.verbose:
                logging.info(f"\nStrategy 'all': Target {self.target_batches} total batches.")

        self.total_batches_drawn = 0
        self.current_loader_idx = 0
        self._initialized = True
        self.epoch_counter += 1

    def __len__(self) -> int:
        return self.target_batches if self._initialized else 0
    
    def __iter__(self):
        self._initialize_epoch()
        return self
    
    def __next__(self):
        if self.total_batches_drawn >= self.target_batches:
            raise StopIteration
        
        if self.strategy == 'min':
            num_proteins = len(self.protein_states)
            for _ in range(num_proteins):
                ps = self.protein_states[self.current_loader_idx]
                self.current_loader_idx = (self.current_loader_idx + 1) % num_proteins
                
                if ps.batches_drawn < ps.batches_allocated and ps.remaining_items >= self.batch_size:
                    batch = ps.items[ps.cursor : ps.cursor + self.batch_size]
                    ps.cursor += self.batch_size
                    ps.batches_drawn += 1
                    self.total_batches_drawn += 1
                    return self.collate_fn(batch)
            raise StopIteration
            
        else: # strategy == 'all'
            available = [
                ps for ps in self.protein_states 
                if ps.batches_drawn < ps.batches_allocated and ps.remaining_items >= self.batch_size
            ]
            if not available:
                raise StopIteration
                
            ps = self._rng.choice(available)
            batch = ps.items[ps.cursor : ps.cursor + self.batch_size]
            ps.cursor += self.batch_size
            ps.batches_drawn += 1
            self.total_batches_drawn += 1
            return self.collate_fn(batch)

    def reset_epoch(self) -> None:
        self._initialize_epoch()


class ProteinCyclingDataLoader:
    """
    Simpler cycling dataloader wrapper to balance and iterate over multiple
    DataLoaders based on specified strategies.
    """
    def __init__(self, dataloaders, batch_size, train_list, collate_fn, strategy='min', positional=False):
        self.dataloaders = dataloaders
        self.batch_size = batch_size
        self.train_list = train_list
        self.num_dataloaders = len(dataloaders)
        self.current_loader_idx = 0
        self.strategy = strategy
        self.positional = positional
        self.collate_fn = collate_fn

        # Store lengths instead of computing them repeatedly
        self.dataloader_lengths = [len(dl) for dl in self.dataloaders]
        self.min_length = min(self.dataloader_lengths)
        self.max_length = max(self.dataloader_lengths)
        
        self.batches_drawn = [0] * self.num_dataloaders
        self.total_batches_drawn = 0
        if self.strategy == 'min':
            self.target_batches = self.min_length * self.num_dataloaders
        elif self.strategy == 'all':
            self.target_batches = sum(self.dataloader_lengths)
        else:
            raise AssertionError('strategy must be one of: min, repeat, all')
        self.epoch_ended = False
        
        # Keep track of iterators separately
        self.iterators = {}

        # Probability weights for selecting dataloaders
        self.weights = np.array(self.dataloader_lengths, dtype=np.float32) / sum(self.dataloader_lengths)

        logging.info("Dataloader lengths:")
        for name, length in zip(self.train_list, self.dataloader_lengths):
            logging.info(f"{name}: {length}")
        logging.info(f"Min length: {self.min_length}")
        logging.info(f"Target batches: {self.target_batches}")

    def __len__(self) -> int:
        return self.target_batches

    def reset_dataloader(self, idx: int):
        """Reset a specific dataloader."""
        if idx in self.iterators:
            del self.iterators[idx]
            
        old_dataloader = self.dataloaders[idx]
        
        generator = torch.Generator()
        
        # Create new dataloader
        self.dataloaders[idx] = DataLoader(
            old_dataloader.dataset,
            collate_fn=self.collate_fn,
            generator=generator,
            batch_size=self.batch_size,
        )

        gc.collect()

    def shuffle_all(self):
        """Reset all dataloaders."""
        logging.info('Shuffling dataloaders.')
        
        # Zip dataloaders and train_list to shuffle them in sync
        combined = list(zip(self.dataloaders, self.train_list))
        random.shuffle(combined)
        self.dataloaders, self.train_list = zip(*combined)
        
        # Convert back to lists
        self.dataloaders = list(self.dataloaders)
        self.train_list = list(self.train_list)
        
        self.dataloader_lengths = [len(dl) for dl in self.dataloaders]
        
        logging.info('Dataloader lengths:')
        for name, length in zip(self.train_list, self.dataloader_lengths):
            logging.info(f"{name}: {length}")
            
        self.weights = np.array(self.dataloader_lengths, dtype=np.float32) / sum(self.dataloader_lengths)
        for idx in tqdm(range(self.num_dataloaders)):
            self.reset_dataloader(idx)
        
        self.iterators.clear()
        
        self.total_batches_drawn = 0
        self.batches_drawn = [0] * self.num_dataloaders
        self.current_loader_idx = 0
        
        gc.collect()
        logging.info('Shuffled all dataloaders.')

    def reset_epoch(self):
        """Reset for new epoch."""
        self.epoch_ended = False
        self.shuffle_all()

    def __iter__(self):
        self.reset_epoch()
        return self

    def get_current_iterator(self):
        """Get iterator for current loader, creating if necessary and checking for exhaustion."""
        if self.current_loader_idx not in self.iterators:
            if self.strategy != 'repeat' and self.batches_drawn[self.current_loader_idx] >= len(self.dataloaders[self.current_loader_idx]):
                return None
            self.iterators[self.current_loader_idx] = iter(self.dataloaders[self.current_loader_idx])
        return self.iterators[self.current_loader_idx]

    def update_weights(self):
        """Update weights based on remaining batches."""
        remaining_batches = [max(0, len(dl) - drawn) for dl, drawn in zip(self.dataloaders, self.batches_drawn)]
        total_remaining = sum(remaining_batches)
        self.weights = np.array(remaining_batches, dtype=np.float32) / total_remaining if total_remaining > 0 else np.zeros_like(self.weights)

    def __next__(self):
        if self.total_batches_drawn >= self.target_batches:
            logging.info('Stopping due to drawing target number of batches')
            raise StopIteration
        
        while True:
            logging.info(f'Sampling from {self.train_list[self.current_loader_idx]}')

            if self.strategy == 'all':
                self.update_weights()
                self.current_loader_idx = np.random.choice(self.num_dataloaders, p=self.weights)

            current_iterator = self.get_current_iterator()
            
            if current_iterator is None:
                if self.strategy != 'all':
                    self.current_loader_idx = (self.current_loader_idx + 1) % self.num_dataloaders
                continue

            try:
                batch = next(current_iterator)
            except StopIteration:
                logging.warning(f"Loader {self.train_list[self.current_loader_idx]} unexpectedly exhausted.")
                self.batches_drawn[self.current_loader_idx] = self.dataloader_lengths[self.current_loader_idx]
                continue
                
            self.batches_drawn[self.current_loader_idx] += 1
            self.total_batches_drawn += 1
            if self.strategy != 'all':
                self.current_loader_idx = (self.current_loader_idx + 1) % self.num_dataloaders

            if len(batch['ddG']) < self.batch_size:
                logging.info(f'Smaller than expected batch of size {len(batch["ddG"])} / {self.batch_size} detected!')
                logging.info(f"Loader: {self.train_list[self.current_loader_idx]} has been exhausted")
                
            return batch
        

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