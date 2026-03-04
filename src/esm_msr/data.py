import os
from tqdm import tqdm
import gc
import pickle
import random
from collections import defaultdict
import math
from dataclasses import dataclass, field

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from esm_msr.utils import *


class ProteinStructureMutationEpistasisDatasetChainRule(torch.utils.data.IterableDataset):
    """
    Originals-only base like your prior class, but also emits synthesized mutant-context singles from doubles.
    - Flag `include_doubles`: whether to keep original double examples.
    - Flag `enable_mutctx_masking`: single switch controlling whether structure masking is applied anywhere:
        * If a backbone implies a mutated residue (your prior logic), mask there when True.
        * For synthesized mutant-context singles, also mask at the partner mutation site when True (if we are not using modeled coords).
    - Flag `use_mut_structs`: when True, try to load modeled mutant-context structures from `mut_structs_root`.
    - Flag `include_reversions`: when True, add antisymmetric reversion singles (mt->wt) on the modeled mutant structure with negated ddG.
    """
    def __init__(
        self,
        dms_df,
        tokenizer,
        dms_name,
        device: str = 'cpu',
        score_name: str = 'ddG_ML',
        allow_multi: bool = True,
        path: Optional[str] = None,
        generate: bool = False,
        incl_destab: bool = True,
        domainome: bool = False,
        *,
        structure_encoder = None,
        incl_doubles: bool = True,
        incl_mutctx: bool = False,
        incl_reversions: bool = False,
        enable_mutctx_masking: bool = False,
        use_mut_structs: bool = False,             # try to load modeled mutant context structures
        mut_structs_root: Optional[str] = None,    # root dir for modeled structures
    ):
        self.score_name = score_name
        self.dms_name = dms_name
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.device = device
        self.allow_multi = allow_multi
        self.incl_destab = incl_destab
        self.domainome = domainome

        # Flags
        self.include_doubles = incl_doubles
        self.include_mut_context = incl_mutctx
        self.include_reversions = incl_reversions
        self.enable_mutctx_masking = enable_mutctx_masking
        self.use_mut_structs = use_mut_structs
        self.mut_structs_root = mut_structs_root

        if self.use_mut_structs:
            assert self.mut_structs_root

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        dms_df['ground_truth'] = dms_df['ddG']  # deprecated alias

        if path is None:
            path = '.'
        self.cache_path = os.path.join(
            path, 'cache',
            f"{self.dms_name}_{self.score_name}_doubles{int(self.include_doubles)}_mutctx{int(self.include_mut_context)}_rev{int(self.include_reversions)}"
                +f"_mutS{int(self.use_mut_structs and (self.include_mut_context or self.include_doubles))}_Smasked{int(self.enable_mutctx_masking)}.pkl"
        )
        print(self.cache_path)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.data: List[Dict[str, Any]] = []
        self._mutant_struct_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if generate or not os.path.exists(self.cache_path):
            print(f"Generating and caching data for {self.dms_name} (originals + synthesized mutant-context singles)")
            self.data = self.generate_data(dms_df)
            self._save_data_to_cache()
        else:
            print(f"Loading cached data for {self.dms_name} (originals + synthesized mutant-context singles)")
            self.load_data_from_cache()

        self.rng = np.random.default_rng()

    # ---------------------------
    # Core loading & generation
    # ---------------------------
    def generate_data(self, dms_df):
        if self.score_name == 'ddG_ML':
            # get all instances of domains which may have mutated backbone suffixes
            # warning: can easily match substrings unintentionally
            df = dms_df.loc[dms_df['code'].str.contains(self.dms_name, regex=False)] if not self.domainome else dms_df.loc[dms_df['code'] == self.dms_name]
            # but don't get v2_ versions, they are too dissimilar
            if not self.dms_name.startswith('v2_'):
                df = df.loc[~df['code'].str.startswith('v2_')]
            info = df.head(1)
            df = df.copy()
            df['mutated_sequence'] = df['aa_seq']
            # pdb_file should be the same for all items with the same dms_name
            try:
                pdb_file = info['pdb_file'].item()
            except Exception as e:
                print(info)
                raise e
            data = self._load_data_chainrule(df, pdb_file, 'A', is_predicted=True)
        else:
            print('Not using predicted structures!')
            data = []
            for (code, chain), df_sub in dms_df.groupby(['code', 'chain']):
                df_sub = df_sub.copy()
                df_sub['mutated_sequence'] = df_sub['mut_seq']
                pdb_file = df_sub['pdb_file'].head(1).item()
                data.extend(self._load_data_chainrule(df_sub, pdb_file, chain, incl_chain_in_code=True, is_predicted=False))
        return data

    def _save_data_to_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        indices = list(range(len(self.data)))
        self.rng.shuffle(indices)
        for idx in indices:
            yield self.data[idx]

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _to_three_letter(aa: str) -> str:
        m = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }
        return m[aa.upper()]

    def _try_load_modeled_context(
        self,
        base_code: str,
        chain: str,
        partner_wt: str,
        partner_pos: int,
        partner_mut: str,
        expected_seq: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Attempt to load a modeled PDB for the mutant protein/domain (with caching).

        Returns: (coords_padded, structure_tokens_padded) if successful and sequence matches expected_seq;
                None otherwise.
        """
        if not self.mut_structs_root:
            return None

        # Build path
        dir_lvl1 = f"{base_code}"
        dir_lvl2 = "pdb_models"
        fname = f"{chain}[{partner_wt}{partner_pos}{partner_mut}].pdb"
        pdb_path = os.path.join(self.mut_structs_root, dir_lvl1, dir_lvl2, fname)

        # Check cache first
        cache_key = pdb_path
        if cache_key in self._mutant_struct_cache:
            #print(f"[mut-struct] Loaded from cache: {pdb_path}")
            cached_seq_tokens, cached_coords, cached_plddt, cached_struct_tokens, cached_residue_index = self._mutant_struct_cache[cache_key]
            # Verify sequence still matches
            cached_seq = self.tokenizer.decode(cached_seq_tokens if isinstance(cached_seq_tokens, list) else cached_seq_tokens.tolist())
            cached_seq = ''.join(cached_seq.split(' ')[1:-1])
            if cached_seq == expected_seq:
                return cached_seq_tokens, cached_coords, cached_plddt, cached_struct_tokens, cached_residue_index
            else:
                # Sequence mismatch, remove from cache
                del self._mutant_struct_cache[cache_key]
                print('Unexpected sequence!')
                print(cached_seq)
                print(expected_seq)
                return None

        if not os.path.exists(pdb_path):
            print(f"[mut-struct] Failed to load {pdb_path}: file not found.")
            return None  

        try:
            mut_chain = ProteinChain.from_pdb(pdb_path, chain, is_predicted=True)
        except Exception as e:
            print(f"[mut-struct] Failed to load {pdb_path}: {e}; falling back.")
            return None

        seq_loaded = mut_chain.sequence
        if seq_loaded != expected_seq:
            print(f"[mut-struct] Sequence mismatch for {pdb_path}; falling back.")
            return None

        try:
            coords_m, plddt_m, residue_index_m = mut_chain.to_structure_encoder_inputs()
            # pre-compute the structure tokens
            if self.structure_encoder:
                _, structure_tokens_m = self.structure_encoder.encode(coords_m, residue_index=residue_index_m)
                structure_tokens_m = F.pad(structure_tokens_m, (1, 1), value=0)
                structure_tokens_m[:, 0] = C.STRUCTURE_BOS_TOKEN
                structure_tokens_m[:, -1] = C.STRUCTURE_EOS_TOKEN
            # leave a placeholder indicating that the tokens should be computed during inference for tuning
            else:
                structure_tokens_m = torch.Tensor([-1])
        except Exception as e:
            print(f"[mut-struct] Encoding failed for {pdb_path}: {e}; falling back.")
            return None

        coords_m = F.pad(coords_m, (0, 0, 0, 0, 1, 1), value=torch.inf)
        plddt_m = F.pad(plddt_m, (1, 1), value=0)
        sequence_tokens_m = self.tokenizer.encode(seq_loaded)

        # Store in cache
        self._mutant_struct_cache[cache_key] = (sequence_tokens_m, coords_m, plddt_m, structure_tokens_m, residue_index_m)

        return sequence_tokens_m, coords_m, plddt_m, structure_tokens_m, residue_index_m

    # ---------------------------
    # Internal helpers (core loader)
    # ---------------------------
    def _load_data_chainrule(self, df, pdb_file, chain, is_predicted=False, incl_chain_in_code: bool = False) -> List[Dict[str, Any]]:
        
        # Clear the mutant structure cache for this new protein
        self._mutant_struct_cache.clear()
        print(f'loading data for {pdb_file} (mutant structure cache cleared)')

        data: List[Dict[str, Any]] = []

        if 'mut_structure' not in df.columns:
            df['mut_structure'] = df['pdb_file']
        df.loc[df['mut_structure'].isna(), 'mut_structure'] = df.loc[df['mut_structure'].isna(), 'pdb_file']

        for backbone, group in df.groupby('mut_structure'):
            assert len(group['code'].unique()) == 1
            base_code = group['code'].head(1).item()
            code = base_code
            if incl_chain_in_code:
                code += chain

            protein_chain = ProteinChain.from_pdb(
                group['pdb_file'].head(1).item() if not backbone.endswith('.pdb') else backbone,
                chain,
                is_predicted=is_predicted,
            )
            coords_b, plddt_b, residue_index_b = protein_chain.to_structure_encoder_inputs()
            if self.structure_encoder:
                _, structure_tokens_b = self.structure_encoder.encode(coords_b, residue_index=residue_index_b)
                structure_tokens_b = F.pad(structure_tokens_b, (1, 1), value=0)
                structure_tokens_b[:, 0] = C.STRUCTURE_BOS_TOKEN
                structure_tokens_b[:, -1] = C.STRUCTURE_EOS_TOKEN
            else:
                structure_tokens_b = torch.Tensor([-1])

            coords_b = F.pad(coords_b, (0, 0, 0, 0, 1, 1), value=torch.inf)
            plddt_b = F.pad(plddt_b, (1, 1), value=0)
            corrected_seq = protein_chain.sequence

            mutated_backbone_pos: Optional[int] = None
            if not backbone.endswith('.pdb'):
                if not self.incl_destab:
                    print('Skipped destabilized backbone!', backbone)
                    continue
                mutated_backbone_pos = int(backbone[1:-1])
                wt = backbone[0]
                mt = backbone[-1]
                assert corrected_seq[mutated_backbone_pos-1] == wt
                assert mt in 'ACDEFGHIKLMNPQRSTVWY'
                corrected_seq = list(corrected_seq)
                corrected_seq[mutated_backbone_pos-1] = mt
                print(f"Backbone implies mutated residue at {mutated_backbone_pos}: {wt}->{mt}; preserved in corrected_seq.")
                if self.enable_mutctx_masking:
                    structure_tokens_b[:, mutated_backbone_pos] = C.STRUCTURE_MASK_TOKEN
                    coords_b[:, mutated_backbone_pos, :, :] = np.nan
                    plddt_b[:, mutated_backbone_pos] = 0
                    print(f"Also masking the coordinates and structure tokens at {mutated_backbone_pos}")
                corrected_seq = ''.join(corrected_seq)

            sequence_tokens_orig = self.tokenizer.encode(corrected_seq)

            # --- First pass: collect singles for dddG composition ---
            single_map: Dict[Tuple[int, str], float] = {}  # (pos, mt) -> ddG
            position_to_wt_map: Dict[int, str] = {}

            parsed_rows: List[Tuple[List[Tuple[str,int,str]], float, Any]] = []
            for uid, row in tqdm(group.iterrows()):
                if 'mut_type' not in df.columns:
                    mut_seq = row['mutated_sequence']
                    offset, alignment, _ = custom_end_gap_alignment(mut_seq, protein_chain.sequence)
                    muts = determine_diffs(mut_seq[offset:len(corrected_seq)+offset], corrected_seq)
                else:
                    muts = []
                    for mut in row['mut_type'].split(':'):
                        pos = int(mut[1:-1])
                        wt = mut[0]
                        mt = mut[-1]
                        if corrected_seq[pos-1] != wt:
                            print(f'Warning! Provided wt {wt} != {corrected_seq[pos-1]} in {pdb_file}_{backbone}; ignoring this mutation.')
                            continue
                        muts.append((wt, pos, mt))

                ddG_val = float(row['ddG'])
                parsed_rows.append((muts, ddG_val, row))

                for wt0, pos0, mt0 in muts:
                    position_to_wt_map.setdefault(pos0, wt0)
                if len(muts) == 1:
                    (_, posS, mtS) = muts[0]
                    single_map[(posS, mtS)] = ddG_val

            # --- Second pass: originals + synthesized context-singles (+ optional reversions)
            for muts, ddG_val, row in parsed_rows:
                # (A) originals
                dddG_val = np.nan
                if len(muts) == 2:
                    (wtA, posA, mtA), (wtB, posB, mtB) = muts
                    has_A = (posA, mtA) in single_map
                    has_B = (posB, mtB) in single_map
                    if has_A and has_B:
                        dddG_val = ddG_val - single_map[(posA, mtA)] - single_map[(posB, mtB)]

                if len(muts) == 1:
                    data.append(self._create_data_item(
                        mutations=muts,
                        ddG=ddG_val,
                        dddG=dddG_val,
                        code=code,
                        corrected_seq=corrected_seq,
                        sequence_tokens_orig=sequence_tokens_orig,
                        coords=coords_b,
                        plddt=plddt_b,
                        structure_tokens=structure_tokens_b,
                        residue_index=residue_index_b,
                        subset_type='single'
                    ))
                    if self.include_reversions:
                        wt, pos, mut = muts[0]
                        # Reversion mutation is defined relative to wt_seq
                        rev_mut = [(mut, pos, wt)]
                        rev_ddG = -float(row['ddG'])
                        mut_seq = list(corrected_seq)
                        mut_seq[pos-1] = mut
                        mut_seq = ''.join(mut_seq)
                        sequence_tokens_m = self.tokenizer.encode(mut_seq)
                        coords_m, plddt_m, structure_tokens_m, residue_index_m = coords_b, plddt_b, structure_tokens_b, residue_index_b
                        if self.use_mut_structs:
                            maybe = self._try_load_modeled_context(base_code, chain, wt, pos, mut, expected_seq=mut_seq)
                            if maybe is not None:
                                sequence_tokens_m, coords_m, plddt_m, structure_tokens_m, residue_index_m = maybe
                        data.append(self._create_data_item(
                            mutations=rev_mut,
                            ddG=rev_ddG,
                            dddG=np.nan,
                            code=code,
                            corrected_seq=mut_seq,             # baseline sequence is the mutant A context
                            sequence_tokens_orig=sequence_tokens_m,
                            coords=coords_m,                 # use modeled A structure
                            plddt=plddt_m,
                            structure_tokens=structure_tokens_m,
                            residue_index=residue_index_m,
                            subset_type='reversion'
                        ))

                if len(muts) == 2 and self.include_doubles:
                    data.append(self._create_data_item(
                        mutations=muts,
                        ddG=ddG_val,
                        dddG=dddG_val,
                        code=code,
                        corrected_seq=corrected_seq,
                        sequence_tokens_orig=sequence_tokens_orig,
                        coords=coords_b,
                        plddt=plddt_b,
                        structure_tokens=structure_tokens_b,
                        residue_index=residue_index_b,
                        subset_type='double'
                    ))

                # (B) synthesized mutant-context singles from doubles
                if len(muts) == 2 and self.include_mut_context:
                    (wtA, posA, mtA), (wtB, posB, mtB) = muts
                    has_A = (posA, mtA) in single_map
                    has_B = (posB, mtB) in single_map
                    if has_A and has_B:
                        t_B_given_A = ddG_val - single_map[(posA, mtA)]
                        t_A_given_B = ddG_val - single_map[(posB, mtB)]

                        # Validate WT correctness at positions
                        assert corrected_seq[posA-1] == wtA
                        assert corrected_seq[posB-1] == wtB

                        # --- "B|A": context with A present
                        ctx_seq_A_list = list(corrected_seq)
                        ctx_seq_A_list[posA-1] = mtA
                        ctx_seq_A = ''.join(ctx_seq_A_list)
                        seq_tokens_ctx_A = self.tokenizer.encode(ctx_seq_A)

                        coords_ctx_A = coords_b.clone()
                        plddt_ctx_A = plddt_b.clone()
                        struct_ctx_A = structure_tokens_b.clone()
                        residue_index_ctx_A = residue_index_b.clone()
                        used_modeled_A = False
                        if self.use_mut_structs and self.mut_structs_root:
                            maybe = self._try_load_modeled_context(
                                base_code=base_code, chain=chain,
                                partner_wt=wtA, partner_pos=posA, partner_mut=mtA,
                                expected_seq=ctx_seq_A
                            )
                            if maybe is not None:
                                _, coords_ctx_A, plddt_ctx_A, struct_ctx_A, residue_index_ctx_A = maybe
                                used_modeled_A = True

                        if not used_modeled_A and self.enable_mutctx_masking:
                            struct_ctx_A[:, posA] = C.STRUCTURE_MASK_TOKEN
                            coords_ctx_A[:, posA, :, :] = np.nan

                        data.append(self._create_data_item(
                            mutations=[(wtB, posB, mtB)],
                            ddG=float(t_B_given_A),
                            dddG=np.nan,
                            code=code,
                            corrected_seq=ctx_seq_A,
                            sequence_tokens_orig=seq_tokens_ctx_A,
                            coords=coords_ctx_A,
                            plddt=plddt_ctx_A,
                            structure_tokens=struct_ctx_A,
                            residue_index=residue_index_ctx_A,
                            subset_type='mut_ctx'
                        ))

                        # --- "A|B": context with B present
                        ctx_seq_B_list = list(corrected_seq)
                        ctx_seq_B_list[posB-1] = mtB
                        ctx_seq_B = ''.join(ctx_seq_B_list)
                        seq_tokens_ctx_B = self.tokenizer.encode(ctx_seq_B)

                        coords_ctx_B = coords_b.clone()
                        plddt_ctx_B = plddt_b.clone()
                        struct_ctx_B = structure_tokens_b.clone()
                        residue_index_ctx_B = residue_index_b.clone()
                        used_modeled_B = False
                        if self.use_mut_structs and self.mut_structs_root:
                            maybe = self._try_load_modeled_context(
                                base_code=base_code, chain=chain,
                                partner_wt=wtB, partner_pos=posB, partner_mut=mtB,
                                expected_seq=ctx_seq_B
                            )
                            if maybe is not None:
                                _, coords_ctx_B, plddt_ctx_B, struct_ctx_B, residue_index_ctx_B = maybe
                                used_modeled_B = True

                        if not used_modeled_B and self.enable_mutctx_masking:
                            struct_ctx_B[:, posB] = C.STRUCTURE_MASK_TOKEN
                            coords_ctx_B[:, posB, :, :] = np.nan

                        data.append(self._create_data_item(
                            mutations=[(wtA, posA, mtA)],
                            ddG=float(t_A_given_B),
                            dddG=np.nan,
                            code=code,
                            corrected_seq=ctx_seq_B,
                            sequence_tokens_orig=seq_tokens_ctx_B,
                            coords=coords_ctx_B,
                            plddt=plddt_ctx_B,
                            structure_tokens=struct_ctx_B,
                            residue_index=residue_index_ctx_B,
                            subset_type='mut_ctx'
                        ))

        print(f"Mutant structure cache for {pdb_file}: {len(self._mutant_struct_cache)} unique structures loaded")

        return data

    def _create_data_item(
        self,
        mutations: List[Tuple[str, int, str]],
        ddG: float,
        dddG: float,
        code: str,
        corrected_seq: str,
        sequence_tokens_orig: List[int],
        coords: torch.Tensor,
        plddt: torch.Tensor,
        structure_tokens: torch.Tensor,
        residue_index: torch.Tensor,
        subset_type: str
    ) -> Dict[str, Any]:
        """Create a single data item."""
        #positions = tuple(sorted({pos for (_, pos, _) in mutations}))
        item: Dict[str, Any] = {
            'pdb': code,
            'mutations': mutations,
            #'positions': positions,
            'coords_orig': coords.clone().cpu().numpy(),
            'sequence_tokens_orig': np.array(sequence_tokens_orig, dtype=np.int64),
            'structure_tokens_orig': structure_tokens.clone().cpu().numpy(),
            'residue_index': residue_index.clone().cpu().numpy(),
            'plddt': plddt.clone().cpu().numpy(),
            'ddG': float(ddG),
            'dddG': float(dddG) if dddG == dddG else np.nan,
            'ground_truth': float(ddG),
            'subset_type': subset_type
        }
        return item


def collate_fn_chainrule(batch):
    """
    Collate for originals + synthesized mutant-context singles (no masking here).
    Mirrors your original collate_fn; adds passthrough of new metadata fields.
    """
    try:
        B = len(batch)

        pdb = [item['pdb'] for item in batch]
        mutations = [item['mutations'] for item in batch]
        #positions = [tuple(item.get('positions', ())) for item in batch]

        # --- Labels ---
        if 'ddG' in batch[0]:
            ddG = torch.tensor([float(item['ddG']) for item in batch], dtype=torch.float)
        else:
            ddG = torch.tensor([float(item['ground_truth']) for item in batch], dtype=torch.float)

        if 'dddG' in batch[0]:
            dddG_np = np.array([item['dddG'] for item in batch], dtype=np.float32)
            dddG = torch.from_numpy(dddG_np)  # may contain NaNs
        else:
            dddG = torch.full((B,), float('nan'), dtype=torch.float)

        # sequence tokens
        seq_list = [item['sequence_tokens_orig'] for item in batch]
        try:
            seq_stack = torch.from_numpy(np.stack(seq_list, axis=0)).to(torch.long)
        except Exception as e:
            print('Are sequences the same length?')
            print(pdb)
            print(mutations)
            print(seq_list)
            print([seq.shape for seq in seq_list])
            raise e

        # coords
        crd_list = [item['coords_orig'] for item in batch]
        crd_stack_np = np.stack(crd_list, axis=0)
        if crd_stack_np.dtype != np.float32:
            crd_stack_np = crd_stack_np.astype(np.float32, copy=False)
        crd_stack = torch.from_numpy(crd_stack_np)

        # structure tokens
        str_list = [item['structure_tokens_orig'] for item in batch]
        str_stack_np = np.stack(str_list, axis=0)
        str_stack = torch.from_numpy(str_stack_np).to(torch.long)

        try:
            # residue_index
            ri_list = [item['residue_index'] for item in batch]
            ri_stack_np = np.stack(ri_list, axis=0)
            ri_stack = torch.from_numpy(ri_stack_np).to(torch.long)
        except:
            ri_stack = None       

        subset_type = [item['subset_type'] for item in batch]
        plddt = [item['plddt'] for item in batch]

        collated = {
            'pdb': pdb,
            'mutations': mutations,
            'ddG': ddG,                             # [B]
            'dddG': dddG,                           # [B] (NaN if N/A)
            'sequence_tokens_orig': seq_stack,      # [B, L]
            'coords_orig': crd_stack,               # [B, ...]
            'plddt': plddt,
            'structure_tokens_orig': str_stack,     # [B, K, L] or [B, L]
            'residue_index': ri_stack,
            'ground_truth': ddG,
            'subset_type': subset_type
        }
        return collated

    finally:
        del batch
        gc.collect()


class ProteinStructureMutationEpistasisDatasetChainRuleIndel(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dms_df,
        tokenizer,
        dms_name,
        device: str = 'cpu',
        score_name: str = 'ddG_ML',
        allow_multi: bool = True,
        path: Optional[str] = None,
        generate: bool = False,
        incl_destab: bool = True,
        domainome: bool = False,
        *,
        structure_encoder = None,
        incl_doubles: bool = True,
        incl_mutctx: bool = False,
        incl_reversions: bool = False,
        enable_mutctx_masking: bool = False,
        use_mut_structs: bool = False,
        mut_structs_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.score_name = score_name
        self.dms_name = dms_name
        self.incl_destab = incl_destab
        self.domainome = domainome
        self.include_doubles = incl_doubles
        self.include_mut_context = incl_mutctx
        self.include_reversions = incl_reversions
        self.enable_mutctx_masking = enable_mutctx_masking
        self.use_mut_structs = use_mut_structs
        self.mut_structs_root = mut_structs_root

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        
        if path is None: path = '.'
        self.cache_path = os.path.join(path, 'cache', f"{self.dms_name}_paired_v1.pkl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.data: List[Dict[str, Any]] = []
        self._mutant_struct_cache = {}

        if generate or not os.path.exists(self.cache_path):
            self.data = self.generate_data(dms_df, path)
            self._save_data_to_cache()
        else:
            self.load_data_from_cache()
        self.rng = np.random.default_rng()

    def _save_data_to_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)

    # ---------------------------
    # Core loading & generation
    # ---------------------------
    def generate_data(self, dms_df, path):
        if self.score_name == 'ddG_ML':
            df = dms_df.loc[dms_df['code'].str.contains(self.dms_name, regex=False)] if not self.domainome else dms_df.loc[dms_df['code'] == self.dms_name]
            if not self.dms_name.startswith('v2_'):
                df = df.loc[~df['code'].str.startswith('v2_')]
            info = df.head(1)
            df = df.copy()
            df['mutated_sequence'] = df['aa_seq']
            try:
                pdb_file = info['pdb_file'].item()
            except Exception as e:
                print(info)
                raise e
            data = self._load_data_chainrule(df, pdb_file, 'A', is_predicted=False)
        else:
            print('Not using predicted structures!')
            data = []
            for (code, chain), df_sub in dms_df.groupby(['code', 'chain']):
                df_sub = df_sub.copy()
                df_sub['mutated_sequence'] = df_sub['mut_seq']
                pdb_file = df_sub['pdb_file'].head(1).item()
                data.extend(self._load_data_chainrule(df_sub, pdb_file, chain, incl_chain_in_code=True))
        return data

    def __len__(self): return len(self.data)
    def __iter__(self):
        indices = list(range(len(self.data)))
        self.rng.shuffle(indices)
        for idx in indices: yield self.data[idx]

    def _apply_surgery(self, base_seq, base_coords, base_struct, base_ri, mutations):
        """Generates MT sequence and tensors from a baseline with masking for indels."""
        seq_list = list(base_seq)
        coords, struct, ri = base_coords.clone(), base_struct.clone(), base_ri.clone()
        is_indel = False
        
        # Process mutations from back to front to preserve indices
        for wt, pos, mt in sorted(mutations, key=lambda x: x[1], reverse=True):
            if mt == '-': # Deletion
                is_indel = True
                seq_list.pop(pos-1)
                
                # Create the gap in tensors
                coords = torch.cat([coords[:, :pos], coords[:, pos+1:]], dim=1)
                struct = torch.cat([struct[:, :pos], struct[:, pos+1:]], dim=1)
                ri = torch.cat([ri[:pos-1], ri[pos:]], dim=0)
                
                # MASK NEIGHBORS: The residue at `pos-1` and `pos` in the new tensor
                # are the ones that were stitched together. Mask them to avoid
                # penalizing the "broken" geometry.
                # pos is 1-based. indices: 0..pos-1 (Left), pos (Right)
                # Note: `struct` is padded (BOS at 0), so `pos` index aligns with residue `pos`
                #struct[:, pos-1] = C.STRUCTURE_MASK_TOKEN
                #struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #coords[:, pos-1, :, :] = float('nan')
                #coords[:, pos, :, :] = float('nan')

            elif wt == '-': # Insertion
                is_indel = True
                
                # MASK LEFT ANCHOR (at index `pos`)
                #struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #coords[:, pos, :, :] = float('nan')

                # Insert the new residues
                # Note: We use NaN for coords so `build_affine3d` treats them as missing
                for aa in reversed(mt):
                    seq_list.insert(pos, aa)
                    coords = torch.cat([coords[:, :pos+1], torch.full((1, 1, 37, 3), float("nan")), coords[:, pos+1:]], dim=1)
                    struct = torch.cat([struct[:, :pos+1], torch.tensor([[C.STRUCTURE_MASK_TOKEN]]), struct[:, pos+1:]], dim=1)
                    ri = torch.cat([ri[:pos], ri[pos-1:pos], ri[pos:]], dim=0)
                
                # MASK RIGHT ANCHOR
                # The right anchor was originally at `pos+1`. 
                # After inserting `len(mt)`, it is at `pos + 1 + len(mt)`.
                # (Loop inserts in reverse, pushing right anchor out)
                #idx_right = pos + len(mt) + 1
                #if idx_right < struct.shape[1]: # Bounds check just in case
                #    struct[:, idx_right] = C.STRUCTURE_MASK_TOKEN
                #    coords[:, idx_right, :, :] = float('nan')

            else: # Substitution
                seq_list[pos-1] = mt
                #if self.enable_mutctx_masking:
                #    struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #    coords[:, pos, :, :] = float('nan')
        
        new_seq = "".join(seq_list)
        return new_seq, torch.tensor(self.tokenizer.encode(new_seq)), coords, struct, ri, is_indel

    def _load_data_chainrule(self, df, pdb_file, chain, is_predicted=False, incl_chain_in_code=False):
        self._mutant_struct_cache.clear()
        ref_chain = ProteinChain.from_pdb(pdb_file, chain, is_predicted=is_predicted)
        ref_coords, _, ref_ri = ref_chain.to_structure_encoder_inputs()
        
        # Prepare Structure Tokens
        # If encoder is None, we create dummy tokens (-1) of the correct shape (1, L)
        # so that surgery (slicing) works correctly.
        if self.structure_encoder:
            _, ref_struct = self.structure_encoder.encode(ref_coords, residue_index=ref_ri)
        else:
            # Create dummy tokens of shape (1, L) filled with -1
            ref_struct = torch.full((1, ref_coords.shape[1]), -1, dtype=torch.long)

        # Pad Tensors (BOS/EOS)
        ref_struct = F.pad(ref_struct, (1, 1), value=0 if self.structure_encoder else -1)
        if self.structure_encoder:
            ref_struct[:, 0] = C.STRUCTURE_BOS_TOKEN
            ref_struct[:, -1] = C.STRUCTURE_EOS_TOKEN
        
        ref_coords = F.pad(ref_coords, (0, 0, 0, 0, 1, 1), value=float("inf"))
        
        data = []
        if 'mut_structure' not in df.columns: df['mut_structure'] = df['pdb_file']
        df['mut_structure'] = df['mut_structure'].fillna(df['pdb_file'])
        
        for backbone, group in df.groupby('mut_structure'):
            code = group['code'].head(1).item() + (chain if incl_chain_in_code else "")
            b_chain = ProteinChain.from_pdb(group['pdb_file'].head(1).item() if not backbone.endswith('.pdb') else backbone, chain)
            b_coords, _, b_ri = b_chain.to_structure_encoder_inputs()
            
            # Similar handling for backbone structure tokens
            if self.structure_encoder:
                _, b_struct = self.structure_encoder.encode(b_coords, residue_index=b_ri)
            else:
                b_struct = torch.full((1, b_coords.shape[1]), -1, dtype=torch.long)
            
            b_struct = F.pad(b_struct, (1, 1), value=0 if self.structure_encoder else -1)
            if self.structure_encoder:
                b_struct[:, 0] = C.STRUCTURE_BOS_TOKEN
                b_struct[:, -1] = C.STRUCTURE_EOS_TOKEN
            else:
                # Ensure dummy tokens don't look like valid BOS/EOS to prevent confusion if not handled downstream,
                # but -1 is generally safe.
                pass 

            b_coords = F.pad(b_coords, (0, 0, 0, 0, 1, 1), value=float("inf"))
            b_seq = b_chain.sequence

            # First Pass: Map singles for dddG
            single_map = {}
            parsed = []
            for _, row in group.iterrows():
                if 'mut_type' not in df.columns:
                    offset, alignment, _ = custom_end_gap_alignment(row['mut_seq'], b_seq)
                    muts = determine_diffs(row['mut_seq'][offset:len(b_seq)+offset], b_seq)
                else:
                    muts = []
                    for m in row['mut_type'].split(':'):
                        if m.endswith('-'): muts.append((m[0], int(m[1:-1]), '-'))
                        elif m.startswith('-'):
                            p = int(re.search(r'\d+', m).group())
                            muts.append(('-', p, re.sub(r'[-\d]+', '', m)))
                        else: muts.append((m[0], int(m[1:-1]), m[-1]))
                parsed.append((muts, float(row['ddG'])))
                if len(muts) == 1: single_map[(muts[0][1], muts[0][2])] = float(row['ddG'])

            for muts, ddG in parsed:
                dddG = ddG - single_map.get((muts[0][1], muts[0][2]), 0) - single_map.get((muts[1][1], muts[1][2]), 0) if len(muts)==2 else np.nan
                
                # (A) Originals
                if len(muts) == 1:
                    mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, muts)
                    data.append(self._create_paired_item(code, muts, ddG, dddG, is_ind, 'single', 
                                                       (torch.tensor(self.tokenizer.encode(b_seq)), b_coords, b_struct, b_ri),
                                                       (mt_tok, mt_crd, mt_st, mt_ri)))
                    
                # (A) Originals
                if len(muts) == 2 and self.include_doubles:
                    mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, muts)
                    data.append(self._create_paired_item(code, muts, ddG, dddG, is_ind, 'double',
                                                       (torch.tensor(self.tokenizer.encode(b_seq)), b_coords, b_struct, b_ri),
                                                       (mt_tok, mt_crd, mt_st, mt_ri)))

                # (B) Context Singles (B|A)
                if len(muts) == 2 and self.include_mut_context:
                    for i, j in [(0, 1), (1, 0)]:
                        mA, mB = muts[i], muts[j]
                        if (mA[1], mA[2]) in single_map:
                            # Baseline is context A
                            a_seq, a_tok, a_crd, a_st, a_ri, _ = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, [mA])
                            # Mutant is A + B
                            mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(a_seq, a_crd, a_st, a_ri, [mB])
                            data.append(self._create_paired_item(code, [mB], ddG - single_map[(mA[1], mA[2])], np.nan, is_ind, 'mut_ctx',
                                                               (a_tok, a_crd, a_st, a_ri), (mt_tok, mt_crd, mt_st, mt_ri)))
        return data

    def _create_paired_item(self, code, muts, ddG, dddG, is_indel, subset, wt_pack, mt_pack):
        return {
            'pdb': code, 'mutations': muts, 'ddG': ddG, 'dddG': dddG, 'is_indel': is_indel, 'subset_type': subset,
            'wt_tokens': wt_pack[0].numpy(), 'wt_coords': wt_pack[1].numpy(), 'wt_struct': wt_pack[2].numpy(), 'wt_ri': wt_pack[3].numpy(),
            'mt_tokens': mt_pack[0].numpy(), 'mt_coords': mt_pack[1].numpy(), 'mt_struct': mt_pack[2].numpy(), 'mt_ri': mt_pack[3].numpy()
        }

def collate_fn_chainrule_indel(batch):
    def pad_key(key, pad_val):
        tensors = []
        for item in batch:
            val = item[key]
            # Ensure tensor
            t = torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.as_tensor(val)
            
            # Squeeze leading singleton dim from Dataset (e.g. [1, L] -> [L])
            if t.ndim > 1 and t.shape[0] == 1:
                t = t.squeeze(0)
            tensors.append(t)
            
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)

    # Mutant Tensors
    seq_mut = pad_key('mt_tokens', C.SEQUENCE_PAD_TOKEN)
    crd_mut = pad_key('mt_coords', float("nan"))
    str_mut = pad_key('mt_struct', C.STRUCTURE_PAD_TOKEN)
    ri_mut = pad_key('mt_ri', 0)

    # Wild-Type Tensors
    seq_wt = pad_key('wt_tokens', C.SEQUENCE_PAD_TOKEN)
    crd_wt = pad_key('wt_coords', float("nan"))
    str_wt = pad_key('wt_struct', C.STRUCTURE_PAD_TOKEN)
    ri_wt = pad_key('wt_ri', 0)

    # Handle dddG
    if 'dddG' in batch[0]:
        dddG_np = np.array([item['dddG'] for item in batch], dtype=np.float32)
        dddG = torch.from_numpy(dddG_np)
    else:
        dddG = torch.full((len(batch),), float('nan'), dtype=torch.float)

    return {
        'pdb': [i['pdb'] for i in batch],
        'mutations': [i['mutations'] for i in batch],
        'is_indel': torch.tensor([i['is_indel'] for i in batch], dtype=torch.bool),
        'ddG': torch.tensor([i['ddG'] for i in batch], dtype=torch.float),
        'dddG': dddG,
        'sequence_tokens_mut': seq_mut, 
        'coords_mut': crd_mut, 
        'structure_tokens_mut': str_mut, 
        'residue_index_mut': ri_mut,
        'sequence_tokens_wt': seq_wt, 
        'coords_wt': crd_wt, 
        'structure_tokens_wt': str_wt, 
        'residue_index_wt': ri_wt,
        'subset_type': [i['subset_type'] for i in batch]
    }


class ProteinStructureMutationEpistasisDatasetChainRuleAbsolute(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dms_df,
        tokenizer,
        dms_name,
        device: str = 'cpu',
        score_name: str = 'dG_ML',
        allow_multi: bool = True,
        path: Optional[str] = None,
        generate: bool = False,
        incl_destab: bool = True,
        domainome: bool = False,
        *,
        structure_encoder = None,
        incl_doubles: bool = True,
        incl_mutctx: bool = False,
        incl_reversions: bool = False,
        enable_mutctx_masking: bool = False,
        use_mut_structs: bool = False,
        mut_structs_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.structure_encoder = structure_encoder
        self.score_name = score_name
        self.dms_name = dms_name
        self.incl_destab = incl_destab
        self.domainome = domainome
        self.include_doubles = incl_doubles
        self.include_mut_context = incl_mutctx
        self.include_reversions = incl_reversions
        self.enable_mutctx_masking = enable_mutctx_masking
        self.use_mut_structs = use_mut_structs
        self.mut_structs_root = mut_structs_root

        dms_df = dms_df.copy()
        dms_df['ddG'] = dms_df[self.score_name]
        
        if path is None: path = '.'
        self.cache_path = os.path.join(path, 'cache', f"{self.dms_name}_absolute_v1.pkl")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.data: List[Dict[str, Any]] = []
        self._mutant_struct_cache = {}

        if generate or not os.path.exists(self.cache_path):
            self.data = self.generate_data(dms_df, path)
            self._save_data_to_cache()
        else:
            self.load_data_from_cache()
        self.rng = np.random.default_rng()

    def _save_data_to_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            self.data = pickle.load(f)

    # ---------------------------
    # Core loading & generation
    # ---------------------------
    def generate_data(self, dms_df, path):
        if self.score_name == 'dG_ML':
            df = dms_df.loc[dms_df['code'].str.contains(self.dms_name, regex=False)] if not self.domainome else dms_df.loc[dms_df['code'] == self.dms_name]
            if not self.dms_name.startswith('v2_'):
                df = df.loc[~df['code'].str.startswith('v2_')]
            info = df.head(1)
            df = df.copy()
            df['mutated_sequence'] = df['aa_seq']
            try:
                pdb_file = info['pdb_file'].item()
            except Exception as e:
                print(info)
                raise e
            data = self._load_data_chainrule(df, pdb_file, 'A', is_predicted=False)
        else:
            print('Not using predicted structures!')
            data = []
            for (code, chain), df_sub in dms_df.groupby(['code', 'chain']):
                df_sub = df_sub.copy()
                df_sub['mutated_sequence'] = df_sub['mut_seq']
                pdb_file = df_sub['pdb_file'].head(1).item()
                data.extend(self._load_data_chainrule(df_sub, pdb_file, chain, incl_chain_in_code=True))
        return data

    def __len__(self): return len(self.data)
    def __iter__(self):
        indices = list(range(len(self.data)))
        self.rng.shuffle(indices)
        for idx in indices: yield self.data[idx]

    def _apply_surgery(self, base_seq, base_coords, base_struct, base_ri, mutations):
        """Generates MT sequence and tensors from a baseline with masking for indels."""
        seq_list = list(base_seq)
        coords, struct, ri = base_coords.clone(), base_struct.clone(), base_ri.clone()
        is_indel = False
        
        # Process mutations from back to front to preserve indices
        for wt, pos, mt in sorted(mutations, key=lambda x: x[1], reverse=True):
            assert pos > 0
            assert pos <= len(seq_list)
            if mt == '-': # Deletion
                is_indel = True
                seq_list.pop(pos-1)
                
                # Create the gap in tensors
                coords = torch.cat([coords[:, :pos], coords[:, pos+1:]], dim=1)
                struct = torch.cat([struct[:, :pos], struct[:, pos+1:]], dim=1)
                ri = torch.cat([ri[:pos-1], ri[pos:]], dim=0)
                
                # MASK NEIGHBORS: The residue at `pos-1` and `pos` in the new tensor
                # are the ones that were stitched together. Mask them to avoid
                # penalizing the "broken" geometry.
                # pos is 1-based. indices: 0..pos-1 (Left), pos (Right)
                # Note: `struct` is padded (BOS at 0), so `pos` index aligns with residue `pos`
                #struct[:, pos-1] = C.STRUCTURE_MASK_TOKEN
                #struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #coords[:, pos-1, :, :] = float('nan')
                #coords[:, pos, :, :] = float('nan')

            elif wt == '-': # Insertion
                is_indel = True
                
                # MASK LEFT ANCHOR (at index `pos`)
                #struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #coords[:, pos, :, :] = float('nan')

                # Insert the new residues
                # Note: We use NaN for coords so `build_affine3d` treats them as missing
                for aa in reversed(mt):
                    seq_list.insert(pos, aa)
                    coords = torch.cat([coords[:, :pos+1], torch.full((1, 1, 37, 3), float("nan")), coords[:, pos+1:]], dim=1)
                    struct = torch.cat([struct[:, :pos+1], torch.tensor([[C.STRUCTURE_MASK_TOKEN]]), struct[:, pos+1:]], dim=1)
                    ri = torch.cat([ri[:pos], ri[pos-1:pos], ri[pos:]], dim=0)
                
                # MASK RIGHT ANCHOR
                # The right anchor was originally at `pos+1`. 
                # After inserting `len(mt)`, it is at `pos + 1 + len(mt)`.
                # (Loop inserts in reverse, pushing right anchor out)
                #idx_right = pos + len(mt) + 1
                #if idx_right < struct.shape[1]: # Bounds check just in case
                #    struct[:, idx_right] = C.STRUCTURE_MASK_TOKEN
                #    coords[:, idx_right, :, :] = float('nan')

            else: # Substitution
                seq_list[pos-1] = mt
                #if self.enable_mutctx_masking:
                #    struct[:, pos] = C.STRUCTURE_MASK_TOKEN
                #    coords[:, pos, :, :] = float('nan')
        
        new_seq = "".join(seq_list)
        try:
            assert len(seq_list) + 2 == struct.shape[1]
        except AssertionError as e:
            print(len(seq_list) + 2)
            print(struct.shape[1])
            print(mutations)
        return new_seq, torch.tensor(self.tokenizer.encode(new_seq)), coords, struct, ri, is_indel

    def _load_data_chainrule(self, df, pdb_file, chain, is_predicted=False, incl_chain_in_code=False):
        self._mutant_struct_cache.clear()
        ref_chain = ProteinChain.from_pdb(pdb_file, chain, is_predicted=is_predicted)
        ref_coords, _, ref_ri = ref_chain.to_structure_encoder_inputs()
        
        # Prepare Structure Tokens
        # If encoder is None, we create dummy tokens (-1) of the correct shape (1, L)
        # so that surgery (slicing) works correctly.
        if self.structure_encoder:
            _, ref_struct = self.structure_encoder.encode(ref_coords, residue_index=ref_ri)
        else:
            # Create dummy tokens of shape (1, L) filled with -1
            ref_struct = torch.full((1, ref_coords.shape[1]), -1, dtype=torch.long)

        # Pad Tensors (BOS/EOS)
        ref_struct = F.pad(ref_struct, (1, 1), value=0 if self.structure_encoder else -1)
        if self.structure_encoder:
            ref_struct[:, 0] = C.STRUCTURE_BOS_TOKEN
            ref_struct[:, -1] = C.STRUCTURE_EOS_TOKEN
        
        ref_coords = F.pad(ref_coords, (0, 0, 0, 0, 1, 1), value=float("inf"))
        
        data = []
        if 'mut_structure' not in df.columns: df['mut_structure'] = df['pdb_file']
        df['mut_structure'] = df['mut_structure'].fillna(df['pdb_file'])
        
        for backbone, group in df.groupby('mut_structure'):
            code = group['code'].head(1).item() + (chain if incl_chain_in_code else "")
            b_chain = ProteinChain.from_pdb(group['pdb_file'].head(1).item() if not backbone.endswith('.pdb') else backbone, chain)
            b_coords, _, b_ri = b_chain.to_structure_encoder_inputs()
            
            # Similar handling for backbone structure tokens
            if self.structure_encoder:
                _, b_struct = self.structure_encoder.encode(b_coords, residue_index=b_ri)
            else:
                b_struct = torch.full((1, b_coords.shape[1]), -1, dtype=torch.long)
            
            b_struct = F.pad(b_struct, (1, 1), value=0 if self.structure_encoder else -1)
            if self.structure_encoder:
                b_struct[:, 0] = C.STRUCTURE_BOS_TOKEN
                b_struct[:, -1] = C.STRUCTURE_EOS_TOKEN
            else:
                # Ensure dummy tokens don't look like valid BOS/EOS to prevent confusion if not handled downstream,
                # but -1 is generally safe.
                pass 

            b_coords = F.pad(b_coords, (0, 0, 0, 0, 1, 1), value=float("inf"))
            b_seq = b_chain.sequence

            # First Pass: Map singles for dddG
            single_map = {}
            parsed = []
            for _, row in group.iterrows():
                if 'mut_type' not in df.columns:
                    offset, alignment, _ = custom_end_gap_alignment(row['mut_seq'], b_seq)
                    muts = determine_diffs(row['mut_seq'][offset:len(b_seq)+offset], b_seq)
                else:
                    muts = []
                    for m in row['mut_type'].split(':'):
                        if m.endswith('-'): muts.append((m[0], int(m[1:-1]), '-'))
                        elif m.startswith('-'):
                            p = int(re.search(r'\d+', m).group())
                            muts.append(('-', p, re.sub(r'[-\d]+', '', m)))
                        else: muts.append((m[0], int(m[1:-1]), m[-1]))
                parsed.append((muts, float(row['ddG'])))
                if len(muts) == 1: single_map[(muts[0][1], muts[0][2])] = float(row['ddG'])

            for muts, ddG in parsed:
                dddG = ddG - single_map.get((muts[0][1], muts[0][2]), 0) - single_map.get((muts[1][1], muts[1][2]), 0) if len(muts)==2 else np.nan
                
                # (A) Originals
                if len(muts) == 1:
                    mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, muts)
                    data.append(self._create_paired_item(code, muts, ddG, dddG, is_ind, 'single', 
                                                       (torch.tensor(self.tokenizer.encode(b_seq)), b_coords, b_struct, b_ri),
                                                       (mt_tok, mt_crd, mt_st, mt_ri)))
                    
                # (A) Originals
                if len(muts) == 2 and self.include_doubles:
                    mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, muts)
                    data.append(self._create_paired_item(code, muts, ddG, dddG, is_ind, 'single',
                                                       (torch.tensor(self.tokenizer.encode(b_seq)), b_coords, b_struct, b_ri),
                                                       (mt_tok, mt_crd, mt_st, mt_ri)))

                # (B) Context Singles (B|A)
                if len(muts) == 2 and self.include_mut_context:
                    for i, j in [(0, 1), (1, 0)]:
                        mA, mB = muts[i], muts[j]
                        if (mA[1], mA[2]) in single_map:
                            # Baseline is context A
                            a_seq, a_tok, a_crd, a_st, a_ri, _ = self._apply_surgery(b_seq, b_coords, b_struct, b_ri, [mA])
                            # Mutant is A + B
                            mt_seq, mt_tok, mt_crd, mt_st, mt_ri, is_ind = self._apply_surgery(a_seq, a_crd, a_st, a_ri, [mB])
                            data.append(self._create_paired_item(code, [mB], ddG - single_map[(mA[1], mA[2])], np.nan, is_ind, 'mut_ctx',
                                                               (a_tok, a_crd, a_st, a_ri), (mt_tok, mt_crd, mt_st, mt_ri)))
        return data

    def _create_paired_item(self, code, muts, ddG, dddG, is_indel, subset, wt_pack, mt_pack):
        return {
            'pdb': code, 'mutations': muts, 'ddG': ddG, 'dddG': dddG, 'is_indel': is_indel, 'subset_type': subset,
            'wt_tokens': wt_pack[0].numpy(), 'wt_coords': wt_pack[1].numpy(), 'wt_struct': wt_pack[2].numpy(), 'wt_ri': wt_pack[3].numpy(),
            'mt_tokens': mt_pack[0].numpy(), 'mt_coords': mt_pack[1].numpy(), 'mt_struct': mt_pack[2].numpy(), 'mt_ri': mt_pack[3].numpy()
        }

def collate_fn_chainrule_absolute(batch):
    def pad_key(key, pad_val):
        tensors = []
        for item in batch:
            val = item[key]
            # Ensure tensor
            t = torch.from_numpy(val) if isinstance(val, np.ndarray) else torch.as_tensor(val)
            
            # Squeeze leading singleton dim from Dataset (e.g. [1, L] -> [L])
            if t.ndim > 1 and t.shape[0] == 1:
                t = t.squeeze(0)
            tensors.append(t)
            
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)

    # Mutant Tensors
    seq_mut = pad_key('mt_tokens', C.SEQUENCE_PAD_TOKEN)
    crd_mut = pad_key('mt_coords', float("nan"))
    str_mut = pad_key('mt_struct', C.STRUCTURE_PAD_TOKEN)
    #try:
    #    ri_mut = pad_key('mt_ri', 0)
    #except:
     #   print([item['mt_ri']for item in batch])

    # Handle dddG
    if 'dddG' in batch[0]:
        dddG_np = np.array([item['dddG'] for item in batch], dtype=np.float32)
        dddG = torch.from_numpy(dddG_np)
    else:
        dddG = torch.full((len(batch),), float('nan'), dtype=torch.float)

    return {
        'pdb': [i['pdb'] for i in batch],
        'ddG': torch.tensor([i['ddG'] for i in batch], dtype=torch.float),
        'dddG': dddG,
        'mutations': [[('X', 0, 'X')] for _ in batch],
        'sequence_tokens': seq_mut, 
        'coords': crd_mut, 
        'structure_tokens': str_mut, 
        #'residue_index': ri_mut,
        'subset_type': [i['subset_type'] for i in batch],
        'use_absolute': True
    }


import math
import random
import numpy as np
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class BucketState:
    """Tracks sampling state for a single bucket."""
    indices: List[int] = field(default_factory=list)
    cursor: int = 0
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def remaining(self) -> int:
        return max(0, len(self.indices) - self.cursor)
    
    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.indices)
    
    def shuffle(self, rng: random.Random) -> None:
        rng.shuffle(self.indices)
        self.cursor = 0
    
    def take(self, n: int) -> List[int]:
        """Take up to n indices WITHOUT replacement."""
        available = min(n, self.remaining)
        if available <= 0:
            return []
        result = self.indices[self.cursor:self.cursor + available]
        self.cursor += available
        return result


@dataclass
class ProteinSamplerState:
    """Tracks sampling state for all buckets in a single protein dataset."""
    dataset: Any
    buckets: Dict[str, BucketState] = field(default_factory=dict)
    protein_name: str = ""
    batches_allocated: int = 0
    batches_drawn: int = 0
    
    @property
    def total_remaining(self) -> int:
        return sum(b.remaining for b in self.buckets.values())
    
    @property
    def total_samples(self) -> int:
        return sum(len(b) for b in self.buckets.values())
    
    @property
    def allocation_exhausted(self) -> bool:
        return self.batches_drawn >= self.batches_allocated
    
    @property
    def fully_exhausted(self) -> bool:
        return all(b.exhausted for b in self.buckets.values())
    
    def reset_epoch(self, rng: random.Random) -> None:
        for bucket in self.buckets.values():
            bucket.shuffle(rng)
        self.batches_drawn = 0


class SubsetRestrictedProteinCyclingDataLoader:
    """
    Cycles through multiple protein datasets with per-batch caps on subset types.
    
    This loader does NOT try to hit target subset fractions. Instead, it simply:
    1. Caps restricted subsets (like doubles) to a maximum fraction per batch
    2. Fills the rest of the batch from unrestricted subsets
    
    strategy='equal':
        Each protein contributes the same number of batches, determined by
        the protein with the fewest unrestricted samples. Round-robin sampling.
        
    strategy='all':
        Exhausts all proteins, sampling weighted by remaining items.
        Restricted subsets are still capped per batch.
    """
    
    SUBSET_ORDER = ['single', 'double', 'mut_ctx', 'reversion']
    
    def __init__(
        self,
        dataloaders: List,
        batch_size: int,
        train_list: List[str],
        collate_fn: Callable,
        strategy: str = 'all',
        fraction: float = 1.0,
        *,
        subset_caps: Optional[Dict[str, Optional[float]]] = None,
        rng_seed: Optional[int] = None,
        protein_weighting: str = "sqrt",
        weighting_temperature: float = 1.0,
        verbose: bool = True,
    ):
        """
        Args:
            dataloaders: List of DataLoader objects, one per protein
            batch_size: Samples per batch
            train_list: Protein names corresponding to dataloaders
            collate_fn: Function to collate samples into a batch
            strategy: 'equal' (balanced across proteins) or 'all' (exhaust everything)
            fraction: Fraction of computed batches to actually use
            subset_caps: Per-batch caps as fractions. Default: {'double': 0.1}, others unlimited.
                         Use None for unlimited, or a float in (0, 1] for fraction of batch.
            rng_seed: Random seed for reproducibility
            protein_weighting: For strategy='all', how to weight proteins ('sqrt', 'linear', 'uniform')
            weighting_temperature: Temperature for protein weighting
            verbose: Print logging information
        """
        if strategy not in ('equal', 'all'):
            raise ValueError(f"strategy must be 'equal' or 'all', got '{strategy}'")
        
        self.dataloaders = list(dataloaders)
        self.batch_size = batch_size
        self.train_list = list(train_list)
        self.collate_fn = collate_fn
        self.strategy = strategy
        self.fraction = fraction
        self.verbose = verbose
        
        # RNG
        self._rng = random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)
        
        # Protein weighting for 'all' strategy
        self.protein_weighting = protein_weighting
        self.weighting_temperature = max(float(weighting_temperature), 1e-8)
        
        # Subset caps: only doubles restricted by default
        self.subset_caps: Dict[str, Optional[float]] = {
            'single': None,
            'double': 0.0,
            'mut_ctx': 0.25,
            'reversion': 0.0,
        }
        if subset_caps is not None:
            self.subset_caps.update(subset_caps)
        
        # State
        self.protein_states: List[Optional[ProteinSamplerState]] = [None] * len(dataloaders)
        self.protein_weights: Optional[np.ndarray] = None
        
        # Epoch tracking
        self.total_batches_drawn = 0
        self.target_batches = 0
        self.current_loader_idx = 0
        self._initialized = False

    # -------------------------------------------------------------------------
    # Cap helpers
    # -------------------------------------------------------------------------
    
    def _cap_for_bucket(self, bucket: str) -> int:
        """
        Convert fractional cap to absolute count for a single batch.
        None means unlimited (returns batch_size).
        """
        cap = self.subset_caps.get(bucket)
        if cap is None:
            return self.batch_size
        if isinstance(cap, float):
            if cap >= 1.0:
                return self.batch_size
            return max(1, int(math.floor(cap * self.batch_size)))
        return min(int(cap), self.batch_size)
    
    def _is_restricted(self, bucket: str) -> bool:
        """Check if bucket has a meaningful cap (less than batch_size)."""
        return self._cap_for_bucket(bucket) < self.batch_size
    
    def _count_unrestricted_samples(self, state: ProteinSamplerState) -> int:
        """Count samples in unrestricted buckets for a protein."""
        total = 0
        for bucket_name, bucket_state in state.buckets.items():
            if not self._is_restricted(bucket_name):
                total += len(bucket_state)
        return total
    
    def _count_remaining_unrestricted(self, state: ProteinSamplerState) -> int:
        """Count remaining samples in unrestricted buckets."""
        total = 0
        for bucket_name, bucket_state in state.buckets.items():
            if not self._is_restricted(bucket_name):
                total += bucket_state.remaining
        return total

    # -------------------------------------------------------------------------
    # Protein state building
    # -------------------------------------------------------------------------
    
    def _build_protein_state(self, idx: int) -> ProteinSamplerState:
        """Build bucket structure for a protein dataset."""
        dl = self.dataloaders[idx]
        ds = dl.dataset
        protein_name = self.train_list[idx] if idx < len(self.train_list) else f"protein_{idx}"
        
        # Categorize all indices by subset_type
        buckets: Dict[str, List[int]] = {k: [] for k in self.SUBSET_ORDER}
        
        for i in range(len(ds)):
            try:
                item = ds[i]
                subset_type = item.get('subset_type', 'single')
                if subset_type in buckets:
                    buckets[subset_type].append(i)
                else:
                    # Unknown type → treat as single
                    buckets['single'].append(i)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to index item {i} from {protein_name}: {e}")
        
        # Create bucket states
        bucket_states = {}
        for name, indices in buckets.items():
            state = BucketState(indices=indices.copy())
            state.shuffle(self._rng)
            bucket_states[name] = state

        # Subsample restricted buckets upstream to match the capacity of unrestricted items
        total_unrest = sum(len(b) for name, b in bucket_states.items() if not self._is_restricted(name))
        sum_restr_caps = sum(self._cap_for_bucket(k) for k in self.SUBSET_ORDER if self._is_restricted(k))
        unrest_cap_per_batch = max(1, self.batch_size - sum_restr_caps)
        
        est_batches = total_unrest / unrest_cap_per_batch
        
        for name, b in bucket_states.items():
            if self._is_restricted(name):
                max_items = int(math.ceil(est_batches * self._cap_for_bucket(name)))
                if len(b.indices) > max_items:
                    if self.verbose:
                        print(f"Subsampling '{name}' in {protein_name}: {len(b.indices)} -> {max_items} "
                              f"(based on {total_unrest} unrestricted items)")
                    b.indices = b.indices[:max_items]

        print(f"[DEBUG] Built {protein_name}: "
              f"single={len(bucket_states['single'])}, "
              f"double={len(bucket_states['double'])}, "
              f"mut_ctx={len(bucket_states['mut_ctx'])}, "
              f"reversion={len(bucket_states['reversion'])}")
        
        return ProteinSamplerState(
            dataset=ds,
            buckets=bucket_states,
            protein_name=protein_name,
        )

    # -------------------------------------------------------------------------
    # Batch planning and sampling
    # -------------------------------------------------------------------------
    
    def _plan_batch_counts(self, state: ProteinSamplerState) -> Dict[str, int]:
        """
        Plan sample counts per bucket for one batch.
        
        Strategy:
        1. Add restricted items first (up to their caps)
        2. Fill remaining slots from unrestricted buckets PROPORTIONALLY
        based on their remaining samples
        """
        B = self.batch_size
        plan = {k: 0 for k in self.SUBSET_ORDER}
        remaining_slots = B
        
        # Phase 1: Add from restricted buckets (up to caps)
        for bucket in self.SUBSET_ORDER:
            if not self._is_restricted(bucket):
                continue
            bucket_state = state.buckets.get(bucket)
            if not bucket_state:
                continue
            cap = self._cap_for_bucket(bucket)
            available = bucket_state.remaining
            take = min(available, cap, remaining_slots)
            plan[bucket] = take
            remaining_slots -= take
        
        if remaining_slots <= 0:
            return plan
        
        # Phase 2: Fill remaining slots from unrestricted buckets PROPORTIONALLY
        unrestricted_remaining = {}
        total_unrestricted = 0
        
        for bucket in self.SUBSET_ORDER:
            if self._is_restricted(bucket):
                continue
            bucket_state = state.buckets.get(bucket)
            if bucket_state and bucket_state.remaining > 0:
                unrestricted_remaining[bucket] = bucket_state.remaining
                total_unrestricted += bucket_state.remaining
        
        if total_unrestricted == 0:
            return plan
        
        # Allocate proportionally to remaining samples in each bucket
        allocated = 0
        for bucket, avail in unrestricted_remaining.items():
            proportion = avail / total_unrestricted
            share = int(proportion * remaining_slots)
            take = min(share, avail)
            plan[bucket] = take
            allocated += take
        
        # Distribute rounding remainder to buckets with most remaining capacity
        leftover = remaining_slots - allocated
        if leftover > 0:
            # Sort by remaining (after initial allocation) descending
            sorted_buckets = sorted(
                unrestricted_remaining.keys(),
                key=lambda k: unrestricted_remaining[k] - plan[k],
                reverse=True
            )
            for bucket in sorted_buckets:
                if leftover <= 0:
                    break
                capacity = unrestricted_remaining[bucket] - plan[bucket]
                if capacity > 0:
                    take = min(leftover, capacity)
                    plan[bucket] += take
                    leftover -= take

        total_planned = sum(plan.values())
        if total_planned == 0:
            print(f"[DEBUG] Empty plan for {state.protein_name}!")
            print(f"  Bucket remaining: { {k: state.buckets[k].remaining for k in self.SUBSET_ORDER} }")
            print(f"  Restricted: {[k for k in self.SUBSET_ORDER if self._is_restricted(k)]}")
        
        return plan
    
    def _sample_batch_from_protein(self, idx: int) -> Tuple[List[Any], Dict[str, int]]:
        """Sample one batch from a protein WITHOUT replacement."""
        state = self.protein_states[idx]
        if state is None:
            return [], {}
        
        plan = self._plan_batch_counts(state)
        items = []
        actual_counts = {k: 0 for k in self.SUBSET_ORDER}
        
        for bucket_name, count in plan.items():
            if count <= 0:
                continue
            
            bucket_state = state.buckets.get(bucket_name)
            if not bucket_state:
                continue
            
            indices = bucket_state.take(count)
            for i in indices:
                try:
                    items.append(state.dataset[i])
                    actual_counts[bucket_name] += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to load {state.protein_name}[{i}]: {e}")
        
        # Only count as a drawn batch if we actually got items
        if items:
            state.batches_drawn += 1
            self._rng.shuffle(items)
        
        return items, actual_counts

    # -------------------------------------------------------------------------
    # Strategy='equal' helpers
    # -------------------------------------------------------------------------
    
    def _compute_batches_per_protein_equal(self) -> int:
        """
        For strategy='equal': compute batches per protein based on the
        protein with fewest unrestricted samples.
        """
        min_unrestricted = float('inf')
        limiting_protein = None
        
        for state in self.protein_states:
            if state is None:
                continue
            unrestricted = self._count_unrestricted_samples(state)
            if unrestricted < min_unrestricted:
                min_unrestricted = unrestricted
                limiting_protein = state.protein_name
        
        if min_unrestricted == float('inf') or min_unrestricted == 0:
            if self.verbose:
                print(f"Warning: No unrestricted samples found!")
            return 0
        
        batches = int(min_unrestricted // self.batch_size)
        
        if self.verbose:
            print(f"\nStrategy='equal' analysis:")
            print(f"  Limiting protein: {limiting_protein}")
            print(f"  Unrestricted samples: {int(min_unrestricted)}")
            print(f"  Batches per protein: {batches}")
        
        return batches
    
    def _setup_equal_strategy(self) -> None:
        """Setup for strategy='equal'."""
        batches_per_protein = self._compute_batches_per_protein_equal()
        
        if self.fraction < 1.0:
            batches_per_protein = int(batches_per_protein * self.fraction)
        
        for state in self.protein_states:
            if state is not None:
                state.batches_allocated = batches_per_protein
        
        self.target_batches = batches_per_protein * len(self.dataloaders)

    # -------------------------------------------------------------------------
    # Strategy='all' helpers
    # -------------------------------------------------------------------------
    
    def _compute_protein_weights(self) -> np.ndarray:
        """Compute sampling weights for strategy='all' based on remaining items that can actually be sampled."""
        remaining = np.array([
            ps.total_remaining if (ps is not None and self._can_produce_batch(ps)) else 0
            for ps in self.protein_states
        ], dtype=np.float64)
        
        if self.protein_weighting == "uniform":
            weights = (remaining > 0).astype(np.float64)
        elif self.protein_weighting == "sqrt":
            weights = np.sqrt(np.maximum(remaining, 0))
        elif self.protein_weighting == "linear":
            weights = remaining.copy()
        else:
            weights = np.sqrt(np.maximum(remaining, 0))
        
        # Apply temperature
        if self.weighting_temperature != 1.0:
            weights = np.power(weights + 1e-10, 1.0 / self.weighting_temperature)
        
        # Normalize
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.zeros(len(self.protein_states))
        
        return weights
    
    def _setup_all_strategy(self) -> None:
        """Setup for strategy='all'."""
        # Estimate total batches needed to exhaust all data.
        # Only count items in buckets that have a cap > 0.
        total_drawable = 0
        for ps in self.protein_states:
            if ps is not None:
                for bucket_name, bucket_state in ps.buckets.items():
                    if self._cap_for_bucket(bucket_name) > 0:
                        total_drawable += len(bucket_state)
        
        # Upper bound estimate on batches
        self.target_batches = int(math.ceil(total_drawable / self.batch_size))
        
        if self.fraction < 1.0:
            self.target_batches = int(self.target_batches * self.fraction)
        
        # For 'all', batches_allocated is not a hard limit
        for state in self.protein_states:
            if state is not None:
                state.batches_allocated = self.target_batches  # Large number

    # -------------------------------------------------------------------------
    # Epoch initialization
    # -------------------------------------------------------------------------
    
    def _initialize_epoch(self) -> None:
        """Initialize or reset for a new epoch."""
        if self.verbose:
            print("Initializing epoch...")
        
        # Shuffle protein order
        combined = list(zip(self.dataloaders, self.train_list))
        self._rng.shuffle(combined)
        if combined:
            self.dataloaders, self.train_list = map(list, zip(*combined))
        
        # Build protein states
        self.protein_states = [None] * len(self.dataloaders)
        for idx in range(len(self.dataloaders)):
            self.protein_states[idx] = self._build_protein_state(idx)
        
        # Strategy-specific setup
        if self.strategy == 'equal':
            self._setup_equal_strategy()
        else:
            self._setup_all_strategy()
        
        self.total_batches_drawn = 0
        self.current_loader_idx = 0
        self._initialized = True
        
        if self.verbose:
            self._print_epoch_summary()
    
    def _print_epoch_summary(self) -> None:
        """Print summary of epoch setup."""
        # Identify restricted buckets
        restricted = [k for k in self.SUBSET_ORDER if self._is_restricted(k)]
        unrestricted = [k for k in self.SUBSET_ORDER if not self._is_restricted(k)]
        
        print(f"\nEpoch initialized (strategy='{self.strategy}'):")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Restricted subsets: {restricted} (caps: {[self._cap_for_bucket(k) for k in restricted]})")
        print(f"  Unrestricted subsets: {unrestricted}")
        print(f"  Target batches (Estimate): {self.target_batches}")
        
        if self.strategy == 'equal':
            batches_per = self.protein_states[0].batches_allocated if self.protein_states[0] else 0
            print(f"  Batches per protein: {batches_per}")
        
        print(f"\n{'Protein':<30} {'single':>8} {'double':>8} {'mut_ctx':>8} {'rev':>8} {'unrest':>10}")
        print("-" * 82)
        
        for state in self.protein_states:
            if state is None:
                continue
            counts = {k: len(state.buckets.get(k, [])) for k in self.SUBSET_ORDER}
            unrestricted_count = self._count_unrestricted_samples(state)
            print(f"{state.protein_name:<30} {counts['single']:>8} {counts['double']:>8} "
                  f"{counts['mut_ctx']:>8} {counts['reversion']:>8} {unrestricted_count:>10}")

    # -------------------------------------------------------------------------
    # Iterator protocol
    # -------------------------------------------------------------------------
    
    def __len__(self) -> int:
        return self.target_batches if self._initialized else 0
    
    def __iter__(self):
        self._initialize_epoch()
        return self
    
    def __next__(self):
        # We only strictly enforce StopIteration via target_batches for 'equal' strategy, 
        # or if fraction < 1.0 was explicitly passed.
        # Otherwise, we rely on the exhaustion checks in _next_all.
        if self.total_batches_drawn >= self.target_batches:
            if self.strategy == 'equal' or self.fraction < 1.0:
                if self.verbose:
                    print(f"\nEpoch complete: {self.total_batches_drawn} batches")
                raise StopIteration
        
        if self.strategy == 'equal':
            return self._next_equal()
        else:
            return self._next_all()
    
    def _can_produce_batch(self, state: ProteinSamplerState) -> bool:
        """Check if this protein can produce a non-empty batch."""
        for bucket_name in self.SUBSET_ORDER:
            bucket_state = state.buckets.get(bucket_name)
            # Must have remaining items AND the cap must allow drawing at least 1 item
            if bucket_state and bucket_state.remaining > 0 and self._cap_for_bucket(bucket_name) > 0:
                return True
        return False

    def _next_equal(self):
        num_proteins = len(self.protein_states)

        for _ in range(num_proteins):
            state = self.protein_states[self.current_loader_idx]
            idx = self.current_loader_idx
            
            self.current_loader_idx = (self.current_loader_idx + 1) % num_proteins
            
            if state is None:
                continue
            if state.allocation_exhausted:
                continue
            if not self._can_produce_batch(state):
                continue  # Skip without burning allocation
            
            items, counts = self._sample_batch_from_protein(idx)
            
            if not items:  
                continue
            
            self.total_batches_drawn += 1
            return self.collate_fn(items)
        
        # If we looped through all proteins without returning, we're done
        raise StopIteration
    
    def _next_all(self):
        """
        Sample from proteins weighted by remaining items until exhausted.
        """
        while True:
            self.protein_weights = self._compute_protein_weights()
            
            if self.protein_weights.sum() == 0:
                if self.verbose:
                    print(f"\nAll valid proteins exhausted after {self.total_batches_drawn} batches")
                raise StopIteration
            
            idx = int(self._np_rng.choice(len(self.protein_states), p=self.protein_weights))
            state = self.protein_states[idx]
            
            items, counts = self._sample_batch_from_protein(idx)
            
            if items:
                break
        
        self.total_batches_drawn += 1
        
        if self.verbose and self.total_batches_drawn % 100 == 0:
            remaining = sum(ps.total_remaining for ps in self.protein_states if ps)
            print(f"Batch {self.total_batches_drawn} [{state.protein_name}]: {counts}, "
                  f"{remaining} total samples remaining")
        
        return self.collate_fn(items)

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def reset_epoch(self) -> None:
        """Manually reset for a new epoch."""
        self._initialize_epoch()
    
    def get_epoch_stats(self) -> Dict[str, Any]:
        """Get statistics about current epoch state."""
        stats = {
            'strategy': self.strategy,
            'batch_size': self.batch_size,
            'target_batches': self.target_batches,
            'batches_drawn': self.total_batches_drawn,
            'subset_caps': {k: self._cap_for_bucket(k) for k in self.SUBSET_ORDER},
            'proteins': {},
        }
        
        for state in self.protein_states:
            if state is None:
                continue
            stats['proteins'][state.protein_name] = {
                'total': state.total_samples,
                'remaining': state.total_remaining,
                'batches_drawn': state.batches_drawn,
                'buckets': {k: len(state.buckets.get(k, [])) for k in self.SUBSET_ORDER},
            }
        
        return stats


class ProteinCyclingDataLoader:
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
        #elif self.strategy == 'repeat':
        #    self.target_batches = self.max_length * self.num_dataloaders
        elif self.strategy == 'all':
            self.target_batches = sum(self.dataloader_lengths)
        else:
            raise AssertionError('strategy must be one of: min, repeat, all')
        self.epoch_ended = False
        
        # Keep track of iterators separately
        self.iterators = {}

        # Probability weights for selecting dataloaders
        self.weights = np.array(self.dataloader_lengths, dtype=np.float32) / sum(self.dataloader_lengths)

        print("Dataloader lengths:")
        for name, length in zip(self.train_list, self.dataloader_lengths):
            print(f"{name}: {length}")
        print(f"Min length: {self.min_length}")
        print(f"Target batches: {self.target_batches}")

    def __len__(self):
        return self.target_batches

    def reset_dataloader(self, idx):
        """Reset a specific dataloader"""
        if idx in self.iterators:
            del self.iterators[idx]
            
        old_dataloader = self.dataloaders[idx]
        
        # Create new generator with random seed
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
        """Reset all dataloaders"""
        print('Shuffling dataloaders.')
        
        # Zip dataloaders and train_list to shuffle them in sync
        combined = list(zip(self.dataloaders, self.train_list))
        random.shuffle(combined)
        self.dataloaders, self.train_list = zip(*combined)
        
        # Convert back to lists (zip returns tuples) to ensure mutability if needed later
        self.dataloaders = list(self.dataloaders)
        self.train_list = list(self.train_list)
        
        self.dataloader_lengths = [len(dl) for dl in self.dataloaders]
        
        print('Dataloader lengths:')
        for name, length in zip(self.train_list, self.dataloader_lengths):
            print(f"{name}: {length}")
            
        self.weights = np.array(self.dataloader_lengths, dtype=np.float32) / sum(self.dataloader_lengths)
        for idx in tqdm(range(self.num_dataloaders)):
            self.reset_dataloader(idx)
        
        # Clear all iterators
        self.iterators.clear()
        
        # Reset counters
        self.total_batches_drawn = 0
        self.batches_drawn = [0] * self.num_dataloaders
        self.current_loader_idx = 0
        
        gc.collect()
        print('Shuffled all dataloaders.')

    def reset_epoch(self):
        """Reset for new epoch"""
        self.epoch_ended = False
        self.shuffle_all()

    def __iter__(self):
        self.reset_epoch()
        return self

    def get_current_iterator(self):
        """Get iterator for current loader, creating if necessary and checking for exhaustion."""
        if self.current_loader_idx not in self.iterators:
            # Check if this dataloader has been exhausted
            if self.strategy != 'repeat' and self.batches_drawn[self.current_loader_idx] >= len(self.dataloaders[self.current_loader_idx]):
                return None
            #elif self.strategy == 'repeat' and self.batches_drawn[self.current_loader_idx] >= len(self.dataloaders[self.current_loader_idx]):
            #    self.reset_dataloader(self.current_loader_idx)
            # Otherwise, create a new iterator
            self.iterators[self.current_loader_idx] = iter(self.dataloaders[self.current_loader_idx])
        return self.iterators[self.current_loader_idx]

    def update_weights(self):
        # Update weights based on remaining batches
        remaining_batches = [max(0, len(dl) - drawn) for dl, drawn in zip(self.dataloaders, self.batches_drawn)]
        total_remaining = sum(remaining_batches)
        self.weights = np.array(remaining_batches, dtype=np.float32) / total_remaining if total_remaining > 0 else np.zeros_like(self.weights)

    def __next__(self):
        if self.total_batches_drawn >= self.target_batches:
            print('Stopping due to drawing target number of batches')
            raise StopIteration
        
        while True:
            
            print('Sampling from', self.train_list[self.current_loader_idx])

            if self.strategy == 'all':
                self.update_weights()
                self.current_loader_idx = np.random.choice(self.num_dataloaders, p=self.weights)

            current_iterator = self.get_current_iterator()
            batch = next(current_iterator)
            self.batches_drawn[self.current_loader_idx] += 1
            self.total_batches_drawn += 1
            if self.strategy != 'all':
                self.current_loader_idx = (self.current_loader_idx + 1) % self.num_dataloaders

            if len(batch['ddG']) < self.batch_size:
                print(f'Smaller than expected batch of size {len(batch["ddG"])} / {self.batch_size} detected!')
                print(f"Loader: {self.train_list[self.current_loader_idx]} has been exhausted")
                
            return batch


class PooledDataLoader:
    """
    Pools samples from multiple dataloaders and yields padded batches without
    repetition within an epoch. Compatible with your masking/forward pipeline.

    Key outputs per batch:
      - sequence_tokens_orig:  [B, Lmax] (long)
      - structure_tokens_orig: [B, K, Lmax] (long)  (K>=1; we standardize 1D -> K=1)
      - coords_orig:           [B, Lmax, ...] or [B, Kc, Lmax, ...] (float)
      - positions:             list[tuple] of 0-based residue columns per item
      - mutations:             carried through (we don't alter semantics)
      - attention_mask:        [B, Lmax] (1 for real tokens, 0 for pad)
      - position_ids:          [B, Lmax] (0..L_i-1 then 0 or any filler for pads)
      - lengths:               [B] (original residues length, BEFORE pad)
      - ddG / dddG (if present), and legacy 'ground_truth' is aliased to ddG
    """

    def __init__(
        self,
        dataloaders: List[Iterable],
        batch_size: int,
        train_list: Optional[List[str]] = None,
        strategy: str = "all",                 # {"all", "min"}  (kept from your version)
        *,
        # --- Token / mask IDs (pass from your tokenizer/constants) ---
        # If you have tokenizer.vocab["<pad>"], pass that here.
        seq_pad_token_id: int = C.SEQUENCE_PAD_TOKEN,
        # Structure: use a dedicated PAD that is distinct from STRUCTURE_MASK_TOKEN
        structure_pad_token_id: int = C.STRUCTURE_PAD_TOKEN,
        # Coordinates pad (inf is typical for "absent")
        coord_pad_value: float = float("inf"),
        # Optional sanity check: provide a callable int->AA (e.g., tokenizer inverse map)
        token_id_to_aa: Optional[Callable[[int], str]] = None,
        # Are mutation positions in incoming samples 1-based (most of your code is)? If True, we convert to 0-based.
        mutations_are_1_based: bool = True,
        # If present in items, should we respect existing *_orig fields instead of reusing masked ones?
        prefer_orig_fields: bool = True,
        # Set this true to print a couple of first-batch diagnostics
        debug_first_batches: int = 0,
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

        self.token_id_to_aa = token_id_to_aa
        self.mutations_are_1_based = bool(mutations_are_1_based)
        self.prefer_orig_fields = bool(prefer_orig_fields)
        self.debug_first_batches = int(debug_first_batches)

        # storage
        self.dataset_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pooled_data: List[Dict[str, Any]] = []
        self.current_indices: List[int] = []
        self.current_batch = 0
        self._batch_count = 0
        self.rng = random.Random()

        # Load data from all dataloaders (retain your original logic/behavior)
        print("Loading data from dataloaders...")
        for i, dl in enumerate(self.dataloaders):
            dataset_name = self.train_list[i]
            print(f"Loading data from {dataset_name}")
            try:
                dataset = dl.dataset
                for data in dataset:
                    self.dataset_samples[dataset_name].append(data)
            except AttributeError:
                print(f"Iterating through dataloader {i} to get samples")
                for batch in dl:
                    if isinstance(batch, dict):
                        # Unbatch dicts
                        bsz = len(batch.get("pdb", [])) or len(batch.get("ground_truth", [])) or 1
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

        # Balance (keeps your options)
        self._balance_datasets()

        # Pool
        for _, samples in self.dataset_samples.items():
            self.pooled_data.extend(samples)

        print(f"Total pooled samples: {len(self.pooled_data)}")
        print("Samples per dataset after balancing:")
        for name, samples in self.dataset_samples.items():
            print(f"  {name}: {len(samples)} samples")

        # Number of batches per epoch
        self.batches_per_epoch = len(self.pooled_data) // self.batch_size + (1 if len(self.pooled_data) % self.batch_size else 0)
        print(f"Batches per epoch: {self.batches_per_epoch}")

        # Prepare epoch indices
        self.current_indices = list(range(len(self.pooled_data)))
        self.current_batch = 0

    # --------------------------
    # Balancing (unchanged logic)
    # --------------------------
    def _balance_datasets(self):
        if not self.strategy or self.strategy == "all":
            print("No balancing strategy selected - using all available data")
            return

        dataset_sizes = {name: len(samples) for name, samples in self.dataset_samples.items()}
        print("Dataset sizes before balancing:")
        for name, size in dataset_sizes.items():
            print(f"  {name}: {size} samples")

        if self.strategy == "min" and dataset_sizes:
            min_size = min(dataset_sizes.values())
            print(f"Balancing datasets by subsampling to {min_size} samples each")
            for name, samples in list(self.dataset_samples.items()):
                if len(samples) > min_size:
                    self.dataset_samples[name] = self.rng.sample(samples, min_size)

    # --------------------------
    # Iteration
    # --------------------------
    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        self.rng.shuffle(self.current_indices)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.batches_per_epoch:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.current_indices))
        batch_idxs = self.current_indices[start_idx:end_idx]
        items = [self.pooled_data[i] for i in batch_idxs]

        batch = self._collate_with_padding(items)

        self.current_batch += 1
        return batch

    def shuffle_all(self):
        self.rng.shuffle(self.current_indices)
        gc.collect()
        print("Shuffled all data.")

    def reset_epoch(self):
        self.shuffle_all()

    # --------------------------
    # Collation helpers
    # --------------------------
    @staticmethod
    def _to_tensor_long(x) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) and x.dtype == torch.long else torch.as_tensor(x, dtype=torch.long)

    @staticmethod
    def _to_tensor_float(x) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) and x.dtype.is_floating_point else torch.as_tensor(x, dtype=torch.float)

    @staticmethod
    def _ensure_2d_structure(st: Union[np.ndarray, torch.Tensor, List[int]]) -> torch.Tensor:
        """Return [K, L], upgrading [L] -> [1, L]."""
        t = PooledDataLoader._to_tensor_long(st)
        if t.ndim == 1:
            t = t.unsqueeze(0)  # [1, L]
        assert t.ndim == 2, f"structure tokens must be [K,L] or [L]; got shape {tuple(t.shape)}"
        return t

    @staticmethod
    def _ensure_coords_residue_axis(coords: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Accepts:
          - [L, ...]  -> we will later add batch dim
          - [Kc, L, ...]
        We standardize at collate time to have a batch dim first; we ONLY pad along L.
        """
        t = PooledDataLoader._to_tensor_float(coords)
        assert t.ndim in (3, 4), f"coords must be 3D or 4D with residue axis present; got {tuple(t.shape)}"
        return t

    @staticmethod
    def _right_pad_last_dim(x: torch.Tensor, Lmax: int, pad_val: int | float) -> torch.Tensor:
        """Pad only the last dimension to length Lmax."""
        need = Lmax - x.size(-1)
        if need <= 0:
            return x
        return F.pad(x, (0, need), value=pad_val)

    def _positions_from_mutations(self, muts: List[Tuple[str, int, str]], L_res: int) -> Tuple[int, ...]:
        """
        Compute 0-based residue columns from a mutation list.
        Assumes incoming positions are 1-based unless self.mutations_are_1_based=False.
        """
        pos_cols = []
        for (_, pos, _) in (muts or []):
            p = int(pos)
            if self.mutations_are_1_based:
                p = p - 1
            assert 0 <= p < L_res, f"Mutation index {pos} -> {p} out of bounds for length {L_res}"
            pos_cols.append(p)
        return tuple(sorted(set(pos_cols)))

    # --------------------------
    # Core collate
    # --------------------------
    def _collate_with_padding(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate variable-length proteins by **right-padding the residue axis (L)** only.
        Emits *_orig tensors and indexing helpers expected by your training code.
        """
        if not batch:
            return {}

        B = len(batch)

        # Choose the "orig" sources (if present) so we never re-pad already-masked working tensors by accident.
        def get_seq_arr(item):
            if self.prefer_orig_fields and "sequence_tokens_orig" in item:
                return np.asarray(item["sequence_tokens_orig"])
            return np.asarray(item["sequence_tokens"])

        def get_str_arr(item):
            key = "structure_tokens_orig" if (self.prefer_orig_fields and "structure_tokens_orig" in item) else "structure_tokens"
            return np.asarray(item[key])

        def get_crd_arr(item):
            key = "coords_orig" if (self.prefer_orig_fields and "coords_orig" in item) else "coords"
            return np.asarray(item[key])

        # --- Determine per-sample residue lengths (L_i) safely ---
        lengths: List[int] = []
        seq_list_1d: List[torch.Tensor] = []
        for it in batch:
            s = self._to_tensor_long(get_seq_arr(it))
            assert s.ndim == 1, f"sequence must be [L]; got shape {tuple(s.shape)}"
            seq_list_1d.append(s)
            lengths.append(int(s.size(0)))
        Lmax = max(lengths)

        # --- Sequences: pad on residue axis to [B, Lmax] ---
        seq_pad = []
        for s in seq_list_1d:
            seq_pad.append(self._right_pad_last_dim(s, Lmax, self.seq_pad_token_id))
        sequence_tokens_orig = torch.stack(seq_pad, dim=0)  # [B, Lmax] long

        # --- Structure tokens: standardize to [K, L_i], then pad L -> [K, Lmax], finally stack -> [B, K, Lmax] ---
        struct_pad = []
        for it in batch:
            st = self._ensure_2d_structure(get_str_arr(it))  # [K, L]
            K, L = st.shape
            assert L == len(np.asarray(get_seq_arr(it))), "Structure/sequence residue length mismatch before pad."
            st_p = self._right_pad_last_dim(st, Lmax, self.structure_pad_token_id)  # [K, Lmax]
            struct_pad.append(st_p)
        structure_tokens_orig = torch.stack(struct_pad, dim=0)  # [B, K, Lmax] long

        # --- Structure tokens: standardize to [K, L_i], then pad L -> [K, Lmax], finally stack -> [B, K, Lmax] ---
        plddt_pad = []
        for it in batch:
            pt = it['plddt']  # [K, L]
            pt = self._ensure_2d_structure(pt)  # [K, L]
            K, L = pt.shape
            assert L == len(np.asarray(get_seq_arr(it))), "Structure/sequence residue length mismatch before pad."
            pt_p = self._right_pad_last_dim(pt, Lmax, 0)  # [K, Lmax]
            plddt_pad.append(pt_p)
        plddt = torch.stack(plddt_pad, dim=0)  # [B, K, Lmax] long

        # --- Coordinates: accept [L, ...] or [Kc, L, ...]; pad along L only; stack with batch in front ---
        coords_list = []
        for it in batch:
            c = self._ensure_coords_residue_axis(get_crd_arr(it))  # 3D or 4D
            if c.ndim == 3:
                # [L, A1, A2] -> add batch dimension later
                assert c.size(0) == len(np.asarray(get_seq_arr(it))), "Coords/sequence residue length mismatch before pad."
                c_p = self._right_pad_last_dim(c, Lmax, self.coord_pad_value)  # [Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))  # [1, Lmax, A1, A2]
            else:
                # [Kc, L, A1, A2] -> pad along L (dim=1)
                assert c.size(1) == len(np.asarray(get_seq_arr(it))), "Coords/sequence residue length mismatch before pad."
                # move residue L to last, pad, then move back to preserve-only-L padding
                c_perm = c.permute(0, 2, 3, 1)            # [Kc, A1, A2, L]
                c_p = self._right_pad_last_dim(c_perm, Lmax, self.coord_pad_value)  # [Kc, A1, A2, Lmax]
                c_p = c_p.permute(0, 3, 1, 2)             # [Kc, Lmax, A1, A2]
                coords_list.append(c_p.unsqueeze(0))       # [1, Kc, Lmax, A1, A2]
        # Stack; result is either [B, Lmax, ...] or [B, Kc, Lmax, ...] depending on inputs
        coords_orig = torch.cat(coords_list, dim=0)

        # --- Basic metadata & labels ---
        pdb = [it.get("pdb", f"unk_{i}") for i, it in enumerate(batch)]
        st = [it.get("subset_type", None) for i, it in enumerate(batch)]

        # ddG / ground_truth reconciliation
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

        # --- Mutations & positions (0-based) ---
        mutations = [it.get("mutations", []) for it in batch]
        # positions are the **residue columns** (0-based) for all sites in the item
        positions = [self._positions_from_mutations(m, L_res=lengths[i]) for i, m in enumerate(mutations)]

        # --- Attention mask & position ids ---
        attention_mask = torch.zeros((B, Lmax), dtype=torch.long)
        position_ids = torch.zeros((B, Lmax), dtype=torch.long)
        for i, Li in enumerate(lengths):
            attention_mask[i, :Li] = 1
            position_ids[i, :Li] = torch.arange(Li, dtype=torch.long)

        # --- Optional WT sanity check (only if you gave token_id_to_aa) ---
        if self.token_id_to_aa is not None:
            for i, muts in enumerate(mutations):
                Li = lengths[i]
                for (wt, pos, _mt) in (muts or []):
                    p0 = pos - 1 if self.mutations_are_1_based else pos
                    if 0 <= p0 < Li:
                        tok = int(sequence_tokens_orig[i, p0].item())
                        aa = self.token_id_to_aa(tok)
                        assert aa == wt, f"WT mismatch @row {i}, pos {pos} (0-based {p0}): seq has '{aa}', WT says '{wt}'"

        # --- Final shape assertions (protects your downstream masking) ---
        # 1) structure L matches sequence L
        assert structure_tokens_orig.size(-1) == sequence_tokens_orig.size(1), "Structure/sequence L mismatch after pad."
        # 2) coords residue axis matches sequence L
        if coords_orig.ndim == 4:      # [B, Lmax, ...]
            assert coords_orig.size(-3) == sequence_tokens_orig.size(1), "Coords/sequence L mismatch after pad."
        elif coords_orig.ndim == 5:    # [B, Kc, Lmax, ...]
            assert coords_orig.size(-3) == sequence_tokens_orig.size(1), "Coords/sequence L mismatch after pad."
        else:
            raise AssertionError("coords_orig must be 3D or 4D after collation.")

        # --- Diagnostics (optional) ---
        if self._batch_count < self.debug_first_batches:
            print(f"\n[DEBUG] Batch {self._batch_count} diagnostics:")
            self._batch_count += 1
            i0 = 0
            print(f"  Lmax: {Lmax} | lengths[0]: {lengths[i0]}")
            print(f"  sequence_tokens_orig[0].shape: {tuple(sequence_tokens_orig[i0].shape)}")
            print(f"  structure_tokens_orig[0].shape: {tuple(structure_tokens_orig[i0].shape)}")
            print(f"  coords_orig.shape: {tuple(coords_orig.shape)}")
            print(f"  positions[0]: {positions[i0] if positions else '[]'}")

        # --- Return collated batch with *_orig fields and helpers ---
        collated = {
            "pdb": pdb,
            "mutations": mutations,                        # untouched; your trainer rewrites for masking when needed
            "positions": positions,                        # 0-based residue columns (tuple per item)
            "sequence_tokens_orig": sequence_tokens_orig,  # [B, Lmax] long
            "structure_tokens_orig": structure_tokens_orig,# [B, K, Lmax] long
            "coords_orig": coords_orig,                    # [B, Lmax, ...] or [B, Kc, Lmax, ...] float
            "attention_mask": attention_mask,              # [B, Lmax] long (1 for real, 0 for pad)
            "position_ids": position_ids,                  # [B, Lmax] long (0..L_i-1 then filler)
            "lengths": torch.as_tensor(lengths, dtype=torch.long),  # [B]
            # Labels (ddG primary, ground_truth alias):
            "ddG": ddG,
            "dddG": dddG,
            "ground_truth": ddG,                           # legacy alias (as in your dataset class)
            "subset_type": st,
            "plddt": plddt
        }
        return collated    