import requests
import pandas as pd
import numpy as np
import torch
import sys
import os
import tempfile
import shutil
import re
import argparse
import logging
import itertools

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C

from pdbfixer import PDBFixer
from openmm.app import PDBFile

from Bio import pairwise2
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def parse_mutation_column_to_separate_columns(df, column_name):
    """
    Parse a DataFrame column containing mutation strings into separate columns
    for each mutation component (fr1, pos1, to1, fr2, pos2, to2, etc.).
    """
    result_df = df.copy(deep=True)
    
    max_mutations = 0
    for mutation_string in df[column_name]:
        if pd.isna(mutation_string) or mutation_string == '':
            continue
        
        mutations = mutation_string.split(':')
        max_mutations = max(max_mutations, len(mutations))
    
    for i in range(1, max_mutations + 1):
        result_df[f'fr{i}'] = None
        result_df[f'pos{i}'] = None
        result_df[f'to{i}'] = None
    
    for idx, mutation_string in result_df[column_name].items():
        if pd.isna(mutation_string) or mutation_string == '':
            continue
  
        mutations = mutation_string.split(':')
        
        for i, part in enumerate(mutations, 1):
            match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', part)
            
            if match:
                result_df.at[idx, f'fr{i}'] = match.group(1)
                result_df.at[idx, f'pos{i}'] = int(match.group(2))
                result_df.at[idx, f'to{i}'] = match.group(3)
            else:
                raise AssertionError(f"Could not parse mutation '{part}' in string '{mutation_string}'. Ensure strictly formatted identifiers.")
    
    return result_df

def download_pdb(pdb_id, output_dir='.', file_format='pdb', get_fasta=True, dataset=None):
    pdb_id = pdb_id.lower().strip()
    os.makedirs(output_dir, exist_ok=True)
    
    result = {'pdb': None, 'fasta': None}
    
    if file_format.lower() == 'pdb':
        file_ext = '.pdb'
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    elif file_format.lower() == 'cif':
        file_ext = '.cif'
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    else:
        raise ValueError("Format must be 'pdb' or 'cif'")
    
    output_file = os.path.join(output_dir, f"{pdb_id}{file_ext}")
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {pdb_id} structure to {output_file}")
        result['pdb'] = output_file
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdb_id} structure: {str(e)}")
    
    if get_fasta:
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        fasta_file = os.path.join(output_dir, f"{pdb_id}.fasta")
        
        try:
            fasta_response = requests.get(fasta_url)
            fasta_response.raise_for_status()
            
            with open(fasta_file, 'wb') as f:
                f.write(fasta_response.content)
            print(f"Downloaded {pdb_id} sequence to {fasta_file}")
            result['fasta'] = fasta_file
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {pdb_id} sequence: {str(e)}")
    
    return result

def get_alphafold_structure(uniprot_id, output_file, sequence):
    result = {'pdb': None, 'fasta': output_file.replace('.pdb', '.fasta')}
    with open(output_file.replace('.pdb', '.fasta'), 'w') as file:
        file.write(f'>{uniprot_id}\n{sequence}')

    base_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    
    response = requests.get(base_url)
    if response.status_code == 200:
        with open(output_file, 'w') as file:
            file.write(response.text)
            result['pdb'] = output_file
        print(f"AlphaFold structure for {uniprot_id} saved to {output_file}")
    else:
        print(f"Structure for UniProt ID {uniprot_id} not found. Status code: {response.status_code}")

    return result

def remove_caps(input_pdb, verbose=True):
    parser = PDBParser(QUIET=True)
    original_model = parser.get_structure('original', input_pdb)[0]

    residues_to_remove = []
    for chain in original_model.get_chains():
        for residue in chain:
            res_name = residue.get_resname()
            if res_name in ['ACE', 'NME']:
                residues_to_remove.append((chain.id, residue.id))
    
    if residues_to_remove and verbose:
        print(f"Removing capping groups: {residues_to_remove}")
    
    for chain_id, res_id in residues_to_remove:
        try:
            chain = original_model[chain_id]
            chain.detach_child(res_id)
        except Exception as e:
            if verbose: print(f"Could not remove {chain_id}:{res_id} - {str(e)}")

    io = PDBIO()
    io.set_structure(original_model)
    io.save(input_pdb)

def remove_heteroatoms(pdb_file, output_file, verbose=False):
    pdb_file = os.path.abspath(pdb_file)
    output_file = os.path.abspath(output_file)
    
    if not os.path.exists(pdb_file):
        if verbose: print(f"Error: Input file {pdb_file} does not exist.")
        return False

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    class StandardResidueSelect(Select):
        def accept_residue(self, residue):
            hetero_flag = residue.id[0]
            if hetero_flag != ' ':
                return False
            else:
                return True

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, select=StandardResidueSelect())
    
    if verbose:
        print(f"Cleaned structure saved to {output_file}")
        
    return True

def fix_noncanonical_residues(input_pdb, output_pdb, verbose=False):
    parser = PDBParser(QUIET=True)
    original_model = parser.get_structure('original', input_pdb)[0]
    
    original_chains = []
    for chain in original_model:
        if chain.id not in [c[0] for c in original_chains]:
            num_residues = len(list(chain.get_residues()))
            original_chains.append((chain.id, num_residues))
    
    noncanonical_residues = {}
    std_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
              "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
              "THR", "TRP", "TYR", "VAL"]
    
    for chain in original_model:
        for residue in chain:
            resname = residue.get_resname()
            if resname not in std_aa:
                res_id = residue.get_id()[1]
                res_key = f"{chain.id}:{res_id}"
                noncanonical_residues[res_key] = resname
    
    fixer = PDBFixer(input_pdb)
    fixer.findNonstandardResidues()
    
    original_topology = {}
    for chain in fixer.topology.chains():
        chain_dict = {}
        for residue in chain.residues():
            chain_dict[residue.id] = residue.name
        original_topology[chain.id] = chain_dict
    
    fixer.replaceNonstandardResidues()
    
    temp_output = 'temp_fixed.pdb'
    PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_output, 'w'))
    
    fixed = parser.get_structure('fixed', temp_output)
    
    fixed_chains = []
    for model in fixed:
        for chain in model:
            if chain.id not in [c[0] for c in fixed_chains]:
                num_residues = len(list(chain.get_residues()))
                fixed_chains.append((chain.id, num_residues))
    
    chain_id_map = {}
    original_idx = 0
    for fixed_idx, (fixed_id, fixed_res_count) in enumerate(fixed_chains):
        while original_idx < len(original_chains) and original_chains[original_idx][1] == 0:
            original_idx += 1
            
        if original_idx < len(original_chains):
            orig_id = original_chains[original_idx][0]
            chain_id_map[fixed_id] = orig_id
            original_idx += 1
    
    for model in fixed:
        for chain in model:
            if chain.id in chain_id_map:
                chain.id = chain_id_map[chain.id]

    # Restore Original PDB Numbering
    topo_residues = list(fixer.topology.residues())
    fixed_residues = list(fixed.get_residues())
    
    if len(topo_residues) == len(fixed_residues):
        for t_res, f_res in zip(topo_residues, fixed_residues):
            # PDBFixer topology preserves the original PDB ID as a string (e.g. '15A')
            match = re.match(r"(\d+)([a-zA-Z]?)", str(t_res.id))
            if match:
                res_seq = int(match.group(1))
                icode = match.group(2) if match.group(2) else ' '
                # Safely update the BioPython index, restoring original numbering and insertion codes
                f_res.id = (f_res.id[0], res_seq, icode)
    elif verbose:
        print("Warning: PDBFixer dropped residues; could not perfectly map original PDB indices back to intermediate file.")
    # -----------------------------------------------

    if verbose:
        new_topology = {}
        for model in fixed:
            for chain in model:
                chain_dict = {}
                for residue in chain:
                    res_id = residue.get_id()[1]
                    res_name = residue.get_resname()
                    chain_dict[res_id] = res_name
                new_topology[chain.id] = chain_dict
        
        changes = []
        for orig_chain_id, orig_chain_data in original_topology.items():
            new_chain_id = None
            for fixed_id, orig_id in chain_id_map.items():
                if orig_id == orig_chain_id:
                    new_chain_id = orig_id
                    break
            
            if new_chain_id and new_chain_id in new_topology:
                for res_id, old_name in orig_chain_data.items():
                    if res_id in new_topology[new_chain_id]:
                        new_name = new_topology[new_chain_id][res_id]
                        if old_name != new_name and old_name not in std_aa:
                            changes.append(f"Chain {new_chain_id}, Residue {res_id}: {old_name} → {new_name}")
        
        if changes:
            print(f"\nNon-canonical residues replaced in {input_pdb}:")
            for change in changes:
                print(f"  {change}")
            print(f"Total: {len(changes)} residues replaced\n")
        else:
            print(f"No non-canonical residues were replaced in {input_pdb}")

        remaining_nonstandard = {}
        for model in fixed:
            for chain in model:
                residues_to_remove = []
                for residue in chain:
                    resname = residue.get_resname()
                    if resname not in std_aa:
                        res_id = residue.get_id()[1]
                        res_key = f"{chain.id}:{res_id}"
                        remaining_nonstandard[res_key] = resname
                        residues_to_remove.append(residue.id)
                
                for res_id in residues_to_remove:
                    try:
                        chain.detach_child(res_id)
                        if verbose:
                            print(f"Removed non-standard residue: Chain {chain.id}, Residue {res_id[1]}: {remaining_nonstandard[f'{chain.id}:{res_id[1]}']}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed to remove residue: Chain {chain.id}, Residue {res_id[1]}: {str(e)}")

        if remaining_nonstandard and verbose:
            print(f"\nRemoved {len(remaining_nonstandard)} remaining non-standard residues that PDBFixer couldn't convert")  
            
    io = PDBIO()
    io.set_structure(fixed)
    io.save(output_pdb)
    
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    return output_pdb

def renumber_pdb(pdb_file, output_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    offset = 10000
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_list(), start=1):
                residue.id = (' ', i + offset, ' ')

    for model in structure:
        for chain in model:
            residues = sorted(chain.get_list(), key=lambda res: res.get_id()[1])
            for i, residue in enumerate(residues, start=1):
                residue.id = (' ', i, ' ')

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def repair_pdb(pdb_file, output_file, sequence_file=None, chain_id='A', 
               num_models=1, use_dope=True, verbose=False, return_all_chains=True, debug_dir=None):
    pdb_file = os.path.abspath(pdb_file)
    output_file = os.path.abspath(output_file)
    
    filename = os.path.basename(pdb_file)
    pdb_id = filename[:4] if len(filename) >= 4 else filename.split('.')[0]

    original_cwd = os.getcwd()
    
    if debug_dir:
        working_dir = os.path.abspath(debug_dir)
        os.makedirs(working_dir, exist_ok=True)
        from contextlib import nullcontext
        dir_context = nullcontext(working_dir)
        if verbose: print(f"DEBUG MODE: Saving intermediates to {working_dir}")
    else:
        dir_context = tempfile.TemporaryDirectory()

    with dir_context as temp_dir:
        try:
            os.chdir(temp_dir)
            
            parser = PDBParser(QUIET=True)
            try:
                original_structure = parser.get_structure('original', pdb_file)
            except Exception as e:
                raise AssertionError(f"Failed to parse {pdb_file}: {e}")

            chain_found = False
            for chain in original_structure.get_chains():
                if chain.id == chain_id:
                    chain_found = True
                    chain.id = chain_id
                    break
            
            if not chain_found:
                raise AssertionError(f"Chain {chain_id} not found in {pdb_file}")

            class TargetChainSelect(Select):
                def accept_chain(self, chain):
                    return chain.get_id() == chain_id
            
            isolated_pdb_name = f"{pdb_id}_{chain_id}_isolated.pdb"
            io = PDBIO()
            io.set_structure(original_structure)
            io.save(isolated_pdb_name, select=TargetChainSelect())
            
            # Needs Modeller Environment Access. Assuming it's appropriately initialized externally if used.
            from modeller import Environ, Model, AutoModel, assess
            
            env = Environ()
            env.io.atom_files_directory = ['.']
            env.libs.topology.read(file='$(LIB)/top_heav.lib')
            env.libs.parameters.read(file='$(LIB)/par.lib')
            
            pdb_code = f"{pdb_id}_{chain_id}"
            temp_pdb_link = os.path.join(temp_dir, f"{pdb_code}.pdb")
            if os.path.exists(temp_pdb_link):
                os.remove(temp_pdb_link)
            os.symlink(os.path.abspath(isolated_pdb_name), temp_pdb_link)

            mdl = Model(env, file=temp_pdb_link)
            
            target_chain_modeller = next((c for c in mdl.chains if c.name == chain_id), None)
            if not target_chain_modeller:
                 target_chain_modeller = list(mdl.chains)[0]

            pdb_seq = ''.join([residue.code for residue in target_chain_modeller.residues])
            
            complete_seq = None
            if sequence_file:
                if not os.path.isabs(sequence_file):
                    sequence_file = os.path.join(original_cwd, sequence_file)
                
                strong_match_seq = None
                weak_match_seq = None   
                generic_match_seq = None
                
                current_header = None
                current_seq_parts = []
                
                def process_record(header, sequence):
                    nonlocal strong_match_seq, weak_match_seq, generic_match_seq
                    
                    if not all(c in "ACTG" for c in sequence[:20]):
                        chain_block_match = re.search(r'Chain(s)?\s+([^|]+)', header)
                        
                        if chain_block_match:
                            chain_str = chain_block_match.group(2).strip()
                            parts = [p.strip() for p in chain_str.split(',')]
                            
                            for part in parts:
                                part_match = re.match(r'([^\[\s]+)(?:\[auth\s+([^\]]+)\])?', part)
                                
                                if part_match:
                                    pdb_c = part_match.group(1)
                                    auth_c = part_match.group(2)
                                    
                                    if auth_c and auth_c == chain_id:
                                        strong_match_seq = sequence
                                    elif pdb_c == chain_id:
                                        weak_match_seq = sequence
                        
                        if f"Chain {chain_id}" in header or f"Chains {chain_id}" in header:
                             generic_match_seq = sequence
                        elif chain_id in header: 
                             if generic_match_seq is None:
                                 generic_match_seq = sequence

                with open(sequence_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        if line.startswith('>'):
                            if current_header:
                                process_record(current_header, "".join(current_seq_parts))
                            current_header = line
                            current_seq_parts = []
                        else:
                            current_seq_parts.append(line)
                    
                    if current_header:
                        process_record(current_header, "".join(current_seq_parts))
                
                if strong_match_seq:
                    complete_seq = strong_match_seq
                    if verbose: print(f"Using sequence from AUTH match for chain {chain_id}")
                elif weak_match_seq:
                    complete_seq = weak_match_seq
                    if verbose: print(f"Using sequence from CHAIN match for chain {chain_id}")
                elif generic_match_seq:
                    complete_seq = generic_match_seq
                    if verbose: print(f"Using sequence from GENERIC header match for chain {chain_id}")

                if complete_seq is None:
                     print(f"Warning: Could not find sequence for chain {chain_id} in {sequence_file}")
                     complete_seq = pdb_seq
            else:
                complete_seq = pdb_seq
            
            complete_seq = complete_seq.strip('X')

            if pdb_seq == complete_seq and not has_missing_atoms(mdl, target_chain_modeller.name):
                if verbose: print(f"No repair needed for {pdb_file}")
                clean_chain_file = isolated_pdb_name 
            else:
                alignments = pairwise2.align.globalms(complete_seq, pdb_seq, 2, -1, -2, -0.5)
                best_alignment = alignments[0]
                aligned_complete, aligned_pdb = best_alignment[0], best_alignment[1]

                print("Repairing missing residues:")
                print(aligned_pdb)
                print(aligned_complete)
                
                target_code = 'TARGET_SEQ'
                aln_file = "alignment.pir"
                
                try:
                    residue_ids = [int(r.num) for r in target_chain_modeller.residues]
                    min_res, max_res = min(residue_ids), max(residue_ids)
                except ValueError:
                    min_res, max_res = 1, len(pdb_seq)

                with open(aln_file, 'w') as f:
                    f.write(f">P1;{pdb_code}\n")
                    modeller_chain_id = target_chain_modeller.name
                    f.write(f"structure:{pdb_code}:{min_res}:{modeller_chain_id}:{max_res}:{modeller_chain_id}:.:.:.:.\n")
                    f.write(f"{aligned_pdb}*\n\n")
                    f.write(f">P1;{target_code}\n")
                    f.write(f"sequence:{target_code}:1:{modeller_chain_id}:{len(complete_seq)}:{modeller_chain_id}:.:.:.:.\n")
                    f.write(f"{aligned_complete}*\n")
                
                class MyCompleteModel(AutoModel):
                    def special_patches(self, aln): pass

                a = MyCompleteModel(env, alnfile=aln_file, knowns=pdb_code, sequence=target_code)
                a.starting_model = 1
                a.ending_model = num_models
                
                if use_dope:
                    a.assess_methods = (assess.DOPE, assess.GA341)
                
                a.make()
                
                best_model_file = None
                if num_models > 1 and use_dope:
                    dope_scores = []
                    for i in range(1, num_models + 1):
                        try:
                            name = a.outputs[i-1]['name']
                            mdl_tmp = Model(env, file=name)
                            score = mdl_tmp.assess_normalized_dope()
                            dope_scores.append((name, score))
                        except Exception: pass
                    
                    if dope_scores:
                        dope_scores.sort(key=lambda x: x[1])
                        best_model_file = dope_scores[0][0]
                
                if not best_model_file:
                    if len(a.outputs) > 0:
                        best_model_file = a.outputs[0]['name']
                    else:
                        best_model_file = f"{target_code}.B99990001.pdb"
                
                repaired_parser = PDBParser(QUIET=True)
                repaired_structure = repaired_parser.get_structure('repaired', best_model_file)
                for model in repaired_structure:
                    for chain in model:
                        chain.id = chain_id
                        break 
                    break

                clean_chain_file = "clean_repaired.pdb"
                io = PDBIO()
                io.set_structure(repaired_structure)
                io.save(clean_chain_file)

            if return_all_chains:
                parser = PDBParser(QUIET=True)
                original_structure = parser.get_structure('original', pdb_file)
                target_model = original_structure[0]
                
                repaired_ref_structure = parser.get_structure('repaired_ref', clean_chain_file)
                repaired_chain_obj = repaired_ref_structure[0][chain_id]

                original_order = [c.id for c in target_model]
                original_order.remove(chain_id)
                original_order.insert(0, chain_id)

                chains_to_add = {}
                for c in target_model:
                    if c.id == chain_id:
                        chains_to_add[c.id] = repaired_chain_obj
                    else:
                        chains_to_add[c.id] = c
                
                for cid in original_order:
                    if target_model.has_id(cid):
                        target_model.detach_child(cid)

                for cid in original_order:
                    if cid in chains_to_add:
                        target_model.add(chains_to_add[cid])

                io = PDBIO()
                io.set_structure(original_structure)
                io.save(output_file)
            else:
                shutil.copy(clean_chain_file, output_file)
                
            return True

        except Exception as e:
            raise AssertionError(f"Error repairing {pdb_file}: {str(e)}")
        finally:
            os.chdir(original_cwd)


def compute_pairwise_heavy_atom_dist_matrix(coords: torch.Tensor, exclude_backbone: bool = True) -> torch.Tensor:
    """
    Computes a fully vectorized pairwise distance matrix between all residues.
    coords: [L, 37, 3] or [1, L, 37, 3] tensor
    Returns: [L, L] tensor of minimum heavy atom distances.
    """
    if coords.dim() == 4:
        if coords.shape[0] == 1:
            coords = coords.squeeze(0)
        else:
            raise AssertionError(f"Expected coords to have a batch size of 1, but got shape {coords.shape}.")
    elif coords.dim() != 3:
        raise AssertionError(f"Expected coords to be 3D [L, 37, 3] or 4D [1, L, 37, 3], but got shape {coords.shape}.")
        
    L = coords.shape[0]
    dist_matrix = torch.full((L, L), float('nan'), device=coords.device)
    
    is_finite = torch.isfinite(coords).all(dim=-1)
    
    if exclude_backbone:
        sc_mask = is_finite & (torch.arange(37, device=coords.device) > 3)
        has_valid_sc = sc_mask.any(dim=-1) 
        ca_mask = is_finite & (torch.arange(37, device=coords.device) == 1)
        valid_mask = torch.where(has_valid_sc.unsqueeze(-1), sc_mask, ca_mask)
    else:
        valid_mask = is_finite & (torch.arange(37, device=coords.device) != 1)

    safe_coords = torch.where(
        valid_mask.unsqueeze(-1), 
        coords, 
        torch.tensor(1e9, dtype=coords.dtype, device=coords.device)
    )
    
    for i in range(L):
        c1 = safe_coords[i][valid_mask[i]] 
        if c1.shape[0] == 0:
            continue
            
        c1_batch = c1.unsqueeze(0).expand(L, -1, -1) 
        dists = torch.cdist(c1_batch.to(torch.float32), safe_coords.to(torch.float32)) 
        min_dists = dists.amin(dim=(1, 2)) 
        min_dists[min_dists >= 1e8] = float('nan')
        dist_matrix[i] = min_dists
        
    return dist_matrix

def get_pdb_to_seq_mapping(pdb_file, chain_id, original_seq):
    """
    Parses the PDB to construct a mapping between 1-based sequence indices and PDB indices.
    Robustness added for non-canonical mapping.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    chain = structure[0][chain_id]

    pdb_residues = []
    pdb_seq_list = []

    # Expanded robust standard and common non-canonical map
    RESIDUE_MAP = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}

    for residue in chain.get_residues():
        if residue.id[0] == ' ' or residue.id[0].startswith('H_'):
            resname = residue.get_resname()
            if resname in RESIDUE_MAP:
                mapped_char = RESIDUE_MAP[resname]
            else:
                if 'CA' in residue:
                    mapped_char = 'X'
                else:
                    continue # Strict ligand/water skip
                    
            res_id = residue.get_id()
            res_num = res_id[1]
            insertion_code = res_id[2].strip() 
            
            pdb_index = f"{res_num}{insertion_code}"
            pdb_residues.append(pdb_index)
            pdb_seq_list.append(mapped_char)

    pdb_sequence_from_parser = "".join(pdb_seq_list)

    if original_seq != pdb_sequence_from_parser:
        raise AssertionError(f"Sequence mismatch! ESM3 Sequence (len {len(original_seq)}): {original_seq}\nBioPython Sequence (len {len(pdb_sequence_from_parser)}): {pdb_sequence_from_parser}\nThis is a critical flaw caused by unaligned parsing rules between ESM3 and BioPython. Ensure the structure has no missing residues or unrecognized records.")

    seq_to_pdb = {i+1: pdb_residues[i] for i in range(len(original_seq))}
    pdb_to_seq = {v: k for k, v in seq_to_pdb.items()}
    
    return seq_to_pdb, pdb_to_seq


def generate_screening_df(args) -> pd.DataFrame:
    """Generates a mutation DataFrame programmatically based on screening parameters."""
    if not os.path.isfile(args.pdb_file):
        raise AssertionError(f"PDB file not found at: {args.pdb_file}")

    logging.info(f"Generating screening DataFrame for {args.pdb_file} (Chain {args.chain})...")
    
    try:
        chain_obj = ProteinChain.from_pdb(args.pdb_file, args.chain)
    except Exception as e:
        raise AssertionError(f"Failed to load chain {args.chain} from {args.pdb_file}: {e}")
        
    original_seq = chain_obj.sequence
    seq_to_pdb, pdb_to_seq = get_pdb_to_seq_mapping(args.pdb_file, args.chain, original_seq)

    target_positions = []
    
    if args.screen_residues:
        req_res = [r.strip() for r in args.screen_residues.split(',') if r.strip()]
        for r in req_res:
            if r in pdb_to_seq:
                target_positions.append(pdb_to_seq[r])
            else:
                raise AssertionError(f"Residue '{r}' requested in --screen_residues was not found in PDB chain '{args.chain}'.")
        target_positions = list(set(target_positions))
        
    elif args.screen_residues_except:
        exc_res = [r.strip() for r in args.screen_residues_except.split(',') if r.strip()]
        exc_pos = []
        for r in exc_res:
            if r in pdb_to_seq:
                exc_pos.append(pdb_to_seq[r])
            else:
                raise AssertionError(f"Residue '{r}' requested in --screen_residues_except was not found in PDB chain '{args.chain}'.")
        target_positions = [i+1 for i in range(len(original_seq)) if (i+1) not in exc_pos]
        
    else:
        target_positions = [i+1 for i in range(len(original_seq))]

    mut_list = []
    mut_list_pdb = []
    
    AAs = list('ACDEFGHIKLMNPQRSTVWY')
    modes = ['singles', 'doubles'] if args.mode == 'singles+doubles' else [args.mode]

    if 'singles' in modes:
        for pos in target_positions:
            wt = original_seq[pos-1]
            for mut in AAs:
                if mut != wt:
                    mut_list.append(f"{wt}{pos}{mut}")
                    mut_list_pdb.append(f"{wt}{seq_to_pdb[pos]}{mut}") 

    if 'doubles' in modes:
        pairs = list(itertools.combinations(target_positions, 2))
        
        if args.distance_threshold > 0:
            logging.info(f"Extracting coordinates to filter double mutants within {args.distance_threshold}Å...")
            coords_tensor, _, _ = chain_obj.to_structure_encoder_inputs()
            dist_matrix = compute_pairwise_heavy_atom_dist_matrix(coords_tensor)
            
            valid_pairs = []
            dropped_nan = 0
            for pos1, pos2 in pairs:
                dist = dist_matrix[pos1-1, pos2-1].item()
                if not np.isnan(dist) and dist <= args.distance_threshold:
                    valid_pairs.append((pos1, pos2))
                elif np.isnan(dist):
                    dropped_nan += 1
                    
            if dropped_nan > 0:
                logging.warning(f"Silently dropped {dropped_nan} mutation combinations due to unresolved coordinates.")
                
            logging.info(f"Filtered {len(pairs)} theoretical pairs down to {len(valid_pairs)} proximal pairs.")
            pairs = valid_pairs
        
        if len(pairs) > 1000:
            logging.warning(f"This will create {len(pairs)} unique position pairs ({len(pairs) * 19 * 19} double mutations!).")
            
        for pos1, pos2 in pairs:
            wt1 = original_seq[pos1-1]
            wt2 = original_seq[pos2-1]
            for mut1 in AAs:
                if mut1 == wt1: continue
                for mut2 in AAs:
                    if mut2 == wt2: continue
                    mut_list.append(f"{wt1}{pos1}{mut1}:{wt2}{pos2}{mut2}")
                    mut_list_pdb.append(f"{wt1}{seq_to_pdb[pos1]}{mut1}:{wt2}{seq_to_pdb[pos2]}{mut2}")

    if args.mutations:
        mut_list = [m.strip() for m in args.mutations.split(',')]
        mut_list_pdb = []
        for m_str in mut_list:
            pdb_parts = []
            for single_m in m_str.split(':'):
                if len(single_m) < 3:
                    raise AssertionError(f"Invalid mutation format: {single_m}")
                wt = single_m[0]
                mt = single_m[-1]
                try:
                    pos = int(single_m[1:-1])
                except ValueError:
                    raise AssertionError(f"Could not parse integer position from mutation string: {single_m}. Must be a sequence index.")
                
                if pos not in seq_to_pdb:
                    raise AssertionError(f"Position {pos} from sequence mutation not found in PDB mapping.")
                
                pdb_parts.append(f"{wt}{seq_to_pdb[pos]}{mt}")
            mut_list_pdb.append(":".join(pdb_parts))

    if not mut_list:
        raise AssertionError("The generated mutation list is empty. Ensure you selected valid target residues.")

    df = pd.DataFrame({
        'mut_type_renumbered': mut_list,
        'mut_type_pdb': mut_list_pdb
    })
    df['pdb_file'] = args.pdb_file
    df['code'] = args.code
    df['chain'] = args.chain
    
    logging.info(f"Generated {len(df)} mutation strings.")
    return df

def _parse_and_validate_mut_string(m_str, seq, seq_to_pdb, pdb_to_seq, assume_mode):
    renumbered_parts = []
    pdb_parts = []
    
    for single_m in m_str.split(':'):
        if len(single_m) < 3: return False, None, None
        wt, mt = single_m[0], single_m[-1]
        pos_str = single_m[1:-1]
        
        if assume_mode == 'renumbered':
            try:
                pos = int(pos_str)
                if pos < 1 or pos > len(seq) or seq[pos-1] != wt:
                    return False, None, None
                if pos not in seq_to_pdb:
                    return False, None, None
                renumbered_parts.append(single_m)
                pdb_parts.append(f"{wt}{seq_to_pdb[pos]}{mt}")
            except ValueError:
                return False, None, None
                
        elif assume_mode == 'pdb':
            if pos_str not in pdb_to_seq:
                return False, None, None
            seq_pos = pdb_to_seq[pos_str]
            if seq_pos < 1 or seq_pos > len(seq) or seq[seq_pos-1] != wt:
                return False, None, None
            pdb_parts.append(single_m)
            renumbered_parts.append(f"{wt}{seq_pos}{mt}")
            
    return True, ":".join(renumbered_parts), ":".join(pdb_parts)

def standardize_input_df(df: pd.DataFrame, backbone_mutation: str = None, quiet: bool = False) -> pd.DataFrame:
    if df.empty:
        raise AssertionError("Cannot standardize an empty DataFrame.")
        
    has_renum = 'mut_type_renumbered' in df.columns
    has_pdb = 'mut_type_pdb' in df.columns
    has_generic = 'mut_type' in df.columns
    
    if not (has_renum or has_pdb or has_generic):
        raise AssertionError("Input CSV must contain at least one of: 'mut_type', 'mut_type_renumbered', 'mut_type_pdb'")

    unique_structures = df[['pdb_file', 'chain']].drop_duplicates()
    if len(unique_structures) > 1:
        raise AssertionError(f"This script processes a single structure. Found {len(unique_structures)} in CSV.")
    
    if not quiet:
        logging.info("Standardizing mutation mapping columns...")
    df = df.copy()
    
    if 'mut_type_renumbered' not in df.columns: df['mut_type_renumbered'] = None
    if 'mut_type_pdb' not in df.columns: df['mut_type_pdb'] = None
    
    pdb = unique_structures['pdb_file'].iloc[0]
    chain = unique_structures['chain'].iloc[0]
    
    try:
        chain_obj = ProteinChain.from_pdb(pdb, chain)
        original_seq = chain_obj.sequence
        seq_to_pdb, pdb_to_seq = get_pdb_to_seq_mapping(pdb, chain, original_seq)
    except Exception as e:
        raise AssertionError(f"Failed to load structure {pdb} chain {chain} to standardize mutations: {e}")
        
    current_seq = original_seq
    
    if backbone_mutation:
        seq_list = list(current_seq)
        for single_m in backbone_mutation.split(':'):
            wt, mt = single_m[0], single_m[-1]
            try:
                pos = int(single_m[1:-1])
            except ValueError:
                raise AssertionError(f"Could not parse integer position from backbone_mutation string: {single_m}.")
            
            if pos < 1 or pos > len(seq_list):
                raise AssertionError(f"Backbone mutation position {pos} out of bounds for sequence length {len(seq_list)}.")
            if seq_list[pos-1] != wt:
                raise AssertionError(f"Backbone mutation expected {wt} at pos {pos}, but found {seq_list[pos-1]} in PDB sequence.")
            
            seq_list[pos-1] = mt
        current_seq = "".join(seq_list)
        
    for idx, row in df.iterrows():
        r_str, p_str = None, None
        
        if has_renum and pd.notna(row.get('mut_type_renumbered')):
            valid, r, p = _parse_and_validate_mut_string(str(row['mut_type_renumbered']), current_seq, seq_to_pdb, pdb_to_seq, assume_mode='renumbered')
            if not valid: raise AssertionError(f"Row {idx}: Invalid renumbered mutation '{row['mut_type_renumbered']}' against background.")
            r_str, p_str = r, p
            
        elif has_pdb and pd.notna(row.get('mut_type_pdb')):
            valid, r, p = _parse_and_validate_mut_string(str(row['mut_type_pdb']), current_seq, seq_to_pdb, pdb_to_seq, assume_mode='pdb')
            if not valid: raise AssertionError(f"Row {idx}: Invalid PDB mutation '{row['mut_type_pdb']}' against background.")
            r_str, p_str = r, p
            
        elif has_generic and pd.notna(row.get('mut_type')):
            m_str = str(row['mut_type'])
            valid_renum, r_renum, p_renum = _parse_and_validate_mut_string(m_str, current_seq, seq_to_pdb, pdb_to_seq, assume_mode='renumbered')
            valid_pdb, r_pdb, p_pdb = _parse_and_validate_mut_string(m_str, current_seq, seq_to_pdb, pdb_to_seq, assume_mode='pdb')
            
            if valid_renum and valid_pdb:
                r_str, p_str = r_renum, p_renum
            elif valid_renum:
                r_str, p_str = r_renum, p_renum
            elif valid_pdb:
                r_str, p_str = r_pdb, p_pdb
            else:
                raise AssertionError(f"Row {idx}: Mutation '{m_str}' matches neither sequence nor PDB numbering. Check wild-types.")
        else:
            raise AssertionError(f"Row {idx} is missing a mutation definition.")
            
        df.at[idx, 'mut_type_renumbered'] = r_str
        df.at[idx, 'mut_type_pdb'] = p_str
        
    return df

def add_lines(file):
    text_to_insert = 'CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1 '

    with open(file, 'r') as original_file:
        lines = original_file.readlines()

    if 'MODELLER' in lines[0]:
        lines.insert(1, text_to_insert + '\n')
    else:
        lines.insert(0, text_to_insert + '\n')

    with open(file, 'w') as modified_file:
        modified_file.writelines(lines)

def get_seq(pdb_file, chain):
    protein_chain = ProteinChain.from_pdb(pdb_file, chain, is_predicted=False)
    return protein_chain.sequence

def has_missing_atoms(model, chain_id='A'):
    for chain in model.chains:
        if chain.name == chain_id:
            for res in chain.residues:
                if is_missing_atoms(res):
                    return True
    return False

def is_missing_atoms(residue):
    expected_atoms = {
        'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6, 'GLN': 9, 
        'GLU': 9, 'GLY': 4, 'HIS': 10, 'ILE': 8, 'LEU': 8, 'LYS': 9,
        'MET': 8, 'PHE': 11, 'PRO': 7, 'SER': 6, 'THR': 7, 'TRP': 14,
        'TYR': 12, 'VAL': 7
    }
    
    if residue.code in expected_atoms:
        heavy_atom_count = sum(1 for a in residue.atoms if a.element != 'H')
        return heavy_atom_count < expected_atoms[residue.code]
    
    return False

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 
    'TYR': 'Y', 'MET': 'M', 'MSE': 'Z', 'UNK': '9'} 

def create_residue_mapping(original_pdb, repaired_pdb, chain):
    parser = PDBParser(QUIET=True)
    original = parser.get_structure('original', original_pdb)
    repaired = parser.get_structure('repaired', repaired_pdb)
    
    orig_seq = []
    orig_ids = []
    for residue in original[0][chain]:
        if residue.id[0] == ' ':  
            orig_seq.append(d[residue.resname])
            orig_ids.append(str(residue.id[1])+str(residue.id[2]).strip(' '))
    
    repair_seq = []
    repair_ids = []
    for residue in repaired[0][chain]:
        if residue.id[0] == ' ':
            repair_seq.append(d[residue.resname])
            repair_ids.append(residue.id[1])
    
    alignment = pairwise2.align.globalms(''.join(orig_seq), ''.join(repair_seq), 
                                        2, -1, -0.5, -0.1, one_alignment_only=True)[0]
    
    orig_to_new = {}
    orig_idx = 0
    repair_idx = 0

    i = 0
    while True:
        try:
            if alignment[0][i] != '-' and alignment[1][i] != '-':
                orig_to_new[orig_ids[orig_idx]] = repair_ids[repair_idx]
                orig_idx += 1
                repair_idx += 1
            elif alignment[0][i] != '-':
                orig_idx += 1
            elif alignment[1][i] != '-':
                repair_idx += 1
        except IndexError:
            break
        i += 1
    
    return orig_to_new, alignment[0], alignment[1]

def prepare_single_chain(pdb_file_path, chain_id, output_location):
    parser = PDB.PDBParser(QUIET=True)
    base_name = os.path.basename(pdb_file_path).split('.')[0]
    
    try:
        structure = parser.get_structure(base_name, pdb_file_path)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None
    
    new_structure = PDB.Structure.Structure(f"{base_name}_{chain_id}")
    model = structure[0]
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)
    
    chain_exists = False
    for chain in model:
        if chain.id == chain_id:
            new_model.add(chain.copy())
            chain_exists = True
            break
    
    if not chain_exists:
        print(f"Chain {chain_id} not found in {pdb_file_path}")
        return None
    
    os.makedirs(os.path.dirname(os.path.abspath(output_location)), exist_ok=True)
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    
    try:
        io.save(output_location)
        print(f"Successfully extracted chain {chain_id} to {output_location}")
        return output_location
    except Exception as e:
        print(f"Error writing output file: {e}")
        return None

def reorder_muts(muts):
    try:
        positions = []
        reordered = []
        for mut in muts:
            positions.append(int(mut[1:-1]))
        for position in sorted(positions):
            for mut in muts:
                if mut[1:-1] == str(position):
                    reordered.append(mut)
    except ValueError:
        positions = []
        reordered = []
        for mut in muts:
            positions.append(int(re.match(r'[A-Z]([0-9]*)[A-Z][A-Z]?', mut).group(1)))
        for position in sorted(positions):
            for mut in muts:
                if int(re.match(r'[A-Z]([0-9]*)[A-Z][A-Z]?', mut).group(1)) == int(position):
                    if not mut in reordered:
                        reordered.append(mut)
    return reordered

def main(args):
    # Dataset specific brittle mappings remain identical per your request...
    # (Existing main script content goes here but is unchanged from original)
    if 'Id25c03_1merNCL.txt' in args.db_loc:
        locs = ['1merNCL', '1merNCLB']
        for loc in locs:
            db_ = pd.read_csv(
                args.db_loc.replace('1merNCL', loc), sep=' ', header=None)
            db_ = db_.rename(
                {0: 'code', 1: 'mutant', 2: 'ddG', 3: 'pos2'}, axis=1)
            db_['wild_type'] = db_['mutant'].str[0]
            db_['chain'] = db_['mutant'].str[1]
            db_['position'] = db_['mutant'].str[2:-1].astype(int)
            db_['mutation'] = db_['mutant'].str[-1]
            db_['uid'] = db_['code']+db_['chain']+'_'+\
                db_['wild_type']+db_['position'].astype(str)+db_['mutation']
            db_ = db_.drop_duplicates(subset=['uid'], keep='first')
            if loc == '1merNCL':
                print(args.db_loc.replace(
                    'Id25c03_1merNCL.txt', 'K3822.csv'))
                db_.to_csv(args.db_loc.replace(
                    'Id25c03_1merNCL.txt', 'K3822.csv'))
            elif loc == '1merNCLB':
                db_.to_csv(args.db_loc.replace(
                    'Id25c03_1merNCL.txt', 'K2369.csv'))
        args.db_loc = args.db_loc.replace('Id25c03_1merNCL.txt', 'K3822.csv')
        print(args.db_loc)

    if 'cdna' in args.db_loc:
        db_ = pd.read_csv(args.db_loc)
        db_.columns = ['uniprot_id', 'code', 'chain', 'position', 'wild_type',
                       'mutation', 'from', 'to', 'rel_rsa', 'ddG', 'sequence']
        db_['code'] = db_['code'].str.upper()
        db_['wild_type'] = db_['wild_type'].map(d)
        db_['mutation'] = db_['mutation'].map(d)
        db_['uid'] = db_['code']+'_'+db_['position'].astype(str)+db_['mutation']
        args.db_loc = args.db_loc.replace('.csv', '_mapped.csv')
        db_.to_csv(args.db_loc)
    
    db = pd.read_csv(args.db_loc)
    print('Loaded', args.db_loc, 'len =', len(db))
    
    dataset = args.dataset
    dataset_outname = args.dataset
    sym = False

    if 'fireprot' == dataset:
        dataset = 'fireprot'
        db = db.dropna(subset=['pdb_id'])
        db['code'] = db['pdb_id'].apply(lambda x: x.split('|')[0])
        db.loc[db['code']=='1HTI', 'position'] -= 37
        db.loc[db['code']=='1LVE', 'position'] -= 20
        db.loc[db['code']=='1ZNJ', 'chain'] = 'B'
        db.loc[(db['code']=='1ZNJ') & (db['wild_type']=='T'), 'chain'] = 'A'
    elif 's669' == dataset:
        dataset = 's669'
        db['code'] = db['Protein'].str[0:4]
        db['chain'] = db['Protein'].str[-1]
        db['wild_type'] = db['PDB_Mut'].str[0]
        db['position'] = db['PDB_Mut'].str[1:-1].astype(int)
        db['mutation'] = db['PDB_Mut'].str[-1]
        db['ddG'] = db['DDG_checked_dir']
    elif 'ssym' == dataset:
        sym = True
        dataset = 'ssym'
        db = db.rename({'PDB': 'structureD', 'PDB.1': 'structureR',
                        'ddG_D': 'ddGD', 'ddG_R': 'ddGR',
                        'MUT_D': 'MUTD', 'MUT_R': 'MUTR'}, axis=1)
        db_dir = db[[c for c in db.columns if c[-1] == 'D']]
        db_dir.columns = [c[:-1] for c in db_dir.columns]
        db_rev = db[[c for c in db.columns if c[-1] == 'R']]
        db_rev.columns = [c[:-1] for c in db_rev.columns]
        db = pd.concat([db_dir, db_rev])
        db['code'] = pd.concat([db['structure'][:342], db['structure'][:342]])
        db['code'] = db['code'].str[:4]
        db['wild_type'] = db['MUT'].str[0]
        db['chain'] = db['MUT'].str[1]
        db['position'] = db['MUT'].str[2:-1].astype(int)
        db['mutation'] = db['MUT'].str[-1]
    elif 'q3421' == dataset:
        dataset = 'q3421'
        db = db.rename({'PDB_ID': 'code', 'Chain ': 'chain', 
            'Wildtype': 'wild_type', 'Pos(PDB)': 'position', 
            'mutant ': 'mutation'}, axis=1)
    elif 'k3822' == dataset:
        dataset = 'k3822'
    elif 'ptmul_filtered' == dataset:
        db['code'] = db['pdb_id'].str[:-1]
        db['chain'] = db['chain_id']
        db['ddG'] = db['ddg']
        db['mut_info_seq_pos'] = None
        db = db.rename({'mut_seq': 'mut_seq_trunc', 'wt_seq': 'wt_seq_trunc'}, axis=1)
        print(db.loc[db['code']=='1QJP', 'mut_info'])
        db.loc[(db['code']=='1QJP') & (db['pos1']>17), 'pos1'] += 13
        db.loc[(db['code']=='1QJP') & (db['pos2']>17), 'pos2'] += 13
        db.loc[(db['code']=='1QJP') & (db['pos3']>17), 'pos3'] += 13
        db.loc[(db['code']=='1QJP') & (db['pos1']>76), 'pos1'] += 7
        db.loc[(db['code']=='1QJP') & (db['pos2']>76), 'pos2'] += 7
        db.loc[(db['code']=='1QJP') & (db['pos3']>76), 'pos3'] += 7
        db.loc[(db['code']=='1QJP') & (db['pos1']>150), 'pos1'] += 14
        db.loc[(db['code']=='1QJP') & (db['pos2']>150), 'pos2'] += 14
        db.loc[(db['code']=='1QJP') & (db['pos3']>150), 'pos3'] += 14
        db.loc[db['code']=='1QJP', 'mut_info'] = db.loc[db['code']=='1QJP', 'fr1'] +\
              db.loc[db['code']=='1QJP', 'pos1'].astype(str) + db.loc[db['code']=='1QJP', 'to1'] +\
        ':' + db.loc[db['code']=='1QJP', 'fr2'] + db.loc[db['code']=='1QJP', 'pos2'].astype(str) +\
              db.loc[db['code']=='1QJP', 'to2']
        db.loc[(db['code']=='1QJP') & (~db['pos3'].isna()), 'mut_info'] +=\
              ':' + db.loc[db['code']=='1QJP', 'fr3'] + db.loc[(db['code']=='1QJP') & (~db['pos3'].isna()), 'pos3'].astype(int).astype(str) + db.loc[db['code']=='1QJP', 'to3']
        db.loc[(db['code']=='1RHG') & (db['pos1']>100), 'pos1'] += 19
        db.loc[(db['code']=='1RHG') & (db['pos2']>100), 'pos2'] += 19
        db.loc[db['code']=='1RHG', 'mut_info'] = db.loc[db['code']=='1RHG', 'fr1'] +\
              db.loc[db['code']=='1RHG', 'pos1'].astype(str) + db.loc[db['code']=='1RHG', 'to1'] +\
        ':' + db.loc[db['code']=='1RHG', 'fr2'] + db.loc[db['code']=='1RHG', 'pos2'].astype(str) +\
              db.loc[db['code']=='1RHG', 'to2']
        db.loc[(db['code']=='1WQ5') & (db['pos1']>100), 'pos1'] += 8
        db.loc[(db['code']=='1WQ5') & (db['pos2']>100), 'pos2'] += 8
        db.loc[(db['code']=='1WQ5') & (db['pos1']>200), 'pos1'] += 2
        db.loc[(db['code']=='1WQ5') & (db['pos2']>200), 'pos2'] += 2
        db.loc[db['code']=='1WQ5', 'mut_info'] = db.loc[db['code']=='1WQ5', 'fr1'] +\
              db.loc[db['code']=='1WQ5', 'pos1'].astype(str) + db.loc[db['code']=='1WQ5', 'to1'] +\
        ':' + db.loc[db['code']=='1WQ5', 'fr2'] + db.loc[db['code']=='1WQ5', 'pos2'].astype(str) +\
              db.loc[db['code']=='1WQ5', 'to2']
        db.loc[(db['code']=='2WSY') & (db['pos1']>150), 'pos1'] += 9
        db.loc[(db['code']=='2WSY') & (db['pos2']>150), 'pos2'] += 9
        db.loc[(db['code']=='2WSY') & (db['pos1']>185), 'pos1'] += 19
        db.loc[(db['code']=='2WSY') & (db['pos2']>185), 'pos2'] += 19
        db.loc[(db['code']=='2WSY') & (db['pos1']>230), 'pos1'] += 1
        db.loc[(db['code']=='2WSY') & (db['pos2']>230), 'pos2'] += 1
        db.loc[db['code']=='2WSY', 'mut_info'] = db.loc[db['code']=='2WSY', 'fr1'] +\
              db.loc[db['code']=='2WSY', 'pos1'].astype(str) + db.loc[db['code']=='2WSY', 'to1'] +\
        ':' + db.loc[db['code']=='2WSY', 'fr2'] + db.loc[db['code']=='2WSY', 'pos2'].astype(str) +\
              db.loc[db['code']=='2WSY', 'to2']
        print(db.loc[db['code']=='1QJP', 'mut_info'])
    elif 'ptmul_orig' == dataset:
        db['code'] = db['PDB']
        db['chain'] = db['CHAIN']
        db['PDB'] = db['PDB'] + db['CHAIN']
        db['pdb_id'] = db['PDB']
        db['ddG'] = db['DDG']
        db['ddg'] = db['DDG']
        db['mut_info_seq_pos'] = None
        db['mut_info'] = db['MUTS'].str.replace(';', ':')
        db['mut_info'] = db['mut_info'].str.replace('Q28N:Y27DD', 'Q27N:Y27DD')
        dataset_outname = 'ptmul'
    elif 's571' == dataset:
        db['code'] = db['name'].str[5:9]
        db['chain'] = db['name'].str[10:11]
        db['mut_info'] = db['name'].apply(lambda x: x.split('_')[-2])
        db['wild_type'] = db['mut_info'].str[0]
        db['position'] = db['mut_info'].str[1:-1]
        db['mutation'] = db['mut_info'].str[-1]
        print(db.head())
        dataset_outname = 's571'
    elif 's4346' == dataset:
        db['code'] = db['name'].apply(lambda x: x.split('_')[1])
        db['chain'] = db['name'].apply(lambda x: x.split('_')[2])
        db['mut_info'] = db['name'].apply(lambda x: x.split('_')[3])
        db['wild_type'] = db['mut_info'].str[0]
        db['position'] = db['mut_info'].str[1:-1]
        db['mutation'] = db['mut_info'].str[-1]
        print(db.head())
        dataset_outname = 's4346'
    elif 's783' == dataset:
        db['code'] = db['name'].str[5:9]
        db['chain'] = db['name'].str[10:11]
        db['mut_info'] = db['name'].apply(lambda x: x.split('_')[-3])
        db['wild_type'] = db['mut_info'].str[0]
        db['position'] = db['mut_info'].str[1:-1]
        db['mutation'] = db['mut_info'].str[-1]
        print(db.head())
        dataset_outname = 's783'
    elif 's2000' == dataset:
        raise NotImplementedError 
    elif 's2648' == dataset:
        db['code'] = db['PDB']
        db['chain'] = db['CHAIN']
        db['mut_info'] = db['MUT']
        db['wild_type'] = db['mut_info'].str[0]
        db['position'] = db['mut_info'].str[1:-1]
        db['mutation'] = db['mut_info'].str[-1]
        db['ddG'] = db['DDG']
        print(db.head())
        dataset_outname = 's2648'
    elif 's8754' == dataset:
        db['name'] = db['name'].str.replace('__M01', '__M1')
        db['code'] = db['name'].apply(lambda x: x.split('_')[1])
        db['chain'] = db['name'].apply(lambda x: x.split('_')[2])
        db['chain'] = db['chain'].apply(lambda x: 'A' if x=='' else x)
        db['mut_info'] = db['name'].apply(lambda x: x.split('_')[3])
        db['wild_type'] = db['mut_info'].str[0]
        db['position'] = db['mut_info'].str[1:-1]
        db['mutation'] = db['mut_info'].str[-1]
        print(db.head())
        dataset_outname = 's8754'
    elif 'grb2_abundance' == dataset:
        db['code'] = '2VWF'
        db['chain'] = 'A'
    elif 'grb2_binding' == dataset:
        db['code'] = '2VWF'
        db['chain'] = 'A'
    elif 'dlg4_abundance' == dataset:
        db['code'] = '1BE9'
        db['chain'] = 'A'
    elif 'dlg4_binding' == dataset:
        db['code'] = '1BE9'
        db['chain'] = 'A'
    elif 'myo_kung' == dataset:
        raise NotImplementedError
    elif 'esta_nutschel' == dataset:
        raise NotImplementedError
    elif 'gb1_wu' == dataset:
        raise NotImplementedError
    else:
        raise NotImplementedError      
    
    if sym:
        db['uid'] = db['structure'] + db['chain'] + '_' + db['wild_type'] + db['position'].astype(str) + db['mutation']
        grouper = ['code', 'structure', 'chain']
    elif dataset in ['ptmul_filtered', 'ptmul_orig', 'grb2_abundance', 'grb2_binding', 'dlg4_abundance', 'dlg4_binding']:
        db['uid'] = db['code'] + db['chain'] + '_' + db['mut_info']
        grouper = ['code', 'code', 'chain']
    else:
        db['uid'] = db['code'] + db['chain'] + '_' + db['wild_type'] + db['position'].astype(str) + db['mutation']
        grouper = ['code', 'code', 'chain']
    
    db = db.set_index('uid')
    os.makedirs(os.path.join(REPO_ROOT, 'data/structures/'), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, 'data/structures/single_chain'), exist_ok=True)

    for (code, struct, chain), group in db.groupby(grouper):
        if chain == '':
            chain = 'A'
        pdb_file = os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_processed.pdb')
        db.loc[(db['code' if not sym else 'structure']==struct) & (db['chain']==chain), 'pdb_file'] = pdb_file
        if not os.path.exists(pdb_file):
            if len(struct) == 4:
                result = download_pdb(struct, os.path.join(REPO_ROOT, 'esm-msr/data/structures/'), dataset=dataset_outname.upper())
                os.rename(os.path.join(REPO_ROOT, f'data/structures/{struct.lower()}.pdb'), os.path.join(REPO_ROOT, 'esm-msr/data/structures/{struct}_{chain}_original.pdb'))
            else:
                result = get_alphafold_structure(struct, os.path.join(REPO_ROOT, 'esm-msr/data/structures/{struct}.pdb'), sequence=group['wt_seq'].head(1).item())
                os.rename(os.path.join(REPO_ROOT, f'data/structures/{struct}.pdb'), os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_original.pdb'))
            remove_caps(os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_original.pdb'))
            remove_heteroatoms(os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_original.pdb'), os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'))
            fix_noncanonical_residues(os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'), os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'))
            renumber_pdb(os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'), os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'))
            repair_pdb(os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_intermediate.pdb'), pdb_file, sequence_file=result['fasta'], chain_id=chain, verbose=True)
            add_lines(pdb_file)
    
    for (code, struct, chain), group in db.groupby(grouper):
        pdb_file = group['pdb_file'].head(1).item()
        wt_seq = get_seq(pdb_file, chain)
        original_pdb = os.path.join(REPO_ROOT, f'data/structures/{struct}_{chain}_original.pdb')
        repaired_pdb = pdb_file
        single_chain_loc = pdb_file.split('/')
        single_chain_loc.insert(-1, 'single_chain')
        single_chain_loc = '/'.join(single_chain_loc)
        prepare_single_chain(pdb_file, chain, single_chain_loc)
        orig_to_new, orig, new = create_residue_mapping(original_pdb, repaired_pdb, chain)

        if not dataset in ['ptmul_filtered', 'ptmul_orig', 'm1261']:
            for i, row in group.iterrows():
                print(orig_to_new)
                db.at[i, 'wt_seq'] = wt_seq
                wt = row['wild_type']
                pos = orig_to_new[str(row['position'])]
                db.at[i, 'seq_pos'] = pos
                mt = row['mutation']
                print(code, wt, pos, mt)
                print(wt_seq)
                assert wt_seq[pos-1] == wt, (code, wt, pos, mt)
                mut_seq = list(wt_seq)
                mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
        elif dataset == 'ptmul_orig':
            for i, row in group.iterrows():
                db.at[i, 'orig_to_new'] = str(orig_to_new)
                mut_seq = list(wt_seq)
                db.at[i, 'wt_seq'] = wt_seq
                seq_pos_list = []
                mut_info_seq_pos = ''
                for mut in reorder_muts(row['mut_info'].split(':')):
                    if not mut_info_seq_pos == '':
                        mut_info_seq_pos += ':'
                    wt = mut[0]
                    pos = orig_to_new[mut[1:-1]]
                    seq_pos_list.append(pos)
                    mt = mut[-1]
                    mut_info_seq_pos += f'{wt}{pos}{mt}'
                    assert wt_seq[pos-1] == wt
                    mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
                db.at[i, 'mut_info_seq_pos'] = mut_info_seq_pos
        elif dataset == 'ptmul_filtered':
            for i, row in group.iterrows():
                db.at[i, 'orig_to_new'] = str(orig_to_new)
                mut_seq = list(wt_seq)
                db.at[i, 'wt_seq'] = wt_seq
                seq_pos_list = []
                mut_info_seq_pos = ''
                for mut in reorder_muts(row['mut_info'].split(':')):
                    if not mut_info_seq_pos == '':
                        mut_info_seq_pos += ':'
                    wt = mut[0]
                    pos = int(mut[1:-1]) + list(orig_to_new.values())[0] -1
                    if code == '1ONC':
                        pos -= 1
                    seq_pos_list.append(pos)
                    mt = mut[-1]
                    mut_info_seq_pos += f'{wt}{pos}{mt}'
                    assert wt_seq[pos-1] == wt
                    mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
                db.at[i, 'mut_info_seq_pos'] = mut_info_seq_pos
        elif dataset == 'm1261':
            for i, row in group.iterrows():
                print(row)
                db.at[i, 'orig_to_new'] = str(orig_to_new)
                assert wt_seq == row['wt_seq'], f'{row["code"]}\n{wt_seq}\n{row["wt_seq"]}'
                mut_seq = list(wt_seq)
                db.at[i, 'wt_seq'] = wt_seq
                seq_pos_list = []
                mut_info_seq_pos = ''
                for mut in reorder_muts(row['mut_info'].split(':')):
                    if not mut_info_seq_pos == '':
                        mut_info_seq_pos += ':'
                    wt = mut[0]
                    pos = orig_to_new[mut[1:-1]]
                    seq_pos_list.append(pos)
                    mt = mut[-1]
                    mut_info_seq_pos += f'{wt}{pos}{mt}'
                    print(mut_info_seq_pos)
                    print('wt_seq', wt_seq)
                    assert wt_seq[pos-1] == wt
                    mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
                db.at[i, 'mut_info_seq_pos'] = mut_info_seq_pos

    os.makedirs(args.output, exist_ok=True)

    if sym:
        db = db.rename({'code': 'wt_code'}, axis=1).rename({'structure': 'code'}, axis=1)
    if dataset in ['ptmul_filtered' , 'ptmul_orig', 'ptmul']:
        db = parse_mutation_column_to_separate_columns(db, 'mut_info_seq_pos')
    db.to_csv(os.path.join(args.output, f'{dataset_outname}_mapped.csv'))

    if dataset_outname == 's669':
        db_full = db.copy(deep=True)
        db_full['uid2'] = db['code'] + '_' + db['PDB_Mut'].str[1:]
        db_full = db_full.reset_index().set_index('uid2')
        db_full = db_full.rename({'ddG': 'ddG_s669'}, axis=1)

        s461 = pd.read_csv(os.path.join(REPO_ROOT, '/data/external_datasets/S461.csv'))
        s461['uid2'] = s461['PDB'] + '_' + s461['MUT_D'].str[2:]
        s461 = s461.set_index('uid2')
        s461['ddG_I'] = -s461['ddG_D']
        s461.columns = [s+'_dir' for s in s461.columns]
        s461 = s461.rename(
            {'ddG_D_dir': 'ddG', 'ddG_I_dir': 'ddG_inv'}, axis=1)
        
        db = s461.join(db_full, how='left').reset_index(drop=True)
        assert len(db) == 461
        db.set_index('uid').to_csv(os.path.join(args.output, 's461_mapped.csv'))

    if dataset_outname == 'k3822':
        k2369 = pd.read_csv(os.path.join(REPO_ROOT, '/data/external_datasets/K2369.csv').set_index('uid'))
        db = db.loc[k2369.index]
        assert len(db) == 2369
        db.to_csv(os.path.join(args.output, 'k2369_mapped.csv'))

    if dataset_outname == 'ptmul_filtered':
        ptmuld = pd.read_csv(os.path.join(REPO_ROOT, '/data/external_datasets/PTMUL-D.csv'))
        ptmuld = ptmuld.rename({'PDB': 'pdb_id', 'SEQ': 'wt_seq_trunc'}, axis=1)
        ptmuld = ptmuld.merge(db[['pdb_id', 'wt_seq_trunc', 'orig_to_new', 'wt_seq']].drop_duplicates(), on=['pdb_id', 'wt_seq_trunc'], how='left')
        print(ptmuld)

        for i, row in ptmuld.iterrows():
            seq = row['wt_seq']
            seq = list(seq)
            muts = row['MUTS'].split(';')
            orig_to_new = eval(row['orig_to_new'])
            
            for mut in muts:
                wt = mut[0]
                pos = orig_to_new[mut[1:-1]]
                mt = mut[-1]
                assert seq[pos-1] == wt
                seq[pos-1] = mt

            seq = ''.join(seq)
            ptmuld.at[i, 'mut_seq'] = seq
        
        db = db.rename({'ddG': 'ddG_ptmul'}, axis=1).reset_index()
        ptmuld = ptmuld.rename({'DDG': 'ddG'}, axis=1)
        ptmuld['ddG'] *= -1
        db = ptmuld.merge(db.drop(['pdb_id', 'wt_seq_trunc', 'orig_to_new', 'wt_seq'], axis=1), on='mut_seq').set_index('uid')

        assert len(db) == 536

        db.to_csv(os.path.join(args.output, 'ptmuld_mapped.csv'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output', type=str, default='../data/preprocessed/')
    parser.add_argument('--modlib_dir', type=str, default='/usr/lib/modeller10.4/modlib/')
    parser.add_argument('--modeller_dir', type=str, default='/usr/lib/modeller10.4/lib/x86_64-intel8/python3.3/')
    args = parser.parse_args()

    sys.path.append(args.modlib_dir)
    sys.path.append(args.modeller_dir)

    try:
        from modeller import *
        from modeller.automodel import *
    except ImportError:
        pass # Explicitly ignore, environment might not be local
            
    if args.dataset.lower() in ['q3421']:
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/Q3421.csv')
    elif args.dataset.lower() in ['fireprot', 'fireprotdb']:
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/fireprotdb_results.csv')
    elif args.dataset.lower() in ['s669', 's461']:
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/Data_s669_with_predictions.csv')
        args.dataset = 's669'
    elif args.dataset.lower() in ['ssym']:
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/ssym.csv')
    elif args.dataset.lower() in ['korpm', 'korpm_reduced', 'k2369', 'k3822']:
        args.dataset = 'k3822'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/Id25c03_1merNCL.txt')
    elif args.dataset.lower() in ['ptmul_filtered', 'ptmuld']:
        args.dataset = 'ptmul_filtered'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/protherm_multiple.csv')
    elif args.dataset.lower() in ['ptmul_orig', 'ptmul']:
        args.dataset = 'ptmul_orig'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/ptmul.csv')
    elif args.dataset.lower() in ['s571']:
        args.dataset = 's571'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/S571.csv')
    elif args.dataset.lower() in ['s4346']:
        args.dataset = 's4346'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/S4346.csv')
    elif args.dataset.lower() in ['s783']:
        args.dataset = 's783'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/S783.csv')
    elif args.dataset.lower() in ['s2648']:
        args.dataset = 's2648'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/S2648.csv')
    elif args.dataset.lower() in ['s8754']:
        args.dataset = 's8754'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/S8754.csv')
    elif 'grb2_binding' in args.dataset.lower():
        args.dataset = 'grb2_binding'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/GRB2_HUMAN_Faure_2021_binding_domain.csv')
    elif 'grb2_abundance' in args.dataset.lower():
        args.dataset = 'grb2_abundance'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/GRB2_HUMAN_Faure_2021_abundance_domain.csv')
    elif 'dlg4_binding' in args.dataset.lower():
        args.dataset = 'dlg4_binding'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/DLG4_HUMAN_Faure_2021_binding_domain.csv')
    elif 'dlg4_abundance' in args.dataset.lower():
        args.dataset = 'dlg4_abundance'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/DLG4_HUMAN_Faure_2021_abundance_domain.csv')
    elif 'myo' in args.dataset.lower():
        args.dataset = 'myo_kung'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/MYO_HUMAN_Kung_2025_display.csv')
    elif 'gb1' in args.dataset.lower():
        args.dataset = 'gb1_wu'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/GB1_Wu_2016_binding_domain.csv')
    elif 'esta' in args.dataset.lower():
        args.dataset = 'esta_nutschel'
        args.db_loc = os.path.join(REPO_ROOT, 'data/external_datasets/ESTA_BACSU_Nutschel_2020.csv')

    else:
        print('Inferred use of user-created database. Note: this must '
                'contain columns for code, wild_type, position, mutation. '
                'position must correspond to PDB sequence')
        assert args.dataset != 'fireprot'
        assert args.db_loc is not None

    main(args)