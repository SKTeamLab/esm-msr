import requests
import pandas as pd
import sys
import os
import tempfile
import shutil
import re
import argparse

from esm.utils.structure.protein_chain import ProteinChain

from pdbfixer import PDBFixer
from openmm.app import PDBFile

from Bio import pairwise2
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select


def parse_mutation_column_to_separate_columns(df, column_name):
    """
    Parse a DataFrame column containing mutation strings into separate columns
    for each mutation component (fr1, pos1, to1, fr2, pos2, to2, etc.).
    
    Args:
        df (pandas.DataFrame): DataFrame containing the mutation column
        column_name (str): Name of column containing mutation strings
    
    Returns:
        pandas.DataFrame: DataFrame with additional columns for parsed mutations
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy(deep=True)
    
    # First, determine the maximum number of mutations in any entry
    max_mutations = 0
    for mutation_string in df[column_name]:
        if pd.isna(mutation_string) or mutation_string == '':
            continue
        
        mutations = mutation_string.split(':')
        max_mutations = max(max_mutations, len(mutations))
    
    # Initialize empty columns for each mutation component
    for i in range(1, max_mutations + 1):
        result_df[f'fr{i}'] = None
        result_df[f'pos{i}'] = None
        result_df[f'to{i}'] = None
    
    # Parse each row
    for idx, mutation_string in result_df[column_name].items():
        if pd.isna(mutation_string) or mutation_string == '':
            continue
  
        mutations = mutation_string.split(':')
        #print(idx, mutations)
        
        for i, part in enumerate(mutations, 1):
            # Extract components using regex
            match = re.match(r'([A-Za-z])(\d+)([A-Za-z])', part)
            
            if match:
                result_df.at[idx, f'fr{i}'] = match.group(1)
                result_df.at[idx, f'pos{i}'] = int(match.group(2))
                result_df.at[idx, f'to{i}'] = match.group(3)
            else:
                print(f"Warning: Could not parse mutation '{part}'")
    
    return result_df

def download_pdb(pdb_id, output_dir='.', file_format='pdb', get_fasta=True):
    """
    Download a PDB file from the RCSB PDB database and optionally its FASTA sequence.
    
    Parameters:
    -----------
    pdb_id : str
        The 4-character PDB ID code
    output_dir : str
        Directory where the file(s) will be saved
    file_format : str
        Format of the file ('pdb' or 'cif')
    get_fasta : bool
        Whether to also download the FASTA sequence
        
    Returns:
    --------
    dict
        Dictionary with paths to the downloaded files
    """
    # Clean up the PDB ID
    pdb_id = pdb_id.lower().strip()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    result = {'pdb': None, 'fasta': None}
    
    # Determine the file extension and URL for structure
    if file_format.lower() == 'pdb':
        file_ext = '.pdb'
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    elif file_format.lower() == 'cif':
        file_ext = '.cif'
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    else:
        raise ValueError("Format must be 'pdb' or 'cif'")
    
    # Create output filename for structure
    output_file = os.path.join(output_dir, f"{pdb_id}{file_ext}")
    
    # Download the structure file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {pdb_id} structure to {output_file}")
        result['pdb'] = output_file
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdb_id} structure: {str(e)}")
    
    # Download FASTA if requested
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
            if verbose:
                print(f"Could not remove {chain_id}:{res_id} - {str(e)}")

    # Write the final structure
    io = PDBIO()
    io.set_structure(original_model)
    io.save(input_pdb)

def remove_heteroatoms(pdb_file, output_file, verbose=False):
    """
    Removes all heteroatoms, ligands, and water molecules from a PDB file,
    preserving only standard protein/nucleic acid chains.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the input PDB file
    output_file : str
        Path to save the cleaned PDB file
    verbose : bool, optional
        Whether to print detailed progress messages
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    pdb_file = os.path.abspath(pdb_file)
    output_file = os.path.abspath(output_file)
    
    if not os.path.exists(pdb_file):
        if verbose: print(f"Error: Input file {pdb_file} does not exist.")
        return False

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # Define a selector class that filters out HETATMs and Waters
    class StandardResidueSelect(Select):
        def accept_residue(self, residue):
            # Bio.PDB residue ID is a tuple (hetero_flag, sequence_identifier, insertion_code)
            # hetero_flag is ' ' for standard residues (ATOM), 'W' for water, and 'H_...' for heteroatoms
            hetero_flag = residue.id[0]
            
            # Reject waters ('W') and heteroatoms ('H_...')
            if hetero_flag != ' ':
                return False
            else:
                return True

    # Save the structure using the selector
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, select=StandardResidueSelect())
    
    if verbose:
        print(f"Cleaned structure saved to {output_file}")
        
    return True


def fix_noncanonical_residues(input_pdb, output_pdb, verbose=False):
    """
    Fix non-canonical residues while preserving original chain IDs by
    mapping sequential PDBFixer chains (A,B,C...) back to the original chains.
    
    Parameters:
    -----------
    input_pdb : str
        Path to input PDB file
    output_pdb : str
        Path to output PDB file
    verbose : bool
        Whether to print information about replaced residues
    """
    # Step 1: Collect original chain order
    parser = PDBParser(QUIET=True)
    original_model = parser.get_structure('original', input_pdb)[0]
    
    # Get original chains in the order they appear
    original_chains = []
    for chain in original_model:
        if chain.id not in [c[0] for c in original_chains]:
            # Count residues to handle potential empty chains
            num_residues = len(list(chain.get_residues()))
            original_chains.append((chain.id, num_residues))
    
    # Step 2: Collect information about non-canonical residues in the original structure
    noncanonical_residues = {}
    std_aa = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
              "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
              "THR", "TRP", "TYR", "VAL"]
    
    for chain in original_model:
        for residue in chain:
            resname = residue.get_resname()
            # Check if non-standard
            if resname not in std_aa:
                # Store position and name
                res_id = residue.get_id()[1]
                res_key = f"{chain.id}:{res_id}"
                noncanonical_residues[res_key] = resname
    
    # Step 3: Use PDBFixer to fix non-canonical residues
    fixer = PDBFixer(input_pdb)
    fixer.findNonstandardResidues()
    
    # Store a copy of the topology before replacement
    original_topology = {}
    for chain in fixer.topology.chains():
        chain_dict = {}
        for residue in chain.residues():
            # Store both name and ID
            chain_dict[residue.id] = residue.name
        original_topology[chain.id] = chain_dict
    
    # Replace non-standard residues
    fixer.replaceNonstandardResidues()
    
    # Create temporary fixed PDB
    temp_output = 'temp_fixed.pdb'
    PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_output, 'w'))
    
    # Step 4: Map the sequential chains back to original chains
    fixed = parser.get_structure('fixed', temp_output)
    
    # Get the fixed chains in order
    fixed_chains = []
    for model in fixed:
        for chain in model:
            if chain.id not in [c[0] for c in fixed_chains]:
                num_residues = len(list(chain.get_residues()))
                fixed_chains.append((chain.id, num_residues))
    
    # Create chain ID mapping based on order and residue counts
    chain_id_map = {}
    
    # Match chains by their position and approximate size
    original_idx = 0
    for fixed_idx, (fixed_id, fixed_res_count) in enumerate(fixed_chains):
        # Skip to next original chain if current one is exhausted
        while original_idx < len(original_chains) and original_chains[original_idx][1] == 0:
            original_idx += 1
            
        if original_idx < len(original_chains):
            orig_id = original_chains[original_idx][0]
            chain_id_map[fixed_id] = orig_id
            original_idx += 1
    
    # Apply the mapping to the fixed structure
    for model in fixed:
        for chain in model:
            if chain.id in chain_id_map:
                chain.id = chain_id_map[chain.id]
    
    # Step 5: Compare topologies to find all changes
    if verbose:
        # Store the new topology
        new_topology = {}
        for model in fixed:
            for chain in model:
                chain_dict = {}
                for residue in chain:
                    # Get residue ID (number)
                    res_id = residue.get_id()[1]
                    # Get residue name
                    res_name = residue.get_resname()
                    chain_dict[res_id] = res_name
                new_topology[chain.id] = chain_dict
        
        # Find all changes by comparing original to fixed structure
        changes = []
        
        # For each original chain
        for orig_chain_id, orig_chain_data in original_topology.items():
            # Find the corresponding chain in the new structure
            new_chain_id = None
            for fixed_id, orig_id in chain_id_map.items():
                if orig_id == orig_chain_id:
                    new_chain_id = orig_id
                    break
            
            if new_chain_id and new_chain_id in new_topology:
                # Compare residues
                for res_id, old_name in orig_chain_data.items():
                    if res_id in new_topology[new_chain_id]:
                        new_name = new_topology[new_chain_id][res_id]
                        if old_name != new_name and old_name not in std_aa:
                            changes.append(f"Chain {new_chain_id}, Residue {res_id}: {old_name} → {new_name}")
        
        # Print a summary of changes
        if changes:
            print(f"\nNon-canonical residues replaced in {input_pdb}:")
            for change in changes:
                print(f"  {change}")
            print(f"Total: {len(changes)} residues replaced\n")
        else:
            print(f"No non-canonical residues were replaced in {input_pdb}")

        # Add this code before writing the final structure
        # Check for any remaining non-standard residues after PDBFixer
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
                
                # Remove non-standard residues that PDBFixer couldn't handle
                for res_id in residues_to_remove:
                    try:
                        chain.detach_child(res_id)
                        if verbose:
                            print(f"Removed non-standard residue: Chain {chain.id}, Residue {res_id[1]}: {remaining_nonstandard[f'{chain.id}:{res_id[1]}']}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed to remove residue: Chain {chain.id}, Residue {res_id[1]}: {str(e)}")

        # Report on removed residues
        if remaining_nonstandard and verbose:
            print(f"\nRemoved {len(remaining_nonstandard)} remaining non-standard residues that PDBFixer couldn't convert")  
            
    # Write the final structure
    io = PDBIO()
    io.set_structure(fixed)
    io.save(output_pdb)
    
    # Clean up
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    return output_pdb

def renumber_pdb(pdb_file, output_file):
    """
    Renumbers the residues in the PDB file sequentially
    """

    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # Temporarily renumber the residues with a large offset to avoid conflicts
    # (where two residues share the same identity)
    offset = 10000
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_list(), start=1):
                residue.id = (' ', i + offset, ' ')

    # Sequentially renumber the residues (starting from 1)
    for model in structure:
        for chain in model:
            residues = sorted(chain.get_list(), key=lambda res: res.get_id()[1])
            for i, residue in enumerate(residues, start=1):
                residue.id = (' ', i, ' ')

    # Write the output file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def repair_pdb(pdb_file, output_file, sequence_file=None, chain_id='A', 
               num_models=1, use_dope=True, verbose=False, return_all_chains=True, debug_dir=None):
    """
    Repairs missing atoms and residues in a PDB file using Modeller.
    
    Now correctly handles PDB-style FASTA headers like:
    >1FH5_2|Chain B[auth H]|MONOCLONAL ANTIBODY...
    
    If chain_id='H', it prioritizes the [auth H] match over Chain B.
    """
    # 1. Setup and Validation
    pdb_file = os.path.abspath(pdb_file)
    output_file = os.path.abspath(output_file)
    
    filename = os.path.basename(pdb_file)
    pdb_id = filename[:4] if len(filename) >= 4 else filename.split('.')[0]

    original_cwd = os.getcwd()
    
    # Handle Debug/Temp Directory
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
            
            # --- STEP A: ISOLATE THE TARGET CHAIN ---
            parser = PDBParser(QUIET=True)
            try:
                original_structure = parser.get_structure('original', pdb_file)
            except Exception as e:
                if verbose: print(f"Failed to parse {pdb_file}: {e}")
                return False

            # Check if chain exists
            chain_found = False
            for chain in original_structure.get_chains():
                if chain.id == chain_id:
                    chain_found = True
                    chain.id = chain_id # Force rename to A for isolation
                    break
            
            if not chain_found:
                if verbose: print(f"Chain {chain_id} not found in {pdb_file}")
                return False

            # Save the isolated chain
            class TargetChainSelect(Select):
                def accept_chain(self, chain):
                    return chain.get_id() == chain_id
            
            isolated_pdb_name = f"{pdb_id}_{chain_id}_isolated.pdb"
            io = PDBIO()
            io.set_structure(original_structure)
            io.save(isolated_pdb_name, select=TargetChainSelect())
            
            # --- SETUP MODELLER ---
            env = Environ()
            env.io.atom_files_directory = ['.']
            env.libs.topology.read(file='$(LIB)/top_heav.lib')
            env.libs.parameters.read(file='$(LIB)/par.lib')
            
            pdb_code = f"{pdb_id}_{chain_id}"
            temp_pdb_link = os.path.join(temp_dir, f"{pdb_code}.pdb")
            if os.path.exists(temp_pdb_link):
                os.remove(temp_pdb_link)
            os.symlink(os.path.abspath(isolated_pdb_name), temp_pdb_link)

            # Analyze the input 
            mdl = Model(env, file=temp_pdb_link)
            
            # Find the chain in the Modeller model (we forced it to A)
            target_chain_modeller = next((c for c in mdl.chains if c.name == chain_id), None)
            if not target_chain_modeller:
                 target_chain_modeller = list(mdl.chains)[0]

            pdb_seq = ''.join([residue.code for residue in target_chain_modeller.residues])
            
            # 2. Sequence Handling (FIXED: Robust FASTA Parsing)
            complete_seq = None
            if sequence_file:
                if not os.path.isabs(sequence_file):
                    sequence_file = os.path.join(original_cwd, sequence_file)
                
                # We need to store matches to prioritize Auth > PDB > Generic
                strong_match_seq = None # Matches [auth ID]
                weak_match_seq = None   # Matches Chain ID
                generic_match_seq = None # Matches header string
                
                current_header = None
                current_seq_parts = []
                
                def process_record(header, sequence):
                    nonlocal strong_match_seq, weak_match_seq, generic_match_seq
                    
                    if not all(c in "ACTG" for c in sequence[:20]):
                        # 1. Extract the chain list part: "Chain A" or "Chains A, B"
                        # Matches: "Chain(s) <content> |" or end of string
                        # Group 2 contains the list string e.g. "A, B[auth C]"
                        chain_block_match = re.search(r'Chain(s)?\s+([^|]+)', header)
                        
                        if chain_block_match:
                            chain_str = chain_block_match.group(2).strip()
                            
                            # Split by commas to handle "A, B"
                            parts = [p.strip() for p in chain_str.split(',')]
                            
                            for part in parts:
                                # Parse specific item: "A" or "A[auth B]"
                                # Group 1: PDB ID
                                # Group 2: Auth ID (optional)
                                part_match = re.match(r'([^\[\s]+)(?:\[auth\s+([^\]]+)\])?', part)
                                
                                if part_match:
                                    pdb_c = part_match.group(1)
                                    auth_c = part_match.group(2)
                                    
                                    # Check priorities
                                    if auth_c and auth_c == chain_id:
                                        strong_match_seq = sequence
                                    elif pdb_c == chain_id:
                                        weak_match_seq = sequence
                        
                        # Fallback simple string check
                        if f"Chain {chain_id}" in header or f"Chains {chain_id}" in header:
                             # This is slightly safer than just checking "A" which matches everything
                             generic_match_seq = sequence
                        elif chain_id in header: 
                             # Last resort generic match
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
                    
                    # Process last record
                    if current_header:
                        process_record(current_header, "".join(current_seq_parts))
                
                # SELECT THE BEST MATCH
                if strong_match_seq:
                    complete_seq = strong_match_seq
                    if verbose: print(f"Using sequence from AUTH match for chain {chain_id}")
                elif weak_match_seq:
                    complete_seq = weak_match_seq
                    if verbose: print(f"Using sequence from CHAIN match for chain {chain_id}")
                elif generic_match_seq:
                    complete_seq = generic_match_seq
                    if verbose: print(f"Using sequence from GENERIC header match for chain {chain_id}")
                else:
                    # If strictly checking, we might want to fail here, but fallback to PDB seq is current behavior
                    pass

                if complete_seq is None:
                     print(f"Warning: Could not find sequence for chain {chain_id} in {sequence_file}")
                     complete_seq = pdb_seq
            else:
                complete_seq = pdb_seq
            
            complete_seq = complete_seq.strip('X')

            # 3. Check if Repair is Needed
            if pdb_seq == complete_seq and not has_missing_atoms(mdl, target_chain_modeller.name):
                if verbose: print(f"No repair needed for {pdb_file}")
                clean_chain_file = isolated_pdb_name 
            else:
                # 4. Alignment
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
                
                # 5. Run Modeller
                class MyCompleteModel(AutoModel):
                    def special_patches(self, aln): pass

                a = MyCompleteModel(env, alnfile=aln_file, knowns=pdb_code, sequence=target_code)
                a.starting_model = 1
                a.ending_model = num_models
                
                if use_dope:
                    a.assess_methods = (assess.DOPE, assess.GA341)
                
                a.make()
                
                # 6. Select Best Model
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
                
                # 7. Post-Processing: Restore Original ID
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

            # --- MERGING LOGIC ---
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
            if verbose: print(f"Error repairing {pdb_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            os.chdir(original_cwd)

def add_lines(file):
        # add in a CRYST1 line so that DSSP will accept the file
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
    """Check if a structure has missing atoms in any residue"""
    for chain in model.chains:
        if chain.name == chain_id:
            for res in chain.residues:
                # Check if residue has standard number of atoms
                if is_missing_atoms(res):
                    return True
    return False

def is_missing_atoms(residue):
    """Check if a single residue has missing atoms based on residue type"""
    # Dictionary of expected heavy atom counts for each residue type
    expected_atoms = {
        'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6, 'GLN': 9, 
        'GLU': 9, 'GLY': 4, 'HIS': 10, 'ILE': 8, 'LEU': 8, 'LYS': 9,
        'MET': 8, 'PHE': 11, 'PRO': 7, 'SER': 6, 'THR': 7, 'TRP': 14,
        'TYR': 12, 'VAL': 7
    }
    
    if residue.code in expected_atoms:
        # Count heavy atoms (non-hydrogen)
        heavy_atom_count = sum(1 for a in residue.atoms if a.element != 'H')
        return heavy_atom_count < expected_atoms[residue.code]
    
    return False

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 
    'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 
    'TYR': 'Y', 'MET': 'M', 'MSE': 'Z', 'UNK': '9'} 

def create_residue_mapping(original_pdb, repaired_pdb, chain):
    """Create mapping between original and repaired residue numbering"""
    
    # Extract sequences with residue IDs
    parser = PDBParser(QUIET=True)
    original = parser.get_structure('original', original_pdb)
    repaired = parser.get_structure('repaired', repaired_pdb)
    
    # Get sequences with corresponding residue IDs
    orig_seq = []
    orig_ids = []
    for residue in original[0][chain]:
        if residue.id[0] == ' ':  # Standard residue
            orig_seq.append(d[residue.resname])
            orig_ids.append(str(residue.id[1])+str(residue.id[2]).strip(' '))
    
    repair_seq = []
    repair_ids = []
    for residue in repaired[0][chain]:
        if residue.id[0] == ' ':
            repair_seq.append(d[residue.resname])
            repair_ids.append(residue.id[1])
    
    # Align sequences
    alignment = pairwise2.align.globalms(''.join(orig_seq), ''.join(repair_seq), 
                                        2, -1, -0.5, -0.1, one_alignment_only=True)[0]
    
    # Create mapping
    orig_to_new = {}
    orig_idx = 0
    repair_idx = 0

    #print('original', alignment[0])
    #print('repaired', alignment[1])
    
    i = 0
    while True:
        try:
            if alignment[0][i] != '-' and alignment[1][i] != '-':
                # Both sequences have a residue here
                orig_to_new[orig_ids[orig_idx]] = repair_ids[repair_idx]
                orig_idx += 1
                repair_idx += 1
            elif alignment[0][i] != '-':
                # Gap in repaired sequence
                orig_idx += 1
            elif alignment[1][i] != '-':
                # Gap in original sequence
                repair_idx += 1
        except IndexError:
            break
        i += 1
    
    return orig_to_new, alignment[0], alignment[1]

def prepare_single_chain(pdb_file_path, chain_id, output_location):
    """
    Extract a single chain from a PDB file and save it to the specified output location.
    
    Args:
        pdb_file_path (str): Path to the input PDB file
        chain_id (str): Chain identifier to extract (e.g., 'A', 'B', etc.)
        output_location (str): Path where the single-chain PDB should be saved
    
    Returns:
        str: Path to the output file if successful, None otherwise
    """
    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)
    
    # Get the base filename without extension
    base_name = os.path.basename(pdb_file_path).split('.')[0]
    
    # Parse the PDB file
    try:
        structure = parser.get_structure(base_name, pdb_file_path)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None
    
    # Create a new structure for the selected chain
    new_structure = PDB.Structure.Structure(f"{base_name}_{chain_id}")
    
    # Get the first model (PDB files can have multiple models)
    model = structure[0]
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)
    
    # Check if the requested chain exists
    chain_exists = False
    for chain in model:
        if chain.id == chain_id:
            new_model.add(chain.copy())
            chain_exists = True
            break
    
    if not chain_exists:
        print(f"Chain {chain_id} not found in {pdb_file_path}")
        return None
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_location)), exist_ok=True)
    
    # Create a PDBIO object
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    
    # Write the structure to the output file
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
    # to start preprocessing from the provided KORPM datasets, extra steps
    # are needed to convert to a CSV file
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
            # correct wrong index
            #db_.loc[db_['code']=='1IV7', 'position'] -= 100
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
    
    # original database needs to be at this location and can be obtained from
    # the FireProtDB website or from Pancotti et al.
    db = pd.read_csv(args.db_loc)
    print('Loaded', args.db_loc, 'len =', len(db))
    
    dataset = args.dataset
    dataset_outname = args.dataset
    sym = False

    if 'fireprot' == dataset:
        dataset = 'fireprot'
        # some entries in FireProt do not have associated structures
        db = db.dropna(subset=['pdb_id'])
        # get the first PDB from the list (others might be alternate structures)
        db['code'] = db['pdb_id'].apply(lambda x: x.split('|')[0])
        # correct for using the 1LVE structure sequence rather than UniProt
        db.loc[db['code']=='1HTI', 'position'] -= 37
        db.loc[db['code']=='1LVE', 'position'] -= 20
        db.loc[db['code']=='1ZNJ', 'chain'] = 'B'
        #db.loc[~(db['code']=='1ZNJ') & (db['wild_type']=='T'), 'chain'] = 'B'
        db.loc[(db['code']=='1ZNJ') & (db['wild_type']=='T'), 'chain'] = 'A'
    elif 's669' == dataset:
        dataset = 's669'
        db['code'] = db['Protein'].str[0:4]
        db['chain'] = db['Protein'].str[-1]
        db['wild_type'] = db['PDB_Mut'].str[0]
        db['position'] = db['PDB_Mut'].str[1:-1].astype(int)
        #db.loc[db['code']=='1IV7', 'position'] = db.loc[db['code']=='1IV7', \
        #     'PDB_Mut'].str[1:-1].astype(int)
        db['mutation'] = db['PDB_Mut'].str[-1]
        #db.loc[db['code']=='3K82', 'position'] -= 1
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
        #db.loc[db['code']=='1IV7', 'chain'] = 'A'
    elif 'ptmul_filtered' == dataset:
        db['code'] = db['pdb_id'].str[:-1]
        db['chain'] = db['chain_id']
        db['ddG'] = -db['ddg']
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
        #db = db.rename({'mut_seq': 'mut_seq_trunc', 'wt_seq': 'wt_seq_trunc'}, axis=1)
        #print(db.loc[db['code']=='1QJP', 'mut_info'])
    else:
        raise NotImplementedError      
    
    if sym:
        db['uid'] = db['structure'] + db['chain'] + '_' + db['wild_type'] + db['position'].astype(str) + db['mutation']
        grouper = ['code', 'structure', 'chain']
    elif dataset in ['ptmul_filtered', 'ptmul_orig']:
        db['uid'] = db['code'] + db['chain'] + '_' + db['mut_info']
        grouper = ['code', 'code', 'chain']
    else:
        db['uid'] = db['code'] + db['chain'] + '_' + db['wild_type'] + db['position'].astype(str) + db['mutation']
        grouper = ['code', 'code', 'chain']
    
    db = db.set_index('uid')
    os.makedirs('/home/sareeves/software/esm-msr/data/structures/', exist_ok=True)
    os.makedirs('/home/sareeves/software/esm-msr/data/structures/single_chain', exist_ok=True)

    for (code, struct, chain), group in db.groupby(grouper):
        pdb_file = f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_processed.pdb'
        db.loc[(db['code' if not sym else 'structure']==struct) & (db['chain']==chain), 'pdb_file'] = pdb_file
        if not os.path.exists(pdb_file):
            result = download_pdb(struct, f'/home/sareeves/software/esm-msr/data/structures/')
            os.rename(f'/home/sareeves/software/esm-msr/data/structures/{struct.lower()}.pdb', f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_original.pdb')
            remove_caps(f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_original.pdb')
            remove_heteroatoms(f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_original.pdb', f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb')
            fix_noncanonical_residues(f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb', f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb')
            renumber_pdb(f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb', f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb')
            repair_pdb(f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_intermediate.pdb', pdb_file, sequence_file=result['fasta'], chain_id=chain, verbose=True)
            add_lines(pdb_file)
    
    for (code, struct, chain), group in db.groupby(grouper):
        pdb_file = group['pdb_file'].head(1).item()
        wt_seq = get_seq(pdb_file, chain)
        original_pdb = f'/home/sareeves/software/esm-msr/data/structures/{struct}_{chain}_original.pdb'
        repaired_pdb = pdb_file
        single_chain_loc = pdb_file.split('/')
        single_chain_loc.insert(-1, 'single_chain')
        single_chain_loc = '/'.join(single_chain_loc)
        prepare_single_chain(pdb_file, chain, single_chain_loc)
        orig_to_new, orig, new = create_residue_mapping(original_pdb, repaired_pdb, chain)

        if not dataset in ['ptmul_filtered', 'ptmul_orig']:
            for i, row in group.iterrows():
                print(orig_to_new)
                db.at[i, 'wt_seq'] = wt_seq
                #print(code, wt, pos, mt)
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
                #assert wt_seq == row['wt_seq'], f'{row["code"]}\n{wt_seq}\n{row['wt_seq']}'
                mut_seq = list(wt_seq)
                db.at[i, 'wt_seq'] = wt_seq
                seq_pos_list = []
                mut_info_seq_pos = ''
                for mut in reorder_muts(row['mut_info'].split(':')):
                    if not mut_info_seq_pos == '':
                        mut_info_seq_pos += ':'
                    wt = mut[0]
                    pos = orig_to_new[mut[1:-1]]
                    #if code == '1ONC':
                    #    pos -= 1
                    seq_pos_list.append(pos)
                    mt = mut[-1]
                    mut_info_seq_pos += f'{wt}{pos}{mt}'
                    #print(mut_info_seq_pos)
                    #print('wt_seq', wt_seq, 'mut', f'{wt}{pos}{mt} {(mut[1:-1])}')
                    #print(wt, wt_seq[pos-1], mt)
                    assert wt_seq[pos-1] == wt
                    mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
                #db.at[i, 'seq_pos_list'] = seq_pos_list
                db.at[i, 'mut_info_seq_pos'] = mut_info_seq_pos
        #       assert mut_seq == row['mut_seq'], f'{row["code"]}\n{mut_seq}\n{row['mut_seq']}'           
        elif dataset == 'ptmul_filtered':
            for i, row in group.iterrows():
                db.at[i, 'orig_to_new'] = str(orig_to_new)
                #assert wt_seq == row['wt_seq'], f'{row["code"]}\n{wt_seq}\n{row['wt_seq']}'
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
                    #print(mut_info_seq_pos)
                    #print('wt_seq', wt_seq)
                    assert wt_seq[pos-1] == wt
                    mut_seq[pos-1] = mt
                mut_seq = ''.join(mut_seq)
                db.at[i, 'mut_seq'] = mut_seq
                #db.at[i, 'seq_pos_list'] = seq_pos_list
                db.at[i, 'mut_info_seq_pos'] = mut_info_seq_pos
        #       assert mut_seq == row['mut_seq'], f'{row["code"]}\n{mut_seq}\n{row['mut_seq']}'

    os.makedirs(args.output, exist_ok=True)

    if sym:
        db = db.rename({'code': 'wt_code'}, axis=1).rename({'structure': 'code'}, axis=1)
    if dataset in ['ptmul_filtered' , 'ptmul_orig', 'ptmul']:
        db = parse_mutation_column_to_separate_columns(db, 'mut_info_seq_pos')
    db.to_csv(os.path.join(args.output, f'{dataset_outname}_mapped_new.csv'))

    if dataset_outname == 's669':

        # create and use a third index for matching with the S461 subset
        db_full = db.copy(deep=True)
        db_full['uid2'] = db['code'] + '_' + db['PDB_Mut'].str[1:]
        db_full = db_full.reset_index().set_index('uid2')
        db_full = db_full.rename({'ddG': 'ddG_s669'}, axis=1)

        # preprocess S461 to align with S669
        s461 = pd.read_csv('/home/sareeves/PSLMs/data/external_datasets/S461.csv')
        s461['uid2'] = s461['PDB'] + '_' + s461['MUT_D'].str[2:]
        s461 = s461.set_index('uid2')
        s461['ddG_I'] = -s461['ddG_D']
        s461.columns = [s+'_dir' for s in s461.columns]
        s461 = s461.rename(
            {'ddG_D_dir': 'ddG', 'ddG_I_dir': 'ddG_inv'}, axis=1)
        

        db = s461.join(db_full, how='left').reset_index(drop=True)
        assert len(db) == 461
        db.set_index('uid').to_csv(os.path.join(args.output, 's461_mapped_new.csv'))

    if dataset_outname == 'k3822':

        k2369 = pd.read_csv('/home/sareeves/PSLMs/data/external_datasets/K2369.csv').set_index('uid')
        db = db.loc[k2369.index]
        assert len(db) == 2369
        db.to_csv(os.path.join(args.output, 'k2369_mapped_new.csv'))

    if dataset_outname == 'ptmul_filtered':

        ptmuld = pd.read_csv('/home/sareeves/PSLMs/data/external_datasets/PTMUL-D.csv')
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
        db = ptmuld.merge(db.drop(['pdb_id', 'wt_seq_trunc', 'orig_to_new', 'wt_seq'], axis=1), on='mut_seq').set_index('uid')

        assert len(db) == 536

        db.to_csv(os.path.join(args.output, 'ptmuld_mapped_new.csv'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output', type=str, default='../data/preprocessed/')
    parser.add_argument('--modlib_dir', type=str, default='/usr/lib/modeller10.4/modlib/')
    parser.add_argument('--modeller_dir', type=str, default='/usr/lib/modeller10.4/lib/x86_64-intel8/python3.3/')
    args = parser.parse_args()

    sys.path.append(args.modlib_dir)
    sys.path.append(args.modeller_dir)

    from modeller import *
    from modeller.automodel import *
            
    if args.dataset.lower() in ['q3421']:
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/Q3421.csv'
    elif args.dataset.lower() in ['fireprot', 'fireprotdb']:
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/fireprotdb_results.csv'
    elif args.dataset.lower() in ['s669', 's461']:
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/Data_s669_with_predictions.csv'
        args.dataset = 's669'
    elif args.dataset.lower() in ['ssym']:
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/ssym.csv'
    elif args.dataset.lower() in ['korpm', 'korpm_reduced', 'k2369', 'k3822']:
        args.dataset = 'k3822'
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/Id25c03_1merNCL.txt'
    elif args.dataset.lower() in ['ptmul_filtered', 'ptmuld']:
        args.dataset = 'ptmul_filtered'
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/protherm_multiple.csv'
    elif args.dataset.lower() in ['ptmul_orig', 'ptmul']:
        args.dataset = 'ptmul_orig'
        args.db_loc = '/home/sareeves/PSLMs/data/external_datasets/ptmul.csv'
    else:
        print('Inferred use of user-created database. Note: this must '
                'contain columns for code, wild_type, position, mutation. '
                'position must correspond to PDB sequence')
        assert args.dataset != 'fireprot'
        assert args.db_loc is not None

    main(args)
