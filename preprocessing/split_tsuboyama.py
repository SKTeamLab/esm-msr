import pandas as pd
from esm.utils.structure.protein_chain import ProteinChain
import os
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
from typing import Dict, List, Set
from collections import defaultdict
import argparse


def dataframe_to_fasta(df, name_col, seq_col, output_file):
    """
    Convert a DataFrame to a FASTA file.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame
    name_col (str): The name of the column containing sequence names
    seq_col (str): The name of the column containing sequences
    output_file (str): The name of the output FASTA file
    
    Returns:
    None
    """
    # Check if the specified columns exist in the DataFrame
    if name_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"Columns '{name_col}' or '{seq_col}' not found in the DataFrame")
    
    # Create a list to store SeqRecord objects
    records = []
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Create a SeqRecord object for each row
        record = SeqRecord(
            Seq(row[seq_col]),
            id=row[name_col],
            name=row[name_col],
            description=""
        )
        records.append(record)
    
    # Write the records to the FASTA file
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    
    print(f"FASTA file '{output_file}' has been created successfully.")

def get_identity(id1: str, id2: str, id_matrix: pd.DataFrame, debug: bool = False) -> float:
    row = id_matrix[(id_matrix['code1'] == id1) & (id_matrix['code2'] == id2)]
    if row.empty:
        row = id_matrix[(id_matrix['code1'] == id2) & (id_matrix['code2'] == id1)]
    output = row['identity'].iloc[0] if not row.empty else 0.0
    if debug:
        print(row)
    #print(id1, id2, output)
    return output

def generate_splits(candidate_datasets: pd.DataFrame, 
                    id_matrix: pd.DataFrame, 
                    identity_threshold: float = 0.25,
                    allow_redundancy: bool = True) -> pd.DataFrame:

    def check_test(code):
        overlap_df = id_matrix.loc[id_matrix['test_overlap']]
        overlap_df = overlap_df.loc[(overlap_df['code1']==code) | (overlap_df['code2']==code)]
        overlap_df = overlap_df.loc[overlap_df['identity'] >= identity_threshold]
        overlap = set(list(overlap_df['code1'])+list(overlap_df['code2']))
        if code in overlap:
            overlap.remove(code)
        
        if len(overlap_df) > 0:
            print('Rejected', code, ': overlaps', overlap, 'in external test')
            return False
        else:
            return True

    def check_identity(existing_id_set: Set[str], id_new: str, other_sets, kind='training') -> bool:
        # reject any candidate protein with any identity to any protein in existing_id_set or other_sets
        for key, other_set in other_sets.items():
            for existing_id in other_set:
                if get_identity(existing_id, id_new, id_matrix) >= identity_threshold:
                    print('Rejected', id_new, 'for', kind, 'because it was too similar to', existing_id, 'in', key)
                    return False
        for existing_id in existing_id_set:
            if get_identity(existing_id, id_new, id_matrix) >= identity_threshold:
                if allow_redundancy:
                    print(id_new, 'similar to', existing_id, 'already in', kind, 'data')
                    return True
                else:
                    print('Rejected', id_new, 'similar to', existing_id, 'already in', kind, 'data')
                    return False
        return True

    # should be only one row (stability) here
    counts = candidate_datasets.groupby('selection').count()
    splits = pd.DataFrame(index=counts.index, columns=['training', 'validation', 'testing'])

    for sel, row in counts.iterrows():
        print(f"Processing selection: {sel}, DMS_id count: {row['name']}")
        options = candidate_datasets.loc[candidate_datasets['selection'] == sel].copy()
        
        # Group entries by pdb_file
        pdb_file_groups = defaultdict(list)

        for _, row in options.iterrows():
            pdb_file_groups[row['name']].append((row['DMS_id'], row['pdb_seq_len'], row['reserve_for_test']))

        print(pdb_file_groups)
        
        # Initialize sets for each split
        training_set = set()
        validation_set = set()
        testing_set = set()

        # Process all pdb_files
        proteins = list(pdb_file_groups.keys())
        np.random.seed(24) # originally 42!
        np.random.shuffle(proteins)
        
        for prot in proteins:
            
            entries = pdb_file_groups[prot]
            for dms_id, seq_len, reserve_for_test in entries:
                
                if reserve_for_test:
                    print(dms_id, 'reserved for test')
                    testing_set.add(dms_id)
                    continue

                if len(validation_set) < 10:    
                    if check_identity(validation_set, prot, {'training_set': training_set, 'test_set': testing_set}, kind='validation') and len(validation_set) < 10:
                        if sel == 'stability':
                            if check_test(prot):
                                validation_set.add(dms_id)
                                continue
                        else:
                            validation_set.add(dms_id)
                            continue

                # only add to training set if it has less than x entries
                if len(training_set) < 1000:
                    # verify that the entry prot has no overlap with existing datasets
                    if check_identity(training_set, prot, {'validation_set': validation_set, 'test_set': testing_set}, kind='training'):
                        if sel == 'stability':
                            if check_test(prot):
                                training_set.add(dms_id)
                                continue
                        else:
                            training_set.add(dms_id)
                            continue

                # it is okay if there are redundancies in the test set
                if check_identity(set(), prot, {'training_set': training_set, 'validation_set': validation_set}, kind='testing'):
                    testing_set.add(dms_id)
                    continue
                else:
                    print(f'Rejected {dms_id} from all sets')

        # Check if we have enough sequences in each set
        if len(training_set) < 1 or len(validation_set) < 1 or len(testing_set) < 1:
            raise ValueError(f"Not enough sequences with <{identity_threshold:.0%} identity for selection {sel}. "
                             f"Training: {len(training_set)}, Validation: {len(validation_set)}, "
                             f"Testing: {len(testing_set)}")

        # Assign to splits DataFrame
        splits.loc[sel, 'training'] = list(training_set)
        splits.loc[sel, 'validation'] = list(validation_set)
        splits.loc[sel, 'testing'] = list(testing_set)
        print(f"Split sizes - Training: {len(training_set)}, Validation: {len(validation_set)}, Testing: {len(testing_set)}")

    return splits

def verify_splits(splits: pd.DataFrame, 
                  id_matrix: pd.DataFrame, 
                  identity_threshold: float = 0.25) -> Dict[str, List[tuple]]:

    violations = {}

    for selection in splits.index:
        selection_violations = []
        
        # Get sequences for each set
        training_seqs = splits.loc[selection, 'training']
        validation_seqs = splits.loc[selection, 'validation']
        testing_seqs = splits.loc[selection, 'testing']
        
        # Check training vs validation
        for train_seq in training_seqs:
            for val_seq in validation_seqs:
                similarity = get_identity(train_seq, val_seq, id_matrix)
                if similarity > identity_threshold:
                    selection_violations.append(('train-val', train_seq, val_seq, similarity))
        
        # Check training vs testing
        for train_seq in training_seqs:
            for test_seq in testing_seqs:
                similarity = get_identity(train_seq, test_seq, id_matrix)
                if similarity > identity_threshold:
                    selection_violations.append(('train-test', train_seq, test_seq, similarity))
        
        # Check validation vs testing
        for val_seq in validation_seqs:
            for test_seq in testing_seqs:
                similarity = get_identity(val_seq, test_seq, id_matrix)
                if similarity > identity_threshold:
                    selection_violations.append(('val-test', val_seq, test_seq, similarity))
        
        if selection_violations:
            violations[selection] = selection_violations

    return violations

def main(args):
    if os.path.exists("/home/sareeves/PSLMs/data/tsuboyama/homology"):
        import shutil
        shutil.rmtree("/home/sareeves/PSLMs/data/tsuboyama/homology")
    os.makedirs("/home/sareeves/PSLMs/data/tsuboyama/homology")
    # load the dataframe of all substitution mutations in Tsuboyama
    df = pd.read_csv(args.database, index_col=0)
    # get the first instance for each unique structure (use the first mutant sequence)
    ref = df.groupby('code').first()
    # extract a name based on the structure file. Structure files with suffixes (mutant backbones) will be combined into 1 since the suffix is after .pdb
    if 'name' not in ref.columns:
        ref['name'] = ref['WT_name'].apply(lambda x: x.split('.pdb')[0])
    else:
        ref['name'] = ref['name'].apply(lambda x: x.split('.pdb')[0])
    # create a fasta file in order to determine sequence identities
    dataframe_to_fasta(ref.reset_index(), 'name', 'aa_seq', '../data/tsuboyama/homology/tsuboyama_seqs.fasta')


    if args.train_doubles:
        print('Multimutant train set!')
        # any protein that has any single mutants is in the test set; not used for training
        test1 = df.loc[~df['mut_type'].str.contains(':'), ['code', 'chain', 'pdb_file']]
        test = pd.concat([test1]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()    
    elif args.test_doubles:
        print('Multimutant test set!')
        # any protein that has any double mutants is in the test set; not used for training
        test1 = df.loc[df['mut_type'].str.contains(':'), ['code', 'chain', 'pdb_file']]
        test2 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/ptmuld_mapped.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        test = pd.concat([test1, test2]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    elif args.train_synthetic:
        print('Synthetic train set!')
        test1 = df.loc[df['code'].str.len()==4, ['code', 'chain', 'pdb_file']]
        test2 = df.loc[df['code'].str.startswith('v2_'), ['code', 'chain', 'pdb_file']]
        test = pd.concat([test1, test2]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    elif args.test_synthetic:
        print('Synthetic test set!')
        test1 = df.loc[~(df['code'].str.len()==4) & ~(df['code'].str.startswith('v2_')), ['code', 'chain', 'pdb_file']]
        #test2 = df.loc[df['code'].str.startswith('v2_'), ['code', 'chain', 'pdb_file']]
        test = pd.concat([test1]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()
    else:
        # repeat the process with all testing data
        test1 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/k3822_mapped_new.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        test2 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/q3421_mapped_new.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        test3 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/s669_mapped_new.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        test4 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/ssym_mapped_new.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        test5 = pd.read_csv('/home/sareeves/PSLMs/data/preprocessed/ptmul_mapped_new.csv', index_col=0)[['code', 'chain', 'pdb_file']]
        # concatenate the data together into a single dataframe of testing sequences
        test = pd.concat([test1, test2, test3, test4, test5]).drop_duplicates().groupby(['code', 'chain']).first().reset_index()

    # get the correct PDB sequence directly from the structure files
    #test['pdb_file'] = test['pdb_file'].apply(lambda x: '/home/sareeves/PSLMs/structures/' + x.split('structures/')[-1])
    test['pdb_seq'] = test.apply(lambda x: ProteinChain.from_pdb(os.path.join('/home/sareeves/PSLMs/data/proteingym/structures', x['pdb_file']), x['chain']).sequence, axis=1)
    test['name'] = 'test_' + test['code'] + '_' + test['chain']
    # create a fasta file in order to determine sequence identities
    dataframe_to_fasta(test, 'name', 'pdb_seq', '../data/tsuboyama/homology/test_seqs.fasta')

    # Read and combine sequences into a single file for comparison using MMSeqs
    sequences = []
    for fname in ['../data/tsuboyama/homology/tsuboyama_seqs.fasta', '../data/tsuboyama/homology/test_seqs.fasta']:
        sequences.extend(list(SeqIO.parse(fname, "fasta")))

    # Write combined sequences
    SeqIO.write(sequences, "../data/tsuboyama/homology/combined.fasta", "fasta")

    commands = [
        "cd /home/sareeves/PSLMs/data/tsuboyama/homology && " +
        "mmseqs createdb tsuboyama_seqs.fasta queryDB && " +
        "mmseqs createdb combined.fasta targetDB && " +
        "mmseqs search queryDB targetDB resultDB tmp -s 9.5 -e 10 --realign --alignment-mode 3 --seq-id-mode 1 -a --min-ungapped-score 0 && " +
        "mmseqs convertalis queryDB targetDB resultDB alignment.txt"
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
            print(f"Command: {cmd}")
            print(f"Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error while executing: {cmd}")
            print(e.stderr)

    matrix = pd.read_csv('../data/tsuboyama/homology/alignment.txt', sep='\t', header=None)
    matrix = matrix.iloc[:, [0,1,2,3,4,5]]
    matrix.columns = ['code1', 'code2', 'identity', 'alignment_length', 'mismatches', 'gap_openings']
    # we don't care about identity between identical proteins
    matrix = matrix.loc[~(matrix['code1']==matrix['code2'])].drop_duplicates()
    # we don't care about where test sets intersect with each other
    matrix = matrix.loc[~((matrix['code1'].str.contains('test')) & (matrix['code2'].str.contains('test')))]
    # we want to get entries where only one of the sequences is in the test set
    matrix['test_overlap'] = matrix['code1'].str.contains('test') | matrix['code2'].str.contains('test')
    # we aren't interested in overlaps of less than 25%
    matrix = matrix.loc[matrix['identity'] >= 0.25]

    # configure ref (reference set of all proteins and their data) to resemble proteingym data
    candidate_datasets = ref
    candidate_datasets['selection'] = 'stability'
    candidate_datasets['pdb_seq_len'] = candidate_datasets['aa_seq'].apply(lambda x: len(x))
    # don't reserve anything for test - test data is based on the prefix
    candidate_datasets['reserve_for_test'] = False
    candidate_datasets['DMS_id'] = candidate_datasets['name']

    splits = generate_splits(candidate_datasets, matrix, allow_redundancy=args.redundancy)
    print("Violations", verify_splits(splits, matrix))
    splits.to_csv(f'../data/{args.output}')

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--output', type=str)
        parser.add_argument('--database', default='/home/sareeves/software/esm-msr/data/preprocessed/tsuboyama_all_subs_final.csv')
        parser.add_argument('--redundancy', action='store_true')
        parser.add_argument('--train_doubles', action='store_true')
        parser.add_argument('--test_doubles', action='store_true')
        parser.add_argument('--train_synthetic', action='store_true')
        parser.add_argument('--test_synthetic', action='store_true')
        #parser.add_argument('--property', type=str, choices=['activity', 'expression', 'organismalfitness', 'binding', 'stability'])
        #parser.add_argument('--learning_rate', type=float, default=1e-4)
        #parser.add_argument('--n_subsets', type=int, default=32)

        args = parser.parse_args()
        if args.redundancy:
            print('Allowing redundancy!')
        else:
            print('Not allowing redundancy!')
        
        main(args)