import re
import os
import argparse
import pandas as pd

from inference_scripts.utils import is_fake_mutation, is_improper_mutation

def main(args):
    df = pd.read_csv(args.tsuboyama_dataset_loc)

    # extract entries where there is a wild-type crystal structure
    df2 = df[['aa_seq', 'aa_seq_full', 'mut_type', 'WT_name', 'ddG_ML']]

    #df2.loc[:, 'dG_ML'] = pd.to_numeric(df2['dG_ML'], errors='coerce')
    df2.loc[:, 'ddG_ML'] = pd.to_numeric(df2['ddG_ML'], errors='coerce')

    df3 = df2.loc[df2['ddG_ML'].notna()]
    df3 = df3.loc[~df3['mut_type'].str.contains('wt')]
    df3 = df3.loc[~df3['mut_type'].str.contains('ins')]
    df3 = df3.loc[~df3['mut_type'].str.contains('del')]

    df3['code'] = df3['WT_name']#.str.replace('|', '+')
    df3['pdb_file'] = '/home/sareeves/PSLMs/data/tsuboyama/AlphaFold_model_PDBs/' + df3['code'].apply(lambda x: x.split('.pdb')[0]).str.replace('|', '_') + '.pdb'
    df3['mut_structure'] = df3['code'].apply(lambda x: x.split('.pdb_')[1] if '.pdb_' in x else None)
    df3['code'] = df3['code'].apply(lambda x: x.split('.pdb')[0])
    df3.loc[~df3['mut_structure'].isna(), 'code'] += '_' + df3['mut_structure']

    df3['uid'] = df3['code'] + '_' + df3['mut_type']
    df3['chain'] = 'A' 
    #df3['code'].map(lambda x: {'1YU5': 'X', '2I5L': 'X', '2L7F': 'P', '2L7M': 'P', '2LVN': 'C', '1A0N': 'B', '1BSA': 'C', '1BSB': 'C', '1NFI': 'F'}.get(x, 'A'))
    df3.set_index('uid')

    def remove_duplicates_with_mean(df, groupby_cols, mean_col):
        """
        Remove duplicates by taking the mean of a specific column while keeping other shared values.
        
        Parameters:
        df (pandas.DataFrame): Input DataFrame
        groupby_cols (list): Columns to group by (all columns except the one to average)
        mean_col (str): Column name where mean should be taken for duplicates
        
        Returns:
        pandas.DataFrame: DataFrame with duplicates removed and means calculated
        """
        # Group by all columns except the one we want to average
        result = df.groupby(groupby_cols, as_index=False, dropna=False)[mean_col].mean()
        
        return result, df[df.duplicated(subset=groupby_cols, keep=False)]

    def convert_uid_to_mutation(uid: str) -> str:
        """
        Standardizes various UID formats into a unified mutation string.
        Supports:
        - Deletions: 1A32_delA45 -> A45-
        - Insertions: v2_6IVS_insG5 -> -5G
        - Single Subs: 1A23_C12A -> C12A
        - Double Subs: 1BF4_D14E:T41F -> D14E:T41F
        """
        if not isinstance(uid, str):
            return uid

        # 1. Isolate the mutation part.
        # DMS UIDs often look like [ID]_[Mutation]. We take parts that look like mutations.
        parts = uid.split('_')
        
        # Heuristic: The mutation info is usually in the last part, 
        # but we handle cases like 'v2_6IVS_insG5' by checking segments.
        mut_candidates = []
        for part in parts:
            # Check if part contains a mutation-like structure (del, ins, or A123B)
            if any(x in part for x in ['del', 'ins', ':']) or re.search(r'[A-Z]\d+[A-Z]', part):
                mut_candidates.append(part)
        
        if not mut_candidates:
            return uid  # Fallback if no mutation pattern is identified

        # Use the last identified candidate (the most likely mutation string)
        mut_string = mut_candidates[-1]

        # 2. Process each mutation event (handling multiples separated by ':')
        processed_events = []
        events = mut_string.split(':')
        
        for event in events:
            # Handle Deletions: delA45 -> A45-
            del_match = re.search(r'del([A-Z])(\d+)', event)
            if del_match:
                res, pos = del_match.groups()
                processed_events.append(f"{res}{pos}-")
                continue

            # Handle Insertions: insG5 -> -5G
            ins_match = re.search(r'ins([A-Z]+)(\d+)', event)
            if ins_match:
                res_ins, pos = ins_match.groups()
                processed_events.append(f"-{pos}{res_ins}")
                continue

            # Handle Substitutions: C12A -> C12A (kept as is)
            sub_match = re.search(r'([A-Z])(\d+)([A-Z])', event)
            if sub_match:
                processed_events.append(event)
                continue
            
            # Fallback for unidentified fragments
            processed_events.append(event)

        # 3. Join processed events back with colons
        return ":".join(processed_events)

    print(f'Len of all substitutions with duplicate mutants: {len(df3)}')

    groupby_cols = list(df3.columns.drop(['aa_seq_full', 'ddG_ML']))
    df4, dup = remove_duplicates_with_mean(df3, groupby_cols, 'ddG_ML')
    df4 = df4.sort_values('uid')
    df4['mut_type'] = df4['uid'].apply(convert_uid_to_mutation)

    print(f'Len of all substitutions (deduplicated): {len(df4)}')

    df4 = df4.loc[~df4['mut_structure'].fillna('').str.startswith('pross')]

    print(f'Len of all substitutions after removing PROSS designs: {len(df4)}')

    print('Examples of invalid mutations\n', df4.loc[df4['mut_type'].apply(is_fake_mutation)])
    df4 = df4.loc[~df4['mut_type'].apply(is_fake_mutation)]

    #print(df4.loc[df4['mut_type'].apply(is_improper_mutation)])
    df4 = df4.loc[~df4['mut_type'].apply(is_improper_mutation)]

    print(f'Len of all substitutions after removing invalid substitutions: {len(df4)}')

    df4['uid'] = df4['uid'].str.replace('|', '_')
    df4['code'] = df4['code'].str.replace('|', '_')
    df4.set_index('uid').to_csv(os.path.join(args.output, 'tsuboyama_all_subs_final.csv'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--tsuboyama_dataset_loc', type=str, default='~/PSLMs/data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
    parser.add_argument('--output', type=str, default='../data/preprocessed/')
    args = parser.parse_args()
            
    main(args)