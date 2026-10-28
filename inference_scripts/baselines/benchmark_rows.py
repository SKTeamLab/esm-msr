"""The benchmark rows every baseline scores, built once so every model sees exactly the same ones.

Each function returns a frame with id, code, chain, pdb_file, mut_type, ddG, where mut_type is the mutation string in
structure-sequence numbering (`A12G` or `A12G:L40V`, sorted by position) and ddG is the target (positive = stabilizing).
These are the rows esm_msr_testing.py scores for the same set, keyed the same way:

  external_rows(name)   one of the 12 external sets (EXTERNAL)
  dms_rows(assay)       one of the 7 DMS assays (DMS), from the complete tables with the completeness guard (dms_source)
  domainome_rows()      the Domainome assay (single mutants on 522 domains)

ProteinGym comes from pgym_inputs through pgym_runner; the Tsuboyama splits are model-specific (ThermoMPNN and
MutateEverything read prepared item tables, SPURS and ProteinMPNN the ESM-MSR preprocessor).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dms_source  # noqa: E402
import paths  # noqa: E402

EXTERNAL = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
SEQPOS_SETS = {'s669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 's571', 's783', 's2648', 's8754'}


def external_rows(name):
    """The rows and mutation strings (structure-sequence numbering) esm_msr_testing.py scores for one external set."""
    df = pd.read_csv(paths.PREPROCESSED / f'{name}_mapped.csv')
    df['pdb_file'] = paths.fix_pdb_column(df['pdb_file'])
    if name in SEQPOS_SETS:
        df['mut_type'] = df['wild_type'] + df['seq_pos'].astype(int).astype(str) + df['mutation']
        if 'dTm' in df.columns:
            df['ddG'] = df['dTm']
    else:
        df['mut_type'] = df['mut_info_seq_pos'].map(lambda s: ':'.join(sorted(s.split(':'), key=lambda m: int(m[1:-1]))))
    df['id'] = df['code'] + df['chain'] + '_' + df['mut_type']
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']]


DMS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
       'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
       'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']


def dms_rows(prot):
    """The rows esm_msr_testing.py's DMS block scores for one assay, keyed and scored the same way (id = code_mut_info,
    deduplicated, target ddG_ML). Audited: for all seven assays every mutation's position and wild-type residue match
    the parsed chain exactly, and the chain sequence equals wt_seq wherever the table carries one, so the positions are
    already in structure numbering and need no renumbering."""
    # dms_source.load: the complete tables, with the completeness guard (the old software/esm-msr copy of GB1 was
    # missing V39A/V39C/V39D, which silently cost 152 of its 2,091 scorable doubles).
    df = dms_source.load(prot)
    df['mut_type'] = df['mut_info'].map(lambda s: ':'.join(sorted(str(s).split(':'), key=lambda m: int(m[1:-1]))))
    df['id'] = df['code'] + '_' + df['mut_info']
    df = df[~df['id'].duplicated(keep='first')]
    df['ddG'] = df['ddG_ML']
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']].reset_index(drop=True)


def domainome_rows():
    """ESM-MSR's domainome rows (domainome_mapped_2026.csv, code = domain_ID with '/'->'_', target scaled_fitness),
    keyed and deduplicated as esm_msr_testing.py's domainome block does. All 536k rows are single mutants; audited:
    every row's position and wild-type residue match its spurs_structures chain A."""
    df = pd.read_csv(paths.require(paths.DOMAINOME_TABLE, 'the Domainome table'),
                     usecols=['domain_ID', 'mut_type', 'scaled_fitness', 'pdb_file', 'position', 'wild_type',
                              'mutation'], low_memory=False)
    df['code'] = df['domain_ID'].str.replace('/', '_', regex=False)
    df = df.dropna(subset=['pdb_file', 'position'])
    df['pdb_file'] = paths.fix_pdb_column(df['pdb_file'])
    df['chain'] = 'A'
    df['ddG'] = df['scaled_fitness']
    df['id'] = df['code'] + '_' + df['mut_type']
    df = df[~df['id'].duplicated(keep='first')]
    return df[['id', 'code', 'chain', 'pdb_file', 'mut_type', 'ddG']].reset_index(drop=True)


def library_sequence(data, mut_structure, mut_col='mut_type'):
    """(sequence, pdb_file) a Tsuboyama library group is measured against: the AlphaFold chain whose residue at the
    first measured single is the *mutant* (the table's aa_seq), reverted to wild type, and, for a mutant-background
    library (mut_structure like 'L42A'), carrying the background substitution. `data` holds the rows of one
    (library, mut_structure) group; mut_type is relative to the returned sequence."""
    singles = data.loc[~data[mut_col].str.contains(':')]
    seq = list(singles['aa_seq'].head(1).item())
    first = singles[mut_col].head(1).item()
    wt, pos, mut = first[0], int(first[1:-1]), first[-1]
    assert seq[pos - 1] == mut, f'Expected position {pos} to be {mut}, but got {seq[pos - 1]}'
    seq[pos - 1] = wt
    if mut_structure != '-':
        wt, pos, mut = mut_structure[0], int(mut_structure[1:-1]), mut_structure[-1]
        assert seq[pos - 1] == mut, f'Expected position {pos} to be {mut}, but got {seq[pos - 1]}'
        seq[pos - 1] = mut
    return ''.join(seq), singles['pdb_file'].head(1).item()

