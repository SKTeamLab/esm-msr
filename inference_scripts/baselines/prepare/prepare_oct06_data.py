"""
Build ESM-MSR-matched training / validation / test tables for the MutateEverything and ThermoMPNN-D baselines.

Row-level filtering is ESM-MSR's own (MegaScaleDatasetPreprocessor + create_training_splits, out-of-range rows
excluded). Item-level rules follow MutationStabilityDataset as run by scripts/run_arm.sh (subsets single, cond,
native_cond; --min_additive_dG -1.0 --subfloor_rank_only; no censor floor):

  * single       measured single on a WT-backbone library                       -> kept
  * native_cond  measured single on a mutant-background library ('2K52_L42A')   -> kept, WT sequence = background
  * cond         derived from a double whose two singles are measured (uncensored); ESM-MSR regresses on it unless
                 dG_wt + ddG_A + ddG_B <= -1 (reg_ok=False, rank-only). The baselines have no rank loss and take
                 doubles natively, so the double itself is kept iff ESM-MSR would regress on its cond items.
  * censored ('<-1' / '>5') rows, doubles without both singles, sub-floor doubles, reversions   -> dropped

The test table is what inference_scripts/esm_msr_testing.py scores (all uncensored singles and doubles of the test
split, every backbone group), so baseline and ESM-MSR numbers come from identical rows.

Structures: ESM-MSR looks for a modeled background structure under <mut_structures_root>/<code incl. background>/,
which never exists, so it always uses the WT AlphaFold backbone with the background-mutated sequence (unmasked).
For ThermoMPNN-D we write that same backbone with the background residue renamed, so its WT sequence matches.

--mut_backbones exclude drops every row of a mutant-background library from train/val (see --mut_backbone_splits),
for the ablation that asks what those technically-incorrect backbones contribute. test is untouched by default so
both conditions are scored on identical rows.
"""
import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths
import paths  # noqa: E402

REPO = str(paths.REPO_ROOT)
sys.path.insert(0, os.path.join(REPO, 'src'))
from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor  # noqa: E402

BG_RE = re.compile(r'_([A-Z])([0-9]+)([A-Z])$')
THREE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
         'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
         'TYR': 'Y', 'VAL': 'V'}
ONE = {v: k for k, v in THREE.items()}
MIN_ADDITIVE_DG = -1.0


def pdb_sequence(path, chain='A'):
    """Sequence of `chain`, one residue per (resnum, icode) with a CA atom - what ProteinChain.from_pdb reads."""
    seq, seen = [], set()
    with open(path) as f:
        for line in f:
            if not line.startswith('ATOM') or line[21] != chain or line[12:16].strip() != 'CA':
                continue
            key = line[22:27]
            if key in seen:
                continue
            seen.add(key)
            seq.append(THREE.get(line[17:20], 'X'))
    return ''.join(seq)


def write_renamed_pdb(src, dst, pos, wt_aa, mt_aa, chain='A'):
    """Copy `src` with residue `pos` (1-based, sequential over CA residues) renamed wt_aa -> mt_aa.
    Side-chain atoms beyond CB are dropped at that residue (and CB too for Gly) so no atom contradicts the new name;
    ThermoMPNN-D reads backbone atoms only, so this changes nothing but the residue identity it sees."""
    out, idx, last = [], 0, None
    keep_sc = {'N', 'CA', 'C', 'O'} | (set() if mt_aa == 'G' else {'CB'})
    with open(src) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and line[21] == chain:
                key = line[22:27]
                if key != last:
                    idx += 1
                    last = key
                if idx == pos:
                    assert THREE.get(line[17:20]) == wt_aa, f'{src}: residue {pos} is {line[17:20]}, expected {wt_aa}'
                    if line[12:16].strip() not in keep_sc:
                        continue
                    line = line[:17] + ONE[mt_aa] + line[20:]
            out.append(line)
    with open(dst, 'w') as f:
        f.writelines(out)


def parse_muts(mut_type, ref_seq):
    """ESM-MSR's MutationStabilityDataset._parse_mutations: entries whose WT residue disagrees with ref_seq are dropped."""
    return [(m[0], int(m[1:-1]), m[-1]) for m in mut_type.split(':')
            if len(m) >= 3 and ref_seq[int(m[1:-1]) - 1] == m[0]]


def build_items(df, dG_wt, af_dir, split_name, stats):
    """Expand one split's rows into ESM-MSR-matched items (train rules)."""
    rows = []
    for code, g in df.groupby('code', sort=True):
        code_wt = g['code_wt'].iloc[0]
        wt_seq_struct = pdb_sequence(os.path.join(af_dir, f'{code_wt}.pdb'))
        m = BG_RE.search(code) if code != code_wt else None
        bb = None
        ref_seq = wt_seq_struct
        if m:
            bb = m.group(0)[1:]
            w, p, t = m.group(1), int(m.group(2)), m.group(3)
            assert wt_seq_struct[p - 1] == w, f'{code}: background {bb} does not match structure ({wt_seq_struct[p-1]})'
            ref_seq = wt_seq_struct[:p - 1] + t + wt_seq_struct[p:]
        parsed = []
        singles = {}
        for _, r in g.iterrows():
            muts = parse_muts(r['mut_type'], ref_seq)
            n_listed = r['mut_type'].count(':') + 1
            cens = int(r['cens']) if np.isfinite(r['cens']) else 0
            if len(muts) != n_listed:
                stats[f'{split_name}/mismatched_entry_rows'] += 1
                continue  # ESM-MSR would turn some of these into mislabeled singles; drop instead (counted)
            parsed.append((r, muts, cens))
            if len(muts) == 1 and cens == 0:
                singles[(muts[0][1], muts[0][2])] = float(r['ddG_ML'])
        dgw = dG_wt.get(code)
        dgw = float(dgw) if dgw is not None and np.isfinite(dgw) else None
        for r, muts, cens in parsed:
            base = dict(code=code, code_wt=code_wt, bb_mut=bb, wt_seq=ref_seq, mut_type=r['mut_type'],
                        ddG_ML=float(r['ddG_ML']), dG_wt=dgw if dgw is not None else np.nan)
            if cens != 0:
                stats[f'{split_name}/drop_censored'] += 1
                continue
            if len(muts) == 1:
                kind = 'native_cond' if bb else 'single'
                rows.append({**base, 'kind': kind, 'ddG_A': np.nan, 'ddG_B': np.nan})
                stats[f'{split_name}/{kind}'] += 1
            elif len(muts) == 2:
                a = singles.get((muts[0][1], muts[0][2]), np.nan)
                b = singles.get((muts[1][1], muts[1][2]), np.nan)
                if not (np.isfinite(a) and np.isfinite(b)):
                    stats[f'{split_name}/drop_double_no_singles'] += 1
                    continue
                if dgw is not None and dgw + a + b <= MIN_ADDITIVE_DG:
                    stats[f'{split_name}/drop_double_subfloor'] += 1
                    continue
                rows.append({**base, 'kind': 'double', 'ddG_A': a, 'ddG_B': b})
                stats[f'{split_name}/double'] += 1
            else:
                stats[f'{split_name}/drop_higher_order'] += 1
    return pd.DataFrame(rows)


def build_test(df, af_dir):
    """All uncensored singles/doubles, as esm_msr_testing.py scores them (no item-level filtering)."""
    out = []
    for code, g in df.groupby('code', sort=True):
        code_wt = g['code_wt'].iloc[0]
        wt_seq_struct = pdb_sequence(os.path.join(af_dir, f'{code_wt}.pdb'))
        m = BG_RE.search(code) if code != code_wt else None
        bb, ref_seq = None, wt_seq_struct
        if m:
            bb = m.group(0)[1:]
            p, t = int(m.group(2)), m.group(3)
            ref_seq = wt_seq_struct[:p - 1] + t + wt_seq_struct[p:]
        for _, r in g.iterrows():
            if np.isfinite(r['cens']) and int(r['cens']) != 0:
                continue
            n = r['mut_type'].count(':') + 1
            if n > 2 or len(parse_muts(r['mut_type'], ref_seq)) != n:
                continue
            out.append(dict(code=code, code_wt=code_wt, bb_mut=bb, wt_seq=ref_seq, mut_type=r['mut_type'],
                            ddG_ML=float(r['ddG_ML']), kind='double' if n == 2 else ('native_cond' if bb else 'single')))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=os.path.join(REPO, 'data/tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'))
    ap.add_argument('--af_dir', default=os.path.join(REPO, 'data/tsuboyama/AlphaFold_model_PDBs'))
    ap.add_argument('--split', default=os.path.join(REPO, 'data/splits_oct06_structure.pkl'))
    ap.add_argument('--out', default=str(paths.BASELINE_DATA))
    ap.add_argument('--mut_backbones', choices=['include', 'exclude'], default='include',
                    help="Mutant-background libraries ('2K52_L42A': the WT AlphaFold backbone carrying a "
                         "background-mutated sequence, so the structure is a wild type the measurements were never "
                         "made against). 'exclude' drops every row of such a library from the splits named by "
                         "--mut_backbone_splits. Default 'include' reproduces the oct06/capped tables already on disk.")
    ap.add_argument('--mut_backbone_splits', default='train,val',
                    help='Which splits --mut_backbones exclude applies to. test is left out by default so a model '
                         'trained without these examples is still scored on exactly the rows ESM-MSR scores.')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(args.out, exist_ok=True)

    pre = MegaScaleDatasetPreprocessor(args.raw, args.af_dir, censor_range=False)
    pre.create_training_splits(args.split, -1)
    stats = Counter()

    tables = {}
    for split in ['train', 'val']:
        tables[split] = build_items(pre.split_dfs[split], pre.dG_wt, args.af_dir, split, stats)
    tables['test'] = build_test(pre.split_dfs['test'], args.af_dir)

    # A mutant-background library's rows are measured against a sequence its backbone does not have. 'exclude' drops
    # them wholesale; the libraries themselves stay in libraries.csv (and so keep their PDB/MSA) when any other split
    # still uses them, which keeps the on-disk layout identical between the two conditions.
    if args.mut_backbones == 'exclude':
        for split in [s for s in args.mut_backbone_splits.split(',') if s]:
            t = tables[split]
            keep = t['bb_mut'].isna()
            stats[f'{split}/drop_mut_backbone_rows'] = int((~keep).sum())
            stats[f'{split}/drop_mut_backbone_libraries'] = int(t.loc[~keep, 'code'].nunique())
            tables[split] = t.loc[keep].reset_index(drop=True)
    print(f"mut_backbones={args.mut_backbones} splits={args.mut_backbone_splits}")

    for split, t in tables.items():
        t.to_csv(os.path.join(args.out, f'items_{split}.csv'), index=False)
        print(split, len(t), dict(Counter(t['kind'])), 'libraries:', t['code'].nunique(), 'wt:', t['code_wt'].nunique())

    # One WT-sequence "library" per code: its sequence, and for ThermoMPNN-D a PDB whose sequence matches it.
    pdb_dir = os.path.join(args.out, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)
    libs = pd.concat(tables.values())[['code', 'code_wt', 'bb_mut', 'wt_seq']].drop_duplicates('code')
    for _, r in libs.iterrows():
        dst = os.path.join(pdb_dir, f"{r['code']}.pdb")
        src = os.path.join(args.af_dir, f"{r['code_wt']}.pdb")
        if isinstance(r['bb_mut'], str):
            bb = r['bb_mut']
            write_renamed_pdb(src, dst, int(bb[1:-1]), bb[0], bb[-1])
        elif not os.path.lexists(dst):
            os.symlink(src, dst)
        assert pdb_sequence(dst) == r['wt_seq'], f"{r['code']}: PDB sequence does not match the library sequence"
    libs.to_csv(os.path.join(args.out, 'libraries.csv'), index=False)
    with open(os.path.join(args.out, 'prep_stats.json'), 'w') as f:
        json.dump(dict(stats), f, indent=1, sort_keys=True)
    print(json.dumps(dict(stats), indent=1, sort_keys=True))
    print('libraries:', len(libs), 'with background:', libs['bb_mut'].notna().sum())


if __name__ == '__main__':
    main()
