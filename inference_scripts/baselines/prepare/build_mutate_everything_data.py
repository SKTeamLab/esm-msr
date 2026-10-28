"""
Write a prepared baseline variant (prepare_baseline_data.py: one split, its flags) in upstream MutateEverything's on-disk format
(data/<name>/{mutations,fasta,msa}); <name> is the variant's name (variant.json) unless --name says otherwise.

pdb_id is the ESM-MSR library code, so a mutant-background library ('2K52_L42A') is its own protein whose wild type is
the background sequence, with its own fasta and its own ColabFold MSA (query = background sequence). Upstream data.py
then needs no change: it groups by pdb_id and requires exactly one wt_seq per pdb_id.

ddg follows the upstream convention (positive = destabilizing), i.e. ddg = -ddG_ML.

  --mode fasta   write the query fasta for library sequences with no exact-match MSA yet (for colabfold_search)
  --mode build   write mutations/*.csv, fasta/, msa/ (every MSA must exist)
"""
import argparse
import json
import os
import shutil

import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # paths
import paths  # noqa: E402


def sort_muts(mut_type):
    ms = sorted(mut_type.split(':'), key=lambda m: int(m[1:-1]))
    return ':'.join(ms), ms


def to_me(items):
    rows = []
    for _, r in items.iterrows():
        mut_info, ms = sort_muts(r['mut_type'])
        seq = list(r['wt_seq'])
        for m in ms:
            assert seq[int(m[1:-1]) - 1] == m[0], (r['code'], m)
            seq[int(m[1:-1]) - 1] = m[-1]
        row = dict(pdb_id=r['code'], code_wt=r['code_wt'], bb_mut=r['bb_mut'], kind=r['kind'], mut_info=mut_info,
                   ddg=-float(r['ddG_ML']), ddG_ML=float(r['ddG_ML']), wt_seq=r['wt_seq'], mut_seq=''.join(seq))
        for i, m in enumerate(ms, 1):
            row.update({f'fr{i}': m[0], f'pos{i}': int(m[1:-1]), f'to{i}': m[-1]})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fasta', 'build'], required=True)
    ap.add_argument('--data_dir', default=None, help='the prepare_baseline_data.py output (default: $MSR_BASELINE_DATA)')
    ap.add_argument('--name', default=None, help='data/<name> in the MutateEverything checkout (default: the variant name in variant.json)')
    ap.add_argument('--me_root', default=str(paths.MUTATE_EVERYTHING_DIR))
    ap.add_argument('--new_msas', default=os.environ.get('MSR_ME_NEW_MSAS'),
                    help='ColabFold MSAs <code>.a3m of the library sequences without one yet (a data input, not in the repo; '
                         'default $MSR_ME_NEW_MSAS, else <data_dir>/new_msas)')
    ap.add_argument('--skip_msa', action='store_true', help='write mutations/fasta before the MSAs exist (pre-flight only)')
    args = ap.parse_args()

    DATA = paths.baseline_data(args.data_dir)
    args.name = args.name or paths.variant_info(DATA).get('name')
    if not args.name:
        sys.exit(f'{DATA} has no variant.json: pass --name')
    args.new_msas = args.new_msas or os.path.join(DATA, 'new_msas')
    libs = pd.read_csv(os.path.join(DATA, 'libraries.csv'))
    index = json.load(open(os.path.join(DATA, 'msa_index.json')))
    if os.path.isdir(args.new_msas):
        for code in libs['code']:
            p = os.path.join(args.new_msas, f'{code}.a3m')
            if os.path.exists(p):
                seq = libs.loc[libs.code == code, 'wt_seq'].item()
                index.setdefault(seq, p)

    missing = libs[~libs['wt_seq'].isin(index)]
    if args.mode == 'fasta':
        fa = os.path.join(DATA, f'{args.name}_missing_msas.fasta')
        with open(fa, 'w') as f:
            for _, r in missing.iterrows():
                f.write(f">{r['code']}\n{r['wt_seq']}\n")
        print(f'{len(missing)} sequences without an MSA -> {fa}')
        return

    assert missing.empty or args.skip_msa, f'no MSA for {missing.code.tolist()}'
    root = os.path.join(args.me_root, 'data', args.name)
    for sub in ['mutations', 'fasta', 'msa']:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for _, r in libs.iterrows():
        code = r['code']
        os.makedirs(os.path.join(root, 'fasta', code), exist_ok=True)
        with open(os.path.join(root, 'fasta', code, f'{code}.fasta'), 'w') as f:
            f.write(f">{code}\n{r['wt_seq']}\n")
        if r['wt_seq'] not in index:
            continue
        msa_d = os.path.join(root, 'msa', code)
        os.makedirs(msa_d, exist_ok=True)
        shutil.copyfile(index[r['wt_seq']], os.path.join(msa_d, f'{code}.a3m'))
    if os.path.exists(os.path.join(DATA, 'variant.json')):      # lets the inference script check the split it is asked to score
        shutil.copyfile(os.path.join(DATA, 'variant.json'), os.path.join(root, 'variant.json'))
    for split in ['train', 'val', 'valfull', 'test']:
        if split == 'valfull' and not os.path.exists(os.path.join(DATA, 'items_valfull.csv')):
            continue                                            # tables prepared before valfull existed
        df = to_me(pd.read_csv(os.path.join(DATA, f'items_{split}.csv'), low_memory=False))
        out = os.path.join(root, 'mutations', f'{args.name}_{split}.csv')
        df.to_csv(out, index=False)
        print(split, len(df), df['pdb_id'].nunique(), 'proteins ->', out)

    # Every single that is a component of a test double, so the model's own additive prediction (and from it the
    # predicted epistasis) exists for every test double. Unmeasured singles get a placeholder ddg (measured=False);
    # this file is only read for its predictions, never scored.
    test = pd.read_csv(os.path.join(DATA, 'items_test.csv'), low_memory=False)
    measured = {(r.code, r.mut_type): r.ddG_ML for r in test.itertuples() if ':' not in r.mut_type}
    comp = {}
    for r in test.itertuples():
        for m in r.mut_type.split(':'):
            comp.setdefault((r.code, m), dict(code=r.code, code_wt=r.code_wt, bb_mut=r.bb_mut, wt_seq=r.wt_seq,
                                              mut_type=m, kind='component', ddG_ML=measured.get((r.code, m), 0.0)))
    comp = to_me(pd.DataFrame(list(comp.values())))
    comp['measured'] = [(c, m) in measured for c, m in zip(comp.pdb_id, comp.mut_info)]
    out = os.path.join(root, 'mutations', f'{args.name}_testcomp.csv')
    comp.to_csv(out, index=False)
    print('test components', len(comp), 'measured', int(comp.measured.sum()), '->', out)


if __name__ == '__main__':
    main()
