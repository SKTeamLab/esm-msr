"""Calibrating the batch sizes: run the sizer once per dataset on its worst-case proteins, keep the answer in a table.

`--batch_mode table` (the default of the baseline scripts) reads one batch size per model and dataset from
`batch_sizes_gpu4.json`; ProteinGym stays dynamic (`auto`) because its proteins differ most. This module builds the table:

  python <model>_inference.py --calibrate_batch_sizes OUT.json ...   # on the target GPU, one job per model
  python calibration.py merge OUT_a.json OUT_b.json ... -o batch_sizes_gpu4.json

For each dataset the script takes its `k` longest structures (memory per item grows with length), builds a saturated work list
for each (every single mutation, plus random double mutants, `cap` variants), runs its normal scoring path with the
AutoBatchSizer, and keeps the smallest batch the sizer settled on: one size that fits the heaviest protein of the dataset. A list
the sizer gets through before memory stops it only gives a lower bound (the sizer cannot grow past the items it has left), so a
run whose largest batch is more than a quarter of its items is repeated with a list four times longer, up to `max_cap`. The
table's `_meta` says which GPU, headroom and precision it is for; a different GPU or precision needs a new table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
       'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
DEFAULT_TABLE = Path(__file__).resolve().parent / 'batch_sizes_gpu4.json'


def pdb_chain_seq(path, chain='A'):
    """One-letter sequence of `chain`: one residue per (resnum, icode) with a CA atom, 'X' for non-standard residues. The index
    in this string (1-based) is the position in structure-sequence numbering that the benchmark rows use."""
    seq, seen = [], set()
    with open(path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
                key = line[22:27]
                if key not in seen:
                    seen.add(key)
                    seq.append(AA3.get(line[17:20].strip(), 'X'))
    return ''.join(seq)


def mixed_variants(seq, cap=4096, seed=0):
    """A saturated work list for `seq`: every single substitution (subsampled to cap // 2 if there are more, repeated up to cap // 2
    if there are fewer: a protein has only 19 L of them, and a list the sizer exhausts before memory does gives a lower bound, not
    the limit; a method that deduplicates just ignores the repeats) and cap // 2 random double mutants at distinct positions, as
    mutation strings ('A12G', 'A12G:C30D', sorted by position). Deterministic."""
    rng = np.random.default_rng(seed)
    pos = [i for i, a in enumerate(seq) if a in AA]
    singles = [(i, a, b) for i in pos for b in AA if b != seq[i] for a in [seq[i]]]
    if len(singles) > cap // 2:
        singles = [singles[j] for j in sorted(rng.choice(len(singles), cap // 2, replace=False))]
    if singles and len(singles) < cap // 2:
        singles = (singles * -(-(cap // 2) // len(singles)))[:cap // 2]
    out = [f'{a}{i + 1}{b}' for i, a, b in singles]
    if len(pos) >= 2:
        for _ in range(cap // 2):
            p, q = sorted(rng.choice(pos, 2, replace=False))
            out.append(f'{seq[p]}{p + 1}{rng.choice(list(AA.replace(seq[p], "")))}:{seq[q]}{q + 1}{rng.choice(list(AA.replace(seq[q], "")))}')
    return out


def longest(structures, k=2):
    """The k longest of `structures`, a list of dicts with at least `seq` (kept in order of length, longest first)."""
    return sorted(structures, key=lambda s: -len(s['seq']))[:k]


def structures_of(frame, chain_col='chain', seq_fn=None):
    """Distinct structures of a benchmark rows frame (pdb_file, code, chain) as dicts with their sequence."""
    seq_fn = seq_fn or (lambda r: pdb_chain_seq(r['pdb_file'], r[chain_col]))
    out, seen = [], set()
    for r in frame[['pdb_file', 'code', chain_col]].drop_duplicates().to_dict('records'):
        key = (r['pdb_file'], r[chain_col])
        if key not in seen:
            seen.add(key)
            out.append({**r, 'chain': r[chain_col], 'seq': seq_fn(r)})
    return out


def worst(infos):
    """One info dict out of the sizer reports of one scoring call (a call can run several groups, e.g. singles and doubles): the
    smallest batch with the item count of the group it came from, OOMs summed."""
    infos = [i for i in infos if i]
    w = min(infos, key=lambda i: i['batch_max'])
    return {'batch_max': w['batch_max'], 'n_items': w.get('n_items', 0), 'batch_ooms': sum(i.get('batch_ooms', 0) for i in infos)}


def list_bounded(info):
    """Whether the work list, not memory, may have stopped the sizer: its largest batch is over a quarter of the items it had."""
    return int(info['batch_max']) * 4 > int(info.get('n_items', 0)) > 0


def calibrate(datasets, run_one, k=2, cap=4096, max_cap=65536, log=print):
    """{dataset: batch} for `datasets` = {name: [structure dicts]}: for each dataset's k longest structures,
    run_one(structure, mutation_strings) -> the sizer's info dict ({'batch_max', 'n_items', 'batch_ooms'}); the smallest batch
    wins. A list that bounds the answer is repeated four times longer, up to max_cap variants (`list_bounded` in the detail if
    it still bounds it)."""
    sizes, detail = {}, {}
    for name, structs in datasets.items():
        found = []
        for s in longest(structs, k):
            t0, c = time.time(), cap
            while True:
                info = run_one(s, mixed_variants(s['seq'], c))
                if not list_bounded(info) or c >= max_cap:
                    break
                log(f'  [calibrate] {name}: {s.get("code", "?")} batch {info["batch_max"]} of {info["n_items"]} items: list too short, x4')
                c *= 4
            found.append(dict(code=s.get('code', '?'), length=len(s['seq']), batch=int(info['batch_max']), ooms=int(info.get('batch_ooms', 0)),
                              variants=c, list_bounded=list_bounded(info)))
            log(f'  [calibrate] {name}: {s.get("code", "?")} L={len(s["seq"])} -> batch {info["batch_max"]} '
                f'(ooms {info.get("batch_ooms", 0)}, {c} variants, {time.time() - t0:.0f}s)')
        if found:
            sizes[name] = min(f['batch'] for f in found)
            detail[name] = found
    return sizes, detail


def gpu_meta(headroom, dtype, k, cap):
    import datetime
    import subprocess
    import torch
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        commit = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {'gpu': p.name, 'gpu_gb': round(p.total_memory / 2 ** 30, 2), 'headroom': headroom, 'dtype': dtype, 'k_longest': k,
            'cap_variants': cap, 'date': datetime.date.today().isoformat(), 'commit': commit}


def add_calibration_args(parser):
    parser.add_argument('--calibrate_batch_sizes', metavar='OUT.json', default=None,
                        help='Instead of benchmarking: run the sizer on the worst-case proteins of every dataset and write the '
                             'batch sizes to OUT.json (calibration.py merge builds the table). Run it on the target GPU.')
    parser.add_argument('--calibration_datasets', default=None, help='comma-separated dataset names to calibrate (default all)')
    parser.add_argument('--calibration_k', type=int, default=2, help='worst-case proteins (longest) per dataset')
    parser.add_argument('--calibration_cap', type=int, default=4096, help='variants in the first saturated work list of each structure')
    parser.add_argument('--calibration_max_cap', type=int, default=65536,
                        help='a list that bounds the sizer (largest batch > 1/4 of its items) is made 4x longer, up to this many variants')
    return parser


def rows_structures(only=None):
    """{dataset: (structures, has_doubles)} for the datasets every script shares: the 12 external sets, the 7 DMS assays and
    Domainome (benchmark_rows.py). `only`: a collection of dataset names to keep."""
    import benchmark_rows as br
    out = {}
    for n in br.EXTERNAL:
        if only is None or n in only:
            f = br.external_rows(n)
            out[n] = (structures_of(f), bool(f.mut_type.str.contains(':').any()))
    for a in br.DMS:
        if only is None or a in only:
            f = br.dms_rows(a)
            out[a] = (structures_of(f), bool(f.mut_type.str.contains(':').any()))
    if only is None or 'domainome' in only:
        out['domainome'] = (structures_of(br.domainome_rows()), False)
    return out


def parse_only(arg):
    return {x for x in arg.split(',') if x} if arg else None


def save_entries(path, entries, details, meta):
    """Write {model: {dataset: batch}} as a standalone JSON (merge builds the table); ProteinGym is `auto` for every model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = {'_meta': {'models': {m: meta for m in entries}}, '_detail': details}
    for m, sizes in entries.items():
        out[m] = {**sizes, 'proteingym': 'auto'}
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f'wrote {path}')


def merge(paths, out):
    table = {'_meta': {'models': {}}, '_detail': {}}
    for p in paths:
        d = json.load(open(p))
        table['_meta']['models'].update(d['_meta']['models'])
        table['_detail'].update(d.get('_detail', {}))
        table.update({k: v for k, v in d.items() if not k.startswith('_')})
    gpus = {m['gpu'] for m in table['_meta']['models'].values()}
    heads = {m['headroom'] for m in table['_meta']['models'].values()}
    assert len(gpus) == 1 and len(heads) == 1, f'entries from different GPUs or headrooms: {gpus} {heads}'
    table['_meta'].update({'gpu': gpus.pop(), 'headroom': heads.pop()})
    with open(out, 'w') as f:
        json.dump(table, f, indent=1, sort_keys=True)
    print(f'wrote {out}: ' + ', '.join(f'{m} ({len([k for k in v])})' for m, v in table.items() if not m.startswith('_')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('merge')
    m.add_argument('files', nargs='+')
    m.add_argument('-o', '--out', default=str(DEFAULT_TABLE))
    a = ap.parse_args()
    merge(a.files, a.out)
