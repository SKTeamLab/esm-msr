"""Build or convert the per-library item cache that training reads. CPU work only.

Training no longer touches the raw table, the structures or the structure encoder (``docs/trim_audit.md`` §2.4): it reads
``<cache_path>/<code>_ddG_ML_<version>.pkl`` for every library of a split and ``<cache_path>/manifest_<version>.json``, which lists
those libraries in loader order. This script makes both.

    # from the raw table (what training used to do on its first launch)
    python analysis_scripts/build_cache.py build --raw_data_file ... --af_model_folder ... --mut_structures_root ... \\
        --split_file data/splits_oct06_capped.pkl --cache_path CACHE --benchmark_data_path data/preprocessed --cache_version c6

    # from an older cache, without the structure encoder: c5 -> c6 drops the `cond` items (nothing trains on them)
    python analysis_scripts/build_cache.py convert --raw_data_file ... --af_model_folder ... --split_file ... \\
        --cache_path CACHE --from_version c5 --to_version c6

``convert`` still reads the raw table, but only to list the libraries of each split in the order the training loaders always had (that order
fixes the sampler's random stream). The items themselves are copied from the old cache unchanged except for the dropped subset.
"""
import argparse
import logging
import os
import pickle
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import pandas as pd

from esm_msr.preprocess_megascale import MegaScaleDatasetPreprocessor, write_manifest

BENCHMARKS = {'ptmuld': 'ptmuld_mapped.csv', 's461': 's461_mapped.csv', 'ssym': 'ssym_mapped.csv'}


def _preprocessor(a):
    pre = MegaScaleDatasetPreprocessor(a.raw_data_file, a.af_model_folder, spurs_override=a.remove_spurs_homologs, censor_range=True)
    pre.create_training_splits(a.split_file)
    return pre


def build(a):
    from esm.pretrained import ESM3_structure_encoder_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    from esm_msr.cache_build import LibraryCacheBuilder
    from tqdm import tqdm
    pre = _preprocessor(a)
    tokenizer, structure_encoder = EsmSequenceTokenizer("cpu"), ESM3_structure_encoder_v0("cpu")
    os.makedirs(a.cache_path, exist_ok=True)
    for scaffold in a.splits:
        df = pre.split_dfs[scaffold]
        for wt_code in tqdm(list(df['code_wt'].unique()), desc=f"Building {scaffold} caches"):
            df_prot = df.loc[df['code_wt'] == wt_code]
            for code in df_prot['code'].unique():
                cache_file = os.path.join(a.cache_path, f"{code}_ddG_ML_{a.cache_version}.pkl")
                if getattr(a, 'skip_existing', False) and os.path.exists(cache_file):
                    continue
                LibraryCacheBuilder(
                    code, tokenizer, a.mut_structures_root, a.cache_path, score_name='ddG_ML', structure_encoder=structure_encoder,
                    incl_destab_bb=True, dG_wt=pre.dG_wt.get(code), cache_version=a.cache_version,
                ).build(df_prot.loc[df_prot['code'] == code])
    found = []
    for name, filename in BENCHMARKS.items():
        cache_file = os.path.join(a.cache_path, f"{name}_ddG_{a.cache_version}.pkl")
        if getattr(a, 'skip_existing', False) and os.path.exists(cache_file):
            found.append(name)
            continue
        path = os.path.join(a.benchmark_data_path, filename) if a.benchmark_data_path else None
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            if 'ddG' in df.columns:
                LibraryCacheBuilder(name, tokenizer, '', a.cache_path, score_name='ddG', structure_encoder=structure_encoder,
                                    incl_destab_bb=False, cache_version=a.cache_version).build(df)
                found.append(name)
        else:
            logging.warning(f"benchmark {name}: {path} not found; left out of the manifest")
    out = write_manifest(a.cache_path, a.cache_version, {s: pre.library_order(s) for s in ('train', 'val')}, found, a.split_file,
                         note=f"built from {os.path.basename(a.raw_data_file)}")
    logging.info(f"wrote {out}")


def convert(a):
    pre = _preprocessor(a)
    splits = {s: pre.library_order(s) for s in ('train', 'val')}
    names = sorted({c for v in splits.values() for c in v}) + [b for b in BENCHMARKS]
    dropped = {}
    found = []
    for code in names:
        score = 'ddG' if code in BENCHMARKS else 'ddG_ML'
        src = os.path.join(a.cache_path, f"{code}_{score}_{a.from_version}.pkl")
        if not os.path.exists(src):
            if code in BENCHMARKS:
                continue
            raise FileNotFoundError(f"{src}: the {a.from_version} cache lacks library {code}; build it with `build` instead")
        with open(src, 'rb') as f:
            blob = pickle.load(f)
        items = blob['items']
        keep = [i for i in items if i.get('subset_type') not in a.drop]
        for i in items:
            if i.get('subset_type') in a.drop:
                dropped[i['subset_type']] = dropped.get(i['subset_type'], 0) + 1
        with open(os.path.join(a.cache_path, f"{code}_{score}_{a.to_version}.pkl"), 'wb') as f:
            pickle.dump({'items': keep, 'floor_stats': {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        if code in BENCHMARKS:
            found.append(code)
    out = write_manifest(a.cache_path, a.to_version, splits, found, a.split_file,
                         note=f"converted from {a.from_version}; dropped {dropped}")
    logging.info(f"wrote {out}; dropped items: {dropped}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('build', 'convert'):
        p = sub.add_parser(name)
        p.add_argument('--raw_data_file', required=True)
        p.add_argument('--af_model_folder', required=True)
        p.add_argument('--split_file', required=True)
        p.add_argument('--cache_path', required=True)
        p.add_argument('--remove_spurs_homologs', action='store_true')
    b = sub.choices['build']
    b.add_argument('--mut_structures_root', required=True)
    b.add_argument('--benchmark_data_path', default=None)
    b.add_argument('--cache_version', default='c6')
    b.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val', 'test'])
    b.add_argument('--skip_existing', action='store_true', default=False)
    c = sub.choices['convert']
    c.add_argument('--from_version', default='c5')
    c.add_argument('--to_version', default='c6')
    c.add_argument('--drop', nargs='*', default=['cond'])
    a = ap.parse_args(argv)
    {'build': build, 'convert': convert}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
