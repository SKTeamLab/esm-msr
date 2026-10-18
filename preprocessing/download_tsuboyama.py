#!/usr/bin/env python3
"""Download the Tsuboyama et al. 2023 mega-scale data that training reads, from Zenodo record 7992926.

Writes, under --out (default <repo>/data/tsuboyama):

    Tsuboyama2023_Dataset2_Dataset3_20230416.csv    -> --raw_data_file
    AlphaFold_model_PDBs/                           -> --af_model_folder

The structures keep their Zenodo names (':' where the table's WT_name has '|'); the preprocessor maps between them.
Each zip is checked against the md5 Zenodo publishes. Run on a node with internet (a login node on the Alliance
clusters); steps whose output already exists are skipped.

Usage:
    python preprocessing/download_tsuboyama.py [--out DIR] [--keep_zips]
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

RECORD = 'https://zenodo.org/api/records/7992926'
CSV_NAME = 'Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
PDB_DIR = 'AlphaFold_model_PDBs'


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def fetch(entry: dict, dest: Path) -> None:
    print(f"Downloading {entry['key']} ({entry['size'] / 1e6:.0f} MB) ...", flush=True)
    urllib.request.urlretrieve(entry['links']['self'], dest)
    expected = entry['checksum'].split(':', 1)[1]
    if md5sum(dest) != expected:
        raise RuntimeError(f"md5 mismatch for {dest}")


def main():
    default_out = Path(__file__).resolve().parent.parent / 'data' / 'tsuboyama'
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=default_out)
    p.add_argument('--keep_zips', action='store_true', help="keep the downloaded zips in <out>/zips")
    args = p.parse_args()

    with urllib.request.urlopen(RECORD) as r:
        files = {f['key']: f for f in json.load(r)['files']}
    out = args.out
    zips = out / 'zips'
    zips.mkdir(parents=True, exist_ok=True)

    if not (out / PDB_DIR).is_dir() or not any((out / PDB_DIR).iterdir()):
        z = zips / 'AlphaFold_model_PDBs.zip'
        fetch(files[z.name], z)
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.startswith(PDB_DIR + '/') and m.endswith('.pdb'):
                    zf.extract(m, out)
        print(f"{PDB_DIR}: {len(list((out / PDB_DIR).glob('*.pdb')))} structures")
    else:
        print(f"{out / PDB_DIR} exists, skipping")

    if not (out / CSV_NAME).exists():
        z = zips / 'Processed_K50_dG_datasets.zip'
        fetch(files[z.name], z)
        with zipfile.ZipFile(z) as zf, zf.open(f'Processed_K50_dG_datasets/{CSV_NAME}') as src, \
                open(out / CSV_NAME, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        print(f"wrote {out / CSV_NAME}")
    else:
        print(f"{out / CSV_NAME} exists, skipping")

    if not args.keep_zips:
        shutil.rmtree(zips, ignore_errors=True)
    print(f"Use: --raw_data_file {out / CSV_NAME} --af_model_folder {out / PDB_DIR}")


if __name__ == '__main__':
    main()
