#!/usr/bin/env python3
"""Download and extract the 522 Domainome 1.0 domain structures used in benchmarking.

Domainome 1.0 benchmarking (Beltran et al., Nature 2025) uses the 522 pre-sliced domain PDB
structures published in the official Domainome 1.0 Zenodo release (`https://zenodo.org/records/14356805`,
concept DOI `10.5281/zenodo.11043642`, `files_to_reproduce_analyses.tar.gz`, under
`files_to_reproduce_analyses/analysis_files/pdb_files/<domain_ID>.pdb`). This structure set covers
all 522 human Pfam and designed control domains assayed in Domainome 1.0, including constructs from
>2,700-residue human proteins and de novo designed domains absent from single-fragment AlphaFold DB
entries.

Writes, under `--out` (default `<repo>/data/domainome1/alphafold_structures`):

    <domain_ID>.pdb    (522 raw domain structures consumed by preprocess_domainome.py)

With `--download_table`, also downloads and extracts the raw Domainome 1.0 Supplementary Table 5
TSV (`Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt`) from Zenodo record 14356805
to `--table_out` (default `<repo>/data/domainome1/SupplementaryTable5.txt`).

Usage:
    python preprocessing/download_domainome_structures.py [--out DIR] [--download_table]
    python preprocessing/download_domainome_structures.py --archive /path/to/files_to_reproduce_analyses.tar.gz [--out DIR]
"""
import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

EXPECTED_PDB_COUNT = 522

STRUCTURES_ZENODO = {
    "url": "https://zenodo.org/api/records/14356805/files/files_to_reproduce_analyses.tar.gz/content",
    "filename": "files_to_reproduce_analyses.tar.gz",
    "algo": "md5",
    "checksum": "330ac11247bdccc7ea994c5f9d700048",
    "pdb_suffix": "files_to_reproduce_analyses/analysis_files/pdb_files/",
    "description": "Domainome 1.0 Zenodo record 14356805 (files_to_reproduce_analyses.tar.gz, ~4.1 GB)",
}

TABLE_ZENODO = {
    "url": (
        "https://zenodo.org/api/records/14356805/files/"
        "Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip/content"
    ),
    "filename": "Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip",
    "algo": "md5",
    "checksum": "2f3ff661e25aa5e6d92b664dc0709c17",
}


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download_with_checksum(url: str, dest: Path, algo: str, expected: str, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        actual = file_digest(dest, algo)
        if actual == expected:
            print(f"Archive {dest} already exists and matches {algo}; reusing.", flush=True)
            return
        print(f"Existing {dest} failed {algo} check ({actual} != {expected}); re-downloading.", flush=True)
        dest.unlink()

    print(f"Downloading {label} -> {dest} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "esm-msr-download/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1 << 20)

    actual = file_digest(dest, algo)
    if actual != expected:
        raise RuntimeError(f"{algo} mismatch for {dest}: expected {expected}, got {actual}")


def extract_domainome_pdbs(
    archive_path: Path,
    out_dir: Path,
    pdb_dir_suffix: str = STRUCTURES_ZENODO["pdb_suffix"],
    clean_stale: bool = True,
) -> int:
    """Extract only Domainome `<domain_ID>.pdb` files from `files_to_reproduce_analyses.tar.gz` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_names = set()
    with tarfile.open(archive_path, mode="r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            norm_name = member.name.replace("\\", "/")
            if pdb_dir_suffix not in norm_name:
                continue
            base = os.path.basename(norm_name)
            if (
                not base.endswith(".pdb")
                or base.startswith("._")
                or base.startswith(".")
                or ":Zone.Identifier" in base
                or base.endswith("_truncated_renumbered.pdb")
            ):
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            dest = out_dir / base
            with src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_names.add(base)

    if clean_stale and len(extracted_names) == EXPECTED_PDB_COUNT:
        for p in out_dir.glob("*.pdb"):
            if not p.name.endswith("_truncated_renumbered.pdb") and p.name not in extracted_names:
                p.unlink()

    return len(extracted_names)


def download_supplementary_table5(table_out: Path, cache_dir: Path, keep_archive: bool = False) -> None:
    if table_out.exists():
        print(f"{table_out} exists, skipping table download.", flush=True)
        return
    table_out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = cache_dir / TABLE_ZENODO["filename"]
    download_with_checksum(
        TABLE_ZENODO["url"],
        zip_dest,
        TABLE_ZENODO["algo"],
        TABLE_ZENODO["checksum"],
        "Domainome 1.0 Supplementary Table 5 (~40 MB)",
    )
    with zipfile.ZipFile(zip_dest) as zf:
        txt_members = [
            m for m in zf.namelist()
            if m.endswith(".txt") and not os.path.basename(m).startswith("._") and "__MACOSX" not in m
        ]
        if not txt_members:
            raise RuntimeError(f"No .txt member found inside {zip_dest}")
        with zf.open(txt_members[0]) as src, open(table_out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not keep_archive:
        zip_dest.unlink(missing_ok=True)
    print(f"Wrote {table_out}", flush=True)


def count_raw_pdbs(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(
        1 for p in out_dir.glob("*.pdb")
        if not p.name.endswith("_truncated_renumbered.pdb") and not p.name.startswith("._")
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "data" / "domainome1" / "alphafold_structures"
    default_table = repo_root / "data" / "domainome1" / "SupplementaryTable5.txt"

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output directory for extracted <domain_ID>.pdb files (default: {default_out}).",
    )
    p.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional local path to an already-downloaded files_to_reproduce_analyses.tar.gz.",
    )
    p.add_argument(
        "--download_table",
        action="store_true",
        help="Also download Domainome 1.0 Supplementary Table 5 from Zenodo record 14356805.",
    )
    p.add_argument(
        "--table_out",
        type=Path,
        default=default_table,
        help=f"Destination for Supplementary Table 5 TSV when --download_table is set (default: {default_table}).",
    )
    p.add_argument(
        "--keep_archive",
        action="store_true",
        help="Keep downloaded archive(s) in <out>/_downloads instead of deleting after extraction.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract structures even if 522 PDBs already exist in --out.",
    )
    args = p.parse_args()

    out_dir = args.out.resolve()
    cache_dir = out_dir / "_downloads"

    existing = count_raw_pdbs(out_dir)
    if existing == EXPECTED_PDB_COUNT and not args.force:
        print(f"{out_dir} already has {existing} raw Domainome PDB structures, skipping.", flush=True)
    else:
        if args.archive is not None:
            archive_path = args.archive.resolve()
            if not archive_path.is_file():
                raise FileNotFoundError(f"Archive not found: {archive_path}")
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            archive_path = cache_dir / STRUCTURES_ZENODO["filename"]
            download_with_checksum(
                STRUCTURES_ZENODO["url"],
                archive_path,
                STRUCTURES_ZENODO["algo"],
                STRUCTURES_ZENODO["checksum"],
                STRUCTURES_ZENODO["description"],
            )

        print(f"Extracting Domainome PDB structures from {archive_path} into {out_dir} ...", flush=True)
        n_extracted = extract_domainome_pdbs(archive_path, out_dir, STRUCTURES_ZENODO["pdb_suffix"])
        print(f"Extracted {n_extracted} PDB structures to {out_dir}.", flush=True)
        if n_extracted != EXPECTED_PDB_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_PDB_COUNT} Domainome PDB structures, extracted {n_extracted}."
            )
        if args.archive is None and not args.keep_archive:
            shutil.rmtree(cache_dir, ignore_errors=True)

    if args.download_table:
        download_supplementary_table5(args.table_out.resolve(), cache_dir, keep_archive=args.keep_archive)
        if not args.keep_archive:
            shutil.rmtree(cache_dir, ignore_errors=True)

    print(
        f"Next step: python preprocessing/preprocess_domainome.py --structures_dir {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
