#!/usr/bin/env python3
"""Download and extract the 522 Domainome 1.0 domain structures used in benchmarking.

Domainome benchmarking (Beltran et al., Nature 2025) uses the pre-sliced domain PDB
structures distributed in the SPURS benchmark archive (`cyclization9/SPURS` on HuggingFace:
`data/dataset/Domainome/files_to_reproduce_analyses/analysis_files/pdb_files/<domain_ID>.pdb`,
522 structures covering all human and designed Rocklin domains in Domainome 1.0, including
constructs from >2,700-aa proteins absent from AlphaFold DB single-fragment entries). The
same `files_to_reproduce_analyses/analysis_files/pdb_files/` directory is also published in
the upstream Domainome 1.0 Zenodo record (14356805 / DOI 10.5281/zenodo.11043642), alongside
`Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip`.

Writes, under `--out` (default `<repo>/data/domainome1/alphafold_structures`):

    <domain_ID>.pdb    (522 raw domain structures consumed by preprocess_domainome.py)

With `--download_table`, also downloads and extracts the raw Domainome Supplementary Table 5
TSV (`Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt`) from Zenodo record 14356805
to `--table_out` (default `<repo>/data/domainome1/SupplementaryTable5.txt`).

Usage:
    python preprocessing/download_domainome_structures.py [--out DIR] [--download_table]
    python preprocessing/download_domainome_structures.py --source zenodo --download_table
    python preprocessing/download_domainome_structures.py --archive /path/to/data.tar.gz [--out DIR]
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

SOURCES = {
    "spurs-hf": {
        "url": "https://huggingface.co/cyclization9/SPURS/resolve/main/data.tar.gz",
        "filename": "spurs_data.tar.gz",
        "algo": "sha256",
        "checksum": "f2e2c1e32320392df7d837fa9b66b65076cb6d3e5e638163888cd40b2a857330",
        "pdb_suffix": "files_to_reproduce_analyses/analysis_files/pdb_files/",
        "description": "SPURS HuggingFace archive (cyclization9/SPURS data.tar.gz, ~12.1 GB)",
    },
    "spurs-dropbox": {
        "url": (
            "https://www.dropbox.com/scl/fi/uo4e6lvptyy9df5xfulsc/data.tar.gz"
            "?rlkey=voi6fxu6ojbzwdk67jlooy8kb&st=4iinnpbc&dl=1"
        ),
        "filename": "spurs_data.tar.gz",
        "algo": "sha256",
        "checksum": "f2e2c1e32320392df7d837fa9b66b65076cb6d3e5e638163888cd40b2a857330",
        "pdb_suffix": "files_to_reproduce_analyses/analysis_files/pdb_files/",
        "description": "SPURS Dropbox mirror (data.tar.gz, ~12.1 GB)",
    },
    "zenodo": {
        "url": "https://zenodo.org/api/records/14356805/files/files_to_reproduce_analyses.tar.gz/content",
        "filename": "files_to_reproduce_analyses.tar.gz",
        "algo": "md5",
        "checksum": "330ac11247bdccc7ea994c5f9d700048",
        "pdb_suffix": "files_to_reproduce_analyses/analysis_files/pdb_files/",
        "description": "Domainome 1.0 Zenodo record 14356805 (files_to_reproduce_analyses.tar.gz, ~4.1 GB)",
    },
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
    pdb_dir_suffix: str = "files_to_reproduce_analyses/analysis_files/pdb_files/",
) -> int:
    """Extract only Domainome `<domain_ID>.pdb` files from a `.tar.gz` archive into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
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
            extracted += 1
    return extracted


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
        "--source",
        choices=list(SOURCES.keys()),
        default="spurs-hf",
        help=(
            "Upstream archive source when --archive is not passed: 'spurs-hf' (HuggingFace "
            "cyclization9/SPURS data.tar.gz, default), 'spurs-dropbox' (SPURS Dropbox mirror), "
            "or 'zenodo' (Domainome 1.0 Zenodo record 14356805 files_to_reproduce_analyses.tar.gz)."
        ),
    )
    p.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional local path to an already-downloaded data.tar.gz or files_to_reproduce_analyses.tar.gz.",
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
    if existing >= EXPECTED_PDB_COUNT and not args.force:
        print(f"{out_dir} already has {existing} raw Domainome PDB structures, skipping.", flush=True)
    else:
        spec = SOURCES[args.source]
        if args.archive is not None:
            archive_path = args.archive.resolve()
            if not archive_path.is_file():
                raise FileNotFoundError(f"Archive not found: {archive_path}")
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            archive_path = cache_dir / spec["filename"]
            download_with_checksum(
                spec["url"], archive_path, spec["algo"], spec["checksum"], spec["description"]
            )

        print(f"Extracting Domainome PDB structures from {archive_path} into {out_dir} ...", flush=True)
        n_extracted = extract_domainome_pdbs(archive_path, out_dir, spec["pdb_suffix"])
        print(f"Extracted {n_extracted} PDB structures to {out_dir}.", flush=True)
        if n_extracted < EXPECTED_PDB_COUNT:
            raise RuntimeError(
                f"Expected at least {EXPECTED_PDB_COUNT} Domainome PDB structures, extracted {n_extracted}."
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
