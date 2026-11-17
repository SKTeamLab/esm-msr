#!/usr/bin/env python3
"""Download and prepare the MGnify Stability Dataset (Cho, Tsuboyama et al., 2026).

Downloads:
  1. Metadata & stability tables from Zenodo record 19411306:
       - mgnify_training_index.csv (138 MB)
       - 230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv (2.23 GB)
  2. AlphaFold2 structure shards (.tar of .cif.gz) from the official Google Drive folder
     (1SPHsaC2tcwtnQuU0gbZeEFUuol-bhSZM) and converts needed structures to 1-indexed
     Chain-A PDB files under <out>/structures/<bucket>/<stem>.pdb (bucket = id % 1000).

Usage:
    python preprocessing/download_mgnify.py [--out data/mgnify]
    python preprocessing/download_mgnify.py --tables_only [--out data/mgnify]
    python preprocessing/download_mgnify.py --structures_only [--needed_stems stems.txt] [--out data/mgnify]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
import urllib.request

from Bio.PDB import MMCIFParser, PDBIO

ZENODO_BASE = "https://zenodo.org/api/records/19411306/files"

TABLES = {
    "mgnify_training_index.csv": {
        "url": f"{ZENODO_BASE}/mgnify_training_index.csv/content",
        "size": 137_910_739,
        "md5": "c585736d10302a92aa935f29eeddcde6",
    },
    "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv": {
        "url": f"{ZENODO_BASE}/230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv/content",
        "size": 2_233_150_308,
        "md5": "371a3ebbfdb5c3ca1f0f2af8b2b8d3a8",
    },
}

# (tar_filename, google_drive_file_id, expected_bytes, index_splits_covered)
SHARDS: List[Tuple[str, str, int, Tuple[str, ...]]] = [
    ("structures_test_shard_0.tar", "1BrnBAlfIwZaeZRgF6lAx7ckQKtQA0Pfk", 43_290_112, ("test",)),
    (
        "structures_validation_shard_0.tar",
        "1abGA2ywlab7GkOYWnYbkSCfmLXmgRPjf",
        31_060_480,
        ("validation", "validation_online"),
    ),
    ("structures_train_shard_0.tar", "1igUS_PHhujBfuFzZvVdwFzdgN40LiW6h", 634_121_728, ("train",)),
    ("structures_train_shard_1.tar", "1sXzsJao5DeKxpZm9WoN-66XPXwRjOixS", 634_487_808, ("train",)),
    ("structures_train_shard_2.tar", "1zMijkqabtRvidlydkI4BpIayh7VsscBF", 634_634_240, ("train",)),
    ("structures_train_shard_3.tar", "1TB6wWYhGg9wrvX9jCOm-PpUDX39Uj9td", 634_320_896, ("train",)),
    ("structures_train_shard_4.tar", "1LEP9TfWogpiDWmPQk9WOtOF_HXBlLmi8", 634_217_984, ("train",)),
    ("structures_train_shard_5.tar", "1VaPWhDzCRBIF2QjbiF5Nfl5wTtw4DuYW", 634_431_488, ("train",)),
    ("structures_train_shard_6.tar", "1n2tzn_sXkUQeJViUAkDXEVr55LjHeGd3", 633_852_416, ("train",)),
    ("structures_train_shard_7.tar", "1XKfq74qKCpNg0lTsyZJ1-rlQceWP_byJ", 634_807_296, ("train",)),
    ("structures_train_shard_8.tar", "12tw2XQoPVcFcT3DXd9B-9KRdREqFFNZG", 634_600_448, ("train",)),
    ("structures_train_shard_9.tar", "1iO9i1BMGACjG31QncNXofK_w4MNjhlb8", 634_503_168, ("train",)),
    ("structures_train_shard_10.tar", "10GWIs4NrdWmXLM7FS_QuvuxT4fQACSDn", 634_841_600, ("train",)),
    ("structures_train_shard_11.tar", "1My1oKSoYf3vzGda3Gr9wwHNATg8f-fN4", 635_268_096, ("train",)),
    ("structures_train_shard_12.tar", "14qz2n5IhgCn9m_V2drk5oHgrC_I-tlbR", 634_868_736, ("train",)),
    ("structures_train_shard_13.tar", "1YJQvHu74nqQ1GilpqhTWzZTlFGYfBQ2v", 635_287_552, ("train",)),
    ("structures_train_shard_14.tar", "1uUM9eXanE1-tgAdsTeM1c1PUrfZgafvw", 634_964_480, ("train",)),
    ("structures_train_shard_15.tar", "1JerwdReeQbnif6fpX7yhT8TwRlmz4kiN", 634_864_640, ("train",)),
    ("structures_train_shard_16.tar", "1nS1Mk6lUnaDZrAuy3Qaq3yqZNacVYobj", 634_917_376, ("train",)),
    ("structures_train_shard_17.tar", "1rgjiAjI8NtweBMZ0g1yG-inRP4AfV0Vq", 635_466_240, ("train",)),
    ("structures_train_shard_18.tar", "1aZhVJ-S2BvZ7XHIbhUaRLWn2y7kWyd9h", 635_136_512, ("train",)),
    ("structures_train_shard_19.tar", "1DmtkLoAg8_RjMBjQAmCT-VTK5Q9fDn_r", 632_307_712, ("train",)),
]

STEM_NUM_RE = re.compile(r"rocklin_batch2_(\d+)")
BASE_NAME_RE = re.compile(r"^(rocklin_batch2_\d+)(?:_(.+))?$")


def scaffold_bucket(stem: str) -> str:
    """Return a 3-digit directory bucket (`id % 1000`) for a scaffold stem."""
    m = STEM_NUM_RE.search(stem)
    if m:
        return f"{int(m.group(1)) % 1000:03d}"
    return "other"


def structure_rel_path(stem: str) -> str:
    """Relative path under `<out_dir>` for a converted `.pdb` file."""
    return f"structures/{scaffold_bucket(stem)}/{stem}.pdb"


def md5sum(path: Path) -> str:
    """Compute the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _download_zenodo_file(url: str, dest: Path, expected_md5: str, expected_size: int) -> None:
    """Download a file from Zenodo using urllib and verify its size and MD5 checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()

    print(f"Downloading {dest.name} ({expected_size / 1e6:.1f} MB) from Zenodo ...", flush=True)
    h = hashlib.md5()
    downloaded = 0
    next_report = 100 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out_f:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out_f.write(chunk)
            h.update(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                pct = 100.0 * downloaded / expected_size if expected_size else 0.0
                print(f"  {dest.name}: {downloaded / 1e6:.0f} / {expected_size / 1e6:.0f} MB ({pct:.1f}%)", flush=True)
                next_report += 100 * 1024 * 1024

    actual_md5 = h.hexdigest()
    if actual_md5 != expected_md5:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual_md5}")
    if expected_size and downloaded != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {dest.name}: expected {expected_size}, got {downloaded}")
    part.replace(dest)
    print(f"Verified {dest} (md5={actual_md5})", flush=True)


def download_tables(out_dir: Path, local_index_fallback: Optional[Path] = None) -> Dict[str, Path]:
    """Download `mgnify_training_index.csv` and `230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv`."""
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if local_index_fallback is None:
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root.parent / "leak_audit" / "data" / "raw" / "mgnify_index.csv"
        if candidate.is_file():
            local_index_fallback = candidate

    paths: Dict[str, Path] = {}
    for fname, meta in TABLES.items():
        dest = raw_dir / fname
        paths[fname] = dest
        if dest.is_file() and dest.stat().st_size == meta["size"] and md5sum(dest) == meta["md5"]:
            print(f"{dest} already exists and matches MD5, skipping.", flush=True)
            continue

        if fname == "mgnify_training_index.csv" and local_index_fallback is not None and local_index_fallback.is_file():
            if local_index_fallback.stat().st_size == meta["size"] and md5sum(local_index_fallback) == meta["md5"]:
                print(f"Copying verified local index {local_index_fallback} -> {dest}", flush=True)
                part = dest.with_name(dest.name + ".part")
                shutil.copy2(local_index_fallback, part)
                part.replace(dest)
                continue

        _download_zenodo_file(meta["url"], dest, meta["md5"], meta["size"])
    return paths


def _fast_cif_to_pdb(cif_text: str) -> Optional[str]:
    """Convert a standard single-chain AlphaFold mmCIF text to 80-col PDB format; return None on non-standard layout."""
    cols: List[str] = []
    in_loop = False
    atom_cols_done = False
    out_lines: List[str] = []
    serial = 0

    i_grp = i_sym = i_atm = i_res = i_seq = i_x = i_y = i_z = i_occ = i_b = -1
    i_alt: Optional[int] = None

    for ln in cif_text.splitlines():
        s = ln.strip()
        if s == "loop_":
            cols = []
            in_loop = True
            atom_cols_done = False
            continue
        if in_loop and s.startswith("_atom_site."):
            cols.append(s[len("_atom_site.") :])
            continue
        if cols and not atom_cols_done:
            atom_cols_done = True
            idx = {c: i for i, c in enumerate(cols)}
            try:
                i_grp = idx["group_PDB"]
                i_sym = idx["type_symbol"]
                i_atm = idx["label_atom_id"]
                i_alt = idx.get("label_alt_id")
                i_res = idx["label_comp_id"]
                i_seq = idx["label_seq_id"]
                i_x, i_y, i_z = idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"]
                i_occ = idx["occupancy"]
                i_b = idx["B_iso_or_equiv"]
            except KeyError:
                return None

        if atom_cols_done and (ln.startswith("ATOM ") or ln.startswith("HETATM")):
            if "'" in ln or '"' in ln:
                return None
            toks = ln.split()
            if len(toks) < len(cols):
                return None
            if toks[i_grp] != "ATOM":
                continue
            serial += 1
            if serial > 99_999:
                return None
            elem = toks[i_sym].upper()
            atom = toks[i_atm]
            atom_fld = f" {atom:<3s}" if (len(elem) == 1 and len(atom) < 4) else f"{atom:<4s}"
            alt_raw = toks[i_alt] if i_alt is not None else "."
            alt = " " if alt_raw in (".", "?") else alt_raw[0]
            resn = toks[i_res]
            resi = int(toks[i_seq])
            x, y, z = float(toks[i_x]), float(toks[i_y]), float(toks[i_z])
            occ = float(toks[i_occ])
            bfac = float(toks[i_b])
            out_lines.append(
                f"ATOM  {serial:5d} {atom_fld}{alt}{resn:>3s} A{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {elem:>2s}  \n"
            )
        elif atom_cols_done and (s == "#" or s.startswith("_") or s == "loop_"):
            in_loop = False
            atom_cols_done = False
            cols = []

    if not out_lines:
        return None
    out_lines.append("TER\nEND\n")
    return "".join(out_lines)


def cif_gz_bytes_to_pdb_str(raw_gz_bytes: bytes) -> str:
    """Decompress `.cif.gz` bytes and convert to 1-indexed Chain-A PDB string (BioPython fallback)."""
    cif_text = gzip.decompress(raw_gz_bytes).decode("utf-8")
    pdb_str = _fast_cif_to_pdb(cif_text)
    if pdb_str is not None:
        return pdb_str

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("mgnify", io.StringIO(cif_text))
    for model in structure:
        for chain in model:
            chain.id = "A"
            residues = list(chain)
            for res in residues:
                chain.detach_child(res.id)
            for idx_r, res in enumerate(residues, start=1):
                res.id = (res.id[0], idx_r, " ")
                chain.add(res)
            break
        break
    buf = io.StringIO()
    io_pdb = PDBIO()
    io_pdb.set_structure(structure)
    io_pdb.save(buf)
    return buf.getvalue()


def _convert_and_write_one(task: Tuple[bytes, str]) -> None:
    raw_gz_bytes, dest_path = task
    pdb_text = cif_gz_bytes_to_pdb_str(raw_gz_bytes)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(pdb_text)


def _download_gdrive_shard(filename: str, file_id: str, expected_bytes: int, dest: Path, max_attempts: int = 4) -> None:
    """Download a Google Drive structure shard using IPv4 curl and verify its exact byte size and tar readability."""
    if dest.is_file() and dest.stat().st_size == expected_bytes:
        try:
            with tarfile.open(dest, "r:") as tf:
                tf.next()
            print(f"Reusing existing shard {dest.name} ({expected_bytes / 1e6:.1f} MB)", flush=True)
            return
        except Exception:
            dest.unlink()

    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        if part.exists():
            part.unlink()
        print(
            f"Downloading {filename} ({expected_bytes / 1e6:.1f} MB, attempt {attempt}/{max_attempts}) ...",
            flush=True,
        )
        cmd = [
            "curl",
            "-4",
            "-fSL",
            "--connect-timeout",
            "30",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-o",
            str(part),
            url,
        ]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and part.is_file():
            sz = part.stat().st_size
            if sz == expected_bytes:
                try:
                    with tarfile.open(part, "r:") as tf:
                        tf.next()
                    part.replace(dest)
                    return
                except Exception as exc:
                    print(f"  Invalid tar for {filename}: {exc}", flush=True)
            else:
                print(
                    f"  Size mismatch for {filename}: got {sz} bytes, expected {expected_bytes}",
                    flush=True,
                )
        else:
            print(f"  curl exited with code {res.returncode} for {filename}", flush=True)

        part.unlink(missing_ok=True)
        if attempt < max_attempts:
            wait_s = 5 * attempt
            print(f"  Retrying in {wait_s}s ...", flush=True)
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download valid shard {filename} (id={file_id}) after {max_attempts} attempts.")


def infer_needed_stems_from_tables(out_dir: Path, all_wt: bool = False) -> Tuple[Set[str], Dict[str, str]]:
    """Determine needed structure stems from `mgnify_training_index.csv` (and `concat260429.csv` if present)."""
    index_csv = out_dir / "raw" / "mgnify_training_index.csv"
    if not index_csv.is_file():
        raise FileNotFoundError(f"Index CSV not found at {index_csv}; run --tables_only first.")

    best_stem_by_scaf: Dict[str, str] = {}
    split_by_stem: Dict[str, str] = {}
    mutated_in_index: Set[str] = set()

    with open(index_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nm = row["name"]
            pdb_name = row["PDB_name"]
            sp = row["split"]
            split_by_stem[nm] = sp
            prev = best_stem_by_scaf.get(pdb_name)
            if prev is None or nm < prev:
                best_stem_by_scaf[pdb_name] = nm
            if nm != pdb_name:
                mutated_in_index.add(pdb_name)

    if all_wt:
        needed = set(best_stem_by_scaf.values())
        return needed, split_by_stem

    mutated_scafs = set(mutated_in_index)
    concat_csv = out_dir / "raw" / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv"
    if concat_csv.is_file():
        with open(concat_csv, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["mgnify"] != "True":
                    continue
                m = BASE_NAME_RE.match(row["name"])
                if m and m.group(2) is not None:
                    mutated_scafs.add(m.group(1))

    needed = {best_stem_by_scaf[scaf] for scaf in mutated_scafs if scaf in best_stem_by_scaf}
    return needed, split_by_stem


def _split_map_from_index(out_dir: Path) -> Dict[str, str]:
    index_csv = out_dir / "raw" / "mgnify_training_index.csv"
    if not index_csv.is_file():
        return {}
    splits: Dict[str, str] = {}
    with open(index_csv, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            splits[row["name"]] = row["split"]
    return splits


def download_structures(
    out_dir: Path,
    needed_stems: Set[str],
    *,
    shards_dir: Optional[Path] = None,
    keep_tars: bool = False,
    workers: int = 8,
) -> Dict[str, int]:
    """Download Google Drive shards as needed and extract `needed_stems` into `<out_dir>/structures/<bucket>/<stem>.pdb`."""
    struct_root = out_dir / "structures"
    struct_root.mkdir(parents=True, exist_ok=True)
    for b in range(1000):
        (out_dir / "structures" / f"{b:03d}").mkdir(exist_ok=True)
    (out_dir / "structures" / "other").mkdir(exist_ok=True)

    if shards_dir is None:
        shards_dir = out_dir / "raw" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = struct_root / ".shard_catalog.json"
    catalog: Dict[str, List[str]] = {}
    if catalog_path.is_file():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    missing_stems = {s for s in needed_stems if not (out_dir / structure_rel_path(s)).is_file()}
    print(
        f"Structures requested: {len(needed_stems):,} | already on disk: {len(needed_stems) - len(missing_stems):,} | "
        f"to extract: {len(missing_stems):,}",
        flush=True,
    )
    if not missing_stems:
        return {"requested": len(needed_stems), "extracted": 0, "remaining_missing": 0}

    stem_splits = _split_map_from_index(out_dir)

    extracted_total = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for shard_name, fid, expected_bytes, shard_splits in SHARDS:
            if not missing_stems:
                break
            if stem_splits and all(s in stem_splits for s in missing_stems):
                if not any(stem_splits[s] in shard_splits for s in missing_stems):
                    print(f"Skipping {shard_name} (no remaining missing stems in splits {shard_splits})", flush=True)
                    continue

            if shard_name in catalog:
                shard_missing = [s for s in catalog[shard_name] if s in missing_stems]
                if not shard_missing:
                    print(f"Skipping {shard_name} (0 missing stems via shard catalog)", flush=True)
                    continue

            tar_path = shards_dir / shard_name
            _download_gdrive_shard(shard_name, fid, expected_bytes, tar_path)

            shard_members: List[str] = []
            tasks: List[Tuple[bytes, str]] = []
            extracted_here: List[str] = []

            with tarfile.open(tar_path, "r:") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    if not base.endswith(".cif.gz"):
                        continue
                    stem = base[: -len(".cif.gz")]
                    shard_members.append(stem)
                    if stem in missing_stems:
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        raw_bytes = fobj.read()
                        dest_path = str(out_dir / structure_rel_path(stem))
                        tasks.append((raw_bytes, dest_path))
                        extracted_here.append(stem)
                        if len(tasks) >= 2000:
                            list(pool.map(_convert_and_write_one, tasks, chunksize=64))
                            tasks.clear()

            if tasks:
                list(pool.map(_convert_and_write_one, tasks, chunksize=64))
                tasks.clear()

            catalog[shard_name] = shard_members
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f)

            for s in extracted_here:
                missing_stems.discard(s)
            extracted_total += len(extracted_here)
            print(
                f"Processed {shard_name}: {len(shard_members):,} entries, extracted {len(extracted_here):,} PDBs "
                f"({len(missing_stems):,} remaining)",
                flush=True,
            )

            if not keep_tars:
                tar_path.unlink(missing_ok=True)

    return {
        "requested": len(needed_stems),
        "extracted": extracted_total,
        "remaining_missing": len(missing_stems),
        "sample_missing": sorted(missing_stems)[:10],
    }


def main() -> None:
    default_out = Path(__file__).resolve().parent.parent / "data" / "mgnify"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=default_out, help="Target root directory (default: data/mgnify).")
    ap.add_argument("--tables_only", action="store_true", help="Only download raw CSV tables from Zenodo.")
    ap.add_argument("--structures_only", action="store_true", help="Only download and convert AF2 structure shards.")
    ap.add_argument(
        "--needed_stems",
        type=Path,
        default=None,
        help="Optional text file with one structure stem per line to extract.",
    )
    ap.add_argument(
        "--all_wt",
        action="store_true",
        help="Extract all 532,988 scaffold structures from mgnify_training_index.csv when --needed_stems is unset.",
    )
    ap.add_argument("--shards_dir", type=Path, default=None, help="Temporary directory for downloaded .tar shards.")
    ap.add_argument("--keep_tars", action="store_true", help="Keep .tar shards after extraction.")
    ap.add_argument("--workers", type=int, default=8, help="Worker processes for CIF-to-PDB conversion.")
    ap.add_argument(
        "--local_index_fallback",
        type=Path,
        default=None,
        help="Optional local copy of mgnify_training_index.csv to reuse if MD5 matches.",
    )
    args = ap.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.structures_only:
        download_tables(out_dir, local_index_fallback=args.local_index_fallback)

    if not args.tables_only:
        if args.needed_stems is not None:
            stems = {ln.strip() for ln in open(args.needed_stems, encoding="utf-8") if ln.strip()}
        else:
            stems, _ = infer_needed_stems_from_tables(out_dir, all_wt=args.all_wt)
        stats = download_structures(
            out_dir,
            stems,
            shards_dir=args.shards_dir,
            keep_tars=args.keep_tars,
            workers=args.workers,
        )
        print("Structure summary:", json.dumps(stats, indent=2), flush=True)
        if stats["remaining_missing"] > 0:
            raise RuntimeError(
                f"{stats['remaining_missing']} requested structures were not found in the 22 shards: "
                f"{stats['sample_missing']}"
            )


if __name__ == "__main__":
    main()
