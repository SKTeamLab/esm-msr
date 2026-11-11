#!/usr/bin/env python3
"""Preprocess Domainome 1.0 (Beltran et al., Nature 2025) into `domainome_mapped_2026.csv`.

Maps every missense mutation in Domainome 1.0 Supplementary Table 5 onto the 522 domain PDB
structures published by Beltran et al. (Nature 2025; Zenodo record 14356805, DOI
10.5281/zenodo.11043642; downloaded via `download_domainome_structures.py`), truncating and
sequentially renumbering each domain structure to 1-based construct coordinates
(`<domain_ID>_truncated_renumbered.pdb`) and validating wild-type/mutant sequence agreement.

Inputs:
  - `--raw_table`: Domainome 1.0 Supplementary Table 5 TSV (`SupplementaryTable5.txt`,
    `Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt`, or the `.txt.zip` archive
    from Zenodo record 14356805).
  - `--structures_dir`: Directory containing `<domain_ID>.pdb` files extracted by
    `preprocessing/download_domainome_structures.py` (default: `<repo>/data/domainome1/alphafold_structures`).

Outputs:
  - `<processed_structures_dir>/<domain_ID>_truncated_renumbered.pdb` (522 structures)
  - `--out`: Mapped benchmark table (default: `<repo>/data/domainome1/domainome_mapped_2026.csv`),
    consumed by `inference_scripts/esm_msr_testing.py` (`MSR_DOMAINOME_TABLE`),
    `inference_scripts/baselines/`, and `preprocessing/split_tsuboyama.py`.

Usage:
    python preprocessing/preprocess_domainome.py \\
        [--raw_table data/domainome1/SupplementaryTable5.txt] \\
        [--structures_dir data/domainome1/alphafold_structures] \\
        [--out data/domainome1/domainome_mapped_2026.csv]
"""
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select
from esm.utils.structure.protein_chain import ProteinChain
from tqdm import tqdm


def align_sequences(seq1: str, seq2: str, max_mismatches: int = 1) -> Tuple[str, str, int, int]:
    """Find the aligned region where `seq2` occurs as a contiguous window of `seq1` (<= max_mismatches)."""
    best_start = -1
    best_end = -1
    best_aligned_seq1 = ""
    best_aligned_seq2 = ""

    for start in range(len(seq1) - len(seq2) + 1):
        subseq1 = seq1[start : start + len(seq2)]
        mismatches = sum(1 for a, b in zip(subseq1, seq2) if a != b)
        if mismatches <= max_mismatches and (best_end - best_start) < len(seq2):
            best_start = start
            best_end = start + len(seq2)
            best_aligned_seq1 = subseq1
            best_aligned_seq2 = seq2

    aligned_seq1 = ("-" * best_start) + best_aligned_seq1 + ("-" * (len(seq1) - best_end))
    aligned_seq2 = (" " * best_start) + best_aligned_seq2 + (" " * (len(seq1) - best_end))
    return aligned_seq1, aligned_seq2, best_start, best_end


def assert_diff_by_one(str1: str, str2: str) -> None:
    """Assert two equal-length sequences differ at exactly one position (single-mutant check)."""
    if len(str1) != len(str2):
        raise AssertionError(f"Sequence lengths differ ({len(str1)} vs {len(str2)})")
    diff_count = sum(1 for a, b in zip(str1, str2) if a != b)
    if diff_count != 1:
        raise AssertionError(
            f"Sequences do not differ by exactly one residue (diffs={diff_count}):\n{str1}\n{str2}"
        )


class ResidueRangeSelect(Select):
    """Retain only residues whose numeric PDB residue ID falls in `[start_residue, end_residue]`."""

    def __init__(self, start_residue: int, end_residue: int, chain_id: Optional[str] = None):
        self.start_residue = start_residue
        self.end_residue = end_residue
        self.chain_id = chain_id

    def accept_residue(self, residue) -> bool:
        res_id = residue.get_id()[1]
        if self.chain_id is None or residue.get_full_id()[2] == self.chain_id:
            return self.start_residue <= res_id <= self.end_residue
        return False


def truncate_pdb(
    input_pdb: Path,
    output_pdb: Path,
    start_residue: int,
    end_residue: int,
    chain_id: Optional[str] = None,
) -> None:
    """Truncate a PDB file to retain only residues in `[start_residue, end_residue]`."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(input_pdb))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb), ResidueRangeSelect(start_residue, end_residue, chain_id))


def renumber_pdb(pdb_file: Path, output_file: Path) -> None:
    """Renumber residues in each chain sequentially starting at 1, detaching first to avoid ID collisions."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_file))

    for model in structure:
        for chain in model:
            residues = list(chain)
            for res in residues:
                chain.detach_child(res.id)
            for i, residue in enumerate(residues, start=1):
                residue.id = (residue.id[0], i, " ")
                chain.add(residue)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_file))


def load_raw_domainome_table(raw_table: Path) -> pd.DataFrame:
    """Load Domainome Supplementary Table 5 from a `.txt` TSV or Zenodo `.txt.zip` archive."""
    if not raw_table.exists():
        candidates = [
            raw_table.parent / "Extended_data_Table_5_aPCA_vs_variant_effect_predictors.txt",
            raw_table.parent / "Supplementary_Table_5_aPCA_vs_variant_effect_predictors.txt.zip",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Domainome raw table not found at {raw_table}. Download it with:\n"
                f"  python preprocessing/download_domainome_structures.py --download_table"
            )
        raw_table = found

    if raw_table.suffix == ".zip":
        with zipfile.ZipFile(raw_table) as zf:
            members = [
                m for m in zf.namelist()
                if m.endswith(".txt") and not Path(m).name.startswith("._") and "__MACOSX" not in m
            ]
            if not members:
                raise RuntimeError(f"No .txt file found inside {raw_table}")
            with zf.open(members[0]) as fh:
                return pd.read_csv(fh, sep="\t", low_memory=False)
    return pd.read_csv(raw_table, sep="\t", low_memory=False)


def apply_construct_boundary_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the two known 1-residue construct-offset corrections so domain sequences align to PDBs."""
    # P03034_PF01381_4: construct carries a leading residue before PDB residue 4
    m_p03034 = df["domain_ID"] == "P03034_PF01381_4"
    if m_p03034.any():
        df.loc[m_p03034, "uniprot_ID_mutation"] = df.loc[m_p03034, "uniprot_ID_mutation"].apply(
            lambda x: x.split("_")[0]
            + "_"
            + x.split("_")[-1][0]
            + str(int(x.split("_")[-1][1:-1]) - 1)
            + x.split("_")[-1][-1]
        )
        df.loc[m_p03034, "aa_seq"] = df.loc[m_p03034, "aa_seq"].str.slice(1)

    # Q13472_PF06839_896: PDB begins one residue upstream ('S') relative to construct sequence
    m_q13472 = df["domain_ID"] == "Q13472_PF06839_896"
    if m_q13472.any():
        df.loc[m_q13472, "uniprot_ID_mutation"] = df.loc[m_q13472, "uniprot_ID_mutation"].apply(
            lambda x: x.split("_")[0]
            + "_"
            + x.split("_")[-1][0]
            + str(int(x.split("_")[-1][1:-1]) + 1)
            + x.split("_")[-1][-1]
        )
        df.loc[m_q13472, "aa_seq"] = "S" + df.loc[m_q13472, "aa_seq"].str.slice(0, -1)

    return df


def preprocess_domainome(
    raw_table: Path,
    structures_dir: Path,
    processed_structures_dir: Path,
    out_csv: Path,
    max_domains: int = 0,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, list]:
    df = load_raw_domainome_table(raw_table)
    df = df.loc[~df["uniprot_ID_mutation"].str.endswith("*", na=False)].copy()
    df = df.loc[~df["uniprot_ID_mutation"].str.contains("NANANA", na=False)].copy()

    if max_domains and max_domains > 0:
        keep_doms = list(dict.fromkeys(df["domain_ID"]))[:max_domains]
        df = df.loc[df["domain_ID"].isin(keep_doms)].copy()

    df = apply_construct_boundary_fixes(df)

    processed_structures_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for col in ["pdb_file", "wt_seq", "wild_type", "mutation", "mut_type", "uniprot_seq"]:
        df[col] = pd.Series(index=df.index, dtype="object")
    for col in ["position", "offset_up"]:
        df[col] = pd.Series(index=df.index, dtype="float64")

    failed_assertions = []
    groups = list(df.groupby(["domain_ID", "uniprot_ID"]))
    iterator = groups if quiet else tqdm(groups, desc="Mapping Domainome domains")

    for (domain_id, _uniprot_id), group in iterator:
        offset = int(str(domain_id).split("_")[-1])
        subseq = str(group["aa_seq"].iloc[0])

        src_pdb = structures_dir / f"{domain_id}.pdb"
        if not src_pdb.exists():
            raise FileNotFoundError(
                f"Missing structure {src_pdb}. Run preprocessing/download_domainome_structures.py first."
            )

        wt_prot = ProteinChain.from_pdb(str(src_pdb))
        full_seq = wt_prot.sequence
        _, _, start_idx, end_idx = align_sequences(full_seq, subseq, max_mismatches=1)
        if start_idx < 0:
            raise RuntimeError(f"Could not align construct sequence for {domain_id} against {src_pdb}")

        start_res = start_idx + offset
        end_res = end_idx + (offset - 1)

        truncated_pdb = processed_structures_dir / f"{domain_id}_truncated_renumbered.pdb"
        truncate_pdb(src_pdb, truncated_pdb, start_res, end_res)
        renumber_pdb(truncated_pdb, truncated_pdb)
        try:
            trunc_prot = ProteinChain.from_pdb(str(truncated_pdb))
        except ValueError:
            # Structures already numbered 1..L rather than offset..offset+L-1 (e.g. O95714_PF11515_2555)
            shutil.copy2(src_pdb, truncated_pdb)
            renumber_pdb(truncated_pdb, truncated_pdb)
            trunc_prot = ProteinChain.from_pdb(str(truncated_pdb))

        trunc_seq = trunc_prot.sequence
        assert_diff_by_one(subseq, trunc_seq)

        dom_mask = df["domain_ID"] == domain_id
        df.loc[dom_mask, "pdb_file"] = str(truncated_pdb.resolve())

        mut_tokens = group["uniprot_ID_mutation"].str.split("_").str[-1]
        for idx, tok, aa_seq in zip(group.index, mut_tokens, group["aa_seq"]):
            wt = tok[0]
            try:
                pos_uniprot = int(tok[1:-1])
            except ValueError:
                continue
            pos0 = pos_uniprot - offset
            mut = tok[-1]

            if not (0 <= pos0 < len(trunc_seq)) or not (0 <= pos0 < len(aa_seq)):
                failed_assertions.append((domain_id, wt, pos0, mut))
                continue
            if trunc_seq[pos0] != wt or aa_seq[pos0] != mut:
                failed_assertions.append((domain_id, wt, pos0, mut))
                continue

            pos1 = pos0 + 1
            df.at[idx, "wt_seq"] = trunc_seq
            df.at[idx, "wild_type"] = wt
            df.at[idx, "position"] = float(pos1)
            df.at[idx, "mutation"] = mut
            df.at[idx, "mut_type"] = f"{wt}{pos1}{mut}"
            df.at[idx, "uniprot_seq"] = full_seq
            df.at[idx, "offset_up"] = float(-offset + 1)

    df.to_csv(out_csv)
    return df, failed_assertions


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_dom_dir = repo_root / "data" / "domainome1"
    default_structs = default_dom_dir / "alphafold_structures"

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw_table",
        type=Path,
        default=default_dom_dir / "SupplementaryTable5.txt",
        help="Path to Domainome 1.0 Supplementary Table 5 (.txt TSV or Zenodo .txt.zip).",
    )
    ap.add_argument(
        "--structures_dir",
        type=Path,
        default=default_structs,
        help=f"Directory containing raw <domain_ID>.pdb files (default: {default_structs}).",
    )
    ap.add_argument(
        "--processed_structures_dir",
        type=Path,
        default=None,
        help="Output directory for <domain_ID>_truncated_renumbered.pdb (default: --structures_dir).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_dom_dir / "domainome_mapped_2026.csv",
        help="Output CSV path (default: <repo>/data/domainome1/domainome_mapped_2026.csv).",
    )
    ap.add_argument(
        "--max_domains",
        type=int,
        default=0,
        help="Process only the first N domains (for dry-run/testing; 0 = all 522 domains).",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress progress bar.")
    args = ap.parse_args()

    structures_dir = args.structures_dir.resolve()
    proc_dir = (args.processed_structures_dir or structures_dir).resolve()
    out_csv = args.out.resolve()

    df, failed = preprocess_domainome(
        raw_table=args.raw_table.resolve(),
        structures_dir=structures_dir,
        processed_structures_dir=proc_dir,
        out_csv=out_csv,
        max_domains=args.max_domains,
        quiet=args.quiet,
    )
    n_mapped = int(df["position"].notna().sum())
    n_scored = int(df.dropna(subset=["position", "pdb_file", "scaled_fitness"]).shape[0])
    print(
        f"Wrote {out_csv}: {len(df)} total rows across {df['domain_ID'].nunique()} domains "
        f"({n_mapped} mapped to structure, {n_scored} with scaled_fitness, "
        f"{len(failed)} boundary-shifted rows left unmapped)."
    )


if __name__ == "__main__":
    main()
