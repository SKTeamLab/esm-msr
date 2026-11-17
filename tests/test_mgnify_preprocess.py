"""CPU-only unit tests for MGnify Stability Dataset download & preprocessing utilities."""
import csv
import gzip
import io
import json
from pathlib import Path
import sys

from Bio.PDB import MMCIFParser, PDBIO
import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROC_DIR = REPO_ROOT / "preprocessing"
if str(PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROC_DIR))

import download_mgnify  # noqa: E402
import preprocess_mgnify  # noqa: E402
from esm.utils.structure.protein_chain import ProteinChain  # noqa: E402
from esm_msr import inference, preprocess  # noqa: E402


def _make_synthetic_mmcif(n_res: int = 8) -> str:
    """Build a minimal AlphaFold-style single-chain mmCIF with N, CA, C, O backbone atoms."""
    resnames = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
    header = [
        "data_synthetic",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    rows = []
    aid = 1
    atom_offsets = [
        ("N", "N", 0.0, 0.0, 0.0),
        ("C", "CA", 1.458, 0.0, 0.0),
        ("C", "C", 2.009, 1.420, 0.0),
        ("O", "O", 1.251, 2.389, 0.0),
    ]
    for r_idx in range(1, n_res + 1):
        rn = resnames[(r_idx - 1) % len(resnames)]
        bfac = 82.5 + 0.5 * r_idx
        base_x = 3.8 * (r_idx - 1)
        for sym, aname, dx, dy, dz in atom_offsets:
            rows.append(
                f"ATOM {aid} {sym} {aname} . {rn} A 1 {r_idx} ? "
                f"{base_x + dx:.3f} {dy:.3f} {dz:.3f} 1.00 {bfac:.2f} {r_idx} A 1"
            )
            aid += 1
    rows.append("#")
    return "\n".join(header + rows) + "\n"


def test_fast_cif_to_pdb_matches_biopython_and_proteinchain(tmp_path):
    cif_text = _make_synthetic_mmcif(n_res=8)
    raw_gz = gzip.compress(cif_text.encode("utf-8"))

    fast_pdb_str = download_mgnify.cif_gz_bytes_to_pdb_str(raw_gz)
    fast_pdb = tmp_path / "fast.pdb"
    fast_pdb.write_text(fast_pdb_str, encoding="utf-8")

    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("syn", io.StringIO(cif_text))
    bio_buf = io.StringIO()
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(bio_buf)
    bio_pdb = tmp_path / "bio.pdb"
    bio_pdb.write_text(bio_buf.getvalue(), encoding="utf-8")

    pc_fast = ProteinChain.from_pdb(str(fast_pdb), "A", is_predicted=True)
    pc_bio = ProteinChain.from_pdb(str(bio_pdb), "A", is_predicted=True)

    assert pc_fast.sequence == pc_bio.sequence == "ACDEFGHI"
    c_fast, p_fast, idx_fast = pc_fast.to_structure_encoder_inputs()
    c_bio, p_bio, idx_bio = pc_bio.to_structure_encoder_inputs()
    assert torch.allclose(c_fast, c_bio, atol=1e-4, equal_nan=True)
    assert torch.allclose(p_fast, p_bio, atol=1e-4)
    assert torch.equal(idx_fast, idx_bio)

    res_map, aa_map = preprocess.get_pdb_to_seq_mapping(str(fast_pdb), "A", pc_fast.sequence)
    assert len(res_map) == len(pc_fast.sequence) == 8
    assert res_map == {i: str(i) for i in range(1, 9)}
    assert len(aa_map) == 8


def test_dms7_indel_guard_and_linker_trim():
    wt = "DPETKARAEGLAGQLQDAIAGLEKDLAAARAAGDAKKIAETEAALTARRAWLEQVLRSAK"
    # Fixed-length indel (linker-sliced to same length as WT, with shifted tail) must still be caught!
    assert preprocess_mgnify.is_dms7_indel("delG46", wt, wt) is True
    assert preprocess_mgnify.is_dms7_indel("insH12", wt, "A" * len(wt)) is True
    assert preprocess_mgnify.is_dms7_indel("Q26R_insA5", wt, "A" * len(wt)) is True
    assert preprocess_mgnify.is_dms7_indel("Q26R", wt, "A" * (len(wt) - 1)) is True
    assert preprocess_mgnify.is_dms7_indel("Q26R", wt, wt) is False
    assert preprocess_mgnify.is_dms7_indel("D63E_A38L", wt, wt) is False
    assert preprocess_mgnify.is_dms7_indel("mutant1_A35I", wt, wt) is False

    padded = "SAGGSAGGSA" + wt[:10] + "W" + wt[11:] + "SAGGSAGGSA"
    trimmed = preprocess_mgnify.trim_dms7_linkers(wt, padded)
    assert len(trimmed) == len(wt)
    assert preprocess_mgnify.diff_to_mut_type(wt, trimmed) == (f"{wt[10]}11W", 1)


def test_annotate_additivity_keeps_multis_without_singles():
    df = pd.DataFrame({
        "code_wt": ["s1", "s1", "s1", "s1", "s2", "s2"],
        "mut_type": ["A1G", "C2D", "A1G:C2D", "A1G:E3K", "F4Y:G5S", "F4Y:G5S:H6N"],
        "n_mut": [1, 1, 2, 2, 2, 3],
        "ddG_ML": [-0.5, -1.0, -1.2, -2.0, -0.8, -3.1],
    })
    out = preprocess_mgnify.annotate_additivity(df)
    assert len(out) == 6
    # s1 double with both singles present
    r_with = out.loc[out["mut_type"] == "A1G:C2D"].iloc[0]
    assert bool(r_with["has_all_singles"]) is True
    assert r_with["ddG_additive_ML"] == pytest.approx(-1.5)
    assert r_with["dddG_ML"] == pytest.approx(0.3)

    # s1 double missing E3K, plus s2 double & triple with zero singles: all must remain!
    for mt in ("A1G:E3K", "F4Y:G5S", "F4Y:G5S:H6N"):
        r = out.loc[out["mut_type"] == mt].iloc[0]
        assert bool(r["has_all_singles"]) is False
        assert np.isnan(r["ddG_additive_ML"])
        assert np.isnan(r["dddG_ML"])


def test_backbone_mutation_fallback_roundtrip():
    pdb_seq = "ACDEFGHIK"  # e.g., AF2 structure for rocklin_batch2_X_mutant1
    wt_seq = "MCDEFGHIK"   # true WT sequence (PDB has A1M diff? i.e., pdb=A, wt=M -> A1M)
    bb_mut = preprocess_mgnify.compute_backbone_mutation(pdb_seq, wt_seq)
    assert bb_mut == "A1M"

    coords = torch.zeros((1, len(pdb_seq) + 2, 37, 3), dtype=torch.float32)
    struct_toks = torch.zeros((1, len(pdb_seq) + 2), dtype=torch.long)
    corr_seq, _, _ = inference._handle_mutated_backbone(
        pdb_seq, coords, struct_toks, backbone_mutation=bb_mut, assert_wt=True
    )
    assert corr_seq == wt_seq


def test_pigeonhole_matcher_accuracy_and_preference():
    wt_a = "ACDEFGHIKLMNPQRSTVWY"
    wt_b = "WCDEFGHIKLMNPQRSTVWY"
    wt_dup = wt_a
    # scaf_2 shares sequence with scaf_1, but only scaf_2 has a structure
    wt_seqs = {"rocklin_batch2_1": wt_dup, "rocklin_batch2_2": wt_a, "rocklin_batch2_3": wt_b}
    struct_set = {"rocklin_batch2_2", "rocklin_batch2_3"}
    exact_map, q_idx = preprocess_mgnify.build_pigeonhole_index(
        wt_seqs, lambda s, _seq: s in struct_set
    )

    assert exact_map[wt_a] == ("rocklin_batch2_2", True)

    # HD=0 -> wt_identical
    assert preprocess_mgnify.match_pigeonhole(wt_a, exact_map, q_idx)[0] == "wt_identical"
    # HD=1 against wt_a (closer to wt_a than wt_b at pos 1 + 19)
    mut_hd1 = wt_a[:-1] + "A"
    st, sc, hd = preprocess_mgnify.match_pigeonhole(mut_hd1, exact_map, q_idx, max_hd=3)
    assert (st, sc, hd) == ("matched", "rocklin_batch2_2", 1)
    # HD=3 against wt_b (differs at pos 1, 2, 3 from wt_b and pos 1 is W->W vs A->W so closer to wt_b = 2 vs 3; let's make 3 diffs from wt_b and 4 from wt_a)
    mut_hd3_b = "W" + "AAA" + wt_b[4:]
    st3, sc3, hd3 = preprocess_mgnify.match_pigeonhole(mut_hd3_b, exact_map, q_idx, max_hd=3)
    assert (st3, sc3, hd3) == ("matched", "rocklin_batch2_3", 3)
    # HD=4 -> unmatched
    mut_hd4 = "KKKK" + wt_a[4:]
    assert preprocess_mgnify.match_pigeonhole(mut_hd4, exact_map, q_idx, max_hd=3)[0] == "unmatched"


def test_end_to_end_synthetic_preprocess_retains_all_edge_cases(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    wt1 = "ACDEFGHIKLMNPQRSTVWYACDE"
    wt2 = "MNPQRSTVWYACDEFGHIKLMNPQ"
    wt1_mutpdb = "VCDEFGHIKLMNPQRSTVWYACDE"  # A1V diff in PDB -> backbone_mutation = V1A

    # Write minimal index (wt1 uses mutant1 fallback PDB; wt2 has normal WT PDB)
    with open(raw / "mgnify_training_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "seq", "dG", "PDB_name"])
        w.writerow(["rocklin_batch2_10_mutant1", "test", wt1_mutpdb, "2.0", "rocklin_batch2_10"])
        w.writerow(["rocklin_batch2_20", "train", wt2, "1.5", "rocklin_batch2_20"])

    # Synthetic concat CSV exercising: WT, dms7 fixed-len indel, dms7 double w/o singles, dms5 hd=3 unindexed
    cols = [
        "name", "dna_seq", "log10_K50_t", "log10_K50_t_95CI_high", "log10_K50_t_95CI_low",
        "log10_K50_t_95CI", "fitting_error_t", "rmsd_t", "log10_K50unfolded_t", "deltaG_t",
        "deltaG_t_95CI_high", "deltaG_t_95CI_low", "deltaG_t_95CI", "log10_K50_c",
        "log10_K50_c_95CI_high", "log10_K50_c_95CI_low", "log10_K50_c_95CI", "fitting_error_c",
        "rmsd_c", "log10_K50unfolded_c", "deltaG_c", "deltaG_c_95CI_high", "deltaG_c_95CI_low",
        "deltaG_c_95CI", "log10_A0_K50", "deltaG", "deltaG_95CI_high", "deltaG_95CI_low",
        "deltaG_95CI", "lib", "aa_seq", "aa_seq_full", "mgnify", "dm_design", "train_name",
        "split", "train_seq", "train_dG_from_earlier_processing",
    ]

    def _row(name, lib, aa_seq, dg, ci="0.1", train_name="", split="test"):
        d = {c: "" for c in cols}
        d.update({
            "name": name, "lib": lib, "aa_seq": aa_seq, "aa_seq_full": aa_seq,
            "deltaG": str(dg), "deltaG_t": str(dg), "deltaG_c": str(dg),
            "deltaG_95CI": ci, "mgnify": "True", "dm_design": "False",
            "train_name": train_name, "split": split,
        })
        return [d[c] for c in cols]

    wt1_double = wt1[:2] + "K" + wt1[3:5] + "Y" + wt1[6:]  # D3K:G6Y
    wt1_fake_indel = wt1[:-2] + "WW"  # same len as wt1 but named _delE10 -> must go to indels!
    wt2_triple = "A" + "A" + "A" + wt2[3:]  # M1A:N2A:P3A via dms5 unindexed hd=3

    with open(raw / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(_row("rocklin_batch2_10", "dms4", wt1, 2.5))
        w.writerow(_row("rocklin_batch2_20", "dms4", wt2, 1.5, split="train"))
        w.writerow(_row("rocklin_batch2_10_D3K_G6Y", "dms7", wt1_double, 1.0))
        w.writerow(_row("rocklin_batch2_10_delE10", "dms7", wt1_fake_indel, 0.5))
        w.writerow(_row("rocklin_batch3_essential_1", "dms5", wt2_triple, -1.5, split=""))

    summary = preprocess_mgnify.preprocess_mgnify(
        tmp_path, do_download=False, skip_homolog_pairs=True, spot_check_pdbs=0
    )
    mapped = pd.read_csv(tmp_path / "mgnify_mapped.csv")
    multis = pd.read_csv(tmp_path / "mgnify_multis_only.csv")
    indels = pd.read_csv(tmp_path / "mgnify_indels.csv.gz")

    assert len(indels) == 1
    assert indels.iloc[0]["indel_spec"] == "delE10"
    assert len(mapped) == len(multis) == 2
    r_dbl = mapped.loc[mapped["code_wt"] == "rocklin_batch2_10"].iloc[0]
    assert r_dbl["mut_type"] == "D3K:G6Y"
    assert r_dbl["backbone_mutation"] == "V1A"
    assert r_dbl["ddG_ML"] == pytest.approx(-1.5)
    assert bool(r_dbl["has_all_singles"]) is False

    r_trp = mapped.loc[mapped["code_wt"] == "rocklin_batch2_20"].iloc[0]
    assert int(r_trp["n_mut"]) == 3
    assert int(r_trp["cens"]) == -1  # dG_wt=1.5 in range, dG_mut=-1.5 < -1
    assert summary["mgnify_mapped"]["multis_without_all_singles"] == 2
