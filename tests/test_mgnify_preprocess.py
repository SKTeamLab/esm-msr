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
import fold_rocklin_denovo  # noqa: E402
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
    assert r_dbl["cens"] == "default"

    r_trp = mapped.loc[mapped["code_wt"] == "rocklin_batch2_20"].iloc[0]
    assert int(r_trp["n_mut"]) == 3
    assert r_trp["cens"] == "mut_below_neg1"  # dG_wt=1.5 in range, dG_mut=-1.5 < -1
    assert summary["mgnify_mapped"]["multis_without_all_singles"] == 2


def test_annotate_censoring_descriptive_labels():
    assert preprocess_mgnify.annotate_censoring_and_ddg(1.2, 2.0)["cens"] == "default"
    assert preprocess_mgnify.annotate_censoring_and_ddg(-1.2, 2.0)["cens"] == "mut_below_neg1"
    assert preprocess_mgnify.annotate_censoring_and_ddg(5.3, 2.0)["cens"] == "mut_above_pos5"
    assert preprocess_mgnify.annotate_censoring_and_ddg(1.2, -1.5)["cens"] == "wt_out_of_range"
    assert preprocess_mgnify.annotate_censoring_and_ddg(1.2, 5.4)["cens"] == "wt_out_of_range"


def test_classify_source():
    assert preprocess_mgnify.classify_source("True", "False", "rocklin_batch2_1") == (True, "mgnify")
    assert preprocess_mgnify.classify_source("False", "True", "rocklin_batch2_9") == (False, "dm_design")
    assert preprocess_mgnify.classify_source("False", "False", "design_id_10") == (False, "rocklin_denovo")
    assert preprocess_mgnify.classify_source("False", "False", "rocklin_design_batch1_1") == (False, "rocklin_denovo")
    assert preprocess_mgnify.classify_source("False", "False", "doublevillin_S1A") == (False, "villin_control")
    assert preprocess_mgnify.classify_source("False", "False", "villintile_1") == (False, "villin_control")
    assert preprocess_mgnify.classify_source("False", "False", "6i0a_A") == (False, "pdb_uniprot_control")


def test_dms5_wt_incorporation_and_nonmgnify_sources(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    wt_mgn = "ACDEFGHIKLMNPQRSTVWYACDE"
    wt_des = "MNPQRSTVWYACDEFGHIKLMNPQ"
    wt_vil = "GMLSDEDFKAVFGMTRSAFANLPL"

    with open(raw / "mgnify_training_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "seq", "dG", "PDB_name"])
        w.writerow(["rocklin_batch2_100", "train", wt_mgn, "2.0", "rocklin_batch2_100"])

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

    def _row(name, lib, aa_seq, dg, mgn="True", dm="False", train_name="", split="train"):
        d = {c: "" for c in cols}
        d.update({
            "name": name, "lib": lib, "aa_seq": aa_seq, "aa_seq_full": aa_seq,
            "deltaG": str(dg), "deltaG_t": str(dg), "deltaG_c": str(dg),
            "deltaG_95CI": "0.1", "mgnify": mgn, "dm_design": dm,
            "train_name": train_name, "split": split,
        })
        return [d[c] for c in cols]

    with open(raw / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        # mgnify=True WT in dms4 (2.0) + indexed dms5 hd=0 (2.6) + unindexed dms5 hd=0 (2.9) -> mean dG_wt = 2.5 across 3 reps
        w.writerow(_row("rocklin_batch2_100", "dms4", wt_mgn, 2.0))
        w.writerow(_row("rocklin_batch3_essential_1", "dms5", wt_mgn, 2.6, train_name="rocklin_batch2_100"))
        w.writerow(_row("rocklin_batch3_essential_2", "dms5", wt_mgn, 2.9, train_name="", split=""))
        # Single mutant of rocklin_batch2_100
        w.writerow(_row("rocklin_batch2_100_A1K", "dms7", "K" + wt_mgn[1:], 1.0))
        # Non-MGnify rows across all 4 sources
        w.writerow(_row("rocklin_batch2_999", "dms4", "QQQ" + wt_des[3:], 1.2, mgn="False", dm="True", split=""))
        w.writerow(_row("design_id_5", "dms7", wt_des, 3.0, mgn="False", split=""))
        w.writerow(_row("design_id_5_M1A", "dms7", "A" + wt_des[1:], 2.0, mgn="False", split=""))
        w.writerow(_row("doublevillin_WT", "dms7", wt_vil, 4.0, mgn="False", split=""))
        w.writerow(_row("doublevillin_G1A", "dms7", "A" + wt_vil[1:], 3.2, mgn="False", split=""))
        w.writerow(_row("6i0a_A", "dms7", "WWWW" + wt_vil[4:], 1.1, mgn="False", split=""))

    preprocess_mgnify.preprocess_mgnify(
        tmp_path, do_download=False, skip_homolog_pairs=True, spot_check_pdbs=0
    )

    mapped = pd.read_csv(tmp_path / "mgnify_mapped.csv")
    all_col = pd.read_csv(tmp_path / "mgnify_all_collapsed.csv.gz")
    wts = pd.read_csv(tmp_path / "mgnify_wildtypes.csv.gz")

    assert len(mapped) == 1
    r_m = mapped.iloc[0]
    assert bool(r_m["mgnify"]) is True
    assert r_m["source"] == "mgnify"
    assert int(r_m["n_wt_replicates"]) == 3
    assert r_m["dG_wt"] == pytest.approx(2.5)
    assert r_m["ddG_ML"] == pytest.approx(-1.5)  # 1.0 - 2.5

    assert len(all_col) == 3
    assert set(all_col["source"]) == {"mgnify", "rocklin_denovo", "villin_control"}
    r_des = all_col.loc[all_col["source"] == "rocklin_denovo"].iloc[0]
    assert bool(r_des["mgnify"]) is False
    assert r_des["mut_type"] == "M1A"
    assert r_des["ddG_ML"] == pytest.approx(-1.0)
    assert pd.isna(r_des["pdb_file"]) or r_des["pdb_file"] == ""

    r_vil = all_col.loc[all_col["source"] == "villin_control"].iloc[0]
    assert bool(r_vil["mgnify"]) is False
    assert r_vil["ddG_ML"] == pytest.approx(-0.8)

    assert set(wts["source"]) == {"mgnify", "dm_design", "rocklin_denovo", "villin_control", "pdb_uniprot_control"}


def test_compose_trajectory_and_infer_design_id_core_bounds():
    assert fold_rocklin_denovo.compose_trajectory_mutations(
        "design_id_10", "design_id_10_R45L_K23G_G23H"
    ) == [(23, "K", "H"), (45, "R", "L")]
    assert fold_rocklin_denovo.compose_trajectory_mutations(
        "design_id_10", "design_id_10_K34N_N34K_G29W"
    ) == [(29, "G", "W")]

    # Case 1: left_pad = 8, L_core = 63, right_pad = 9
    core63 = ("AEKL" * 16)[:63]
    wt80_p = "SAGGSAGG" + core63 + "SAGGSAGGS"
    assert len(wt80_p) == 80
    assert fold_rocklin_denovo.infer_design_id_core_bounds(
        "design_id_1",
        wt80_p,
        ["design_id_1_A1G", "design_id_1_K3D_D3F"],
        ["A9G", "K11F"],
    ) == (8, 9, 63, core63)

    # Case 2: left_pad = 0, ends with 'S' with max_core_pos = 79 -> L_core = 79
    core79 = ("MDEK" * 20)[:79]
    wt80_79 = core79 + "S"
    assert fold_rocklin_denovo.infer_design_id_core_bounds(
        "design_id_2",
        wt80_79,
        ["design_id_2_M1A", f"design_id_2_{core79[78]}79K"],
        ["M1A", f"{core79[78]}79K"],
    ) == (0, 1, 79, core79)

    # Case 3: left_pad = 0, ends with 'S' with max_core_pos = 80 -> L_core = 80
    wt80_80 = core79 + "S"
    assert fold_rocklin_denovo.infer_design_id_core_bounds(
        "design_id_3",
        wt80_80,
        ["design_id_3_M1A", "design_id_3_S80K"],
        ["M1A", "S80K"],
    ) == (0, 0, 80, wt80_80)


def _write_synthetic_core_pdb(path: Path, seq: str, plddt: float = 91.5) -> None:
    aa1_to_3 = {v: k for k, v in preprocess_mgnify.AA3_TO_1.items()}
    lines = []
    aid = 1
    offsets = [("N", 0.0, 0.0, 0.0), ("CA", 1.458, 0.0, 0.0), ("C", 2.009, 1.420, 0.0), ("O", 1.251, 2.389, 0.0)]
    for r_idx, ch in enumerate(seq, 1):
        rn = aa1_to_3[ch]
        bx = 3.8 * (r_idx - 1)
        for aname, dx, dy, dz in offsets:
            lines.append(
                f"ATOM  {aid:5d} {aname:<4s} {rn:>3s} A{r_idx:4d}    "
                f"{bx + dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00{plddt:6.2f}           {aname[0]:>2s}"
            )
            aid += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_fold_rocklin_denovo_prepare_collect_apply_and_preprocess_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    # Build 80-AA MGnify control (left untouched) + padded 64-AA core design_id_10
    wt_mgn = ("ACDEFGHIKLMNPQRSTVWY" * 4)[:80]
    core64 = ("AEKLVDQIFGHMNPQRTWYC" * 4)[:64]
    wt_des = "SAGGSAGG" + core64 + "SAGGSAGG"
    assert len(wt_des) == 80

    with open(raw / "mgnify_training_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "seq", "dG", "PDB_name"])
        w.writerow(["rocklin_batch2_1", "train", wt_mgn, "2.0", "rocklin_batch2_1"])

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

    def _row(name, lib, aa_seq, dg, mgn="True"):
        d = {c: "" for c in cols}
        d.update({
            "name": name, "lib": lib, "aa_seq": aa_seq, "aa_seq_full": aa_seq,
            "deltaG": str(dg), "deltaG_t": str(dg), "deltaG_c": str(dg),
            "deltaG_95CI": "0.1", "mgnify": mgn, "dm_design": "False", "split": "train" if mgn == "True" else "",
        })
        return [d[c] for c in cols]

    # Mutations in core: core pos 1 (A->G), core pos 2 (E->D), double A1G:E2D, composed K3M_M3W on pos 3
    m_s1 = wt_des[:8] + "G" + wt_des[9:]
    m_s2 = wt_des[:9] + "D" + wt_des[10:]
    m_d = wt_des[:8] + "G" + "D" + wt_des[10:]
    m_c3 = wt_des[:10] + "W" + wt_des[11:]

    with open(raw / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(_row("rocklin_batch2_1", "dms4", wt_mgn, 2.0, mgn="True"))
        w.writerow(_row("rocklin_batch2_1_A1G", "dms7", "G" + wt_mgn[1:], 1.5, mgn="True"))
        w.writerow(_row("design_id_10", "dms7", wt_des, 1.0, mgn="False"))
        w.writerow(_row("design_id_10_A1G", "dms7", m_s1, 0.8, mgn="False"))
        w.writerow(_row("design_id_10_E2D", "dms7", m_s2, 0.7, mgn="False"))
        w.writerow(_row("design_id_10_A1G_E2D", "dms7", m_d, 0.6, mgn="False"))
        w.writerow(_row("design_id_10_K3M_M3W", "dms7", m_c3, 0.9, mgn="False"))

    preprocess_mgnify.preprocess_mgnify(tmp_path, do_download=False, skip_homolog_pairs=True, spot_check_pdbs=0)
    prep = fold_rocklin_denovo.cmd_prepare(tmp_path, n_shards=1)
    assert prep["n_scaffolds"] == 1
    assert prep["flank_mut_errors"] == 0
    assert prep["shifted_mut_errors"] == 0

    pred_dir = tmp_path / "af2_rocklin_denovo" / "raw_predictions"
    pred_dir.mkdir(parents=True)
    _write_synthetic_core_pdb(pred_dir / "design_id_10_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb", core64)
    (pred_dir / "design_id_10_scores_rank_001_alphafold2_ptm_model_1_seed_000.json").write_text(
        json.dumps({"plddt": [92.0] * 64, "ptm": 0.87}), encoding="utf-8"
    )

    col_stats = fold_rocklin_denovo.cmd_collect(tmp_path, spot_check_pdbs=1)
    assert col_stats["n_collected"] == 1
    assert col_stats["n_spot_checked"] == 1

    app_stats = fold_rocklin_denovo.cmd_apply(tmp_path)
    # Idempotency check: second apply should succeed identically
    app_stats_2 = fold_rocklin_denovo.cmd_apply(tmp_path)
    assert app_stats_2["mgnify_mapped_rows"] == app_stats["mgnify_mapped_rows"] == 5

    mapped = pd.read_csv(tmp_path / "mgnify_mapped.csv")
    assert len(mapped) == 5
    rd_m = mapped[mapped["source"] == "rocklin_denovo"].copy()
    assert len(rd_m) == 4
    assert (rd_m["wt_seq"] == core64).all()
    assert (rd_m["pdb_file"] == "structures_af2_denovo/010/design_id_10.pdb").all()
    dbl = rd_m[rd_m["n_mut"] == 2].iloc[0]
    assert dbl["mut_type"] == "A1G:E2D"
    assert bool(dbl["has_all_singles"]) is True
    assert dbl["dddG_ML"] == pytest.approx((-0.4) - ((-0.2) + (-0.3)))
    assert "K3W" in set(rd_m["mut_type"])
    # MGnify row untouched
    assert mapped[mapped["source"] == "mgnify"].iloc[0]["wt_seq"] == wt_mgn

    # Also test preprocess_mgnify auto-detecting existing manifest + PDBs on re-run
    preprocess_mgnify.preprocess_mgnify(tmp_path, do_download=False, skip_homolog_pairs=True, spot_check_pdbs=0)
    mapped_re = pd.read_csv(tmp_path / "mgnify_mapped.csv")
    rd_re = mapped_re[mapped_re["source"] == "rocklin_denovo"]
    assert len(rd_re) == 4
    assert (rd_re["wt_seq"] == core64).all()
    assert set(rd_re["mut_type"]) == {"A1G", "E2D", "A1G:E2D", "K3W"}


def test_incremental_partial_collect_and_prefill_apply(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    core64 = ("AEKLVDQIFGHMNPQRTWYC" * 4)[:64]
    core66 = ("MNPQRTWYCAEKLVDQIFGH" * 4)[:66]
    wt1 = "SAGGSAGG" + core64 + "SAGGSAGG"
    wt2 = "SAGGSAG" + core66 + "SAGGSAG"
    assert len(wt1) == len(wt2) == 80

    with open(raw / "mgnify_training_index.csv", "w", newline="") as f:
        csv.writer(f).writerow(["name", "split", "seq", "dG", "PDB_name"])

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

    def _row(name, aa_seq, dg):
        d = {c: "" for c in cols}
        d.update({
            "name": name, "lib": "dms7", "aa_seq": aa_seq, "aa_seq_full": aa_seq,
            "deltaG": str(dg), "deltaG_t": str(dg), "deltaG_c": str(dg),
            "deltaG_95CI": "0.1", "mgnify": "False", "dm_design": "False",
        })
        return [d[c] for c in cols]

    with open(raw / "230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(_row("design_id_10", wt1, 1.0))
        w.writerow(_row("design_id_10_A1G", wt1[:8] + "G" + wt1[9:], 0.5))
        w.writerow(_row("design_id_20", wt2, 1.2))
        w.writerow(_row("design_id_20_M1W", wt2[:7] + "W" + wt2[8:], 0.8))

    preprocess_mgnify.preprocess_mgnify(tmp_path, do_download=False, skip_homolog_pairs=True, spot_check_pdbs=0)
    fold_rocklin_denovo.cmd_prepare(tmp_path, n_shards=2)

    pred_dir = tmp_path / "af2_rocklin_denovo" / "raw_predictions"
    pred_dir.mkdir(parents=True)
    _write_synthetic_core_pdb(pred_dir / "design_id_10_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb", core64)
    (pred_dir / "design_id_10_scores_rank_001_alphafold2_ptm_model_1_seed_000.json").write_text(
        json.dumps({"plddt": [93.0] * 64, "ptm": 0.88}), encoding="utf-8"
    )

    c1 = fold_rocklin_denovo.cmd_collect(tmp_path, spot_check_pdbs=1, allow_partial=True)
    assert (c1["n_target"], c1["n_collected"], c1["n_newly_installed"], c1["n_missing"]) == (2, 1, 1, 1)

    dst10 = tmp_path / "structures_af2_denovo/010/design_id_10.pdb"
    inode10 = dst10.stat().st_ino

    app = fold_rocklin_denovo.cmd_apply(tmp_path, allow_missing_pdbs=True)
    assert app["mgnify_mapped_rows"] == 2
    assert app["rocklin_denovo_af2"]["n_structures_on_disk"] == 1
    mapped = pd.read_csv(tmp_path / "mgnify_mapped.csv")
    assert set(mapped["pdb_file"]) == {
        "structures_af2_denovo/010/design_id_10.pdb",
        "structures_af2_denovo/020/design_id_20.pdb",
    }

    # Add 2nd PDB incrementally; 1st PDB inode must remain untouched
    _write_synthetic_core_pdb(pred_dir / "design_id_20_relaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb", core66)
    c2 = fold_rocklin_denovo.cmd_collect(tmp_path, spot_check_pdbs=1, allow_partial=True)
    assert (c2["n_target"], c2["n_collected"], c2["n_newly_installed"], c2["n_missing"]) == (2, 2, 1, 0)
    assert dst10.stat().st_ino == inode10
