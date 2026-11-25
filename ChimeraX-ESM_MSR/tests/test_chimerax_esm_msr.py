"""Automated headless/offscreen verification suite for ChimeraX-ESM-MSR (v0.5.0).

Run everything (unit + headless PyQt6 + real ChimeraX --offscreen integration + isolated wheel CLI):
    env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \\
      /usr/lib/ucsf-chimerax/bin/python3.11 -m pytest \\
      ChimeraX-ESM_MSR/tests/test_chimerax_esm_msr.py -v

Or run the ChimeraX-session integration tests directly inside ChimeraX:
    env -u DISPLAY -u WAYLAND_DISPLAY QT_QPA_PLATFORM=offscreen \\
      chimerax --offscreen --exit --script \\
      ChimeraX-ESM_MSR/tests/test_chimerax_esm_msr.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np

# Ensure ChimeraX user site-packages (e.g. pandas 3.x) and bundle src are importable
_BUNDLE_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _BUNDLE_ROOT / "src"
_REPO_ROOT = _BUNDLE_ROOT.parent

for _ver in ("1.11", "1.10", "1.9"):
    _user_sp = Path.home() / ".local" / "share" / "ChimeraX" / _ver / "lib" / "python3.11" / "site-packages"
    if _user_sp.is_dir() and str(_user_sp) not in sys.path:
        sys.path.append(str(_user_sp))


def _ensure_synthetic_pkg() -> str:
    pkg_name = "_test_chimerax_esm_msr_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_SRC_DIR)]
        sys.modules[pkg_name] = pkg
    return pkg_name


def _load_submodule(name: str):
    pkg_name = _ensure_synthetic_pkg()
    full = f"{pkg_name}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _SRC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pure_modules():
    """Import pure-Python modules (parsing, fast_doubles, heatmap) without requiring ChimeraX session."""
    return _load_submodule("parsing"), _load_submodule("fast_doubles"), _load_submodule("heatmap")


def _load_bundle_modules():
    """Import src/parsing.py, src/commands.py, and src/tool.py as a synthetic package without installing."""
    parsing_mod = _load_submodule("parsing")
    _load_submodule("heatmap")
    commands_mod = _load_submodule("commands")
    tool_mod = _load_submodule("tool")
    return parsing_mod, tool_mod, commands_mod


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures
# ---------------------------------------------------------------------------

def _backbone_and_cb(atom_idx, resname, chain, resnum, icode, x0, y0, z0, bfac=20.0):
    lines = []
    coords = [
        ("N ", x0 + 0.000, y0 + 0.000, z0 + 0.000, "N"),
        ("CA", x0 + 1.458, y0 + 0.000, z0 + 0.000, "C"),
        ("C ", x0 + 2.009, y0 + 1.420, z0 + 0.000, "C"),
        ("O ", x0 + 1.246, y0 + 2.390, z0 + 0.000, "O"),
    ]
    if resname != "GLY":
        coords.append(("CB", x0 + 1.986, y0 - 0.760, z0 - 1.215, "C"))
    for aname, x, y, z, elem in coords:
        ic = icode if icode else " "
        lines.append(
            f"ATOM  {atom_idx:5d}  {aname:<3s} {resname:>3s} {chain:1s}"
            f"{resnum:4d}{ic:1s}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2s}\n"
        )
        atom_idx += 1
    return atom_idx, lines


def generate_synthetic_fixtures(out_dir: str | Path) -> dict[str, Path]:
    """Create deterministic multi-chain PDB and 6 schema-accurate CSVs (incl. two-stage fast doubles)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdb_path = out / "multi_chain_edge.pdb"
    atom_idx = 1
    lines = []
    # Chain A: negative resnum (-2), insertion code (15A), Glycine (16), close spatial spacing
    chain_a = [
        ("ALA", -2, ""),
        ("LEU", 14, ""),
        ("VAL", 15, ""),
        ("GLU", 15, "A"),
        ("GLY", 16, ""),
        ("TRP", 17, ""),
    ]
    for i, (rn, num, ic) in enumerate(chain_a):
        atom_idx, rlines = _backbone_and_cb(atom_idx, rn, "A", num, ic, i * 2.5, 0.0, 0.0)
        lines.extend(rlines)
    lines.append("TER\n")

    # Chain B: separate chain for multi-chain filtering & transparency isolation
    chain_b = [("ALA", 10, ""), ("VAL", 11, ""), ("LEU", 12, "")]
    for i, (rn, num, ic) in enumerate(chain_b):
        atom_idx, rlines = _backbone_and_cb(atom_idx, rn, "B", num, ic, i * 2.5, 10.0, 0.0)
        lines.extend(rlines)
    lines.append("TER\nEND\n")
    pdb_path.write_text("".join(lines), encoding="utf-8")

    # 1. Full singles+doubles matching current src/esm_msr/inference.py output schema
    full_csv = out / "singles_and_doubles.csv"
    full_csv.write_text(
        "pdb_file,code,chain,mut_type_renumbered,mut_type_pdb,"
        "wt_lora_pred_additive,wt_lora_dddg_pred,wt_lora_pred,"
        "mt_lora_pred_additive,mt_lora_dddg_pred,mt_lora_pred,"
        "combined_pred_additive,combined_dddg_pred,combined_pred,epi_pred,dist\n"
        f"{pdb_path},edge,A,A1V,A-2V,,0.0,1.20,,0.0,1.40,,0.0,1.30,0.0,\n"
        f"{pdb_path},edge,A,A1D,A-2D,,0.0,-0.40,,0.0,-0.60,,0.0,-0.50,0.0,\n"
        f"{pdb_path},edge,A,L2A,L14A,,0.0,-1.80,,0.0,-2.20,,0.0,-2.00,0.0,\n"
        f"{pdb_path},edge,A,L2I,L14I,,0.0,0.00,,0.0,0.00,,0.0,0.00,0.0,\n"
        f"{pdb_path},edge,A,V3I,V15I,,0.0,0.85,,0.0,0.95,,0.0,0.90,0.0,\n"
        f"{pdb_path},edge,A,E4K,E15AK,,0.0,1.50,,0.0,1.70,,0.0,1.60,0.0,\n"
        f"{pdb_path},edge,A,E4D,E15AD,,0.0,-0.90,,0.0,-1.10,,0.0,-1.00,0.0,\n"
        f"{pdb_path},edge,A,G5A,G16A,,0.0,0.70,,0.0,0.90,,0.0,0.80,0.0,\n"
        f"{pdb_path},edge,A,W6F,W17F,,0.0,-1.20,,0.0,-1.40,,0.0,-1.30,0.0,\n"
        # Doubles: epi_pred != combined_dddg_pred so metric selector can be verified
        f"{pdb_path},edge,A,A1V:E4K,A-2V:E15AK,2.70,0.0,2.70,3.10,1.60,4.30,2.90,0.60,3.50,0.80,7.5\n"
        f"{pdb_path},edge,A,L2A:G5A,L14A:G16A,-1.10,0.0,-1.10,-1.30,-1.80,-2.90,-1.20,-0.80,-2.00,-0.90,7.5\n"
        f"{pdb_path},edge,A,V3I:W6F,V15I:W17F,-0.35,0.0,-0.35,-0.45,1.20,0.85,-0.40,0.65,0.25,0.60,7.5\n"
        # Chain B rows
        f"{pdb_path},edge,B,A1V,A10V,,0.0,2.50,,0.0,2.50,,0.0,2.50,0.0,\n"
        f"{pdb_path},edge,B,V2L,V11L,,0.0,-2.50,,0.0,-2.50,,0.0,-2.50,0.0,\n"
        f"{pdb_path},edge,B,A1V:V2L,A10V:V11L,0.0,0.0,0.0,0.0,2.20,2.20,0.0,1.10,1.10,1.10,2.5\n",
        encoding="utf-8",
    )

    # 2. --skip_reverse output (only wt_lora_pred survives dropna)
    skip_rev_csv = out / "skip_reverse.csv"
    skip_rev_csv.write_text(
        "pdb_file,code,chain,mut_type_renumbered,mut_type_pdb,wt_lora_pred,combined_dddg_pred,epi_pred,mt_lora_dddg_pred\n"
        f"{pdb_path},edge,A,A1V,A-2V,1.25,0.0,0.0,0.0\n"
        f"{pdb_path},edge,A,E4K,E15AK,-0.75,0.0,0.0,0.0\n"
        f"{pdb_path},edge,A,G5A,G16A,0.00,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    # 3. Legacy CSV with combined_dddg_pred only
    legacy_csv = out / "legacy_dddg_only.csv"
    legacy_csv.write_text(
        "pdb_file,code,chain,mut_type_renumbered,mut_type_pdb,wt_lora_pred,mt_lora_pred,combined_pred,combined_dddg_pred\n"
        f"{pdb_path},edge,A,L2A,L14A,-1.0,-1.2,-1.1,0.0\n"
        f"{pdb_path},edge,A,E4K,E15AK,1.1,1.3,1.2,0.0\n"
        f"{pdb_path},edge,A,L2A:E4K,L14A:E15AK,0.1,1.5,0.8,0.7\n",
        encoding="utf-8",
    )

    # 4. epi_pred only
    epi_only_csv = out / "epi_only.csv"
    epi_only_csv.write_text(
        "pdb_file,code,chain,mut_type_renumbered,mut_type_pdb,wt_lora_pred,mt_lora_pred,combined_pred,epi_pred\n"
        f"{pdb_path},edge,A,L2A,L14A,-1.0,-1.2,-1.1,0.0\n"
        f"{pdb_path},edge,A,E4K,E15AK,1.1,1.3,1.2,0.0\n"
        f"{pdb_path},edge,A,L2A:E4K,L14A:E15AK,0.1,1.5,0.8,0.95\n",
        encoding="utf-8",
    )

    # 5. Edge cases: case-duplicate columns, NaN, Inf, exact 0.0, malformed mutations
    edge_csv = out / "edge_nan_zero_dupcase.csv"
    edge_csv.write_text(
        "pdb_file,code,CHAIN,chain,mut_type_pdb,wt_lora_pred,mt_lora_pred,combined_pred,epi_pred,combined_dddg_pred\n"
        f"{pdb_path},edge,X,A,A-2V,0.0,0.0,0.0,0.0,0.0\n"
        f"{pdb_path},edge,X,A,L14A,nan,nan,nan,0.0,0.0\n"
        f"{pdb_path},edge,X,A,V15I,inf,-inf,inf,0.0,0.0\n"
        f"{pdb_path},edge,X,A,E15AK,1.1,1.3,1.2,0.0,0.0\n"
        f"{pdb_path},edge,X,A,BADMUT,0.5,0.5,0.5,0.0,0.0\n"
        f"{pdb_path},edge,X,A,A-2V:E15AK,1.1,2.0,1.55,nan,inf\n"
        f"{pdb_path},edge,X,A,E15AK:G16A,1.1,2.0,1.55,0.85,0.75\n",
        encoding="utf-8",
    )

    # 6. Unified two-stage fast_doubles CSV (Stage-1 singles with wt_lora_pred only + Stage-2 re-ranked doubles)
    two_stage_csv = out / "two_stage_fast_doubles.csv"
    two_stage_csv.write_text(
        "pdb_file,code,chain,mut_type_renumbered,mut_type_pdb,wt_lora_pred,mt_lora_pred,"
        "combined_pred,epi_pred,combined_dddg_pred,mt_lora_dddg_pred,dist,"
        "stage1_additive_pred,additive_rank,full_rank,rank_change\n"
        f"{pdb_path},edge,A,A1V,A-2V,1.20,,,0.0,0.0,0.0,,,,,\n"
        f"{pdb_path},edge,A,L2I,L14I,0.40,,,0.0,0.0,0.0,,,,,\n"
        f"{pdb_path},edge,A,V3I,V15I,0.90,,,0.0,0.0,0.0,,,,,\n"
        f"{pdb_path},edge,A,E4K,E15AK,1.50,,,0.0,0.0,0.0,,,,,\n"
        f"{pdb_path},edge,A,G5A,G16A,0.75,,,0.0,0.0,0.0,,,,,\n"
        f"{pdb_path},edge,A,W6F,W17F,0.60,,,0.0,0.0,0.0,,,,,\n"
        # Doubles: #1 (A-2V:E15AK), #2 (A-2V:V15I shares -2!), #3 (V15I:G16A disjoint), #4 (L14I:W17F disjoint)
        f"{pdb_path},edge,A,A1V:E4K,A-2V:E15AK,2.70,3.10,2.90,0.20,0.20,0.40,12.5,2.70,1,1,0\n"
        f"{pdb_path},edge,A,A1V:V3I,A-2V:V15I,2.10,2.70,2.40,0.30,0.30,0.60,9.5,2.10,2,2,0\n"
        f"{pdb_path},edge,A,V3I:G5A,V15I:G16A,1.65,2.55,2.10,0.45,0.45,0.90,11.0,1.65,4,3,1\n"
        f"{pdb_path},edge,A,L2I:W6F,L14I:W17F,1.00,2.20,1.60,0.60,0.60,1.20,14.2,1.00,5,4,1\n",
        encoding="utf-8",
    )

    return {
        "pdb": pdb_path,
        "full": full_csv,
        "skip_reverse": skip_rev_csv,
        "legacy_dddg": legacy_csv,
        "epi_only": epi_only_csv,
        "edge": edge_csv,
        "two_stage": two_stage_csv,
    }


# ---------------------------------------------------------------------------
# Part 1: Pure unit tests for parsing, fast_doubles, heatmap, palettes & env discovery
# ---------------------------------------------------------------------------

def test_parse_mut_string_edge_cases():
    parsing, _, _ = _load_pure_modules()

    m = parsing.parse_mut_string("G289A")
    assert len(m) == 1
    assert m[0] == {"wt": "G", "pos": 289, "icode": "", "pos_key": (289, ""), "mut": "A"}

    m_neg = parsing.parse_mut_string("G-2A")
    assert len(m_neg) == 1
    assert m_neg[0]["pos"] == -2 and m_neg[0]["icode"] == "" and m_neg[0]["pos_key"] == (-2, "")

    m_ic = parsing.parse_mut_string("A15AC")
    assert len(m_ic) == 1
    assert m_ic[0] == {"wt": "A", "pos": 15, "icode": "A", "pos_key": (15, "A"), "mut": "C"}

    m_dbl = parsing.parse_mut_string("a-2v:e15ak")
    assert len(m_dbl) == 2
    assert m_dbl[0]["pos_key"] == (-2, "") and m_dbl[0]["mut"] == "V"
    assert m_dbl[1]["pos_key"] == (15, "A") and m_dbl[1]["mut"] == "K"

    assert parsing.parse_mut_string("A12C:BADTOKEN") == []
    assert parsing.parse_mut_string("A15ABC") == []
    assert parsing.parse_mut_string("") == []
    assert parsing.parse_mut_string(None) == []
    assert parsing.parse_mut_string(float("nan")) == []

    assert parsing.format_res_pos((-2, "")) == "-2"
    assert parsing.format_res_pos((15, "A")) == "15A"
    assert parsing.format_res_pos((15, "a")) == "15A"


def test_parse_csv_all_synthetic_schemas(tmp_path):
    parsing, _, _ = _load_pure_modules()
    fx = generate_synthetic_fixtures(tmp_path)

    # 1. Full singles+doubles, Chain A filtering
    res_a = parsing.parse_esm_msr_csv(str(fx["full"]), target_chain="A", mode="MT epistasis")
    assert res_a.ok, res_a.error
    assert res_a.chains_in_csv == ["A", "B"]
    assert res_a.effective_score_col == "combined_pred"
    assert res_a.effective_epi_col == "epi_pred"
    assert res_a.n_singles == 9
    assert res_a.n_positions == 6
    assert res_a.n_doubles == 3
    assert res_a.all_singles_scores[("A", (14, ""), "I")] == 0.0
    assert res_a.residue_scores_data[("A", (-2, ""))] == (1.30, "V")
    assert res_a.residue_scores_data[("A", (15, "A"))] == (1.60, "K")
    assert ("B", (10, "")) not in res_a.residue_scores_data
    row_a2 = res_a.epistasis_df[res_a.epistasis_df["pos1_key"] == (-2, "")].iloc[0]
    assert np.isclose(row_a2["dddg_pred"], 0.80)

    res_dddg = parsing.parse_esm_msr_csv(
        str(fx["full"]), requested_epi_col="combined_dddg_pred", target_chain="A", mode="MT epistasis"
    )
    assert res_dddg.ok and res_dddg.effective_epi_col == "combined_dddg_pred"
    row_a2_d = res_dddg.epistasis_df[res_dddg.epistasis_df["pos1_key"] == (-2, "")].iloc[0]
    assert np.isclose(row_a2_d["dddg_pred"], 0.60)

    res_z = parsing.parse_esm_msr_csv(str(fx["full"]), target_chain="Z", mode="Singles")
    assert not res_z.ok and not res_z.chain_matched and res_z.chains_in_csv == ["A", "B"]
    assert "A, B" in res_z.error

    # 2. --skip_reverse auto-fallback to wt_lora_pred + preserves 0.0
    res_sr = parsing.parse_esm_msr_csv(str(fx["skip_reverse"]), requested_score_col="combined_pred", target_chain="A")
    assert res_sr.ok, res_sr.error
    assert res_sr.effective_score_col == "wt_lora_pred"
    assert res_sr.auto_fallback_score_col == "wt_lora_pred"
    assert res_sr.available_score_cols == {"wt_lora_pred"}
    assert res_sr.all_singles_scores[("A", (16, ""), "A")] == 0.0

    # 3. Legacy CSV (combined_dddg_pred only)
    res_leg = parsing.parse_esm_msr_csv(str(fx["legacy_dddg"]), target_chain="A", mode="MT epistasis")
    assert res_leg.ok and res_leg.effective_epi_col == "combined_dddg_pred"
    assert np.isclose(res_leg.epistasis_df.iloc[0]["dddg_pred"], 0.7)

    # 4. epi_pred only
    res_epi = parsing.parse_esm_msr_csv(str(fx["epi_only"]), target_chain="A", mode="MT epistasis")
    assert res_epi.ok and res_epi.effective_epi_col == "epi_pred"
    assert np.isclose(res_epi.epistasis_df.iloc[0]["dddg_pred"], 0.95)

    # 5. Edge CSV
    res_edge = parsing.parse_esm_msr_csv(str(fx["edge"]), target_chain="A", mode="MT epistasis")
    assert res_edge.ok, res_edge.error
    assert any("case-duplicate" in w for w in res_edge.warnings)
    assert res_edge.all_singles_scores[("A", (-2, ""), "V")] == 0.0
    assert res_edge.n_singles == 2
    assert res_edge.n_doubles == 1

    # 6. Unified two-stage CSV: Stage-1 singles use wt_lora_pred while Stage-2 doubles use combined_pred!
    res_ts = parsing.parse_esm_msr_csv(str(fx["two_stage"]), requested_score_col="combined_pred", target_chain="A")
    assert res_ts.ok, res_ts.error
    assert res_ts.is_two_stage_csv is True
    assert res_ts.singles_score_col == "wt_lora_pred"
    assert res_ts.effective_score_col == "combined_pred"
    assert res_ts.n_singles == 6 and res_ts.n_doubles == 4

    # Disjoint pair selection skips A-2V:V15I (shares -2 with #1 A-2V:E15AK)
    disjoint = parsing.select_top_m_pairs(res_ts, top_m=5, sort_by="combined_pred", diverse_positions=True)
    assert [p.mut_str for p in disjoint] == ["A-2V:E15AK", "V15I:G16A", "L14I:W17F"]
    assert [p.color_hex for p in disjoint] == list(parsing.CATEGORICAL_PAIR_PALETTE[:3])

    # Non-disjoint includes A-2V:V15I and marks (-2, '') as shared_positions
    all_pairs = parsing.select_top_m_pairs(res_ts, top_m=3, sort_by="combined_pred", diverse_positions=False)
    assert [p.mut_str for p in all_pairs] == ["A-2V:E15AK", "A-2V:V15I", "V15I:G16A"]
    assert (-2, "") in all_pairs[0].shared_positions and (-2, "") in all_pairs[1].shared_positions


def test_path_python_env_and_tqdm_helpers(tmp_path):
    parsing, _, _ = _load_pure_modules()

    resolved = parsing.resolve_user_path("LoRA_models/esm-msr-small/lora_seed1.safetensors", str(tmp_path))
    assert resolved == str(tmp_path / "LoRA_models" / "esm-msr-small" / "lora_seed1.safetensors")

    fake_py = tmp_path / "my_python"
    fake_py.write_text("#!/bin/sh\n", encoding="utf-8")
    prog, args, err = parsing.resolve_python_command(str(fake_py))
    assert err is None and prog == str(fake_py) and args == ["-u"]

    venv_dir = tmp_path / "fake_venv"
    (venv_dir / "bin").mkdir(parents=True)
    venv_py = venv_dir / "bin" / "python"
    venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    prog, args, err = parsing.resolve_python_command(str(venv_dir))
    assert err is None and prog == str(venv_py) and args == ["-u"]

    prog, args, err = parsing.resolve_python_command("./does_not_exist_env", str(tmp_path))
    assert prog is None and err is not None and "does not exist" in err

    prog, args, err = parsing.resolve_python_command("")
    assert err is None and prog is not None and "-u" in args

    with patch("shutil.which", return_value="/usr/bin/conda"):
        prog, args, err = parsing.resolve_python_command("msr_venv")
        assert err is None and prog == "/usr/bin/conda"
        assert args == ["run", "--no-capture-output", "-n", "msr_venv", "python", "-u"]

    assert parsing.parse_tqdm_progress("Scoring:  42%|████      | 42/100 [00:05<00:07, 8.1it/s]") == 42
    assert parsing.parse_tqdm_progress("100%|██████████| 10/10 [00:01<00:00]") == 100
    assert parsing.parse_tqdm_progress("Loading checkpoint...") is None


def test_fast_doubles_exactness_filters_and_scaling(tmp_path):
    import pandas as pd

    _, fast_doubles, _ = _load_pure_modules()
    fx = generate_synthetic_fixtures(tmp_path)

    # 1. Exactness vs brute-force across negative resnums, icodes, Cys/Pro filters, min_seq_sep
    singles_rows = [
        ("p.pdb", "c", "A", "A1V", "A-2V", 1.50),
        ("p.pdb", "c", "A", "A1C", "A-2C", 1.45),
        ("p.pdb", "c", "A", "C2I", "C5I", 1.30),
        ("p.pdb", "c", "A", "P3A", "P6A", 1.25),
        ("p.pdb", "c", "A", "E4K", "E15AK", 1.20),
        ("p.pdb", "c", "A", "E4P", "E15AP", 1.15),
        ("p.pdb", "c", "A", "G5W", "G16W", 0.90),
        ("p.pdb", "c", "A", "W6Y", "W17Y", 0.70),
    ]
    sdf = pd.DataFrame(
        singles_rows,
        columns=["pdb_file", "code", "chain", "mut_type_renumbered", "mut_type_pdb", "wt_lora_pred"],
    )

    top_all = fast_doubles.select_top_n_additive_doubles(sdf, top_n=10)
    assert len(top_all) == 10
    assert top_all["additive_rank"].tolist() == list(range(1, 11))
    # Monotonically non-increasing and exact top pair is A-2V:C5I (1.50 + 1.30 = 2.80)
    sums = top_all["stage1_additive_pred"].to_numpy()
    assert np.all(np.diff(sums) <= 1e-12)
    assert top_all.iloc[0]["mut_type_pdb"] == "A-2V:C5I"
    assert np.isclose(top_all.iloc[0]["stage1_additive_pred"], 2.80)
    assert len(set(top_all["mut_type_pdb"])) == 10  # Zero duplicates

    # Cys/Pro filters exclude C5I, P6A, A-2C, E15AP
    top_filt = fast_doubles.select_top_n_additive_doubles(
        sdf, top_n=10, exclude_wt_cys=True, exclude_wt_pro=True, exclude_mut_cys=True, exclude_mut_pro=True
    )
    assert len(top_filt) == 6  # C(4, 2) = 6 valid pairs from {A-2V, E15AK, G16W, W17Y}
    assert top_filt.iloc[0]["mut_type_pdb"] == "A-2V:E15AK"
    assert all("C" not in m and "P" not in m for m in top_filt["mut_type_pdb"])

    # min_seq_sep excludes adjacent (16, 17)
    top_sep = fast_doubles.select_top_n_additive_doubles(
        sdf, top_n=10, exclude_wt_cys=True, exclude_wt_pro=True, exclude_mut_cys=True, exclude_mut_pro=True, min_seq_sep=2
    )
    assert "G16W:W17Y" not in set(top_sep["mut_type_pdb"])

    # Heavy-atom PDB distance filtering on multi_chain_edge.pdb
    allowed = fast_doubles.compute_allowed_pos_pairs_within_distance(str(fx["pdb"]), "A", 4.0)
    assert allowed is not None and len(allowed) > 0
    assert ((-2, ""), (17, "")) not in allowed  # Far apart (>10 A)

    # merge_stage1_and_stage2 computes full_rank and rank_change = additive_rank - full_rank
    sel3 = top_filt.head(3).copy()
    stage2 = pd.DataFrame({
        "pdb_file": ["tmp.pdb"] * 3,
        "code": ["c"] * 3,
        "chain": ["A"] * 3,
        "mut_type_pdb": sel3["mut_type_pdb"].tolist(),
        "wt_lora_pred": sel3["stage1_additive_pred"].tolist(),
        "mt_lora_pred": [2.0, 2.5, 4.0],
        "combined_pred": [2.1, 2.3, 3.2],  # Reverse order of Stage-1!
        "epi_pred": [-0.6, -0.1, +1.0],
        "dist": [12.0, 14.0, 16.0],
    })
    unified, reranked = fast_doubles.merge_stage1_and_stage2(sdf, sel3, stage2, original_pdb_file="orig.pdb")
    assert len(reranked) == 3
    assert reranked.iloc[0]["full_rank"] == 1
    assert reranked.iloc[0]["additive_rank"] == 3
    assert reranked.iloc[0]["rank_change"] == 2
    assert len(unified) == len(sdf) + 3
    assert (unified["pdb_file"] == "orig.pdb").all()

    # 2. Scaling test: 1,900 singles (~1.8M pairs) -> top_n=100,000 in well under 2 seconds!
    rng = np.random.default_rng(42)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    big_muts = [f"A{pos}{aa}" for pos in range(1, 101) for aa in aas[:19]]
    big_df = pd.DataFrame({
        "pdb_file": "big.pdb",
        "code": "big",
        "chain": "A",
        "mut_type_pdb": big_muts,
        "wt_lora_pred": rng.normal(0.0, 1.0, size=len(big_muts)),
    })
    t0 = time.perf_counter()
    res_100k = fast_doubles.select_top_n_additive_doubles(big_df, top_n=100_000)
    elapsed = time.perf_counter() - t0
    assert len(res_100k) == 100_000
    assert elapsed < 3.0, f"select_top_n_additive_doubles(100_000) took {elapsed:.2f}s"
    vals = res_100k["stage1_additive_pred"].to_numpy()
    assert np.all(np.diff(vals) <= 1e-12)
    assert res_100k["mut_type_pdb"].nunique() == 100_000


def test_heatmap_pair_card_palettes_and_env_discovery(tmp_path):
    from PIL import Image

    parsing, _, heatmap = _load_pure_modules()
    fx = generate_synthetic_fixtures(tmp_path)

    # 1. All 5 COLOR_PALETTES use zero-network ChimeraX specs (explicit hex/color stops)
    assert list(parsing.COLOR_PALETTES.keys()) == [
        "blue_white_red",
        "purple_white_green",
        "brown_white_teal",
        "viridis",
        "cividis",
    ]
    for key, spec in parsing.COLOR_PALETTES.items():
        assert ":" in spec["chimerax"], f"Palette {key} should use colon-delimited stops: {spec}"
        assert parsing.get_color_palette(key)["key"] == key
        assert parsing.get_color_palette(spec["mpl"])["key"] == key

    # 2. Heatmap export (PNG, PDF, SVG + custom vmin/vmax + unified 2-stage CSV)
    parsed = parsing.parse_esm_msr_csv(str(fx["two_stage"]), requested_score_col="combined_pred", target_chain="A")
    assert parsed.ok
    for ext, pal_key in [("png", "purple_white_green"), ("pdf", "viridis"), ("svg", "brown_white_teal")]:
        out_hm = tmp_path / f"hm_{pal_key}.{ext}"
        res_path = heatmap.export_mutational_heatmap(
            parsed, str(out_hm), palette_key=pal_key, vmin=-2.0, vmax=2.0, dpi=120
        )
        assert Path(res_path).is_file() and Path(res_path).stat().st_size > 1000
    arr = np.array(Image.open(tmp_path / "hm_purple_white_green.png"))
    assert arr.shape[0] > 200 and arr.shape[1] > 400
    assert len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)) > 40

    # 3. Pair legend card export
    pairs = parsing.select_top_m_pairs(parsed, top_m=5, sort_by="combined_pred", diverse_positions=True)
    out_card = tmp_path / "pair_card.png"
    card_path = heatmap.export_pair_legend_card(pairs, str(out_card), dpi=120)
    assert Path(card_path).is_file() and Path(card_path).stat().st_size > 1000
    carr = np.array(Image.open(out_card))
    assert len(np.unique(carr.reshape(-1, carr.shape[-1]), axis=0)) > 20

    # 4. discover_python_environments with synthetic conda/venv dirs (incl. editable .pth esm_msr)
    fake_home = tmp_path / "fake_home"
    env_plain = fake_home / "miniconda3" / "envs" / "plain_env"
    env_torch = fake_home / "miniconda3" / "envs" / "torch_env"
    env_msr = fake_home / "miniconda3" / "envs" / "msr_top_env"
    for ed in (env_plain, env_torch, env_msr):
        (ed / "bin").mkdir(parents=True)
        (ed / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
        (ed / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    (env_torch / "lib" / "python3.11" / "site-packages" / "torch").mkdir()
    (env_msr / "lib" / "python3.11" / "site-packages" / "torch").mkdir()
    (env_msr / "lib" / "python3.11" / "site-packages" / "esm").mkdir()
    (env_msr / "lib" / "python3.11" / "site-packages" / "__editable__.esm_msr-0.3.0.pth").write_text("/tmp\n")

    with patch.dict(os.environ, {"HOME": str(fake_home)}, clear=False):
        os.environ.pop("CONDA_PREFIX", None)
        os.environ.pop("VIRTUAL_ENV", None)
        found = parsing.discover_python_environments(base_repo_path=None)
    names = [e.name for e in found]
    assert names[:2] == ["msr_top_env", "torch_env"], f"Unexpected ranking: {names}"
    assert found[0].has_esm_msr and found[0].has_esm and found[0].has_torch


# ---------------------------------------------------------------------------
# Part 2 & 3: Real ChimeraX --offscreen + PyQt6 offscreen integration suite
# ---------------------------------------------------------------------------

def run_chimerax_integration_suite(session) -> None:
    """Runs inside a real ChimeraX --offscreen session with QT_QPA_PLATFORM=offscreen."""
    from PyQt6.QtWidgets import QApplication, QWidget, QScrollArea
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PIL import Image

    from chimerax.core.commands import run
    from chimerax.atomic import Structure, PseudobondGroup, selected_atoms

    parsing, tool_mod, commands_mod = _load_bundle_modules()

    app = QApplication.instance() or QApplication(["chimerax_esm_msr_test"])

    class _ShimMainToolWindow:
        def __init__(self, tool_instance, **kwargs):
            self.tool_instance = tool_instance
            self.ui_area = QWidget()
            self.ui_area.resize(520, 780)

        def manage(self, placement=None, **kwargs):
            self.ui_area.show()

        def cleanup(self):
            self.ui_area.close()

    orig_mtw = tool_mod.MainToolWindow
    tool_mod.MainToolWindow = _ShimMainToolWindow

    work_dir = Path(tempfile.mkdtemp(prefix="cx_esm_msr_inttest_"))
    xdg_dir = work_dir / "xdg"
    xdg_dir.mkdir()
    old_xdg = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(xdg_dir)

    try:
        fx = generate_synthetic_fixtures(work_dir / "fixtures")

        opened = run(session, f'open "{fx["pdb"]}"')
        wt_model = opened[0]
        assert isinstance(wt_model, Structure)

        # User-owned pseudobond group to verify ESM-MSR NEVER deletes user pbonds
        user_pbg = session.pb_manager.get_group("User Custom Bonds")
        if getattr(user_pbg, "id", None) is None:
            session.models.add([user_pbg])
        ca_atoms = [a for a in wt_model.atoms if a.name == "CA" and a.residue.chain_id == "B"]
        user_pbg.new_pseudobond(ca_atoms[0], ca_atoms[1])
        assert len(user_pbg.pseudobonds) == 1

        # Instantiate real ESM_MSR_Tool
        tool = tool_mod.ESM_MSR_Tool(session, "ESM_MSR")
        app.processEvents()

        # Verify QScrollArea wrapping, v0.4 + v0.5 widgets, and collapsible log console
        assert isinstance(tool.tabs.widget(0), QScrollArea)
        assert isinstance(tool.tabs.widget(1), QScrollArea)
        assert tool.model_dtype_combobox.count() == 3
        assert [tool.model_dtype_combobox.itemText(i) for i in range(3)] == ["float32", "bfloat16", "float16"]
        assert tool.mask_strategy_combobox.itemText(0) == "Checkpoint default"
        assert tool.display_mode_combo.findText("Top-M Ranked Pairs") >= 0
        assert tool.palette_combobox.count() == 5

        # Collapsible log console: collapsed by default, toggles cleanly, appends & clears
        assert not tool.toggle_log_button.isChecked()
        assert not tool.log_console.isVisible()
        tool.toggle_log_button.setChecked(True)
        app.processEvents()
        assert tool.log_console.isVisible()
        assert "Hide" in tool.toggle_log_button.text()
        tool._append_log("[Stage 1/2] test log line")
        assert "[Stage 1/2] test log line" in tool.log_console.toPlainText()
        tool.clear_log_button.click()
        assert tool.log_console.toPlainText() == ""
        tool.toggle_log_button.setChecked(False)

        # Bi-directional sync between Tab 1 fast_top_m_spinbox and Tab 2 top_m_pairs_spinbox
        tool.fast_top_m_spinbox.setValue(7)
        assert tool.top_m_pairs_spinbox.value() == 7
        tool.top_m_pairs_spinbox.setValue(4)
        assert tool.fast_top_m_spinbox.value() == 4

        # Auto-Detect Python env button
        tool._handle_auto_detect_python_env()
        app.processEvents()
        if tool._discovered_envs:
            assert tool.detected_env_combo.count() == len(tool._discovered_envs)
            assert tool.python_env_edit.text().strip() == tool._discovered_envs[0].python_path

        # _grab_selection scopes to target model & chain and preserves negative resnum + icode
        run(session, f"select #{wt_model.id_string}/A:-2,15A | #{wt_model.id_string}/B:10")
        QTest.mouseClick(tool.grab_sel_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool.selected_residues_edit.text() == "-2,15A", f"Got: {tool.selected_residues_edit.text()!r}"

        # Pre-flight validation
        tool.base_repo_path_edit.setText("")
        QTest.mouseClick(tool.run_prediction_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool._temp_dir_to_cleanup is None
        assert "Error:" in tool.status_label.text()

        # ---- Subprocess argv verification (Modes 1, zero-shot, and Mode 4 Fast Top-N Doubles) ----
        started_calls = []

        class _FakeProc:
            def __init__(self):
                self._state = tool_mod._PROC_NOT_RUNNING
                self._wd = ""
                self._env = None
                self.readyReadStandardOutput = type("_Sig", (), {"connect": lambda self, fn: None})()
                self.errorOccurred = type("_Sig", (), {"connect": lambda self, fn: None})()
                self.finished = type("_Sig", (), {"connect": lambda self, fn: None})()

            def setProcessChannelMode(self, mode):
                self._mode = mode

            def setWorkingDirectory(self, wd):
                self._wd = wd

            def setProcessEnvironment(self, env):
                self._env = env

            def state(self):
                return self._state

            def processId(self):
                return 0  # Test P0 #1 pid=0 guard

            def kill(self):
                self._state = tool_mod._PROC_NOT_RUNNING

            def waitForFinished(self, _ms):
                return True

            def start(self, program, args):
                self._state = 2
                started_calls.append((program, list(args), self._wd, self._env.value("PYTHONPATH", "")))

            def deleteLater(self):
                pass

        out_csv_target = work_dir / "pred_out.csv"
        tool.base_repo_path_edit.setText(str(_REPO_ROOT))
        tool.python_env_edit.setText(sys.executable)
        tool.script_output_csv_path_edit.setText(str(out_csv_target))
        tool.checkpoint_path_edit.setText("LoRA_models/esm-msr-small/lora_seed1.safetensors")
        tool.config_file_edit.setText("")
        tool.model_dtype_combobox.setCurrentText("bfloat16")
        tool.sigma_spinbox.setValue(0.5)
        tool.radio_full.setChecked(True)
        tool.mode_combobox.setCurrentText("singles+doubles")

        with patch.object(tool_mod, "QProcess", _FakeProc):
            tool._initiate_run_prediction_script()
            assert len(started_calls) == 1
            prog, args, wd, pypath = started_calls[-1]
            assert prog == os.path.normpath(os.path.abspath(sys.executable))
            assert args[0] == "-u"
            assert "--model_dtype" in args and args[args.index("--model_dtype") + 1] == "bfloat16"
            assert "--checkpoint_path" in args
            assert "--mode" in args and args[args.index("--mode") + 1] == "singles+doubles"
            assert "--mask_strategy" not in args
            assert os.path.isdir(wd) and wd.startswith(tempfile.gettempdir())
            assert str(_REPO_ROOT / "src") in pypath

            with patch.object(tool_mod.subprocess, "run") as mock_sub_run:
                tool._stop_prediction_script(silent=False)
                mock_sub_run.assert_not_called()

        # Mode 4 (Fast Top-N Doubles) argv with top_n=1_000_000, Cys/Pro filters, and distance threshold
        tool.radio_fast_doubles.setChecked(True)
        tool.fast_top_n_spinbox.setValue(1_000_000)
        tool.fast_filter_cys_pro_checkbox.setChecked(True)
        tool.fast_enable_dist_checkbox.setChecked(True)
        tool.fast_dist_spinbox.setValue(12.5)
        with patch.object(tool_mod, "QProcess", _FakeProc):
            tool._initiate_run_prediction_script()
            assert len(started_calls) == 2
            _prog4, args4, _wd4, _py4 = started_calls[-1]
            assert args4[0] == "-u"
            assert args4[1].endswith("fast_doubles.py")
            assert "--top_n" in args4 and args4[args4.index("--top_n") + 1] == "1000000"
            assert "--exclude_wt_cys" in args4 and "--exclude_mut_pro" in args4
            assert "--distance_threshold" in args4 and args4[args4.index("--distance_threshold") + 1] == "12.5"
            assert "--repo_dir" in args4
            tool._stop_prediction_script(silent=True)

        # Reset Mode 4 filters and test zero-shot mode (empty checkpoint + hparams.yaml)
        tool.fast_filter_cys_pro_checkbox.setChecked(False)
        tool.fast_enable_dist_checkbox.setChecked(False)
        tool.radio_full.setChecked(True)
        tool.checkpoint_path_edit.setText("")
        tool.config_file_edit.setText("LoRA_models/esm-msr-small/hparams.yaml")
        tool.sigma_spinbox.setValue(0.0)
        with patch.object(tool_mod, "QProcess", _FakeProc):
            tool._initiate_run_prediction_script()
            assert len(started_calls) == 3
            _prog, args2, _wd2, _pypath2 = started_calls[-1]
            assert "--checkpoint_path" not in args2
            assert "--hparams_path" in args2
            tool._stop_prediction_script(silent=True)

        # ---- Load synthetic CSVs & exercise Singles, Epistasis, Palettes, Top-M Pairs, Table, Exports ----
        # (a) --skip_reverse CSV auto-falls back cleanly
        tool.display_mode_combo.setCurrentText("MT epistasis")
        assert tool._load_csv_file(str(fx["skip_reverse"]))
        assert tool.display_mode_combo.currentText() == "Singles"
        assert tool.radio_wt_lora.isChecked()
        assert not tool.radio_dual_view.isEnabled()

        # (b) Edge CSV
        assert tool._load_csv_file(str(fx["edge"]))
        assert tool.all_singles_scores[("A", (-2, ""), "V")] == 0.0

        # (c) Full CSV -> Singles with PRGn palette + custom range + backbone + nearby (preserves user selection)
        tool.display_mode_combo.setCurrentText("Singles")
        pidx = tool.palette_combobox.findData("purple_white_green")
        assert pidx >= 0
        tool.palette_combobox.setCurrentIndex(pidx)
        tool.auto_range_checkbox.setChecked(False)
        tool.range_min_spinbox.setValue(-2.5)
        tool.range_max_spinbox.setValue(2.5)
        tool.color_backbone_checkbox.setChecked(True)
        tool.show_nearby_checkbox.setChecked(True)
        tool.nearby_distance_spinbox.setValue(5.0)
        run(session, f"select #{wt_model.id_string}/B:12")
        before_sel_count = len(selected_atoms(session))
        assert before_sel_count > 0

        assert tool._load_csv_file(str(fx["full"]))
        assert len(selected_atoms(session)) == before_sel_count
        assert len(tool.mutated_model_id_strings) == 1
        assert tool.viz_model_combobox.count() == 1
        clone = next(m for m in session.models.list(type=Structure) if m.id_string == tool.mutated_model_id_strings[0])
        clone_map = {(r.chain_id, r.number, r.insertion_code.strip()): r.name for r in clone.residues}
        assert clone_map[("A", -2, "")] == "VAL"
        assert clone_map[("A", 15, "A")] == "LYS"

        # Verify programmatic heatmap & pair card exports from GUI tool
        hm_out = work_dir / "gui_export_hm.png"
        card_out = work_dir / "gui_export_card.png"
        assert Path(tool.export_heatmap(str(hm_out))).is_file()
        assert Path(tool.export_pair_card(str(card_out))).is_file()
        tool.auto_range_checkbox.setChecked(True)

        # (d) Top-M Ranked Pairs on unified two-stage CSV!
        tool.tabs.setCurrentIndex(1)
        tool.display_mode_combo.setCurrentText("Top-M Ranked Pairs")
        tool.top_m_pairs_spinbox.setValue(5)
        tool.top_m_disjoint_checkbox.setChecked(True)
        app.processEvents()
        assert tool.top_m_group.isVisible()
        assert tool._load_csv_file(str(fx["two_stage"]))
        app.processEvents()

        # 3 disjoint pairs rendered: A-2V:E15AK, V15I:G16A, L14I:W17F
        assert len(tool.current_ranked_pairs) == 3
        assert len(tool.mutated_model_id_strings) == 3
        assert tool.viz_model_combobox.count() == 1  # Clones hidden from combobox!

        pbgs_topm = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Ranked Pairs" in pbgs_topm
        assert len(pbgs_topm["ESM-MSR Ranked Pairs"].pseudobonds) == 3
        assert "User Custom Bonds" in pbgs_topm and len(pbgs_topm["User Custom Bonds"].pseudobonds) == 1

        leg_models = [m for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")]
        assert len(leg_models) == 4  # hdr + 3 pairs

        # Ranked Results table checks: row 0 is #1 (ascending sort!), filter works, row click selects residues
        assert tool.results_table.rowCount() == 3
        assert tool.results_table.item(0, 0).text() == "#1", f"Expected #1 first, got {tool.results_table.item(0, 0).text()!r}"
        tool._on_results_row_clicked(0, 0)
        app.processEvents()
        assert tool.isolate_pair_button.isEnabled()
        sel_after_click = selected_atoms(session)
        assert sel_after_click is not None and len(sel_after_click) > 0

        # Filter table to "G16A" -> 1 row; clear -> 3 rows; switch View -> All Singles -> 6 rows
        tool.results_filter_edit.setText("G16A")
        app.processEvents()
        assert tool.results_table.rowCount() == 1
        tool.results_filter_edit.setText("")
        tool.results_view_combo.setCurrentText("All Singles")
        app.processEvents()
        assert tool.results_table.rowCount() == 6
        tool.results_view_combo.setCurrentText("Auto")
        app.processEvents()
        assert tool.results_table.rowCount() == 3

        # Isolate Pair #1 -> hides other pair clones & legend rows; Show All -> restores all
        assert tool.isolate_pair_rank(1)
        leg_vis = {m.name: m.display for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")}
        assert leg_vis["esm_msr_leg_hdr"] is True and leg_vis["esm_msr_leg_1"] is True
        assert leg_vis["esm_msr_leg_2"] is False and leg_vis["esm_msr_leg_3"] is False
        assert tool.show_all_pairs()
        leg_vis_all = {m.name: m.display for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")}
        assert all(leg_vis_all.values())

        # (e) MT epistasis & WT epistasis modes (using Viridis palette)
        tool.display_mode_combo.setCurrentText("MT epistasis")
        tool.show_nearby_checkbox.setChecked(False)
        tool.pos_threshold_spinbox.setValue(0.0)
        tool.neg_threshold_spinbox.setValue(0.0)
        assert tool._load_csv_file(str(fx["full"]))
        pbgs = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" in pbgs and len(pbgs["ESM-MSR Epistasis"].pseudobonds) >= 2
        assert "ESM-MSR Ranked Pairs" not in pbgs
        assert "User Custom Bonds" in pbgs and len(pbgs["User Custom Bonds"].pseudobonds) == 1

        tool.display_mode_combo.setCurrentText("WT epistasis")
        tool.mut_color_edit.setText("#00aa88")
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        pbgs_wt = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" in pbgs_wt and len(pbgs_wt["ESM-MSR Epistasis"].pseudobonds) == 1
        tool.mut_color_edit.setText("")

        # (f) Chain mismatch retains loaded_csv_path
        tool.display_mode_combo.setCurrentText("Singles")
        tool.viz_chain_id_combobox.setCurrentText("Z")
        assert not tool._load_csv_file(str(fx["full"]))
        assert tool.loaded_csv_path == str(fx["full"])
        assert tool.update_viz_button.isEnabled()
        tool.viz_chain_id_combobox.setCurrentText("B")
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert "Chain B" in tool.csv_label.text()

        # (g) Reset / Clear Viz cleans clones, ESM-MSR pbonds, and 2D legends while keeping user pbonds!
        tool.viz_chain_id_combobox.setCurrentText("A")
        tool.display_mode_combo.setCurrentText("Top-M Ranked Pairs")
        assert tool._load_csv_file(str(fx["two_stage"]))
        QTest.mouseClick(tool.reset_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool.mutated_model_id_strings == []
        pbgs_after_reset = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" not in pbgs_after_reset
        assert "ESM-MSR Ranked Pairs" not in pbgs_after_reset
        assert "User Custom Bonds" in pbgs_after_reset and len(pbgs_after_reset["User Custom Bonds"].pseudobonds) == 1
        assert not [m for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")]

        # (h) Lifecycle delete() cleans ESM-MSR legends & pbonds and unregisters triggers
        assert tool._load_csv_file(str(fx["two_stage"]))
        tool.delete()
        assert tool._closing is True
        assert tool._models_added_handler is None
        assert tool._models_removed_handler is None
        assert not [m for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")]
        pbgs_after_del = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Ranked Pairs" not in pbgs_after_del
        assert "User Custom Bonds" in pbgs_after_del

        # (i) Headless CLI handler functions (all 5 non-GUI handlers: load, visualize, pairs, heatmap, reset)
        commands_mod.cmd_esm_msr_reset(session)
        assert commands_mod.cmd_esm_msr_load(session, str(fx["two_stage"]), model=[wt_model], chain="A", mode="Singles")
        assert commands_mod.cmd_esm_msr_visualize(
            session, mode="Singles", palette="purple_white_green", minScore=-2.0, maxScore=2.0
        )
        cli_pairs = commands_mod.cmd_esm_msr_pairs(
            session, str(fx["two_stage"]), model=[wt_model], chain="A", topM=3,
            sortBy="combined_pred", diverse=True, connectors="dashed", labels=True, legend=True, isolate=1,
        )
        assert len(cli_pairs) == 3
        cli_hm = work_dir / "cli_hm.png"
        assert Path(
            commands_mod.cmd_esm_msr_heatmap(session, str(fx["two_stage"]), str(cli_hm), chain="A", palette="cividis")
        ).is_file()
        commands_mod.cmd_esm_msr_reset(session)
        assert not [m for m in session.models.list() if str(getattr(m, "name", "")).startswith("esm_msr_leg_")]

        # (j) Real PDB + CSV smoke tests (if available in parent workspace)
        real_root = _REPO_ROOT.parent
        csn4_pdb = real_root / "ProteinGym" / "ProteinGym_AF2_structures" / "CSN4_MOUSE.pdb"
        csn4_csv = real_root / "final_pgym_comb_sigma0.5" / "FINAL_p0" / "s1" / "CSN4_MOUSE_Tsuboyama_2023_1UFM.csv"
        if csn4_pdb.is_file() and csn4_csv.is_file():
            run(session, "close all")
            m_csn4 = run(session, f'open "{csn4_pdb}"')[0]
            assert commands_mod.cmd_esm_msr_load(session, str(csn4_csv), model=[m_csn4], chain="A", mode="Singles")
            render_csn4 = work_dir / "csn4_singles.png"
            run(session, f'view; save "{render_csn4}" width 320 height 240')
            arr = np.array(Image.open(render_csn4))
            assert len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)) > 50

        print("CHIMERAX_INTEGRATION_SUITE_PASSED")
    finally:
        tool_mod.MainToolWindow = orig_mtw
        if old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = old_xdg
        shutil.rmtree(work_dir, ignore_errors=True)


def test_chimerax_offscreen_end_to_end():
    """Invoke ChimeraX --offscreen headlessly to run the full ChimeraX+PyQt6 integration suite."""
    chimerax_bin = shutil.which("chimerax")
    assert chimerax_bin is not None, "chimerax executable not found on PATH"

    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run(
        [chimerax_bin, "--offscreen", "--exit", "--script", str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"ChimeraX exited with {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "CHIMERAX_INTEGRATION_SUITE_PASSED" in proc.stdout, f"Missing PASS marker\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_isolated_wheel_install_and_native_cli_commands(tmp_path):
    """Build & install v0.5.0 wheel into an isolated HOME and run all native `esm_msr` CLI commands."""
    chimerax_bin = shutil.which("chimerax")
    assert chimerax_bin is not None, "chimerax executable not found on PATH"

    fx = generate_synthetic_fixtures(tmp_path / "fx")
    bundle_copy = tmp_path / "bundle_src"
    shutil.copytree(_BUNDLE_ROOT, bundle_copy, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))

    iso_home = tmp_path / "iso_home"
    iso_home.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(iso_home)
    env["XDG_CONFIG_HOME"] = str(iso_home / ".config")
    env["XDG_DATA_HOME"] = str(iso_home / ".local" / "share")
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env["QT_QPA_PLATFORM"] = "offscreen"

    # 1. Build + install wheel cleanly into isolated HOME
    p_inst = subprocess.run(
        [
            chimerax_bin, "--nogui", "--exit", "--cmd",
            f'devel install "{bundle_copy}" user true noDeps true',
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p_inst.returncode == 0, f"devel install failed:\n{p_inst.stdout}\n{p_inst.stderr}"
    wheels = list((bundle_copy / "dist").glob("chimerax_esm_msr-0.5.0-*.whl"))
    assert len(wheels) == 1, f"Expected v0.5.0 wheel, got {wheels}"

    # Symlink ONLY pandas (+ dist-info) from host ChimeraX user site-packages into iso_home
    # (never touches host ~/.local/share/ChimeraX and never loads host chimerax bundles!)
    for iso_sp in (iso_home / ".local" / "share" / "ChimeraX").glob("*/lib/python*/site-packages"):
        cx_ver = iso_sp.parents[2].name
        host_sp = Path.home() / ".local" / "share" / "ChimeraX" / cx_ver / "lib" / iso_sp.parent.name / "site-packages"
        if host_sp.is_dir():
            for pat in ("pandas", "pandas-*.dist-info"):
                for src_item in host_sp.glob(pat):
                    dst_item = iso_sp / src_item.name
                    if not dst_item.exists():
                        os.symlink(src_item, dst_item)

    # 2. Execute native ChimeraX CLI commands end-to-end in a fresh --offscreen session
    cli_hm = tmp_path / "native_cli_heatmap.png"
    cli_render = tmp_path / "native_cli_pairs.png"
    cmds = (
        f'open "{fx["pdb"]}"; '
        f'esm_msr load "{fx["two_stage"]}" chain A mode Singles; '
        f'esm_msr visualize mode Singles palette purple_white_green minScore -2.0 maxScore 2.0; '
        f'esm_msr pairs "{fx["two_stage"]}" chain A topM 3 sortBy combined_pred diverse true connectors dashed labels true legend true; '
        f'view; save "{cli_render}" width 400 height 300; '
        f'esm_msr heatmap "{fx["two_stage"]}" "{cli_hm}" chain A palette purple_white_green; '
        f'esm_msr reset'
    )
    p_cli = subprocess.run(
        [chimerax_bin, "--offscreen", "--exit", "--cmd", cmds],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p_cli.returncode == 0, f"Native CLI run failed:\n{p_cli.stdout}\n{p_cli.stderr}"
    assert "Unknown command" not in p_cli.stdout and "Unknown command" not in p_cli.stderr
    assert cli_hm.is_file() and cli_hm.stat().st_size > 2000
    assert cli_render.is_file() and cli_render.stat().st_size > 2000


# Allow running directly via `chimerax --offscreen --exit --script test_chimerax_esm_msr.py`
if "session" in globals():
    run_chimerax_integration_suite(globals()["session"])
