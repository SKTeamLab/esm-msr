"""Automated headless/offscreen verification suite for ChimeraX-ESM-MSR (v0.4.0).

Run everything (unit + headless PyQt6 + real ChimeraX --offscreen integration):
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


def _load_bundle_modules():
    """Import src/parsing.py and src/tool.py as a synthetic package without installing."""
    pkg_name = "_test_chimerax_esm_msr_pkg"
    if pkg_name not in sys.modules:
        import types

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_SRC_DIR)]
        sys.modules[pkg_name] = pkg

    def _load_sub(name: str):
        full = f"{pkg_name}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _SRC_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    parsing_mod = _load_sub("parsing")
    tool_mod = _load_sub("tool")
    return parsing_mod, tool_mod


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
    """Create deterministic multi-chain PDB and 5 schema-accurate CSVs."""
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

    return {
        "pdb": pdb_path,
        "full": full_csv,
        "skip_reverse": skip_rev_csv,
        "legacy_dddg": legacy_csv,
        "epi_only": epi_only_csv,
        "edge": edge_csv,
    }


# ---------------------------------------------------------------------------
# Part 1: Pure unit tests for src/parsing.py
# ---------------------------------------------------------------------------

def test_parse_mut_string_edge_cases():
    parsing, _ = _load_bundle_modules()

    # Standard single
    m = parsing.parse_mut_string("G289A")
    assert len(m) == 1
    assert m[0] == {"wt": "G", "pos": 289, "icode": "", "pos_key": (289, ""), "mut": "A"}

    # Negative residue number
    m_neg = parsing.parse_mut_string("G-2A")
    assert len(m_neg) == 1
    assert m_neg[0]["pos"] == -2 and m_neg[0]["icode"] == "" and m_neg[0]["pos_key"] == (-2, "")

    # Insertion code (previously misparsed as pos=15, mut='A'!)
    m_ic = parsing.parse_mut_string("A15AC")
    assert len(m_ic) == 1
    assert m_ic[0] == {"wt": "A", "pos": 15, "icode": "A", "pos_key": (15, "A"), "mut": "C"}

    # Double with both negative resnum and insertion code
    m_dbl = parsing.parse_mut_string("a-2v:e15ak")
    assert len(m_dbl) == 2
    assert m_dbl[0]["pos_key"] == (-2, "") and m_dbl[0]["mut"] == "V"
    assert m_dbl[1]["pos_key"] == (15, "A") and m_dbl[1]["mut"] == "K"

    # Malformed token in double must NOT silently downgrade to a single
    assert parsing.parse_mut_string("A12C:BADTOKEN") == []
    assert parsing.parse_mut_string("A15ABC") == []
    assert parsing.parse_mut_string("") == []
    assert parsing.parse_mut_string(None) == []
    assert parsing.parse_mut_string(float("nan")) == []

    # Formatting helper
    assert parsing.format_res_pos((-2, "")) == "-2"
    assert parsing.format_res_pos((15, "A")) == "15A"
    assert parsing.format_res_pos((15, "a")) == "15A"


def test_parse_csv_all_synthetic_schemas(tmp_path):
    parsing, _ = _load_bundle_modules()
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
    # Valid exact 0.0 score preserved!
    assert res_a.all_singles_scores[("A", (14, ""), "I")] == 0.0
    # Negative resnum & insertion code preserved!
    assert res_a.residue_scores_data[("A", (-2, ""))] == (1.30, "V")
    assert res_a.residue_scores_data[("A", (15, "A"))] == (1.60, "K")
    # Chain B excluded when Chain A requested
    assert ("B", (10, "")) not in res_a.residue_scores_data
    # Verify epi_pred vs combined_dddg_pred values differ as expected
    row_a2 = res_a.epistasis_df[res_a.epistasis_df["pos1_key"] == (-2, "")].iloc[0]
    assert np.isclose(row_a2["dddg_pred"], 0.80)

    res_dddg = parsing.parse_esm_msr_csv(
        str(fx["full"]), requested_epi_col="combined_dddg_pred", target_chain="A", mode="MT epistasis"
    )
    assert res_dddg.ok and res_dddg.effective_epi_col == "combined_dddg_pred"
    row_a2_d = res_dddg.epistasis_df[res_dddg.epistasis_df["pos1_key"] == (-2, "")].iloc[0]
    assert np.isclose(row_a2_d["dddg_pred"], 0.60)

    # Chain mismatch reports available chains cleanly
    res_z = parsing.parse_esm_msr_csv(str(fx["full"]), target_chain="Z", mode="Singles")
    assert not res_z.ok
    assert not res_z.chain_matched
    assert res_z.chains_in_csv == ["A", "B"]
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

    # 5. Edge CSV: case-duplicate CHAIN/chain (exact lowercase 'chain'='A' wins!), NaN/Inf filtered, 0.0 kept
    res_edge = parsing.parse_esm_msr_csv(str(fx["edge"]), target_chain="A", mode="MT epistasis")
    assert res_edge.ok, res_edge.error
    assert any("case-duplicate" in w for w in res_edge.warnings)
    assert res_edge.all_singles_scores[("A", (-2, ""), "V")] == 0.0
    assert res_edge.n_singles == 2  # A-2V (0.0) and E15AK (1.2); NaN, Inf, BADMUT excluded
    assert res_edge.n_doubles == 1  # E15AK:G16A (0.85); NaN/Inf double excluded


def test_path_python_env_and_tqdm_helpers(tmp_path):
    parsing, _ = _load_bundle_modules()

    # Relative path resolution against base_repo_path
    resolved = parsing.resolve_user_path("LoRA_models/esm-msr-small/lora_seed1.safetensors", str(tmp_path))
    assert resolved == str(tmp_path / "LoRA_models" / "esm-msr-small" / "lora_seed1.safetensors")

    # Direct python executable file
    fake_py = tmp_path / "my_python"
    fake_py.write_text("#!/bin/sh\n", encoding="utf-8")
    prog, args, err = parsing.resolve_python_command(str(fake_py))
    assert err is None and prog == str(fake_py) and args == ["-u"]

    # Directory containing bin/python
    venv_dir = tmp_path / "fake_venv"
    (venv_dir / "bin").mkdir(parents=True)
    venv_py = venv_dir / "bin" / "python"
    venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    prog, args, err = parsing.resolve_python_command(str(venv_dir))
    assert err is None and prog == str(venv_py) and args == ["-u"]

    # Non-existent path gives clean error (doesn't fall through to conda run)
    prog, args, err = parsing.resolve_python_command("./does_not_exist_env", str(tmp_path))
    assert prog is None and err is not None and "does not exist" in err

    # Empty spec uses system python with -u
    prog, args, err = parsing.resolve_python_command("")
    assert err is None and prog is not None and "-u" in args

    # Conda env name (when conda is on PATH)
    with patch("shutil.which", return_value="/usr/bin/conda"):
        prog, args, err = parsing.resolve_python_command("msr_venv")
        assert err is None and prog == "/usr/bin/conda"
        assert args == ["run", "--no-capture-output", "-n", "msr_venv", "python", "-u"]

    # tqdm progress parsing
    assert parsing.parse_tqdm_progress("Scoring:  42%|████      | 42/100 [00:05<00:07, 8.1it/s]") == 42
    assert parsing.parse_tqdm_progress("100%|██████████| 10/10 [00:01<00:00]") == 100
    assert parsing.parse_tqdm_progress("Loading checkpoint...") is None


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

    parsing, tool_mod = _load_bundle_modules()

    app = QApplication.instance() or QApplication(["chimerax_esm_msr_test"])

    # Shim MainToolWindow with a real QWidget
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

        # Open multi-chain PDB with negative resnum (-2), insertion code (15A), and Glycine (16)
        opened = run(session, f'open "{fx["pdb"]}"')
        wt_model = opened[0]
        assert isinstance(wt_model, Structure)

        # Create a user-owned pseudobond group to verify ESM-MSR NEVER deletes user pbonds!
        user_pbg = session.pb_manager.get_group("User Custom Bonds")
        if getattr(user_pbg, "id", None) is None:
            session.models.add([user_pbg])
        ca_atoms = [a for a in wt_model.atoms if a.name == "CA" and a.residue.chain_id == "B"]
        user_pb = user_pbg.new_pseudobond(ca_atoms[0], ca_atoms[1])
        assert len(user_pbg.pseudobonds) == 1

        # Instantiate real ESM_MSR_Tool
        tool = tool_mod.ESM_MSR_Tool(session, "ESM_MSR")
        app.processEvents()

        # Verify QScrollArea wrapping and key widgets
        assert isinstance(tool.tabs.widget(0), QScrollArea)
        assert isinstance(tool.tabs.widget(1), QScrollArea)
        assert tool.model_dtype_combobox.count() == 3
        assert [tool.model_dtype_combobox.itemText(i) for i in range(3)] == ["float32", "bfloat16", "float16"]
        assert tool.mask_strategy_combobox.itemText(0) == "Checkpoint default"

        # Verify _grab_selection scopes to target model & chain and preserves negative resnum + icode
        run(session, f"select #{wt_model.id_string}/A:-2,15A | #{wt_model.id_string}/B:10")
        QTest.mouseClick(tool.grab_sel_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool.selected_residues_edit.text() == "-2,15A", f"Got: {tool.selected_residues_edit.text()!r}"

        # ---- Pre-flight validation (must NOT create temp dirs or crash) ----
        tool.base_repo_path_edit.setText("")
        QTest.mouseClick(tool.run_prediction_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool._temp_dir_to_cleanup is None
        assert "Error:" in tool.status_label.text()

        # ---- Subprocess argv verification (mock QProcess.start so no GPU run occurs) ----
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
                return 0  # Test P0 #1 pid=0 guard!

            def kill(self):
                self._state = tool_mod._PROC_NOT_RUNNING

            def waitForFinished(self, _ms):
                return True

            def start(self, program, args):
                self._state = 2  # Running
                started_calls.append((program, list(args), self._wd, self._env.value("PYTHONPATH", "")))

            def deleteLater(self):
                pass

        # Test standard run with real repo, real checkpoint, bfloat16, -u preserved, writable CWD
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
            assert args[0] == "-u", f"-u dropped: {args}"
            assert "--model_dtype" in args and args[args.index("--model_dtype") + 1] == "bfloat16"
            assert "--checkpoint_path" in args
            assert "--mode" in args and args[args.index("--mode") + 1] == "singles+doubles"
            assert "--mask_strategy" not in args  # Checkpoint default omits --mask_strategy
            assert os.path.isdir(wd) and wd.startswith(tempfile.gettempdir())
            assert str(_REPO_ROOT / "src") in pypath

            # Test P0 #1: STOP with pid == 0 must NOT invoke subprocess.run(["kill", "-9", "0"])!
            with patch.object(tool_mod.subprocess, "run") as mock_sub_run:
                tool._stop_prediction_script(silent=False)
                mock_sub_run.assert_not_called()

        # Test P0 #3: zero-shot mode (empty checkpoint + hparams.yaml) omits --checkpoint_path
        tool.checkpoint_path_edit.setText("")
        tool.config_file_edit.setText("LoRA_models/esm-msr-small/hparams.yaml")
        tool.sigma_spinbox.setValue(0.0)
        with patch.object(tool_mod, "QProcess", _FakeProc):
            tool._initiate_run_prediction_script()
            assert len(started_calls) == 2
            _prog, args2, _wd2, _pypath2 = started_calls[-1]
            assert "--checkpoint_path" not in args2
            assert "--hparams_path" in args2
            tool._stop_prediction_script(silent=True)

        # ---- Load all 5 synthetic CSVs and exercise real ChimeraX visualization ----
        # (a) --skip_reverse CSV loaded while in MT epistasis mode: auto-falls back to Singles & WT LoRA,
        #     disables Dual-view & MT LoRA & epistasis modes cleanly without error
        tool.display_mode_combo.setCurrentText("MT epistasis")
        assert tool._load_csv_file(str(fx["skip_reverse"]))
        assert tool.display_mode_combo.currentText() == "Singles"
        assert tool.radio_wt_lora.isChecked()
        assert not tool.radio_dual_view.isEnabled()
        assert not tool.radio_mt_lora.isEnabled()
        assert not tool.display_mode_combo.model().item(1).isEnabled()
        assert not tool.display_mode_combo.model().item(2).isEnabled()
        assert "wt_lora_pred" in tool.csv_label.text()

        # (b) Edge CSV (case-duplicate CHAIN/chain, NaN, Inf, exact 0.0)
        assert tool._load_csv_file(str(fx["edge"]))
        assert ("A", (-2, ""), "V") in tool.all_singles_scores
        assert tool.all_singles_scores[("A", (-2, ""), "V")] == 0.0

        # (c) Full singles+doubles CSV -> Singles mode with backbone coloring + nearby atoms (incl. Gly 16)
        tool.display_mode_combo.setCurrentText("Singles")
        tool.color_backbone_checkbox.setChecked(True)
        tool.show_nearby_checkbox.setChecked(True)
        tool.nearby_distance_spinbox.setValue(5.0)
        # Set a known user selection to verify _apply_nearby preserves it!
        run(session, f"select #{wt_model.id_string}/B:12")
        before_sel_count = len(selected_atoms(session))
        assert before_sel_count > 0

        assert tool._load_csv_file(str(fx["full"]))
        after_sel = selected_atoms(session)
        assert len(after_sel) == before_sel_count, "User selection was clobbered by _apply_nearby!"
        # Verify mutant clone tracked and hidden from model comboboxes!
        assert len(tool.mutated_model_id_strings) == 1
        assert tool.viz_model_combobox.count() == 1
        # Verify -2 and 15A were actually swapped on the mutant clone!
        clone = next(m for m in session.models.list(type=Structure) if m.id_string == tool.mutated_model_id_strings[0])
        clone_map = {(r.chain_id, r.number, r.insertion_code.strip()): r.name for r in clone.residues}
        assert clone_map[("A", -2, "")] == "VAL", f"Negative resnum swapaa failed: {clone_map}"
        assert clone_map[("A", 15, "A")] == "LYS", f"Insertion code swapaa failed: {clone_map}"

        # (d) MT epistasis mode -> verify dedicated "ESM-MSR Epistasis" PseudobondGroup + user pbond preserved!
        tool.display_mode_combo.setCurrentText("MT epistasis")
        tool.show_nearby_checkbox.setChecked(False)
        tool.pos_threshold_spinbox.setValue(0.0)
        tool.neg_threshold_spinbox.setValue(0.0)
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()

        pbgs = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" in pbgs, f"Missing dedicated pbg: {list(pbgs)}"
        assert len(pbgs["ESM-MSR Epistasis"].pseudobonds) >= 2
        assert "User Custom Bonds" in pbgs and len(pbgs["User Custom Bonds"].pseudobonds) == 1

        # (e) WT epistasis mode with custom Mut Color & Hide Fully (P1 #14)
        tool.display_mode_combo.setCurrentText("WT epistasis")
        tool.mut_color_edit.setText("#00aa88")
        tool.mut_hide_checkbox.setChecked(False)
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        pbgs_wt = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" in pbgs_wt and len(pbgs_wt["ESM-MSR Epistasis"].pseudobonds) == 1
        assert "User Custom Bonds" in pbgs_wt and len(pbgs_wt["User Custom Bonds"].pseudobonds) == 1

        # Also test Hide Fully in WT epistasis
        tool.mut_hide_checkbox.setChecked(True)
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        tool.mut_hide_checkbox.setChecked(False)
        tool.mut_color_edit.setText("")

        # (f) Chain mismatch retains loaded_csv_path so switching chain + Update Viz succeeds immediately
        tool.display_mode_combo.setCurrentText("Singles")
        tool.viz_chain_id_combobox.setCurrentText("Z")
        assert not tool._load_csv_file(str(fx["full"]))
        assert tool.loaded_csv_path == str(fx["full"])
        assert tool.update_viz_button.isEnabled()
        tool.viz_chain_id_combobox.setCurrentText("B")
        QTest.mouseClick(tool.update_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert "Chain B" in tool.csv_label.text()

        # (g) Reset / Clear Viz closes clones & deletes ONLY "ESM-MSR Epistasis" (keeps user pbonds!)
        QTest.mouseClick(tool.reset_viz_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert tool.mutated_model_id_strings == []
        pbgs_after_reset = {g.name: g for g in session.models.list(type=PseudobondGroup)}
        assert "ESM-MSR Epistasis" not in pbgs_after_reset
        assert "User Custom Bonds" in pbgs_after_reset and len(pbgs_after_reset["User Custom Bonds"].pseudobonds) == 1

        # (h) Real PDB + CSV smoke tests (if available in parent workspace)
        real_root = _REPO_ROOT.parent
        csn4_pdb = real_root / "ProteinGym" / "ProteinGym_AF2_structures" / "CSN4_MOUSE.pdb"
        csn4_csv = real_root / "final_pgym_comb_sigma0.5" / "FINAL_p0" / "s1" / "CSN4_MOUSE_Tsuboyama_2023_1UFM.csv"
        grb2_pdb = real_root / "ProteinGym" / "ProteinGym_AF2_structures" / "GRB2_HUMAN.pdb"
        grb2_csv = real_root / "final_pgym_comb_sigma0.5" / "FINAL_p0" / "s1" / "GRB2_HUMAN_Faure_2021.csv"

        if csn4_pdb.is_file() and csn4_csv.is_file():
            run(session, "close all")
            m_csn4 = run(session, f'open "{csn4_pdb}"')[0]
            tool._refresh_models()
            tool.display_mode_combo.setCurrentText("Singles")
            tool.pos_threshold_spinbox.setValue(0.3)
            tool.neg_threshold_spinbox.setValue(-2.5)
            assert tool._load_csv_file(str(csn4_csv))
            render_csn4 = work_dir / "csn4_singles.png"
            run(session, f'view; save "{render_csn4}" width 320 height 240')
            arr = np.array(Image.open(render_csn4))
            assert len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)) > 50

        if grb2_pdb.is_file() and grb2_csv.is_file():
            run(session, "close all")
            m_grb2 = run(session, f'open "{grb2_pdb}"')[0]
            tool._refresh_models()
            tool.display_mode_combo.setCurrentText("MT epistasis")
            tool.pos_threshold_spinbox.setValue(0.5)
            tool.neg_threshold_spinbox.setValue(-0.5)
            tool.epi_max_edges.setValue(2)
            assert tool._load_csv_file(str(grb2_csv))
            pbgs_grb2 = {g.name: g for g in session.models.list(type=PseudobondGroup)}
            assert "ESM-MSR Epistasis" in pbgs_grb2 and len(pbgs_grb2["ESM-MSR Epistasis"].pseudobonds) > 0
            render_grb2 = work_dir / "grb2_epistasis.png"
            run(session, f'view; save "{render_grb2}" width 320 height 240')
            arr2 = np.array(Image.open(render_grb2))
            assert len(np.unique(arr2.reshape(-1, arr2.shape[-1]), axis=0)) > 50

        # (i) Lifecycle delete(): removes triggers cleanly, subsequent model ops do not raise!
        tool.delete()
        assert tool._closing is True
        assert tool._models_added_handler is None
        assert tool._models_removed_handler is None
        run(session, f'open "{fx["pdb"]}"')
        run(session, "close all")

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


# Allow running directly via `chimerax --offscreen --exit --script test_chimerax_esm_msr.py`
if "session" in globals():
    run_chimerax_integration_suite(globals()["session"])
