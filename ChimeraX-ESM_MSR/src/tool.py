# vim: set expandtab shiftwidth=4 softtabstop=4:
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict

import numpy as np

from chimerax.ui import MainToolWindow
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.atomic import Structure, Residue, PseudobondGroup, selected_atoms

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFileDialog, QLineEdit, QFrame, QGroupBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QTabWidget, QRadioButton, QButtonGroup,
    QScrollArea, QProgressBar, QColorDialog, QMessageBox, QSizePolicy,
)
from Qt.QtCore import QProcess, QSettings, QProcessEnvironment
from Qt.QtGui import QColor

from .parsing import (
    format_res_pos,
    normalize_icode,
    parse_esm_msr_csv,
    parse_tqdm_progress,
    resolve_python_command,
    resolve_user_path,
)


def _qt_enum(cls, group_name, member_name):
    """Return Qt6 scoped enum (cls.Group.Member) with Qt5 unscoped fallback."""
    group = getattr(cls, group_name, None)
    if group is not None and hasattr(group, member_name):
        return getattr(group, member_name)
    return getattr(cls, member_name)


_FRAME_HLINE = _qt_enum(QFrame, "Shape", "HLine")
_FRAME_NOFRAME = _qt_enum(QFrame, "Shape", "NoFrame")
_FRAME_SUNKEN = _qt_enum(QFrame, "Shadow", "Sunken")
_PROC_NOT_RUNNING = _qt_enum(QProcess, "ProcessState", "NotRunning")
_PROC_MERGED = _qt_enum(QProcess, "ProcessChannelMode", "MergedChannels")
_PROC_NORMAL_EXIT = _qt_enum(QProcess, "ExitStatus", "NormalExit")
_SIZE_EXPANDING = _qt_enum(QSizePolicy, "Policy", "Expanding")
_SIZE_FIXED = _qt_enum(QSizePolicy, "Policy", "Fixed")
_COMBO_ADJUST_MIN = _qt_enum(QComboBox, "SizeAdjustPolicy", "AdjustToMinimumContentsLengthWithIcon")

SCORE_ATTRIBUTE_NAME = "residue_score_viz_score"
EPISTASIS_PBG_NAME = "ESM-MSR Epistasis"

ONE_TO_THREE_LETTER_AA = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

_STATUS_STYLES = {
    "info": "color: #222222;",
    "success": "color: #1b7a3d; font-weight: bold;",
    "warning": "color: #9a6700; font-weight: bold;",
    "error": "color: #c62828; font-weight: bold;",
}


class ESM_MSR_Tool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    display_name = "ESM-MSR Predictor & Visualizer"

    def __init__(self, session, tool_registered_name):
        super().__init__(session, tool_registered_name)

        # 1. INITIALIZE SETTINGS
        self.settings = QSettings("ESM_MSR_Tools", "ESM_MSR")

        # 2. LOAD STATE VARIABLES
        self.base_repo_path = str(self.settings.value("base_repo_path", "") or "")
        self.python_env = str(self.settings.value("python_env", "") or "")
        self.script_output_csv_path = str(self.settings.value("script_output_csv_path", "") or "")
        self.checkpoint_path = str(self.settings.value("checkpoint_path", "") or "")
        self.config_file = str(self.settings.value("config_file", "") or "")
        self.base_model_loc = str(self.settings.value("base_model_loc", "") or "")

        self._closing = False
        self._suppress_model_refresh = False
        self._user_stopped = False
        self._temp_dir_to_cleanup = None
        self._models_added_handler = None
        self._models_removed_handler = None
        self._attr_registered = False

        self.residue_scores_data = {}
        self.all_singles_scores = {}
        self.epistasis_df = None
        self.loaded_csv_path = ""
        self.predicted_output_path = ""
        self.mutated_model_id_strings = []
        self.proc = None

        # CSV capability tracking (used to enable/disable controls intelligently)
        self._available_score_cols = None
        self._available_epi_cols = None
        self._has_doubles = None
        self._active_run_output_csv = ""

        # 3. BUILD THE UI
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        self._build_ui(parent)
        self.tool_window.manage(None)

        # 4. POPULATE UI WITH DYNAMIC PATHS
        self._update_paths_from_base_repo(self.base_repo_path, is_init=True)

        # 5. INITIAL PATH VALIDATION & BUTTON STATE
        self._validate_file_paths()
        self._refresh_load_last_output_button()

        # 6. REGISTER CHIMERAX EVENT HOOKS
        try:
            self._models_added_handler = self.session.triggers.add_handler("add models", self._refresh_models)
            self._models_removed_handler = self.session.triggers.add_handler("remove models", self._refresh_models)
        except Exception as e:
            self.session.logger.warning(f"Could not register model event hooks: {e}")

        # Initial populate
        self._refresh_models()

    # ---------------- Lifecycle Cleanup -----------------
    def delete(self):
        self._closing = True

        # Remove ChimeraX triggers safely
        for attr in ("_models_added_handler", "_models_removed_handler"):
            handler = getattr(self, attr, None)
            if handler is not None:
                try:
                    self.session.triggers.remove_handler(handler)
                except Exception:
                    pass
                setattr(self, attr, None)

        # Terminate any running subprocess safely
        try:
            self._stop_prediction_script(silent=True)
        except Exception as e:
            self.session.logger.warning(f"Error stopping prediction process during tool close: {e}")

        # Clean up temporary directories
        self._cleanup_temp_dir()

        super().delete()

    def _cleanup_temp_dir(self):
        if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
            try:
                shutil.rmtree(self._temp_dir_to_cleanup)
            except Exception as e:
                self.session.logger.warning(
                    f"Could not remove temporary directory {self._temp_dir_to_cleanup}: {e}"
                )
        self._temp_dir_to_cleanup = None

    # ---------------- Status & Messaging Helpers -----------------
    def _set_status(self, message, level="info"):
        if self._closing or not hasattr(self, "status_label"):
            return
        prefix = "Status: " if not message.startswith(("Status:", "Error:", "Warning:")) else ""
        self.status_label.setText(f"{prefix}{message}")
        self.status_label.setStyleSheet(_STATUS_STYLES.get(level, _STATUS_STYLES["info"]))

    def _show_warning_dialog(self, title, message):
        """Show a QMessageBox warning only in interactive GUI mode (never blocks headless/offscreen runs)."""
        ui = getattr(self.session, "ui", None)
        is_gui = bool(getattr(ui, "is_gui", False))
        if is_gui and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            parent = getattr(ui, "main_window", None)
            QMessageBox.warning(parent, title, message)

    def _report_error(self, title, message, popup=False):
        self.session.logger.error(f"{title}: {message}")
        self._set_status(f"Error: {message}", level="error")
        if popup:
            self._show_warning_dialog(title, message)

    # ---------------- Color Swatch Helper -----------------
    def _create_color_swatch(self, line_edit, default_fallback=""):
        btn = QPushButton("")
        btn.setFixedSize(22, 22)
        btn.setToolTip("Pick a color from a color dialog (clear text box to restore default/score coloring).")

        def _sync_swatch(text=None):
            txt = (line_edit.text() if text is None else text).strip()
            color = QColor(txt) if txt else QColor(default_fallback)
            if txt and color.isValid():
                btn.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #666666; border-radius: 2px;"
                )
            elif not txt:
                btn.setStyleSheet(
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    "stop:0 #FF1493, stop:0.5 #FFFFFF, stop:1 #008000); "
                    "border: 1px solid #666666; border-radius: 2px;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: #dddddd; border: 1px solid #999999; border-radius: 2px;"
                )

        def _pick_color():
            cur = line_edit.text().strip() or default_fallback or "white"
            initial = QColor(cur)
            if not initial.isValid():
                initial = QColor("white")
            chosen = QColorDialog.getColor(initial, self.tool_window.ui_area, "Select Color")
            if chosen.isValid():
                line_edit.setText(chosen.name())

        line_edit.textChanged.connect(_sync_swatch)
        btn.clicked.connect(_pick_color)
        _sync_swatch()
        return btn

    # ---------------- UI Construction -----------------
    def _build_ui(self, parent_widget):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        parent_widget.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Wrap Tab 1 in QScrollArea so narrow/short docked panels never clip
        self.tab_io = QWidget()
        self._build_io_tab(self.tab_io)
        self.tab_io_scroll = QScrollArea()
        self.tab_io_scroll.setWidgetResizable(True)
        self.tab_io_scroll.setFrameShape(_FRAME_NOFRAME)
        self.tab_io_scroll.setWidget(self.tab_io)
        self.tabs.addTab(self.tab_io_scroll, "Execution / IO")

        # Wrap Tab 2 in QScrollArea
        self.tab_viz = QWidget()
        self._build_viz_tab(self.tab_viz)
        self.tab_viz_scroll = QScrollArea()
        self.tab_viz_scroll.setWidgetResizable(True)
        self.tab_viz_scroll.setFrameShape(_FRAME_NOFRAME)
        self.tab_viz_scroll.setWidget(self.tab_viz)
        self.tabs.addTab(self.tab_viz_scroll, "Visualization")

        separator = QFrame()
        separator.setFrameShape(_FRAME_HLINE)
        separator.setFrameShadow(_FRAME_SUNKEN)
        main_layout.addWidget(separator)

        # Global Action Bar — split into two clean rows to fit ~520px width comfortably
        exec_row1 = QHBoxLayout()
        exec_row1.setSpacing(6)
        self.run_prediction_button = QPushButton("Run Prediction Script")
        self.run_prediction_button.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.run_prediction_button.setToolTip("Export target structure (if needed) and launch src/esm_msr/inference.py.")
        self.run_prediction_button.clicked.connect(self._initiate_run_prediction_script)
        exec_row1.addWidget(self.run_prediction_button)

        self.stop_prediction_button = QPushButton("STOP")
        self.stop_prediction_button.setStyleSheet(
            "background-color: #c62828; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.stop_prediction_button.setToolTip("Terminate the running inference process.")
        self.stop_prediction_button.clicked.connect(lambda: self._stop_prediction_script(silent=False))
        self.stop_prediction_button.setVisible(False)
        exec_row1.addWidget(self.stop_prediction_button)

        self.load_button = QPushButton("Load CSV")
        self.load_button.setToolTip("Browse for an ESM-MSR output CSV and visualize it on the target model.")
        self.load_button.clicked.connect(self._handle_load_and_visualize)
        exec_row1.addWidget(self.load_button)

        self.load_last_output_button = QPushButton("Load Last Output")
        self.load_last_output_button.setToolTip(
            "Directly load the last completed prediction output CSV (or the configured Output CSV if it exists)."
        )
        self.load_last_output_button.setEnabled(False)
        self.load_last_output_button.clicked.connect(self._handle_load_last_output)
        exec_row1.addWidget(self.load_last_output_button)
        main_layout.addLayout(exec_row1)

        exec_row2 = QHBoxLayout()
        exec_row2.setSpacing(6)
        self.update_viz_button = QPushButton("Update Visualization")
        self.update_viz_button.setToolTip("Re-apply the current visualization settings to the loaded CSV.")
        self.update_viz_button.clicked.connect(self._update_visualization)
        self.update_viz_button.setEnabled(False)
        exec_row2.addWidget(self.update_viz_button)

        self.reset_viz_button = QPushButton("Reset / Clear Viz")
        self.reset_viz_button.setToolTip(
            "Close ESM-MSR generated mutant clones, remove the 'ESM-MSR Epistasis' pseudobond group, "
            "and restore the target model to clean white ribbon without touching other models or user pseudobonds."
        )
        self.reset_viz_button.clicked.connect(self._reset_visualization)
        exec_row2.addWidget(self.reset_viz_button)
        main_layout.addLayout(exec_row2)

        # Compact live progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(14)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        self.prediction_output_label = QLabel("Predicted output file: None")
        self.prediction_output_label.setWordWrap(True)
        self.prediction_output_label.setStyleSheet("color: #555555; font-size: 11px;")
        main_layout.addWidget(self.prediction_output_label)

    def _build_io_tab(self, parent_widget):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        parent_widget.setLayout(layout)

        # 1. Local Path Configuration
        paths_group = QGroupBox("Local Path Configuration")
        paths_grid = QGridLayout()
        paths_grid.setSpacing(5)
        paths_grid.setColumnStretch(1, 1)
        paths_group.setLayout(paths_grid)

        paths_grid.addWidget(QLabel("Base Repo Dir:"), 0, 0)
        self.base_repo_path_edit = QLineEdit(self.base_repo_path)
        self.base_repo_path_edit.setPlaceholderText("Path to esm-msr repository root")
        self.base_repo_path_edit.setToolTip("Root directory of the esm-msr repository (containing src/esm_msr/inference.py).")
        self.base_repo_path_edit.textChanged.connect(self._validate_file_paths)
        paths_grid.addWidget(self.base_repo_path_edit, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_base_repo)
        paths_grid.addWidget(btn, 0, 2)

        paths_grid.addWidget(QLabel("Python Env:"), 1, 0)
        self.python_env_edit = QLineEdit(self.python_env)
        self.python_env_edit.setPlaceholderText("Venv/conda dir, python binary, or conda env name")
        self.python_env_edit.setToolTip(
            "Accepts: (1) direct path to a python/python.exe executable, "
            "(2) path to a venv or conda environment folder, "
            "(3) a conda environment name (via 'conda run'), or blank for system python."
        )
        paths_grid.addWidget(self.python_env_edit, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_python_env)
        paths_grid.addWidget(btn, 1, 2)

        paths_grid.addWidget(QLabel("Output CSV:"), 2, 0)
        self.script_output_csv_path_edit = QLineEdit(self.script_output_csv_path)
        self.script_output_csv_path_edit.setPlaceholderText("Destination path for prediction CSV")
        self.script_output_csv_path_edit.setToolTip("Output CSV path for ESM-MSR predictions. Relative paths resolve against Base Repo Dir.")
        self.script_output_csv_path_edit.textChanged.connect(self._refresh_load_last_output_button)
        paths_grid.addWidget(self.script_output_csv_path_edit, 2, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_script_output_csv)
        paths_grid.addWidget(btn, 2, 2)

        paths_grid.addWidget(QLabel("ESM3 Weights:"), 3, 0)
        self.base_model_loc_edit = QLineEdit(self.base_model_loc)
        self.base_model_loc_edit.setPlaceholderText("Optional: local ESM3 weights folder")
        self.base_model_loc_edit.setToolTip(
            "Optional local directory containing data/weights/esm3_sm_open_v1.pth (leave blank to download via esm/HuggingFace)."
        )
        paths_grid.addWidget(self.base_model_loc_edit, 3, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(
            lambda: self._browse_generic_dir(self.base_model_loc_edit, "Select Model Directory", "base_model_loc")
        )
        paths_grid.addWidget(btn, 3, 2)

        layout.addWidget(paths_group)

        # 2. Compute Environment
        device_group = QGroupBox("Compute Environment")
        device_grid = QGridLayout()
        device_grid.setSpacing(6)
        device_group.setLayout(device_grid)

        device_grid.addWidget(QLabel("Compute Device:"), 0, 0)
        self.device_combobox = QComboBox()
        self.device_combobox.addItems(["cuda", "cuda:0", "cuda:1", "mps", "cpu"])
        self.device_combobox.setToolTip("PyTorch device passed via --device.")
        device_grid.addWidget(self.device_combobox, 0, 1)

        device_grid.addWidget(QLabel("Model Precision (dtype):"), 0, 2)
        self.model_dtype_combobox = QComboBox()
        self.model_dtype_combobox.addItems(["float32", "bfloat16", "float16"])
        self.model_dtype_combobox.setToolTip(
            "Inference precision passed via --model_dtype. "
            "float32 (default, CPU/GPU safe); bfloat16 (faster & lower VRAM on Ampere+ CUDA GPUs); float16 (older GPUs)."
        )
        device_grid.addWidget(self.model_dtype_combobox, 0, 3)

        device_grid.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 512)
        self.batch_size_spinbox.setValue(16)
        self.batch_size_spinbox.setToolTip("Inference batch size (--batch_size). Lower this if you encounter CUDA out-of-memory errors.")
        device_grid.addWidget(self.batch_size_spinbox, 1, 1)
        device_grid.setColumnStretch(1, 1)
        device_grid.setColumnStretch(3, 1)

        layout.addWidget(device_group)

        # 3. Model Configuration Files
        files_group = QGroupBox("Model Configuration Files")
        files_grid = QGridLayout()
        files_grid.setSpacing(5)
        files_grid.setColumnStretch(1, 1)
        files_group.setLayout(files_grid)

        files_grid.addWidget(QLabel("Checkpoint:"), 0, 0)
        self.checkpoint_path_edit = QLineEdit(self.checkpoint_path)
        self.checkpoint_path_edit.setPlaceholderText(".ckpt or .safetensors (leave blank for zero-shot)")
        self.checkpoint_path_edit.setToolTip(
            "Path to LoRA checkpoint (.safetensors or .ckpt). Leave blank (with a LoRA Config) or set Sigma = 0.0 for zero-shot ESM3."
        )
        self.checkpoint_path_edit.textChanged.connect(self._validate_file_paths)
        self.checkpoint_path_edit.textEdited.connect(self._on_checkpoint_text_edited)
        files_grid.addWidget(self.checkpoint_path_edit, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_checkpoint_path)
        files_grid.addWidget(btn, 0, 2)

        files_grid.addWidget(QLabel("LoRA Config:"), 1, 0)
        self.config_file_edit = QLineEdit(self.config_file)
        self.config_file_edit.setPlaceholderText("Optional JSON/YAML (auto-detects hparams.yaml)")
        self.config_file_edit.setToolTip(
            "Optional LoRA config JSON (--lora_config) or Lightning hparams.yaml (--hparams_path). "
            "Leave blank to auto-detect hparams.yaml beside the checkpoint."
        )
        self.config_file_edit.textChanged.connect(self._validate_file_paths)
        files_grid.addWidget(self.config_file_edit, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(
            lambda: self._browse_generic(
                self.config_file_edit,
                "Select Config",
                "JSON/YAML (*.json *.yaml *.yml);;All (*)",
                "config_file",
            )
        )
        files_grid.addWidget(btn, 1, 2)

        self.file_warning_label = QLabel("")
        self.file_warning_label.setWordWrap(True)
        self.file_warning_label.setStyleSheet("color: #c62828; font-weight: bold;")
        self.file_warning_label.setVisible(False)
        files_grid.addWidget(self.file_warning_label, 2, 0, 1, 3)

        layout.addWidget(files_group)

        # 4. Screening Config
        screening_config_group = QGroupBox("Screening Config")
        screening_config_layout = QVBoxLayout()
        screening_config_layout.setSpacing(6)
        screening_config_group.setLayout(screening_config_layout)

        # Target Selection
        target_group = QGroupBox("Target Selection")
        target_layout = QHBoxLayout()
        target_layout.setSpacing(5)
        target_group.setLayout(target_layout)

        target_layout.addWidget(QLabel("Target Model:"))
        self.pred_model_combobox = QComboBox()
        self.pred_model_combobox.setSizePolicy(_SIZE_EXPANDING, _SIZE_FIXED)
        self.pred_model_combobox.currentIndexChanged.connect(self._on_model_selected)
        target_layout.addWidget(self.pred_model_combobox, stretch=1)

        self.pred_refresh_models_button = QPushButton("Refresh")
        self.pred_refresh_models_button.setToolTip("Refresh the list of open ChimeraX atomic models and chains.")
        self.pred_refresh_models_button.clicked.connect(lambda: self._refresh_models())
        target_layout.addWidget(self.pred_refresh_models_button)

        target_layout.addWidget(QLabel("Chain:"))
        self.pred_chain_id_combobox = QComboBox()
        self.pred_chain_id_combobox.setEditable(True)
        self.pred_chain_id_combobox.setMaximumWidth(75)
        self.pred_chain_id_combobox.currentTextChanged.connect(self._on_pred_chain_changed)
        target_layout.addWidget(self.pred_chain_id_combobox)

        screening_config_layout.addWidget(target_group)

        # Mutation Scope
        scope_group = QGroupBox("Mutation Scope (Mutually Exclusive)")
        scope_layout = QVBoxLayout()
        scope_layout.setSpacing(5)
        scope_group.setLayout(scope_layout)

        self.scope_btn_group = QButtonGroup()

        # Mode 1: Full Screen
        meth1_row1 = QHBoxLayout()
        self.radio_full = QRadioButton("1. Full Screen:")
        self.scope_btn_group.addButton(self.radio_full)
        meth1_row1.addWidget(self.radio_full)

        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(["singles", "singles+doubles"])
        self.mode_combobox.setToolTip(
            "'singles' runs site-saturation mutagenesis; 'singles+doubles' also screens double mutants for epistasis."
        )
        self.mode_combobox.currentTextChanged.connect(self._update_scope_ui)
        meth1_row1.addWidget(self.mode_combobox)

        self.screen_except_checkbox = QCheckBox("Invert (Except)")
        self.screen_except_checkbox.setToolTip("If checked, passes --screen_residues_except instead of --screen_residues.")
        meth1_row1.addWidget(self.screen_except_checkbox)
        meth1_row1.addStretch()
        scope_layout.addLayout(meth1_row1)

        meth1_row2 = QHBoxLayout()
        meth1_row2.addSpacing(20)
        meth1_row2.addWidget(QLabel("Positions:"))
        self.selected_residues_edit = QLineEdit()
        self.selected_residues_edit.setPlaceholderText("e.g. 11,12,15A (blank = all)")
        self.selected_residues_edit.setToolTip(
            "Comma-separated PDB residue positions (supports negative indices & insertion codes like -2,15A). Leave empty to screen all residues."
        )
        meth1_row2.addWidget(self.selected_residues_edit, stretch=1)

        self.grab_sel_button = QPushButton("Grab Selection")
        self.grab_sel_button.setToolTip(
            "Populate Positions from currently selected atoms in the selected Target Model and Chain (preserving insertion codes)."
        )
        self.grab_sel_button.clicked.connect(self._grab_selection)
        meth1_row2.addWidget(self.grab_sel_button)
        scope_layout.addLayout(meth1_row2)

        meth1_row3 = QHBoxLayout()
        meth1_row3.addSpacing(20)
        self.enable_distance_checkbox = QCheckBox("Filter doubles by WT heavy-atom distance (Å):")
        self.enable_distance_checkbox.setChecked(False)
        self.enable_distance_checkbox.setToolTip(
            "Restrict double-mutant pairs to residues within the specified WT heavy-atom distance (--distance_threshold + --calculate_distances)."
        )
        self.enable_distance_checkbox.toggled.connect(self._toggle_distance_spinbox)
        meth1_row3.addWidget(self.enable_distance_checkbox)

        self.distance_threshold_spinbox = QDoubleSpinBox()
        self.distance_threshold_spinbox.setRange(0.0, 100.0)
        self.distance_threshold_spinbox.setValue(6.0)
        self.distance_threshold_spinbox.setEnabled(False)
        self.distance_threshold_spinbox.setToolTip("Maximum WT heavy-atom distance in Angstroms for double-mutant pairing.")
        meth1_row3.addWidget(self.distance_threshold_spinbox)
        meth1_row3.addStretch()
        scope_layout.addLayout(meth1_row3)

        # Mode 2: Input CSV
        meth2_layout = QHBoxLayout()
        self.radio_csv = QRadioButton("2. Input CSV:")
        self.scope_btn_group.addButton(self.radio_csv)
        meth2_layout.addWidget(self.radio_csv)

        self.subset_df_edit = QLineEdit()
        self.subset_df_edit.setPlaceholderText("Path to input mutations CSV...")
        self.subset_df_edit.setToolTip("CSV with columns pdb_file, code, chain, and mut_type/mut_type_pdb (--input_csv).")
        meth2_layout.addWidget(self.subset_df_edit, stretch=1)
        self.subset_df_btn = QPushButton("Browse...")
        self.subset_df_btn.clicked.connect(self._browse_subset_df)
        meth2_layout.addWidget(self.subset_df_btn)
        scope_layout.addLayout(meth2_layout)

        # Mode 3: Direct Mutations
        meth3_layout = QHBoxLayout()
        self.radio_explicit = QRadioButton("3. Direct Mutations:")
        self.scope_btn_group.addButton(self.radio_explicit)
        meth3_layout.addWidget(self.radio_explicit)

        self.mutations_edit = QLineEdit()
        self.mutations_edit.setPlaceholderText("e.g. A12C,A12C:D15E")
        self.mutations_edit.setToolTip("Comma-separated PDB-numbered mutations passed via --mutations (e.g. A12C,A12C:D15E).")
        meth3_layout.addWidget(self.mutations_edit, stretch=1)
        scope_layout.addLayout(meth3_layout)

        screening_config_layout.addWidget(scope_group)

        self.radio_full.toggled.connect(self._update_scope_ui)
        self.radio_csv.toggled.connect(self._update_scope_ui)
        self.radio_explicit.toggled.connect(self._update_scope_ui)
        self.radio_full.setChecked(True)

        # Screening Parameters
        runtime_group = QGroupBox("Screening Parameters")
        runtime_layout = QVBoxLayout()
        runtime_layout.setSpacing(5)
        runtime_group.setLayout(runtime_layout)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Mask Strategy:"))
        self.mask_strategy_combobox = QComboBox()
        self.mask_strategy_combobox.addItems(["Checkpoint default", "marginal", "independent"])
        self.mask_strategy_combobox.setToolTip(
            "'Checkpoint default' omits --mask_strategy so inference uses the checkpoint's hparams.yaml setting; "
            "'marginal' and 'independent' explicitly override the masking strategy."
        )
        param_row.addWidget(self.mask_strategy_combobox)

        param_row.addSpacing(8)
        param_row.addWidget(QLabel("Sigma (σ):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.0, 100.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setToolTip(
            "LoRA alpha scaling factor sigma (σ). 1.0 = standard fine-tuned model; 0.0 = zero-shot ESM3 backbone control."
        )
        param_row.addWidget(self.sigma_spinbox)
        param_row.addStretch()
        runtime_layout.addLayout(param_row)

        self.fix_noncanonical_checkbox = QCheckBox("Replace non-canonical residues")
        self.fix_noncanonical_checkbox.setChecked(True)
        self.fix_noncanonical_checkbox.setToolTip(
            "Replace non-canonical residues (e.g. MSE -> MET) via PDBFixer before inference. Uncheck to pass --skip_fix_noncanonical."
        )
        runtime_layout.addWidget(self.fix_noncanonical_checkbox)

        self.skip_reverse_checkbox = QCheckBox("Skip MT LoRA Pass (Additive Approximation)")
        self.skip_reverse_checkbox.setToolTip(
            "Pass --skip_reverse: fast single-pass WT LoRA approximation. "
            "Produces wt_lora_pred only (no mt_lora_pred, combined_pred, or epistasis)."
        )
        runtime_layout.addWidget(self.skip_reverse_checkbox)

        screening_config_layout.addWidget(runtime_group)
        layout.addWidget(screening_config_group)
        layout.addStretch()

    def _build_viz_tab(self, parent_widget):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        parent_widget.setLayout(layout)

        self.csv_label = QLabel("No CSV loaded.")
        self.csv_label.setWordWrap(True)
        self.csv_label.setStyleSheet(
            "background-color: #f4f6f8; border: 1px solid #d0d7de; border-radius: 3px; padding: 5px;"
        )
        layout.addWidget(self.csv_label)

        # 1. Target Configuration
        target_group = QGroupBox("Target Configuration (Synced with Screening)")
        target_layout = QHBoxLayout()
        target_layout.setSpacing(5)
        target_group.setLayout(target_layout)

        target_layout.addWidget(QLabel("Target Model:"))
        self.viz_model_combobox = QComboBox()
        self.viz_model_combobox.setSizePolicy(_SIZE_EXPANDING, _SIZE_FIXED)
        self.viz_model_combobox.currentIndexChanged.connect(self._on_viz_model_selected)
        target_layout.addWidget(self.viz_model_combobox, stretch=1)

        self.viz_refresh_models_button = QPushButton("Refresh")
        self.viz_refresh_models_button.setToolTip("Refresh the list of open ChimeraX atomic models and chains.")
        self.viz_refresh_models_button.clicked.connect(lambda: self._refresh_models())
        target_layout.addWidget(self.viz_refresh_models_button)

        target_layout.addWidget(QLabel("Chain:"))
        self.viz_chain_id_combobox = QComboBox()
        self.viz_chain_id_combobox.setEditable(True)
        self.viz_chain_id_combobox.setMaximumWidth(75)
        self.viz_chain_id_combobox.currentTextChanged.connect(self._on_viz_chain_changed)
        target_layout.addWidget(self.viz_chain_id_combobox)

        layout.addWidget(target_group)

        # 2. Display Mode and Priority
        mode_group = QGroupBox("Display Mode and Priority")
        mode_grid = QGridLayout()
        mode_grid.setSpacing(5)
        mode_grid.setColumnStretch(1, 1)
        mode_grid.setColumnStretch(3, 1)
        mode_group.setLayout(mode_grid)

        mode_grid.addWidget(QLabel("Mode:"), 0, 0)
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setSizeAdjustPolicy(_COMBO_ADJUST_MIN)
        self.display_mode_combo.setMinimumContentsLength(7)
        self.display_mode_combo.addItems(["Singles", "WT epistasis", "MT epistasis"])
        self.display_mode_combo.setToolTip(
            "Singles: top single mutant per residue. "
            "WT epistasis: Ala/Gly double-mutant scan mapped on WT sidechains. "
            "MT epistasis: mutated double-mutant sidechains with epistasis pseudobonds."
        )
        self.display_mode_combo.currentTextChanged.connect(self._on_display_mode_changed)
        mode_grid.addWidget(self.display_mode_combo, 0, 1)

        mode_grid.addWidget(QLabel("Select Pairs By:"), 0, 2)
        self.pair_selection_combo = QComboBox()
        self.pair_selection_combo.setSizeAdjustPolicy(_COMBO_ADJUST_MIN)
        self.pair_selection_combo.setMinimumContentsLength(7)
        self.pair_selection_combo.addItems(["Epistasis", "Stability"])
        self.pair_selection_combo.setToolTip(
            "Threshold and rank double-mutant pairs by their epistasis score or total stability (combined_pred)."
        )
        mode_grid.addWidget(self.pair_selection_combo, 0, 3)

        mode_grid.addWidget(QLabel("Display Priority:"), 1, 0)
        self.display_priority_combo = QComboBox()
        self.display_priority_combo.setSizeAdjustPolicy(_COMBO_ADJUST_MIN)
        self.display_priority_combo.setMinimumContentsLength(7)
        self.display_priority_combo.addItems(["High score", "Low score", "Magnitude"])
        self.display_priority_combo.setToolTip("Ordering used when selecting up to Max Interactions per Position.")
        mode_grid.addWidget(self.display_priority_combo, 1, 1)

        mode_grid.addWidget(QLabel("Epistasis Metric:"), 1, 2)
        self.epi_metric_combo = QComboBox()
        self.epi_metric_combo.setSizeAdjustPolicy(_COMBO_ADJUST_MIN)
        self.epi_metric_combo.setMinimumContentsLength(7)
        self.epi_metric_combo.addItem("epi_pred (1× ensemble)", "epi_pred")
        self.epi_metric_combo.addItem("combined_dddg_pred", "combined_dddg_pred")
        self.epi_metric_combo.setToolTip(
            "epi_pred: authoritative 1x ensemble epistasis = comb(AB) - WT(AB) (default when present). "
            "combined_dddg_pred: additive-subtracted ΔΔΔG = comb(AB) - (comb(A) + comb(B)) (legacy fallback)."
        )
        mode_grid.addWidget(self.epi_metric_combo, 1, 3)

        layout.addWidget(mode_group)

        # 3. Global Filters
        filters_group = QGroupBox("Global Filters")
        filters_layout = QHBoxLayout()
        filters_layout.setSpacing(8)

        self.exclude_wt_cys_checkbox = QCheckBox("No WT Cys")
        self.exclude_wt_cys_checkbox.setToolTip("Exclude mutations whose wild-type residue is Cysteine (C).")
        self.exclude_wt_pro_checkbox = QCheckBox("No WT Pro")
        self.exclude_wt_pro_checkbox.setToolTip("Exclude mutations whose wild-type residue is Proline (P).")
        self.exclude_mut_cys_checkbox = QCheckBox("No Mut Cys")
        self.exclude_mut_cys_checkbox.setToolTip("Exclude mutations introducing Cysteine (C).")
        self.exclude_mut_pro_checkbox = QCheckBox("No Mut Pro")
        self.exclude_mut_pro_checkbox.setToolTip("Exclude mutations introducing Proline (P).")

        filters_layout.addWidget(self.exclude_wt_cys_checkbox)
        filters_layout.addWidget(self.exclude_wt_pro_checkbox)
        filters_layout.addWidget(self.exclude_mut_cys_checkbox)
        filters_layout.addWidget(self.exclude_mut_pro_checkbox)
        filters_layout.addStretch()

        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)

        # 4. Score Selection
        score_group = QGroupBox("Score Selection (Additive Base)")
        score_layout = QVBoxLayout()
        score_layout.setSpacing(3)
        score_group.setLayout(score_layout)

        self.viz_score_btn_group = QButtonGroup()

        self.radio_dual_view = QRadioButton("Dual-view predictions (recommended; required for epistasis)")
        self.radio_dual_view.setToolTip("Uses combined_pred (WT + MT dual-view ensemble).")
        self.viz_score_btn_group.addButton(self.radio_dual_view)
        score_layout.addWidget(self.radio_dual_view)

        self.radio_wt_lora = QRadioButton("WT LoRA predictions (additive)")
        self.radio_wt_lora.setToolTip("Uses wt_lora_pred (fast WT-only pass; always available including --skip_reverse).")
        self.viz_score_btn_group.addButton(self.radio_wt_lora)
        score_layout.addWidget(self.radio_wt_lora)

        self.radio_mt_lora = QRadioButton("MT LoRA predictions (not recommended)")
        self.radio_mt_lora.setToolTip("Uses mt_lora_pred (MT-only pass; not available with --skip_reverse).")
        self.viz_score_btn_group.addButton(self.radio_mt_lora)
        score_layout.addWidget(self.radio_mt_lora)

        self.radio_dual_view.setChecked(True)
        layout.addWidget(score_group)

        # 5. Thresholds, Contacts, and Networks
        tcn_group = QGroupBox("Global Thresholds and Networks (kcal/mol, positive = stabilizing)")
        tcn_layout = QVBoxLayout()
        tcn_layout.setSpacing(5)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Pos Threshold:"))
        self.pos_threshold_spinbox = QDoubleSpinBox()
        self.pos_threshold_spinbox.setRange(0.0, 1000.0)
        self.pos_threshold_spinbox.setSingleStep(0.05)
        self.pos_threshold_spinbox.setValue(0.0)
        self.pos_threshold_spinbox.setToolTip(
            "Display mutations/pairs with score >= Pos Threshold (kcal/mol; positive = stabilizing)."
        )
        thresh_row.addWidget(self.pos_threshold_spinbox)

        thresh_row.addSpacing(8)
        thresh_row.addWidget(QLabel("Neg Threshold:"))
        self.neg_threshold_spinbox = QDoubleSpinBox()
        self.neg_threshold_spinbox.setRange(-1000.0, 0.0)
        self.neg_threshold_spinbox.setSingleStep(0.05)
        self.neg_threshold_spinbox.setValue(0.0)
        self.neg_threshold_spinbox.setToolTip(
            "Display mutations/pairs with score <= Neg Threshold (kcal/mol; negative = destabilizing)."
        )
        thresh_row.addWidget(self.neg_threshold_spinbox)
        thresh_row.addStretch()
        tcn_layout.addLayout(thresh_row)

        nontarget_row = QHBoxLayout()
        nontarget_row.addWidget(QLabel("Non-Target Chain Transp %:"))
        self.non_target_alpha_spinbox = QSpinBox()
        self.non_target_alpha_spinbox.setRange(0, 100)
        self.non_target_alpha_spinbox.setSingleStep(10)
        self.non_target_alpha_spinbox.setValue(90)
        self.non_target_alpha_spinbox.setToolTip("Transparency (0-100%) applied to chains other than the Target Chain.")
        self.non_target_alpha_spinbox.valueChanged.connect(self._on_transp_changed)
        nontarget_row.addWidget(self.non_target_alpha_spinbox)

        self.hide_non_target_chains_checkbox = QCheckBox("Hide Fully")
        self.hide_non_target_chains_checkbox.setToolTip("Completely hide non-target chains in the active model.")
        self.hide_non_target_chains_checkbox.toggled.connect(self._on_hide_fully_toggled)
        nontarget_row.addWidget(self.hide_non_target_chains_checkbox)
        nontarget_row.addStretch()
        tcn_layout.addLayout(nontarget_row)

        net_row = QHBoxLayout()
        net_row.addWidget(QLabel("Max Interactions / Pos:"))
        self.epi_max_edges = QSpinBox()
        self.epi_max_edges.setRange(1, 20)
        self.epi_max_edges.setValue(1)
        self.epi_max_edges.setToolTip("Maximum number of epistasis pseudobond connections drawn per residue position.")
        net_row.addWidget(self.epi_max_edges)
        net_row.addStretch()
        tcn_layout.addLayout(net_row)

        self.color_backbone_checkbox = QCheckBox("Color Backbone by Highest Additive ΔΔG")
        self.color_backbone_checkbox.setChecked(False)
        self.color_backbone_checkbox.setToolTip("Color the target chain ribbon by each residue's best single-mutant ΔΔG.")
        tcn_layout.addWidget(self.color_backbone_checkbox)

        contacts_row = QHBoxLayout()
        self.show_nearby_checkbox = QCheckBox("Nearby atoms within:")
        self.show_nearby_checkbox.setChecked(False)
        self.show_nearby_checkbox.setToolTip("Highlight WT atoms and residues contacting displayed mutant sidechains (or Gly CA).")
        contacts_row.addWidget(self.show_nearby_checkbox)

        self.nearby_distance_spinbox = QDoubleSpinBox()
        self.nearby_distance_spinbox.setRange(0.0, 50.0)
        self.nearby_distance_spinbox.setSingleStep(0.5)
        self.nearby_distance_spinbox.setValue(3.0)
        self.nearby_distance_spinbox.setEnabled(False)
        contacts_row.addWidget(self.nearby_distance_spinbox)

        contacts_row.addWidget(QLabel("Å"))
        contacts_row.addSpacing(6)

        self.exclude_nearby_backbone_checkbox = QCheckBox("Exclude backbone")
        self.exclude_nearby_backbone_checkbox.setChecked(True)
        self.exclude_nearby_backbone_checkbox.setToolTip("Exclude N, CA, C, O, OXT atoms when finding contacting atoms.")
        contacts_row.addWidget(self.exclude_nearby_backbone_checkbox)
        contacts_row.addStretch()
        tcn_layout.addLayout(contacts_row)

        tcn_group.setLayout(tcn_layout)
        layout.addWidget(tcn_group)

        self.show_nearby_checkbox.toggled.connect(self._on_show_nearby_toggled)

        # 6. Consolidated Rendering and Styling (compact 6-column grid fits ~440px with zero clipping)
        styling_group = QGroupBox("Rendering and Styling")
        style_grid = QGridLayout()
        style_grid.setSpacing(4)
        style_grid.setColumnStretch(1, 1)
        styling_group.setLayout(style_grid)

        for col_idx, header in enumerate(["Element", "Color (name/#hex)", "", "Style", "Transp %", "Hide"]):
            lbl = QLabel(header)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            style_grid.addWidget(lbl, 0, col_idx)

        # WT row
        style_grid.addWidget(QLabel("WT:"), 1, 0)
        self.wt_color_edit = QLineEdit("white")
        self.wt_color_edit.setMinimumWidth(65)
        style_grid.addWidget(self.wt_color_edit, 1, 1)
        self.wt_color_btn = self._create_color_swatch(self.wt_color_edit, "white")
        style_grid.addWidget(self.wt_color_btn, 1, 2)
        self.wt_style_combo = QComboBox()
        self.wt_style_combo.addItems(["stick", "ball", "sphere", "wire"])
        style_grid.addWidget(self.wt_style_combo, 1, 3)
        self.wt_stick_alpha_spinbox = QSpinBox()
        self.wt_stick_alpha_spinbox.setRange(0, 100)
        self.wt_stick_alpha_spinbox.setValue(100)
        style_grid.addWidget(self.wt_stick_alpha_spinbox, 1, 4)
        self.wt_hide_checkbox = QCheckBox("")
        self.wt_hide_checkbox.setToolTip("Hide WT sidechains completely.")
        self.wt_hide_checkbox.setChecked(True)
        style_grid.addWidget(self.wt_hide_checkbox, 1, 5)

        # Mutant row
        style_grid.addWidget(QLabel("Mutant:"), 2, 0)
        self.mut_color_edit = QLineEdit("")
        self.mut_color_edit.setMinimumWidth(65)
        self.mut_color_edit.setPlaceholderText("Blank = score")
        self.mut_color_edit.setToolTip("Leave blank to color mutant sidechains by additive ΔΔG score; or enter a color name/#hex.")
        style_grid.addWidget(self.mut_color_edit, 2, 1)
        self.mut_color_btn = self._create_color_swatch(self.mut_color_edit, "")
        style_grid.addWidget(self.mut_color_btn, 2, 2)
        self.mut_style_combo = QComboBox()
        self.mut_style_combo.addItems(["stick", "ball", "sphere", "wire"])
        style_grid.addWidget(self.mut_style_combo, 2, 3)
        self.mut_stick_alpha_spinbox = QSpinBox()
        self.mut_stick_alpha_spinbox.setRange(0, 100)
        self.mut_stick_alpha_spinbox.setValue(0)
        style_grid.addWidget(self.mut_stick_alpha_spinbox, 2, 4)
        self.mut_hide_checkbox = QCheckBox("")
        self.mut_hide_checkbox.setToolTip("Hide mutant sidechains completely.")
        self.mut_hide_checkbox.setChecked(False)
        style_grid.addWidget(self.mut_hide_checkbox, 2, 5)

        # Nearby Atom row
        style_grid.addWidget(QLabel("Near Atom:"), 3, 0)
        self.nearby_atom_color_edit = QLineEdit("purple")
        self.nearby_atom_color_edit.setMinimumWidth(65)
        style_grid.addWidget(self.nearby_atom_color_edit, 3, 1)
        self.nearby_atom_color_btn = self._create_color_swatch(self.nearby_atom_color_edit, "purple")
        style_grid.addWidget(self.nearby_atom_color_btn, 3, 2)
        self.nearby_atom_style_combo = QComboBox()
        self.nearby_atom_style_combo.addItems(["ball", "sphere", "stick", "wire"])
        style_grid.addWidget(self.nearby_atom_style_combo, 3, 3)
        self.nearby_atom_alpha_spinbox = QSpinBox()
        self.nearby_atom_alpha_spinbox.setRange(0, 100)
        self.nearby_atom_alpha_spinbox.setValue(0)
        style_grid.addWidget(self.nearby_atom_alpha_spinbox, 3, 4)

        # Nearby Residue row
        style_grid.addWidget(QLabel("Near Res:"), 4, 0)
        self.nearby_res_color_edit = QLineEdit("grey")
        self.nearby_res_color_edit.setMinimumWidth(65)
        style_grid.addWidget(self.nearby_res_color_edit, 4, 1)
        self.nearby_res_color_btn = self._create_color_swatch(self.nearby_res_color_edit, "grey")
        style_grid.addWidget(self.nearby_res_color_btn, 4, 2)
        self.nearby_res_style_combo = QComboBox()
        self.nearby_res_style_combo.addItems(["stick", "ball", "sphere", "wire"])
        style_grid.addWidget(self.nearby_res_style_combo, 4, 3)
        self.nearby_res_alpha_spinbox = QSpinBox()
        self.nearby_res_alpha_spinbox.setRange(0, 100)
        self.nearby_res_alpha_spinbox.setValue(80)
        style_grid.addWidget(self.nearby_res_alpha_spinbox, 4, 4)
        self.nearby_hide_checkbox = QCheckBox("")
        self.nearby_hide_checkbox.setToolTip("Hide nearby contacting atoms and residues.")
        self.nearby_hide_checkbox.setChecked(False)
        style_grid.addWidget(self.nearby_hide_checkbox, 4, 5)

        layout.addWidget(styling_group)
        layout.addStretch()

        # Connections for Styling
        self.wt_stick_alpha_spinbox.valueChanged.connect(self._on_wt_transp_changed)
        self.wt_hide_checkbox.toggled.connect(self._on_wt_hide_toggled)
        self.mut_stick_alpha_spinbox.valueChanged.connect(self._on_mut_transp_changed)
        self.mut_hide_checkbox.toggled.connect(self._on_mut_hide_toggled)
        self.nearby_atom_alpha_spinbox.valueChanged.connect(self._on_nearby_transp_changed)
        self.nearby_res_alpha_spinbox.valueChanged.connect(self._on_nearby_transp_changed)
        self.nearby_hide_checkbox.toggled.connect(self._on_nearby_hide_toggled)

        self._on_display_mode_changed()
        self._on_show_nearby_toggled(self.show_nearby_checkbox.isChecked())

    # ---------------- Validation & UI State Sync -----------------
    def _on_checkpoint_text_edited(self, _text):
        """Clear LoRA config when the user edits the checkpoint path and sync QSettings."""
        if self.config_file_edit.text():
            self.config_file_edit.clear()
        self.config_file = ""
        self.settings.setValue("config_file", "")

    def _validate_file_paths(self):
        if self._closing or not hasattr(self, "file_warning_label"):
            return
        warnings = []
        base_repo = self.base_repo_path_edit.text().strip() if hasattr(self, "base_repo_path_edit") else self.base_repo_path
        ckpt_raw = self.checkpoint_path_edit.text().strip()
        cfg_raw = self.config_file_edit.text().strip()

        ckpt = resolve_user_path(ckpt_raw, base_repo) if ckpt_raw else ""
        cfg = resolve_user_path(cfg_raw, base_repo) if cfg_raw else ""

        if ckpt and not os.path.exists(ckpt):
            warnings.append("Checkpoint file does not exist.")
        if cfg and not os.path.exists(cfg):
            warnings.append("Config file does not exist.")
        if not ckpt and not cfg:
            warnings.append("No checkpoint or LoRA config set (zero-shot requires a LoRA Config, or set Sigma = 0.0 with a checkpoint).")

        if warnings:
            self.file_warning_label.setText("WARNING: " + " | ".join(warnings))
            self.file_warning_label.setVisible(True)
        else:
            self.file_warning_label.setText("")
            self.file_warning_label.setVisible(False)

    def _refresh_load_last_output_button(self):
        if not hasattr(self, "load_last_output_button"):
            return
        candidate = self._get_last_output_candidate()
        self.load_last_output_button.setEnabled(bool(candidate))

    def _get_last_output_candidate(self):
        if self.predicted_output_path and os.path.isfile(self.predicted_output_path):
            return self.predicted_output_path
        if hasattr(self, "script_output_csv_path_edit"):
            raw = self.script_output_csv_path_edit.text().strip()
            if raw:
                base_repo = self.base_repo_path_edit.text().strip() if hasattr(self, "base_repo_path_edit") else self.base_repo_path
                resolved = resolve_user_path(raw, base_repo)
                if os.path.isfile(resolved):
                    return resolved
        return ""

    def _update_scope_ui(self):
        is_full = self.radio_full.isChecked()
        self.mode_combobox.setEnabled(is_full)
        self.selected_residues_edit.setEnabled(is_full)
        self.grab_sel_button.setEnabled(is_full)
        self.screen_except_checkbox.setEnabled(is_full)

        current_mode = self.mode_combobox.currentText()
        is_doubles_mode = (current_mode == "singles+doubles")
        should_enable_dist = is_full and is_doubles_mode

        self.enable_distance_checkbox.setEnabled(should_enable_dist)
        if not should_enable_dist:
            self.enable_distance_checkbox.setChecked(False)
            self.distance_threshold_spinbox.setEnabled(False)
        else:
            self.distance_threshold_spinbox.setEnabled(self.enable_distance_checkbox.isChecked())

        is_csv = self.radio_csv.isChecked()
        self.subset_df_edit.setEnabled(is_csv)
        self.subset_df_btn.setEnabled(is_csv)

        is_explicit = self.radio_explicit.isChecked()
        self.mutations_edit.setEnabled(is_explicit)

    def _toggle_distance_spinbox(self, checked):
        self.distance_threshold_spinbox.setEnabled(checked)

    def _on_transp_changed(self, val):
        self.hide_non_target_chains_checkbox.blockSignals(True)
        self.hide_non_target_chains_checkbox.setChecked(val == 100)
        self.hide_non_target_chains_checkbox.blockSignals(False)

    def _on_hide_fully_toggled(self, checked):
        if checked:
            self.non_target_alpha_spinbox.blockSignals(True)
            self.non_target_alpha_spinbox.setValue(100)
            self.non_target_alpha_spinbox.blockSignals(False)

    def _on_wt_transp_changed(self, val):
        self.wt_hide_checkbox.blockSignals(True)
        self.wt_hide_checkbox.setChecked(val == 100)
        self.wt_hide_checkbox.blockSignals(False)

    def _on_wt_hide_toggled(self, checked):
        if checked:
            self.wt_stick_alpha_spinbox.blockSignals(True)
            self.wt_stick_alpha_spinbox.setValue(100)
            self.wt_stick_alpha_spinbox.blockSignals(False)

    def _on_mut_transp_changed(self, val):
        self.mut_hide_checkbox.blockSignals(True)
        self.mut_hide_checkbox.setChecked(val == 100)
        self.mut_hide_checkbox.blockSignals(False)

    def _on_mut_hide_toggled(self, checked):
        if checked:
            self.mut_stick_alpha_spinbox.blockSignals(True)
            self.mut_stick_alpha_spinbox.setValue(100)
            self.mut_stick_alpha_spinbox.blockSignals(False)

    def _on_nearby_transp_changed(self, _val):
        all_hidden = (
            self.nearby_atom_alpha_spinbox.value() == 100
            and self.nearby_res_alpha_spinbox.value() == 100
        )
        self.nearby_hide_checkbox.blockSignals(True)
        self.nearby_hide_checkbox.setChecked(all_hidden)
        self.nearby_hide_checkbox.blockSignals(False)

    def _on_nearby_hide_toggled(self, checked):
        if checked:
            self.nearby_atom_alpha_spinbox.blockSignals(True)
            self.nearby_atom_alpha_spinbox.setValue(100)
            self.nearby_atom_alpha_spinbox.blockSignals(False)
            self.nearby_res_alpha_spinbox.blockSignals(True)
            self.nearby_res_alpha_spinbox.setValue(100)
            self.nearby_res_alpha_spinbox.blockSignals(False)

    def _on_show_nearby_toggled(self, checked):
        self.nearby_distance_spinbox.setEnabled(checked)
        self.exclude_nearby_backbone_checkbox.setEnabled(checked)
        self.nearby_atom_color_edit.setEnabled(checked)
        self.nearby_atom_color_btn.setEnabled(checked)
        self.nearby_atom_style_combo.setEnabled(checked)
        self.nearby_atom_alpha_spinbox.setEnabled(checked)
        self.nearby_res_color_edit.setEnabled(checked)
        self.nearby_res_color_btn.setEnabled(checked)
        self.nearby_res_style_combo.setEnabled(checked)
        self.nearby_res_alpha_spinbox.setEnabled(checked)
        self.nearby_hide_checkbox.setEnabled(checked)

    # ---------------- UI Callbacks -----------------
    def _grab_selection(self):
        try:
            atoms = selected_atoms(self.session)
            if atoms is None or len(atoms) == 0:
                self.selected_residues_edit.setText("")
                self._set_status("No residues selected.", level="warning")
                return

            target_model_id = self.pred_model_combobox.currentData()
            target_chain = self.pred_chain_id_combobox.currentText().strip()

            pos_keys = set()
            other_count = 0
            for res in atoms.residues:
                struct_id = getattr(getattr(res, "structure", None), "id_string", None)
                if target_model_id and struct_id and struct_id != target_model_id:
                    other_count += 1
                    continue
                if target_chain and res.chain_id != target_chain:
                    other_count += 1
                    continue
                pos_keys.add((res.number, normalize_icode(getattr(res, "insertion_code", ""))))

            if not pos_keys:
                self.selected_residues_edit.setText("")
                scope_desc = f"model #{target_model_id}" if target_model_id else "target model"
                if target_chain:
                    scope_desc += f" chain {target_chain}"
                self._set_status(
                    f"No selected residues in {scope_desc} ({other_count} residue(s) selected in other model/chain).",
                    level="warning",
                )
                return

            ordered = sorted(pos_keys, key=lambda pk: (pk[0], pk[1]))
            formatted = [format_res_pos(pk) for pk in ordered]
            self.selected_residues_edit.setText(",".join(formatted))
            extra = f" (ignored {other_count} outside target model/chain)" if other_count else ""
            self._set_status(f"Grabbed {len(formatted)} unique residue position(s){extra}.", level="success")
        except Exception as e:
            self.session.logger.warning(f"Could not grab selection: {e}")
            self._set_status("Failed to grab selection.", level="error")

    def _is_generated_clone(self, model):
        if not self.mutated_model_id_strings:
            return False
        mid = model.id_string
        top_id = mid.split(".")[0]
        return mid in self.mutated_model_id_strings or top_id in self.mutated_model_id_strings

    def _refresh_models(self, trigger_name=None, trigger_data=None):
        if self._closing or self._suppress_model_refresh:
            return

        try:
            current_model_id = self.pred_model_combobox.currentData()

            self.pred_model_combobox.blockSignals(True)
            self.pred_model_combobox.clear()

            models = [
                m for m in self.session.models.list(type=Structure)
                if not self._is_generated_clone(m)
            ]

            for m in models:
                self.pred_model_combobox.addItem(f"#{m.id_string} {m.name}", m.id_string)

            idx = self.pred_model_combobox.findData(current_model_id)
            if idx >= 0:
                self.pred_model_combobox.setCurrentIndex(idx)
            elif self.pred_model_combobox.count() > 0:
                self.pred_model_combobox.setCurrentIndex(0)

            self.pred_model_combobox.blockSignals(False)

            if hasattr(self, "viz_model_combobox"):
                self.viz_model_combobox.blockSignals(True)
                self.viz_model_combobox.clear()
                for i in range(self.pred_model_combobox.count()):
                    self.viz_model_combobox.addItem(
                        self.pred_model_combobox.itemText(i),
                        self.pred_model_combobox.itemData(i),
                    )
                self.viz_model_combobox.setCurrentIndex(self.pred_model_combobox.currentIndex())
                self.viz_model_combobox.blockSignals(False)

            self._on_model_selected()

        except RuntimeError as e:
            if "deleted" in str(e).lower():
                self.session.logger.warning("Model refresh trigger skipped: UI widget already deleted.")
                return
            self.session.logger.error(f"Error refreshing models: {e}")

    def _on_model_selected(self, index=None):
        if self._closing:
            return

        current_chain = self.pred_chain_id_combobox.currentText()
        self.pred_chain_id_combobox.blockSignals(True)
        self.pred_chain_id_combobox.clear()

        model_id = self.pred_model_combobox.currentData()
        if not model_id:
            self.pred_chain_id_combobox.blockSignals(False)
            if hasattr(self, "viz_chain_id_combobox"):
                self.viz_chain_id_combobox.blockSignals(True)
                self.viz_chain_id_combobox.clear()
                self.viz_chain_id_combobox.blockSignals(False)
            return

        models = self.session.models.list(type=Structure)
        target_model = next((m for m in models if m.id_string == model_id), None)

        chains = set()
        if target_model:
            for c in target_model.chains:
                chains.add(c.chain_id)

        if chains:
            self.pred_chain_id_combobox.addItems(sorted(chains))
        else:
            self.pred_chain_id_combobox.addItems(["A"])

        if current_chain in chains:
            self.pred_chain_id_combobox.setCurrentText(current_chain)

        self.pred_chain_id_combobox.blockSignals(False)

        if hasattr(self, "viz_model_combobox"):
            idx = self.pred_model_combobox.currentIndex()
            if self.viz_model_combobox.currentIndex() != idx:
                self.viz_model_combobox.blockSignals(True)
                self.viz_model_combobox.setCurrentIndex(idx)
                self.viz_model_combobox.blockSignals(False)

            self.viz_chain_id_combobox.blockSignals(True)
            self.viz_chain_id_combobox.clear()
            self.viz_chain_id_combobox.addItems(
                [self.pred_chain_id_combobox.itemText(i) for i in range(self.pred_chain_id_combobox.count())]
            )
            self.viz_chain_id_combobox.setCurrentText(self.pred_chain_id_combobox.currentText())
            self.viz_chain_id_combobox.blockSignals(False)

    def _on_viz_model_selected(self, index=None):
        if self._closing:
            return
        idx = self.viz_model_combobox.currentIndex()
        if self.pred_model_combobox.currentIndex() != idx:
            self.pred_model_combobox.blockSignals(True)
            self.pred_model_combobox.setCurrentIndex(idx)
            self.pred_model_combobox.blockSignals(False)
            self._on_model_selected()

    def _on_pred_chain_changed(self, text):
        if self._closing:
            return
        if hasattr(self, "viz_chain_id_combobox") and self.viz_chain_id_combobox.currentText() != text:
            self.viz_chain_id_combobox.blockSignals(True)
            self.viz_chain_id_combobox.setCurrentText(text)
            self.viz_chain_id_combobox.blockSignals(False)

    def _on_viz_chain_changed(self, text):
        if self._closing:
            return
        if self.pred_chain_id_combobox.currentText() != text:
            self.pred_chain_id_combobox.blockSignals(True)
            self.pred_chain_id_combobox.setCurrentText(text)
            self.pred_chain_id_combobox.blockSignals(False)

    def _browse_base_repo(self):
        w = getattr(self.session.ui, "main_window", None)
        folder_path = QFileDialog.getExistingDirectory(w, "Select Base esm-msr Repository Folder", self.base_repo_path)
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            self.base_repo_path = folder_path
            self.base_repo_path_edit.setText(folder_path)
            self.settings.setValue("base_repo_path", folder_path)
            self.session.logger.info(f"Base repository path persistently saved: {folder_path}")
            self._update_paths_from_base_repo(folder_path, is_init=False)

    def _update_paths_from_base_repo(self, base_path, is_init=False):
        if not base_path:
            self.python_script_path = ""
            if not is_init:
                self.checkpoint_path = ""
                self.script_output_csv_path = ""
                self.python_env = ""
        else:
            self.python_script_path = os.path.normpath(os.path.join(base_path, "src", "esm_msr", "inference.py"))
            if not is_init or not self.checkpoint_path:
                self.checkpoint_path = os.path.normpath(
                    os.path.join(base_path, "LoRA_models", "esm-msr-small", "lora_seed1.safetensors")
                )
            if not is_init or not self.script_output_csv_path:
                self.script_output_csv_path = os.path.normpath(os.path.join(base_path, "example_inference.csv"))
            if not is_init or not self.python_env:
                self.python_env = os.path.normpath(os.path.join(base_path, "msr_venv"))

        if hasattr(self, "checkpoint_path_edit"):
            self.checkpoint_path_edit.setText(self.checkpoint_path)
        if hasattr(self, "script_output_csv_path_edit"):
            self.script_output_csv_path_edit.setText(self.script_output_csv_path)
        if hasattr(self, "python_env_edit"):
            self.python_env_edit.setText(self.python_env)

    # ---------------- Generic File Browsers -----------------
    def _browse_script_output_csv(self):
        w = getattr(self.session.ui, "main_window", None)
        fp, _ = QFileDialog.getSaveFileName(
            w, "Specify Prediction Output CSV File Path",
            self.script_output_csv_path_edit.text(), "CSV Files (*.csv);;All Files (*)",
        )
        if fp:
            if not fp.lower().endswith(".csv"):
                fp += ".csv"
            norm_fp = os.path.normpath(fp)
            self.script_output_csv_path = norm_fp
            self.script_output_csv_path_edit.setText(norm_fp)
            self.settings.setValue("script_output_csv_path", norm_fp)

    def _browse_checkpoint_path(self):
        w = getattr(self.session.ui, "main_window", None)
        fp, _ = QFileDialog.getOpenFileName(
            w, "Select Checkpoint File",
            self.checkpoint_path_edit.text(), "Checkpoint Files (*.ckpt *.safetensors);;All Files (*)",
        )
        if fp:
            norm_fp = os.path.normpath(fp)
            self.checkpoint_path_edit.setText(norm_fp)
            self.settings.setValue("checkpoint_path", norm_fp)
            self.config_file_edit.clear()
            self.config_file = ""
            self.settings.setValue("config_file", "")

    def _browse_python_env(self):
        w = getattr(self.session.ui, "main_window", None)
        folder_path = QFileDialog.getExistingDirectory(
            w, "Select Python/Conda Environment Folder", self.python_env_edit.text()
        )
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            self.python_env_edit.setText(folder_path)
            self.python_env = folder_path
            self.settings.setValue("python_env", folder_path)

    def _browse_subset_df(self):
        w = getattr(self.session.ui, "main_window", None)
        fp, _ = QFileDialog.getOpenFileName(
            w, "Select Subset DataFrame CSV", self.base_repo_path, "CSV Files (*.csv);;All Files (*)"
        )
        if fp:
            self.subset_df_edit.setText(os.path.normpath(fp))

    def _browse_generic(self, line_edit, title, filter_str, settings_key=None):
        w = getattr(self.session.ui, "main_window", None)
        fp, _ = QFileDialog.getOpenFileName(w, title, self.base_repo_path, filter_str)
        if fp:
            norm_fp = os.path.normpath(fp)
            line_edit.setText(norm_fp)
            if settings_key:
                self.settings.setValue(settings_key, norm_fp)

    def _browse_generic_dir(self, line_edit, title, settings_key=None):
        w = getattr(self.session.ui, "main_window", None)
        folder_path = QFileDialog.getExistingDirectory(w, title, self.base_repo_path)
        if folder_path:
            norm_fp = os.path.normpath(folder_path)
            line_edit.setText(norm_fp)
            if settings_key:
                self.settings.setValue(settings_key, norm_fp)

    # ---------------- Subprocess Management -----------------
    def _reset_run_controls(self):
        if self._closing:
            return
        self.run_prediction_button.setVisible(True)
        self.run_prediction_button.setEnabled(True)
        self.stop_prediction_button.setVisible(False)
        self.stop_prediction_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._refresh_load_last_output_button()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def _stop_prediction_script(self, silent=False):
        if self.proc and self.proc.state() != _PROC_NOT_RUNNING:
            self._user_stopped = True
            if not silent:
                self.session.logger.warning("Terminating prediction process tree...")
                self._set_status("Terminating prediction process...", level="warning")

            try:
                pid = self.proc.processId()
                # CRITICAL P0 GUARD: never send kill -9 0 (which would SIGKILL ChimeraX's own process group!)
                if pid and pid > 0:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                    else:
                        subprocess.run(["pkill", "-9", "-P", str(pid)], capture_output=True)
                        subprocess.run(["kill", "-9", str(pid)], capture_output=True)

                self.proc.kill()
                self.proc.waitForFinished(1000)
            except Exception as e:
                self.session.logger.error(f"Failed while stopping prediction process: {e}")
            finally:
                self._cleanup_temp_dir()
                if not silent:
                    self._reset_run_controls()
                    self.prediction_output_label.setText("Predicted output file: Terminated")
                    self._set_status("Execution stopped by user.", level="warning")

    def _initiate_run_prediction_script(self):
        self.session.logger.info("****** Running prediction ******")

        if self.proc and self.proc.state() != _PROC_NOT_RUNNING:
            self._set_status("Prediction is already running.", level="warning")
            self.session.logger.warning("Prediction already running; ignoring new request.")
            return

        # Save current user fields to QSettings
        raw_repo = self.base_repo_path_edit.text().strip()
        raw_env = self.python_env_edit.text().strip()
        raw_out_csv = self.script_output_csv_path_edit.text().strip()
        raw_ckpt = self.checkpoint_path_edit.text().strip()
        raw_cfg = self.config_file_edit.text().strip()
        raw_base_loc = self.base_model_loc_edit.text().strip()

        self.settings.setValue("python_env", raw_env)
        self.settings.setValue("base_repo_path", raw_repo)
        self.settings.setValue("script_output_csv_path", raw_out_csv)
        self.settings.setValue("checkpoint_path", raw_ckpt)
        self.settings.setValue("config_file", raw_cfg)
        self.settings.setValue("base_model_loc", raw_base_loc)

        # ---- PRE-FLIGHT VALIDATION (all checks BEFORE creating temp dirs or exporting PDB) ----
        if not raw_repo or not os.path.isdir(os.path.expanduser(raw_repo)):
            self._report_error(
                "Invalid Repository Path",
                "Base Repo Dir does not exist. Select the root directory of the esm-msr repository.",
                popup=True,
            )
            return

        self.base_repo_path = os.path.normpath(os.path.abspath(os.path.expanduser(raw_repo)))
        self.python_script_path = os.path.normpath(os.path.join(self.base_repo_path, "src", "esm_msr", "inference.py"))
        if not os.path.isfile(self.python_script_path):
            self._report_error(
                "Inference Script Missing",
                f"Could not find inference script at '{self.python_script_path}'. Check Base Repo Dir.",
                popup=True,
            )
            return

        if not raw_out_csv:
            self._report_error(
                "Missing Output CSV",
                "An Output CSV path is required to run inference.",
                popup=True,
            )
            return
        resolved_out_csv = resolve_user_path(raw_out_csv, self.base_repo_path)
        out_dir = os.path.dirname(resolved_out_csv)
        if out_dir and not os.path.isdir(out_dir):
            self._report_error(
                "Invalid Output Directory",
                f"Output CSV parent directory does not exist: '{out_dir}'.",
                popup=True,
            )
            return

        # Resolve Python executable / conda environment
        self.python_env = raw_env
        program, base_args, env_err = resolve_python_command(raw_env, self.base_repo_path)
        if env_err or not program:
            self._report_error("Invalid Python Environment", env_err or "Could not resolve Python executable.", popup=True)
            return

        # Resolve & validate checkpoint / config (P0 #3 zero-shot safety)
        resolved_ckpt = resolve_user_path(raw_ckpt, self.base_repo_path) if raw_ckpt else ""
        resolved_cfg = resolve_user_path(raw_cfg, self.base_repo_path) if raw_cfg else ""

        if resolved_ckpt and not os.path.isfile(resolved_ckpt):
            self._report_error("Missing Checkpoint", f"Checkpoint file does not exist: '{resolved_ckpt}'.", popup=True)
            return
        if resolved_cfg and not os.path.isfile(resolved_cfg):
            self._report_error("Missing Config File", f"LoRA config file does not exist: '{resolved_cfg}'.", popup=True)
            return

        if not resolved_ckpt:
            if not resolved_cfg:
                self._report_error(
                    "Zero-Shot Configuration Required",
                    "No checkpoint was provided. To run zero-shot inference, either:\n"
                    "  1. Provide a LoRA Config (JSON/YAML, e.g. hparams.yaml) and set Sigma (σ) = 0.0, OR\n"
                    "  2. Select a checkpoint and set Sigma (σ) = 0.0 for the zero-shot ESM3 backbone control.",
                    popup=True,
                )
                return
            self.session.logger.warning(
                "No checkpoint provided; omitting --checkpoint_path for zero-shot / uninitialized weights mode "
                "(tip: set Sigma = 0.0 for pure zero-shot ESM3 backbone control)."
            )
        elif not resolved_cfg:
            sibling_hparams = os.path.join(os.path.dirname(resolved_ckpt), "hparams.yaml")
            if not os.path.isfile(sibling_hparams):
                self._report_error(
                    "Missing LoRA Config",
                    f"No LoRA Config was specified and 'hparams.yaml' was not found alongside '{resolved_ckpt}'. "
                    "Please specify a LoRA Config (JSON/YAML).",
                    popup=True,
                )
                return

        resolved_base_loc = resolve_user_path(raw_base_loc, self.base_repo_path) if raw_base_loc else ""
        if resolved_base_loc and not os.path.exists(resolved_base_loc):
            self._report_error(
                "Invalid ESM3 Weights Location",
                f"Specified ESM3 Weights path does not exist: '{resolved_base_loc}'.",
                popup=True,
            )
            return

        # Scope-specific validation
        needs_pdb_export = self.radio_full.isChecked() or self.radio_explicit.isChecked()
        current_model = None
        target_chain = self.pred_chain_id_combobox.currentText().strip()
        resolved_subset_df = ""
        mutations_val = ""

        if needs_pdb_export:
            model_id = self.pred_model_combobox.currentData()
            models = self.session.models.list(type=Structure)
            current_model = next((m for m in models if m.id_string == model_id), None)
            if not current_model:
                self._report_error(
                    "No Target Model",
                    "Selected target model not found. Open a structure in ChimeraX and click Refresh.",
                    popup=True,
                )
                return
            if not target_chain:
                self._report_error("Missing Chain", "Please specify a Target Chain before running inference.", popup=True)
                return
            if self.radio_explicit.isChecked():
                mutations_val = self.mutations_edit.text().strip()
                if not mutations_val:
                    self._report_error(
                        "Missing Mutations",
                        "Direct Mutations mode is selected, but no mutations were entered (e.g. A12C,A12C:D15E).",
                        popup=True,
                    )
                    return
        elif self.radio_csv.isChecked():
            raw_subset = self.subset_df_edit.text().strip()
            if not raw_subset:
                self._report_error("Missing Input CSV", "Input CSV mode is selected, but no file path was provided.", popup=True)
                return
            resolved_subset_df = resolve_user_path(raw_subset, self.base_repo_path)
            if not os.path.isfile(resolved_subset_df):
                self._report_error("Input CSV Not Found", f"Input CSV file does not exist: '{resolved_subset_df}'.", popup=True)
                return

        # ---- ALL VALIDATIONS PASSED: CREATE TEMP DIR / WORKDIR AND EXPORT PDB ----
        self._cleanup_temp_dir()
        try:
            self._temp_dir_to_cleanup = tempfile.mkdtemp(prefix="chimerax_esm_msr_run_")
            run_workdir = os.path.join(self._temp_dir_to_cleanup, "work")
            os.makedirs(run_workdir, exist_ok=True)

            self.script_input_structure_path = ""
            if needs_pdb_export and current_model is not None:
                safe_id = current_model.id_string.replace(":", "_").replace("/", "_")
                temp_model_filename = f"current_model_input_{safe_id}.pdb"
                temp_pdb_path = os.path.join(self._temp_dir_to_cleanup, temp_model_filename)
                run(self.session, f'save "{temp_pdb_path}" models #{current_model.id_string} format pdb')
                self.script_input_structure_path = temp_pdb_path
        except Exception as e:
            self._cleanup_temp_dir()
            self._report_error("PDB Export Failed", f"Failed to export temporary PDB: {e}", popup=True)
            return

        # ---- BUILD CLI ARGUMENTS ----
        script_args = [self.python_script_path, "--output_csv", resolved_out_csv]

        # P0 #3: omit --checkpoint_path entirely when empty
        if resolved_ckpt:
            script_args += ["--checkpoint_path", resolved_ckpt]

        if self.radio_full.isChecked():
            script_args += [
                "--pdb_file", self.script_input_structure_path,
                "--code", os.path.splitext(current_model.name)[0] if current_model.name else "protein",
                "--chain", target_chain,
                "--mode", self.mode_combobox.currentText(),
            ]
            selected_res = self.selected_residues_edit.text().strip()
            if selected_res:
                flag = "--screen_residues_except" if self.screen_except_checkbox.isChecked() else "--screen_residues"
                script_args += [flag, selected_res]
        elif self.radio_csv.isChecked():
            script_args += ["--input_csv", resolved_subset_df]
        elif self.radio_explicit.isChecked():
            script_args += [
                "--pdb_file", self.script_input_structure_path,
                "--code", os.path.splitext(current_model.name)[0] if current_model.name else "protein",
                "--chain", target_chain,
                "--mutations", mutations_val,
            ]

        script_args += [
            "--batch_size", str(self.batch_size_spinbox.value()),
            "--device", self.device_combobox.currentText(),
            "--model_dtype", self.model_dtype_combobox.currentText(),
            "--sigma", str(self.sigma_spinbox.value()),
        ]

        mask_strat = self.mask_strategy_combobox.currentText()
        if mask_strat != "Checkpoint default":
            script_args += ["--mask_strategy", mask_strat]

        if not self.fix_noncanonical_checkbox.isChecked():
            script_args += ["--skip_fix_noncanonical"]

        if self.enable_distance_checkbox.isChecked():
            script_args += ["--distance_threshold", str(self.distance_threshold_spinbox.value())]
            script_args += ["--calculate_distances"]
        else:
            script_args += ["--distance_threshold", "-1"]

        if resolved_base_loc:
            script_args += ["--base_model_loc", resolved_base_loc]

        if resolved_cfg:
            if resolved_cfg.lower().endswith(".json"):
                script_args += ["--lora_config", resolved_cfg]
            elif resolved_cfg.lower().endswith((".yaml", ".yml")):
                script_args += ["--hparams_path", resolved_cfg]
            else:
                self.session.logger.warning(f"Unrecognized config extension for {resolved_cfg}; passing via --hparams_path.")
                script_args += ["--hparams_path", resolved_cfg]

        if self.skip_reverse_checkbox.isChecked():
            script_args += ["--skip_reverse"]

        # P1 #9: ALWAYS preserve base_args (including '-u') regardless of program name!
        full_args = list(base_args) + script_args

        # Log exact command for reproducibility
        cmd_str = " ".join(shlex.quote(x) for x in [program] + full_args)
        self.session.logger.info(f"Working directory: {run_workdir}")
        self.session.logger.info(f"Launching ESM-MSR command: {cmd_str}")

        self._active_run_output_csv = resolved_out_csv
        self._user_stopped = False

        self.proc = QProcess()
        self.proc.setProcessChannelMode(_PROC_MERGED)
        # P0 #4: writable temp working directory so preprocess.py's temp_fixed.pdb never fails or pollutes repo
        self.proc.setWorkingDirectory(run_workdir)

        # Inject PYTHONPATH to support esm_msr absolute imports
        process_env = QProcessEnvironment.systemEnvironment()
        src_path = os.path.normpath(os.path.join(self.base_repo_path, "src"))
        current_pythonpath = process_env.value("PYTHONPATH", "")
        new_pythonpath = f"{src_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else src_path
        process_env.insert("PYTHONPATH", new_pythonpath)
        process_env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(process_env)

        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.errorOccurred.connect(self._on_proc_error)
        self.proc.finished.connect(self._on_proc_finished)

        self._set_status("Initializing ESM-MSR model and running inference...", level="info")
        self.run_prediction_button.setVisible(False)
        self.stop_prediction_button.setVisible(True)
        self.load_button.setEnabled(False)
        self.load_last_output_button.setEnabled(False)
        self.prediction_output_label.setText("Predicted output file: Processing...")

        # Indeterminate progress bar during model initialization
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

        self.proc.start(program, full_args)

    # ---------------- QProcess Slots -----------------
    def _on_proc_output(self):
        if self._closing or self.proc is None:
            return
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if not out:
                return

            for chunk in out.replace("\r", "\n").split("\n"):
                clean_chunk = chunk.strip()
                if not clean_chunk:
                    continue

                pct = parse_tqdm_progress(clean_chunk)
                if pct is not None:
                    if self.progress_bar.maximum() == 0:
                        self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(pct)

                if "|" in clean_chunk and ("it/s" in clean_chunk or "s/it" in clean_chunk or "%" in clean_chunk):
                    self._set_status(clean_chunk, level="info")
                else:
                    self.session.logger.info(clean_chunk)
        except Exception as e:
            self.session.logger.error(f"Error reading process output: {e}")

    def _on_proc_error(self, err):
        self.session.logger.error(f"Prediction process encountered QProcess error: {err}")
        self._set_status(f"Process Error ({err}). Check ChimeraX Log.", level="error")
        self._reset_run_controls()
        self.prediction_output_label.setText("Predicted output file: Failed to start / error")

        # Note: finished() is NOT emitted on FailedToStart, so clean up temp dir & proc when NotRunning.
        if self.proc is None or self.proc.state() == _PROC_NOT_RUNNING:
            self._cleanup_temp_dir()
            if self.proc is not None:
                try:
                    self.proc.deleteLater()
                except Exception:
                    pass
                self.proc = None

    def _on_proc_finished(self, exitCode, exitStatus):
        self._cleanup_temp_dir()

        if exitStatus == _PROC_NORMAL_EXIT and exitCode == 0:
            out_path = self._active_run_output_csv or resolve_user_path(
                self.script_output_csv_path_edit.text().strip(), self.base_repo_path
            )
            self.predicted_output_path = out_path
            self._reset_run_controls()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(True)
            self.prediction_output_label.setText(f"Predicted output: {os.path.basename(out_path)}")
            self._set_status("Prediction completed. Click 'Load Last Output' or 'Load CSV' to visualize.", level="success")
        elif self._user_stopped:
            self._reset_run_controls()
            self.prediction_output_label.setText("Predicted output file: Terminated")
            self._set_status("Execution stopped by user.", level="warning")
        else:
            self._reset_run_controls()
            self.prediction_output_label.setText("Predicted output file: Failed/Terminated")
            self._set_status(f"Script error or stopped (exit={exitCode}). Check ChimeraX Log.", level="error")
        self._user_stopped = False

        if self.proc is not None:
            try:
                self.proc.deleteLater()
            except Exception as e:
                self.session.logger.error(f"Failed cleaning up QProcess: {e}")
            self.proc = None

    # ---------------- Visualization Controls Sync -----------------
    def _set_combobox_item_enabled(self, combo, index, enabled):
        model = combo.model()
        if model is not None and hasattr(model, "item"):
            item = model.item(index)
            if item is not None:
                item.setEnabled(enabled)

    def _apply_csv_capabilities_to_ui(self, parsed):
        """Enable/disable score radios, epistasis modes, and epistasis metric combo based on loaded CSV."""
        self._available_score_cols = set(parsed.available_score_cols)
        self._available_epi_cols = list(parsed.available_epi_cols)
        self._has_doubles = bool(parsed.n_doubles_total > 0 and parsed.available_epi_cols)

        # Update epistasis metric selector availability
        for i in range(self.epi_metric_combo.count()):
            col_key = self.epi_metric_combo.itemData(i)
            available = col_key in self._available_epi_cols
            self._set_combobox_item_enabled(self.epi_metric_combo, i, available)
        if parsed.effective_epi_col:
            idx = self.epi_metric_combo.findData(parsed.effective_epi_col)
            if idx >= 0:
                self.epi_metric_combo.blockSignals(True)
                self.epi_metric_combo.setCurrentIndex(idx)
                self.epi_metric_combo.blockSignals(False)

        # Enable/disable epistasis display modes when CSV lacks doubles or epistasis columns
        for mode_idx in (1, 2):  # "WT epistasis", "MT epistasis"
            self._set_combobox_item_enabled(self.display_mode_combo, mode_idx, self._has_doubles)

        if (not self._has_doubles or parsed.auto_fallback_mode == "Singles") and self.display_mode_combo.currentIndex() in (1, 2):
            self.display_mode_combo.blockSignals(True)
            self.display_mode_combo.setCurrentIndex(0)  # Singles
            self.display_mode_combo.blockSignals(False)

        # Sync selected score radio if auto-fallback happened
        if parsed.effective_score_col == "wt_lora_pred":
            self.radio_wt_lora.setChecked(True)
        elif parsed.effective_score_col == "mt_lora_pred":
            self.radio_mt_lora.setChecked(True)
        elif parsed.effective_score_col == "combined_pred":
            self.radio_dual_view.setChecked(True)

        self._on_display_mode_changed()

    def _on_display_mode_changed(self):
        mode = self.display_mode_combo.currentText()
        is_wt_epi = (mode == "WT epistasis")
        is_epi = ("epistasis" in mode.lower())

        self.wt_color_edit.setEnabled(not is_wt_epi)
        self.wt_color_btn.setEnabled(not is_wt_epi)
        self.wt_style_combo.setEnabled(not is_wt_epi)
        self.wt_stick_alpha_spinbox.setEnabled(not is_wt_epi)
        self.wt_hide_checkbox.setEnabled(not is_wt_epi)

        avail_scores = self._available_score_cols
        has_dual = (avail_scores is None) or ("combined_pred" in avail_scores)
        has_wt = (avail_scores is None) or ("wt_lora_pred" in avail_scores)
        has_mt = (avail_scores is None) or ("mt_lora_pred" in avail_scores)

        if mode == "Singles":
            self.radio_dual_view.setEnabled(has_dual)
            self.radio_wt_lora.setEnabled(has_wt)
            self.radio_mt_lora.setEnabled(has_mt)
            self.radio_dual_view.setToolTip(
                "Uses combined_pred (WT + MT dual-view ensemble)."
                if has_dual else "Unavailable: 'combined_pred' is absent in the loaded CSV (e.g. --skip_reverse)."
            )
            self.radio_wt_lora.setToolTip(
                "Uses wt_lora_pred (additive WT pass)."
                if has_wt else "Unavailable: 'wt_lora_pred' is absent in the loaded CSV."
            )
            self.radio_mt_lora.setToolTip(
                "Uses mt_lora_pred (MT-only pass)."
                if has_mt else "Unavailable: 'mt_lora_pred' is absent in the loaded CSV (e.g. --skip_reverse)."
            )
            self.color_backbone_checkbox.setEnabled(True)
            self.epi_max_edges.setEnabled(False)
            self.display_priority_combo.setEnabled(False)
            self.pair_selection_combo.setEnabled(False)
            self.epi_metric_combo.setEnabled(False)
        else:
            if has_dual:
                self.radio_dual_view.setChecked(True)
            self.radio_wt_lora.setEnabled(False)
            self.radio_mt_lora.setEnabled(False)
            self.radio_dual_view.setEnabled(has_dual)
            self.color_backbone_checkbox.setEnabled(is_wt_epi)
            self.epi_max_edges.setEnabled(True)
            self.display_priority_combo.setEnabled(True)
            self.pair_selection_combo.setEnabled(True)
            self.epi_metric_combo.setEnabled(bool(is_epi and (self._available_epi_cols is None or len(self._available_epi_cols) > 0)))

    # ---------------- CSV Loading & Parsing -----------------
    def _get_requested_score_col(self):
        if self.radio_wt_lora.isChecked():
            return "wt_lora_pred"
        if self.radio_mt_lora.isChecked():
            return "mt_lora_pred"
        return "combined_pred"

    def _get_requested_epi_col(self):
        return self.epi_metric_combo.currentData() or "epi_pred"

    def _format_csv_summary(self, parsed):
        fname = os.path.basename(parsed.filepath)
        chains_str = ", ".join(parsed.chains_in_csv) if parsed.chains_in_csv else "?"
        chain_note = f"Chain {parsed.selected_chain}" if parsed.selected_chain else f"Chains: {chains_str}"
        parts = [
            f"<b>{fname}</b>",
            chain_note,
            f"{parsed.n_singles:,} singles ({parsed.n_positions} pos)",
        ]
        if parsed.n_doubles > 0:
            parts.append(f"{parsed.n_doubles:,} doubles")
        if parsed.score_min is not None and parsed.score_max is not None:
            parts.append(f"range [{parsed.score_min:+.2f}, {parsed.score_max:+.2f}] kcal/mol")
        summary = " | ".join(parts)
        notes = []
        if parsed.auto_fallback_score_col:
            notes.append(f"Score auto-selected: {parsed.effective_score_col}")
        if parsed.auto_fallback_mode:
            notes.append(f"Mode auto-switched to {parsed.auto_fallback_mode} (no epistasis doubles in CSV)")
        if notes:
            summary += f"<br><span style='color:#9a6700;'>{' | '.join(notes)}</span>"
        return summary

    def _load_csv_file(self, filepath):
        self._set_status("Parsing CSV...", level="info")
        parsed = self._parse_csv(filepath)
        if parsed is None:
            return False

        # Only update loaded_csv_path and csv_label AFTER parsing succeeds (or chain mismatch allows quick chain switch)
        if not parsed.ok:
            if not parsed.chain_matched and parsed.chains_in_csv:
                # Keep filepath so user can simply switch Target Chain and click Update Visualization
                self.loaded_csv_path = filepath
                self.update_viz_button.setEnabled(True)
                self._apply_csv_capabilities_to_ui(parsed)
                chains_str = ", ".join(parsed.chains_in_csv)
                self.csv_label.setText(
                    f"<b>{os.path.basename(filepath)}</b> | "
                    f"<span style='color:#9a6700;'>Selected chain '{parsed.selected_chain}' not in CSV "
                    f"(available: {chains_str}). Switch Chain and click Update Visualization.</span>"
                )
                self._set_status(parsed.error, level="warning")
                return False

            self._report_error("CSV Load Error", parsed.error, popup=True)
            return False

        self.loaded_csv_path = filepath
        self.csv_label.setText(self._format_csv_summary(parsed))
        self.update_viz_button.setEnabled(True)

        mode = self.display_mode_combo.currentText()
        if mode == "Singles":
            self._apply_singles()
        elif "epistasis" in mode.lower():
            self._apply_epistasis()
        return True

    def _handle_load_and_visualize(self):
        self.session.logger.info("****** _handle_load_and_visualize called ******")
        w = getattr(self.session.ui, "main_window", None)
        initial_dir, default_filename = "", ""

        candidate = self._get_last_output_candidate()
        if candidate:
            default_filename = candidate
            initial_dir = os.path.dirname(candidate)
        elif self.loaded_csv_path and os.path.isfile(self.loaded_csv_path):
            initial_dir = os.path.dirname(self.loaded_csv_path)

        fp, _ = QFileDialog.getOpenFileName(
            w, "Open Residue Score CSV", default_filename or initial_dir, "CSV Files (*.csv);;All Files (*)"
        )
        if fp:
            self._load_csv_file(fp)
        else:
            self._set_status("CSV loading cancelled.", level="info")

    def _handle_load_last_output(self):
        candidate = self._get_last_output_candidate()
        if not candidate:
            self._report_error(
                "Output CSV Not Found",
                "No completed prediction output CSV exists yet. Run inference or use 'Load CSV'.",
                popup=False,
            )
            return
        self.tabs.setCurrentIndex(1)
        self._load_csv_file(candidate)

    def _update_visualization(self):
        if not self.loaded_csv_path or not os.path.isfile(self.loaded_csv_path):
            self._report_error(
                "No CSV Loaded",
                "No valid CSV file is currently loaded. Click 'Load CSV' first.",
                popup=True,
            )
            return
        self._load_csv_file(self.loaded_csv_path)

    def _parse_csv(self, filepath):
        self.residue_scores_data = {}
        self.all_singles_scores = {}
        self.epistasis_df = None

        ex_wt = set()
        ex_mt = set()
        if self.exclude_wt_cys_checkbox.isChecked():
            ex_wt.add("C")
        if self.exclude_mut_cys_checkbox.isChecked():
            ex_mt.add("C")
        if self.exclude_wt_pro_checkbox.isChecked():
            ex_wt.add("P")
        if self.exclude_mut_pro_checkbox.isChecked():
            ex_mt.add("P")

        target_chain = self.viz_chain_id_combobox.currentText().strip()
        mode = self.display_mode_combo.currentText()

        parsed = parse_esm_msr_csv(
            filepath=filepath,
            requested_score_col=self._get_requested_score_col(),
            requested_epi_col=self._get_requested_epi_col(),
            mode=mode,
            target_chain=target_chain,
            exclude_wt_aa=ex_wt,
            exclude_mt_aa=ex_mt,
        )

        for w in parsed.warnings:
            self.session.logger.warning(w)

        if not parsed.ok:
            return parsed

        self.residue_scores_data = parsed.residue_scores_data
        self.all_singles_scores = parsed.all_singles_scores
        self.epistasis_df = parsed.epistasis_df
        self._apply_csv_capabilities_to_ui(parsed)
        return parsed

    # ---------------- Non-Destructive ChimeraX Rendering Helpers -----------------
    def _get_spec(self, model_id_string, res_keys):
        if not res_keys:
            return "None"
        by_chain = defaultdict(list)
        for c, pos_key in res_keys:
            by_chain[c].append(format_res_pos(pos_key))
        specs = []
        for c, positions in by_chain.items():
            specs.append(f"#{model_id_string}/{c}:{','.join(positions)}")
        return " | ".join(specs)

    def _build_residue_lookup(self, model):
        return {
            (r.chain_id, (r.number, normalize_icode(getattr(r, "insertion_code", "")))): r
            for r in model.residues
        }

    def _set_residue_scores(self, model, score_map):
        """Assign numeric SCORE_ATTRIBUTE_NAME on residues via ChimeraX Python API (no per-residue CLI spam)."""
        if not score_map:
            return
        if not self._attr_registered:
            try:
                Residue.register_attr(self.session, SCORE_ATTRIBUTE_NAME, "ESM_MSR", attr_type=float)
                self._attr_registered = True
            except Exception as e:
                self.session.logger.warning(f"Attribute registration note: {e}")

        res_lookup = self._build_residue_lookup(model)
        for key, val in score_map.items():
            if not np.isfinite(val):
                continue
            res = res_lookup.get(key)
            if res is not None:
                setattr(res, SCORE_ATTRIBUTE_NAME, float(val))

    def _clear_epistasis_pbonds(self):
        """Delete ONLY our dedicated 'ESM-MSR Epistasis' pseudobond group (never user pbonds)."""
        for group in list(self.session.models.list(type=PseudobondGroup)):
            if group.name == EPISTASIS_PBG_NAME:
                try:
                    self.session.models.close([group])
                except Exception as e:
                    self.session.logger.warning(f"Failed closing pseudobond group '{EPISTASIS_PBG_NAME}': {e}")

    def _get_or_create_epistasis_pbg(self):
        pbg = self.session.pb_manager.get_group(EPISTASIS_PBG_NAME)
        if getattr(pbg, "id", None) is None:
            self.session.models.add([pbg])
        pbg.dashes = 0
        return pbg

    def _close_generated_clones(self):
        if not self.mutated_model_id_strings:
            return
        existing_ids = {m.id_string for m in self.session.models.list()}
        for mid in list(self.mutated_model_id_strings):
            if mid in existing_ids:
                try:
                    run(self.session, f"close #{mid}")
                except Exception as e:
                    self.session.logger.warning(f"Could not close generated model #{mid}: {e}")
        self.mutated_model_id_strings = []

    def _hide_active_hydrogens(self, wt_model):
        """Scope hydrogen hiding strictly to the active target model and ESM-MSR clones."""
        active_ids = [wt_model.id_string] + list(self.mutated_model_id_strings)
        specs = [f"#{mid}" for mid in active_ids if mid]
        if specs:
            run(self.session, f"hide ({' | '.join(specs)}) & H")

    def _reset_visualization(self):
        """Close generated clones, delete our epistasis pbond group, and reset WT appearance."""
        try:
            self._clear_epistasis_pbonds()
            self._close_generated_clones()

            model_id = self.viz_model_combobox.currentData()
            wt_model = next(
                (m for m in self.session.models.list(type=Structure) if m.id_string == model_id),
                None,
            )
            if wt_model is not None:
                run(self.session, f"show #{wt_model.id_string} models")
                run(self.session, f"color #{wt_model.id_string} white")
                run(self.session, f"transparency #{wt_model.id_string} 0 target ac")
                run(self.session, f"ribbon style #{wt_model.id_string}")
                run(self.session, f"show #{wt_model.id_string} ribbons")
                run(self.session, f"hide #{wt_model.id_string} atoms")
            self._refresh_models()
            self._set_status("Cleared ESM-MSR clones and epistasis lines.", level="success")
        except Exception as e:
            self._report_error("Reset Visualization Failed", str(e), popup=False)

    def _setup_base_wt_model(self):
        self._close_generated_clones()

        wt_candidates = [
            m for m in self.session.models.list(type=Structure)
            if not self._is_generated_clone(m)
        ]

        model_id = self.viz_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        if not wt_model and wt_candidates:
            wt_model = wt_candidates[0]
        if not wt_model:
            return None

        run(self.session, f"show #{wt_model.id_string} models")
        run(self.session, f"color #{wt_model.id_string} white")
        run(self.session, f"ribbon style #{wt_model.id_string}")
        run(self.session, f"hide #{wt_model.id_string} atoms")
        return wt_model

    def _create_mutated_model(self, wt_model, suffix="viz"):
        safe_id = wt_model.id_string.replace(":", "_").replace("/", "_")
        fd, temp_pdb = tempfile.mkstemp(prefix=f"chimerax_esm_msr_clone_{safe_id}_", suffix=".pdb")
        os.close(fd)
        # Suppress synchronous 'add models' refresh until mut_id is registered as a generated clone
        self._suppress_model_refresh = True
        try:
            try:
                run(self.session, f'save "{temp_pdb}" models #{wt_model.id_string} format pdb')
                safe_name = (wt_model.name or "model").replace('"', "")
                opened = run(self.session, f'open "{temp_pdb}" name "{safe_name}_{suffix}"')
            finally:
                if os.path.exists(temp_pdb):
                    try:
                        os.remove(temp_pdb)
                    except OSError:
                        pass

            mut_model = None
            if isinstance(opened, (list, tuple)):
                mut_model = next((m for m in opened if isinstance(m, Structure)), None)
            elif isinstance(opened, Structure):
                mut_model = opened

            if mut_model is None:
                atomic_models = self.session.models.list(type=Structure)
                mut_model = atomic_models[-1] if atomic_models else None
            if mut_model is None:
                return None

            mut_id = mut_model.id_string.split(".")[0]
            self.mutated_model_id_strings.append(mut_id)
        finally:
            self._suppress_model_refresh = False

        run(self.session, f"color #{mut_id} white")
        run(self.session, f"transparency #{mut_id} 100 target a")
        run(self.session, f"hide #{mut_id} atoms")
        return mut_model

    def _apply_swapaa(self, mutated_model_id, mutated_model, mutation_plan):
        res_lookup = self._build_residue_lookup(mutated_model)
        for (chain_val, pos_key), tgt_aa in mutation_plan.items():
            res_wt = res_lookup.get((chain_val, pos_key))
            three_letter = ONE_TO_THREE_LETTER_AA.get(tgt_aa, "")
            if res_wt is not None and three_letter and three_letter != res_wt.name:
                spec = f"#{mutated_model_id}/{chain_val}:{format_res_pos(pos_key)}"
                run(self.session, f"swapaa {spec} {three_letter.lower()} log false")

    def _apply_nearby(self, wt_model, target_keys, target_spec_bare):
        if not self.show_nearby_checkbox.isChecked() or not target_keys or target_spec_bare == "None":
            return
        if self.nearby_hide_checkbox.isChecked():
            return

        # Preserve user's current selection across our temporary selection queries
        prev_selected = selected_atoms(self.session)
        try:
            dist = self.nearby_distance_spinbox.value()
            wt_mut_spec = self._get_spec(wt_model.id_string, target_keys)

            # Include sidechain atoms for non-GLY residues, and CA for GLY so Glycine mutants still find contacts
            target_sidechains = (
                f"(({target_spec_bare}) & ~@N,CA,C,O,OXT) | (({target_spec_bare}) & :GLY & @CA)"
            )
            candidate_atoms = f"#{wt_model.id_string} & protein & ~({wt_mut_spec})"
            if self.exclude_nearby_backbone_checkbox.isChecked():
                candidate_atoms += " & ~@N,CA,C,O,OXT"

            nearby_atoms_query = f"({target_sidechains}) @<{dist} & {candidate_atoms}"
            run(self.session, f"select {nearby_atoms_query}")

            sel_atoms = selected_atoms(self.session)
            if sel_atoms is None or len(sel_atoms) == 0:
                return

            # 1. Nearby Residue Styling
            run(self.session, "select up")
            res_color = self.nearby_res_color_edit.text().strip() or "grey"
            res_style = self.nearby_res_style_combo.currentText()
            res_alpha = self.nearby_res_alpha_spinbox.value()

            if res_alpha < 100:
                run(self.session, "show sel atoms")
                run(self.session, f"style sel {res_style}")
                if res_style == "stick":
                    run(self.session, "size sel stickRadius 0.25")
                try:
                    run(self.session, f"color sel {res_color} target a")
                except Exception:
                    run(self.session, "color sel grey target a")
                run(self.session, "color sel & ~C byelement target a")
                run(self.session, f"transparency sel {res_alpha} target a")
            else:
                run(self.session, "hide sel atoms")

            # 2. Nearby Atom Styling
            run(self.session, f"select {nearby_atoms_query}")
            if self.exclude_nearby_backbone_checkbox.isChecked():
                run(self.session, "select sel & ~@N,CA,C,O,OXT")

            atom_color = self.nearby_atom_color_edit.text().strip() or "purple"
            atom_style = self.nearby_atom_style_combo.currentText()
            atom_alpha = self.nearby_atom_alpha_spinbox.value()

            if atom_alpha < 100:
                run(self.session, "show sel atoms")
                run(self.session, f"style sel {atom_style}")
                if atom_style == "stick":
                    run(self.session, "size sel stickRadius 0.25")
                try:
                    run(self.session, f"color sel {atom_color} target a")
                except Exception:
                    run(self.session, "color sel purple target a")
                run(self.session, "color sel & ~C byelement target a")
                run(self.session, f"transparency sel {atom_alpha} target ab")
            else:
                run(self.session, "hide sel atoms")
        finally:
            run(self.session, "select clear")
            if prev_selected is not None and len(prev_selected) > 0:
                try:
                    prev_selected.selected = True
                except Exception:
                    pass

    def _resolve_and_apply_styles(self, wt_model, mut_model_ids, mutation_plans):
        all_mutant_keys = set()
        for plan in mutation_plans:
            all_mutant_keys.update(plan.keys())

        wt_color = self.wt_color_edit.text().strip() or "white"
        wt_style = self.wt_style_combo.currentText()
        wt_alpha = self.wt_stick_alpha_spinbox.value()
        wt_hide = self.wt_hide_checkbox.isChecked()

        mut_color = self.mut_color_edit.text().strip()
        mut_style = self.mut_style_combo.currentText()
        mut_alpha = self.mut_stick_alpha_spinbox.value()
        mut_hide = self.mut_hide_checkbox.isChecked()

        mut_specs = []
        for mut_id, plan in zip(mut_model_ids, mutation_plans):
            mutant_keys = set(plan.keys())
            if mutant_keys:
                spec = self._get_spec(mut_id, mutant_keys)
                mut_specs.append(spec)

                if mut_hide or mut_alpha >= 99:
                    run(self.session, f"hide ({spec}) atoms")
                else:
                    run(self.session, f"size ({spec}) stickRadius 0.2")
                    if mut_color:
                        try:
                            run(self.session, f"color ({spec}) {mut_color} target a")
                        except Exception:
                            run(self.session, f"color ({spec}) orange target a")
                    run(self.session, f"color ({spec}) & ~C byelement target a")
                    run(self.session, f"show ({spec}) atoms")
                    run(self.session, f"style ({spec}) {mut_style}")
                    run(self.session, f"transparency ({spec}) {mut_alpha} target a")

        if all_mutant_keys:
            spec = self._get_spec(wt_model.id_string, all_mutant_keys)
            if wt_hide or wt_alpha >= 99:
                run(self.session, f"hide ({spec}) atoms")
            else:
                run(self.session, f"size ({spec}) stickRadius 0.2")
                try:
                    run(self.session, f"color ({spec}) {wt_color} target a")
                except Exception:
                    run(self.session, f"color ({spec}) white target a")
                run(self.session, f"color ({spec}) & ~C byelement target a")
                run(self.session, f"show ({spec}) atoms")
                run(self.session, f"style ({spec}) {wt_style}")
                run(self.session, f"transparency ({spec}) {wt_alpha} target a")

        mut_spec_bare = " | ".join(mut_specs) if mut_specs else "None"
        return all_mutant_keys, mut_spec_bare

    def _draw_epistasis_line(self, pbg, c1, p1_key, c2, p2_key, score, max_abs_score, thresh_pos, thresh_neg, model_residues):
        if not np.isfinite(score):
            return False

        res1 = model_residues.get((c1, p1_key))
        res2 = model_residues.get((c2, p2_key))
        if not res1 or not res2 or not res1.atoms or not res2.atoms:
            self.session.logger.warning(
                f"Skipping epistasis line: missing residue(s) {c1}:{format_res_pos(p1_key)} or {c2}:{format_res_pos(p2_key)}."
            )
            return False

        def _candidate_atoms(res):
            sidechain = [a for a in res.atoms if a.name not in ("N", "CA", "C", "O", "OXT") and not a.name.startswith("H")]
            if sidechain:
                return sidechain
            ca = [a for a in res.atoms if a.name == "CA"]
            return ca if ca else list(res.atoms)

        atoms1 = _candidate_atoms(res1)
        atoms2 = _candidate_atoms(res2)
        if not atoms1 or not atoms2:
            return False

        coords1 = np.array([a.scene_coord for a in atoms1])
        coords2 = np.array([a.scene_coord for a in atoms2])
        dists = np.sqrt(np.sum((coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]) ** 2, axis=2))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        a1, a2 = atoms1[min_idx[0]], atoms2[min_idx[1]]

        rel_thresh = thresh_pos if score > 0 else abs(thresh_neg)
        if max_abs_score <= rel_thresh:
            norm_score = 0.0
        else:
            norm_score = min(1.0, max(0.0, (abs(score) - rel_thresh) / (max_abs_score - rel_thresh)))

        radius_val = 0.05 + (0.35 * norm_score)
        alpha_255 = int(round(255.0 * (0.30 + 0.70 * norm_score)))
        c_inv = int(round(255.0 * (1.0 - norm_score)))

        if score > 0:
            rgba = (c_inv, c_inv, 255, alpha_255)
        else:
            g_val = int(round(255.0 * (1.0 - 0.5 * norm_score)))
            rgba = (255, g_val, c_inv, alpha_255)

        pb = pbg.new_pseudobond(a1, a2)
        pb.color = rgba
        pb.radius = radius_val
        pb.shown_when_atoms_hidden = True
        return True

    def _apply_transparency_isolation(self, wt_model):
        target_chain = self.viz_chain_id_combobox.currentText().strip()
        alpha_val = self.non_target_alpha_spinbox.value()
        hide_fully = self.hide_non_target_chains_checkbox.isChecked()

        if not target_chain:
            self.session.logger.info("No target chain selected; skipping chain isolation.")
            return

        active_model_ids = [wt_model.id_string] + list(self.mutated_model_id_strings)
        hide_models = [
            f"#{m.id_string}"
            for m in self.session.models.list(type=Structure)
            if m.id_string not in active_model_ids
        ]

        try:
            for model_spec in hide_models:
                run(self.session, f"hide {model_spec} models")

            for m in self.session.models.list(type=Structure):
                if m.id_string in active_model_ids:
                    has_non_target = any(c.chain_id != target_chain for c in m.chains)
                    if has_non_target:
                        non_target_spec = f"#{m.id_string} & ~#{m.id_string}/{target_chain}"
                        if hide_fully or alpha_val >= 99:
                            run(self.session, f"hide {non_target_spec} atoms")
                            run(self.session, f"hide {non_target_spec} ribbons")
                        else:
                            run(self.session, f"show {non_target_spec} ribbons")
                            run(self.session, f"transparency {non_target_spec} {alpha_val} target ac")
        except Exception as e:
            self.session.logger.error(f"ChimeraX command failed in _apply_transparency_isolation: {e}")

    # ---------------- Visualization Subroutines -----------------
    def _apply_singles(self):
        try:
            wt_model = self._setup_base_wt_model()
            if wt_model is None:
                self._report_error("No Model Open", "Open a protein structure in ChimeraX before visualizing.", popup=True)
                return

            self._clear_epistasis_pbonds()
            pos_thresh = self.pos_threshold_spinbox.value()
            neg_thresh = self.neg_threshold_spinbox.value()

            mutation_plan = {}
            for (chain, pos_key), (score, tgt_aa) in self.residue_scores_data.items():
                if score >= pos_thresh or score <= neg_thresh:
                    mutation_plan[(chain, pos_key)] = tgt_aa

            mutated_model = self._create_mutated_model(wt_model, "singles_viz")
            if mutated_model is None:
                self._report_error("Clone Error", "Failed to create mutant visualization clone.", popup=False)
                return
            mut_id = mutated_model.id_string.split(".")[0]

            self._apply_swapaa(mut_id, mutated_model, mutation_plan)

            scores = [s for s, _ in self.residue_scores_data.values() if np.isfinite(s)]
            if scores:
                scores_abs = [abs(s) for s in scores]
                max_abs_add = float(np.percentile(scores_abs, 98))
                if max_abs_add <= 0.05:
                    max_abs_add = max(scores_abs)
                if max_abs_add <= 0.0:
                    max_abs_add = 0.01
            else:
                max_abs_add = 0.01
            color_range = f"{-max_abs_add:.3f},{max_abs_add:.3f}"

            wt_score_map = {k: score for k, (score, _) in self.residue_scores_data.items()}
            self._set_residue_scores(wt_model, wt_score_map)

            if self.color_backbone_checkbox.isChecked() and self.residue_scores_data:
                chains_present = sorted({c for c, _ in self.residue_scores_data.keys()})
                chain_spec = ",".join(chains_present)
                run(
                    self.session,
                    f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone "
                    f"palette #FF1493:white:green range {color_range} key false",
                )

            all_mutant_keys, mut_spec_bare = self._resolve_and_apply_styles(wt_model, [mut_id], [mutation_plan])

            mut_score_map = {}
            for (chain, pos_key), tgt_aa in mutation_plan.items():
                exact_score = self.all_singles_scores.get((chain, pos_key, tgt_aa))
                if exact_score is None:
                    self.session.logger.warning(
                        f"Single mutant score missing for {chain}:{format_res_pos(pos_key)}->{tgt_aa}. Defaulting to 0.0."
                    )
                    exact_score = 0.0
                mut_score_map[(chain, pos_key)] = exact_score
            self._set_residue_scores(mutated_model, mut_score_map)

            if not self.mut_color_edit.text().strip() and not self.mut_hide_checkbox.isChecked():
                mut_spec = self._get_spec(mut_id, mutation_plan.keys())
                if mut_spec != "None":
                    run(
                        self.session,
                        f"color byattribute {SCORE_ATTRIBUTE_NAME} ({mut_spec}) "
                        f"palette #FF1493:white:green range {color_range} key false target a",
                    )
                    run(self.session, f"color ({mut_spec}) & ~C byelement target a")

            try:
                run(self.session, f"key #FF1493:{-max_abs_add:.2f} white:0 green:{max_abs_add:.2f} pos 0.05,0.05")
            except Exception as e:
                self.session.logger.warning(f"Failed to draw colorbar key: {e}")

            self._apply_nearby(wt_model, all_mutant_keys, mut_spec_bare)
            self._apply_transparency_isolation(wt_model)
            self._hide_active_hydrogens(wt_model)
            self._set_status(
                f"Singles visualization complete ({len(mutation_plan)} displayed mutation(s)).",
                level="success",
            )
        except Exception as e:
            self._report_error("Visualization Error", str(e), popup=False)

    def _apply_epistasis(self):
        try:
            wt_model = self._setup_base_wt_model()
            if wt_model is None:
                self._report_error("No Model Open", "Open a protein structure in ChimeraX before visualizing.", popup=True)
                return

            self._clear_epistasis_pbonds()
            df = self.epistasis_df
            if df is None or df.empty:
                self._report_error(
                    "No Epistasis Data",
                    "No valid double-mutant epistasis rows are loaded for this chain.",
                    popup=False,
                )
                return

            is_wt_epi = (self.display_mode_combo.currentText() == "WT epistasis")
            if is_wt_epi:
                valid_mask = (
                    ((df["wt1"] != "A") & (df["mut1"] == "A")) | ((df["wt1"] == "A") & (df["mut1"] == "G"))
                ) & (
                    ((df["wt2"] != "A") & (df["mut2"] == "A")) | ((df["wt2"] == "A") & (df["mut2"] == "G"))
                )
                df = df[valid_mask].copy()
                if df.empty:
                    self._set_status(
                        "WT Epistasis requires Ala/Gly double mutations (X->A or A->G); none found.",
                        level="warning",
                    )
                    return

            metric_target = self.pair_selection_combo.currentText()
            sort_col = "combined_pred" if metric_target == "Stability" else "dddg_pred"

            pos_thresh = self.pos_threshold_spinbox.value()
            neg_thresh = self.neg_threshold_spinbox.value()
            filtered_df = df[(df[sort_col] >= pos_thresh) | (df[sort_col] <= neg_thresh)].copy()

            if filtered_df.empty:
                self._set_status(
                    f"No residue pairs found outside thresholds using {metric_target}.",
                    level="warning",
                )
                return

            priority = self.display_priority_combo.currentText()
            if priority == "High score":
                sorted_df = filtered_df.sort_values(by=sort_col, ascending=False)
            elif priority == "Low score":
                sorted_df = filtered_df.sort_values(by=sort_col, ascending=True)
            else:
                sorted_df = filtered_df.assign(abs_score_sort=filtered_df[sort_col].abs()).sort_values(
                    by="abs_score_sort", ascending=False
                )

            max_abs_epi = float(sorted_df["dddg_pred"].abs().max())
            if not np.isfinite(max_abs_epi) or max_abs_epi <= 0.0:
                max_abs_epi = 1.0

            scores = [s for s, _ in self.residue_scores_data.values() if np.isfinite(s)]
            if scores:
                max_abs_add = float(max(abs(s) for s in scores))
                if max_abs_add <= 0.0:
                    max_abs_add = 0.01
            else:
                max_abs_add = 0.01
            color_range = f"{-max_abs_add:.3f},{max_abs_add:.3f}"

            wt_score_map = {k: score for k, (score, _) in self.residue_scores_data.items()}
            self._set_residue_scores(wt_model, wt_score_map)

            max_edges = self.epi_max_edges.value()
            drawn_count = 0

            if is_wt_epi:
                participating_positions = set()
                pairs_to_draw = []
                connection_counts = defaultdict(int)
                seen_pairs = set()
                mutation_plan = {}

                for row in sorted_df.itertuples(index=False):
                    c1 = str(row.chain_id).strip()
                    p1_key = row.pos1_key
                    m1 = str(row.mut1).upper()
                    c2 = str(row.chain_id).strip()
                    p2_key = row.pos2_key
                    m2 = str(row.mut2).upper()
                    epi_score = float(row.dddg_pred)

                    pair_key = tuple(sorted([(c1, p1_key), (c2, p2_key)]))
                    if pair_key in seen_pairs:
                        continue
                    if connection_counts[(c1, p1_key)] >= max_edges or connection_counts[(c2, p2_key)] >= max_edges:
                        continue

                    seen_pairs.add(pair_key)
                    connection_counts[(c1, p1_key)] += 1
                    connection_counts[(c2, p2_key)] += 1
                    pairs_to_draw.append((c1, p1_key, m1, c2, p2_key, m2, epi_score))
                    participating_positions.add((c1, p1_key))
                    participating_positions.add((c2, p2_key))
                    mutation_plan[(c1, p1_key)] = m1
                    mutation_plan[(c2, p2_key)] = m2

                if (
                    self.color_backbone_checkbox.isEnabled()
                    and self.color_backbone_checkbox.isChecked()
                    and self.residue_scores_data
                ):
                    chains_present = sorted({c for c, _ in self.residue_scores_data.keys()})
                    chain_spec = ",".join(chains_present)
                    run(
                        self.session,
                        f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone "
                        f"palette #FF1493:white:green range {color_range} key false",
                    )

                spec = self._get_spec(wt_model.id_string, participating_positions)
                if spec != "None":
                    epi_wt_scores = {}
                    for chain, pos_key in participating_positions:
                        tgt_aa = mutation_plan[(chain, pos_key)]
                        exact_score = self.all_singles_scores.get((chain, pos_key, tgt_aa))
                        if exact_score is None:
                            self.session.logger.warning(
                                f"Single Ala/Gly score missing for {chain}:{format_res_pos(pos_key)}. Defaulting to 0.0."
                            )
                            exact_score = 0.0
                        epi_wt_scores[(chain, pos_key)] = exact_score
                    self._set_residue_scores(wt_model, epi_wt_scores)

                    # P1 #14: Honor custom Mut Color and Hide Fully consistently in WT Epistasis mode
                    mut_color = self.mut_color_edit.text().strip()
                    mut_style = self.mut_style_combo.currentText()
                    mut_alpha = self.mut_stick_alpha_spinbox.value()
                    mut_hide = self.mut_hide_checkbox.isChecked()

                    if mut_hide or mut_alpha >= 99:
                        run(self.session, f"hide ({spec}) atoms")
                    else:
                        run(self.session, f"size ({spec}) stickRadius 0.2")
                        if mut_color:
                            try:
                                run(self.session, f"color ({spec}) {mut_color} target a")
                            except Exception:
                                run(self.session, f"color ({spec}) orange target a")
                        else:
                            run(
                                self.session,
                                f"color byattribute {SCORE_ATTRIBUTE_NAME} ({spec}) "
                                f"palette #FF1493:white:green range {color_range} key false target a",
                            )
                        run(self.session, f"color ({spec}) & ~C byelement target a")
                        run(self.session, f"show ({spec}) atoms")
                        run(self.session, f"style ({spec}) {mut_style}")
                        run(self.session, f"transparency ({spec}) {mut_alpha} target a")

                self._apply_nearby(wt_model, participating_positions, spec)

                if pairs_to_draw:
                    pbg = self._get_or_create_epistasis_pbg()
                    wt_residues = self._build_residue_lookup(wt_model)
                    for c1, p1_key, _m1, c2, p2_key, _m2, score in pairs_to_draw:
                        if self._draw_epistasis_line(
                            pbg, c1, p1_key, c2, p2_key, score, max_abs_epi, pos_thresh, neg_thresh, wt_residues
                        ):
                            drawn_count += 1
                num_layers = 1
            else:
                layers = []
                pairs_to_draw = []
                connection_counts = defaultdict(int)
                seen_pairs = set()

                for row in sorted_df.itertuples(index=False):
                    c1 = str(row.chain_id).strip()
                    p1_key = row.pos1_key
                    m1 = str(row.mut1).upper()
                    c2 = str(row.chain_id).strip()
                    p2_key = row.pos2_key
                    m2 = str(row.mut2).upper()
                    epi_score = float(row.dddg_pred)

                    # P2 #19: deduplicate residue-pair edges cleanly
                    pair_key = tuple(sorted([(c1, p1_key), (c2, p2_key)]))
                    if pair_key in seen_pairs:
                        continue
                    if connection_counts[(c1, p1_key)] >= max_edges or connection_counts[(c2, p2_key)] >= max_edges:
                        continue

                    assigned_layer = -1
                    for i, layer in enumerate(layers):
                        if layer.get((c1, p1_key), m1) == m1 and layer.get((c2, p2_key), m2) == m2:
                            assigned_layer = i
                            break
                    if assigned_layer == -1:
                        assigned_layer = len(layers)
                        layers.append({})

                    layers[assigned_layer][(c1, p1_key)] = m1
                    layers[assigned_layer][(c2, p2_key)] = m2

                    seen_pairs.add(pair_key)
                    connection_counts[(c1, p1_key)] += 1
                    connection_counts[(c2, p2_key)] += 1
                    pairs_to_draw.append((c1, p1_key, m1, c2, p2_key, m2, epi_score, assigned_layer))

                mutated_models_objs = []
                for i, layer in enumerate(layers):
                    mut_model = self._create_mutated_model(wt_model, f"epi_layer_{i}")
                    if mut_model is None:
                        continue
                    mut_id = mut_model.id_string.split(".")[0]
                    self._apply_swapaa(mut_id, mut_model, layer)
                    mutated_models_objs.append(mut_model)

                if (
                    self.color_backbone_checkbox.isEnabled()
                    and self.color_backbone_checkbox.isChecked()
                    and self.residue_scores_data
                ):
                    chains_present = sorted({c for c, _ in self.residue_scores_data.keys()})
                    chain_spec = ",".join(chains_present)
                    run(
                        self.session,
                        f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone "
                        f"palette #FF1493:white:green range {color_range} key false",
                    )

                all_mutant_keys, mut_spec_bare = self._resolve_and_apply_styles(
                    wt_model, self.mutated_model_id_strings, layers
                )

                for i, (mut_model, plan) in enumerate(zip(mutated_models_objs, layers)):
                    if not plan:
                        continue
                    mut_id = self.mutated_model_id_strings[i]
                    layer_scores = {}
                    for (chain, pos_key), tgt_aa in plan.items():
                        exact_score = self.all_singles_scores.get((chain, pos_key, tgt_aa))
                        if exact_score is None:
                            self.session.logger.warning(
                                f"Single mutant score missing for {chain}:{format_res_pos(pos_key)}->{tgt_aa}. Defaulting to 0.0."
                            )
                            exact_score = 0.0
                        layer_scores[(chain, pos_key)] = exact_score
                    self._set_residue_scores(mut_model, layer_scores)

                    if not self.mut_color_edit.text().strip() and not self.mut_hide_checkbox.isChecked():
                        mut_spec = self._get_spec(mut_id, plan.keys())
                        if mut_spec != "None":
                            run(
                                self.session,
                                f"color byattribute {SCORE_ATTRIBUTE_NAME} ({mut_spec}) "
                                f"palette #FF1493:white:green range {color_range} key false target a",
                            )
                            run(self.session, f"color ({mut_spec}) & ~C byelement target a")

                self._apply_nearby(wt_model, all_mutant_keys, mut_spec_bare)

                if pairs_to_draw:
                    pbg = self._get_or_create_epistasis_pbg()
                    # Precompute residue lookup dict ONCE per layer model
                    layer_res_lookups = [self._build_residue_lookup(m) for m in mutated_models_objs]
                    for c1, p1_key, _m1, c2, p2_key, _m2, score, layer_idx in pairs_to_draw:
                        if layer_idx < len(layer_res_lookups):
                            if self._draw_epistasis_line(
                                pbg,
                                c1, p1_key, c2, p2_key,
                                score, max_abs_epi, pos_thresh, neg_thresh,
                                layer_res_lookups[layer_idx],
                            ):
                                drawn_count += 1
                num_layers = len(layers)

            try:
                run(self.session, f"key orange:{-max_abs_epi:.2f} white:0 blue:{max_abs_epi:.2f} pos 0.05,0.05")
            except Exception as e:
                self.session.logger.warning(f"Failed to draw epistasis colorbar key: {e}")

            self._apply_transparency_isolation(wt_model)
            self._hide_active_hydrogens(wt_model)
            self._set_status(
                f"Epistasis visualization complete ({drawn_count} interaction(s) across {num_layers} layer(s)).",
                level="success",
            )
        except Exception as e:
            self._report_error("Epistasis Visualization Error", str(e), popup=False)
