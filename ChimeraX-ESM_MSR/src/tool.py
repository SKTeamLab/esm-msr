# vim: set expandtab shiftwidth=4 softstop=4:
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import re
from collections import defaultdict

from chimerax.ui import MainToolWindow
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run

from chimerax.atomic import Structure

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QLineEdit, QFrame, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTabWidget, QRadioButton, QButtonGroup
)
from Qt.QtCore import QProcess, QSettings

SCORE_ATTRIBUTE_NAME = "residue_score_viz_score"

ONE_TO_THREE_LETTER_AA = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL'
}

class ESM_MSR_Tool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    display_name = "Residue Score Visualizer & Predictor"

    def __init__(self, session, tool_registered_name):
        super().__init__(session, tool_registered_name)
        #self.session.logger.info(f"****** RSVTool __init__ ({tool_registered_name}) ******")

        # 1. INITIALIZE SETTINGS
        self.settings = QSettings("ESM_MSR_Tools", "ESM_MSR")
        
        # 2. LOAD STATE VARIABLES
        self.base_repo_path = self.settings.value("base_repo_path", "")
        self.python_env = self.settings.value("python_env", "")
        self.script_output_csv_path = self.settings.value("script_output_csv_path", "")
        self.checkpoint_path = self.settings.value("checkpoint_path", "")
        self.config_file = self.settings.value("config_file", "")
        self.base_model_loc = self.settings.value("base_model_loc", "")
        
        self._closing = False
        self._temp_dir_to_cleanup = None
        self.residue_scores_data = {}
        self.epistasis_df = None  
        self.loaded_csv_path = ""
        self.mutated_model_id_string = None
        self.predicted_output_path = ""
        self.proc = None  

        # 3. BUILD THE UI
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        self._build_ui(parent)
        self.tool_window.manage(None)

        # 4. POPULATE UI WITH DYNAMIC PATHS
        self._update_paths_from_base_repo(self.base_repo_path, is_init=True)
        
        # 5. INITIAL PATH VALIDATION
        self._validate_file_paths()
        
        # 6. REGISTER CHIMERAX EVENT HOOKS
        try:
            self._models_added_handler = self.session.triggers.add_handler('add models', self._refresh_models)
            self._models_removed_handler = self.session.triggers.add_handler('remove models', self._refresh_models)
        except Exception as e:
            self.session.logger.warning(f"Could not register model event hooks: {e}")
            raise AssertionError(f"Critical failure registering ChimeraX triggers: {e}")

        # Initial populate
        self._refresh_models()

    # ---------------- UI -----------------
    def _build_ui(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Build Sub-Tabs
        self.tab_io = QWidget()
        self._build_io_tab(self.tab_io)
        self.tabs.addTab(self.tab_io, "Execution / IO")

        self.tab_screening = QWidget()
        self._build_screening_tab(self.tab_screening)
        self.tabs.addTab(self.tab_screening, "Screening Config")

        self.tab_viz = QWidget()
        self._build_viz_tab(self.tab_viz)
        self.tabs.addTab(self.tab_viz, "Visualization")

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Global Status and Execution Layout
        global_exec_layout = QHBoxLayout()
        self.run_prediction_button = QPushButton("Run Prediction Script")
        self.run_prediction_button.setStyleSheet("font-weight: bold; padding: 5px;")
        self.run_prediction_button.clicked.connect(self._initiate_run_prediction_script)
        global_exec_layout.addWidget(self.run_prediction_button)

        self.stop_prediction_button = QPushButton("STOP")
        self.stop_prediction_button.setStyleSheet("background-color: red; color: white; font-weight: bold; padding: 5px;")
        self.stop_prediction_button.clicked.connect(self._stop_prediction_script)
        self.stop_prediction_button.setVisible(False)
        global_exec_layout.addWidget(self.stop_prediction_button)

        self.load_button = QPushButton("Load CSV + Visualize Scores")
        self.load_button.clicked.connect(self._handle_load_and_visualize)
        global_exec_layout.addWidget(self.load_button)

        main_layout.addLayout(global_exec_layout)

        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)
        
        self.prediction_output_label = QLabel("Predicted output file: None")
        main_layout.addWidget(self.prediction_output_label)

    def _build_io_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        # Paths Group
        paths_group = QGroupBox("Environment & Paths")
        paths_layout = QVBoxLayout()
        paths_group.setLayout(paths_layout)

        base_repo_layout = QHBoxLayout()
        base_repo_layout.addWidget(QLabel("Base Repo Dir:"))
        self.base_repo_path_edit = QLineEdit(self.base_repo_path)
        base_repo_layout.addWidget(self.base_repo_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_base_repo)
        base_repo_layout.addWidget(btn)
        paths_layout.addLayout(base_repo_layout)

        py_env_layout = QHBoxLayout()
        py_env_layout.addWidget(QLabel("Python Env:"))
        self.python_env_edit = QLineEdit(self.python_env)
        py_env_layout.addWidget(self.python_env_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_python_env)
        py_env_layout.addWidget(btn)
        paths_layout.addLayout(py_env_layout)

        out_csv_layout = QHBoxLayout()
        out_csv_layout.addWidget(QLabel("Output CSV:"))
        self.script_output_csv_path_edit = QLineEdit(self.script_output_csv_path)
        out_csv_layout.addWidget(self.script_output_csv_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_script_output_csv)
        out_csv_layout.addWidget(btn)
        paths_layout.addLayout(out_csv_layout)

        layout.addWidget(paths_group)

        # Model Source Group
        source_group = QGroupBox("Model Weights Source")
        source_layout = QVBoxLayout()
        source_group.setLayout(source_layout)

        self.source_btn_group = QButtonGroup()

        # HF Token Radio
        hf_layout = QHBoxLayout()
        self.radio_hf = QRadioButton("HuggingFace Token:")
        self.source_btn_group.addButton(self.radio_hf)
        hf_layout.addWidget(self.radio_hf)
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setPlaceholderText("For ESM3 weights; can leave blank if logged in via cli")
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        hf_layout.addWidget(self.hf_token_edit)
        source_layout.addLayout(hf_layout)

        # Base Model Location Radio
        base_loc_layout = QHBoxLayout()
        self.radio_base_loc = QRadioButton("Base Model Location (data folder):")
        self.source_btn_group.addButton(self.radio_base_loc)
        base_loc_layout.addWidget(self.radio_base_loc)
        self.base_model_loc_edit = QLineEdit(self.base_model_loc)
        self.base_model_loc_edit.setPlaceholderText("Path to offline model folder")
        base_loc_layout.addWidget(self.base_model_loc_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_generic_dir(self.base_model_loc_edit, "Select Model Directory", "base_model_loc"))
        base_loc_layout.addWidget(btn)
        source_layout.addLayout(base_loc_layout)

        layout.addWidget(source_group)

        # Compute Environment (Moved from Model & Checkpoint)
        device_group = QGroupBox("Compute Environment")
        device_layout = QHBoxLayout()
        device_group.setLayout(device_layout)
        
        device_layout.addWidget(QLabel("Compute Device:"))
        self.device_combobox = QComboBox()
        self.device_combobox.addItems(['cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu'])
        device_layout.addWidget(self.device_combobox)
        
        device_layout.addSpacing(20)
        
        device_layout.addWidget(QLabel("Batch Size (lower this if you get CUDA OOM):"))
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 512)
        self.batch_size_spinbox.setValue(16)
        device_layout.addWidget(self.batch_size_spinbox)
        
        layout.addWidget(device_group)

        # Config Files (Moved from Model & Checkpoint)
        files_group = QGroupBox("Model Configuration Files")
        files_layout = QVBoxLayout()
        files_group.setLayout(files_layout)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Checkpoint (.ckpt/.safetensors):"))
        self.checkpoint_path_edit = QLineEdit(self.checkpoint_path)
        self.checkpoint_path_edit.textChanged.connect(self._validate_file_paths)
        row1.addWidget(self.checkpoint_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_checkpoint_path)
        row1.addWidget(btn)
        files_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("LoRA Config (JSON/YAML):"))
        self.config_file_edit = QLineEdit(self.config_file)
        self.config_file_edit.textChanged.connect(self._validate_file_paths)
        self.config_file_edit.setPlaceholderText("Optional config or hparams file")
        row2.addWidget(self.config_file_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_generic(self.config_file_edit, "Select Config", "JSON/YAML (*.json *.yaml *.yml);;All (*)", "config_file"))
        row2.addWidget(btn)
        files_layout.addLayout(row2)

        # File validation warning label
        self.file_warning_label = QLabel("")
        self.file_warning_label.setStyleSheet("color: red; font-weight: bold;")
        files_layout.addWidget(self.file_warning_label)

        layout.addWidget(files_group)
        layout.addStretch()

        # Connect UI logic
        self.radio_hf.toggled.connect(self._update_source_ui)
        self.radio_base_loc.toggled.connect(self._update_source_ui)
        
        # Set default
        self.radio_hf.setChecked(True)
        self._update_source_ui()

    def _validate_file_paths(self):
        warnings = []
        ckpt = self.checkpoint_path_edit.text().strip()
        cfg = self.config_file_edit.text().strip()

        if ckpt and not os.path.exists(ckpt):
            warnings.append("Checkpoint file does not exist.")
        if cfg and not os.path.exists(cfg):
            warnings.append("Config file does not exist.")

        if warnings:
            self.file_warning_label.setText("WARNING: " + " | ".join(warnings))
        else:
            self.file_warning_label.setText("")

    def _update_source_ui(self):
        is_hf = self.radio_hf.isChecked()
        self.hf_token_edit.setEnabled(is_hf)
        
        is_base_loc = self.radio_base_loc.isChecked()
        self.base_model_loc_edit.setEnabled(is_base_loc)
        self.base_model_loc_edit.parent().findChildren(QPushButton)[0].setEnabled(is_base_loc) # Disable browse button if needed

    def _build_screening_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        # Target Selection Group
        target_group = QGroupBox("Target Selection")
        target_layout = QHBoxLayout()
        target_group.setLayout(target_layout)

        target_layout.addWidget(QLabel("Target Model:"))
        self.pred_model_combobox = QComboBox()
        self.pred_model_combobox.currentIndexChanged.connect(self._on_model_selected)
        target_layout.addWidget(self.pred_model_combobox)

        target_layout.addWidget(QLabel("Chain:"))
        self.pred_chain_id_combobox = QComboBox()
        self.pred_chain_id_combobox.setEditable(True)
        target_layout.addWidget(self.pred_chain_id_combobox)
        
        layout.addWidget(target_group)

        # Mutations Scope Group
        scope_group = QGroupBox("Mutation Scope (Mutually Exclusive Inputs)")
        scope_layout = QVBoxLayout()
        scope_group.setLayout(scope_layout)

        self.scope_btn_group = QButtonGroup()

        # Mode 1: Automated Screening
        meth1_layout = QVBoxLayout()
        
        meth1_row1 = QHBoxLayout()
        self.radio_full = QRadioButton("1. Full Screen:")
        self.scope_btn_group.addButton(self.radio_full)
        meth1_row1.addWidget(self.radio_full)

        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['singles', 'singles+doubles']) #'doubles',
        self.mode_combobox.currentTextChanged.connect(self._update_scope_ui)
        meth1_row1.addWidget(self.mode_combobox)
        
        meth1_row1.addWidget(QLabel("Positions:"))
        self.selected_residues_edit = QLineEdit()
        self.selected_residues_edit.setPlaceholderText("e.g. 11,12 (Empty=All)")
        meth1_row1.addWidget(self.selected_residues_edit)

        self.grab_sel_button = QPushButton("Grab Selection")
        self.grab_sel_button.clicked.connect(self._grab_selection)
        meth1_row1.addWidget(self.grab_sel_button)

        self.screen_except_checkbox = QCheckBox("Invert (Except)")
        meth1_row1.addWidget(self.screen_except_checkbox)
        meth1_layout.addLayout(meth1_row1)

        # Mode 1 - Distance Filter Row
        meth1_row2 = QHBoxLayout()
        meth1_row2.addSpacing(20) # Indent to show it belongs to mode 1
        self.enable_distance_checkbox = QCheckBox("Filter doubles by distance (Å):")
        self.enable_distance_checkbox.setChecked(False)
        self.enable_distance_checkbox.toggled.connect(self._toggle_distance_spinbox)
        meth1_row2.addWidget(self.enable_distance_checkbox)

        self.distance_threshold_spinbox = QDoubleSpinBox()
        self.distance_threshold_spinbox.setRange(0.0, 100.0)
        self.distance_threshold_spinbox.setValue(6.0)
        self.distance_threshold_spinbox.setEnabled(False) # Default off
        meth1_row2.addWidget(self.distance_threshold_spinbox)
        meth1_row2.addStretch()
        meth1_layout.addLayout(meth1_row2)

        scope_layout.addLayout(meth1_layout)

        # Mode 2: Input CSV
        meth2_layout = QHBoxLayout()
        self.radio_csv = QRadioButton("2. Specify mutations in CSV:")
        self.scope_btn_group.addButton(self.radio_csv)
        meth2_layout.addWidget(self.radio_csv)

        self.subset_df_edit = QLineEdit()
        self.subset_df_edit.setPlaceholderText("Path to input DataFrame...")
        meth2_layout.addWidget(self.subset_df_edit)
        self.subset_df_btn = QPushButton("Browse...")
        self.subset_df_btn.clicked.connect(self._browse_subset_df)
        meth2_layout.addWidget(self.subset_df_btn)
        scope_layout.addLayout(meth2_layout)

        # Mode 3: Specific Mutations
        meth3_layout = QHBoxLayout()
        self.radio_explicit = QRadioButton("3. Input Mutations Directly:")
        self.scope_btn_group.addButton(self.radio_explicit)
        meth3_layout.addWidget(self.radio_explicit)

        self.mutations_edit = QLineEdit()
        self.mutations_edit.setPlaceholderText("e.g., A12C,A12C:D15E")
        meth3_layout.addWidget(self.mutations_edit)
        scope_layout.addLayout(meth3_layout)
        
        layout.addWidget(scope_group)

        # Connect UI logic
        self.radio_full.toggled.connect(self._update_scope_ui)
        self.radio_csv.toggled.connect(self._update_scope_ui)
        self.radio_explicit.toggled.connect(self._update_scope_ui)
        
        # Set default
        self.radio_full.setChecked(True)

        # Runtime Options
        runtime_group = QGroupBox("Screening Parameters")
        runtime_layout = QVBoxLayout()
        runtime_group.setLayout(runtime_layout)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mask Strategy:"))
        self.mask_strategy_combobox = QComboBox()
        self.mask_strategy_combobox.addItems(['Default (unmasked)', 'marginal', 'chain'])
        row1.addWidget(self.mask_strategy_combobox)
        
        # TODO: Add helper text for Mask Strategy selection here. 
        # (e.g., replace the stretch below with a QLabel containing your helper text)
        row1.addStretch()

        runtime_layout.addLayout(row1)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Artificial Background Mutation:"))
        self.backbone_mutation_edit = QLineEdit()
        self.backbone_mutation_edit.setPlaceholderText("e.g., A15G")
        row3.addWidget(self.backbone_mutation_edit)
        runtime_layout.addLayout(row3)

        # Execution flags moved into Screening Parameters
        flags_layout = QHBoxLayout()
        self.skip_additive_checkbox = QCheckBox("Approximate Epistasis (Not Recommended)")
        flags_layout.addWidget(self.skip_additive_checkbox)
        self.skip_reverse_checkbox = QCheckBox("Skip MT LoRA Pass (Use Additive Approximation)")
        flags_layout.addWidget(self.skip_reverse_checkbox)
        flags_layout.addStretch()
        runtime_layout.addLayout(flags_layout)
        
        # Protein Complex Mode
        complex_layout = QHBoxLayout()
        complex_layout.addWidget(QLabel("Protein Complex Mode (Experimental):"))
        self.quaternary_mode_combobox = QComboBox()
        self.quaternary_mode_combobox.addItems(["single_chain", "complex"])
        complex_layout.addWidget(self.quaternary_mode_combobox)
        complex_layout.addStretch()
        runtime_layout.addLayout(complex_layout)

        layout.addWidget(runtime_group)
        layout.addStretch()

    def _update_scope_ui(self):
        # Enable Full Screen inputs
        is_full = self.radio_full.isChecked()
        self.mode_combobox.setEnabled(is_full)
        self.selected_residues_edit.setEnabled(is_full)
        self.grab_sel_button.setEnabled(is_full)
        self.screen_except_checkbox.setEnabled(is_full)

        # Handle distance checkbox logic safely
        current_mode = self.mode_combobox.currentText()
        is_doubles_mode = current_mode == 'singles+doubles'
        should_enable_dist = is_full and is_doubles_mode
        
        self.enable_distance_checkbox.setEnabled(should_enable_dist)
        if not should_enable_dist:
            self.enable_distance_checkbox.setChecked(False)
            self.distance_threshold_spinbox.setEnabled(False)
        else:
            self.distance_threshold_spinbox.setEnabled(self.enable_distance_checkbox.isChecked())

        # Enable CSV inputs
        is_csv = self.radio_csv.isChecked()
        self.subset_df_edit.setEnabled(is_csv)
        self.subset_df_btn.setEnabled(is_csv)

        # Enable Explicit inputs
        is_explicit = self.radio_explicit.isChecked()
        self.mutations_edit.setEnabled(is_explicit)

    def _toggle_distance_spinbox(self, checked):
        self.distance_threshold_spinbox.setEnabled(checked)

    # ---------------- UI Callbacks -----------------
    def _grab_selection(self):
        try:
            from chimerax.atomic import selected_atoms
            atoms = selected_atoms(self.session)
            if atoms is None or len(atoms) == 0:
                self.selected_residues_edit.setText("")
                self.status_label.setText("Status: No residues selected.")
                return
            
            res_nums = sorted(list(set(atoms.residues.numbers)))
            if res_nums:
                self.selected_residues_edit.setText(",".join(map(str, res_nums)))
                self.status_label.setText(f"Status: Grabbed {len(res_nums)} unique residue indices.")
        except Exception as e:
            self.session.logger.warning(f"Could not grab selection: {e}")
            self.status_label.setText("Status: Failed to grab selection.")

    def _refresh_models(self, trigger_name=None, trigger_data=None):
        if self._closing: return
        
        current_model_id = self.pred_model_combobox.currentData()
        
        self.pred_model_combobox.blockSignals(True)
        self.pred_model_combobox.clear()
        
        models = [m for m in self.session.models.list(type=Structure) 
                  if not (self.mutated_model_id_string and m.id_string == self.mutated_model_id_string)]
        
        for m in models:
            self.pred_model_combobox.addItem(f"#{m.id_string} {m.name}", m.id_string)
            
        idx = self.pred_model_combobox.findData(current_model_id)
        if idx >= 0:
            self.pred_model_combobox.setCurrentIndex(idx)
        elif self.pred_model_combobox.count() > 0:
            self.pred_model_combobox.setCurrentIndex(0)
            
        self.pred_model_combobox.blockSignals(False)
        self._on_model_selected()

    def _on_model_selected(self, index=None):
        if self._closing: return
        
        current_chain = self.pred_chain_id_combobox.currentText()
        self.pred_chain_id_combobox.clear()
        
        model_id = self.pred_model_combobox.currentData()
        if not model_id:
            return
            
        models = self.session.models.list(type=Structure)
        target_model = next((m for m in models if m.id_string == model_id), None)
        
        chains = set()
        if target_model:
            for c in target_model.chains:
                chains.add(c.chain_id)
                
        if chains:
            self.pred_chain_id_combobox.addItems(sorted(list(chains)))
        else:
            self.pred_chain_id_combobox.addItems(['A'])
            
        if current_chain in chains:
            self.pred_chain_id_combobox.setCurrentText(current_chain)

    def _browse_base_repo(self):
        w = self.session.ui.main_window
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
                self.checkpoint_path = os.path.normpath(os.path.join(base_path, "LoRA_models", "esm-msr-small", "epoch=03-val_rho_combined_avg=0.816.ckpt"))
            if not is_init or not self.script_output_csv_path:
                self.script_output_csv_path = os.path.normpath(os.path.join(base_path, "example_inference.csv"))
            if not is_init or not self.python_env:
                self.python_env = os.path.normpath(os.path.join(base_path, "msr_venv"))

        if hasattr(self, 'checkpoint_path_edit'):
            self.checkpoint_path_edit.setText(self.checkpoint_path)
        if hasattr(self, 'script_output_csv_path_edit'):
            self.script_output_csv_path_edit.setText(self.script_output_csv_path)
        if hasattr(self, 'python_env_edit'):
            self.python_env_edit.setText(self.python_env)

    # ---------------- generic browsers -----------------
    def _browse_script_output_csv(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getSaveFileName(w, "Specify Prediction Output CSV File Path", self.script_output_csv_path_edit.text(), "CSV Files (*.csv);;All Files (*)")
        if fp:
            if not fp.lower().endswith('.csv'): fp += '.csv'
            norm_fp = os.path.normpath(fp)
            self.script_output_csv_path = norm_fp
            self.script_output_csv_path_edit.setText(norm_fp)
            self.settings.setValue("script_output_csv_path", norm_fp)

    def _browse_checkpoint_path(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, "Select Checkpoint File", self.checkpoint_path_edit.text(), "Checkpoint Files (*.ckpt *.safetensors);;All Files (*)")
        if fp:
            norm_fp = os.path.normpath(fp)
            self.checkpoint_path_edit.setText(norm_fp)
            self.settings.setValue("checkpoint_path", norm_fp)
            
            # Auto-populate config file using parent directory of the checkpoint
            parent_dir = os.path.dirname(norm_fp)
            assumed_config_path = os.path.join(parent_dir, "hparams.yaml")
            self.config_file_edit.setText(assumed_config_path)
            self.settings.setValue("config_file", assumed_config_path)

    def _browse_python_env(self):
        w = self.session.ui.main_window
        folder_path = QFileDialog.getExistingDirectory(w, "Select Python/Conda Environment Folder", self.python_env_edit.text())
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            self.python_env_edit.setText(folder_path)
            self.python_env = folder_path
            self.settings.setValue("python_env", folder_path)

    def _browse_subset_df(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, "Select Subset DataFrame CSV", self.base_repo_path, "CSV Files (*.csv);;All Files (*)")
        if fp:
            self.subset_df_edit.setText(os.path.normpath(fp))

    def _browse_generic(self, line_edit, title, filter_str, settings_key=None):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, title, self.base_repo_path, filter_str)
        if fp:
            norm_fp = os.path.normpath(fp)
            line_edit.setText(norm_fp)
            if settings_key:
                self.settings.setValue(settings_key, norm_fp)
            
    def _browse_generic_dir(self, line_edit, title, settings_key=None):
        w = self.session.ui.main_window
        folder_path = QFileDialog.getExistingDirectory(w, title, self.base_repo_path)
        if folder_path:
            norm_fp = os.path.normpath(folder_path)
            line_edit.setText(norm_fp)
            if settings_key:
                self.settings.setValue(settings_key, norm_fp)

    # -------------- process termination --------------
    def _stop_prediction_script(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.session.logger.warning("Force terminating prediction process tree...")
            self.status_label.setText("Status: Force terminating process...")
            
            try:
                pid = self.proc.processId()
                import platform
                import subprocess
                
                # Attempt to aggressively kill process tree to prevent zombie PyTorch workers
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                else:
                    # POSIX: pkill -P kills direct children before killing the parent
                    subprocess.run(["pkill", "-9", "-P", str(pid)], capture_output=True)
                    subprocess.run(["kill", "-9", str(pid)], capture_output=True)
                
                # Fallback to standard QProcess kill to ensure Qt state is updated
                self.proc.kill()
                self.proc.waitForFinished(1000)
            except Exception as e:
                self.session.logger.error(f"Failed to execute process kill commands: {e}")
                raise AssertionError(f"Process tree kill failed: {e}")
            finally:
                self.run_prediction_button.setVisible(True)
                self.stop_prediction_button.setVisible(False)
                self.run_prediction_button.setEnabled(True)
                self.load_button.setEnabled(True)
                self.prediction_output_label.setText("Predicted output file: Terminated")
                self.status_label.setText("Status: Execution Stopped by User.")

    # -------------- run prediction (QProcess) --------------
    def _initiate_run_prediction_script(self):
        self.session.logger.info("****** Running prediction ******")

        # explicitly save path values to QSettings in case user typed them manually rather than using Browse
        self.settings.setValue("python_env", self.python_env_edit.text().strip())
        self.settings.setValue("base_repo_path", self.base_repo_path_edit.text().strip())
        self.settings.setValue("script_output_csv_path", self.script_output_csv_path_edit.text().strip())
        self.settings.setValue("checkpoint_path", self.checkpoint_path_edit.text().strip())
        self.settings.setValue("config_file", self.config_file_edit.text().strip())
        self.settings.setValue("base_model_loc", self.base_model_loc_edit.text().strip())

        # cleanup old temp
        if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
            try:
                shutil.rmtree(self._temp_dir_to_cleanup)
            except Exception as e:
                self.session.logger.error(f"Failed to remove previous temp directory: {e}")
            finally:
                self._temp_dir_to_cleanup = None

        self.base_repo_path = self.base_repo_path_edit.text().strip()
        self.python_script_path = os.path.normpath(os.path.join(self.base_repo_path, "src", "esm_msr", "inference.py"))
        self.python_env = self.python_env_edit.text().strip()

        model_id = self.pred_model_combobox.currentData()
        models = self.session.models.list(type=Structure)
        current_model = next((m for m in models if m.id_string == model_id), None)
        
        if not current_model:
            self.status_label.setText("Error: Selected model not found. Please ensure a valid model is open.")
            raise AssertionError("Selected model not found. Please ensure a valid model is open.")

        # Create temporary PDB input for the prediction script
        try:
            self._temp_dir_to_cleanup = tempfile.mkdtemp(prefix="chimerax_rsv_input_")
            temp_model_filename = f"current_model_input_{current_model.id_string.replace(':', '_').replace('/', '_')}.pdb"
            _temp_script_input_structure_path = os.path.join(self._temp_dir_to_cleanup, temp_model_filename)
            run(self.session, f"save \"{_temp_script_input_structure_path}\" models #{current_model.id_string} format pdb")
            self.script_input_structure_path = _temp_script_input_structure_path
        except Exception as e:
            self.session.logger.error(f"Failed to save temporary PDB structure: {e}")
            if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
                shutil.rmtree(self._temp_dir_to_cleanup)
                self._temp_dir_to_cleanup = None
            raise AssertionError(f"Failed to write temporary PDB. Aborting. Trace: {e}")

        # Program selection
        program = None
        args = []
        env = self.python_env
        
        if env:
            if os.path.isdir(env):
                cand = [
                    os.path.join(env, 'python.exe'),
                    os.path.join(env, 'Scripts', 'python.exe'),
                    os.path.join(env, 'bin', 'python')
                ]
                pyexe = next((p for p in cand if os.path.isfile(p)), None)
                
                if pyexe:
                    program = pyexe
                    args.append('-u') # Force unbuffered stdout/stderr
                else:
                    self.session.logger.warning(f"Could not find python binary inside {env}. Falling back to default 'python'.")
                    program = 'python'
                    args.append('-u')
            else:
                program = 'conda'
                # Conda explicitly captures output by default. We must disable it and pass -u to python.
                args.extend(['run', '--no-capture-output', '-n', env, 'python', '-u'])
        else:
            program = 'python'
            args.append('-u')

        # Script Arguments Assembly
        script_args = [self.python_script_path]

        # Critical Paths
        if not self.script_output_csv_path_edit.text().strip():
            raise AssertionError("An Output CSV path is required to run inference.")
        script_args += ['--output_csv', self.script_output_csv_path_edit.text().strip()]

        if not self.checkpoint_path_edit.text().strip():
            #raise AssertionError("A checkpoint path is required to run inference.")
            self.session.logger.warning("No checkpoint was provided; running in zero-shot mode.")
        script_args += ['--checkpoint_path', self.checkpoint_path_edit.text().strip()]

        # Target Config - Enforced by Radio Buttons
        if self.radio_full.isChecked():
            script_args += ['--pdb_file', self.script_input_structure_path]
            script_args += ['--code', os.path.splitext(current_model.name)[0] if current_model.name else 'protein']
            script_args += ['--chain', self.pred_chain_id_combobox.currentText().strip()]
            script_args += ['--mode', self.mode_combobox.currentText()]
            selected_res = self.selected_residues_edit.text().strip()
            if selected_res:
                if self.screen_except_checkbox.isChecked():
                    script_args += ['--screen_residues_except', selected_res]
                else:
                    script_args += ['--screen_residues', selected_res]
                    
        elif self.radio_csv.isChecked():
            subset_df_val = self.subset_df_edit.text().strip()
            if not subset_df_val:
                raise AssertionError("CSV input selected but no path provided.")
            script_args += ['--input_csv', subset_df_val]
            
        elif self.radio_explicit.isChecked():
            script_args += ['--pdb_file', self.script_input_structure_path]
            script_args += ['--code', os.path.splitext(current_model.name)[0] if current_model.name else 'protein']
            script_args += ['--chain', self.pred_chain_id_combobox.currentText().strip()]
            mutations_val = self.mutations_edit.text().strip()
            if not mutations_val:
                raise AssertionError("Explicit mutations selected but none provided.")
            script_args += ['--mutations', mutations_val]

        # Engine Params
        script_args += ['--batch_size', str(self.batch_size_spinbox.value())]
        script_args += ['--device', self.device_combobox.currentText()]
        
        mask_strat = self.mask_strategy_combobox.currentText()
        if mask_strat != 'Default (unmasked)':
            script_args += ['--mask_strategy', mask_strat]

        # Distance threshold
        if self.enable_distance_checkbox.isChecked():
            script_args += ['--distance_threshold', str(self.distance_threshold_spinbox.value())]
            script_args += ['--calculate_distances']
        else:
            script_args += ['--distance_threshold', '-1']

        backbone_mut = self.backbone_mutation_edit.text().strip()
        if backbone_mut:
            script_args += ['--backbone_mutation', backbone_mut]

        # Model Source Resolution (HF vs Base Model)
        if self.radio_hf.isChecked():
            hf_token_val = self.hf_token_edit.text().strip()
            if hf_token_val:
                script_args += ['--hf_token', hf_token_val]
            else:
                self.session.logger.warning("HuggingFace Token selected but left empty. Passing without token.")
        elif self.radio_base_loc.isChecked():
            base_model_loc_val = self.base_model_loc_edit.text().strip()
            if not base_model_loc_val:
                raise AssertionError("Base Model Location selected but no path provided.")
            script_args += ['--base_model_loc', base_model_loc_val]

        # Model Configs
        config_file = self.config_file_edit.text().strip()
        if config_file:
            if config_file.lower().endswith('.json'):
                script_args += ['--lora_config', config_file]
            elif config_file.lower().endswith(('.yaml', '.yml')):
                script_args += ['--hparams_path', config_file]
            else:
                self.session.logger.warning(f"Unrecognized config extension for {config_file}. Assuming YAML.")
                script_args += ['--hparams_path', config_file + '.yaml']

        script_args += ['--quaternary_mode', self.quaternary_mode_combobox.currentText()]

        # Flags
        if self.skip_additive_checkbox.isChecked(): script_args += ['--skip_additive']
        if self.skip_reverse_checkbox.isChecked(): script_args += ['--skip_reverse']

        full_args = args + script_args if program != 'python' else script_args

        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.session.logger.warning("Prediction already running; ignoring new request.")
            return
            
        self.proc = QProcess()
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.errorOccurred.connect(self._on_proc_error)
        self.proc.finished.connect(self._on_proc_finished)

        self.status_label.setText("Status: Running prediction script...")
        
        # Toggle RUN/STOP buttons
        self.run_prediction_button.setVisible(False)
        self.stop_prediction_button.setVisible(True)
        self.load_button.setEnabled(False)
        self.prediction_output_label.setText("Predicted output file: Processing...")

        #self.session.logger.info(f"QProcess starting: {program} {' '.join(full_args)}")
        self.proc.start(program, full_args)

    # QProcess slots
    def _on_proc_output(self):
        try:
            out = bytes(self.proc.readAllStandardOutput()).decode('utf-8', errors='replace')
            if not out:
                return
                
            # tqdm uses \r to overwrite lines. Replace with \n so we can split the stream cleanly.
            chunks = out.replace('\r', '\n').split('\n')
            
            for chunk in chunks:
                clean_chunk = chunk.strip()
                if not clean_chunk: 
                    continue
                    
                # Heuristic: Detect tqdm progress bars using common structural markers
                if '|' in clean_chunk and ('it/s' in clean_chunk or 's/it' in clean_chunk or '%' in clean_chunk):
                    # Route to the GUI status label to simulate in-place updates
                    self.status_label.setText(f"Status: {clean_chunk}")
                else:
                    # Route standard logging back to the ChimeraX console
                    self.session.logger.info(clean_chunk)
                    
        except Exception as e:
            self.session.logger.error(f"Error reading process output: {e}")

    def _on_proc_error(self, err):
        self.session.logger.error(f"Prediction process encountered QProcess error: {err}")
        self.status_label.setText(f"Status: Process Error ({err}). See Log.")
        self.run_prediction_button.setVisible(True)
        self.stop_prediction_button.setVisible(False)

    def _on_proc_finished(self, exitCode, exitStatus):
        self.run_prediction_button.setEnabled(True)
        self.run_prediction_button.setVisible(True)
        self.stop_prediction_button.setVisible(False)
        self.stop_prediction_button.setEnabled(True) # Re-enable for next execution
        self.load_button.setEnabled(True)
        
        if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
            try:
                shutil.rmtree(self._temp_dir_to_cleanup)
            except Exception as e:
                self.session.logger.warning(f"Could not remove temporary directory {self._temp_dir_to_cleanup}: {e}")
            self._temp_dir_to_cleanup = None

        if exitStatus == QProcess.NormalExit and exitCode == 0:
            self.predicted_output_path = self.script_output_csv_path_edit.text()
            self.prediction_output_label.setText(f"Predicted output: {os.path.basename(self.predicted_output_path)}")
            self.status_label.setText("Status: Prediction script completed. Load CSV to visualize.")
        else:
            self.prediction_output_label.setText("Predicted output file: Failed/Terminated")
            self.status_label.setText(f"Status: Script Error/Stop (exit={exitCode}). See log.")

        try:
            self.proc.deleteLater()
        except Exception as e:
            self.session.logger.error(f"Failed cleaning up QProcess: {e}")
        self.proc = None

    # -------------- CSV parse + viz --------------
    def _build_viz_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        self.csv_label = QLabel("No CSV loaded.")
        layout.addWidget(self.csv_label)

        # --- Display Mode Selection ---
        mode_group = QGroupBox("Display Mode and Priority")
        mode_layout = QHBoxLayout()
        
        mode_layout.addWidget(QLabel("Mode:"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems([
            "Singles", 
            "WT epistasis", 
            "MT epistasis"
        ])
        self.display_mode_combo.currentTextChanged.connect(self._on_display_mode_changed)
        mode_layout.addWidget(self.display_mode_combo)
        
        mode_layout.addSpacing(10)
        mode_layout.addWidget(QLabel("Select Pairs By:"))
        self.pair_selection_combo = QComboBox()
        self.pair_selection_combo.addItems(["Epistasis", "Stability"])
        mode_layout.addWidget(self.pair_selection_combo)

        mode_layout.addSpacing(10)
        mode_layout.addWidget(QLabel("Display Priority:"))
        self.display_priority_combo = QComboBox()
        self.display_priority_combo.addItems(["High score", "Low score", "Magnitude"])
        mode_layout.addWidget(self.display_priority_combo)
        
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # --- Global Filters Group ---
        filters_group = QGroupBox("Global Filters")
        filters_layout = QHBoxLayout()
        
        self.exclude_wt_cys_checkbox = QCheckBox("No WT Cys")
        self.exclude_wt_pro_checkbox = QCheckBox("No WT Pro")
        self.exclude_mut_cys_checkbox = QCheckBox("No Mut Cys")
        self.exclude_mut_pro_checkbox = QCheckBox("No Mut Pro")

        filters_layout.addWidget(self.exclude_wt_cys_checkbox)
        filters_layout.addWidget(self.exclude_wt_pro_checkbox)
        filters_layout.addSpacing(10)
        filters_layout.addWidget(self.exclude_mut_cys_checkbox)
        filters_layout.addWidget(self.exclude_mut_pro_checkbox)
        filters_layout.addStretch()

        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)

        # --- Score Selection Group ---
        score_group = QGroupBox("Score Selection (Additive Base)")
        score_layout = QVBoxLayout()
        score_group.setLayout(score_layout)

        self.viz_score_btn_group = QButtonGroup()

        self.radio_dual_view = QRadioButton("Dual-view predictions (recommended) (required for epistasis)")
        self.viz_score_btn_group.addButton(self.radio_dual_view)
        score_layout.addWidget(self.radio_dual_view)

        self.radio_wt_lora = QRadioButton("WT LoRA predictions (additive)")
        self.viz_score_btn_group.addButton(self.radio_wt_lora)
        score_layout.addWidget(self.radio_wt_lora)

        self.radio_mt_lora = QRadioButton("MT LoRA predictions (not recommended)")
        self.viz_score_btn_group.addButton(self.radio_mt_lora)
        score_layout.addWidget(self.radio_mt_lora)

        self.radio_dual_view.setChecked(True) # Default
        layout.addWidget(score_group)

        # --- Thresholds, Contacts, and Networks Group ---
        tcn_group = QGroupBox("Global Thresholds and Networks (kcal/mol, positive=stable)")
        tcn_layout = QVBoxLayout()
        
        # Row 1: General Thresholds and Transparency
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Pos Threshold:"))
        self.pos_threshold_spinbox = QDoubleSpinBox()
        self.pos_threshold_spinbox.setRange(0.0, 1000.0)
        self.pos_threshold_spinbox.setSingleStep(0.05)
        self.pos_threshold_spinbox.setValue(0.0)
        thresh_row.addWidget(self.pos_threshold_spinbox)
        
        thresh_row.addWidget(QLabel("Neg Threshold:"))
        self.neg_threshold_spinbox = QDoubleSpinBox()
        self.neg_threshold_spinbox.setRange(-1000.0, 0.0)
        self.neg_threshold_spinbox.setSingleStep(0.05)
        self.neg_threshold_spinbox.setValue(0.0)
        thresh_row.addWidget(self.neg_threshold_spinbox)

        thresh_row.addSpacing(20)
        thresh_row.addWidget(QLabel("Non-Target Chain Transp %:"))
        self.non_target_alpha_spinbox = QSpinBox()
        self.non_target_alpha_spinbox.setRange(0, 100)
        self.non_target_alpha_spinbox.setSingleStep(10)
        self.non_target_alpha_spinbox.setValue(90)
        thresh_row.addWidget(self.non_target_alpha_spinbox)
        
        thresh_row.addStretch()
        tcn_layout.addLayout(thresh_row)
        
        # Row 2: Network Edges
        net_row = QHBoxLayout()
        net_row.addWidget(QLabel("Max Interactions per Position:"))
        self.epi_max_edges = QSpinBox()
        self.epi_max_edges.setRange(1, 20)
        self.epi_max_edges.setValue(1)
        net_row.addWidget(self.epi_max_edges)
        net_row.addStretch()
        tcn_layout.addLayout(net_row)
        
        # Row 3: Contacts Visualization
        contacts_row = QHBoxLayout()
        self.show_contacts_checkbox = QCheckBox("Visualize residues within:")
        self.show_contacts_checkbox.setChecked(False)
        contacts_row.addWidget(self.show_contacts_checkbox)

        self.contact_distance_spinbox = QDoubleSpinBox()
        self.contact_distance_spinbox.setRange(0.0, 50.0)
        self.contact_distance_spinbox.setSingleStep(0.5)
        self.contact_distance_spinbox.setValue(3.0)
        self.contact_distance_spinbox.setEnabled(False)
        contacts_row.addWidget(self.contact_distance_spinbox)
        
        contacts_row.addWidget(QLabel("Å of displayed mutants"))
        contacts_row.addStretch()
        tcn_layout.addLayout(contacts_row)
        
        tcn_group.setLayout(tcn_layout)
        layout.addWidget(tcn_group)

        self.show_contacts_checkbox.toggled.connect(self.contact_distance_spinbox.setEnabled)

        self.color_backbone_checkbox = QCheckBox("Color Backbone by Highest Additive ΔΔG")
        self.color_backbone_checkbox.setChecked(False)
        layout.addWidget(self.color_backbone_checkbox)

        # --- Consolidated Styling Group ---
        styling_group = QGroupBox("Rendering and Styling")
        styling_layout = QVBoxLayout()
        
        wt_row = QHBoxLayout()
        wt_row.addWidget(QLabel("WT Color (name/RGB):"))
        self.wt_color_edit = QLineEdit("white")
        wt_row.addWidget(self.wt_color_edit)
        wt_row.addWidget(QLabel("Style:"))
        self.wt_style_combo = QComboBox()
        self.wt_style_combo.addItems(["stick", "ball", "sphere", "wire"])
        wt_row.addWidget(self.wt_style_combo)
        wt_row.addWidget(QLabel("Transp %:"))
        self.wt_stick_alpha_spinbox = QSpinBox()
        self.wt_stick_alpha_spinbox.setRange(0, 100)
        self.wt_stick_alpha_spinbox.setValue(70)
        wt_row.addWidget(self.wt_stick_alpha_spinbox)
        styling_layout.addLayout(wt_row)

        mut_row = QHBoxLayout()
        mut_row.addWidget(QLabel("Mut Color (name/RGB):"))
        self.mut_color_edit = QLineEdit("")
        self.mut_color_edit.setPlaceholderText("LEAVE BLANK FOR ADDITIVE SCORE")
        mut_row.addWidget(self.mut_color_edit)
        mut_row.addWidget(QLabel("Style:"))
        self.mut_style_combo = QComboBox()
        self.mut_style_combo.addItems(["stick", "ball", "sphere", "wire"])
        mut_row.addWidget(self.mut_style_combo)
        mut_row.addWidget(QLabel("Transp %:"))
        self.mut_stick_alpha_spinbox = QSpinBox()
        self.mut_stick_alpha_spinbox.setRange(0, 100)
        self.mut_stick_alpha_spinbox.setValue(30)
        mut_row.addWidget(self.mut_stick_alpha_spinbox)
        styling_layout.addLayout(mut_row)
        
        contact_row = QHBoxLayout()
        contact_row.addWidget(QLabel("Contact Color:"))
        self.contact_color_edit = QLineEdit("purple")
        contact_row.addWidget(self.contact_color_edit)
        contact_row.addWidget(QLabel("Style:"))
        self.contact_style_combo = QComboBox()
        self.contact_style_combo.addItems(["ball", "sphere", "stick", "wire"])
        contact_row.addWidget(self.contact_style_combo)
        contact_row.addWidget(QLabel("Transp %:"))
        self.contact_stick_alpha_spinbox = QSpinBox()
        self.contact_stick_alpha_spinbox.setRange(0, 100)
        self.contact_stick_alpha_spinbox.setValue(70)
        contact_row.addWidget(self.contact_stick_alpha_spinbox)
        styling_layout.addLayout(contact_row)

        styling_group.setLayout(styling_layout)
        layout.addWidget(styling_group)
        layout.addStretch()

        # Initialize UI State
        self._on_display_mode_changed()

    def _on_display_mode_changed(self):
        mode = self.display_mode_combo.currentText()
        
        if mode == "Singles":
            self.radio_wt_lora.setEnabled(True)
            self.radio_mt_lora.setEnabled(True)
            self.radio_dual_view.setEnabled(True)
            self.color_backbone_checkbox.setEnabled(True)
            self.epi_max_edges.setEnabled(False)
            self.display_priority_combo.setEnabled(False)
            self.pair_selection_combo.setEnabled(False)
            
        elif mode == "WT epistasis":
            self.radio_dual_view.setChecked(True)
            self.radio_wt_lora.setEnabled(False)
            self.radio_mt_lora.setEnabled(False)
            self.radio_dual_view.setEnabled(True)
            self.color_backbone_checkbox.setEnabled(True)
            self.epi_max_edges.setEnabled(True)
            self.display_priority_combo.setEnabled(True)
            self.pair_selection_combo.setEnabled(True)
            
        elif mode == "MT epistasis":
            self.radio_dual_view.setChecked(True)
            self.radio_wt_lora.setEnabled(False)
            self.radio_mt_lora.setEnabled(False)
            self.radio_dual_view.setEnabled(True)
            self.color_backbone_checkbox.setEnabled(False)
            self.epi_max_edges.setEnabled(True)
            self.display_priority_combo.setEnabled(True)
            self.pair_selection_combo.setEnabled(True)

    def _handle_load_and_visualize(self):
        self.session.logger.info("****** _handle_load_and_visualize called ******")
        w = getattr(self.session.ui, 'main_window', None)
        initial_dir, default_filename = "", ""
        if getattr(self, 'predicted_output_path', None) and os.path.exists(self.predicted_output_path):
            initial_dir, default_filename = os.path.dirname(self.predicted_output_path), self.predicted_output_path
        elif getattr(self, 'script_output_csv_path_edit', None) and self.script_output_csv_path_edit.text() and os.path.exists(self.script_output_csv_path_edit.text()):
            default_filename = self.script_output_csv_path_edit.text()
            initial_dir = os.path.dirname(default_filename)
        elif getattr(self, 'loaded_csv_path', None) and os.path.exists(self.loaded_csv_path):
            initial_dir = os.path.dirname(self.loaded_csv_path)

        fp, _ = QFileDialog.getOpenFileName(w, "Open Residue Score CSV", default_filename or initial_dir, "CSV Files (*.csv);;All Files (*)")
        if fp:
            self.loaded_csv_path = fp
            self.csv_label.setText(f"Loaded: {os.path.basename(fp)}")
            self.status_label.setText("Status: Parsing CSV...")
            
            if self._parse_csv(fp):
                self.status_label.setText("Status: Applying visualization...")
                mode = self.display_mode_combo.currentText()
                if mode == "Singles":
                    self._apply_singles()
                elif "epistasis" in mode:
                    self._apply_epistasis()
            else:
                if not self.status_label.text().startswith("Status: Error"):
                    self.status_label.setText("Status: Error parsing CSV. Check Log.")
        else:
            self.status_label.setText("Status: CSV loading cancelled.")

    def _parse_csv(self, filepath):
        try:
            import pandas as pd
        except ImportError:
            msg = "Pandas is not installed in the ChimeraX Python environment. Please run 'pip install pandas' in the ChimeraX shell."
            self.session.logger.error(msg)
            raise AssertionError(msg)

        self.residue_scores_data = {}
        self.all_singles_scores = {}
        self.epistasis_df = None
        mode = self.display_mode_combo.currentText()

        try:
            df = pd.read_csv(filepath)
            df.columns = [c.lower().strip() for c in df.columns]

            def parse_mut_string(m_str):
                muts = []
                for m in str(m_str).split(':'):
                    if len(m) < 3: continue
                    match = re.match(r"([a-zA-Z])(\d+)([a-zA-Z])", m.strip())
                    if match:
                        wt, pos, mt = match.groups()
                        muts.append({'wt': wt, 'pos': int(pos), 'mut': mt})
                return muts

            # REPLACED mut_type with mut_type_pdb as requested
            if 'mut_type_pdb' not in df.columns:
                raise AssertionError("Missing required column 'mut_type_pdb'. Ensure your CSV was generated with the updated inference.py script.")
            df['parsed_muts'] = df['mut_type_pdb'].apply(parse_mut_string)

            # Apply Global Exclusions explicitly for WT vs Mut
            def has_excluded_aa(muts, excl_wt, excl_mt):
                for m in muts:
                    if m['wt'] in excl_wt or m['mut'] in excl_mt: return True
                return False

            ex_wt = set()
            ex_mt = set()
            if self.exclude_wt_cys_checkbox.isChecked(): ex_wt.add('C')
            if self.exclude_mut_cys_checkbox.isChecked(): ex_mt.add('C')
            if self.exclude_wt_pro_checkbox.isChecked(): ex_wt.add('P')
            if self.exclude_mut_pro_checkbox.isChecked(): ex_mt.add('P')
            
            if ex_wt or ex_mt:
                df = df[~df['parsed_muts'].apply(lambda x: has_excluded_aa(x, ex_wt, ex_mt))]
                if df.empty:
                    raise AssertionError("Global filters removed all mutations from the loaded CSV.")

            # Base check for singles columns, which are always required now
            base_req = {'pdb_file', 'code', 'chain', 'mut_type_pdb', 'wt_lora_pred'}
            if not base_req.issubset(set(df.columns)):
                raise AssertionError(f"Missing required base columns in CSV. Expected at least: {base_req}.")

            if self.radio_wt_lora.isChecked(): target_score_col = 'wt_lora_pred'
            elif self.radio_mt_lora.isChecked(): target_score_col = 'mt_lora_pred'
            else: target_score_col = 'combined_pred'

            if target_score_col not in df.columns:
                raise AssertionError(f"Selected score '{target_score_col}' not found in CSV.")

            # ALWAYS extract singles logic to support additive atom/backbone coloring
            singles_df = df[df['parsed_muts'].apply(len) == 1].copy()
            if singles_df.empty:
                raise AssertionError("No single mutations found in the CSV. Single mutation data is REQUIRED in all modes to color individual mutant sidechains. Ensure you ran inference with mode 'both' or 'singles'.")

            singles_df['chain_id'] = singles_df['chain'].astype(str).str.strip()
            singles_df['pos1_pdb'] = singles_df['parsed_muts'].apply(lambda x: x[0]['pos'])
            singles_df['mut1'] = singles_df['parsed_muts'].apply(lambda x: x[0]['mut'])
            singles_df['viz_score'] = singles_df[target_score_col]

            # Save full dictionary of individual mutations mapped to scores for exact retrieval
            for _, row in singles_df.iterrows():
                c = str(row['chain_id']).strip()
                p = int(row['pos1_pdb'])
                m = str(row['mut1']).upper()
                s = float(row['viz_score'])
                self.all_singles_scores[(c, p, m)] = s

            pivot_df = singles_df.pivot_table(index=['chain_id', 'pos1_pdb'], columns='mut1', values='viz_score')
            if pivot_df.empty: raise AssertionError("Parsed CSV resulted in an empty pivot table.")

            max_scores = pivot_df.max(axis=1)
            top_aas = pivot_df.idxmax(axis=1)

            count = 0
            for idx in max_scores.index:
                chain_id_val, pos = idx
                score = max_scores[idx]
                if pd.isna(score) or pd.isna(top_aas[idx]): continue
                if score != 0.0:
                    self.residue_scores_data[(chain_id_val, int(pos))] = (float(score), str(top_aas[idx]).upper())
                    count += 1
            
            if count == 0: raise AssertionError("Parsed CSV, but no valid non-zero singles scores found.")

            if "epistasis" in mode:
                if 'combined_dddg_pred' not in df.columns:
                    raise AssertionError(f"{mode} mode missing 'combined_dddg_pred' column.")

                doubles_df = df[df['parsed_muts'].apply(len) == 2].copy()
                if doubles_df.empty:
                    raise AssertionError(f"{mode} mode requested, but no double mutations were found in the CSV.")
                    
                doubles_df['chain_id'] = doubles_df['chain'].astype(str).str.strip()
                doubles_df['pos1_pdb'] = doubles_df['parsed_muts'].apply(lambda x: x[0]['pos'])
                doubles_df['wt1'] = doubles_df['parsed_muts'].apply(lambda x: x[0]['wt'])
                doubles_df['mut1'] = doubles_df['parsed_muts'].apply(lambda x: x[0]['mut'])
                
                doubles_df['pos2_pdb'] = doubles_df['parsed_muts'].apply(lambda x: x[1]['pos'])
                doubles_df['wt2'] = doubles_df['parsed_muts'].apply(lambda x: x[1]['wt'])
                doubles_df['mut2'] = doubles_df['parsed_muts'].apply(lambda x: x[1]['mut'])
                
                doubles_df['dddg_pred'] = doubles_df['combined_dddg_pred']
                doubles_df['combined_pred'] = doubles_df['combined_pred']

                self.epistasis_df = doubles_df
                self.session.logger.info(f"Parsed {mode.split()[0]} dataframe with {len(doubles_df)} double mutation rows.")
                
            return True

        except Exception as e:
            self.session.logger.error(f"Error parsing CSV with Pandas: {e}")
            self.status_label.setText("Status: Error parsing CSV (see log).")
            raise AssertionError(f"CSV Parsing failed: {e}")

    # --- Unified Rendering Pipeline ---

    def _get_spec(self, model_id_string, res_keys):
        """Builds a robust ChimeraX spec string from a set of (chain, pos) tuples."""
        if not res_keys: return "None"
        by_chain = defaultdict(list)
        for c, p in res_keys:
            by_chain[c].append(str(p))
        specs = []
        for c, positions in by_chain.items():
            specs.append(f"#{model_id_string}/{c}:{','.join(positions)}")
        return " | ".join(specs)

    def _setup_base_wt_model(self):
        """Fetches the WT model and clears out legacy visualization state across all previous layers."""
        
        # Ensure mutated tracking list exists to avoid init discrepancies
        if not hasattr(self, 'mutated_model_id_strings'):
            self.mutated_model_id_strings = []
            
        # Clean up legacy single string if present
        if getattr(self, 'mutated_model_id_string', None):
            mid = self.mutated_model_id_string
            if any(m.id_string == mid for m in self.session.models.list()):
                run(self.session, f"close #{mid}")
            self.mutated_model_id_string = None

        wt_candidates = [m for m in self.session.models.list(type=Structure)
                         if m.id_string not in self.mutated_model_id_strings]
        
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        if not wt_model and wt_candidates: wt_model = wt_candidates[0]
        if not wt_model: raise AssertionError("No suitable WT model open.")

        for mid in self.mutated_model_id_strings:
            if any(m.id_string == mid for m in self.session.models.list()):
                run(self.session, f"close #{mid}")
        self.mutated_model_id_strings = []

        run(self.session, f"color #{wt_model.id_string} white")
        run(self.session, f"ribbon style #{wt_model.id_string}")
        run(self.session, f"hide #{wt_model.id_string} atoms")
        
        return wt_model

    def _create_mutated_model(self, wt_model, suffix="viz"):
        """Clones the WT model cleanly via a temporary file to preserve polymer metadata."""
        temp_pdb = os.path.join(tempfile.gettempdir(), f"clone_{wt_model.id_string.replace(':', '_')}.pdb")
        run(self.session, f"save {temp_pdb} models #{wt_model.id_string} format pdb")
        run(self.session, f"open {temp_pdb} name \"{wt_model.name}_{suffix}\"")
        
        atomic_models = self.session.models.list(type=Structure)
        mut_model = atomic_models[-1]
        
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)

        mut_id = mut_model.id_string.split('.')[0]
        run(self.session, f"color #{mut_id} white")
        run(self.session, f"transparency #{mut_id} 100 target a")
        run(self.session, f"hide #{mut_id} atoms")
        
        return mut_model

    def _apply_swapaa(self, mutated_model_id, mutated_model, mutation_plan):
        """Applies sidechain swaps in the cloned model based on the unified mutation plan."""
        for (chain_val, pos), tgt_aa in mutation_plan.items():
            res_wt = next((r for r in mutated_model.residues if r.number == pos and r.chain_id == chain_val), None)
            if res_wt and ONE_TO_THREE_LETTER_AA.get(tgt_aa, '') != res_wt.name:
                spec = f"#{mutated_model_id}/{chain_val}:{pos}"
                run(self.session, f"swapaa {spec} {ONE_TO_THREE_LETTER_AA.get(tgt_aa, 'ALA').lower()} log false")

    def _apply_contacts(self, wt_model, target_keys, target_spec_bare):
        """Isolates the contact finding and styling logic across any mode."""
        if not self.show_contacts_checkbox.isChecked() or not target_keys:
            return

        dist = self.contact_distance_spinbox.value()
        contact_query = f"({target_spec_bare}) @<{dist} & #{wt_model.id_string} & protein"
        
        wt_mut_spec_bare = self._get_spec(wt_model.id_string, target_keys)
        if wt_mut_spec_bare != "None":
            contact_query += f" & ~({wt_mut_spec_bare})"
            
        from chimerax.atomic import selected_atoms
        contact_keys = set()
        run(self.session, f"select {contact_query}")
        sel_atoms = selected_atoms(self.session)
        if sel_atoms and len(sel_atoms) > 0:
            for r in set(sel_atoms.residues):
                if (r.chain_id, r.number) not in target_keys:
                    contact_keys.add((r.chain_id, r.number))
        run(self.session, "select clear")

        if contact_keys:
            spec = self._get_spec(wt_model.id_string, contact_keys)
            c_color = self.contact_color_edit.text().strip() or "purple"
            c_alpha = self.contact_stick_alpha_spinbox.value()

            try: run(self.session, f"color ({spec}) {c_color} target a")
            except: run(self.session, f"color ({spec}) purple target a")
            run(self.session, f"color ({spec}) & ~C byelement target a")
            run(self.session, f"show ({spec}) atoms")
            
            run(self.session, f"style ({spec}) stick")
            run(self.session, f"size ({spec}) stickRadius 0.15") 
            run(self.session, f"transparency ({spec}) {c_alpha} target ab")
            
        if contact_query:
            c_style = self.contact_style_combo.currentText()
            run(self.session, f"style {contact_query} & ~backbone {c_style}")
            if c_style == "stick":
                run(self.session, f"size {contact_query} stickRadius 0.25")
            run(self.session, f"transparency {contact_query} 0 target a")

    def _resolve_and_apply_styles(self, wt_model, mut_model_ids, mutation_plans):
        """
        Calculates mutually exclusive rendering sets and apply styles across ALL layers.
        Priority: Mutants (highest) -> Contacts -> Wild-Type.
        Mutant colors are optionally overridden via UI, otherwise managed by additive score.
        """
        all_mutant_keys = set()
        for plan in mutation_plans:
            all_mutant_keys.update(plan.keys())

        wt_color = self.wt_color_edit.text().strip() or "white"
        wt_style = self.wt_style_combo.currentText()
        wt_alpha = self.wt_stick_alpha_spinbox.value()
        
        mut_color = self.mut_color_edit.text().strip()
        mut_style = self.mut_style_combo.currentText()
        mut_alpha = self.mut_stick_alpha_spinbox.value()

        # A. Style the Primary Mutants inside each Mutated Model Layer
        mut_specs = []
        for mut_id, plan in zip(mut_model_ids, mutation_plans):
            mutant_keys = set(plan.keys())
            if mutant_keys:
                spec = self._get_spec(mut_id, mutant_keys)
                mut_specs.append(spec)
                run(self.session, f"size ({spec}) stickRadius 0.2")
                
                # Apply explicit user color if provided, else ensure non-carbons format properly for byattribute fallback
                if mut_color:
                    try: run(self.session, f"color ({spec}) {mut_color} target a")
                    except: run(self.session, f"color ({spec}) orange target a")
                
                run(self.session, f"color ({spec}) & ~C byelement target a")
                run(self.session, f"show ({spec}) atoms")
                run(self.session, f"style ({spec}) {mut_style}")
                run(self.session, f"transparency ({spec}) {mut_alpha} target a")
            
        # B. Style the corresponding WT Ghost Models underneath the mutants globally
        if all_mutant_keys:
            spec = self._get_spec(wt_model.id_string, all_mutant_keys)
            run(self.session, f"size ({spec}) stickRadius 0.2")
            try: run(self.session, f"color ({spec}) {wt_color} target a")
            except: run(self.session, f"color ({spec}) white target a")
            run(self.session, f"color ({spec}) & ~C byelement target a")
            run(self.session, f"show ({spec}) atoms")
            run(self.session, f"style ({spec}) {wt_style}")
            run(self.session, f"transparency ({spec}) {wt_alpha} target a")
            
        # C. Find and Apply Contacts based on total mutant span
        mut_spec_bare = " | ".join(mut_specs) if mut_specs else "None"
        self._apply_contacts(wt_model, all_mutant_keys, mut_spec_bare)

    def _draw_epistasis_line(self, mut_model_id, c1, p1, c2, p2, score, max_abs_score, thresh_pos, thresh_neg, model_residues):
        res1, res2 = model_residues.get((c1, p1)), model_residues.get((c2, p2))
        if not res1 or not res2 or not res1.atoms or not res2.atoms: return

        # Explicitly exclude all backbone atoms. No fallback.
        atoms1 = [a for a in res1.atoms if a.name not in ('N', 'CA', 'C', 'O')]
        atoms2 = [a for a in res2.atoms if a.name not in ('N', 'CA', 'C', 'O')]
        
        if not atoms1 or not atoms2:
            self.session.logger.warning(f"Skipping epistasis line between {c1}:{p1} and {c2}:{p2} because one or both lack sidechain heavy atoms (e.g. Glycine).")
            return
        
        coords1 = np.array([a.scene_coord for a in atoms1])
        coords2 = np.array([a.scene_coord for a in atoms2])
        dists = np.sqrt(np.sum((coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :])**2, axis=2))
        
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        a1, a2 = atoms1[min_idx[0]], atoms2[min_idx[1]]
        
        rel_thresh = thresh_pos if score > 0 else abs(thresh_neg)
        if max_abs_score <= rel_thresh:
            norm_score = 0.0
        else:
            norm_score = min(1.0, max(0.0, (abs(score) - rel_thresh) / (max_abs_score - rel_thresh)))
            
        radius_val = 0.05 + (0.95 * (norm_score ** 2)) 
        alpha = int(30 + 225 * norm_score)             
        
        c_inv = int(255 * (1.0 - norm_score))
        if score > 0:
            # Positive Epistasis -> Orange gradient interpolation
            color_spec = f"255,{int(255 - 90*norm_score)},{c_inv},{alpha}" 
        else:
            # Negative Epistasis -> Blue gradient interpolation
            color_spec = f"{c_inv},{c_inv},255,{alpha}" 
        
        cmd = f"pbond #{mut_model_id}/{c1}:{p1}@{a1.name} #{mut_model_id}/{c2}:{p2}@{a2.name} reveal true color {color_spec} radius {radius_val:.3f} name \"{score:.2f}\""
        run(self.session, cmd)

    def _apply_transparency_isolation(self, wt_model):
        """Dims non-target chains based on user preferences."""
        target_chain = self.pred_chain_id_combobox.currentText().strip()
        alpha_val = self.non_target_alpha_spinbox.value()
        if target_chain:
            exclude_spec = f"~#{wt_model.id_string}/{target_chain}"
            for mid in getattr(self, 'mutated_model_id_strings', []):
                exclude_spec += f" & ~#{mid}"
            if alpha_val >= 99:
                run(self.session, f"hide {exclude_spec}")
            else:
                run(self.session, f"show {exclude_spec} ribbons")
                run(self.session, f"transparency {exclude_spec} {alpha_val} target ac")

    # --- Execution Subroutines ---

    def _apply_singles(self):
        try:
            wt_model = self._setup_base_wt_model()
            pos_thresh = self.pos_threshold_spinbox.value()
            neg_thresh = self.neg_threshold_spinbox.value()
            
            mutation_plan = {}
            for (chain, pos), (score, tgt_aa) in self.residue_scores_data.items():
                if score >= pos_thresh or score <= neg_thresh:
                    mutation_plan[(chain, pos)] = tgt_aa

            mutated_model = self._create_mutated_model(wt_model, "singles_viz")
            mut_id = mutated_model.id_string.split('.')[0]
            self.mutated_model_id_strings.append(mut_id)
            
            self._apply_swapaa(mut_id, mutated_model, mutation_plan)
            self._resolve_and_apply_styles(wt_model, [mut_id], [mutation_plan])

            scores = [s for s, _ in self.residue_scores_data.values()]
            if scores:
                scores_abs = [abs(s) for s in scores]
                max_abs_add = float(np.percentile(scores_abs, 98))
                if max_abs_add <= 0.05: max_abs_add = max(scores_abs)
            else:
                max_abs_add = 0.01
            color_range = f"{-max_abs_add:.3f},{max_abs_add:.3f}"
            
            # Map exact singles scores directly to the mutants using the parsed singles mapping
            for (chain, pos), tgt_aa in mutation_plan.items():
                if (chain, pos, tgt_aa) not in self.all_singles_scores:
                    self.session.logger.warning(f"Single mutant score missing for {chain}:{pos}->{tgt_aa}. Defaulting to 0.0.")
                    exact_score = 0.0
                else:
                    exact_score = self.all_singles_scores[(chain, pos, tgt_aa)]
                run(self.session, f"setattr #{mut_id}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {exact_score} create true")

            # Map the aggregated max singles scores to the WT backbone
            for (chain, pos), (score, _) in self.residue_scores_data.items():
                run(self.session, f"setattr #{wt_model.id_string}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {score} create true")

            if not self.mut_color_edit.text().strip():
                mut_spec = self._get_spec(mut_id, mutation_plan.keys())
                if mut_spec != "None":
                    run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} ({mut_spec}) palette red:white:green range {color_range} key false target a")
                    run(self.session, f"color ({mut_spec}) & ~C byelement target a")

            if self.color_backbone_checkbox.isChecked():
                chains_present = set(c for c, p in self.residue_scores_data.keys())
                chain_spec = ",".join(chains_present)
                run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone palette red:white:green range {color_range} key false")

            try:
                run(self.session, f"key red:{-max_abs_add:.2f} white:0 green:{max_abs_add:.2f} pos 0.05,0.05")
            except Exception as e:
                self.session.logger.warning(f"Failed to draw colorbar key: {e}")

            self._apply_transparency_isolation(wt_model)
            self.status_label.setText("Status: Singles Visualization complete.")

        except Exception as e:
            self.session.logger.error(f"Visualization Error: {e}")
            self.status_label.setText("Status: Visualization Error.")

    def _apply_epistasis(self):
        try:
            wt_model = self._setup_base_wt_model()
            df = self.epistasis_df
            is_wt_epi = self.display_mode_combo.currentText() == "WT epistasis"
            
            if is_wt_epi:
                def is_valid_wt_epi(row):
                    valid1 = (row['wt1'] != 'A' and row['mut1'] == 'A') or (row['wt1'] == 'A' and row['mut1'] == 'G')
                    valid2 = (row['wt2'] != 'A' and row['mut2'] == 'A') or (row['wt2'] == 'A' and row['mut2'] == 'G')
                    return valid1 and valid2
                df = df[df.apply(is_valid_wt_epi, axis=1)].copy()
                if df.empty: raise AssertionError("WT Epistasis requires mutations to Alanine/Glycine. None found.")

            # Filtering Selection Metric
            metric_target = self.pair_selection_combo.currentText()
            if metric_target == "Stability":
                sort_col = 'combined_pred'
            else:
                sort_col = 'dddg_pred'

            pos_thresh = self.pos_threshold_spinbox.value()
            neg_thresh = self.neg_threshold_spinbox.value()
            filtered_df = df[(df[sort_col] >= pos_thresh) | (df[sort_col] <= neg_thresh)].copy()

            if filtered_df.empty:
                self.status_label.setText(f"Status: No residue pairs found outside the specified thresholds using {metric_target}.")
                return

            priority = self.display_priority_combo.currentText()
            if priority == "High score":
                sorted_df = filtered_df.sort_values(by=sort_col, ascending=False)
            elif priority == "Low score":
                sorted_df = filtered_df.sort_values(by=sort_col, ascending=True)
            elif priority == "Magnitude":
                filtered_df['abs_score_sort'] = filtered_df[sort_col].abs()
                sorted_df = filtered_df.sort_values(by='abs_score_sort', ascending=False)

            # Epistasis line colorbar is ALWAYS scaled to dddg_pred regardless of filtering metric
            max_abs_epi = float(sorted_df['dddg_pred'].abs().quantile(0.98))
            if pd.isna(max_abs_epi) or max_abs_epi <= 0.05:
                max_abs_epi = float(sorted_df['dddg_pred'].abs().max())
            if max_abs_epi == 0: max_abs_epi = 1.0

            # --- Base Additive Scoring Extraction ---
            scores = [s for s, _ in self.residue_scores_data.values()]
            if scores:
                scores_abs = [abs(s) for s in scores]
                max_abs_add = float(np.percentile(scores_abs, 98))
                if max_abs_add <= 0.05: max_abs_add = max(scores_abs)
            else:
                max_abs_add = 0.01
            color_range = f"{-max_abs_add:.3f},{max_abs_add:.3f}"

            if is_wt_epi:
                # -------------------------
                # WT EPISTASIS (NO LAYERS)
                # -------------------------
                participating_positions = set()
                pairs_to_draw = []
                connection_counts = defaultdict(int)
                max_edges = self.epi_max_edges.value()
                mutation_plan = {} # Used only for score lookup mapping

                for _, row in sorted_df.iterrows():
                    c1, p1, m1 = str(row['chain_id']).strip(), int(row['pos1_pdb']), str(row['mut1']).upper()
                    c2, p2, m2 = str(row['chain_id']).strip(), int(row['pos2_pdb']), str(row['mut2']).upper()
                    epi_score = float(row['dddg_pred'])
                    
                    if connection_counts[(c1, p1)] >= max_edges or connection_counts[(c2, p2)] >= max_edges: continue
                    
                    connection_counts[(c1, p1)] += 1
                    connection_counts[(c2, p2)] += 1
                    pairs_to_draw.append((c1, p1, m1, c2, p2, m2, epi_score))
                    participating_positions.add((c1, p1))
                    participating_positions.add((c2, p2))
                    mutation_plan[(c1, p1)] = m1
                    mutation_plan[(c2, p2)] = m2

                spec = self._get_spec(wt_model.id_string, participating_positions)
                if spec != "None":
                    for (chain, pos) in participating_positions:
                        tgt_aa = mutation_plan[(chain, pos)]
                        if (chain, pos, tgt_aa) not in self.all_singles_scores:
                            self.session.logger.warning(f"Single Ala/Gly score missing for {chain}:{pos}. Defaulting to 0.0.")
                            exact_score = 0.0
                        else:
                            exact_score = self.all_singles_scores[(chain, pos, tgt_aa)]
                        run(self.session, f"setattr #{wt_model.id_string}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {exact_score} create true")
                    
                    mut_style = self.mut_style_combo.currentText()
                    mut_alpha = self.mut_stick_alpha_spinbox.value()
                    
                    run(self.session, f"size ({spec}) stickRadius 0.2")
                    if not self.mut_color_edit.text().strip():
                        run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} ({spec}) palette red:white:green range {color_range} key false target a")
                        run(self.session, f"color ({spec}) & ~C byelement target a")
                    run(self.session, f"show ({spec}) atoms")
                    run(self.session, f"style ({spec}) {mut_style}")
                    run(self.session, f"transparency ({spec}) {mut_alpha} target a")
                    
                self._apply_contacts(wt_model, participating_positions, spec)
                
                if self.color_backbone_checkbox.isChecked():
                    chains_present = set(c for c, p in self.residue_scores_data.keys())
                    chain_spec = ",".join(chains_present)
                    for (chain, pos), (score, _) in self.residue_scores_data.items():
                        run(self.session, f"setattr #{wt_model.id_string}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {score} create true")
                    run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone palette red:white:green range {color_range} key false")

                model_residues = {(r.chain_id, r.number): r for r in wt_model.residues}
                for c1, p1, m1, c2, p2, m2, score in pairs_to_draw:
                    self._draw_epistasis_line(wt_model.id_string, c1, p1, c2, p2, score, max_abs_epi, pos_thresh, neg_thresh, model_residues)

            else:
                # -------------------------
                # MT EPISTASIS (LAYERS)
                # -------------------------
                layers = [] 
                pairs_to_draw = [] 
                connection_counts = defaultdict(int)
                max_edges = self.epi_max_edges.value()
                
                for _, row in sorted_df.iterrows():
                    c1, p1, m1 = str(row['chain_id']).strip(), int(row['pos1_pdb']), str(row['mut1']).upper()
                    c2, p2, m2 = str(row['chain_id']).strip(), int(row['pos2_pdb']), str(row['mut2']).upper()
                    epi_score = float(row['dddg_pred'])
                    
                    if connection_counts[(c1, p1)] >= max_edges or connection_counts[(c2, p2)] >= max_edges: continue
                    
                    assigned_layer = -1
                    for i, layer in enumerate(layers):
                        if layer.get((c1, p1), m1) == m1 and layer.get((c2, p2), m2) == m2:
                            assigned_layer = i
                            break
                    
                    if assigned_layer == -1:
                        assigned_layer = len(layers)
                        layers.append({})
                    
                    layers[assigned_layer][(c1, p1)] = m1
                    layers[assigned_layer][(c2, p2)] = m2
                    
                    connection_counts[(c1, p1)] += 1
                    connection_counts[(c2, p2)] += 1
                    pairs_to_draw.append((c1, p1, m1, c2, p2, m2, epi_score, assigned_layer))

                mutated_models_objs = []
                for i, layer in enumerate(layers):
                    mut_model = self._create_mutated_model(wt_model, f"epi_layer_{i}")
                    mut_id = mut_model.id_string.split('.')[0]
                    self.mutated_model_id_strings.append(mut_id)
                    self._apply_swapaa(mut_id, mut_model, layer)
                    mutated_models_objs.append(mut_model)
                    
                self._resolve_and_apply_styles(wt_model, self.mutated_model_id_strings, layers)
                
                for (chain, pos), (score, _) in self.residue_scores_data.items():
                    run(self.session, f"setattr #{wt_model.id_string}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {score} create true")

                for i, plan in enumerate(layers):
                    if not plan: continue
                    mut_id = self.mutated_model_id_strings[i]
                    
                    for (chain, pos), tgt_aa in plan.items():
                        if (chain, pos, tgt_aa) not in self.all_singles_scores:
                            self.session.logger.warning(f"Single mutant score missing for {chain}:{pos}->{tgt_aa}. Defaulting to 0.0.")
                            exact_score = 0.0
                        else:
                            exact_score = self.all_singles_scores[(chain, pos, tgt_aa)]
                        run(self.session, f"setattr #{mut_id}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {exact_score} create true")
                    
                    if not self.mut_color_edit.text().strip():
                        mut_spec = self._get_spec(mut_id, plan.keys())
                        if mut_spec != "None":
                            run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} ({mut_spec}) palette red:white:green range {color_range} key false target a")
                            run(self.session, f"color ({mut_spec}) & ~C byelement target a")

                for c1, p1, m1, c2, p2, m2, score, layer_idx in pairs_to_draw:
                    mut_id = self.mutated_model_id_strings[layer_idx]
                    model_residues = {(r.chain_id, r.number): r for r in mutated_models_objs[layer_idx].residues}
                    self._draw_epistasis_line(mut_id, c1, p1, c2, p2, score, max_abs_epi, pos_thresh, neg_thresh, model_residues)

            # Unified Colorbars with opposite positions along the bottom
            try:
                run(self.session, f"key red:{-max_abs_add:.2f} white:0 green:{max_abs_add:.2f} pos 0.05,0.05")
                run(self.session, f"key blue:{-max_abs_epi:.2f} white:0 orange:{max_abs_epi:.2f} pos 0.55,0.05")
            except Exception as e:
                self.session.logger.warning(f"Failed to draw dual colorbar keys: {e}")

            self._apply_transparency_isolation(wt_model)
            self.status_label.setText(f"Status: Epistasis Viz Complete ({len(pairs_to_draw)} interactions across {len(layers) if not is_wt_epi else 1} layers).")

        except Exception as e:
            self.session.logger.error(f"Critical failure in Epistasis Visualization: {e}")
            self.status_label.setText("Status: Visualization Error.")