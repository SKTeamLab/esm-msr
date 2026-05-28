# vim: set expandtab shiftwidth=4 softstop=4:
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import re

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

class ESM_MSR_VisualizerTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    display_name = "Residue Score Visualizer & Predictor"

    def __init__(self, session, tool_registered_name):
        super().__init__(session, tool_registered_name)
        #self.session.logger.info(f"****** RSVTool __init__ ({tool_registered_name}) ******")

        # 1. INITIALIZE SETTINGS
        self.settings = QSettings("ESM_MSR_Tools", "ESM_MSR_Visualizer")
        
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
        self.tabs.addTab(self.tab_io, "Execution & IO")

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
        self.hf_token_edit.setPlaceholderText("Optional: HuggingFace Token (for ESM3 weights)")
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
        self.mode_combobox.addItems(['singles', 'doubles', 'both'])
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
        is_doubles_mode = current_mode in ['doubles', 'both']
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

    def _build_viz_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        self.csv_label = QLabel("No CSV loaded.")
        layout.addWidget(self.csv_label)

        # --- NEW: Score Selection Group ---
        score_group = QGroupBox("Score Selection (Single Mutations)")
        score_layout = QVBoxLayout()
        score_group.setLayout(score_layout)

        self.viz_score_btn_group = QButtonGroup()

        self.radio_wt_lora = QRadioButton("WT LoRA predictions (additive)")
        self.viz_score_btn_group.addButton(self.radio_wt_lora)
        score_layout.addWidget(self.radio_wt_lora)

        self.radio_mt_lora = QRadioButton("MT LoRA predictions (not recommended)")
        self.viz_score_btn_group.addButton(self.radio_mt_lora)
        score_layout.addWidget(self.radio_mt_lora)

        self.radio_dual_view = QRadioButton("Dual-view predictions (recommended)")
        self.viz_score_btn_group.addButton(self.radio_dual_view)
        score_layout.addWidget(self.radio_dual_view)

        self.radio_dual_view.setChecked(True) # Default
        layout.addWidget(score_group)
        # ----------------------------------

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Score Threshold (kcal/mol, positive=stable):"))
        self.score_threshold_spinbox = QDoubleSpinBox()
        self.score_threshold_spinbox.setRange(-1000.0, 1000.0)
        self.score_threshold_spinbox.setSingleStep(0.05)
        self.score_threshold_spinbox.setValue(0.5)
        self.score_threshold_spinbox.setDecimals(2)
        threshold_layout.addWidget(self.score_threshold_spinbox)
        
        threshold_layout.addSpacing(20)
        
        threshold_layout.addWidget(QLabel("Non-Target Chain Transparency %:"))
        self.non_target_alpha_spinbox = QSpinBox()
        self.non_target_alpha_spinbox.setRange(0, 100)
        self.non_target_alpha_spinbox.setSingleStep(10)
        self.non_target_alpha_spinbox.setValue(90)
        threshold_layout.addWidget(self.non_target_alpha_spinbox)
        threshold_layout.addStretch()
        
        layout.addLayout(threshold_layout)

        self.color_backbone_checkbox = QCheckBox("Color Backbone by Highest ΔΔG vs Wild-Type")
        self.color_backbone_checkbox.setChecked(False)
        layout.addWidget(self.color_backbone_checkbox)

        stick_layout = QHBoxLayout()
        self.show_sticks_checkbox = QCheckBox("Show Sticks for Highest-Scoring Mutations")
        self.show_sticks_checkbox.setChecked(True)
        stick_layout.addWidget(self.show_sticks_checkbox)

        stick_layout.addWidget(QLabel("WT Stick Transp %:"))
        self.wt_stick_alpha_spinbox = QSpinBox()
        self.wt_stick_alpha_spinbox.setRange(0, 100)
        self.wt_stick_alpha_spinbox.setValue(70) 
        stick_layout.addWidget(self.wt_stick_alpha_spinbox)

        stick_layout.addWidget(QLabel("MUT Stick Transp %:"))
        self.mut_stick_alpha_spinbox = QSpinBox()
        self.mut_stick_alpha_spinbox.setRange(0, 100)
        self.mut_stick_alpha_spinbox.setValue(30)
        stick_layout.addWidget(self.mut_stick_alpha_spinbox)

        layout.addLayout(stick_layout)

        # Contact Selection Group
        contacts_layout = QHBoxLayout()
        self.show_contacts_checkbox = QCheckBox("Visualize residues within:")
        self.show_contacts_checkbox.setChecked(False)
        contacts_layout.addWidget(self.show_contacts_checkbox)

        self.contact_distance_spinbox = QDoubleSpinBox()
        self.contact_distance_spinbox.setRange(0.0, 50.0)
        self.contact_distance_spinbox.setSingleStep(0.5)
        self.contact_distance_spinbox.setValue(3.0)
        self.contact_distance_spinbox.setEnabled(False)
        contacts_layout.addWidget(self.contact_distance_spinbox)
        
        contacts_layout.addWidget(QLabel("Å of displayed mutants"))
        contacts_layout.addStretch()
        layout.addLayout(contacts_layout)
        
        # Connect the checkbox to enable/disable the spinbox
        self.show_contacts_checkbox.toggled.connect(self.contact_distance_spinbox.setEnabled)

        epistasis_groupbox = QGroupBox("Epistasis Analysis")
        epistasis_layout = QVBoxLayout()
        epistasis_groupbox.setLayout(epistasis_layout)
        layout.addWidget(epistasis_groupbox)

        self.epistasis_checkbox = QCheckBox("Epistasis Mode (Mutually Exclusive UI)")
        self.epistasis_checkbox.setChecked(False)
        self.epistasis_checkbox.toggled.connect(self._on_epistasis_toggled)
        epistasis_layout.addWidget(self.epistasis_checkbox)

        epist_thresh_layout = QHBoxLayout()
        epist_thresh_layout.addWidget(QLabel("Epistasis dddg_pred Threshold:"))
        self.epistasis_threshold_spinbox = QDoubleSpinBox()
        self.epistasis_threshold_spinbox.setRange(0.0, 1000.0) 
        self.epistasis_threshold_spinbox.setSingleStep(0.1)
        self.epistasis_threshold_spinbox.setValue(1.0)
        epist_thresh_layout.addWidget(self.epistasis_threshold_spinbox)
        epist_thresh_layout.addStretch()
        epistasis_layout.addLayout(epist_thresh_layout)
        layout.addStretch()


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

    def _on_epistasis_toggled(self, checked):
        self.color_backbone_checkbox.setEnabled(not checked)
        self.show_sticks_checkbox.setEnabled(not checked)
        self.show_contacts_checkbox.setEnabled(not checked)
        self.epistasis_threshold_spinbox.setEnabled(checked)

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
                else:
                    self.session.logger.warning(f"Could not find python binary inside {env}. Falling back to default 'python'.")
                    program = 'python'
            else:
                program = 'conda'
                args.extend(['run', '-n', env, 'python'])
        else:
            program = 'python'

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
            # Assumes mode isn't needed if input_csv overrides it, but add it if backend requires it:
            # script_args += ['--mode', 'singles'] # Uncomment if required as fallback
            
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
            if out:
                self.session.logger.info(out.rstrip())
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
            
            is_epistasis = self.epistasis_checkbox.isChecked()
            if self._parse_csv(fp, is_epistasis=is_epistasis):
                self.status_label.setText("Status: Applying visualization...")
                if is_epistasis:
                    self._apply_epistasis_visualization()
                else:
                    self._apply_visualization()
            else:
                if not self.status_label.text().startswith("Status: Error"):
                    self.status_label.setText("Status: Error parsing CSV. Check Log.")
        else:
            self.status_label.setText("Status: CSV loading cancelled.")

    def _parse_csv(self, filepath, is_epistasis=False):
        try:
            import pandas as pd
        except ImportError:
            msg = "Pandas is not installed in the ChimeraX Python environment. Please run 'pip install pandas' in the ChimeraX shell."
            self.session.logger.error(msg)
            raise AssertionError(msg)

        self.residue_scores_data = {}
        self.epistasis_df = None
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

            df['parsed_muts'] = df['mut_type'].apply(parse_mut_string)

            if is_epistasis:
                # Epistasis validation
                if not {'chain', 'mut_type', 'combined_dddg_pred'}.issubset(set(df.columns)):
                    raise AssertionError(f"Epistasis mode missing required columns. Found: {list(df.columns)}")

                df = df[df['parsed_muts'].apply(len) == 2].copy()
                if df.empty:
                    raise AssertionError("Epistasis mode requested, but no double mutations (e.g., A1C:D2E) were found in the CSV.")
                    
                df['chain_id'] = df['chain'].astype(str).str.strip()
                df['pos1_pdb'] = df['parsed_muts'].apply(lambda x: x[0]['pos'])
                df['mut1'] = df['parsed_muts'].apply(lambda x: x[0]['mut'])
                df['pos2_pdb'] = df['parsed_muts'].apply(lambda x: x[1]['pos'])
                df['mut2'] = df['parsed_muts'].apply(lambda x: x[1]['mut'])
                df['dddg_pred'] = df['combined_dddg_pred']

                self.epistasis_df = df
                self.session.logger.info(f"Parsed epistasis dataframe with {len(df)} rows.")
                return True
                
            else:
                # Standard (Single Mutation) validation
                base_req = {'pdb_file', 'code', 'chain', 'mut_type', 'wt_lora_pred'}
                if not base_req.issubset(set(df.columns)):
                    raise AssertionError(f"Missing required base columns in CSV. Expected at least: {base_req}. Found: {list(df.columns)}")

                # Determine target score column based on Radio Button selection
                if self.radio_wt_lora.isChecked():
                    target_score_col = 'wt_lora_pred'
                elif self.radio_mt_lora.isChecked():
                    target_score_col = 'mt_lora_pred'
                else:
                    target_score_col = 'combined_pred'

                # Graceful crash if the required column wasn't generated by inference.py
                if target_score_col not in df.columns:
                    raise AssertionError(f"Selected score '{target_score_col}' not found in CSV. Did you skip the MT pass during inference? If so, select 'WT LoRA predictions'.")

                # Isolate single mutations
                df = df[df['parsed_muts'].apply(len) == 1].copy()
                if df.empty:
                    raise AssertionError("Single mutation mode requested, but no single mutations were found in the CSV.")

                df['chain_id'] = df['chain'].astype(str).str.strip()
                df['pos1_pdb'] = df['parsed_muts'].apply(lambda x: x[0]['pos'])
                df['mut1'] = df['parsed_muts'].apply(lambda x: x[0]['mut'])
                df['viz_score'] = df[target_score_col]

                pivot_df = df.pivot_table(index=['chain_id', 'pos1_pdb'], columns='mut1', values='viz_score')
                
                if pivot_df.empty:
                    raise AssertionError("Parsed CSV resulted in an empty pivot table.")

                max_scores = pivot_df.max(axis=1)
                top_aas = pivot_df.idxmax(axis=1)

                count = 0
                for idx in max_scores.index:
                    chain_id_val, pos = idx
                    score = max_scores[idx]
                    if pd.isna(score):
                        continue
                    
                    top_aa = top_aas[idx]
                    if pd.isna(top_aa):
                        continue

                    if score != 0.0:
                        self.residue_scores_data[(chain_id_val, int(pos))] = (float(score), str(top_aa).upper())
                        count += 1
                
                if count == 0:
                    raise AssertionError("Parsed CSV, but no valid non-zero scores found.")

                self.session.logger.info(f"Parsed {count} single-mutation scores using metric: {target_score_col}.")
                return True

        except Exception as e:
            self.session.logger.error(f"Error parsing CSV with Pandas: {e}")
            self.status_label.setText("Status: Error parsing CSV (see log).")
            raise AssertionError(f"CSV Parsing failed: {e}")

    def _apply_epistasis_visualization(self):
        """
        Visualizes epistatic interactions by creating a mutated model and drawing
        scaled pseudobonds between residues based on their coupling scores.
        """
        wt_candidates = [m for m in self.session.models.list(type=Structure)
                        if not (getattr(self, 'mutated_model_id_string', None) and m.id_string == self.mutated_model_id_string)]
                        
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        
        if not wt_model:
            self.status_label.setText("Status: Error - WT model not found.")
            raise AssertionError("Cannot apply visualization: Selected WT model is not open.")
        
        if getattr(self, 'mutated_model_id_string', None) and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
            run(self.session, f"close #{self.mutated_model_id_string}")
            self.mutated_model_id_string = None

        threshold = self.epistasis_threshold_spinbox.value()
        df = self.epistasis_df
        if df is None or df.empty:
            raise AssertionError("No epistasis data loaded. Please load a valid CSV first.")

        filtered_df = df[df['dddg_pred'].abs() >= threshold].copy()

        if filtered_df.empty:
            self.status_label.setText("Status: No residue pairs found exceeding threshold.")
            return

        filtered_df['abs_score'] = filtered_df['dddg_pred'].abs()
        sorted_df = filtered_df.sort_values(by='abs_score', ascending=False)
        
        max_abs_score = sorted_df['abs_score'].max()
        if max_abs_score <= threshold:
            max_abs_score = threshold + 0.0001 

        mutation_plan = {} 
        pairs_to_draw = [] 

        for _, row in sorted_df.iterrows():
            c1, p1, m1 = str(row['chain_id']).strip(), int(row['pos1_pdb']), str(row['mut1']).upper()
            c2, p2, m2 = str(row['chain_id']).strip(), int(row['pos2_pdb']), str(row['mut2']).upper()
            score = float(row['dddg_pred'])
            
            comp1 = ((c1, p1) not in mutation_plan) or (mutation_plan[(c1, p1)] == m1)
            comp2 = ((c2, p2) not in mutation_plan) or (mutation_plan[(c2, p2)] == m2)
            
            if comp1 and comp2:
                mutation_plan[(c1, p1)] = m1
                mutation_plan[(c2, p2)] = m2
                pairs_to_draw.append((c1, p1, m1, c2, p2, m2, score))

        try:
            # FIX: Use safe PDB clone to preserve polymer metadata instead of combine
            temp_pdb = os.path.join(tempfile.gettempdir(), f"clone_{wt_model.id_string.replace(':', '_')}.pdb")
            run(self.session, f"save {temp_pdb} models #{wt_model.id_string} format pdb")
            run(self.session, f"open {temp_pdb} name \"{wt_model.name}_epistasis_viz\"")
            
            mutated_model = self.session.models.list()[-1]
            self.mutated_model_id_string = mutated_model.id_string.split('.')[0]
            
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)
            
            run(self.session, f"color #{self.mutated_model_id_string} white")
            run(self.session, f"transparency #{self.mutated_model_id_string} 70 target a")
            run(self.session, f"hide #{self.mutated_model_id_string} atoms")

            for (chain_val, pos), tgt_aa in mutation_plan.items():
                res_wt = next((r for r in mutated_model.residues if r.number == pos and r.chain_id == chain_val), None)
                if res_wt and ONE_TO_THREE_LETTER_AA.get(tgt_aa, '') != res_wt.name:
                    spec = f"#{self.mutated_model_id_string}/{chain_val}:{pos}"
                    run(self.session, f"swapaa {spec} {ONE_TO_THREE_LETTER_AA[tgt_aa].lower()} log false")

            if mutation_plan:
                spec_list = [f"#{self.mutated_model_id_string}/{c}:{p}" for (c, p) in mutation_plan.keys()]
                spec_all = " | ".join(spec_list)
                run(self.session, f"show {spec_all} atoms; style {spec_all} stick; color {spec_all} byelement")
                run(self.session, f"transparency {spec_all} 0 target a")

            model_residues = {(r.chain_id, r.number): r for r in mutated_model.residues}
            count_lines = 0
            
            for c1, p1, m1, c2, p2, m2, score in pairs_to_draw:
                res1, res2 = model_residues.get((c1, p1)), model_residues.get((c2, p2))
                if not res1 or not res2 or not res1.atoms or not res2.atoms:
                    continue

                atoms1 = [a for a in res1.atoms if a.name not in ('N', 'CA', 'C', 'O')]
                atoms2 = [a for a in res2.atoms if a.name not in ('N', 'CA', 'C', 'O')]
                if not atoms1: atoms1 = list(res1.atoms)
                if not atoms2: atoms2 = list(res2.atoms)
                
                coords1 = np.array([a.scene_coord for a in atoms1])
                coords2 = np.array([a.scene_coord for a in atoms2])
                diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                a1, a2 = atoms1[min_idx[0]], atoms2[min_idx[1]]
                
                norm_score = min(1.0, max(0.0, (abs(score) - threshold) / (max_abs_score - threshold)))
                radius_val = 0.05 + (0.95 * (norm_score ** 2)) 
                intensity = int(100 + 155 * norm_score)        
                alpha = int(30 + 225 * norm_score)             
                
                if score > 0:
                    color_spec = f"0,{intensity},0,{alpha}"
                else:
                    color_spec = f"{intensity},0,0,{alpha}"
                
                cmd = (f"pbond #{self.mutated_model_id_string}/{c1}:{p1}@{a1.name} "
                       f"#{self.mutated_model_id_string}/{c2}:{p2}@{a2.name} "
                       f"reveal true color {color_spec} "
                       f"radius {radius_val:.3f} name \"{score:.2f}\"")
                
                run(self.session, cmd)
                count_lines += 1
                
            target_chain = self.pred_chain_id_combobox.currentText().strip()
            alpha_val = self.non_target_alpha_spinbox.value()
            if target_chain:
                exclude_spec = f"~#{wt_model.id_string}/{target_chain}"
                if getattr(self, 'mutated_model_id_string', None):
                    exclude_spec += f" & ~#{self.mutated_model_id_string}"
                if alpha_val >= 99:
                    run(self.session, f"hide {exclude_spec}")
                else:
                    run(self.session, f"show {exclude_spec} ribbons; transparency {exclude_spec} {alpha_val} target ac")

            self.status_label.setText(f"Status: Epistasis Viz Complete ({count_lines} interactions).")

        except Exception as e:
            self.session.logger.error(f"Critical failure in Epistasis Visualization: {e}")
            self.status_label.setText("Status: Visualization Error.")
            raise AssertionError(f"Epistasis visualization subroutine failed: {e}")


    def _apply_visualization(self):
        wt_candidates = [m for m in self.session.models.list(type=Structure)
                         if not (getattr(self, 'mutated_model_id_string', None) and m.id_string == self.mutated_model_id_string)]
                         
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        
        if not wt_model and wt_candidates:
             wt_model = wt_candidates[0]
             
        if not wt_model:
            self.status_label.setText("Status: No suitable WT model open.")
            raise AssertionError("Failed to apply Visualization. No suitable WT model open.")
            
        if not getattr(self, 'residue_scores_data', None):
            self.status_label.setText("Status: No scores to apply.")
            return

        threshold = self.score_threshold_spinbox.value()
        color_backbone = self.color_backbone_checkbox.isChecked()
        show_sticks = self.show_sticks_checkbox.isChecked()
        wt_stick_alpha = self.wt_stick_alpha_spinbox.value()
        mut_stick_alpha = self.mut_stick_alpha_spinbox.value()

        if getattr(self, 'mutated_model_id_string', None) and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
            try:
                run(self.session, f"close #{self.mutated_model_id_string}")
            except Exception as e:
                self.session.logger.warning(f"Could not close previous mutated: {e}")
            finally:
                self.mutated_model_id_string = None

        try:
            run(self.session, f"color #{wt_model.id_string} white")
            run(self.session, f"ribbon style #{wt_model.id_string}")
            run(self.session, f"hide #{wt_model.id_string} atoms")
        except Exception as e:
            self.status_label.setText("Status: Error styling WT model.")
            raise AssertionError(f"Visualization aborted, failed to style initial WT structure: {e}")

        scores = [s for s, _ in self.residue_scores_data.values()]
        if not scores:
            self.status_label.setText("Status: No valid scores in data.")
            return
            
        max_abs = max(abs(min(scores)), abs(max(scores))) or 0.01
        color_range = f"{-max_abs:.3f},{max_abs:.3f}"

        for (chain, pos), (score, _) in self.residue_scores_data.items():
            spec = f"#{wt_model.id_string}/{chain}:{pos}"
            try:
                run(self.session, f"setattr {spec} r {SCORE_ATTRIBUTE_NAME} {score} create true")
            except Exception as e:
                self.session.logger.warning(f"Failed to set custom residue attribute on WT spec: {spec}")

        mut_model = None
        mut_spec = None
        wt_spec = None
        if show_sticks:
            try:
                # FIX: Use safe PDB clone to preserve polymer metadata instead of combine
                temp_pdb = os.path.join(tempfile.gettempdir(), f"clone_{wt_model.id_string.replace(':', '_')}.pdb")
                run(self.session, f"save {temp_pdb} models #{wt_model.id_string} format pdb")
                run(self.session, f"open {temp_pdb} name \"{wt_model.name}_mutated_viz\"")
                
                mut_model = self.session.models.list()[-1]
                self.mutated_model_id_string = mut_model.id_string.split('.')[0]
                
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)

                run(self.session, f"color #{self.mutated_model_id_string} lightgray")
                run(self.session, f"ribbon style #{self.mutated_model_id_string}")
                run(self.session, f"hide #{self.mutated_model_id_string} atoms")

                muts_by_chain = {}
                for (chain, pos), (score, tgt_aa) in self.residue_scores_data.items():
                    if score >= threshold:
                        res_wt = next((r for r in wt_model.residues if r.number == pos and r.chain_id == chain), None)
                        if res_wt and tgt_aa != res_wt.one_letter_code:
                            spec = f"#{self.mutated_model_id_string}/{chain}:{pos}"
                            try:
                                aa_code = ONE_TO_THREE_LETTER_AA.get(tgt_aa)
                                if aa_code:
                                    run(self.session, f"swapaa {spec} {aa_code.lower()} log true")
                                    muts_by_chain.setdefault(chain, []).append(pos)
                                else:
                                    self.session.logger.warning(f"Unknown target aa {tgt_aa} requested")
                            except Exception as e:
                                self.session.logger.error(f"swapaa failed at chain {chain} pos {pos}: {e}")
                        elif res_wt:
                            muts_by_chain.setdefault(chain, []).append(pos)

                if muts_by_chain:
                    wt_specs = []
                    mut_specs = []
                    for chain, muts in muts_by_chain.items():
                        lst = ",".join(map(str, muts))
                        wt_specs.append(f"#{wt_model.id_string}/{chain}:{lst}")
                        mut_specs.append(f"#{self.mutated_model_id_string}/{chain}:{lst}")
                    
                    wt_spec = " | ".join(wt_specs)
                    mut_spec = " | ".join(mut_specs)

                    for chain, muts in muts_by_chain.items():
                        for pos in muts:
                            score, _ = self.residue_scores_data[(chain, pos)]
                            run(self.session, f"setattr #{self.mutated_model_id_string}/{chain}:{pos} r {SCORE_ATTRIBUTE_NAME} {score} create true")
                            
                    key_val = 'true' if not color_backbone else 'false'
                    run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} {mut_spec} & sideonly palette red:white:green range {color_range} key {key_val} target a")
                    run(self.session, f"color {mut_spec} & ~C & sideonly byelement target a")
                    run(self.session, f"show {mut_spec} atoms")
                    run(self.session, f"style {mut_spec} stick")
                    run(self.session, f"transparency {mut_spec} {mut_stick_alpha} target a")

                    run(self.session, f"color {wt_spec} & ~C & sideonly white")
                    run(self.session, f"color {wt_spec} & ~C & sideonly byelement")
                    run(self.session, f"show {wt_spec} atoms")
                    run(self.session, f"style {wt_spec} stick")
                    run(self.session, f"transparency {wt_spec} {wt_stick_alpha} target a")

                # The `match` command is now redundant as both structures are perfectly aligned
                # However we leave it here for safety just in case of slight numerical shift
                run(self.session, f"match #{self.mutated_model_id_string} to #{wt_model.id_string}")
            except Exception as e:
                self.status_label.setText("Status: Error showing sticks.")
                self.session.logger.error(f"Error in stick viz: {e}")
                if mut_model and any(m.id_string == getattr(self, 'mutated_model_id_string', None) for m in self.session.models.list()):
                    run(self.session, f"close #{self.mutated_model_id_string}")
                self.mutated_model_id_string = None
                raise AssertionError(f"Visualization logic failed mid-execution: {e}")

        if self.color_backbone_checkbox.isChecked():
            chains_present = set(c for c, p in self.residue_scores_data.keys())
            chain_spec = ",".join(chains_present)
            run(self.session, f"color byattribute {SCORE_ATTRIBUTE_NAME} #{wt_model.id_string}/{chain_spec} & backbone palette red:white:green range {color_range} key true")

        if show_sticks and self.show_contacts_checkbox.isChecked():
            try:
                if mut_spec:
                    dist = self.contact_distance_spinbox.value()
                    
                    # 1. Zone selection: Distance check explicitly using the exact syntax ordering you requested.
                    run(self.session, f"select ({mut_spec}) @<{dist} & ~backbone & #{wt_model.id_string} & protein & ~({mut_spec}) & ~({wt_spec})")
                    run(self.session, "select up")
                    run(self.session, "select up")
                    # Style the entire residue context as thin, semi-transparent lines
                    run(self.session, "show sel")
                    run(self.session, "style sel stick")
                    run(self.session, "size sel stickRadius 0.1") # Increased from 0.05 to ensure visibility
                    run(self.session, "color sel byelement")
                    run(self.session, f"transparency sel {wt_stick_alpha} target ab") # Explicitly target atoms AND bonds for transparency
                    
                    run(self.session, f"select ({mut_spec}) @<{dist} & ~backbone & #{wt_model.id_string} & protein & ~({mut_spec}) & ~({wt_spec})")
                    run(self.session, "style sel ball")
                    run(self.session, "transparency sel 0 target a")

                    # Clean up view state
                    run(self.session, "hide @h*")
                    run(self.session, "select clear")
            except Exception as e:
                self.session.logger.error(f"Error displaying contacts: {e}")

        target_chain = self.pred_chain_id_combobox.currentText().strip()
        alpha_val = self.non_target_alpha_spinbox.value()
        if target_chain:
            exclude_spec = f"~#{wt_model.id_string}/{target_chain}"
            if getattr(self, 'mutated_model_id_string', None):
                exclude_spec += f" & ~#{self.mutated_model_id_string}"
            try:
                if alpha_val >= 99:
                    run(self.session, f"hide {exclude_spec}")
                else:
                    run(self.session, f"show {exclude_spec} ribbons")
                    run(self.session, f"transparency {exclude_spec} {alpha_val} target ac")
            except Exception as e:
                self.session.logger.warning(f"Failed setting transparency for isolation: {e}")

        run(self.session, "select clear; hide @H")
        self.status_label.setText("Status: Visualization complete.")