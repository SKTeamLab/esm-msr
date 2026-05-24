# vim: set expandtab shiftwidth=4 softstop=4:
import os
import tempfile
import shutil
import numpy as np

from chimerax.ui import MainToolWindow
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run

from chimerax.atomic import Structure

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QLineEdit, QFrame, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTabWidget
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
        self.session.logger.info(f"****** RSVTool __init__ ({tool_registered_name}) ******")

        # 1. INITIALIZE SETTINGS
        self.settings = QSettings("ESM_MSR_Tools", "ESM_MSR_Visualizer")
        
        # 2. LOAD STATE VARIABLES
        self.base_repo_path = self.settings.value("base_repo_path", "")
        self.python_env = self.settings.value("python_env", "")
        
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
        
        # 5. REGISTER CHIMERAX EVENT HOOKS
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

        self.tab_advanced = QWidget()
        self._build_advanced_tab(self.tab_advanced)
        self.tabs.addTab(self.tab_advanced, "Model & Checkpoint")

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
        self.script_output_csv_path_edit = QLineEdit()
        out_csv_layout.addWidget(self.script_output_csv_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_script_output_csv)
        out_csv_layout.addWidget(btn)
        paths_layout.addLayout(out_csv_layout)
        
        hf_layout = QHBoxLayout()
        hf_layout.addWidget(QLabel("HF Token:"))
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setPlaceholderText("Optional: HuggingFace Token (for ESM3 weights)")
        self.hf_token_edit.setEchoMode(QLineEdit.Password)
        hf_layout.addWidget(self.hf_token_edit)
        paths_layout.addLayout(hf_layout)

        layout.addWidget(paths_group)

        # Target Selection Group
        target_group = QGroupBox("Target Selection")
        target_layout = QVBoxLayout()
        target_group.setLayout(target_layout)

        model_chain_layout = QHBoxLayout()
        model_chain_layout.addWidget(QLabel("Target Model:"))
        self.pred_model_combobox = QComboBox()
        self.pred_model_combobox.currentIndexChanged.connect(self._on_model_selected)
        model_chain_layout.addWidget(self.pred_model_combobox)

        model_chain_layout.addWidget(QLabel("Chain:"))
        self.pred_chain_id_combobox = QComboBox()
        self.pred_chain_id_combobox.setEditable(True)
        model_chain_layout.addWidget(self.pred_chain_id_combobox)
        target_layout.addLayout(model_chain_layout)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Compute Device:"))
        self.device_combobox = QComboBox()
        self.device_combobox.addItems(['cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu'])
        device_layout.addWidget(self.device_combobox)
        target_layout.addLayout(device_layout)
        
        layout.addWidget(target_group)
        layout.addStretch()

    def _build_screening_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        # Mutations Scope Group
        scope_group = QGroupBox("Mutation Scope (Mutually Exclusive Inputs)")
        scope_layout = QVBoxLayout()
        scope_group.setLayout(scope_layout)

        # Mode 1: Automated Screening
        meth1_layout = QHBoxLayout()
        meth1_layout.addWidget(QLabel("1. Auto Mode:"))
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['singles', 'doubles', 'both'])
        meth1_layout.addWidget(self.mode_combobox)
        
        meth1_layout.addWidget(QLabel("Positions (CSV):"))
        self.selected_residues_edit = QLineEdit()
        self.selected_residues_edit.setPlaceholderText("e.g. 11,12 (Empty=All)")
        meth1_layout.addWidget(self.selected_residues_edit)

        self.grab_sel_button = QPushButton("Grab Selection")
        self.grab_sel_button.clicked.connect(self._grab_selection)
        meth1_layout.addWidget(self.grab_sel_button)

        self.screen_except_checkbox = QCheckBox("Invert (Except)")
        meth1_layout.addWidget(self.screen_except_checkbox)
        scope_layout.addLayout(meth1_layout)

        # Mode 2: Input CSV
        meth2_layout = QHBoxLayout()
        meth2_layout.addWidget(QLabel("2. Input CSV:"))
        self.subset_df_edit = QLineEdit()
        self.subset_df_edit.setPlaceholderText("Path to input DataFrame...")
        meth2_layout.addWidget(self.subset_df_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_subset_df)
        meth2_layout.addWidget(btn)
        scope_layout.addLayout(meth2_layout)

        # Mode 3: Specific Mutations
        meth3_layout = QHBoxLayout()
        meth3_layout.addWidget(QLabel("3. Explicit Muts:"))
        self.mutations_edit = QLineEdit()
        self.mutations_edit.setPlaceholderText("e.g., A12C,A12C:D15E")
        meth3_layout.addWidget(self.mutations_edit)
        scope_layout.addLayout(meth3_layout)
        
        layout.addWidget(scope_group)

        # Runtime Options
        runtime_group = QGroupBox("Screening Parameters")
        runtime_layout = QVBoxLayout()
        runtime_group.setLayout(runtime_layout)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Mask Strategy:"))
        self.mask_strategy_combobox = QComboBox()
        self.mask_strategy_combobox.addItems(['None', 'marginal', 'chain'])
        row1.addWidget(self.mask_strategy_combobox)

        row1.addWidget(QLabel("Batch Size:"))
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 512)
        self.batch_size_spinbox.setValue(16)
        row1.addWidget(self.batch_size_spinbox)
        runtime_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Distance Threshold (Å):"))
        self.distance_threshold_spinbox = QDoubleSpinBox()
        self.distance_threshold_spinbox.setRange(0.0, 100.0)
        self.distance_threshold_spinbox.setValue(6.0)
        row2.addWidget(self.distance_threshold_spinbox)

        self.calculate_distances_checkbox = QCheckBox("Calculate Distances (for CSV)")
        row2.addWidget(self.calculate_distances_checkbox)
        runtime_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Global Backbone Mutation:"))
        self.backbone_mutation_edit = QLineEdit()
        self.backbone_mutation_edit.setPlaceholderText("e.g., A15G")
        row3.addWidget(self.backbone_mutation_edit)
        runtime_layout.addLayout(row3)

        layout.addWidget(runtime_group)
        layout.addStretch()

    def _build_advanced_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        # Config Files
        files_group = QGroupBox("Model Configuration Files")
        files_layout = QVBoxLayout()
        files_group.setLayout(files_layout)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Checkpoint (.ckpt):"))
        self.checkpoint_path_edit = QLineEdit()
        row1.addWidget(self.checkpoint_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_checkpoint_path)
        row1.addWidget(btn)
        files_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("LoRA Config (JSON):"))
        self.lora_config_edit = QLineEdit()
        self.lora_config_edit.setPlaceholderText("Optional config file")
        row2.addWidget(self.lora_config_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_generic(self.lora_config_edit, "Select LoRA JSON", "JSON (*.json);;All (*)"))
        row2.addWidget(btn)
        files_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("HParams (YAML):"))
        self.hparams_path_edit = QLineEdit()
        self.hparams_path_edit.setPlaceholderText("Optional hparams.yaml")
        row3.addWidget(self.hparams_path_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_generic(self.hparams_path_edit, "Select HParams", "YAML (*.yaml *.yml);;All (*)"))
        row3.addWidget(btn)
        files_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Base Model Location (Folder):"))
        self.base_model_loc_edit = QLineEdit()
        self.base_model_loc_edit.setPlaceholderText("Optional path to offline model folder")
        row4.addWidget(self.base_model_loc_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_generic_dir(self.base_model_loc_edit, "Select Model Directory"))
        row4.addWidget(btn)
        files_layout.addLayout(row4)

        layout.addWidget(files_group)

        # Architectural parameters
        arch_group = QGroupBox("Architectural Parameters")
        arch_layout = QVBoxLayout()
        arch_group.setLayout(arch_layout)

        arow1 = QHBoxLayout()
        arow1.addWidget(QLabel("Model Dtype:"))
        self.model_dtype_combobox = QComboBox()
        self.model_dtype_combobox.addItems(["float32", "bfloat16", "float16"])
        arow1.addWidget(self.model_dtype_combobox)
        
        arow1.addWidget(QLabel("Adapter Mode:"))
        self.adapter_mode_combobox = QComboBox()
        self.adapter_mode_combobox.addItems(["dual", "fused"])
        arow1.addWidget(self.adapter_mode_combobox)
        
        arow1.addWidget(QLabel("LoRA Mode:"))
        self.lora_mode_combobox = QComboBox()
        self.lora_mode_combobox.addItems(["ensemble", "corrector"])
        arow1.addWidget(self.lora_mode_combobox)

        #arow2 = QHBoxLayout()
        arow1.addWidget(QLabel("Quaternary Mode:"))
        self.quaternary_mode_combobox = QComboBox()
        self.quaternary_mode_combobox.addItems(["single_chain", "complex"])
        arow1.addWidget(self.quaternary_mode_combobox)

        arch_layout.addLayout(arow1)

        #arow2.addWidget(QLabel("Shared Scale Init:"))
        #self.shared_scale_init_spinbox = QDoubleSpinBox()
        #self.shared_scale_init_spinbox.setRange(-10.0, 10.0)
        #self.shared_scale_init_spinbox.setValue(0.3)
        #self.shared_scale_init_spinbox.setSingleStep(0.1)
        #arow2.addWidget(self.shared_scale_init_spinbox)
        
        #arow2.addWidget(QLabel("Shared Bias Init:"))
        #self.shared_bias_init_spinbox = QDoubleSpinBox()
        #self.shared_bias_init_spinbox.setRange(-10.0, 10.0)
        #self.shared_bias_init_spinbox.setValue(0.0)
        #self.shared_bias_init_spinbox.setSingleStep(0.1)
        #arow2.addWidget(self.shared_bias_init_spinbox)
        #arch_layout.addLayout(arow2)

        layout.addWidget(arch_group)

        # Execution Flags
        flags_group = QGroupBox("Execution Flags")
        flags_layout = QVBoxLayout()
        flags_group.setLayout(flags_layout)
        
        grid_layout = QHBoxLayout()
        
        col1 = QVBoxLayout()
        self.load_wt_only_checkbox = QCheckBox("Load WT Only")
        self.freeze_lora_checkbox = QCheckBox("Freeze LoRA Init")
        col1.addWidget(self.load_wt_only_checkbox)
        col1.addWidget(self.freeze_lora_checkbox)
        grid_layout.addLayout(col1)

        col2 = QVBoxLayout()
        self.no_optimize_wt_pass_checkbox = QCheckBox("No Optimize WT Pass")
        self.skip_additive_checkbox = QCheckBox("Skip Additive Math")
        self.skip_reverse_checkbox = QCheckBox("Skip Reverse/MT Pass")
        self.log_likelihood_checkbox = QCheckBox("Log Likelihoods")
        col2.addWidget(self.no_optimize_wt_pass_checkbox)
        col2.addWidget(self.skip_additive_checkbox)
        col2.addWidget(self.skip_reverse_checkbox)
        col2.addWidget(self.log_likelihood_checkbox)
        grid_layout.addLayout(col2)
        
        col3 = QVBoxLayout()
        self.use_plddt_checkbox = QCheckBox("Use pLDDT")
        col3.addWidget(self.use_plddt_checkbox)
        col3.addStretch()
        grid_layout.addLayout(col3)

        flags_layout.addLayout(grid_layout)
        layout.addWidget(flags_group)
        layout.addStretch()

    def _build_viz_tab(self, parent_widget):
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)

        self.csv_label = QLabel("No CSV loaded.")
        layout.addWidget(self.csv_label)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Score Threshold:"))
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

        self.show_contacts_checkbox = QCheckBox("Visualize Contacts")
        self.show_contacts_checkbox.setChecked(False)
        layout.addWidget(self.show_contacts_checkbox)

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
            self.checkpoint_path = ""
            self.script_output_csv_path = ""
            if not is_init:
                self.python_env = ""
        else:
            self.python_script_path = os.path.normpath(os.path.join(base_path, "src", "esm_msr", "inference.py"))
            self.checkpoint_path = os.path.normpath(os.path.join(base_path, "LoRA_models", "msr_singles_only", "seed3_epoch=08-val_rho_avg=0.754.ckpt"))
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
            self.script_output_csv_path = fp
            self.script_output_csv_path_edit.setText(fp)

    def _browse_checkpoint_path(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, "Select Checkpoint File", self.checkpoint_path_edit.text(), "Checkpoint Files (*.ckpt *.pt *.pth *.h5);;All Files (*)")
        if fp:
            self.checkpoint_path_edit.setText(os.path.normpath(fp))

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

    def _browse_generic(self, line_edit, title, filter_str):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, title, self.base_repo_path, filter_str)
        if fp:
            line_edit.setText(os.path.normpath(fp))
            
    def _browse_generic_dir(self, line_edit, title):
        w = self.session.ui.main_window
        folder_path = QFileDialog.getExistingDirectory(w, title, self.base_repo_path)
        if folder_path:
            line_edit.setText(os.path.normpath(folder_path))

    # -------------- run prediction (QProcess) --------------
    def _initiate_run_prediction_script(self):
        self.session.logger.info("****** _initiate_run_prediction_script called ******")

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
        self.settings.setValue("python_env", self.python_env)
        self.settings.setValue("base_repo_path", self.base_repo_path)

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
            raise AssertionError("A checkpoint path is required to run inference.")
        script_args += ['--checkpoint_path', self.checkpoint_path_edit.text().strip()]

        # Data Inputs
        subset_df_val = self.subset_df_edit.text().strip()
        if subset_df_val:
            script_args += ['--input_csv', subset_df_val]
        else:
            # If no input CSV, fall back to pdb logic
            script_args += ['--pdb_file', self.script_input_structure_path]
            script_args += ['--code', os.path.splitext(current_model.name)[0] if current_model.name else 'protein']
            script_args += ['--chain', self.pred_chain_id_combobox.currentText().strip()]

        # Target Config
        script_args += ['--mode', self.mode_combobox.currentText()]
        
        mutations_val = self.mutations_edit.text().strip()
        if mutations_val:
            script_args += ['--mutations', mutations_val]

        selected_res = self.selected_residues_edit.text().strip()
        if selected_res:
            if self.screen_except_checkbox.isChecked():
                script_args += ['--screen_residues_except', selected_res]
            else:
                script_args += ['--screen_residues', selected_res]

        # Engine Params
        script_args += ['--batch_size', str(self.batch_size_spinbox.value())]
        script_args += ['--device', self.device_combobox.currentText()]
        
        mask_strat = self.mask_strategy_combobox.currentText()
        if mask_strat != 'None':
            script_args += ['--mask_strategy', mask_strat]

        if self.distance_threshold_spinbox.value() >= 0:
            script_args += ['--distance_threshold', str(self.distance_threshold_spinbox.value())]

        if self.calculate_distances_checkbox.isChecked():
            script_args += ['--calculate_distances']

        backbone_mut = self.backbone_mutation_edit.text().strip()
        if backbone_mut:
            script_args += ['--backbone_mutation', backbone_mut]

        hf_token_val = self.hf_token_edit.text().strip()
        if hf_token_val:
            script_args += ['--hf_token', hf_token_val]

        # Model Configs
        lora_cfg = self.lora_config_edit.text().strip()
        if lora_cfg:
            script_args += ['--lora_config', lora_cfg]
            
        hparams = self.hparams_path_edit.text().strip()
        if hparams:
            script_args += ['--hparams_path', hparams]
            
        base_model_loc_val = self.base_model_loc_edit.text().strip()
        if base_model_loc_val:
            script_args += ['--base_model_loc', base_model_loc_val]

        script_args += ['--model_dtype', self.model_dtype_combobox.currentText()]
        script_args += ['--adapter_mode', self.adapter_mode_combobox.currentText()]
        script_args += ['--lora_mode', self.lora_mode_combobox.currentText()]
        script_args += ['--quaternary_mode', self.quaternary_mode_combobox.currentText()]
        #script_args += ['--shared_scale_init', str(self.shared_scale_init_spinbox.value())]
        #script_args += ['--shared_bias_init', str(self.shared_bias_init_spinbox.value())]

        # Flags
        if self.load_wt_only_checkbox.isChecked(): script_args += ['--load_wt_only']
        if self.freeze_lora_checkbox.isChecked(): script_args += ['--freeze_lora']
        if self.no_optimize_wt_pass_checkbox.isChecked(): script_args += ['--no_optimize_wt_pass']
        if self.skip_additive_checkbox.isChecked(): script_args += ['--skip_additive']
        if self.skip_reverse_checkbox.isChecked(): script_args += ['--skip_reverse']
        if self.log_likelihood_checkbox.isChecked(): script_args += ['--log_likelihood']
        if self.use_plddt_checkbox.isChecked(): script_args += ['--use_plddt']

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
        self.run_prediction_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.prediction_output_label.setText("Predicted output file: Processing...")

        self.session.logger.info(f"QProcess starting: {program} {' '.join(full_args)}")
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

    def _on_proc_finished(self, exitCode, exitStatus):
        self.run_prediction_button.setEnabled(True)
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
            self.prediction_output_label.setText("Predicted output file: Failed")
            self.status_label.setText(f"Status: Script Error (exit={exitCode}). See log.")

        try:
            self.proc.deleteLater()
        except Exception as e:
            self.session.logger.error(f"Failed cleaning up QProcess: {e}")
        self.proc = None

    # -------------- CSV parse + viz --------------
    def _handle_load_and_visualize(self):
        self.session.logger.info("****** _handle_load_and_visualize called ******")
        w = self.session.ui.main_window
        initial_dir, default_filename = "", ""
        if self.predicted_output_path and os.path.exists(self.predicted_output_path):
            initial_dir, default_filename = os.path.dirname(self.predicted_output_path), self.predicted_output_path
        elif self.script_output_csv_path_edit.text() and os.path.exists(self.script_output_csv_path_edit.text()):
            default_filename = self.script_output_csv_path_edit.text()
            initial_dir = os.path.dirname(default_filename)
        elif self.loaded_csv_path and os.path.exists(self.loaded_csv_path):
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
        import pandas as pd
        import re

        self.residue_scores_data = {}
        self.epistasis_df = None
        try:
            df = pd.read_csv(filepath)
            df.columns = [c.lower().strip() for c in df.columns]

            # 1. Base column validation based on new inference.py outputs
            if not {'chain', 'mut_type', 'combined_dddg_pred'}.issubset(set(df.columns)):
                msg = f"Missing columns in CSV. Expected at least: {{'chain', 'mut_type', 'combined_dddg_pred'}}. Found: {list(df.columns)}"
                self.session.logger.error(msg)
                raise AssertionError(msg)

            # 2. Robust helper to parse mut_type strings (e.g., "A12C" or "A12C:D15E")
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
                # Isolate double mutations
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
                # Isolate single mutations
                df = df[df['parsed_muts'].apply(len) == 1].copy()
                if df.empty:
                    raise AssertionError("Single mutation mode requested, but no single mutations were found in the CSV.")

                df['chain_id'] = df['chain'].astype(str).str.strip()
                df['pos1_pdb'] = df['parsed_muts'].apply(lambda x: x[0]['pos'])
                df['mut1'] = df['parsed_muts'].apply(lambda x: x[0]['mut'])
                df['ddg_pred'] = df['combined_dddg_pred']

                pivot_df = df.pivot_table(index=['chain_id', 'pos1_pdb'], columns='mut1', values='ddg_pred')
                
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

                self.session.logger.info(f"Parsed scores for {len(self.residue_scores_data)} positions across chains.")
                return True

        except Exception as e:
            self.session.logger.error(f"Error parsing CSV with Pandas: {e}")
            self.status_label.setText("Status: Error parsing CSV (see log).")
            # Explicitly raise to prevent silent UI state corruption
            raise AssertionError(f"CSV Parsing failed: {e}")

    def _apply_epistasis_visualization(self):
            """
            Visualizes epistatic interactions by creating a mutated model and drawing
            scaled pseudobonds between residues based on their coupling scores.
            """
            wt_candidates = [m for m in self.session.models.list(type=Structure)
                            if not (self.mutated_model_id_string and m.id_string == self.mutated_model_id_string)]
                            
            model_id = self.pred_model_combobox.currentData()
            wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
            
            if not wt_model:
                self.status_label.setText("Status: Error - WT model not found.")
                raise AssertionError("Cannot apply visualization: Selected WT model is not open.")
            
            # Cleanup previous mutated model visualization
            if self.mutated_model_id_string and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
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

            # 1. Scaling Logic: Calculate max for normalization
            filtered_df['abs_score'] = filtered_df['dddg_pred'].abs()
            sorted_df = filtered_df.sort_values(by='abs_score', ascending=False)
            
            max_abs_score = sorted_df['abs_score'].max()
            if max_abs_score <= threshold:
                max_abs_score = threshold + 0.0001 

            # 2. Greedy Conflict Resolution
            mutation_plan = {} # (chain, pos) -> mut_aa
            pairs_to_draw = [] # (c1, p1, m1, c2, p2, m2, score)

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

            # 3. Model Generation & Mutation
            try:
                run(self.session, f"combine #{wt_model.id_string} name \"{wt_model.name}_epistasis_viz\"")
                mutated_model = self.session.models.list()[-1]
                self.mutated_model_id_string = mutated_model.id_string
                
                run(self.session, f"color #{self.mutated_model_id_string} white")
                run(self.session, f"transparency #{self.mutated_model_id_string} 70 target a")
                run(self.session, f"hide #{self.mutated_model_id_string} atoms")

                # Apply Mutations via swapaa
                for (chain_val, pos), tgt_aa in mutation_plan.items():
                    res_wt = next((r for r in mutated_model.residues if r.number == pos and r.chain_id == chain_val), None)
                    if res_wt and ONE_TO_THREE_LETTER_AA.get(tgt_aa, '') != res_wt.name:
                        spec = f"#{self.mutated_model_id_string}/{chain_val}:{pos}"
                        run(self.session, f"swapaa {spec} {ONE_TO_THREE_LETTER_AA[tgt_aa].lower()} log false")

                # Display side-chains for all involved positions
                if mutation_plan:
                    spec_list = [f"#{self.mutated_model_id_string}/{c}:{p}" for (c, p) in mutation_plan.keys()]
                    spec_all = " | ".join(spec_list)
                    run(self.session, f"show {spec_all} atoms; style {spec_all} stick; color {spec_all} byelement")
                    run(self.session, f"transparency {spec_all} 0 target a")

                # 4. Draw Intuitive Pseudobonds
                model_residues = {(r.chain_id, r.number): r for r in mutated_model.residues}
                count_lines = 0
                
                for c1, p1, m1, c2, p2, m2, score in pairs_to_draw:
                    res1, res2 = model_residues.get((c1, p1)), model_residues.get((c2, p2))
                    if not res1 or not res2 or not res1.atoms or not res2.atoms:
                        continue

                    # Find closest side-chain atoms (excluding backbone)
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
                    
                    # Visual Scaling Calculations
                    norm_score = min(1.0, max(0.0, (abs(score) - threshold) / (max_abs_score - threshold)))
                    radius_val = 0.05 + (0.95 * (norm_score ** 2)) 
                    intensity = int(100 + 155 * norm_score)        
                    alpha = int(30 + 225 * norm_score)             
                    
                    # Format color: Use comma-separated RGBA (0-255) to bypass all hex parser issues
                    # ChimeraX color-spec: R,G,B,A
                    if score > 0:
                        color_spec = f"0,{intensity},0,{alpha}" # Green
                    else:
                        color_spec = f"{intensity},0,0,{alpha}" # Red
                    
                    cmd = (f"pbond #{self.mutated_model_id_string}/{c1}:{p1}@{a1.name} "
                        f"#{self.mutated_model_id_string}/{c2}:{p2}@{a2.name} "
                        f"reveal true color {color_spec} "
                        f"radius {radius_val:.3f} name \"{score:.2f}\"")
                    
                    run(self.session, cmd)
                    count_lines += 1
                    
                # Apply Transparency to non-target chains
                target_chain = self.pred_chain_id_combobox.currentText().strip()
                alpha_val = self.non_target_alpha_spinbox.value()
                if target_chain:
                    exclude_spec = f"~#{wt_model.id_string}/{target_chain}"
                    if self.mutated_model_id_string:
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
                         if not (self.mutated_model_id_string and m.id_string == self.mutated_model_id_string)]
                         
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        
        if not wt_model and wt_candidates:
             wt_model = wt_candidates[0]
             
        if not wt_model:
            self.status_label.setText("Status: No suitable WT model open.")
            raise AssertionError("Failed to apply Visualization. No suitable WT model open.")
            
        if not self.residue_scores_data:
            self.status_label.setText("Status: No scores to apply.")
            return

        threshold = self.score_threshold_spinbox.value()
        color_backbone = self.color_backbone_checkbox.isChecked()
        show_sticks = self.show_sticks_checkbox.isChecked()
        wt_stick_alpha = self.wt_stick_alpha_spinbox.value()
        mut_stick_alpha = self.mut_stick_alpha_spinbox.value()

        if self.mutated_model_id_string and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
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
                run(self.session, f"combine #{wt_model.id_string} name \"{wt_model.name}_mutated_viz\"")
                mut_model = self.session.models.list()[-1]
                self.mutated_model_id_string = mut_model.id_string 
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

                run(self.session, f"match #{self.mutated_model_id_string} to #{wt_model.id_string}")
            except Exception as e:
                self.status_label.setText("Status: Error showing sticks.")
                self.session.logger.error(f"Error in stick viz: {e}")
                if mut_model and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
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
                    run(self.session, f"select {mut_spec}")
                    run(self.session, f"contacts sel restrict #{wt_model.id_string} reveal false makePseudobonds false select true")
                    run(self.session, f"select subtract {mut_spec}")
                    run(self.session, f"select subtract {wt_spec}")
                    run(self.session, f"select subtract backbone")
                    run(self.session, "show sel")
                    run(self.session, "style sel ball")
                    run(self.session, "color sel byelement")
                    run(self.session, f"transparency sel 60 target a")
                    run(self.session, "hide @h*")
            except Exception as e:
                self.session.logger.error(f"Error displaying contacts: {e}")

        target_chain = self.pred_chain_id_combobox.currentText().strip()
        alpha_val = self.non_target_alpha_spinbox.value()
        if target_chain:
            exclude_spec = f"~#{wt_model.id_string}/{target_chain}"
            if self.mutated_model_id_string:
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

    # -------------- lifecycle --------------
    def delete(self):
        self.session.logger.info("****** RSVTool delete CALLED ******")
        self._closing = True

        try:
            if hasattr(self, '_models_added_handler'):
                self.session.triggers.remove_handler(self._models_added_handler)
            if hasattr(self, '_models_removed_handler'):
                self.session.triggers.remove_handler(self._models_removed_handler)
        except Exception as e:
            self.session.logger.warning(f"Could not remove event handlers: {e}")

        try:
            from Qt.QtCore import QProcess
            if getattr(self, 'proc', None) and self.proc.state() != QProcess.NotRunning:
                self.proc.kill()
                self.proc.waitForFinished(200)
        except Exception as e:
            self.session.logger.warning(f"Proc kill on delete failed: {e}")

        try:
            if getattr(self, '_temp_dir_to_cleanup', None) and os.path.isdir(self._temp_dir_to_cleanup):
                shutil.rmtree(self._temp_dir_to_cleanup)
        except Exception as e:
            self.session.logger.warning(f"Could not remove temp dir on delete: {e}")
        finally:
            self._temp_dir_to_cleanup = None

        try:
            mid = getattr(self, 'mutated_model_id_string', None)
            if mid:
                if any(m.id_string == mid for m in self.session.models.list()):
                    run(self.session, f"close #{mid}")
        except Exception as e:
            self.session.logger.warning(f"Could not close mutated model on delete: {e}")
        finally:
            self.mutated_model_id_string = None

        super().delete()

print("****** ESM_MSR_VisualizerTool class definition COMPLETE ******")