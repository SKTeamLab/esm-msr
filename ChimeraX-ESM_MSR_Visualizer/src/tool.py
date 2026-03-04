# vim: set expandtab shiftwidth=4 softstop=4:
import os
import subprocess  # kept for CUDA probe via python -c
import tempfile
import shutil
import numpy as np

from chimerax.ui import MainToolWindow
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.core import colors

from chimerax.atomic import Structure

from Qt.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QLineEdit, QFrame, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox
)
from Qt.QtCore import QProcess, QSettings

print("****** ESM_MSR_Visualizer/tool.py TOP LEVEL EXECUTING ******")

SCORE_ATTRIBUTE_NAME = "residue_score_viz_score"

ONE_TO_THREE_LETTER_AA = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL'
}

class ESM_MSR_VisualizerTool(ToolInstance):
    print("****** ESM_MSR_VisualizerTool class DEFINED ******")

    SESSION_ENDURING = False
    SESSION_SAVE = False
    display_name = "Residue Score Visualizer & Predictor"

    def __init__(self, session, tool_registered_name):
        super().__init__(session, tool_registered_name)
        self.session.logger.info(f"****** RSVTool __init__ ({tool_registered_name}) ******")

        # 1. INITIALIZE SETTINGS FIRST
        self.settings = QSettings("ESM_MSR_Tools", "ESM_MSR_Visualizer")
        
        # 2. LOAD STATE VARIABLES
        self.base_repo_path = self.settings.value("base_repo_path", "")
        self.python_env = self.settings.value("python_env", "")
        
        self.lora_alpha = 12
        self.lora_rank = 6
        self.script_input_structure_path = ""
        
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
        
        # 5. SET REMAINING UI DEFAULTS
        self.lora_alpha_spinbox.setValue(self.lora_alpha)
        self.lora_rank_spinbox.setValue(self.lora_rank)
        self.score_threshold_spinbox.setValue(0.0)
        self.python_env_edit.setText(self.python_env)

        self.disallow_cysteine_checkbox.setChecked(False)
        self.color_backbone_checkbox.setChecked(False)
        self.show_sticks_checkbox.setChecked(True)
        
        self.multi_paths_spinbox.setValue(4)
        self.mode_combobox.setCurrentText('singles')
        self.strategy_combobox.setCurrentText('masked')
        self.subset_df_edit.setText('')
        self.mutations_edit.setText('')
        self.hf_token_edit.setText('')

        # 6. REGISTER CHIMERAX EVENT HOOKS
        try:
            self._models_added_handler = self.session.triggers.add_handler('add models', self._refresh_models)
            self._models_removed_handler = self.session.triggers.add_handler('remove models', self._refresh_models)
        except Exception as e:
            self.session.logger.warning(f"Could not register model event hooks: {e}")

        # Initial populate
        self._refresh_models()

        self.session.logger.info("****** RSVTool __init__ COMPLETED. ******")

# ---------------- UI -----------------
    def _build_ui(self, parent_widget):
        main_layout = QVBoxLayout()
        parent_widget.setLayout(main_layout)

        prediction_groupbox = QGroupBox("Run External Prediction")
        prediction_layout = QVBoxLayout()
        prediction_groupbox.setLayout(prediction_layout)
        main_layout.addWidget(prediction_groupbox)

        # Base Repository Path (Persistent)
        base_repo_layout = QHBoxLayout()
        base_repo_layout.addWidget(QLabel("Base Repo Directory:"))
        self.base_repo_path_edit = QLineEdit()
        self.base_repo_path_edit.setPlaceholderText("Select the root esm-msr folder...")
        self.base_repo_path_edit.setText(self.base_repo_path)
        base_repo_layout.addWidget(self.base_repo_path_edit)
        self.browse_base_repo_button = QPushButton("Browse...")
        self.browse_base_repo_button.clicked.connect(self._browse_base_repo)
        base_repo_layout.addWidget(self.browse_base_repo_button)
        prediction_layout.addLayout(base_repo_layout)

        # Input structure (auto-saved)
        script_input_struct_layout = QHBoxLayout()
        script_input_struct_layout.addWidget(QLabel("Input Structure File:"))
        self.script_input_structure_path_edit = QLineEdit()
        self.script_input_structure_path_edit.setPlaceholderText("Current model (will be auto-saved)")
        self.script_input_structure_path_edit.setReadOnly(True)
        script_input_struct_layout.addWidget(self.script_input_structure_path_edit)
        prediction_layout.addLayout(script_input_struct_layout)

        # Output CSV
        script_output_csv_layout = QHBoxLayout()
        script_output_csv_layout.addWidget(QLabel("Prediction Output File (CSV):"))
        self.script_output_csv_path_edit = QLineEdit()
        self.script_output_csv_path_edit.setPlaceholderText("Path where script will save output.csv")
        script_output_csv_layout.addWidget(self.script_output_csv_path_edit)
        self.browse_script_output_csv_button = QPushButton("Browse...")
        self.browse_script_output_csv_button.clicked.connect(self._browse_script_output_csv)
        script_output_csv_layout.addWidget(self.browse_script_output_csv_button)
        prediction_layout.addLayout(script_output_csv_layout)

        # Checkpoint
        lora_path_layout = QHBoxLayout()
        lora_path_layout.addWidget(QLabel("LoRA Path:"))
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("Path to LoRA model .ckpt")
        lora_path_layout.addWidget(self.checkpoint_path_edit)
        self.browse_checkpoint_button = QPushButton("Browse...")
        self.browse_checkpoint_button.clicked.connect(self._browse_checkpoint_path)
        lora_path_layout.addWidget(self.browse_checkpoint_button)
        
        # LoRA
        lora_path_layout.addWidget(QLabel("Alpha:"))
        self.lora_alpha_spinbox = QSpinBox()
        self.lora_alpha_spinbox.setRange(0, 1024)
        self.lora_alpha_spinbox.setSingleStep(1)
        lora_path_layout.addWidget(self.lora_alpha_spinbox)
        lora_path_layout.addWidget(QLabel("Rank:"))
        self.lora_rank_spinbox = QSpinBox()
        self.lora_rank_spinbox.setRange(0, 1024)
        self.lora_rank_spinbox.setSingleStep(1)
        lora_path_layout.addWidget(self.lora_rank_spinbox)
        
        prediction_layout.addLayout(lora_path_layout)
        
        # Inference Strategy Options
        strategy_layout = QHBoxLayout()
        
        strategy_layout.addWidget(QLabel("Model:"))
        self.pred_model_combobox = QComboBox()
        self.pred_model_combobox.currentIndexChanged.connect(self._on_model_selected)
        strategy_layout.addWidget(self.pred_model_combobox)

        strategy_layout.addWidget(QLabel("Chain ID:"))
        self.pred_chain_id_combobox = QComboBox()
        self.pred_chain_id_combobox.setEditable(True)
        strategy_layout.addWidget(self.pred_chain_id_combobox)
        
        strategy_layout.addSpacing(15)
        strategy_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combobox = QComboBox()
        self.strategy_combobox.addItems(['parallel', 'masked', 'direct'])
        strategy_layout.addWidget(self.strategy_combobox)

        strategy_layout.addSpacing(15)
        strategy_layout.addWidget(QLabel("Multi Paths:"))
        self.multi_paths_spinbox = QSpinBox()
        self.multi_paths_spinbox.setRange(1, 100)
        strategy_layout.addWidget(self.multi_paths_spinbox)

        self.disallow_cysteine_checkbox = QCheckBox("Disallow cysteine")
        strategy_layout.addWidget(self.disallow_cysteine_checkbox)

        prediction_layout.addLayout(strategy_layout)

        # --- Mutations to Score (Mutually Exclusive) ---
        mutations_groupbox = QGroupBox("Mutations to Score (Mutually Exclusive)")
        mutations_groupbox.setStyleSheet("QGroupBox { font-weight: bold; }")
        mutations_layout = QVBoxLayout()
        mutations_groupbox.setLayout(mutations_layout)
        
        warning_label = QLabel("Note: Provide input for ONLY ONE of the following three methods.")
        warning_label.setStyleSheet("color: #aa5500; font-style: italic;")
        mutations_layout.addWidget(warning_label)

        # Method 1: Screening Mode & Selected Residues
        meth1_layout = QHBoxLayout()
        meth1_layout.addWidget(QLabel("1. Screening mode:"))
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['singles', 'doubles'])
        meth1_layout.addWidget(self.mode_combobox)
        
        meth1_layout.addSpacing(10)
        meth1_layout.addWidget(QLabel("Selected residues:"))
        self.selected_residues_edit = QLineEdit()
        self.selected_residues_edit.setPlaceholderText("e.g. 11,12,15 (Empty = All)")
        meth1_layout.addWidget(self.selected_residues_edit)
        
        self.grab_sel_button = QPushButton("Grab Selection")
        self.grab_sel_button.setToolTip("Grab currently selected residues from ChimeraX")
        self.grab_sel_button.clicked.connect(self._grab_selection)
        meth1_layout.addWidget(self.grab_sel_button)
        
        self.screen_except_checkbox = QCheckBox("Invert (Except these)")
        meth1_layout.addWidget(self.screen_except_checkbox)
        mutations_layout.addLayout(meth1_layout)

        # Method 2: Subset CSV
        meth2_layout = QHBoxLayout()
        meth2_layout.addWidget(QLabel("2. Subset csv:"))
        self.subset_df_edit = QLineEdit()
        self.subset_df_edit.setPlaceholderText("Optional: Path to CSV of mutations")
        meth2_layout.addWidget(self.subset_df_edit)
        self.browse_subset_button = QPushButton("Browse...")
        self.browse_subset_button.clicked.connect(self._browse_subset_df)
        meth2_layout.addWidget(self.browse_subset_button)
        mutations_layout.addLayout(meth2_layout)
        
        # Method 3: Specific Mutations
        meth3_layout = QHBoxLayout()
        meth3_layout.addWidget(QLabel("3. Specific Mutations:"))
        self.mutations_edit = QLineEdit()
        self.mutations_edit.setPlaceholderText("Optional: Comma-separated (e.g., A12C,A12C:D15E)")
        meth3_layout.addWidget(self.mutations_edit)
        mutations_layout.addLayout(meth3_layout)

        prediction_layout.addWidget(mutations_groupbox)

        # Env
        py_env_layout = QHBoxLayout()
        py_env_layout.addWidget(QLabel("Environment Path or Conda Name:"))
        self.python_env_edit = QLineEdit()
        self.python_env_edit.setPlaceholderText("e.g., C:\\...\\msr_venv OR conda_env_name")
        py_env_layout.addWidget(self.python_env_edit)
        self.browse_env_button = QPushButton("Browse...")
        self.browse_env_button.clicked.connect(self._browse_python_env)
        py_env_layout.addWidget(self.browse_env_button)
        prediction_layout.addLayout(py_env_layout)
        
        # Hardware & Auth
        hw_layout = QHBoxLayout()
        hw_layout.addWidget(QLabel("HF Token (if ESM3 weights not downloaded to repo):"))
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setPlaceholderText("Optional: HuggingFace Token")
        hw_layout.addWidget(self.hf_token_edit)
        prediction_layout.addLayout(hw_layout)

        self.run_prediction_button = QPushButton("Run Prediction Script")
        self.run_prediction_button.clicked.connect(self._initiate_run_prediction_script)
        prediction_layout.addWidget(self.run_prediction_button)

        self.prediction_output_label = QLabel("Predicted output file: None")
        prediction_layout.addWidget(self.prediction_output_label)

        separator = QFrame(); separator.setFrameShape(QFrame.HLine); separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # --- Visualization Section ---
        visualization_groupbox = QGroupBox("Load and Visualize Scores")
        visualization_layout = QVBoxLayout(); visualization_groupbox.setLayout(visualization_layout)
        main_layout.addWidget(visualization_groupbox)

        self.csv_label = QLabel("No CSV loaded.")
        visualization_layout.addWidget(self.csv_label)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Score Threshold:"))
        self.score_threshold_spinbox = QDoubleSpinBox()
        self.score_threshold_spinbox.setRange(-1000.0, 1000.0)
        self.score_threshold_spinbox.setSingleStep(0.05)
        self.score_threshold_spinbox.setValue(0.0)
        self.score_threshold_spinbox.setDecimals(3)
        threshold_layout.addWidget(self.score_threshold_spinbox)
        
        threshold_layout.addSpacing(20)
        
        threshold_layout.addWidget(QLabel("Non-target Chain Alpha (Transparency %):"))
        self.non_target_alpha_spinbox = QSpinBox()
        self.non_target_alpha_spinbox.setRange(0, 100)
        self.score_threshold_spinbox.setSingleStep(10)
        self.non_target_alpha_spinbox.setValue(90)
        threshold_layout.addWidget(self.non_target_alpha_spinbox)
        threshold_layout.addStretch()
        
        visualization_layout.addLayout(threshold_layout)

        # Standard Visualization Options
        self.color_backbone_checkbox = QCheckBox("Color Backbone by Score (Red-to-Green)")
        self.color_backbone_checkbox.setChecked(False)
        visualization_layout.addWidget(self.color_backbone_checkbox)

        self.show_sticks_checkbox = QCheckBox("Show Sticks for High-Scoring Mutations")
        self.show_sticks_checkbox.setChecked(True)
        visualization_layout.addWidget(self.show_sticks_checkbox)

        self.show_contacts_checkbox = QCheckBox("Visualize Contacts")
        self.show_contacts_checkbox.setChecked(False)
        visualization_layout.addWidget(self.show_contacts_checkbox)

        # --- Epistasis Analysis Section ---
        epistasis_groupbox = QGroupBox("Epistasis Analysis")
        epistasis_layout = QVBoxLayout()
        epistasis_groupbox.setLayout(epistasis_layout)
        visualization_layout.addWidget(epistasis_groupbox)

        self.epistasis_checkbox = QCheckBox("Epistasis Mode (Mutually Exclusive)")
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
        # -----------------------------------   

        self.load_button = QPushButton("Load CSV + Visualize Scores")
        self.load_button.clicked.connect(self._handle_load_and_visualize)
        visualization_layout.addWidget(self.load_button)

        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()

    def _grab_selection(self):
        try:
            from chimerax.atomic import selected_atoms
            atoms = selected_atoms(self.session)
            if atoms is None or len(atoms) == 0:
                self.selected_residues_edit.setText("")
                self.status_label.setText("Status: No residues selected.")
                return
            
            # Extract unique residue numbers and sort them
            res_nums = sorted(list(set(atoms.residues.numbers)))
            if res_nums:
                self.selected_residues_edit.setText(",".join(map(str, res_nums)))
                self.status_label.setText(f"Status: Grabbed {len(res_nums)} unique residue indices.")
        except Exception as e:
            self.session.logger.warning(f"Could not grab selection: {e}")
            self.status_label.setText("Status: Failed to grab selection.")

    def _on_epistasis_toggled(self, checked):
        # Enforce mutual exclusivity in UI
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
        if hasattr(self, 'python_script_path_edit'):
            self.python_script_path_edit.setText(self.python_script_path)

    # ---------------- helpers -----------------
    def _browse_python_script(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, "Select Python Script", self.python_script_path_edit.text(), "Python Files (*.py);;All Files (*)")
        if fp:
            self.python_script_path = fp
            self.python_script_path_edit.setText(fp)
            self.session.logger.info(f"Python script selected: {fp}")

    def _browse_script_output_csv(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getSaveFileName(w, "Specify Prediction Output CSV File Path", self.script_output_csv_path_edit.text(), "CSV Files (*.csv);;All Files (*)")
        if fp:
            if not fp.lower().endswith('.csv'):
                fp += '.csv'
            self.script_output_csv_path = fp
            self.script_output_csv_path_edit.setText(fp)
            self.session.logger.info(f"Script output CSV file path set to: {fp}")

    def _browse_checkpoint_path(self):
        w = self.session.ui.main_window
        fp, _ = QFileDialog.getOpenFileName(w, "Select Checkpoint File", self.checkpoint_path_edit.text(), "Checkpoint Files (*.ckpt *.pt *.pth *.h5);;All Files (*)")
        if fp:
            self.checkpoint_path = fp
            self.checkpoint_path_edit.setText(fp)
            self.session.logger.info(f"Checkpoint file selected: {fp}")

    def _browse_python_env(self):
        w = self.session.ui.main_window
        folder_path = QFileDialog.getExistingDirectory(w, "Select Python/Conda Environment Folder", self.python_env_edit.text())
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            self.python_env_edit.setText(folder_path)
            self.python_env = folder_path
            self.settings.setValue("python_env", folder_path)
            self.session.logger.info(f"Environment path persistently saved: {folder_path}")

    def _browse_subset_df(self):
            w = self.session.ui.main_window
            fp, _ = QFileDialog.getOpenFileName(w, "Select Subset DataFrame CSV", self.base_repo_path, "CSV Files (*.csv);;All Files (*)")
            if fp:
                fp = os.path.normpath(fp)
                self.subset_df_edit.setText(fp)
                self.session.logger.info(f"Subset DF selected: {fp}")

    def _early_error_ui_reset(self, message):
        self.status_label.setText(f"Status: {message}")
        self.run_prediction_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.script_input_structure_path_edit.setText("")
        self.script_input_structure_path_edit.setPlaceholderText("Current model (will be auto-saved)")
        self.session.logger.warning(f"Early error occurred: {message}")

    # -------------- run prediction (QProcess) --------------
    def _initiate_run_prediction_script(self):
        self.session.logger.info("****** _initiate_run_prediction_script called ******")

        # cleanup old temp
        if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
            try:
                shutil.rmtree(self._temp_dir_to_cleanup)
            except Exception:
                pass
            finally:
                self._temp_dir_to_cleanup = None
                self.script_input_structure_path = ""
                self.script_input_structure_path_edit.setText("")
                self.script_input_structure_path_edit.setPlaceholderText("Current model (will be auto-saved)")

        self.base_repo_path = self.base_repo_path_edit.text().strip()
        self.python_script_path = os.path.normpath(os.path.join(self.base_repo_path, "src", "esm_msr", "inference.py"))
        
        self.script_output_csv_path = self.script_output_csv_path_edit.text().strip()
        self.checkpoint_path = self.checkpoint_path_edit.text().strip()
        self.lora_alpha = self.lora_alpha_spinbox.value()
        self.lora_rank = self.lora_rank_spinbox.value()
        self.python_env = self.python_env_edit.text().strip()
        self.chain_id = self.pred_chain_id_combobox.currentText().strip()
        self.disallow_cysteine = self.disallow_cysteine_checkbox.isChecked()

        self.settings.setValue("python_env", self.python_env)
        self.settings.setValue("base_repo_path", self.base_repo_path)

        model_id = self.pred_model_combobox.currentData()
        models = self.session.models.list(type=Structure)
        current_model = next((m for m in models if m.id_string == model_id), None)
        
        if not current_model:
            self._early_error_ui_reset("Error: Selected model not found. Please ensure a valid model is open.")
            return

        # create temp input pdb
        try:
            self._temp_dir_to_cleanup = tempfile.mkdtemp(prefix="chimerax_rsv_input_")
            temp_model_filename = f"current_model_input_{current_model.id_string.replace(':', '_').replace('/', '_')}.pdb"
            _temp_script_input_structure_path = os.path.join(self._temp_dir_to_cleanup, temp_model_filename)
            run(self.session, f"save \"{_temp_script_input_structure_path}\" models #{current_model.id_string} format pdb")
            self.script_input_structure_path = _temp_script_input_structure_path
            self.script_input_structure_path_edit.setText(self.script_input_structure_path)
        except Exception as e:
            self.session.logger.error(f"Failed to prepare input structure: {e}")
            self._early_error_ui_reset("Error: Failed to save current model.")
            try:
                if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
                    shutil.rmtree(self._temp_dir_to_cleanup)
            finally:
                self._temp_dir_to_cleanup = None
            return

        if not (self.python_script_path and os.path.exists(self.python_script_path)):
            self.status_label.setText("Status: Python script path is invalid or not found.")
            return
        if not (self.script_input_structure_path and os.path.exists(self.script_input_structure_path)):
            self.status_label.setText("Status: Temporary input structure file is missing after save attempt.")
            return
        if not self.script_output_csv_path:
            self.status_label.setText("Status: Please specify an output CSV file path for the prediction.")
            return
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            self.status_label.setText("Status: Checkpoint path is invalid or not found.")
            return

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
                    self.session.logger.warning(f"Could not find python inside {env}. Falling back to default python.")
                    program = 'python'
            else:
                program = 'conda'
                args.extend(['run', '-n', env, 'python'])
        else:
            program = 'python'

        device = 'cpu'
        try:
            probe_cmd = [program] + (args if program == 'python' else args) + ['-c', 'import torch; print(torch.cuda.is_available())']
            if program == 'python':
                r = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
                device = 'cuda:0' if r.stdout.strip().lower() == 'true' else 'cpu'
            else:
                device = 'cuda:0'
        except Exception as e:
            self.session.logger.warning(f"CUDA probe failed: {e}")
            device = 'cpu'

        if current_model.name:
            code = os.path.splitext(current_model.name)[0]
        else:
            code = 'protein'

        script_args = [
            self.python_script_path,
            '--input_structure', self.script_input_structure_path,
            '--checkpoint', self.checkpoint_path,
            '--lora_alpha', str(self.lora_alpha),
            '--lora_rank', str(self.lora_rank),
            '--multi_paths', str(self.multi_paths_spinbox.value()),
            '--code', code,
            '--chain_id', self.chain_id,
            '--mode', self.mode_combobox.currentText(),
            '--strategy', self.strategy_combobox.currentText(),
            '--device', device,
            '--calculate_distances'
        ]

        if self.script_output_csv_path:
            script_args += ['--output_csv', self.script_output_csv_path]

        if self.disallow_cysteine:
            script_args += ['--disallow_residues', 'C']

        # Selected residues processing
        selected_res = self.selected_residues_edit.text().strip()
        if selected_res:
            if self.screen_except_checkbox.isChecked():
                script_args += ['--screen_residues_except', selected_res]
            else:
                script_args += ['--screen_residues', selected_res]

        subset_df_val = self.subset_df_edit.text().strip()
        if subset_df_val:
            script_args += ['--subset_df', subset_df_val]

        mutations_val = self.mutations_edit.text().strip()
        if mutations_val:
            script_args += ['--mutations', mutations_val]

        hf_token_val = self.hf_token_edit.text().strip()
        if hf_token_val:
            script_args += ['--hf_token', hf_token_val]

        full_args = args + script_args if program != 'python' else script_args

        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.session.logger.warning("Prediction already running; ignoring new request.")
            return
        self.proc = QProcess()
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.errorOccurred.connect(self._on_proc_error)
        self.proc.finished.connect(self._on_proc_finished)

        self.status_label.setText("Status: Running prediction script")
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
        except Exception:
            pass

    def _on_proc_error(self, err):
        try:
            self.session.logger.error(f"Prediction process error: {err}")
        except Exception:
            pass

    def _on_proc_finished(self, exitCode, exitStatus):
        self.run_prediction_button.setEnabled(True)
        self.load_button.setEnabled(True)
        if self._temp_dir_to_cleanup and os.path.isdir(self._temp_dir_to_cleanup):
            try:
                shutil.rmtree(self._temp_dir_to_cleanup)
            except Exception as e:
                self.session.logger.warning(f"Could not remove temporary directory {self._temp_dir_to_cleanup}: {e}")
            self._temp_dir_to_cleanup = None
            self.script_input_structure_path_edit.setText("")
            self.script_input_structure_path_edit.setPlaceholderText("Current model (will be auto-saved)")
            self.script_input_structure_path = ""

        if exitStatus == QProcess.NormalExit and exitCode == 0:
            self.predicted_output_path = self.script_output_csv_path
            self.prediction_output_label.setText(f"Predicted output: {os.path.basename(self.predicted_output_path)}")
            self.status_label.setText("Status: Prediction script completed. Load CSV to visualize.")
        else:
            self.prediction_output_label.setText("Predicted output file: Failed")
            self.status_label.setText(f"Status: Script Error (exit={exitCode}). See log.")

        try:
            self.proc.deleteLater()
        except Exception:
            pass
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

        self.residue_scores_data = {}
        self.epistasis_df = None
        try:
            df = pd.read_csv(filepath)
            df.columns = [c.lower().strip() for c in df.columns]

            if is_epistasis:
                required_cols = {'chain_id', 'pos1_pdb', 'mut1', 'pos2_pdb', 'mut2', 'dddg_pred'}
                if not required_cols.issubset(set(df.columns)):
                    self.session.logger.error(f"Missing columns in CSV for Epistasis. Found: {list(df.columns)}. Expected at least: {required_cols}")
                    self.status_label.setText("Status: Error - Missing epistasis columns (chain_id, pos1/2, mut1/2, dddg_pred).")
                    return False
                df['chain_id'] = df['chain_id'].astype(str).str.strip()
                self.epistasis_df = df
                self.session.logger.info(f"Parsed epistasis dataframe with {len(df)} rows.")
                return True
            else:
                required_cols = {'chain_id', 'pos1_pdb', 'mut1', 'ddg_pred'}
                if not required_cols.issubset(set(df.columns)):
                    self.session.logger.error(f"Missing columns in CSV. Found: {list(df.columns)}. Expected at least: {required_cols}")
                    self.status_label.setText("Status: Error - Missing columns (chain_id, pos1_pdb, mut1, ddg_pred).")
                    return False

                df['chain_id'] = df['chain_id'].astype(str).str.strip()
                pivot_df = df.pivot_table(index=['chain_id', 'pos1_pdb'], columns='mut1', values='ddg_pred')
                
                if pivot_df.empty:
                     self.status_label.setText("Status: Parsed CSV, but data was empty.")
                     return False

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
                    self.status_label.setText("Status: Parsed CSV, but no valid non-zero scores found.")
                    return False

                self.session.logger.info(f"Parsed scores for {len(self.residue_scores_data)} positions across chains.")
                return True

        except Exception as e:
            self.session.logger.error(f"Error reading CSV with Pandas: {e}")
            self.status_label.setText("Status: Error parsing CSV (see log).")
            return False

    def _apply_epistasis_visualization(self):
        wt_candidates = [m for m in self.session.models.list(type=Structure)
                         if not (self.mutated_model_id_string and m.id_string == self.mutated_model_id_string)]
                         
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        
        # Fallback in case models were shifted around
        if not wt_model and wt_candidates:
            wt_model = wt_candidates[0]

        if not wt_model:
            self.status_label.setText("Status: No suitable WT model open.")
            return
        
        # Cleanup previous mutated model
        if self.mutated_model_id_string and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
            try:
                run(self.session, f"close #{self.mutated_model_id_string}")
            except Exception: pass
            finally: self.mutated_model_id_string = None

        # 1. Filter Data by Threshold
        threshold = self.epistasis_threshold_spinbox.value()
        df = self.epistasis_df
        filtered_df = df[df['dddg_pred'].abs() >= threshold].copy()

        if filtered_df.empty:
            self.status_label.setText("Status: No residue pairs found exceeding dddg_pred threshold.")
            return

        # Create an absolute score column to sort by magnitude
        filtered_df['abs_score'] = filtered_df['dddg_pred'].abs()
        sorted_df = filtered_df.sort_values(by='abs_score', ascending=False)
        
        # Get maximum absolute score for dynamic distance radius and color rendering
        #max_abs_score = sorted_df['abs_score'].max()
        #if max_abs_score == 0:
        #    max_abs_score = 1.0

        # 2. Greedy Resolution of Mutation Conflicts
        mutation_plan = {} # (chain, pos) -> mut_aa
        pairs_to_draw = [] # list of (c1, p1, m1, c2, p2, m2, score)

        for _, row in sorted_df.iterrows():
            c1, p1, m1 = str(row['chain_id']).strip(), int(row['pos1_pdb']), str(row['mut1']).upper()
            c2, p2, m2 = str(row['chain_id']).strip(), int(row['pos2_pdb']), str(row['mut2']).upper()
            score = float(row['dddg_pred'])
            
            # Check compatibility: A position is compatible if it's either not mapped yet, 
            # or already mapped to the EXACT SAME amino acid.
            comp1 = ((c1, p1) not in mutation_plan) or (mutation_plan[(c1, p1)] == m1)
            comp2 = ((c2, p2) not in mutation_plan) or (mutation_plan[(c2, p2)] == m2)
            
            if comp1 and comp2:
                # Lock in the specific mutant for these positions
                mutation_plan[(c1, p1)] = m1
                mutation_plan[(c2, p2)] = m2
                pairs_to_draw.append((c1, p1, m1, c2, p2, m2, score))

        # 3. Create Mutated Model
        try:
            run(self.session, f"combine #{wt_model.id_string} name \"{wt_model.name}_epistasis_viz\"")
            mutated_model = self.session.models.list()[-1]
            self.mutated_model_id_string = mutated_model.id_string
            
            run(self.session, f"color #{self.mutated_model_id_string} white")
            run(self.session, f"transparency #{self.mutated_model_id_string} 70 target a")
            run(self.session, f"ribbon style #{self.mutated_model_id_string}")
            run(self.session, f"hide #{self.mutated_model_id_string} atoms")

            # Apply locked mutations
            for (chain_val, pos), tgt_aa in mutation_plan.items():
                res_wt = next((r for r in mutated_model.residues if r.number == pos and r.chain_id == chain_val), None)
                if res_wt and ONE_TO_THREE_LETTER_AA.get(tgt_aa, '') != res_wt.name:
                    spec = f"#{self.mutated_model_id_string}/{chain_val}:{pos}"
                    try:
                        three_letter = ONE_TO_THREE_LETTER_AA[tgt_aa].lower()
                        run(self.session, f"swapaa {spec} {three_letter} log false")
                    except Exception as e:
                        self.session.logger.warning(f"Failed to mutate chain {chain_val} pos {pos} to {tgt_aa}: {e}")

            # Show mutated residues as sticks
            if mutation_plan:
                chain_to_pos = {}
                for (chain_val, pos) in mutation_plan.keys():
                    chain_to_pos.setdefault(chain_val, []).append(str(pos))
                
                specs = []
                for chain_val, pos_list in chain_to_pos.items():
                    specs.append(f"#{self.mutated_model_id_string}/{chain_val}:{','.join(pos_list)}")
                
                spec_all = " | ".join(specs)
                run(self.session, f"show {spec_all} atoms")
                run(self.session, f"style {spec_all} stick")
                run(self.session, f"color {spec_all} byelement")
                run(self.session, f"transparency {spec_all} 0 target a")

            # 4. Draw Connection Lines on Swapped Side Chains
            model_residues = {(r.chain_id, r.number): r for r in mutated_model.residues}
            count_lines = 0
            
            for c1, p1, m1, c2, p2, m2, score in pairs_to_draw:
                res1 = model_residues.get((c1, p1))
                res2 = model_residues.get((c2, p2))
                
                if not res1 or not res2:
                    continue

                atoms1 = res1.atoms
                atoms2 = res2.atoms
                if len(atoms1) == 0 or len(atoms2) == 0:
                    continue
                
                # Filter out backbone to force side-chain to side-chain distance calculation
                valid_atoms1 = [a for a in atoms1 if a.name not in ('CA', 'C', 'N', 'O', 'H', 'HA', 'HB')]
                valid_atoms2 = [a for a in atoms2 if a.name not in ('CA', 'C', 'N', 'O', 'H', 'HA', 'HB')]
                
                # Fallback in case mutation is Glycine (no sidechain)
                if not valid_atoms1: valid_atoms1 = list(atoms1)
                if not valid_atoms2: valid_atoms2 = list(atoms2)
                
                coords1 = np.array([a.scene_coord for a in valid_atoms1])
                coords2 = np.array([a.scene_coord for a in valid_atoms2])
                
                diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                best_a1 = valid_atoms1[min_idx[0]]
                best_a2 = valid_atoms2[min_idx[1]]
                
                if best_a1 and best_a2:
                    # 1. Calculate dynamic thickness (radius)
                    # Adjust the multiplier (0.1) based on the typical magnitude of your scores
                    radius_val = max(0.05, min(0.8, abs(score) * 0.3))
                    
                    # 2. Calculate dynamic color intensity (0 to 255)
                    # Adjust the multiplier (50) so your expected max scores hit ~255
                    intensity = max(50, min(255, int((abs(score)-abs(threshold)) * 100)))
                    if score > 0:
                        hex_color = f"#00{intensity:02x}00"  # Dynamic Green
                    else:
                        hex_color = f"#{intensity:02x}0000"  # Dynamic Red
                    
                    cmd = (f"pbond #{self.mutated_model_id_string}/{c1}:{p1}@{best_a1.name} "
                           f"#{self.mutated_model_id_string}/{c2}:{p2}@{best_a2.name} "
                           f"reveal true color {hex_color} radius {radius_val:.3f} name {score}") #decimalPlaces 0  label false
                    run(self.session, cmd)
                    count_lines += 1
            
            # Apply alpha isolation logic 
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

            self.status_label.setText(f"Status: Epistasis Viz Complete. Drawn {count_lines} pairs.")

        except Exception as e:
            self.session.logger.error(f"Error in epistasis visualization: {e}")
            self.status_label.setText("Status: Error in Epistasis Viz.")


    def _apply_visualization(self):
        wt_candidates = [m for m in self.session.models.list(type=Structure)
                         if not (self.mutated_model_id_string and m.id_string == self.mutated_model_id_string)]
                         
        model_id = self.pred_model_combobox.currentData()
        wt_model = next((m for m in wt_candidates if m.id_string == model_id), None)
        
        # Fallback if combo boxes out of sync
        if not wt_model and wt_candidates:
             wt_model = wt_candidates[0]
             
        if not wt_model:
            self.status_label.setText("Status: No suitable WT model open.")
            return
            
        if not self.residue_scores_data:
            self.status_label.setText("Status: No scores to apply.")
            return

        threshold = self.score_threshold_spinbox.value()
        color_backbone = self.color_backbone_checkbox.isChecked()
        show_sticks = self.show_sticks_checkbox.isChecked()

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
        except Exception:
            self.status_label.setText("Status: Error styling WT model.")
            return

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
            except Exception:
                pass

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
                                run(self.session, f"swapaa {spec} {ONE_TO_THREE_LETTER_AA[tgt_aa].lower()} log true")
                                muts_by_chain.setdefault(chain, []).append(pos)
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
                    run(self.session, f"transparency {mut_spec} 50 target a")

                    run(self.session, f"color {wt_spec} & ~C & sideonly white")
                    run(self.session, f"color {wt_spec} & ~C & sideonly byelement")
                    run(self.session, f"show {wt_spec} atoms")
                    run(self.session, f"style {wt_spec} stick")
                    run(self.session, f"transparency {wt_spec} 50 target a")

                run(self.session, f"match #{self.mutated_model_id_string} to #{wt_model.id_string}")
            except Exception as e:
                self.status_label.setText("Status: Error showing sticks.")
                self.session.logger.error(f"Error in stick viz: {e}")
                if mut_model and any(m.id_string == self.mutated_model_id_string for m in self.session.models.list()):
                    run(self.session, f"close #{self.mutated_model_id_string}")
                self.mutated_model_id_string = None

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

        # Apply alpha isolation logic 
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

        # Unregister event hooks
        try:
            if hasattr(self, '_models_added_handler'):
                self.session.triggers.remove_handler(self._models_added_handler)
            if hasattr(self, '_models_removed_handler'):
                self.session.triggers.remove_handler(self._models_removed_handler)
        except Exception:
            pass

        try:
            from Qt.QtCore import QProcess
            if getattr(self, 'proc', None) and self.proc.state() != QProcess.NotRunning:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc.waitForFinished(200)
        except Exception as e:
            try:
                self.session.logger.warning(f"Proc kill on delete: {e}")
            except Exception:
                pass

        try:
            if getattr(self, '_temp_dir_to_cleanup', None) and os.path.isdir(self._temp_dir_to_cleanup):
                shutil.rmtree(self._temp_dir_to_cleanup)
        except Exception as e:
            self.session.logger.warning(f"Could not remove temp dir on delete: {e}")
        finally:
            self._temp_dir_to_cleanup = None
            if hasattr(self, 'script_input_structure_path_edit'):
                self.script_input_structure_path_edit.setText("")
                self.script_input_structure_path_edit.setPlaceholderText("Current model (will be auto-saved)")
            self.script_input_structure_path = ""

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