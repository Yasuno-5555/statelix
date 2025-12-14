from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QFrame, QHBoxLayout, 
    QPushButton, QFormLayout, QListWidget, QSpinBox, QDoubleSpinBox, QGroupBox
)
from PyQt6.QtCore import pyqtSignal

class ModelPanel(QWidget):
    run_requested = pyqtSignal(dict) # Signal to trigger execution

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("モデルパネル")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # 1. Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("モデル:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "OLS (最小二乗法)", 
            "GLM: Logistic",
            "GLM: Poisson",
            "Ridge Regression",
            "ANOVA (分散分析)", 
            "AR Model (時系列)",
            "--- Graph Analysis ---",
            "Graph: Louvain Communities",
            "Graph: PageRank",
            "--- Causal Inference ---",
            "Causal: IV (2SLS)",
            "Causal: Diff-in-Diff",
            "Causal: PSM (Propensity Matching)",
            "--- Bayesian ---",
            "Bayesian Logistic (HMC)",
            "--- Search ---",
            "Search: Build HNSW Index"
        ])
        
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # 2. Parameters (Dynamic Grid via GroupBox)
        self.param_group = QGroupBox("Hyperparameters")
        self.form_layout = QFormLayout()
        
        # --- Params Widgets ---
        self.spin_k = QSpinBox(); self.spin_k.setRange(2, 50); self.spin_k.setValue(3)
        self.spin_p = QSpinBox(); self.spin_p.setRange(1, 20); self.spin_p.setValue(1)
        self.spin_iter = QSpinBox(); self.spin_iter.setRange(10, 1000); self.spin_iter.setValue(50)
        self.spin_alpha = QDoubleSpinBox(); self.spin_alpha.setRange(0.001, 1000.0); self.spin_alpha.setValue(1.0)
        
        # Graph Params
        self.spin_resolution = QDoubleSpinBox(); self.spin_resolution.setValue(1.0)
        self.spin_damping = QDoubleSpinBox(); self.spin_damping.setValue(0.85); self.spin_damping.setRange(0.01, 0.99)
        
        # HNSW Params
        self.spin_hnsw_m = QSpinBox(); self.spin_hnsw_m.setValue(16)
        self.spin_hnsw_ef = QSpinBox(); self.spin_hnsw_ef.setValue(200); self.spin_hnsw_ef.setRange(10, 2000)

        # HMC Params
        self.spin_hmc_samples = QSpinBox(); self.spin_hmc_samples.setRange(100, 10000); self.spin_hmc_samples.setValue(1000)
        self.spin_hmc_warmup = QSpinBox(); self.spin_hmc_warmup.setRange(10, 5000); self.spin_hmc_warmup.setValue(200)

        # PSM Params (New)
        self.spin_caliper = QDoubleSpinBox(); self.spin_caliper.setRange(0.01, 1.0); self.spin_caliper.setValue(0.2); self.spin_caliper.setSingleStep(0.05)

        # --- Add Rows (Store references to hide later) ---
        self.rows = {}
        def add_row(key, label, widget):
            self.form_layout.addRow(label, widget)
            self.rows[key] = (self.form_layout.labelForField(widget), widget)

        add_row("k", "Clusters (k):", self.spin_k)
        add_row("p", "Order (p):", self.spin_p)
        add_row("iter", "Max Iter:", self.spin_iter)
        add_row("alpha", "Alpha (λ):", self.spin_alpha)
        add_row("res", "Resolution:", self.spin_resolution)
        add_row("damp", "Damping:", self.spin_damping)
        add_row("M", "HNSW M:", self.spin_hnsw_m)
        add_row("ef", "HNSW ef_const:", self.spin_hnsw_ef)
        add_row("samples", "Samples:", self.spin_hmc_samples)
        add_row("warmup", "Warmup:", self.spin_hmc_warmup)
        add_row("caliper", "Caliper (SD):", self.spin_caliper)
        
        self.param_group.setLayout(self.form_layout)
        layout.addWidget(self.param_group)

        # 3. Variable Selection (3 Lists)
        var_layout = QHBoxLayout()
        
        # List 1
        self.l1_layout = QVBoxLayout()
        self.l1_label = QLabel("Target (Y)")
        self.l1_list = QListWidget()
        self.l1_list.setMaximumHeight(80)
        self.l1_layout.addWidget(self.l1_label); self.l1_layout.addWidget(self.l1_list)
        var_layout.addLayout(self.l1_layout)
        
        # List 2
        self.l2_layout = QVBoxLayout()
        self.l2_label = QLabel("Features (X)")
        self.l2_list = QListWidget()
        self.l2_list.setMaximumHeight(80)
        self.l2_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.l2_layout.addWidget(self.l2_label); self.l2_layout.addWidget(self.l2_list)
        var_layout.addLayout(self.l2_layout)

        # List 3 (Aux)
        self.l3_layout = QVBoxLayout()
        self.l3_label = QLabel("Auxiliary")
        self.l3_list = QListWidget()
        self.l3_list.setMaximumHeight(80)
        self.l3_layout.addWidget(self.l3_label); self.l3_layout.addWidget(self.l3_list)
        var_layout.addLayout(self.l3_layout)
        
        layout.addLayout(var_layout)

        # 4. Actions
        action_layout = QHBoxLayout()
        self.run_btn = QPushButton("実行")
        self.run_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self.on_run)
        
        action_layout.addWidget(self.run_btn)
        layout.addLayout(action_layout)

        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize
        self.on_model_changed(0)

    def on_model_changed(self, index):
        model = self.model_combo.currentText()
        
        # 1. Reset Visibility
        for label, widget in self.rows.values():
            label.setVisible(False)
            widget.setVisible(False)
            
        # 2. Reset List Labels
        l1, l2, l3 = "Target (Y)", "Features (X)", "Auxiliary (Unused)"
        self.l3_list.setEnabled(False) # Disable by default
        self.l2_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # 3. Configure based on Task
        if "K-Means" in model:
            self._show("k"); self._show("iter")
            l1 = "(Unused)"; self.l1_list.setEnabled(False)
            
        elif "AR Model" in model:
            self._show("p")
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "Ridge" in model:
            self._show("alpha")

        elif "GLM" in model or "OLS" in model:
            self._show("iter") if "GLM" in model else None

        elif "Louvain" in model:
            self._show("res")
            l1, l2, l3 = "Source Node", "Target Node", "(Unused)"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        elif "PageRank" in model:
            self._show("damp")
            l1, l2, l3 = "Source Node", "Target Node", "(Unused)"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        elif "HNSW" in model:
            self._show("M"); self._show("ef")
            l1, l2, l3 = "(Unused)", "Vector Columns", "(Unused)"
            self.l1_list.setEnabled(False)

        elif "Bayesian" in model: # HMC
            self._show("samples"); self._show("warmup")
            l1, l2, l3 = "Target (0/1)", "Features (X)", "(Unused)"

        elif "IV (2SLS)" in model:
            l1, l2, l3 = "Outcome (Y)", "Endogenous (X)", "Instrument (Z)"
            self.l3_list.setEnabled(True)

        elif "Diff-in-Diff" in model:
            l1, l2, l3 = "Outcome (Y)", "Treatment (D)", "Post-Period (T) / Time"
            self.l3_list.setEnabled(True)
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection) # DID usually 1 treatment col
            
        elif "PSM" in model:
            self._show("caliper")
            l1, l2, l3 = "Outcome (Y)", "Covariates (X)", "Treatment (0/1)"
            self.l3_list.setEnabled(True)
            
        elif "ANOVA" in model:
             l1, l2 = "Data (Y)", "Group (X)"
             self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # Apply list labels
        self.l1_label.setText(l1); self.l1_list.setEnabled(l1 != "(Unused)")
        self.l2_label.setText(l2); self.l2_list.setEnabled(l2 != "(Unused)")
        self.l3_label.setText(l3); self.l3_list.setEnabled(l3 != "(Unused)" and "Unused" not in l3)

    def _show(self, key):
        self.rows[key][0].setVisible(True)
        self.rows[key][1].setVisible(True)

    def update_columns(self, df):
        if df is None: return
        cols = list(df.columns)
        for lst in [self.l1_list, self.l2_list, self.l3_list]:
            lst.clear()
            lst.addItems(cols)

    def on_run(self):
        # Gather Params
        def get_single(lst): return lst.selectedItems()[0].text() if lst.selectedItems() else None
        def get_multi(lst): return [i.text() for i in lst.selectedItems()]

        params = {
            "model": self.model_combo.currentText(),
            "target": get_single(self.l1_list),
            "features": get_multi(self.l2_list),
            "aux": get_single(self.l3_list), # For IV/DID/PSM
            
            # Pack all numeric params (receiver will filter)
            "k": self.spin_k.value(),
            "p": self.spin_p.value(),
            "iter": self.spin_iter.value(),
            "alpha": self.spin_alpha.value(),
            "resolution": self.spin_resolution.value(),
            "damping": self.spin_damping.value(),
            "M": self.spin_hnsw_m.value(),
            "ef": self.spin_hnsw_ef.value(),
            "samples": self.spin_hmc_samples.value(),
            "warmup": self.spin_hmc_warmup.value(),
            "caliper": self.spin_caliper.value()
        }
        self.run_requested.emit(params)
