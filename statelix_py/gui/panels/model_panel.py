from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout, 
    QPushButton, QFormLayout, QListWidget, QSpinBox, QDoubleSpinBox, QGroupBox,
    QScrollArea, QLineEdit, QCompleter, QFrame
)
from PySide6.QtCore import Signal, Qt, QSortFilterProxyModel, QStringListModel

from statelix_py.gui.i18n import t

class ModelPanel(QWidget):
    run_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        self._all_models = [
            # --- Linear Models ---
            "📈 Linear: OLS (最小二乗法)", 
            "📈 Linear: Ridge Regression",
            "📈 Linear: Lasso Regression",
            "📈 Linear: Elastic Net",
            "📈 Linear: Mixed Effects (混合効果)",
            # --- GLM ---
            "📊 GLM: Logistic (二項)",
            "📊 GLM: Poisson (カウント)",
            "📊 GLM: Gamma (正値連続)",
            "📊 GLM: Negative Binomial (過分散)",
            "📊 GLM: Probit",
            "📊 GLM: Quantile Regression (分位点)",
            # --- Discrete Choice ---
            "🎯 Choice: Ordered Model (順序ロジット)",
            "🎯 Choice: Multinomial Logit (多項ロジット)",
            # --- Time Series ---
            "📉 TS: AR Model (自己回帰)",
            "📉 TS: VAR (ベクトル自己回帰)",
            "📉 TS: GARCH (ボラティリティ)",
            "📉 TS: Kalman Filter (状態空間)",
            "📉 TS: Change Point Detection",
            "📉 TS: DTW (動的時間伸縮)",
            "📉 TS: Cointegration (共和分)",
            # --- Panel Data ---
            "📋 Panel: Fixed Effects",
            "📋 Panel: Random Effects",
            "📋 Panel: First Difference",
            # --- Causal Inference ---
            "🔬 Causal: IV (2SLS)",
            "🔬 Causal: Diff-in-Diff",
            "🔬 Causal: PSM (傾向スコア)",
            "🔬 Causal: IPW (逆確率重み付け)",
            "🔬 Causal: Doubly Robust (AIPW)",
            "🔬 Causal: RDD (回帰不連続)",
            "🔬 Causal: GMM",
            "🔬 Causal: Synthetic Control",
            # --- SEM ---
            "🔗 SEM: Path Analysis (パス解析)",
            "🔗 SEM: Mediation Analysis (媒介分析)",
            # --- Survival ---
            "⏱️ Survival: Kaplan-Meier (生存曲線)",
            "⏱️ Survival: Log-Rank Test (検定)",
            "⏱️ Survival: Cox Proportional Hazards",
            # --- Spatial ---
            "🗺️ Spatial: Spatial Regression",
            # --- Bayesian ---
            "🎲 Bayes: MAP (最大事後)",
            "🎲 Bayes: Logistic (HMC)",
            "🎲 Bayes: VI (変分推論)",
            "🎲 Bayes: MCMC (Metropolis)",
            # --- Machine Learning ---
            "🤖 ML: K-Means Clustering",
            "🤖 ML: GBDT (勾配ブースティング)",
            "🤖 ML: Factorization Machines",
            # --- Graph Analysis ---
            "🕸️ Graph: Louvain Communities",
            "🕸️ Graph: PageRank",
            # --- Search ---
            "🔎 Search: Build HNSW Index",
            # --- Signal Processing ---
            "📡 Signal: Wavelet Transform",
            # --- Statistics & Tests ---
            "📐 Stats: ANOVA One-Way (一元配置)",
            "📐 Stats: ANOVA Two-Way (二元配置)",
            "📐 Stats: ANCOVA (共分散分析)",
            "📐 Stats: TukeyHSD (多重比較)",
            "📐 Stats: T-Test (t検定)",
            "📐 Stats: Chi-Squared Test (カイ二乗)",
            "📐 Stats: Mann-Whitney U (順位和)",
            "📐 Stats: Kruskal-Wallis (順位分散)",
            "📐 Stats: Wilcoxon Test (符号順位)",
            # --- Diagnostics ---
            "🔧 Diag: VIF (多重共線性)",
            "🔧 Diag: Durbin-Watson (自己相関)",
            "🔧 Diag: Breusch-Pagan (分散不均一)",
            "🔧 Diag: White Test (分散不均一)",
            "🔧 Diag: Cook's Distance (外れ値)",
        ]
        self.wasm_plugin_models = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Title with icon
        title = QLabel(t("panel.model"))
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #007acc;")
        layout.addWidget(title)

        # 1. Model Selection Group
        model_frame = QFrame()
        model_frame.setStyleSheet("""
            QFrame#ModelSelectionFrame {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        model_frame.setObjectName("ModelSelectionFrame")
        model_layout = QVBoxLayout(model_frame)
        
        # Search Box
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(t("search.model"))
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        model_layout.addWidget(self.search_edit)
        
        # Model Combo
        self.model_combo = QComboBox()
        self.model_combo.addItems(self._all_models)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_frame)
        
        # 2. Parameters Scroll Area
        self.param_group = QGroupBox(t("panel.result.summary")) # Hyperparameters
        self.form_layout = QFormLayout()
        
        # --- Params Widgets ---
        self.spin_k = QSpinBox(); self.spin_k.setRange(2, 50); self.spin_k.setValue(3)
        self.spin_p = QSpinBox(); self.spin_p.setRange(1, 20); self.spin_p.setValue(1)
        self.spin_iter = QSpinBox(); self.spin_iter.setRange(10, 1000); self.spin_iter.setValue(50)
        self.spin_alpha = QDoubleSpinBox(); self.spin_alpha.setRange(0.001, 1000.0); self.spin_alpha.setValue(1.0)
        self.spin_l1_ratio = QDoubleSpinBox(); self.spin_l1_ratio.setRange(0.0, 1.0); self.spin_l1_ratio.setValue(0.5)
        self.spin_tau = QDoubleSpinBox(); self.spin_tau.setRange(0.01, 0.99); self.spin_tau.setValue(0.5)
        
        # Graph Params
        self.spin_resolution = QDoubleSpinBox(); self.spin_resolution.setValue(1.0)
        self.spin_damping = QDoubleSpinBox(); self.spin_damping.setValue(0.85); self.spin_damping.setRange(0.01, 0.99)
        
        # HNSW Params
        self.spin_hnsw_m = QSpinBox(); self.spin_hnsw_m.setValue(16)
        self.spin_hnsw_ef = QSpinBox(); self.spin_hnsw_ef.setValue(200); self.spin_hnsw_ef.setRange(10, 2000)

        # HMC/MCMC Params
        self.spin_hmc_samples = QSpinBox(); self.spin_hmc_samples.setRange(100, 10000); self.spin_hmc_samples.setValue(1000)
        self.spin_hmc_warmup = QSpinBox(); self.spin_hmc_warmup.setRange(10, 5000); self.spin_hmc_warmup.setValue(200)

        # PSM/RDD Params
        self.spin_caliper = QDoubleSpinBox(); self.spin_caliper.setRange(0.01, 1.0); self.spin_caliper.setValue(0.2)
        self.spin_cutoff = QDoubleSpinBox(); self.spin_cutoff.setRange(-1e6, 1e6); self.spin_cutoff.setValue(0.0)
        self.spin_bandwidth = QDoubleSpinBox(); self.spin_bandwidth.setRange(0.01, 100.0); self.spin_bandwidth.setValue(1.0)

        # GARCH Params
        self.spin_garch_p = QSpinBox(); self.spin_garch_p.setRange(1, 5); self.spin_garch_p.setValue(1)
        self.spin_garch_q = QSpinBox(); self.spin_garch_q.setRange(1, 5); self.spin_garch_q.setValue(1)
        
        # GBDT Params
        self.spin_n_estimators = QSpinBox(); self.spin_n_estimators.setRange(10, 500); self.spin_n_estimators.setValue(100)
        self.spin_learning_rate = QDoubleSpinBox(); self.spin_learning_rate.setRange(0.01, 1.0); self.spin_learning_rate.setValue(0.1)
        self.spin_max_depth = QSpinBox(); self.spin_max_depth.setRange(1, 20); self.spin_max_depth.setValue(3)

        # Wavelet Params
        self.spin_wavelet_level = QSpinBox(); self.spin_wavelet_level.setRange(0, 10); self.spin_wavelet_level.setValue(0)

        # --- Add Rows ---
        self.rows = {}
        def add_row(key, label, widget):
            self.form_layout.addRow(label, widget)
            self.rows[key] = (self.form_layout.labelForField(widget), widget)

        add_row("k", "Clusters (k):", self.spin_k)
        add_row("p", "Order (p):", self.spin_p)
        add_row("iter", "Max Iter:", self.spin_iter)
        add_row("alpha", "Alpha (λ):", self.spin_alpha)
        add_row("l1_ratio", "L1 Ratio:", self.spin_l1_ratio)
        add_row("tau", "Quantile (τ):", self.spin_tau)
        add_row("res", "Resolution:", self.spin_resolution)
        add_row("damp", "Damping:", self.spin_damping)
        add_row("M", "HNSW M:", self.spin_hnsw_m)
        add_row("ef", "HNSW ef_const:", self.spin_hnsw_ef)
        add_row("samples", "Samples:", self.spin_hmc_samples)
        add_row("warmup", "Warmup:", self.spin_hmc_warmup)
        add_row("caliper", "Caliper (SD):", self.spin_caliper)
        add_row("cutoff", "Cutoff:", self.spin_cutoff)
        add_row("bandwidth", "Bandwidth:", self.spin_bandwidth)
        add_row("garch_p", "GARCH p:", self.spin_garch_p)
        add_row("garch_q", "GARCH q:", self.spin_garch_q)
        add_row("n_est", "N Estimators:", self.spin_n_estimators)
        add_row("lr", "Learning Rate:", self.spin_learning_rate)
        add_row("depth", "Max Depth:", self.spin_max_depth)
        add_row("level", "Wavelet Level:", self.spin_wavelet_level)
        
        self.param_group.setLayout(self.form_layout)
        scroll = QScrollArea()
        scroll.setWidget(self.param_group)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setMinimumHeight(150)
        layout.addWidget(scroll)

        # 3. Variable Selection (Improved Height)
        var_frame = QFrame()
        var_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget {
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
        """)
        var_layout = QHBoxLayout(var_frame)
        var_layout.setSpacing(10)
        
        # Target (Y) - Single Selection
        self.l1_layout = QVBoxLayout()
        self.l1_label = QLabel(t("label.target"))
        self.l1_label.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        self.l1_list = QListWidget()
        self.l1_list.setMinimumHeight(120)
        self.l1_list.setMaximumHeight(200)
        self.l1_layout.addWidget(self.l1_label)
        self.l1_layout.addWidget(self.l1_list)
        var_layout.addLayout(self.l1_layout)
        
        # Features (X) - Multi Selection
        self.l2_layout = QVBoxLayout()
        self.l2_label = QLabel(t("label.features"))
        self.l2_label.setStyleSheet("font-weight: bold; color: #81c784;")
        self.l2_list = QListWidget()
        self.l2_list.setMinimumHeight(120)
        self.l2_list.setMaximumHeight(200)
        self.l2_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.l2_layout.addWidget(self.l2_label)
        self.l2_layout.addWidget(self.l2_list)
        var_layout.addLayout(self.l2_layout)

        # Auxiliary (Z) - Optional
        self.l3_layout = QVBoxLayout()
        self.l3_label = QLabel(t("label.auxiliary"))
        self.l3_label.setStyleSheet("font-weight: bold; color: #ffb74d;")
        self.l3_list = QListWidget()
        self.l3_list.setMinimumHeight(120)
        self.l3_list.setMaximumHeight(200)
        self.l3_layout.addWidget(self.l3_label)
        self.l3_layout.addWidget(self.l3_list)
        var_layout.addLayout(self.l3_layout)
        
        layout.addWidget(var_frame)

        # 4. Actions
        action_layout = QHBoxLayout()
        self.run_btn = QPushButton(t("btn.run"))
        self.run_btn.clicked.connect(self.on_run)
        action_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton(t("btn.generate_code"))
        self.export_btn.clicked.connect(self.on_export)
        action_layout.addWidget(self.export_btn)
        
        layout.addLayout(action_layout)
        layout.addStretch()
        self.setLayout(layout)
        
        # Initial model update
        self.on_model_changed(0)

    def _on_search_text_changed(self, text):
        """Filter items in model_combo based on search text."""
        self.model_combo.blockSignals(True)
        current = self.model_combo.currentText()
        self.model_combo.clear()
        
        all_items = self._all_models + self.wasm_plugin_models
        if not text:
            self.model_combo.addItems(all_items)
        else:
            filtered = [m for m in all_items if text.lower() in m.lower()]
            if filtered:
                self.model_combo.addItems(filtered)
            else:
                self.model_combo.addItem(f"No results for '{text}'")
        
        # Restore selection
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        else:
            self.model_combo.setCurrentIndex(0)
            
        self.model_combo.blockSignals(False)
        # Ensure UI updates for new selection
        self.on_model_changed(self.model_combo.currentIndex())

        
    def on_model_changed(self, index):
        model = self.model_combo.currentText()
        
        # Skip empty or separators
        if not model or model.startswith("---"):
            return
        
        # Remove emoji prefix for matching (e.g., "📈 Linear: OLS" -> "Linear: OLS")
        model_clean = model
        if len(model) > 2 and model[1] == ' ':
            model_clean = model[2:]  # Remove emoji + space
        
        # 1. Reset Visibility
        for label, widget in self.rows.values():
            label.setVisible(False)
            widget.setVisible(False)
            
        # 2. Reset List Labels
        l1, l2, l3 = "Target (Y)", "Features (X)", "Auxiliary (Unused)"
        self.l3_list.setEnabled(False)
        self.l2_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.l1_list.setEnabled(True)
        self.l2_list.setEnabled(True)

        # 3. Configure based on Model
        # --- Linear ---
        if "OLS" in model:
            pass  # No special params
            
        elif "Ridge" in model:
            self._show("alpha")
            
        elif "Lasso" in model:
            self._show("alpha")
            
        elif "Elastic" in model:
            self._show("alpha"); self._show("l1_ratio")

        # --- GLM ---
        elif "Logistic" in model and "Bayesian" not in model:
            self._show("iter")
            
        elif "Poisson" in model:
            self._show("iter")
            
        elif "Gamma" in model:
            self._show("iter")
            
        elif "Negative Binomial" in model:
            self._show("iter")
            
        elif "Quantile" in model:
            self._show("tau"); self._show("iter")

        # --- Time Series ---
        elif "AR Model" in model:
            self._show("p")
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "VAR" in model:
            self._show("p")
            l1 = "Time Series Columns"
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "GARCH" in model:
            self._show("garch_p"); self._show("garch_q")
            l1 = "Returns (Y)"
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "Kalman" in model:
            l1 = "Observations (Y)"
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "Change Point" in model:
            l1 = "Time Series (Y)"
            l2 = "(Unused)"; self.l2_list.setEnabled(False)
            
        elif "DTW" in model:
            l1, l2 = "Series 1", "Series 2"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            
        elif "Cointegration" in model:
            l1 = "Time Series Columns"

        # --- Panel Data ---
        elif "Panel" in model:
            l1, l2, l3 = "Outcome (Y)", "Regressors (X)", "Unit/Time IDs"
            self.l3_list.setEnabled(True)

        # --- Causal ---
        elif "IV (2SLS)" in model:
            l1, l2, l3 = "Outcome (Y)", "Endogenous (X)", "Instrument (Z)"
            self.l3_list.setEnabled(True)

        elif "Diff-in-Diff" in model:
            l1, l2, l3 = "Outcome (Y)", "Treatment (D)", "Post/Time"
            self.l3_list.setEnabled(True)
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            
        elif "PSM" in model:
            self._show("caliper")
            l1, l2, l3 = "Outcome (Y)", "Covariates (X)", "Treatment (0/1)"
            self.l3_list.setEnabled(True)
            
        elif "RDD" in model:
            self._show("cutoff"); self._show("bandwidth")
            l1, l2, l3 = "Outcome (Y)", "Running Var (X)", "(Unused)"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        elif "GMM" in model:
            l1, l2, l3 = "Outcome (Y)", "Endogenous (X)", "Instruments (Z)"
            self.l3_list.setEnabled(True)

        elif "Synthetic" in model:
            l1, l2, l3 = "Outcome (Y)", "Predictors", "Unit/Time IDs"
            self.l3_list.setEnabled(True)

        # --- Survival ---
        elif "Cox" in model:
            l1, l2, l3 = "Time (T)", "Covariates (X)", "Event (0/1)"
            self.l3_list.setEnabled(True)

        # --- Bayesian ---
        elif "MAP" in model:
            self._show("iter")
            
        elif "HMC" in model or "MCMC" in model:
            self._show("samples"); self._show("warmup")
            l1, l2 = "Target (0/1)", "Features (X)"
            
        elif "VI" in model:
            self._show("iter")

        # --- ML ---
        elif "K-Means" in model:
            self._show("k"); self._show("iter")
            l1 = "(Unused)"; self.l1_list.setEnabled(False)
            
        elif "GBDT" in model:
            self._show("n_est"); self._show("lr"); self._show("depth")
            
        elif "Factorization" in model:
            self._show("k"); self._show("iter"); self._show("lr")

        # --- Graph ---
        elif "Louvain" in model:
            self._show("res")
            l1, l2 = "Source Node", "Target Node"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        elif "PageRank" in model:
            self._show("damp")
            l1, l2 = "Source Node", "Target Node"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # --- Search ---
        elif "HNSW" in model:
            self._show("M"); self._show("ef")
            l1 = "(Unused)"; self.l1_list.setEnabled(False)
            l2 = "Vector Columns"

        # --- Signal ---
        elif "Wavelet" in model:
            self._show("level")
            l1 = "Signal (Y)"
            l2 = "(Unused)"; self.l2_list.setEnabled(False)

        # --- Stats ---
        elif "ANOVA" in model:
            l1, l2 = "Data (Y)", "Group (X)"
            self.l2_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        # Apply list labels
        self.l1_label.setText(l1)
        self.l2_label.setText(l2)
        self.l3_label.setText(l3)
        
        self.l1_list.setEnabled("Unused" not in l1)
        self.l2_list.setEnabled("Unused" not in l2)
        self.l3_list.setEnabled("Unused" not in l3 and self.l3_list.isEnabled())

    def _show(self, key):
        if key in self.rows:
            self.rows[key][0].setVisible(True)
            self.rows[key][1].setVisible(True)

    def update_columns(self, df):
        if df is None: return
        cols = list(df.columns)
        for lst in [self.l1_list, self.l2_list, self.l3_list]:
            lst.clear()
            lst.addItems(cols)

    def on_run(self):
        params = self._get_params()
        self.run_requested.emit(params)

    def _get_params(self):
        def get_single(lst): return lst.selectedItems()[0].text() if lst.selectedItems() else None
        def get_multi(lst): return [i.text() for i in lst.selectedItems()]

        return {
            "model": self.model_combo.currentText(),
            "target": get_single(self.l1_list),
            "features": get_multi(self.l2_list),
            "aux": get_single(self.l3_list),
            
            # All numeric params
            "k": self.spin_k.value(),
            "p": self.spin_p.value(),
            "iter": self.spin_iter.value(),
            "alpha": self.spin_alpha.value(),
            "l1_ratio": self.spin_l1_ratio.value(),
            "tau": self.spin_tau.value(),
            "resolution": self.spin_resolution.value(),
            "damping": self.spin_damping.value(),
            "M": self.spin_hnsw_m.value(),
            "ef": self.spin_hnsw_ef.value(),
            "samples": self.spin_hmc_samples.value(),
            "warmup": self.spin_hmc_warmup.value(),
            "caliper": self.spin_caliper.value(),
            "cutoff": self.spin_cutoff.value(),
            "bandwidth": self.spin_bandwidth.value(),
            "garch_p": self.spin_garch_p.value(),
            "garch_q": self.spin_garch_q.value(),
            "n_estimators": self.spin_n_estimators.value(),
            "learning_rate": self.spin_learning_rate.value(),
            "max_depth": self.spin_max_depth.value(),
            "wavelet_level": self.spin_wavelet_level.value(),
            "is_wasm_plugin": self.model_combo.currentText().startswith("WASM:")
        }

    def on_export(self):
        """Generate Python script and copy to clipboard"""
        params = self._get_params()
        script = self.generate_script(params)
        
        from PySide6.QtWidgets import QApplication, QMessageBox
        cb = QApplication.clipboard()
        cb.setText(script)
        QMessageBox.information(self, "Export", "Python code copied to clipboard!")

    def generate_script(self, params):
        """Map params to Python code string"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        model_name = params['model']
        target = params.get('target', 'None')
        features = params.get('features', [])
        aux = params.get('aux', 'None')
        
        # Basic Template
        code = f'''# Generated by Statelix v2.3 on {timestamp}
import pandas as pd
import numpy as np
from statelix_py.core.data_manager import DataManager

# 1. Load Data (Adjust path as needed)
# df = pd.read_csv("your_data.csv")
'''
        
        code += "\n# 2. Select Variables\n"
        code += f"y_col = '{target}'\n"
        code += f"x_cols = {features}\n"
        if aux: code += f"z_col = '{aux}'\n"
        
        code += f"\n# 3. Model: {model_name}\n"
        
        # logic mapping (simplified for demo)
        if "OLS" in model_name:
            code += "from statelix_py.core import cpp_binding\n"
            code += "X = df[x_cols].values\n"
            code += "y = df[y_col].values\n"
            code += "res = cpp_binding.fit_ols_full(X, y)\n"
            code += "print(f'R2: {res.r_squared}')\n"

        elif "IV" in model_name:
             code += "from statelix.causal import StatelixIV\n"
             code += "iv = StatelixIV()\n"
             code += "iv.fit(df[x_cols].values, df[y_col].values, df[z_col].values)\n"
             code += "print(iv.result_)\n"
             
        elif "PSM" in model_name:
             code += "from statelix.causal import StatelixPSM\n"
             code += f"psm = StatelixPSM(caliper={params['caliper']})\n"
             code += "psm.fit(df[y_col].values, df[z_col].values, df[x_cols].values)\n"
             code += "print(psm.summary)\n"
             
        elif "Bayesian" in model_name:
             code += "from statelix_py.models import BayesianLogisticRegression\n"
             code += f"bayes = BayesianLogisticRegression(n_samples={params['samples']}, warmup={params['warmup']})\n"
             code += "bayes.fit(df[x_cols].values, df[y_col].values)\n"
             code += "print(bayes.summary)\n"
             
        else:
             code += f"# Code generation for '{model_name}' is not yet fully implemented.\n"
             code += "# Please refer to documentation.\n"
             
        return code

    def add_wasm_plugins(self, plugins: dict):
        """Dynamically add WASM plugins to the model list."""
        self.wasm_plugin_models.clear()
        
        for plugin_name, plugin_info in plugins.items():
            exports = plugin_info.get('exports', {})
            for func_name in exports.keys():
                display_name = f"WASM: {plugin_name}::{func_name}"
                self.wasm_plugin_models.append(display_name)
        
        # Refresh the combo box items through the filter logic
        self._on_search_text_changed(self.search_edit.text())
