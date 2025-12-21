import time
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSplitter, QMessageBox, QLabel, QTextEdit, QHBoxLayout, QProgressBar, QPushButton, QFrame, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QKeySequence

from statelix_py.gui.panels.inquiry_panel import InquiryPanel
from statelix_py.gui.panels.data_panel import DataPanel
from statelix_py.gui.panels.model_panel import ModelPanel
from statelix_py.gui.panels.result_panel import ResultPanel
from statelix_py.gui.panels.plot_panel import PlotPanel
from statelix_py.gui.panels.exploratory_panel import ExploratoryPanel
from statelix_py.gui.panels.variable_inspector import VariableInspector
from statelix_py.gui.styles import StatelixTheme
from statelix_py.gui.i18n import t

# --- Worker Thread for Analysis ---
class AnalysisWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int, str) # percent (0-100), message

    def __init__(self, params, df):
        super().__init__()
        self.params = params
        self.df = df
        self._is_aborted = False

    def stop(self):
        self._is_aborted = True

    def run(self):
        try:
            self.progress.emit(5, t("progress.initializing"))
            from statelix_py.models import (
                StatelixGraph, StatelixIV, StatelixDID, StatelixPSM,
                BayesianLogisticRegression, StatelixHNSW
            )
            from statelix.inquiry import Storyteller, WhatIf, CausalAdapter
            from statelix.causal import IV2SLS, DiffInDiff
            from statelix_py.core import cpp_binding 
            
            result_data = {}
            summary = ""
            start_time = time.time()
            
            model = self.params.get('model', '')
            mode = self.params.get('mode', 'Classic')
            
            if self._is_aborted: return

            # === INQUIRY MODE ===
            if mode == 'Inquiry':
                self.progress.emit(20, t("progress.preparing_inquiry"))
                q_id = self.params['question_id']
                y_col = self.params['y']
                x_col = self.params['x']
                z_col = self.params.get('z', '')
                
                Y = self.df[y_col].values
                X = self.df[x_col].values
                
                # 1. Drivers (Association)
                if q_id == 1:
                    self.progress.emit(40, t("progress.calculating_drivers"))
                    import statsmodels.api as sm
                    X_aug = sm.add_constant(X)
                    ols = sm.OLS(Y, X_aug).fit()
                    story = Storyteller(ols, feature_names=["Intercept", x_col])
                    narrative = story.explain()
                    result_data['viz_type'] = 'Drivers'
                    result_data['viz_data'] = {'X': X.tolist(), 'Y': Y.tolist(), 'x_col': x_col, 'y_col': y_col}
                    
                # 2. Causal Inference
                elif q_id == 2:
                    result_data['viz_type'] = 'None'
                    causal_model = None
                    if z_col and "Instrument" in z_col:
                        Z = self.df[z_col].values
                        causal_model = IV2SLS().fit(Y, Endog=X, Instruments=Z)
                        result_data['viz_type'] = 'IV'
                        result_data['viz_data'] = {'Z': Z.tolist(), 'X': X.tolist()}
                    elif "Treatment" in x_col or len(np.unique(X)) == 2:
                        if z_col:
                             Time = self.df[z_col].values
                             causal_model = DiffInDiff().fit(Y, Group=X, Time=Time)
                             result_data['viz_type'] = 'DiD'
                             result_data['viz_data'] = {'Y': Y.tolist(), 'Group': X.tolist(), 'Time': Time.tolist()}
                    
                    if causal_model is None:
                        from statelix.causal import RDD
                        cutoff = 0.0
                        try:
                            causal_model = RDD(cutoff=cutoff).fit(Y, RunVar=X)
                            result_data['viz_type'] = 'RDD'
                            result_data['viz_data'] = {'Y': Y.tolist(), 'RunVar': X.tolist(), 'Cutoff': cutoff}
                        except: pass

                    if causal_model is not None:
                        story = Storyteller(causal_model, feature_names=["Effect", x_col, "Z", "Intercept"])
                        narrative = story.explain()
                    else:
                        narrative = "Could not automatically determine causal strategy."

                # 3. WhatIf
                elif q_id == 3:
                     import statsmodels.api as sm
                     X_model = self.df[[x_col]]
                     X_aug = sm.add_constant(X_model)
                     model_fit = sm.OLS(Y, X_aug).fit()
                     y_pred_base = model_fit.predict(X_aug)
                     
                     X_sim = X_model.copy()
                     sim_val = self.params.get('sim_val', 0.0)
                     X_sim[x_col] = X_sim[x_col] + sim_val
                     X_sim_aug = sm.add_constant(X_sim)
                     y_pred_sim = model_fit.predict(X_sim_aug)
                     
                     mean_diff = np.mean(y_pred_sim - y_pred_base)
                     narrative = f"### What-If Simulation\nIf **{x_col}** increased by {sim_val}, **{y_col}** would change by **{mean_diff:+.2f}** on average."
                     result_data['viz_type'] = 'WhatIf'
                     result_data['viz_data'] = {'Y_base': y_pred_base.tolist(), 'Y_sim': y_pred_sim.tolist(), 'Label': f"Impact of +{sim_val} {x_col}"}

                result_data["type"] = "inquiry"
                result_data["narrative"] = narrative
            
            # === CLASSIC MODE ===
            elif "Graph" in model:
                src_col = self.params.get('target')
                dst_col = self.params.get('features')[0] if self.params.get('features') else None
                graph_model = StatelixGraph()
                graph_model.fit(self.df[src_col].astype(str).values, self.df[dst_col].astype(str).values)
                if "Louvain" in model:
                    res_df = graph_model.louvain(resolution=self.params['resolution'])
                    result_data["table"] = res_df.head(100)
                summary += f"Nodes: {graph_model.n_nodes_}\n"

            elif "Causal" in model:
                y_col = self.params.get('target')
                if "PSM" in model:
                   self.progress.emit(50, t("progress.running_psm"))
                   psm = StatelixPSM(caliper=self.params['caliper'])
                   psm.fit(self.df[y_col].values, self.df[self.params.get('aux')].values, self.df[self.params.get('features')].values)
                   summary += f"ATT: {psm.att:.4f}\n"
                elif "IV" in model:
                     self.progress.emit(50, t("progress.running_iv"))
                     iv = StatelixIV()
                     iv.fit(self.df[y_col].values, self.df[self.params.get('features')].values, self.df[self.params.get('aux')].values)
                     summary += f"Coef: {iv.coef_}\n"
                elif "DID" in model:
                     self.progress.emit(50, t("progress.running_did"))
                     did = StatelixDID()
                     # DID.fit(y, treated, post)
                     did.fit(self.df[y_col].values, self.df[self.params.get('features')].values, self.df[self.params.get('aux')].values)
                     summary += f"ATT: {did.att:.4f}\n"
            else:
                 if "OLS" in model:
                    res = cpp_binding.fit_ols_full(self.df[self.params.get('features')].values, self.df[self.params.get('target')].values)
                    summary += f"R2: {res.r_squared:.4f}\n"
            
            if mode != 'Inquiry':
                result_data["summary"] = summary
            self.finished.emit(result_data)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statelix v2.3 - Explanatory Intelligence")
        self.resize(1100, 700)
        self.worker = None
        self._undo_stack = []
        self._redo_stack = []
        self.setAcceptDrops(True)
        
        from statelix_py.gui.styles import StatelixTheme
        self.setStyleSheet(StatelixTheme.get_stylesheet())
        
        self.init_ui()
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        from PySide6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self._redo)
        QShortcut(QKeySequence("Ctrl+T"), self, self._toggle_theme)

    def _undo(self):
        if not self._undo_stack: return
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is not None: self._redo_stack.append(dm.df.copy())
        dm.df = self._undo_stack.pop()
        self._refresh_all_panels()
        self.show_toast("Undo")

    def _redo(self):
        if not self._redo_stack: return
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is not None: self._undo_stack.append(dm.df.copy())
        dm.df = self._redo_stack.pop()
        self._refresh_all_panels()
        self.show_toast("Redo")

    def on_data_modified(self):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is not None:
             self._undo_stack.append(dm.df.copy())
             if len(self._undo_stack) > 20: self._undo_stack.pop(0)
             self._redo_stack.clear()
        self._refresh_all_panels()

    def _refresh_all_panels(self):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is None: return
        self.data_panel.update_display(dm.filename, dm.df)
        self.model_panel.update_columns(dm.df)
        self.exploratory_panel.on_data_loaded(dm.df)
        self.inquiry_panel.update_columns(dm.df)
        self.inspector_panel.set_data(dm.df)

    def _toggle_theme(self):
        from statelix_py.gui.styles import StatelixTheme
        StatelixTheme.toggle_theme()
        self.setStyleSheet(StatelixTheme.get_stylesheet())

    def show_toast(self, message):
        from statelix_py.gui.components.toast import Toast
        Toast(self, message).show_toast()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.mode_tabs = QTabWidget()
        layout.addWidget(self.mode_tabs)
        
        # Expert Mode
        expert_w = QWidget()
        expert_layout = QVBoxLayout(expert_w)
        expert_layout.setContentsMargins(5, 5, 5, 5)
        
        # Splitter for 3-column layout
        expert_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Left column (Vertical: DataPanel + VariableInspector)
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.data_panel = DataPanel()
        self.inspector_panel = VariableInspector()
        left_layout.addWidget(self.data_panel, 2)
        left_layout.addWidget(self.inspector_panel, 1)
        
        # 2. Middle Column: ModelPanel
        self.model_panel = ModelPanel()
        
        # 3. Right Column: Results & EDA
        self.expert_center = QTabWidget()
        self.result_panel = ResultPanel()
        self.exploratory_panel = ExploratoryPanel()
        
        self.expert_center.addTab(self.result_panel, t("tab.result"))
        self.expert_center.addTab(self.exploratory_panel, t("tab.eda"))
        
        # Add to splitter with initial sizes
        expert_splitter.addWidget(left_col)
        expert_splitter.addWidget(self.model_panel)
        expert_splitter.addWidget(self.expert_center)
        expert_splitter.setStretchFactor(0, 2)
        expert_splitter.setStretchFactor(1, 2)
        expert_splitter.setStretchFactor(2, 5)
        
        expert_layout.addWidget(expert_splitter)
        
        self.mode_tabs.addTab(expert_w, t("tab.expert_mode"))
        
        # Inquiry Mode
        self.inquiry_panel = InquiryPanel()
        self.mode_tabs.addTab(self.inquiry_panel, t("tab.inquiry_mode"))
        
        # Status Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.btn_cancel_analysis = QPushButton(t("btn.cancel"))
        self.btn_cancel_analysis.setVisible(False)
        self.btn_cancel_analysis.clicked.connect(self.cancel_analysis)
        
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.btn_cancel_analysis)
        self.statusBar().showMessage(t("status.ready"))

        # Connections
        self.model_panel.run_requested.connect(self.run_analysis)
        self.inquiry_panel.run_inquiry.connect(self.run_analysis)
        self.inspector_panel.data_changed.connect(self.on_data_modified)
        self.data_panel.data_loaded.connect(self.on_data_modified)
        self.data_panel.data_modified.connect(self.on_data_modified)

    def run_analysis(self, params):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is None: return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.btn_cancel_analysis.setVisible(True)
        self.worker = AnalysisWorker(params, dm.df)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.progress.connect(self._on_analysis_progress)
        self.worker.start()

    def cancel_analysis(self):
        if self.worker: self.worker.stop()
        from statelix_py.gui.i18n import t
        self.statusBar().showMessage(t("status.cancelling"))

    def _on_analysis_progress(self, percent, message):
        """Handle real-time progress updates."""
        if percent >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percent)
        self.statusBar().showMessage(message)

    @Slot(dict)
    def on_analysis_finished(self, result):
        from statelix_py.gui.i18n import t
        self.progress_bar.setVisible(False)
        self.btn_cancel_analysis.setVisible(False)
        self.statusBar().showMessage(t("status.complete"))
        
        if result.get("type") == "inquiry":
            self.inquiry_panel.set_narrative(result.get("narrative", ""))
            viz_type = result.get("viz_type")
            viz_data = result.get("viz_data")
            if not viz_data: return

            if viz_type == "Drivers":
                def plot_drv(ax):
                    ax.scatter(viz_data['X'], viz_data['Y'], alpha=0.5, color='#007acc')
                    ax.set_xlabel(viz_data.get('x_col', 'X'))
                    ax.set_ylabel(viz_data.get('y_col', 'Y'))
                    ax.set_title("Association View")
                self.inquiry_panel.plot_assumption(plot_drv)
            
            elif viz_type == "WhatIf":
                def plot_wi(ax):
                    ax.hist(viz_data['Y_base'], alpha=0.5, label="Base", color='gray')
                    ax.hist(viz_data['Y_sim'], alpha=0.5, label="Simulated", color='#6c5ce7')
                    ax.legend()
                    ax.set_title(viz_data.get('Label', 'What-If Simulation'))
                self.inquiry_panel.plot_assumption(plot_wi)
                
            elif viz_type in ["IV", "DiD", "RDD"]:
                # Simple placeholder for causal viz?
                def plot_causal(ax):
                    ax.text(0.5, 0.5, f"{viz_type} Effect: {result.get('effect', 'N/A')}", ha='center')
                self.inquiry_panel.plot_assumption(plot_causal)
        else:
            self.result_panel.display_result(result)

    @Slot(str)
    def on_analysis_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_cancel_analysis.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

    def save_project(self): pass
    def open_project(self): pass
    def reload_plugins(self): pass
