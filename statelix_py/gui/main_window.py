import time
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSplitter, QMessageBox, QLabel, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from statelix_py.gui.panels.inquiry_panel import InquiryPanel
from statelix_py.gui.panels.data_panel import DataPanel
from statelix_py.gui.panels.model_panel import ModelPanel
from statelix_py.gui.panels.result_panel import ResultPanel
from statelix_py.gui.panels.plot_panel import PlotPanel
from statelix_py.gui.panels.exploratory_panel import ExploratoryPanel

# --- Worker Thread for Analysis ---
class AnalysisWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, params, df):
        super().__init__()
        self.params = params
        self.df = df

    def run(self):
        try:
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
            
            # === INQUIRY MODE ===
            if mode == 'Inquiry':
                q_id = self.params['question_id']
                y_col = self.params['y']
                x_col = self.params['x']
                z_col = self.params.get('z', '')
                
                Y = self.df[y_col].values
                X = self.df[x_col].values
                
                # 1. Drivers (Association)
                if q_id == 1:
                    # "What drives Y?" -> OLS Analysis with Narrative
                    import statsmodels.api as sm
                    X_aug = sm.add_constant(X)
                    ols = sm.OLS(Y, X_aug).fit()
                    
                    # Generate Rich Narrative
                    coef = ols.params[1] if len(ols.params) > 1 else 0
                    pval = ols.pvalues[1] if len(ols.pvalues) > 1 else 1
                    r2 = ols.rsquared
                    
                    sig_indicator = "🟢 Significant" if pval < 0.05 else "🟡 Not Significant"
                    direction = "positively" if coef > 0 else "negatively"
                    
                    narrative = f"## Driver Analysis: {x_col} → {y_col}\n\n"
                    narrative += f"**Question**: What drives **{y_col}**?\n\n"
                    narrative += "### Key Finding\n"
                    narrative += f"> For every 1-unit increase in **{x_col}**, **{y_col}** is expected to change by **{coef:+.4f}** units.\n\n"
                    
                    narrative += "### Statistical Evidence\n"
                    narrative += f"- **Coefficient**: {coef:.4f}\n"
                    narrative += f"- **P-Value**: {pval:.4f} ({sig_indicator})\n"
                    narrative += f"- **Model Fit (R²)**: {r2:.2%}\n\n"
                    
                    if pval < 0.05:
                        narrative += f"> ✅ There is strong statistical evidence that **{x_col}** {direction} affects **{y_col}**.\n"
                    else:
                        narrative += f"> ⚠️ The relationship between **{x_col}** and **{y_col}** is not statistically significant at the 5% level. Consider adding more data or exploring other variables.\n"
                    
                    # Visual: Scatter Plot for Drivers
                    result_data['viz_type'] = 'Drivers'
                    result_data['viz_data'] = {'X': X, 'Y': Y, 'x_col': x_col, 'y_col': y_col}
                    
                # 2. Causal Inference
                elif q_id == 2:
                    # Detect Method
                    result_data['viz_type'] = 'None'
                    
                    if z_col and "Instrument" in z_col: # Heuristic? No, user selected it.
                        # IV
                        Z = self.df[z_col].values
                        model = IV2SLS().fit(Y, Endog=X, Instruments=Z)
                        result_data['viz_type'] = 'IV'
                        result_data['viz_data'] = {'Z': Z, 'X': X, 'resid': Y - model.predict(X)} 
                        
                    elif "Treatment" in x_col or len(np.unique(X)) == 2:
                        # Binary Treatment -> DiD or Matching?
                        # If Z is time...
                        if z_col: # Assume Z is Time or Group
                             # Let's assume DiD structure: Y, Group(X), Time(Z)
                             Time = self.df[z_col].values
                             model = DiffInDiff().fit(Y, Group=X, Time=Time)
                             result_data['viz_type'] = 'DiD'
                             result_data['viz_data'] = {'Y': Y, 'Group': X, 'Time': Time}
                        else:
                             # Just RDD if X is continuous? Or simple difference?
                             pass
                    
                    # RDD Check: X is continuous relative to a Cutoff?
                    # Heuristic: If X has many unique values and "Cutoff" or "Run" is implied?
                    # Or just try RDD if X matches 'RunVar' pattern
                    causal_model = None  # Track model explicitly
                    if causal_model is None:
                        from statelix.causal import RDD
                        # Assume Cutoff 0 or mean? Let's guess 0 for now or user specifies.
                        # For Inquiry Mode, we might need a dedicated input for Cutoff.
                        # For now, default 0.0
                        cutoff = 0.0
                        try:
                            causal_model = RDD(cutoff=cutoff).fit(Y, RunVar=X)
                            result_data['viz_type'] = 'RDD'
                            result_data['viz_data'] = {'Y': Y, 'RunVar': X, 'Cutoff': cutoff}
                        except Exception:
                            pass

                    if causal_model is not None:
                        story = Storyteller(causal_model, feature_names=["Effect/Slope", "X", "Z", "Intercept"])
                        narrative = story.explain()
                    else:
                        narrative = "Could not automatically determine causal strategy. Please ensure inputs are correct."

                # 3. WhatIf
                elif q_id == 3:
                     # Counterfactual Simulation using OLS
                     import statsmodels.api as sm
                     import pandas as pd
                     
                     # Fit Base Model: Y ~ X + Z (if Z exists)
                     if z_col:
                         X_model = self.df[[x_col, z_col]]
                     else:
                         X_model = self.df[[x_col]]
                         
                     X_aug = sm.add_constant(X_model)
                     model_fit = sm.OLS(Y, X_aug).fit()
                     
                     # Predict Baseline
                     y_pred_base = model_fit.predict(X_aug)
                     
                     # Create Counterfactual Data
                     X_sim = X_model.copy()
                     sim_type = self.params.get('sim_type', '')
                     sim_val = self.params.get('sim_val', 0.0)
                     
                     if "Increase by %" in sim_type:
                         X_sim[x_col] = X_sim[x_col] * (1 + sim_val / 100.0)
                         change_desc = f"increased by {sim_val}%"
                     elif "Decrease by %" in sim_type:
                         X_sim[x_col] = X_sim[x_col] * (1 - sim_val / 100.0)
                         change_desc = f"decreased by {sim_val}%"
                     elif "Set to Value" in sim_type:
                         X_sim[x_col] = sim_val
                         change_desc = f"was set to {sim_val}"
                     elif "Increase by Value" in sim_type:
                         X_sim[x_col] = X_sim[x_col] + sim_val
                         change_desc = f"increased by {sim_val} (absolute)"
                     else:
                         change_desc = "changed"
                         
                     # Predict Counterfactual
                     X_sim_aug = sm.add_constant(X_sim) # Constant must match dimensions
                     # Careful with const addition if X_sim changed index/order? No, just copy.
                     # sm.add_constant adds 'const' column.
                     
                     y_pred_sim = model_fit.predict(X_sim_aug)
                     
                     # Calculate Impact
                     diff = y_pred_sim - y_pred_base
                     mean_diff = np.mean(diff)
                     total_diff = np.sum(diff)
                     
                     # Narrative Generation
                     narrative = "### What-If Simulation Results\n"
                     narrative += f"**Scenario**: If **{x_col}** is {change_desc}...\n\n"
                     narrative += f"> The outcome **{y_col}** is expected to change by **{mean_diff:+.2f}** (on average).\n"
                     narrative += f"> **Total Impact**: {total_diff:+.2f} over the entire dataset.\n\n"
                     
                     # Visual Data (Before vs After distribution)
                     result_data['viz_type'] = 'WhatIf'
                     result_data['viz_data'] = {
                         'Y_base': y_pred_base,
                         'Y_sim': y_pred_sim,
                         'Label': f"After {x_col} change"
                     }

                result_data["type"] = "inquiry"
                result_data["narrative"] = narrative
            
            # === CLASSIC MODE ===
            elif "Graph" in model:
                src_col = self.params.get('target') # L1
                dst_col = self.params.get('features')[0] if self.params.get('features') else None # L2
                
                if not src_col or not dst_col:
                    raise ValueError("Source and Target columns required.")

                from statelix_py.core.data_manager import DataManager
                dm = DataManager.instance()
                dm.log_operation(f"graph_model = StatelixGraph()\ngraph_model.fit(df['{src_col}'].astype(str).values, df['{dst_col}'].astype(str).values)")

                graph_model = StatelixGraph()
                graph_model.fit(
                    self.df[src_col].astype(str).values, 
                    self.df[dst_col].astype(str).values,
                    directed=("Louvain" not in model)
                )
                
                summary += f"Model: {model}\nNodes: {graph_model.n_nodes_}\n"

                if "Louvain" in model:
                    res_df = graph_model.louvain(resolution=self.params['resolution'])
                    summary += "Found Communities.\n"
                    result_data["table"] = res_df.head(100)
                    dm.log_operation(f"res_df = graph_model.louvain(resolution={self.params['resolution']})")
                elif "PageRank" in model:
                    res_df = graph_model.pagerank(damping=self.params['damping'])
                    result_data["table"] = res_df.head(100)
                    dm.log_operation(f"res_df = graph_model.pagerank(damping={self.params['damping']})")

            # --- Causal Inference (Refactored) ---
            elif "Causal" in model:
                y_col = self.params.get('target')
                from statelix_py.core.data_manager import DataManager
                dm = DataManager.instance()
                
                if "PSM" in model:
                   x_cols = self.params.get('features')
                   t_col = self.params.get('aux')
                   y = self.df[y_col].values
                   T = self.df[t_col].values
                   X = self.df[x_cols].values
                   psm = StatelixPSM(caliper=self.params['caliper'])
                   psm.fit(y, T, X)
                   # ... (rest of PSM summary logic remains same as before)
                   summary += f"Model: PSM\nATT: {psm.summary['ATT']:.4f}\n"
                   dm.log_operation(f"psm = StatelixPSM(caliper={self.params['caliper']})\npsm.fit(df['{y_col}'].values, df['{t_col}'].values, df[{repr(x_cols)}].values)")

                elif "IV" in model:
                     x_endog_col = self.params.get('features')
                     z_col = self.params.get('aux')
                     iv = StatelixIV()
                     iv.fit(self.df[x_endog_col].values, self.df[y_col].values, self.df[z_col].values)
                     summary += f"Model: IV (2SLS)\nFirst Stage F: {iv.result_.first_stage_f:.4f}\n"
                     dm.log_operation(f"iv = StatelixIV()\niv.fit(df[{repr(x_endog_col)}].values, df['{y_col}'].values, df['{z_col}'].values)")

            # --- Fallback ---
            else:
                 if "OLS" in model:
                    y_col = self.params.get('target')
                    x_cols = self.params.get('features')
                    res = cpp_binding.fit_ols_full(self.df[x_cols].values, self.df[y_col].values)
                    summary += f"Model: OLS\nR2: {res.r_squared:.4f}\n"
                    from statelix_py.core.data_manager import DataManager
                    DataManager.instance().log_operation(f"res = cpp_binding.fit_ols_full(df[{repr(x_cols)}].values, df['{y_col}'].values)")
                 else:
                    summary += "Legacy model selected.\n"

            if mode != 'Inquiry':
                elapsed = time.time() - start_time
                summary += f"\nTime: {elapsed:.3f}s"
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
        self.resize(1100, 700) # Adjusted for smaller screens
        self.worker = None # Keep reference
        
        # Enable Advanced Docking
        from PySide6.QtWidgets import QMainWindow
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks | 
            QMainWindow.DockOption.AllowTabbedDocks | 
            QMainWindow.DockOption.AnimatedDocks
        )
        
        
        
        # --- Apply Theme ---
        # --- Apply Theme ---
        from statelix_py.gui.styles import StatelixTheme
        self.setStyleSheet(StatelixTheme.DARK_STYLESHEET)
        
        self.init_ui()

    def show_toast(self, message, duration=3000):
        from statelix_py.gui.components.toast import Toast
        # Create a new toast
        t = Toast(self, message, duration)
        t.show_toast()

    def init_ui(self):
        from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
        from PySide6.QtCore import Qt
        # Removed FlexiblePanel due to usability issues - Reverting to Standard Splitter

        # --- Central Widget: Mode Tabs ---
        self.mode_tabs = QTabWidget()
        self.setCentralWidget(self.mode_tabs)
        # self.mode_tabs.currentChanged.connect(self.on_mode_changed) # Not needed for Splitter layout usually

        # --- TAB 1: Expert Work Area ---
        expert_widget = QWidget()
        expert_layout = QVBoxLayout(expert_widget)
        expert_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main Layout (Fixed Horizontal)
        # User requested fixed boundaries to prevent issues
        main_layout = QHBoxLayout() # No splitter handles
        
        # 1. Left Column (Vertical Layout: Data / Inspector)
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        from statelix_py.gui.panels.data_panel import DataPanel
        from statelix_py.gui.panels.variable_inspector import VariableInspector
        
        self.data_panel = DataPanel()
        self.inspector_panel = VariableInspector()
        
        left_layout.addWidget(self.data_panel, stretch=2)
        left_layout.addWidget(self.inspector_panel, stretch=1)
        
        # 2. Middle Column (Model)
        from statelix_py.gui.panels.model_panel import ModelPanel
        self.model_panel = ModelPanel()
        
        # 3. Right Column (Results Tabs)
        from statelix_py.gui.panels.result_panel import ResultPanel
        from statelix_py.gui.panels.plot_panel import PlotPanel
        from statelix_py.gui.panels.exploratory_panel import ExploratoryPanel
        
        self.expert_center = QTabWidget() # Internal tabs for output
        self.result_panel = ResultPanel()
        self.plot_panel = PlotPanel()
        self.exploratory_panel = ExploratoryPanel()
        
        self.expert_center.addTab(self.result_panel, "Result")
        self.expert_center.addTab(self.plot_panel, "Plots")
        self.expert_center.addTab(self.exploratory_panel, "EDA")
        
        # Add to Main Layout (Fixed Ratios)
        # Left (Data/Insp) : Middle (Model) : Right (Output)
        main_layout.addWidget(left_column, stretch=20)   # Approx 20%
        main_layout.addWidget(self.model_panel, stretch=25) # Approx 25%
        main_layout.addWidget(self.expert_center, stretch=55) # Approx 55%
        
        expert_layout.addLayout(main_layout)
        
        # 4. Code / Log Console (Bottom)
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(5, 5, 5, 5)
        
        self.code_console = QTextEdit()
        self.code_console.setReadOnly(True)
        self.code_console.setPlaceholderText("Python code for reproducibility will appear here...")
        self.code_console.setMaximumHeight(120)
        self.code_console.setStyleSheet("font-family: monospace; background-color: #1e1e1e; color: #d4d4d4;")
        
        console_layout.addWidget(QLabel("Code Console (Reproducibility Script):"))
        console_layout.addWidget(self.code_console)
        
        expert_layout.addWidget(console_widget)
        
        self.mode_tabs.addTab(expert_widget, "🛠️ Expert Mode")
        
        # --- TAB 2: Inquiry Panel ---
        from statelix_py.gui.panels.inquiry_panel import InquiryPanel
        self.inquiry_panel = InquiryPanel()
        self.mode_tabs.addTab(self.inquiry_panel, "🎓 Inquiry Mode")
        
        # --- Connections ---
        self.data_panel.data_loaded.connect(self.model_panel.update_columns)
        self.data_panel.data_loaded.connect(self.inspector_panel.set_data)
        self.data_panel.data_loaded.connect(self.exploratory_panel.on_data_loaded)
        self.data_panel.data_loaded.connect(self.inquiry_panel.update_columns)
        
        self.inspector_panel.data_changed.connect(self.on_data_modified)
        
        self.model_panel.run_requested.connect(self.run_analysis)
        self.inquiry_panel.run_inquiry.connect(self.run_analysis)

        # --- Plugins (Load Here) ---
        from statelix_py.plugins.loader import WasmPluginLoader
        self.plugin_loader = WasmPluginLoader()
        self.loaded_plugins = self.plugin_loader.scan_and_load()

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        
        open_action = file_menu.addAction("Open Data File...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.data_panel.load_data)
        
        file_menu.addSeparator()
        
        save_project_action = file_menu.addAction("Save Project...")
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self.save_project)
        
        open_project_action = file_menu.addAction("Open Project...")
        open_project_action.setShortcut("Ctrl+Shift+O")
        open_project_action.triggered.connect(self.open_project)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Plugins Menu
        plugin_menu = menu.addMenu("Plugins")
        if not self.loaded_plugins:
             plugin_menu.addAction("No plugins found").setEnabled(False)
        else:
             for name, info in self.loaded_plugins.items():
                 plugin_menu.addAction(f"Loaded: {name}").setEnabled(False)
        
        plugin_menu.addSeparator()
        reload_action = plugin_menu.addAction("Reload Plugins")
        reload_action.triggered.connect(self.reload_plugins)
        
        # --- Connect WASM plugins to model panel ---
        if self.loaded_plugins:
            self.model_panel.add_wasm_plugins(self.loaded_plugins)
        
        self.statusBar().showMessage("Ready")

    def run_analysis(self, params):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is None or dm.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded. Use DataPanel in Expert Mode to load data first.")
            return

        self.statusBar().showMessage("Running Analysis...")
        
        # Start Worker
        self.worker = AnalysisWorker(params, dm.df)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    @Slot(dict)
    def on_analysis_finished(self, result):
        self.statusBar().showMessage("Analysis Completed.")
        
        if result.get("type") == "inquiry":
             # Inquiry Output
             self.inquiry_panel.set_narrative(result.get("narrative", ""))
             
             # Viz
             viz_type = result.get('viz_type')
             viz_data = result.get('viz_data')
             
             if viz_type == 'Drivers':
                 def plot_drivers(ax):
                     import numpy as np
                     X_d = viz_data['X']
                     Y_d = viz_data['Y']
                     x_label = viz_data['x_col']
                     y_label = viz_data['y_col']
                     
                     ax.scatter(X_d, Y_d, alpha=0.5, color='#6C5CE7', label='Data')
                     
                     # Fit Line
                     z = np.polyfit(X_d, Y_d, 1)
                     p = np.poly1d(z)
                     x_line = np.linspace(X_d.min(), X_d.max(), 100)
                     ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
                     
                     ax.set_xlabel(x_label)
                     ax.set_ylabel(y_label)
                     ax.set_title(f"Driver: {x_label} vs {y_label}")
                     ax.legend()
                     
                 self.inquiry_panel.plot_assumption(plot_drivers)

             elif viz_type == 'IV':
                 def plot_iv(ax):
                     # Scatter First Stage
                     ax.scatter(viz_data['Z'], viz_data['X'], alpha=0.5)
                     ax.set_xlabel("Instrument (Z)")
                     ax.set_ylabel("Endogenous (X)")
                     ax.set_title("First Stage Strength")
                 self.inquiry_panel.plot_assumption(plot_iv)
                 
             elif viz_type == 'DiD':
                 def plot_did(ax):
                     # Viz Data: Y, Group, Time
                     Y = viz_data['Y']
                     G = viz_data['Group']
                     T = viz_data['Time']
                     
                     df_viz = pd.DataFrame({'Y': Y, 'Group': G, 'Time': T})
                     means = df_viz.groupby(['Group', 'Time'])['Y'].mean().unstack()
                     
                     # Plot Control
                     if 0 in means.index:
                         ax.plot(means.columns, means.loc[0], 'o--', label='Control', color='gray')
                     # Plot Treated
                     if 1 in means.index:
                         ax.plot(means.columns, means.loc[1], 'o-', label='Treated', color='red')
                         
                     ax.set_xticks(means.columns)
                     ax.set_xlabel("Time Period")
                     ax.set_ylabel("Average Outcome (Y)")
                     ax.set_title("Parallel Trends Check")
                     ax.legend()
                     ax.grid(True, linestyle=':')
                     
                 self.inquiry_panel.plot_assumption(plot_did)

             elif viz_type == 'RDD':
                 def plot_rdd(ax):
                     X = viz_data['RunVar']
                     Y = viz_data['Y']
                     c = viz_data['Cutoff']
                     
                     # Scatter
                     ax.scatter(X, Y, alpha=0.3, s=10, color='gray')
                     ax.axvline(c, color='black', linestyle='--', label=f'Cutoff ({c})')
                     
                     # Trend Lines (Left/Right)
                     mask_left = X < c
                     mask_right = X >= c
                     
                     if np.sum(mask_left) > 1:
                         z = np.polyfit(X[mask_left], Y[mask_left], 1)
                         p = np.poly1d(z)
                         range_l = np.linspace(X.min(), c, 100)
                         ax.plot(range_l, p(range_l), 'b-', linewidth=2, label='Left Trend')
                         
                     if np.sum(mask_right) > 1:
                         z = np.polyfit(X[mask_right], Y[mask_right], 1)
                         p = np.poly1d(z)
                         range_r = np.linspace(c, X.max(), 100)
                         ax.plot(range_r, p(range_r), 'r-', linewidth=2, label='Right Trend')
                         
                     ax.set_xlabel("Running Variable")
                     ax.set_ylabel("Outcome")
                     ax.set_title("Regression Discontinuity Check")
                     ax.legend()
                     
                 self.inquiry_panel.plot_assumption(plot_rdd)

             elif viz_type == 'WhatIf':
                 def plot_whatif(ax):
                     Y_base = viz_data['Y_base']
                     Y_sim = viz_data['Y_sim']
                     label = viz_data['Label']
                     
                     import numpy as np
                     
                     # Comparison Plot (Density or Hist)
                     # Plot Base
                     ax.hist(Y_base, bins=20, alpha=0.5, label='Original', color='gray')
                     # Plot Sim
                     ax.hist(Y_sim, bins=20, alpha=0.5, label='Simulated', color='#6C5CE7')
                     
                     # Add Mean Lines
                     ax.axvline(np.mean(Y_base), color='black', linestyle='--', label='Orig Mean')
                     ax.axvline(np.mean(Y_sim), color='#6C5CE7', linestyle='--', label='Sim Mean')
                     
                     ax.set_xlabel("Predicted Outcome (Y)")
                     ax.set_ylabel("Frequency")
                     ax.set_title(f"Simulation Impact: {label}")
                     ax.legend()
                     
                 self.inquiry_panel.plot_assumption(plot_whatif)


        else:
            # Classic Output
            self.result_panel.display_result(result)
            self.on_data_modified() # Refresh code console with new model logs
            if "trace" in result:
                self.plot_panel.plot_hmc_trace(result['trace'])
                self.expert_center.setCurrentWidget(self.plot_panel)
            else:
                self.expert_center.setCurrentWidget(self.result_panel)

    @Slot(str)
    def on_analysis_error(self, msg):
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)

    def reload_plugins(self):
        self.loaded_plugins = self.plugin_loader.scan_and_load()
        self.model_panel.add_wasm_plugins(self.loaded_plugins)
        QMessageBox.information(self, "Plugins", f"Reloaded. Found {len(self.loaded_plugins)} plugins.")

    def save_project(self):
        from statelix_py.core.project_manager import ProjectManager
        from statelix_py.core.data_manager import DataManager
        from PySide6.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Statelix Project (*.stlx)")
        if path:
            pm = ProjectManager()
            dm = DataManager.instance()
            if pm.save_project(path, dm):
                QMessageBox.information(self, "Success", f"Project saved to {path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save project.")

    def open_project(self):
        from statelix_py.core.project_manager import ProjectManager
        from statelix_py.core.data_manager import DataManager
        from PySide6.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Statelix Project (*.stlx)")
        if path:
            pm = ProjectManager()
            dm = DataManager.instance()
            if pm.load_project(path, dm):
                # Refresh all panels
                if dm.df is not None:
                    self.data_panel.update_display(dm.filename, dm.df)
                    self.model_panel.update_columns(dm.df)
                    self.inspector_panel.set_data(dm.df)
                    self.exploratory_panel.on_data_loaded(dm.df)
                    self.inquiry_panel.update_columns(dm.df)
                    self.on_data_modified()
                QMessageBox.information(self, "Success", f"Project loaded from {path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load project.")

    @Slot()
    def on_data_modified(self):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        df = dm.df
        if df is None: return
        
        # Refresh all panels
        self.data_panel.update_display(dm.filename, df)
        self.model_panel.update_columns(df)
        self.exploratory_panel.on_data_loaded(df)
        self.inquiry_panel.update_columns(df)
        
        # Refresh Code Console
        history = dm.get_history()
        if history:
            code = "import pandas as pd\nimport numpy as np\n\n# Load Data\n"
            code += f"df = pd.read_csv('{dm.filename}')\n"
            code += "\n".join(history)
            self.code_console.setText(code)
        
        self.statusBar().showMessage("Data modified and refreshed.")

    def on_mode_changed(self, index):
        # 0 = Expert, 1 = Inquiry
        # Hide Docks if Inquiry
        is_expert = (index == 0)
        
        # Check if docks exist before toggling (safe-guard)
        if hasattr(self, 'dock_data'): self.dock_data.setVisible(is_expert)
        if hasattr(self, 'dock_model'): self.dock_model.setVisible(is_expert)
        if hasattr(self, 'dock_inspector'): self.dock_inspector.setVisible(is_expert)
