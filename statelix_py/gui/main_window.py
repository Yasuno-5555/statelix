from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSplitter, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from statelix_py.gui.panels.inquiry_panel import InquiryPanel

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
                    # Simple OLS for now
                    # We use cpp_binding.fit_ols_full(X, y)
                    # But Storyteller expects an object.
                    # Use a mock or Python wrapper? 
                    # Let's use pure python Causal model as a placeholder or proper OLS if available.
                    # Ideally: `statelix.linear_model.OLS` but C++ is disabled.
                    # Use Mock/Simple Python OLS for Inquiry?
                    # Or reuse IV2SLS with no instrument? no.
                    # Let's use CausalAdapter with a mock for now or implement PythonOLS.
                    # Or simpler: Just do Correlation analysis?
                    # "What drives Y?" -> usually Lasso or OLS.
                    
                    # Implementation: Use Numpy OLS
                    import statsmodels.api as sm
                    X_aug = sm.add_constant(X)
                    ols = sm.OLS(Y, X_aug).fit()
                    
                    # Adapt to Statelix Storyteller
                    # We need an adapter for statsmodels or map it manually.
                    # Storyteller supports: .coef_, .aic, etc.
                    # statsmodels result has params, aic.
                    
                    # Wrapper class to look like Statelix model
                    class SMWrapper:
                        def __init__(self, res):
                            self.coef_ = res.params
                            self.aic = res.aic
                            self.r2 = res.rsquared
                            
                    story = Storyteller(SMWrapper(ols), feature_names=["Intercept", x_col])
                    narrative = story.explain()
                    
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
                    if 'model' not in locals():
                        from statelix.causal import RDD
                        # Assume Cutoff 0 or mean? Let's guess 0 for now or user specifies.
                        # For Inquiry Mode, we might need a dedicated input for Cutoff.
                        # For now, default 0.0
                        cutoff = 0.0
                        try:
                            model = RDD(cutoff=cutoff).fit(Y, RunVar=X)
                            result_data['viz_type'] = 'RDD'
                            result_data['viz_data'] = {'Y': Y, 'RunVar': X, 'Cutoff': cutoff}
                        except:
                            pass

                    if 'model' in locals():
                        story = Storyteller(model, feature_names=["Effect/Slope", "X", "Z", "Intercept"])
                        narrative = story.explain()
                    else:
                        narrative = "Could not automatically determine causal strategy. Please ensure inputs are correct."

                # 3. WhatIf
                elif q_id == 3:
                     # Counterfactual
                     # Needs a fitted model.
                     # Fit OLS/IV then simulate.
                     pass
                     narrative = "What-If Simulation Engine ... (Coming Soon)"

                result_data["type"] = "inquiry"
                result_data["narrative"] = narrative
            
            # === CLASSIC MODE ===
            elif "Graph" in model:
                src_col = self.params.get('target') # L1
                dst_col = self.params.get('features')[0] if self.params.get('features') else None # L2
                
                if not src_col or not dst_col:
                    raise ValueError("Source and Target columns required.")

                # SDK handles ID mapping automatically now
                graph_model = StatelixGraph()
                graph_model.fit(
                    self.df[src_col].astype(str).values, 
                    self.df[dst_col].astype(str).values,
                    directed=("Louvain" not in model) # Louvain usually undirected
                )
                
                summary += f"Model: {model}\nNodes: {graph_model.n_nodes_}\n"

                if "Louvain" in model:
                    res_df = graph_model.louvain(resolution=self.params['resolution'])
                    summary += f"Found Communities.\n"
                    # Add simple stats if we want, or just rely on table
                    result_data["table"] = res_df.head(100)

                elif "PageRank" in model:
                    res_df = graph_model.pagerank(damping=self.params['damping'])
                    result_data["table"] = res_df.head(100)

            # --- Causal Inference (Refactored) ---
            elif "Causal" in model:
                y_col = self.params.get('target')  # Outcome
                
                if "PSM" in model:
                   # PSM Map: L1=Outcome, L2=Covariates, L3=Treatment(0/1)
                   x_cols = self.params.get('features') # Covariates
                   t_col = self.params.get('aux')       # Treatment
                   
                   if not x_cols or not t_col: raise ValueError("Outcome, Covariates, and Treatment required.")
                   
                   y = self.df[y_col].values
                   T = self.df[t_col].values
                   X = self.df[x_cols].values
                   
                   psm = StatelixPSM(caliper=self.params['caliper'])
                   psm.fit(y, T, X)
                   
                   s = psm.summary
                   summary += f"Model: PSM (Propensity Score Matching)\n"
                   summary += f"ATT: {s['ATT']:.4f} (SE: {s['SE']:.4f})\n"
                   summary += f"N_Treated: {s['n_treated']}, N_Matched: {s['n_matched']}\n"
                   summary += f"Unmatched Ratio: {s['unmatched_ratio']:.2%}\n"
                   summary += "-"*30 + "\n"
                   summary += "Score Summary:\n"
                   summary += f"  Treated Mean: {s['score_summary']['treated_mean']:.3f}\n"
                   summary += f"  Control Mean: {s['score_summary']['control_mean']:.3f}\n"
                   summary += f"  Overlap SD:   {s['score_summary']['overlap_std']:.3f}\n"

                elif "IV" in model:
                     # L1=Y, L2=X_endog, L3=Instruments
                     x_endog_col = self.params.get('features') # List
                     z_col = self.params.get('aux')
                     
                     iv = StatelixIV()
                     iv.fit(
                         self.df[x_endog_col].values,
                         self.df[y_col].values,
                         self.df[z_col].values
                     )
                     
                     res = iv.result_
                     summary += f"Model: IV (2SLS)\nFirst Stage F: {res.first_stage_f:.4f}\n"
                     # Coefs
                     names = ["Intercept"] + x_endog_col
                     for i, val in enumerate(res.coef):
                        name = names[i] if i < len(names) else f"Var{i}"
                        summary += f"{name:<15} {val:.4f}\n"

                elif "Diff-in-Diff" in model:
                     # L1=Y, L2=Treated(D), L3=Post(T)
                     d_col = self.params.get('features')[0]
                     t_col = self.params.get('aux')
                     
                     did = StatelixDID()
                     did.fit(
                         self.df[y_col].values,
                         self.df[d_col].values,
                         self.df[t_col].values
                     )
                     
                     res = did.result_
                     summary += f"Model: DID\nATT: {res.att:.4f} (p={res.p_value:.3f})\n"
                     summary += f"Parallel Trends: {res.parallel_trends_valid}\n"

            # --- Bayesian ---
            elif "Bayesian" in model:
                y_col = self.params.get('target')
                x_cols = self.params.get('features')
                
                y = self.df[y_col].to_numpy(dtype=float)
                X = self.df[x_cols].to_numpy(dtype=float)
                
                bayes_model = BayesianLogisticRegression(
                    n_samples=self.params['samples'], 
                    warmup=self.params['warmup']
                )
                bayes_model.fit(X, y)
                
                s = bayes_model.summary
                summary += f"Model: Bayesian Logistic (HMC)\nSamples: {self.params['samples']}\n"
                summary += f"Acceptance: {s['acceptance']:.2f}\n"
                summary += f"Min ESS: {np.min(s['ess']):.1f}\n"
                
                means = s['mean']
                stds = s['std']
                for i, name in enumerate(x_cols):
                    summary += f"{name:<15} Mean: {means[i]:.3f}  Std: {stds[i]:.3f}\n"
                
                result_data["trace"] = bayes_model.samples_

            # --- Search (HNSW) ---
            elif "HNSW" in model:
                cols = self.params.get('features')
                data = self.df[cols].to_numpy(dtype=float)
                
                hnsw = StatelixHNSW(
                    M=self.params['M'], 
                    ef_construction=self.params['ef']
                )
                hnsw.fit(data)
                
                summary += f"Model: HNSW Index\nBuilt successfully.\n"
                summary += f"Index Size: {data.shape[0]} vectors.\n"

            # --- Fallback (OLS, etc) ---
            else:
                 if "OLS" in model:
                    y_col = self.params.get('target')
                    x_cols = self.params.get('features')
                    y = self.df[y_col].values
                    X = self.df[x_cols].values
                    res = cpp_binding.fit_ols_full(X, y)
                    summary += f"Model: OLS\nR2: {res.r_squared:.4f}\n"
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
        self.resize(1200, 800)
        self.worker = None # Keep reference
        
        self.init_ui()

        # Top Level Tabs for Modes
        self.mode_tabs = QTabWidget()
        self.setCentralWidget(self.mode_tabs)
        
        # --- Plugins (Load First) ---
        from statelix_py.plugins.loader import WasmPluginLoader
        self.plugin_loader = WasmPluginLoader()
        self.loaded_plugins = self.plugin_loader.scan_and_load()
        
        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        plugin_menu = menu.addMenu("Plugins")
        
        if not self.loaded_plugins:
             plugin_menu.addAction("No plugins found").setEnabled(False)
        else:
             for name, info in self.loaded_plugins.items():
                 # sub = plugin_menu.addMenu(name)
                 # For now just list them
                 plugin_menu.addAction(f"Loaded: {name}").setEnabled(False)
        
        plugin_menu.addSeparator()
        reload_action = plugin_menu.addAction("Reload Plugins")
        reload_action.triggered.connect(self.reload_plugins)
        
        # --- Mode 1: Inquiry (New Student GUI) ---
        self.inquiry_panel = InquiryPanel()
        self.mode_tabs.addTab(self.inquiry_panel, "🎓 Inquiry Mode")
        
        # --- Mode 2: Standard (Expert GUI) ---
        classic_widget = QWidget()
        classic_layout = QVBoxLayout(classic_widget)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.data_panel = DataPanel()
        self.model_panel = ModelPanel()
        
        self.output_tabs = QTabWidget()
        self.result_panel = ResultPanel()
        self.plot_panel = PlotPanel()
        self.exploratory_panel = ExploratoryPanel()
        
        self.output_tabs.addTab(self.result_panel, "Result")
        self.output_tabs.addTab(self.plot_panel, "Plots")
        self.output_tabs.addTab(self.exploratory_panel, "EDA")
        
        # Connect Classic
        self.model_panel.run_requested.connect(self.run_analysis)
        self.data_panel.data_loaded.connect(self.model_panel.update_columns)
        self.data_panel.data_loaded.connect(self.exploratory_panel.on_data_loaded)
        
        # Connect Inquiry (Also needs data updates)
        self.data_panel.data_loaded.connect(self.inquiry_panel.update_columns)
        self.inquiry_panel.run_inquiry.connect(self.run_analysis)

        splitter.addWidget(self.data_panel)
        splitter.addWidget(self.model_panel)
        splitter.addWidget(self.output_tabs)
        splitter.setSizes([200, 250, 350])
        
        classic_layout.addWidget(splitter)
        self.mode_tabs.addTab(classic_widget, "🛠️ Expert Mode")
        
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

        self.statusBar().showMessage(f"Running Analysis...")
        
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
             
             if viz_type == 'IV':
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

        else:
            # Classic Output
            self.result_panel.display_result(result)
            if "trace" in result:
                self.plot_panel.plot_hmc_trace(result['trace'])
                self.output_tabs.setCurrentWidget(self.plot_panel)
            else:
                self.output_tabs.setCurrentWidget(self.result_panel)

    @Slot(str)
    def on_analysis_error(self, msg):
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)

    def reload_plugins(self):
        self.loaded_plugins = self.plugin_loader.scan_and_load()
        self.model_panel.add_wasm_plugins(self.loaded_plugins)
        QMessageBox.information(self, "Plugins", f"Reloaded. Found {len(self.loaded_plugins)} plugins.")
