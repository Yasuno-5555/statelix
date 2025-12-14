from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QSplitter, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from statelix_py.gui.panels.data_panel import DataPanel
from statelix_py.gui.panels.model_panel import ModelPanel
from statelix_py.gui.panels.result_panel import ResultPanel
from statelix_py.gui.panels.plot_panel import PlotPanel

import time
import pandas as pd
import numpy as np

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
            # Use High-Level SDK Models
            from statelix_py.models import (
                StatelixGraph, StatelixIV, StatelixDID, StatelixPSM,
                BayesianLogisticRegression, StatelixHNSW
            )
            # Legacy fallback for old models
            from statelix_py.core import cpp_binding 
            
            result_data = {}
            summary = ""
            start_time = time.time()
            
            model = self.params['model']
            
            # --- Graph Analysis (Refactored) ---
            if "Graph" in model:
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
        self.setWindowTitle("Statelix v2.2")
        self.resize(1200, 800)
        self.worker = None # Keep reference
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.data_panel = DataPanel()
        self.model_panel = ModelPanel()
        
        self.output_tabs = QTabWidget()
        self.result_panel = ResultPanel()
        self.plot_panel = PlotPanel()
        
        self.output_tabs.addTab(self.result_panel, "テキスト結果")
        self.output_tabs.addTab(self.plot_panel, "プロット (Viz)")
        
        # Connect
        self.model_panel.run_requested.connect(self.run_analysis)
        self.data_panel.data_loaded.connect(self.model_panel.update_columns)
        
        splitter.addWidget(self.data_panel)
        splitter.addWidget(self.model_panel)
        splitter.addWidget(self.output_tabs)
        splitter.setSizes([200, 250, 350])
        
        main_layout.addWidget(splitter)
        self.statusBar().showMessage("Ready")

    def run_analysis(self, params):
        from statelix_py.core.data_manager import DataManager
        dm = DataManager.instance()
        if dm.df is None or dm.df.empty:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        self.statusBar().showMessage(f"Running {params['model']}...")
        self.model_panel.run_btn.setEnabled(False) # Disable button
        
        # Start Worker
        self.worker = AnalysisWorker(params, dm.df)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    @Slot(dict)
    def on_analysis_finished(self, result):
        self.model_panel.run_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis Completed.")
        
        self.result_panel.display_result(result)
        
        # Handle Viz
        if "trace" in result:
            self.plot_panel.plot_hmc_trace(result['trace'])
            self.output_tabs.setCurrentWidget(self.plot_panel)
        else:
            self.output_tabs.setCurrentWidget(self.result_panel)

    @Slot(str)
    def on_analysis_error(self, msg):
        self.model_panel.run_btn.setEnabled(True)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)
