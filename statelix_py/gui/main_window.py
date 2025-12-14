from statelix_py.gui.panels.data_panel import DataPanel
from statelix_py.gui.panels.model_panel import ModelPanel
from statelix_py.gui.panels.result_panel import ResultPanel
from statelix_py.gui.panels.plot_panel import PlotPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statelix v2.2")
        self.resize(1200, 800) # Slightly wider for plots
        
        self.init_menu()
        self.init_ui()

    def init_menu(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("ファイル")
        file_menu.addAction("新規プロジェクト")
        file_menu.addAction("開く...")
        file_menu.addSeparator()
        file_menu.addAction("終了", self.close)
        
        # Edit Menu
        edit_menu = menubar.addMenu("編集")
        
        # View Menu
        view_menu = menubar.addMenu("表示")
        
        # Help Menu
        help_menu = menubar.addMenu("ヘルプ")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Use QSplitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Initialize Panels
        self.data_panel = DataPanel()
        self.model_panel = ModelPanel()
        
        # Output Tabs (Results + Plots)
        self.output_tabs = QTabWidget()
        self.result_panel = ResultPanel()
        self.plot_panel = PlotPanel()
        
        self.output_tabs.addTab(self.result_panel, "テキスト結果")
        self.output_tabs.addTab(self.plot_panel, "プロット (Viz)")
        
        # Connect Signals
        self.model_panel.run_requested.connect(self.run_analysis)
        self.data_panel.data_loaded.connect(self.model_panel.update_columns)

        # Add to Splitter
        splitter.addWidget(self.data_panel)
        splitter.addWidget(self.model_panel)
        splitter.addWidget(self.output_tabs) # Add Tabs instead of single panel
        
        # Set initial sizes (optional)
        splitter.setSizes([200, 200, 400])
        splitter.setHandleWidth(5)

        main_layout.addWidget(splitter)
        
        # Status Bar
        self.statusBar().showMessage("Ready")

    def run_analysis(self, params):
        self.statusBar().showMessage(f"Running analysis: {params['model']}...")
        
        from statelix_py.core.data_manager import DataManager
        from statelix_py.core import cpp_binding
        import time

        dm = DataManager.instance()
        
        target_col = params.get('target')
        feature_cols = params.get('features', [])
        
        if not target_col and "OLS" in params['model']:
            self.statusBar().showMessage("Error: Target variable is required.")
            return

        if not feature_cols and "OLS" in params['model']:
            self.statusBar().showMessage("Error: At least one feature is required.")
            return

        start_time = time.time()
        result_data = {}
        
        try:
            summary = ""
            
            # --- OLS (最小二乗法) ---
            if "OLS" in params['model']:
                if not target_col or not feature_cols:
                    raise ValueError("Target and at least one Feature required for OLS.")
                    
                y = dm.get_column(target_col).to_numpy(dtype=float)
                X = dm.get_data_matrix(feature_cols)
                
                res = cpp_binding.fit_ols(X, y)
                
                summary += f"Model: OLS Regression\nTarget: {target_col}\n"
                summary += f"R-Squared: {res.r_squared:.4f}  (Adj: {res.adj_r_squared:.4f})\n"
                summary += f"F-Stat: {res.f_statistic:.4f} (p={res.f_pvalue:.4e})\n"
                summary += "-" * 50 + "\n"
                summary += f"{'Variable':<15} {'Coef':<10} {'StdErr':<10} {'t-val':<10} {'p-val':<10}\n"
                summary += f"{'Intercept':<15} {res.intercept:<10.4f} {'-':<10} {'-':<10} {'-':<10}\n"
                for i, name in enumerate(feature_cols):
                    summary += f"{name:<15} {res.coef[i]:<10.4f} {res.std_errors[i]:<10.4f} {res.t_values[i]:<10.4f} {res.p_values[i]:<10.4f}\n"

                # Plot
                self.plot_panel.plot_ols_diagnostics(res.fitted_values, res.residuals)
            
            # --- K-Means (クラスタリング) ---
            elif "K-Means" in params['model']:
                if not feature_cols:
                    raise ValueError("Features required for K-Means.")
                
                X = dm.get_data_matrix(feature_cols)
                k = params.get('k', 3)
                
                res = cpp_binding.fit_kmeans(X, k)
                
                summary += f"Model: K-Means Clustering\nFeatures: {len(feature_cols)}\nK: {k}\n"
                summary += f"Inertia: {res.inertia:.4f}\nIterations: {res.n_iter}\n"
                summary += "-" * 40 + "\n"
                summary += "Cluster Centers:\n"
                for i in range(k):
                    center_vals = ", ".join([f"{x:.2f}" for x in res.centroids[i]])
                    summary += f"Cluster {i}: [{center_vals}]\n"
                    
                # Plot
                self.plot_panel.plot_clustering(X, res.labels, res.centroids)

            # --- ANOVA (分散分析) ---
            elif "ANOVA" in params['model']:
                if not target_col or not feature_cols:
                    raise ValueError("Target and Group (Feature) required for ANOVA.")
                
                data = dm.get_column(target_col).to_numpy(dtype=float)
                # Assume first feature is group. Convert to integer codes if generic
                group_col = dm.get_column(feature_cols[0])
                if group_col.dtype == 'object':
                    groups = group_col.astype('category').cat.codes.to_numpy(dtype='int32')
                else:
                    groups = group_col.to_numpy(dtype='int32')
                    
                res = cpp_binding.f_oneway(data, groups)
                
                summary += f"Model: One-Way ANOVA\nData: {target_col} by Group: {feature_cols[0]}\n"
                summary += f"F-Statistic: {res.f_statistic:.4f}\nP-Value: {res.p_value:.4e}\n"
                summary += "-" * 40 + "\n"
                summary += f"{'Source':<10} {'DF':<5} {'SS':<10} {'MS':<10}\n"
                summary += f"{'Between':<10} {res.df_between:<5} {res.ss_between:<10.2f} {res.ms_between:<10.2f}\n"
                summary += f"{'Within':<10} {res.df_within:<5} {res.ss_within:<10.2f} {res.ms_within:<10.2f}\n"
                summary += f"{'Total':<10} {res.df_total:<5} {res.ss_total:<10.2f}\n"
                
                # Plot
                self.plot_panel.plot_boxplot(data, groups)

            # --- AR Model (時系列) ---
            elif "AR" in params['model']:
                if not target_col:
                    raise ValueError("Target (Time Series) required for AR.")
                
                series = dm.get_column(target_col).to_numpy(dtype=float)
                p = params.get('p', 1)
                
                res = cpp_binding.fit_ar(series, p)
                
                summary += f"Model: Autoregressive AR({p})\nSeries: {target_col}\n"
                summary += f"Sigma^2: {res.sigma2:.6f}\n"
                summary += "-" * 40 + "\n"
                summary += f"Const: {res.params[0]:.4f}\n"
                for i in range(p):
                    summary += f"Phi_{i+1}: {res.params[i+1]:.4f}\n"
                
                # Plot
                self.plot_panel.plot_time_series(series, title=f"AR({p}) - {target_col}")

            # --- Ridge Regression ---
            elif "Ridge" in params['model']:
                if not target_col or not feature_cols:
                    raise ValueError("Target and Features required for Ridge.")
                
                y = dm.get_column(target_col).to_numpy(dtype=float)
                X = dm.get_data_matrix(feature_cols)
                alpha = params.get('alpha', 1.0)
                
                res = cpp_binding.fit_ridge(X, y, alpha)
                
                summary += f"Model: Ridge Regression (L2)\nLambda: {alpha}\n"
                summary += f"{'Variable':<15} {'Coef':<10}\n"
                for i, name in enumerate(feature_cols):
                    summary += f"{name:<15} {res.coef[i]:<10.4f}\n"

            # --- Cox Proportional Hazards ---
            elif "Cox" in params['model']:
                if not target_col or not feature_cols or not params.get('status'):
                    raise ValueError("Time (Target), Status, and Covariates (Features) required for CoxPH.")
                
                time_vec = dm.get_column(target_col).to_numpy(dtype=float)
                status_vec = dm.get_column(params['status']).to_numpy(dtype='int32')
                X = dm.get_data_matrix(feature_cols)
                
                res = cpp_binding.fit_cox_ph(X, time_vec, status_vec)
                
                summary += f"Model: Cox Proportional Hazards\nTime: {target_col}, Status: {params['status']}\n"
                summary += f"{'Variable':<15} {'Hazard Ratio':<15} {'Coef':<10}\n"
                for i, name in enumerate(feature_cols):
                    hr = 2.71828 ** res.coef[i]
                    summary += f"{name:<15} {hr:<15.4f} {res.coef[i]:<10.4f}\n"

            # --- GLM Models (Logistic, Poisson, etc) ---
            elif "Regression" in params['model']: # Matches Logistic, Poisson, etc. (names end in Regression)
                if not target_col or not feature_cols:
                    raise ValueError("Target and Features required for GLM.")
                
                y = dm.get_column(target_col).to_numpy(dtype=float)
                X = dm.get_data_matrix(feature_cols)
                max_iter = params.get('max_iter', 50)
                
                model_name = params['model']
                
                if "Logistic" in model_name:
                    res = cpp_binding.fit_logistic(X, y, max_iter)
                elif "Poisson" in model_name:
                    res = cpp_binding.fit_poisson(X, y, max_iter)
                elif "Negative" in model_name:
                    res = cpp_binding.fit_negbin(X, y) # NegBin has built-in looping
                elif "Gamma" in model_name:
                    res = cpp_binding.fit_gamma(X, y)
                elif "Probit" in model_name:
                    res = cpp_binding.fit_probit(X, y)
                
                summary += f"Model: {model_name}\nIterations: {getattr(res, 'iterations', 'N/A')}\n"
                # Some have converged field
                if hasattr(res, 'converged'):
                     summary += f"Converged: {res.converged}\n"
                
                summary += f"{'Variable':<15} {'Coef':<10}\n"
                for i, name in enumerate(feature_cols):
                    summary += f"{name:<15} {res.coef[i]:<10.4f}\n"

            else:
                 summary = "Unknown Model Selected."

            result_data = {
                "success": True,
                "summary": summary,
                "hash": "cpp_backend_full"
            }
            if "r2" in locals(): result_data["r2"] = "N/A" # Default if not OLS
            
            # Switch to Output Tab
            self.output_tabs.setCurrentIndex(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Clean up error message
            msg = str(e)
            if "RuntimeError:" in msg:
                msg = msg.replace("RuntimeError:", "").strip()
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Execution Error", f"Analysis Failed:\n\n{msg}")
            
            result_data = {
                "success": False, 
                "summary": f"Error: {msg}" 
            }

        elapsed = time.time() - start_time
        summary_footer = f"\nAnalysis completed in {elapsed:.3f}s."
        if "summary" in result_data:
            result_data["summary"] += summary_footer
            
        self.result_panel.display_result(result_data)
        self.statusBar().showMessage("Analysis completed.")
