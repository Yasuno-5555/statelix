
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLabel, QSplitter, 
    QTabWidget, QComboBox, QTextEdit
)
from PySide6.QtCore import Qt, Slot
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from statelix_py.core.data_manager import DataManager
from statelix_py.gui.models.pandas_model import PandasModel

class ExploratoryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.dm = DataManager.instance()
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        # Models for tables
        self.desc_model = PandasModel()
        self.corr_model = PandasModel()
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Splitter: Top (Stats Tables) / Bottom (Plots)
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --- Top: Stats Tabs ---
        stats_tabs = QTabWidget()
        
        # 1. Descriptive Stats
        self.desc_view = QTableView()
        self.desc_view.setModel(self.desc_model)
        stats_tabs.addTab(self.desc_view, "要約統計量 (Describe)")
        
        # 2. Correlation
        self.corr_view = QTableView()
        self.corr_view.setModel(self.corr_model)
        stats_tabs.addTab(self.corr_view, "相関行列")
        
        # 3. Statistical Tests
        self.tests_output = QTextEdit()
        self.tests_output.setReadOnly(True)
        self.tests_output.setPlaceholderText("Select a variable and run tests...")
        stats_tabs.addTab(self.tests_output, "統計検定 (Tests)")
        
        splitter.addWidget(stats_tabs)
        
        # --- Bottom: Visualization ---
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        ctrl_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Histogram", "Box Plot", "Violin Plot", "Scatter Plot", "Pair Plot"])
        self.plot_type.currentTextChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.plot_type)
        
        ctrl_layout.addWidget(QLabel("X Variable:"))
        self.x_var = QComboBox()
        self.x_var.currentTextChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.x_var)
        
        ctrl_layout.addWidget(QLabel("Y Variable:"))
        self.y_var = QComboBox()
        self.y_var.currentTextChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.y_var)
        
        ctrl_layout.addWidget(QLabel("Color By:"))
        self.color_var = QComboBox()
        self.color_var.addItem("(None)")
        self.color_var.currentTextChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.color_var)
        
        viz_layout.addLayout(ctrl_layout)
        
        # Matplotlib Area with Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        # Insight / Fact Box
        self.insight_label = QTextEdit()
        self.insight_label.setReadOnly(True)
        self.insight_label.setMaximumHeight(60)
        self.insight_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 5px;")
        viz_layout.addWidget(self.insight_label)
        
        viz_widget.setLayout(viz_layout)
        
        splitter.addWidget(viz_widget)
        
        layout.addWidget(splitter)
        self.setLayout(layout)

    @Slot(object)
    def on_data_loaded(self, df: pd.DataFrame):
        """Called when data is loaded in MainWindow/DataPanel"""
        if df is None: return
        
        # Update Stats
        try:
            desc = df.describe(include='all').round(3)
            self.desc_model.set_data(desc)
        except:
             pass # numeric errors
        
        # Correlation (numeric only)
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            corr = num_df.corr().round(3)
            self.corr_model.set_data(corr)
        else:
            self.corr_model.set_data(pd.DataFrame(columns=["No numeric data"]))
        
        # Run normality tests on first 3 numeric columns
        try:
            from statelix_py.stats.tests import shapiro_wilk, format_test_result
            test_results = []
            for col in num_df.columns[:3]:
                data = num_df[col].dropna().values
                if len(data) >= 3 and len(data) < 5000: # Shapiro limit
                    result = shapiro_wilk(data)
                    test_results.append(f"<b>{col}</b>\n{format_test_result(result)}")
            
            if test_results:
                self.tests_output.setHtml("<pre>" + "\n\n".join(test_results) + "</pre>")
            else:
                self.tests_output.setText("N < 3 or N > 5000 (Skipping Shapiro).")
        except Exception as e:
            self.tests_output.setText(f"Test error: {e}")
            
        # Update Combo Boxes
        cols = df.columns.tolist()
        self.x_var.blockSignals(True)
        self.y_var.blockSignals(True)
        self.color_var.blockSignals(True)
        
        self.x_var.clear(); self.x_var.addItems(cols)
        self.y_var.clear(); self.y_var.addItems(cols)
        
        # Categorical columns for Color
        self.color_var.clear()
        self.color_var.addItem("(None)")
        # Heuristic: object, category, or int with few unique
        cat_cols = []
        for c in df.columns:
            if df[c].dtype == 'object' or str(df[c].dtype) == 'category':
                cat_cols.append(c)
            elif df[c].nunique() < 20: # discrete numeric likely categorical
                cat_cols.append(c)
        self.color_var.addItems(cat_cols)
        
        self.x_var.blockSignals(False)
        self.y_var.blockSignals(False)
        self.color_var.blockSignals(False)
        
        # Default plot
        self.update_plot()

    def update_plot(self):
        df = self.dm.df
        if df is None or df.empty: return
        
        ptype = self.plot_type.currentText()
        x_col = self.x_var.currentText()
        y_col = self.y_var.currentText()
        color_col = self.color_var.currentText()
        if color_col == "(None)": color_col = None
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        insight_text = ""
        
        # Helper for coloring
        groups = [(None, df)]
        if color_col and color_col in df.columns:
            # Drop NaNs in color col to avoid errors
            valid_df = df.dropna(subset=[color_col])
            # Limit groups to avoid chaos
            if valid_df[color_col].nunique() > 20:
                ax.text(0.5, 0.5, f"Too many groups in {color_col} (>20)", ha='center')
                self.canvas.draw()
                return
            groups = list(valid_df.groupby(color_col))
        
        # Color Cycle
        import itertools
        colors = itertools.cycle(['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2'])
        
        try:
            if ptype == "Histogram":
                if x_col:
                    if color_col:
                        # Stacked or layered hist
                        data_list = []
                        label_list = []
                        for g_name, g_df in groups:
                            d = pd.to_numeric(g_df[x_col], errors='coerce').dropna()
                            if not d.empty:
                                data_list.append(d)
                                label_list.append(str(g_name))
                        
                        if data_list:
                            ax.hist(data_list, bins='auto', alpha=0.7, label=label_list, stacked=True)
                            ax.legend(title=color_col)
                    else:
                        data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                        if not data.empty:
                            ax.hist(data, bins='auto', edgecolor='black', alpha=0.7, color='#007bff')
                    
                    ax.set_title(f"Histogram of {x_col}")
                    ax.set_xlabel(x_col)
                    self.y_var.setEnabled(False)
                    
            elif ptype == "Box Plot":
                if x_col:
                    if color_col:
                         # Grouped Box Plot manually
                         data_list = []
                         labels = []
                         for g_name, g_df in groups:
                             d = pd.to_numeric(g_df[x_col], errors='coerce').dropna()
                             if not d.empty:
                                 data_list.append(d)
                                 labels.append(str(g_name))
                         
                         if data_list:
                             ax.boxplot(data_list, vert=False, labels=labels)
                    else:
                        data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                        if not data.empty:
                            ax.boxplot(data, vert=False)
                    
                    ax.set_title(f"Box Plot of {x_col}")
                    self.y_var.setEnabled(False)

            elif ptype == "Violin Plot":
                if x_col:
                    data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                    if data.empty:
                        ax.text(0.5, 0.5, "No numeric data", ha='center')
                    else:
                        parts = ax.violinplot(data, vert=False, showmeans=True, showmedians=True)
                        for pc in parts['bodies']:
                            pc.set_facecolor('#3273dc')
                            pc.set_alpha(0.7)
                        ax.set_title(f"Violin Plot of {x_col}")
                        
                        skew = stats.skew(data)
                        kurt = stats.kurtosis(data)
                        insight_text = (f"<b>Facts:</b> Skewness={skew:.2f}, Kurtosis={kurt:.2f}. "
                                      f"Mean={data.mean():.2f}, Median={data.median():.2f}.")
                        
                    self.y_var.setEnabled(False)

            elif ptype == "Pair Plot":
                # Use up to 4 numeric columns for pair plot
                num_df = df.select_dtypes(include=[np.number])
                cols = num_df.columns[:4].tolist()
                if len(cols) < 2:
                    ax.text(0.5, 0.5, "Need at least 2 numeric columns", ha='center')
                else:
                    self.figure.clear()
                    n = len(cols)
                    for i, c1 in enumerate(cols):
                        for j, c2 in enumerate(cols):
                            ax = self.figure.add_subplot(n, n, i * n + j + 1)
                            if i == j:
                                ax.hist(num_df[c1].dropna(), bins=15, alpha=0.7)
                            else:
                                ax.scatter(num_df[c2], num_df[c1], alpha=0.5, s=5)
                            if i == n - 1:
                                ax.set_xlabel(c2, fontsize=7)
                            if j == 0:
                                ax.set_ylabel(c1, fontsize=7)
                            ax.tick_params(labelsize=5)
                    self.figure.tight_layout()
                    insight_text = f"<b>Pair Plot:</b> Showing correlations for {', '.join(cols)}."
                    
                self.y_var.setEnabled(False)
                    
            elif ptype == "Scatter Plot":
                if x_col and y_col:
                    for g_name, g_df in groups:
                        c = next(colors) if color_col else '#007bff'
                        lbl = str(g_name) if color_col else None
                        
                        x_data = pd.to_numeric(g_df[x_col], errors='coerce')
                        y_data = pd.to_numeric(g_df[y_col], errors='coerce')
                        
                        valid = ~np.isnan(x_data) & ~np.isnan(y_data)
                        if valid.sum() > 0:
                            ax.scatter(x_data[valid], y_data[valid], alpha=0.6, label=lbl, color=c)
                    
                    if color_col: ax.legend(title=color_col)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} vs {x_col}")
                    
                    # Insight: Overall Correlation
                    full_x = pd.to_numeric(df[x_col], errors='coerce')
                    full_y = pd.to_numeric(df[y_col], errors='coerce')
                    v = ~np.isnan(full_x) & ~np.isnan(full_y)
                    if v.sum() > 2:
                        corr = np.corrcoef(full_x[v], full_y[v])[0, 1]
                        insight_text = f"<b>Overall Correlation:</b> r={corr:.3f}"

                    self.y_var.setEnabled(True)
        
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot Error: {str(e)}", ha='center')
            insight_text = "Error generating insight."
            
        self.canvas.draw()
        self.insight_label.setHtml(insight_text)
