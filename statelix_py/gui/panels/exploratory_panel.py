
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLabel, QSplitter, 
    QTabWidget, QComboBox, QCheckBox, QFrame, QTextEdit
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
        
        splitter.addWidget(stats_tabs)
        
        # --- Bottom: Visualization ---
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        ctrl_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Histogram", "Box Plot", "Scatter Plot"])
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
        desc = df.describe(include='all').round(3)
        self.desc_model.set_data(desc)
        
        # Correlation (numeric only)
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            corr = num_df.corr().round(3)
            self.corr_model.set_data(corr)
        else:
            self.corr_model.set_data(pd.DataFrame(columns=["No numeric data"]))
            
        # Update Combo Boxes
        cols = df.columns.tolist()
        self.x_var.blockSignals(True)
        self.y_var.blockSignals(True)
        self.x_var.clear()
        self.y_var.clear()
        self.x_var.addItems(cols)
        self.y_var.addItems(cols)
        self.x_var.blockSignals(False)
        self.y_var.blockSignals(False)
        
        # Default plot
        self.update_plot()

    def update_plot(self):
        df = self.dm.df
        if df is None or df.empty: return
        
        ptype = self.plot_type.currentText()
        x_col = self.x_var.currentText()
        y_col = self.y_var.currentText()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        insight_text = ""
        
        try:
            if ptype == "Histogram":
                if x_col:
                    data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                    if data.empty:
                        ax.text(0.5, 0.5, "No numeric data", ha='center')
                    else:
                        ax.hist(data, bins='auto', edgecolor='black', alpha=0.7)
                        ax.set_title(f"Histogram of {x_col}")
                        ax.set_xlabel(x_col)
                        
                        # Insight: Skewness and central tendency
                        mean_val = data.mean()
                        median_val = data.median()
                        skew = stats.skew(data)
                        n = len(data)
                        insight_text = (f"<b>Facts:</b> N={n}. Mean={mean_val:.2f}, Median={median_val:.2f}. "
                                      f"Skewness={skew:.2f} (>0 means right-tailed).")
                        
                    self.y_var.setEnabled(False)
                    
            elif ptype == "Box Plot":
                if x_col:
                    data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                    if data.empty:
                        ax.text(0.5, 0.5, "No numeric data", ha='center')
                    else:
                        ax.boxplot(data, vert=False)
                        ax.set_title(f"Box Plot of {x_col}")
                        
                        # Insight: IQR and Outliers
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        insight_text = (f"<b>Facts:</b> Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}. "
                                      f"Range: [{data.min():.2f}, {data.max():.2f}].")
                        
                    self.y_var.setEnabled(False)
                    
            elif ptype == "Scatter Plot":
                if x_col and y_col:
                    x_data = pd.to_numeric(df[x_col], errors='coerce')
                    y_data = pd.to_numeric(df[y_col], errors='coerce')
                    
                    # Drop NaNs for plotting
                    valid = ~np.isnan(x_data) & ~np.isnan(y_data)
                    
                    if valid.sum() == 0:
                         ax.text(0.5, 0.5, "No valid pairs", ha='center')
                    else:
                        ax.scatter(x_data[valid], y_data[valid], alpha=0.6)
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"{y_col} vs {x_col}")
                        
                        # Insight: Correlation
                        corr = np.corrcoef(x_data[valid], y_data[valid])[0, 1]
                        n = valid.sum()
                        insight_text = (f"<b>Facts:</b> N={n}. Pearson Correlation r={corr:.3f}. "
                                      f"R-squared={corr**2:.3f}.")
                        
                    self.y_var.setEnabled(True)
        
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot Error: {str(e)}", ha='center')
            insight_text = "Error generating insight."
            
        self.canvas.draw()
        self.insight_label.setHtml(insight_text)
