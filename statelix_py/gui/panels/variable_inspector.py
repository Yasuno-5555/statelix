from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QLabel, QHBoxLayout, QSplitter,
    QPushButton, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt, Signal
import pandas as pd

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from statelix_py.gui.styles import StatelixTheme
from statelix_py.core.data_manager import DataManager

class VariableInspector(QWidget):
    data_changed = Signal() # Signal to notify other panels (like DataPanel)

    def __init__(self):
        super().__init__()
        self.dm = DataManager.instance()
        self.df = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # layout.setContentsMargins(0,0,0,0)
        
        title = QLabel("Variable Inspector")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Column List
        self.col_list = QListWidget()
        self.col_list.currentItemChanged.connect(self.on_col_selected)
        splitter.addWidget(self.col_list)
        
        # 2. Details Area
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0,0,0,0)
        
        self.stats_label = QLabel("Select a variable...")
        self.stats_label.setWordWrap(True)
        details_layout.addWidget(self.stats_label)
        
        # Preprocessing Tools
        tools_layout = QHBoxLayout()
        self.btn_impute = QPushButton("Impute")
        self.btn_impute.clicked.connect(self.on_impute)
        self.btn_encode = QPushButton("Encode")
        self.btn_encode.clicked.connect(self.on_encode)
        self.btn_convert = QPushButton("Convert")
        self.btn_convert.clicked.connect(self.on_convert)
        
        tools_layout.addWidget(self.btn_impute)
        tools_layout.addWidget(self.btn_encode)
        tools_layout.addWidget(self.btn_convert)
        details_layout.addLayout(tools_layout)
        
        # Mini Plot
        self.figure = Figure(figsize=(3, 2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._setup_ax() # Init Theme
        
        details_layout.addWidget(self.canvas)
        
        splitter.addWidget(details_widget)
        splitter.setSizes([200, 300]) # List smaller, details larger?
        
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _setup_ax(self):
        # Apply Dark Theme to Figure
        self.figure.patch.set_facecolor(StatelixTheme.COLOR_BG_MAIN)
        
    def set_data(self, df: pd.DataFrame):
        self.df = df
        self.col_list.clear()
        if df is not None:
             self.col_list.addItems(df.columns)
        
        self.stats_label.setText("Data loaded. Select a column.")
        self.figure.clear()
        self.canvas.draw()

    def on_impute(self):
        item = self.col_list.currentItem()
        if not item or self.df is None: return
        
        col = item.text()
        strategies = ["mean", "median", "mode", "constant"]
        strategy, ok = QInputDialog.getItem(self, "Impute Missing", "Select Strategy:", strategies, 0, False)
        if not ok: return
        
        fill_value = None
        if strategy == "constant":
            fill_value, ok = QInputDialog.getDouble(self, "Constant Value", "Enter value:", 0.0)
            if not ok: return
            
        if self.dm.impute_missing(col, strategy, fill_value):
            self.df = self.dm.df
            self.on_col_selected(item, None)
            self.data_changed.emit()
            QMessageBox.information(self, "Success", f"Missing values in '{col}' imputed via {strategy}.")

    def on_encode(self):
        item = self.col_list.currentItem()
        if not item or self.df is None: return
        
        col = item.text()
        methods = ["one-hot", "label"]
        method, ok = QInputDialog.getItem(self, "Categorical Encoding", "Select Method:", methods, 0, False)
        if not ok: return
        
        if self.dm.encode_categorical(col, method):
            self.df = self.dm.df
            self.set_data(self.df) # Refresh list as encoding might add columns
            self.data_changed.emit()
            QMessageBox.information(self, "Success", f"Column '{col}' encoded via {method}.")

    def on_convert(self):
        item = self.col_list.currentItem()
        if not item or self.df is None: return
        
        col = item.text()
        types = ["float", "int", "str", "category"]
        new_type, ok = QInputDialog.getItem(self, "Type Conversion", "Select New Type:", types, 0, False)
        if not ok: return
        
        if self.dm.convert_type(col, new_type):
            self.df = self.dm.df
            self.on_col_selected(item, None)
            self.data_changed.emit()
            QMessageBox.information(self, "Success", f"Column '{col}' converted to {new_type}.")
        else:
            QMessageBox.critical(self, "Error", f"Failed to convert '{col}' to {new_type}.")

    def on_col_selected(self, current, previous):
        if not current or self.df is None: return
        
        col = current.text()
        series = self.df[col]
        
        # Stats
        dtype = series.dtype
        n_missing = series.isna().sum()
        n_unique = series.nunique()
        
        stats_text = f"<b>{col}</b><br/>Type: {dtype}<br/>NaN: {n_missing} ({n_missing/len(series):.1%})<br/>Unique: {n_unique}"
        
        # Check for value labels
        labels = self.dm.get_value_labels(col)
        if labels:
            stats_text += f"<br/>Labels: {labels}"
        
        if pd.api.types.is_numeric_dtype(dtype):
            desc = series.describe()
            stats_text += f"<br/>Mean: {desc['mean']:.2f}<br/>Std: {desc['std']:.2f}<br/>Min: {desc['min']:.2f} | Max: {desc['max']:.2f}"
            
            # Plot Histogram
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            # Style
            bg = StatelixTheme.COLOR_BG_MAIN
            fg = StatelixTheme.COLOR_TEXT_MAIN
            ax.set_facecolor(bg)
            ax.spines['bottom'].set_color(fg)
            ax.spines['top'].set_color(bg) 
            ax.spines['left'].set_color(fg)
            ax.spines['right'].set_color(bg)
            ax.tick_params(colors=fg, labelsize=8)
            
            # Draw
            data = series.dropna()
            if len(data) > 0:
                ax.hist(data, bins=20, color=StatelixTheme.COLOR_ACCENT, alpha=0.7)
            ax.set_title("Distribution", color=fg, fontsize=9)
            self.canvas.draw()
            
        else:
            # Categorical - show counts
            counts = series.value_counts().head(5)
            stats_text += "<br/>Top 5:"
            for val, count in counts.items():
                stats_text += f"<br/> {val}: {count}"
                
            self.figure.clear()
            self.canvas.draw()
            
        self.stats_label.setText(stats_text)
