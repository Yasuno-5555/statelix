from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableView, 
    QHeaderView, QHBoxLayout, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import os
from statelix_py.core.data_manager import DataManager
from statelix_py.gui.models.pandas_model import PandasModel

class DataPanel(QWidget):
    data_loaded = Signal(object) # Signal carrying the DataFrame

    def __init__(self):
        super().__init__()
        self.dm = DataManager.instance()
        self.model = PandasModel()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("データパネル")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # File Info Frame
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_layout = QHBoxLayout()
        
        self.file_label = QLabel("データセット: 未選択")
        self.info_label = QLabel("行: - | 列: - | サイズ: -")
        
        info_layout.addWidget(self.file_label)
        info_layout.addStretch()
        info_layout.addWidget(self.info_label)
        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)

        
        # Tools Layout
        tools_layout = QHBoxLayout()
        self.btn_create_col = QPushButton("列作成 (Transform)")
        self.btn_create_col.clicked.connect(self.on_create_column)
        self.btn_filter = QPushButton("フィルタ (Filter)")
        self.btn_filter.clicked.connect(self.on_filter_rows)
        self.btn_reset = QPushButton("リセット (Reset)")
        self.btn_reset.clicked.connect(self.on_reset_data)
        
        tools_layout.addWidget(self.btn_create_col)
        tools_layout.addWidget(self.btn_filter)
        tools_layout.addWidget(self.btn_reset)
        tools_layout.addStretch()
        
        layout.addLayout(tools_layout)

        # Actions
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("データ読み込み")
        self.load_btn.clicked.connect(self.load_data)
        
        # Add Row / Delete Row (Future Work)
        # self.add_row_btn = QPushButton("+ 行")
        # self.del_row_btn = QPushButton("- 行")
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Table View (Spreadsheet)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setAlternatingRowColors(True)
        # self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_view)

        self.setLayout(layout)
        
        # --- Enable Drop ---
        self.setAcceptDrops(True)
        
        # Keep original data for reset
        self._original_df = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            # Optional: Visual cue (handled by QSS usually)
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self._load_file_path(file_path)

    def _load_file_path(self, file_path):
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "File Not Found", f"The file does not exist:\n{file_path}")
            return

        try:
            # Load generic
            if file_path.endswith('.csv'):
                # Try multiple encodings for Japanese environment
                encodings = ['utf-8', 'cp932', 'shift_jis', 'latin1']
                df = None
                last_error = None
                
                for enc in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        break
                    except (UnicodeDecodeError, ValueError) as e:
                        last_error = e
                        continue
                
                if df is None:
                    raise last_error or Exception("Unknown error during CSV loading")
                    
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                QMessageBox.warning(self, "Invalid File", "Only .csv and .xlsx files are supported.")
                return 
            
            # Update Manager and Keep Original
            self._original_df = df.copy()
            self._set_internal_data(df, file_path)
             
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Load Error", 
                                 f"Failed to load data:\n{str(e)}\n\n"
                                 f"Details:\n{error_details[:500]}...")

    def _set_internal_data(self, df, path):
        try:
            self.dm.set_data(df, path)
        except:
            self.dm.df = df
        
        self.update_display(path, df)
        self.data_loaded.emit(df)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
            self._load_file_path(file_path)
            
    def on_create_column(self):
        if self.dm.df is None: return
        from PySide6.QtWidgets import QInputDialog
        
        # Ask for new col name
        col, ok = QInputDialog.getText(self, "New Column", "Name of new column:")
        if not ok or not col: return
        
        # Ask for expression
        expr, ok = QInputDialog.getText(self, "Expression", f"Expression for {col} (e.g. A + B, np.log(GDP)):")
        if not ok or not expr: return
        
        try:
            df = self.dm.df.copy()
            # Use pandas eval or general eval tailored for safety/convenience
            # Supporting 'np' is useful
            import numpy as np
            df[col] = df.eval(expr, engine='python')
            
            self._set_internal_data(df, "Modified Data")
            QMessageBox.information(self, "Success", f"Column '{col}' created.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to transform:\n{e}")

    def on_filter_rows(self):
        if self.dm.df is None: return
        from PySide6.QtWidgets import QInputDialog
        
        # Ask for query
        query, ok = QInputDialog.getText(self, "Filter Rows", "Query (e.g. GDP > 1000 and Age < 50):")
        if not ok or not query: return
        
        try:
            df = self.dm.df.query(query, engine='python')
            self._set_internal_data(df, "Filtered Data")
            QMessageBox.information(self, "Success", f"Filtered to {len(df)} rows.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to filter:\n{e}")
            
    def on_reset_data(self):
        if self._original_df is not None:
             self._set_internal_data(self._original_df.copy(), "Original Data")
        else:
             QMessageBox.warning(self, "Warning", "No original data to reset to.")

    def update_display(self, path, df: pd.DataFrame):
        filename = os.path.basename(path)
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
        except:
            size_mb = 0.0
        
        self.file_label.setText(f"データセット: {filename}")
        self.info_label.setText(f"行: {df.shape[0]} | 列: {df.shape[1]} | サイズ: {size_mb:.2f}MB")
        
        # Update Model
        self.model.set_data(df)
