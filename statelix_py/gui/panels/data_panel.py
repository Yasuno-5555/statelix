from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableView, 
    QHeaderView, QHBoxLayout, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import pandas as pd
import os
from statelix_py.core.data_manager import DataManager
from statelix_py.gui.models.pandas_model import PandasModel

class DataPanel(QWidget):
    data_loaded = pyqtSignal(object) # Signal carrying the DataFrame

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

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        if file_path:
            try:
                # Load generic
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    return 
                
                # Update Manager
                self.dm.set_data(df, file_path) # Assumes set_data takes (df, filename)

                self.update_display(file_path, df)
                
                # Emit signal for other panels
                self.data_loaded.emit(df)
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def update_display(self, path, df: pd.DataFrame):
        filename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        self.file_label.setText(f"データセット: {filename}")
        self.info_label.setText(f"行: {df.shape[0]} | 列: {df.shape[1]} | サイズ: {size_mb:.2f}MB")
        
        # Update Model
        self.model.set_data(df)
