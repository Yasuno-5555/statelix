from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableView, 
    QHBoxLayout, QFrame, QMessageBox, QInputDialog
)
from PySide6.QtCore import Signal, Qt
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
        self.btn_merge = QPushButton("結合 (Merge)")
        self.btn_merge.clicked.connect(self.on_merge)
        self.btn_reshape = QPushButton("変形 (Reshape)")
        self.btn_reshape.clicked.connect(self.on_reshape)
        self.btn_reset = QPushButton("リセット (Reset)")
        self.btn_reset.clicked.connect(self.on_reset_data)
        
        tools_layout.addWidget(self.btn_create_col)
        tools_layout.addWidget(self.btn_filter)
        tools_layout.addWidget(self.btn_merge)
        tools_layout.addWidget(self.btn_reshape)
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
        self.table_view.setSortingEnabled(True) # Enable sorting
        
        # --- Context Menu ---
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.on_context_menu)
        
        layout.addWidget(self.table_view)
        
        # Shortcuts for Copy/Paste
        from PySide6.QtGui import QKeySequence, QAction, QShortcut
        self.copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.table_view)
        self.copy_shortcut.activated.connect(self.copy_selection)
        
        self.paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self.table_view)
        self.paste_shortcut.activated.connect(self.paste_selection)

        self.setLayout(layout)
        
        # --- Enable Drop ---
        self.setAcceptDrops(True)
        
        # Keep original data for reset
        self._original_df = None

    def on_context_menu(self, pos):
        if self.dm.df is None: return
        
        index = self.table_view.indexAt(pos)
        if not index.isValid(): return
        
        col_idx = index.column()
        col_name = self.dm.df.columns[col_idx]
        
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: white; border: 1px solid #ccc; } QMenu::item { padding: 5px 20px; } QMenu::item:selected { background-color: #007bff; color: white; }")
        
        # Header Info
        action_header = menu.addAction(f"Column: {col_name}")
        action_header.setEnabled(False)
        menu.addSeparator()
        
        # Actions
        action_rename = menu.addAction("Rename Column...")
        
        # Excel-like Actions
        action_copy = menu.addAction("Copy")
        action_paste = menu.addAction("Paste")
        menu.addSeparator()

        # Type Conversion Submenu
        type_menu = menu.addMenu("Change Type")
        action_to_numeric = type_menu.addAction("To Numeric")
        action_to_string = type_menu.addAction("To String")
        action_to_datetime = type_menu.addAction("To DateTime")
        action_to_cat = type_menu.addAction("To Categorical")
        
        menu.addSeparator()
        action_drop_nan = menu.addAction("Drop Rows with NaN (This Column)")
        action_drop_col = menu.addAction("Drop Column")
        
        # Execute
        action = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        
        if action == action_rename:
            self._rename_column(col_name)
        elif action == action_copy:
            self.copy_selection()
        elif action == action_paste:
            self.paste_selection()
        elif action == action_to_numeric:
            self._change_type(col_name, 'numeric')
        elif action == action_to_string:
            self._change_type(col_name, 'string')
        elif action == action_to_datetime:
            self._change_type(col_name, 'datetime')
        elif action == action_to_cat:
             self._change_type(col_name, 'category')
        elif action == action_drop_nan:
            self._drop_nan(col_name)
        elif action == action_drop_col:
            self._drop_column(col_name)

    def copy_selection(self):
        selection = self.table_view.selectionModel()
        if not selection.hasSelection(): return
        
        indexes = selection.selectedIndexes()
        if not indexes: return
        
        # Sort to ensure order
        indexes.sort(key=lambda x: (x.row(), x.column()))
        
        # Determine bounds
        rows = sorted(list(set(index.row() for index in indexes)))
        cols = sorted(list(set(index.column() for index in indexes)))
        
        # Construct TSV string
        df = self.dm.df
        text = ""
        for r in rows:
            row_data = []
            for c in cols:
                try:
                    val = str(df.iloc[r, c])
                    row_data.append(val)
                except:
                    row_data.append("")
            text += "\t".join(row_data) + "\n"
            
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def paste_selection(self):
        if self.dm.df is None: return
        
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text: return
        
        # Parse clipboard (assume TSV from Excel/Statelix)
        rows_data = [row.split('\t') for row in text.splitlines()]
        if not rows_data: return

        # Get top-left target
        selection = self.table_view.selectionModel()
        if selection.hasSelection():
            indexes = selection.selectedIndexes()
            start_row = min(index.row() for index in indexes)
            start_col = min(index.column() for index in indexes)
        else:
            start_row = 0
            start_col = 0
            
        try:
            df = self.dm.df # Modify in place if model allows, generally better to copy triggers
            # We must use model.setData to emit signals or reset whole model.
            # Using model directly for bulk updates is faster but needs refresh.
            
            # Simple approach: Loop set data in Model
            model = self.model
            
            for i, row_vals in enumerate(rows_data):
                r = start_row + i
                if r >= df.shape[0]: break # Don't expand rows yet
                
                for j, val in enumerate(row_vals):
                   c = start_col + j
                   if c >= df.shape[1]: break
                   
                   idx = model.index(r, c)
                   model.setData(idx, val, Qt.ItemDataRole.EditRole)
                   
            # Determine if we should emit layoutChanged for bulk?
            # Since we used setData, it emitted dataChanged per cell.
            
        except Exception as e:
            QMessageBox.critical(self, "Paste Error", f"{e}")

    def _rename_column(self, old_name):
        new_name, ok = QInputDialog.getText(self, "Rename Column", "New Name:", text=old_name)
        if ok and new_name and new_name != old_name:
            try:
                df = self.dm.df.copy()
                df = df.rename(columns={old_name: new_name})
                self._set_internal_data(df, "Renamed Column")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _change_type(self, col, target_type):
        try:
            df = self.dm.df.copy()
            if target_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif target_type == 'string':
                df[col] = df[col].astype(str)
            elif target_type == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif target_type == 'category':
                df[col] = df[col].astype('category')
            
            self._set_internal_data(df, f"Converted {col} to {target_type}")
            QMessageBox.information(self, "Success", f"Converted '{col}' to {target_type}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{e}")

    def _drop_nan(self, col):
        try:
            df = self.dm.df.copy()
            initial_len = len(df)
            df = df.dropna(subset=[col])
            dropped = initial_len - len(df)
            self._set_internal_data(df, f"Dropped NaN from {col}")
            QMessageBox.information(self, "Success", f"Dropped {dropped} rows.")
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))

    def _drop_column(self, col):
        res = QMessageBox.question(self, "Confirm", f"Delete column '{col}'?", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            try:
                df = self.dm.df.drop(columns=[col])
                self._set_internal_data(df, f"Dropped {col}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

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
            ext = os.path.splitext(file_path)[1].lower()
            df = None

            if ext in ['.csv', '.txt', '.tsv']:
                # Try multiple encodings for Japanese environment
                encodings = ['utf-8', 'cp932', 'shift_jis', 'latin1']
                sep = ',' if ext == '.csv' else (None if ext == '.txt' else '\t')
                
                last_error = None
                for enc in encodings:
                    try:
                        # For .txt, it might be space or comma or tab, sep=None handles inference
                        df = pd.read_csv(file_path, encoding=enc, sep=sep, engine='python')
                        break
                    except (UnicodeDecodeError, ValueError) as e:
                        last_error = e
                        continue
                
                if df is None:
                    raise last_error or Exception("Unknown error during text loading")
                    
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            elif ext in ['.parquet', '.pq']:
                df = pd.read_parquet(file_path)
            elif ext in ['.feather', '.ftr']:
                df = pd.read_feather(file_path)
            elif ext in ['.h5', '.hdf5']:
                df = pd.read_hdf(file_path)
            elif ext == '.dta':
                df = pd.read_stata(file_path)
            elif ext == '.sav':
                df = pd.read_spss(file_path)
            elif ext in ['.sas7bdat', '.xport']:
                df = pd.read_sas(file_path)
            else:
                supported_exts = ".csv, .txt, .tsv, .xlsx, .xls, .json, .parquet, .feather, .h5, .dta, .sav, .sas7bdat"
                QMessageBox.warning(self, "Unsupported File", 
                                     f"The file extension '{ext}' is not supported.\n"
                                     f"Supported: {supported_exts}")
                return 
            
            # Update Manager and Keep Original
            if df is not None:
                self._original_df = df.copy()
                self._set_internal_data(df, file_path)
            else:
                raise Exception("DataFrame remained None after loading attempt.")
             
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Load Error", 
                                 f"Failed to load data ({os.path.basename(file_path)}):\n{str(e)}\n\n"
                                 f"Details:\n{error_details[:500]}...")

    def _set_internal_data(self, df, path):
        try:
            self.dm.set_data(df, path)
        except Exception:
            self.dm.df = df
        
        self.update_display(path, df)
        self.data_loaded.emit(df)

    def load_data(self):
        filters = (
            "All Supported Files (*.csv *.txt *.tsv *.xlsx *.xls *.json *.parquet *.pq *.feather *.ftr *.h5 *.hdf5 *.dta *.sav *.sas7bdat *.xport);;"
            "CSV Files (*.csv);;"
            "Text Files (*.txt *.tsv);;"
            "Excel Files (*.xlsx *.xls);;"
            "JSON Files (*.json);;"
            "Parquet Files (*.parquet *.pq);;"
            "Feather Files (*.feather *.ftr);;"
            "HDF5 Files (*.h5 *.hdf5);;"
            "Stata Files (*.dta);;"
            "SPSS Files (*.sav);;"
            "SAS Files (*.sas7bdat *.xport);;"
            "All Files (*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", filters
        )
        if file_path:
            self._load_file_path(file_path)
            
    def on_create_column(self):
        if self.dm.df is None: return
        
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
        
        # Ask for query
        query, ok = QInputDialog.getText(self, "Filter Rows", "Query (e.g. GDP > 1000 and Age < 50):")
        if not ok or not query: return
        
        try:
            df = self.dm.df.query(query, engine='python')
            self._set_internal_data(df, "Filtered Data")
            QMessageBox.information(self, "Success", f"Filtered to {len(df)} rows.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to filter:\n{e}")

    def on_merge(self):
        if self.dm.df is None: return
        
        # 1. Select second file
        other_path, _ = QFileDialog.getOpenFileName(self, "Select File to Merge", "", "Data Files (*.csv *.txt *.tsv *.xlsx *.xls *.json *.parquet *.feather *.dta *.sav *.sas7bdat)")
        if not other_path: return
        
        try:
            # Load other df (reusing partial logic or just simple read_csv for now)
            # To be robust, we should probably extract a generic loader
            ext = os.path.splitext(other_path)[1].lower()
            if ext == '.csv': other_df = pd.read_csv(other_path)
            elif ext == '.xlsx': other_df = pd.read_excel(other_path)
            elif ext == '.json': other_df = pd.read_json(other_path)
            elif ext == '.parquet': other_df = pd.read_parquet(other_path)
            elif ext == '.dta': other_df = pd.read_stata(other_path)
            else:
                 QMessageBox.warning(self, "Unsupported", "Merge currently supports common formats.")
                 return
            
            # 2. Ask for key
            common_cols = list(set(self.dm.df.columns) & set(other_df.columns))
            if not common_cols:
                QMessageBox.warning(self, "Error", "No common columns found to join on.")
                return
            
            key, ok = QInputDialog.getItem(self, "Merge Data", "Select Join Key:", common_cols, 0, False)
            if not ok: return
            
            # 3. Ask for how
            how, ok = QInputDialog.getItem(self, "Merge Strategy", "Select Join Type:", ["inner", "left", "right", "outer"], 0, False)
            if not ok: return
            
            if self.dm.merge_data(other_df, on=key, how=how):
                self._set_internal_data(self.dm.df, "Merged Data")
                QMessageBox.information(self, "Success", f"Merged with {os.path.basename(other_path)} on '{key}'.")
            else:
                QMessageBox.critical(self, "Error", "Merge failed.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to merge:\n{e}")

    def on_reshape(self):
        if self.dm.df is None: return
        
        methods = ["melt (Wide to Long)", "pivot (Long to Wide)"]
        method_str, ok = QInputDialog.getItem(self, "Reshape Data", "Select Method:", methods, 0, False)
        if not ok: return
        
        try:
            if "melt" in method_str:
                id_vars, ok = QInputDialog.getText(self, "Melt", "ID Variables (comma separated):")
                if not ok: return
                val_vars, ok = QInputDialog.getText(self, "Melt", "Value Variables (comma separated, empty for all):")
                if not ok: return
                
                kwargs = {'id_vars': [x.strip() for x in id_vars.split(',') if x.strip()]}
                if val_vars.strip():
                    kwargs['value_vars'] = [x.strip() for x in val_vars.split(',') if x.strip()]
                
                if self.dm.reshape_data('melt', **kwargs):
                    self._set_internal_data(self.dm.df, "Reshaped Data (Melt)")
            else:
                idx, ok = QInputDialog.getItem(self, "Pivot", "Index Column:", self.dm.get_columns(), 0, False)
                if not ok: return
                cols, ok = QInputDialog.getItem(self, "Pivot", "Columns Column:", self.dm.get_columns(), 0, False)
                if not ok: return
                vals, ok = QInputDialog.getItem(self, "Pivot", "Values Column:", self.dm.get_columns(), 0, False)
                if not ok: return
                
                if self.dm.reshape_data('pivot', index=idx, columns=cols, values=vals):
                    self._set_internal_data(self.dm.df, "Reshaped Data (Pivot)")
                    
            QMessageBox.information(self, "Success", "Data reshaped.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reshape:\n{e}")

    def on_reset_data(self):
        if self._original_df is not None:
             self._set_internal_data(self._original_df.copy(), "Original Data")
        else:
             QMessageBox.warning(self, "Warning", "No original data to reset to.")

    def update_display(self, path, df: pd.DataFrame):
        filename = os.path.basename(path)
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
        except (OSError, TypeError):
            size_mb = 0.0
        
        self.file_label.setText(f"データセット: {filename}")
        self.info_label.setText(f"行: {df.shape[0]} | 列: {df.shape[1]} | サイズ: {size_mb:.2f}MB")
        
        # Update Model
        self.model.set_data(df)
