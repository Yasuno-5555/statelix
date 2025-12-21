from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
import pandas as pd
import numpy as np

class PandasModel(QAbstractTableModel):
    """Enhanced Pandas Model with Excel-like row/column operations."""
    
    def __init__(self, data=None):
        super().__init__()
        self._data = data

    def set_data(self, data):
        self.beginResetModel()
        self._data = data
        self.endResetModel()
    
    def get_data(self):
        """Return the underlying DataFrame."""
        return self._data
    
    # --- Row Operations ---
    def insert_row(self, row: int):
        """Insert a new empty row at the specified position."""
        if self._data is None:
            return False
        
        self.beginInsertRows(QModelIndex(), row, row)
        
        # Create empty row with same columns
        new_row = pd.DataFrame([[np.nan] * len(self._data.columns)], 
                               columns=self._data.columns)
        
        # Split and concatenate
        if row == 0:
            self._data = pd.concat([new_row, self._data], ignore_index=True)
        elif row >= len(self._data):
            self._data = pd.concat([self._data, new_row], ignore_index=True)
        else:
            top = self._data.iloc[:row]
            bottom = self._data.iloc[row:]
            self._data = pd.concat([top, new_row, bottom], ignore_index=True)
        
        self.endInsertRows()
        return True
    
    def delete_row(self, row: int):
        """Delete the row at the specified position."""
        if self._data is None or row < 0 or row >= len(self._data):
            return False
        
        self.beginRemoveRows(QModelIndex(), row, row)
        self._data = self._data.drop(self._data.index[row]).reset_index(drop=True)
        self.endRemoveRows()
        return True
    
    # --- Column Operations ---
    def insert_column(self, col: int, name: str = None):
        """Insert a new empty column at the specified position."""
        if self._data is None:
            return False
        
        self.beginInsertColumns(QModelIndex(), col, col)
        
        # Generate column name if not provided
        if name is None:
            # Use Excel-style naming (NewCol1, NewCol2, etc.)
            existing = set(self._data.columns)
            i = 1
            while f"NewCol{i}" in existing:
                i += 1
            name = f"NewCol{i}"
        
        # Insert column with NaN values
        new_col = pd.Series([np.nan] * len(self._data), name=name)
        
        # Insert at position
        cols = list(self._data.columns)
        if col <= 0:
            self._data.insert(0, name, new_col)
        elif col >= len(cols):
            self._data[name] = new_col
        else:
            self._data.insert(col, name, new_col)
        
        self.endInsertColumns()
        return True
    
    def delete_column(self, col: int):
        """Delete the column at the specified position."""
        if self._data is None or col < 0 or col >= len(self._data.columns):
            return False
        
        self.beginRemoveColumns(QModelIndex(), col, col)
        col_name = self._data.columns[col]
        self._data = self._data.drop(columns=[col_name])
        self.endRemoveColumns()
        return True

    def rowCount(self, parent=QModelIndex()):
        if self._data is None:
            return 0
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        if self._data is None:
            return 0
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._data is None:
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            val = self._data.iloc[index.row(), index.column()]
            return str(val)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if self._data is None:
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable

    def sort(self, column, order):
        if self._data is None: return
        
        col_name = self._data.columns[column]
        ascending = (order == Qt.SortOrder.AscendingOrder)
        
        self.layoutAboutToBeChanged.emit()
        try:
            self._data.sort_values(by=col_name, ascending=ascending, inplace=True)
            self._data.reset_index(drop=True, inplace=True)
        except Exception:
            pass # sorting failed (e.g. mixed types)
        self.layoutChanged.emit()

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        try:
            row = index.row()
            col = index.column()
            
            # Smart Type Conversion
            current_val = self._data.iloc[row, col]
            # Try to infer type from current value or column dtype
            import pandas as pd
            import numpy as np
            
            # Attempt conversion
            if value == "":
                val = np.nan
            else:
                try:
                    # Try numeric first if column is numeric
                    if pd.api.types.is_numeric_dtype(self._data.iloc[:, col]):
                        if '.' in value:
                            val = float(value)
                        else:
                            val = int(value)
                    else:
                        val = value
                except ValueError:
                    val = value # Fallback to string
                    
            self._data.iloc[row, col] = val
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            return True
        except Exception:
            return False
