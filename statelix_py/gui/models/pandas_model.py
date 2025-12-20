from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

class PandasModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data

    def set_data(self, data):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

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
