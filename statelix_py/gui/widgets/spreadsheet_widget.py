"""
Excel-like Spreadsheet Widget for Statelix
Provides formula bar, status bar, and enhanced table view with Excel-like features.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QLineEdit, QLabel,
    QFrame, QHeaderView, QMenu, QAbstractItemView, QPushButton,
    QStyledItemDelegate, QStyleOptionViewItem
)
from PySide6.QtCore import Qt, Signal, QModelIndex, QItemSelectionModel
from PySide6.QtGui import QKeySequence, QShortcut, QPainter, QColor, QPen
import numpy as np


class ExcelStyleDelegate(QStyledItemDelegate):
    """Custom delegate for Excel-like cell rendering."""
    
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        # Draw cell with grid lines
        painter.save()
        
        # Fill background
        if option.state & QStyleOptionViewItem.State_Selected:
            painter.fillRect(option.rect, QColor("#0078d4"))
        else:
            row = index.row()
            if row % 2 == 0:
                painter.fillRect(option.rect, QColor("#252526"))
            else:
                painter.fillRect(option.rect, QColor("#2d2d30"))
        
        # Draw text
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text:
            if option.state & QStyleOptionViewItem.State_Selected:
                painter.setPen(QColor("#ffffff"))
            else:
                painter.setPen(QColor("#d4d4d4"))
            painter.drawText(option.rect.adjusted(4, 0, -4, 0), 
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                           str(text))
        
        # Draw grid border
        painter.setPen(QPen(QColor("#3e3e42"), 1))
        painter.drawRect(option.rect.adjusted(0, 0, -1, -1))
        
        painter.restore()


class ExcelTableView(QTableView):
    """Enhanced QTableView with Excel-like behavior."""
    
    cell_selected = Signal(int, int, str)  # row, col, value
    selection_changed = Signal(list)  # list of selected values for stats
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_excel_behavior()
        
    def setup_excel_behavior(self):
        # Visual settings
        self.setAlternatingRowColors(False)  # Handled by delegate
        self.setItemDelegate(ExcelStyleDelegate())
        self.setShowGrid(True)
        self.setGridStyle(Qt.PenStyle.SolidLine)
        
        # Selection behavior
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Header behavior
        h_header = self.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h_header.setStretchLastSection(True)
        h_header.setDefaultSectionSize(100)
        h_header.setMinimumSectionSize(30)
        
        v_header = self.verticalHeader()
        v_header.setDefaultSectionSize(25)
        v_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Keyboard navigation
        self.setTabKeyNavigation(True)
        
        # Stylesheets
        self.setStyleSheet("""
            QTableView {
                background-color: #1e1e1e;
                gridline-color: #3e3e42;
                border: 1px solid #3e3e42;
            }
            QTableView::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QTableView::item:focus {
                border: 2px solid #0098ff;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #d4d4d4;
                padding: 4px;
                border: 1px solid #3e3e42;
                font-weight: bold;
            }
            QHeaderView::section:hover {
                background-color: #3e3e42;
            }
        """)
    
    def selectionChanged(self, selected, deselected):
        super().selectionChanged(selected, deselected)
        
        # Emit selected cell info
        indexes = self.selectionModel().selectedIndexes()
        if indexes:
            idx = indexes[0]
            value = idx.data(Qt.ItemDataRole.DisplayRole)
            self.cell_selected.emit(idx.row(), idx.column(), str(value) if value else "")
        
        # Collect all selected values for statistics
        values = []
        for idx in indexes:
            val = idx.data(Qt.ItemDataRole.DisplayRole)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        self.selection_changed.emit(values)
    
    def keyPressEvent(self, event):
        """Handle Excel-like keyboard navigation."""
        key = event.key()
        modifiers = event.modifiers()
        
        current = self.currentIndex()
        
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if self.state() == QAbstractItemView.State.EditingState:
                # Commit edit and move down
                self.commitData(self.indexWidget(current))
                self.closeEditor(self.indexWidget(current), QAbstractItemDelegate.EndEditHint.NoHint)
            # Move to next row
            next_idx = self.model().index(current.row() + 1, current.column())
            if next_idx.isValid():
                self.setCurrentIndex(next_idx)
        
        elif key == Qt.Key.Key_Tab:
            # Move to next column
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                next_idx = self.model().index(current.row(), current.column() - 1)
            else:
                next_idx = self.model().index(current.row(), current.column() + 1)
            if next_idx.isValid():
                self.setCurrentIndex(next_idx)
                event.accept()
                return
        
        elif key == Qt.Key.Key_F2:
            # Start editing
            self.edit(current)
        
        elif key == Qt.Key.Key_Delete:
            # Clear selected cells
            for idx in self.selectionModel().selectedIndexes():
                self.model().setData(idx, "", Qt.ItemDataRole.EditRole)
        
        else:
            super().keyPressEvent(event)


class FormulaBar(QFrame):
    """Excel-style formula bar showing cell reference and value."""
    
    value_edited = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(5)
        
        # Cell reference label (e.g., "A1")
        self.cell_ref_label = QLabel("A1")
        self.cell_ref_label.setFixedWidth(60)
        self.cell_ref_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                padding: 4px 8px;
                font-family: monospace;
                font-weight: bold;
                color: #d4d4d4;
            }
        """)
        
        # Separator
        sep = QLabel("𝑓𝑥")
        sep.setStyleSheet("color: #007acc; font-weight: bold; padding: 0 5px;")
        
        # Formula/value input
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("値を入力...")
        self.value_edit.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                border: 1px solid #3e3e42;
                padding: 4px 8px;
                color: #d4d4d4;
                font-family: monospace;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
        """)
        self.value_edit.returnPressed.connect(self._on_value_edited)
        
        layout.addWidget(self.cell_ref_label)
        layout.addWidget(sep)
        layout.addWidget(self.value_edit, 1)
        
        self.setStyleSheet("""
            FormulaBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
            }
        """)
    
    def set_cell(self, row: int, col: int, value: str):
        """Update formula bar with current cell info."""
        col_letter = self._col_to_excel(col)
        self.cell_ref_label.setText(f"{col_letter}{row + 1}")
        self.value_edit.setText(value)
    
    def _col_to_excel(self, col: int) -> str:
        """Convert column index to Excel letter (0->A, 1->B, 26->AA)."""
        result = ""
        while col >= 0:
            result = chr(col % 26 + ord('A')) + result
            col = col // 26 - 1
        return result
    
    def _on_value_edited(self):
        self.value_edited.emit(self.value_edit.text())


class StatusBar(QFrame):
    """Excel-style status bar showing selection statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setSpacing(20)
        
        # Statistics labels
        self.count_label = QLabel("カウント: -")
        self.sum_label = QLabel("合計: -")
        self.avg_label = QLabel("平均: -")
        self.min_label = QLabel("最小: -")
        self.max_label = QLabel("最大: -")
        
        for label in [self.count_label, self.sum_label, self.avg_label, 
                      self.min_label, self.max_label]:
            label.setStyleSheet("color: #888888; font-size: 9pt;")
        
        layout.addStretch()
        layout.addWidget(self.count_label)
        layout.addWidget(self.sum_label)
        layout.addWidget(self.avg_label)
        layout.addWidget(self.min_label)
        layout.addWidget(self.max_label)
        
        self.setStyleSheet("""
            StatusBar {
                background-color: #007acc;
                border-top: 1px solid #005a9e;
            }
            StatusBar QLabel {
                color: #ffffff;
            }
        """)
    
    def update_stats(self, values: list):
        """Update statistics from selected values."""
        if not values:
            self.count_label.setText("カウント: -")
            self.sum_label.setText("合計: -")
            self.avg_label.setText("平均: -")
            self.min_label.setText("最小: -")
            self.max_label.setText("最大: -")
            return
        
        try:
            arr = np.array(values)
            self.count_label.setText(f"カウント: {len(arr)}")
            self.sum_label.setText(f"合計: {np.sum(arr):.4g}")
            self.avg_label.setText(f"平均: {np.mean(arr):.4g}")
            self.min_label.setText(f"最小: {np.min(arr):.4g}")
            self.max_label.setText(f"最大: {np.max(arr):.4g}")
        except Exception:
            self.count_label.setText(f"カウント: {len(values)}")
            self.sum_label.setText("合計: N/A")
            self.avg_label.setText("平均: N/A")
            self.min_label.setText("最小: N/A")
            self.max_label.setText("最大: N/A")


class SpreadsheetWidget(QWidget):
    """
    Complete Excel-like spreadsheet widget combining:
    - Formula bar
    - Enhanced table view
    - Status bar with statistics
    """
    
    data_modified = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self._current_cell = (0, 0)
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Formula bar
        self.formula_bar = FormulaBar()
        layout.addWidget(self.formula_bar)
        
        # Table view
        self.table_view = ExcelTableView()
        layout.addWidget(self.table_view, 1)
        
        # Status bar
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        # Context menu
        self.table_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self._show_context_menu)
    
    def setup_connections(self):
        self.table_view.cell_selected.connect(self._on_cell_selected)
        self.table_view.selection_changed.connect(self.status_bar.update_stats)
        self.formula_bar.value_edited.connect(self._on_formula_edited)
    
    def set_model(self, model):
        """Set the data model for the table."""
        self.model = model
        self.table_view.setModel(model)
    
    def _on_cell_selected(self, row: int, col: int, value: str):
        """Handle cell selection."""
        self._current_cell = (row, col)
        self.formula_bar.set_cell(row, col, value)
    
    def _on_formula_edited(self, value: str):
        """Handle value edit from formula bar."""
        if self.model is None:
            return
        
        row, col = self._current_cell
        idx = self.model.index(row, col)
        if idx.isValid():
            self.model.setData(idx, value, Qt.ItemDataRole.EditRole)
            self.data_modified.emit()
    
    def _show_context_menu(self, pos):
        """Show Excel-like context menu."""
        if self.model is None:
            return
        
        index = self.table_view.indexAt(pos)
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #252526;
                border: 1px solid #3e3e42;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 30px 8px 20px;
                color: #d4d4d4;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QMenu::separator {
                height: 1px;
                background: #3e3e42;
                margin: 5px 10px;
            }
        """)
        
        # Cut, Copy, Paste
        cut_action = menu.addAction("✂️ 切り取り")
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        
        copy_action = menu.addAction("📋 コピー")
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        
        paste_action = menu.addAction("📥 貼り付け")
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        
        menu.addSeparator()
        
        # Row operations
        row_menu = menu.addMenu("📊 行の操作")
        insert_row_above = row_menu.addAction("上に行を挿入")
        insert_row_below = row_menu.addAction("下に行を挿入")
        delete_row = row_menu.addAction("行を削除")
        
        # Column operations  
        col_menu = menu.addMenu("📏 列の操作")
        insert_col_left = col_menu.addAction("左に列を挿入")
        insert_col_right = col_menu.addAction("右に列を挿入")
        delete_col = col_menu.addAction("列を削除")
        
        menu.addSeparator()
        
        # Clear contents
        clear_action = menu.addAction("🗑️ クリア")
        
        # Execute
        action = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        
        if action == insert_row_above:
            self._insert_row(index.row())
        elif action == insert_row_below:
            self._insert_row(index.row() + 1)
        elif action == delete_row:
            self._delete_row(index.row())
        elif action == insert_col_left:
            self._insert_column(index.column())
        elif action == insert_col_right:
            self._insert_column(index.column() + 1)
        elif action == delete_col:
            self._delete_column(index.column())
        elif action == clear_action:
            self._clear_selection()
        elif action == copy_action:
            self._copy_selection()
        elif action == paste_action:
            self._paste_selection()
    
    def _insert_row(self, row: int):
        """Insert a new row at the specified position."""
        if hasattr(self.model, 'insert_row'):
            self.model.insert_row(row)
            self.data_modified.emit()
    
    def _delete_row(self, row: int):
        """Delete the specified row."""
        if hasattr(self.model, 'delete_row'):
            self.model.delete_row(row)
            self.data_modified.emit()
    
    def _insert_column(self, col: int):
        """Insert a new column at the specified position."""
        if hasattr(self.model, 'insert_column'):
            self.model.insert_column(col)
            self.data_modified.emit()
    
    def _delete_column(self, col: int):
        """Delete the specified column."""
        if hasattr(self.model, 'delete_column'):
            self.model.delete_column(col)
            self.data_modified.emit()
    
    def _clear_selection(self):
        """Clear contents of selected cells."""
        for idx in self.table_view.selectionModel().selectedIndexes():
            self.model.setData(idx, "", Qt.ItemDataRole.EditRole)
        self.data_modified.emit()
    
    def _copy_selection(self):
        """Copy selected cells to clipboard."""
        from PySide6.QtWidgets import QApplication
        
        selection = self.table_view.selectionModel()
        if not selection.hasSelection():
            return
        
        indexes = selection.selectedIndexes()
        if not indexes:
            return
        
        # Sort by row, then column
        indexes.sort(key=lambda x: (x.row(), x.column()))
        
        rows = sorted(set(idx.row() for idx in indexes))
        cols = sorted(set(idx.column() for idx in indexes))
        
        text = ""
        for r in rows:
            row_data = []
            for c in cols:
                idx = self.model.index(r, c)
                val = idx.data(Qt.ItemDataRole.DisplayRole)
                row_data.append(str(val) if val else "")
            text += "\t".join(row_data) + "\n"
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
    
    def _paste_selection(self):
        """Paste from clipboard to selected cells."""
        from PySide6.QtWidgets import QApplication
        
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text:
            return
        
        # Get starting position
        selection = self.table_view.selectionModel()
        if selection.hasSelection():
            indexes = selection.selectedIndexes()
            start_row = min(idx.row() for idx in indexes)
            start_col = min(idx.column() for idx in indexes)
        else:
            start_row = 0
            start_col = 0
        
        # Parse clipboard data
        rows_data = [row.split('\t') for row in text.strip().split('\n')]
        
        for i, row_vals in enumerate(rows_data):
            for j, val in enumerate(row_vals):
                r = start_row + i
                c = start_col + j
                idx = self.model.index(r, c)
                if idx.isValid():
                    self.model.setData(idx, val, Qt.ItemDataRole.EditRole)
        
        self.data_modified.emit()
