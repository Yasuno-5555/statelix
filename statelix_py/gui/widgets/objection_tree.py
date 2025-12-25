
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QHeaderView
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush

class ObjectionTreeWidget(QTreeWidget):
    """
    Tree-structured display for ModelCritic objections.
    """
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setColumnCount(3)
        self.setHeaderLabels(["Objection", "Detail", "Time/Iter"])
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        self.setStyleSheet("""
            QTreeWidget {
                background-color: #1a1a1a;
                color: #ccc;
                border: 1px solid #333;
                font-family: 'Segoe UI', sans-serif;
            }
            QHeaderView::section {
                background-color: #333;
                color: #aaa;
                padding: 4px;
                border: 1px solid #222;
            }
        """)

    def update_objections(self, messages, critic=None):
        self.clear()
        if not messages:
            root = QTreeWidgetItem(["✅ No Objections Found"])
            root.setForeground(0, QBrush(QColor("#4ec9b0")))
            self.addTopLevelItem(root)
            return

        categories = {}
        
        for msg in messages:
            cat_name = "Objections"
            if critic:
                cat_name = critic.categorize_objection(msg)
            
            if cat_name not in categories:
                cat_item = QTreeWidgetItem([cat_name])
                cat_item.setExpanded(True)
                # Set color based on category
                color = "#dcdcaa" # Default Yellow
                if "Critical" in cat_name: color = "#f44747"
                elif "Fit" in cat_name: color = "#569cd6"
                
                cat_item.setForeground(0, QBrush(QColor(color)))
                self.addTopLevelItem(cat_item)
                categories[cat_name] = cat_item
            
            # Add specific objection
            # Message formatting: "Title: Detail"
            parts = msg.split(": ", 1)
            title = parts[0]
            detail = parts[1] if len(parts) > 1 else ""
            
            item = QTreeWidgetItem([title, detail, "-"])
            categories[cat_name].addChild(item)

    def set_historical_objections(self, iteration, messages, critic=None):
        """Display objections for a specific historical iteration."""
        self.update_objections(messages, critic)
        # Update iteration column for all leaf nodes
        root_count = self.topLevelItemCount()
        for i in range(root_count):
            root = self.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                child.setText(2, str(iteration))
