
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt

class ContractItem(QFrame):
    def __init__(self, name, description, depends_on=None):
        super().__init__()
        self.name = name
        self.description = description
        self.depends_on = depends_on
        self.init_ui()
        
    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setToolTip(self.description)
        self.setStyleSheet("""
            ContractItem {
                background-color: #2d2d30;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        layout = QHBoxLayout(self)
        
        self.status_icon = QLabel("❓")
        layout.addWidget(self.status_icon)
        
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(self.name_label, stretch=1)
        
        if self.depends_on:
            dep_label = QLabel(f"→ {self.depends_on}")
            dep_label.setStyleSheet("color: #666; font-size: 10px;")
            layout.addWidget(dep_label)

    def set_status(self, met: bool):
        if met:
            self.status_icon.setText("✅")
            self.setStyleSheet("ContractItem { background-color: #1e3a2f; border: 1px solid #4ec9b0; }")
        else:
            self.status_icon.setText("❌")
            self.setStyleSheet("ContractItem { background-color: #3e2723; border: 1px solid #f44747; }")

class ContractViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        self.items = {
            "Identifiability": ContractItem("Identifiability", "Does the model have a unique solution?", "Data Matrix Rank"),
            "Stability": ContractItem("Stability", "Is the manifold persistent?", "Regularization"),
            "Topology": ContractItem("Topology", "Is the structural structure intact?", "Feature Space"),
            "Geometric Bound": ContractItem("Geometric Bound", "Are coordinate transitions invariant?", "Standardization")
        }
        
        for item in self.items.values():
            layout.addWidget(item)
        layout.addStretch()

    def update_status(self, mci_score_obj):
        # Map MCI scores to contract status (Simplified thresholds)
        self.items["Identifiability"].set_status(mci_score_obj.fit_score > 0.4)
        self.items["Stability"].set_status(mci_score_obj.topology_score > 0.6)
        self.items["Topology"].set_status(mci_score_obj.topology_score > 0.8)
        self.items["Geometric Bound"].set_status(mci_score_obj.geometry_score > 0.7)
