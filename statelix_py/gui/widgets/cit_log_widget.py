
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea
from PySide6.QtCore import Qt

class CITDiscoveryWidget(QWidget):
    """
    Log-style widget for Counter-Intuitive Truth (CIT) discoveries.
    Displays 'Rigor Alarms' in Exploratory mode.
    """
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("CIT Discovery Log: RIGOR ALARMS")
        header.setStyleSheet("color: #eb9e34; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(header)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background: #1e1e1e;")
        
        self.container = QWidget()
        self.log_layout = QVBoxLayout(self.container)
        self.log_layout.addStretch()
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
        
    def update_discoveries(self, discoveries):
        # Clear existing
        while self.log_layout.count() > 1:
            item = self.log_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not discoveries:
            msg = QLabel("No active rigor alarms. The 'Truth' appears stable... for now.")
            msg.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.log_layout.insertWidget(0, msg)
            return

        for disc in discoveries:
            entry = CITEntry(disc)
            self.log_layout.insertWidget(self.log_layout.count()-1, entry)

class CITEntry(QFrame):
    def __init__(self, discovery):
        super().__init__()
        layout = QVBoxLayout(self)
        
        color = "#f44747" if discovery.severity == "Critical" else "#eb9e34"
        
        title = QLabel(f"!! {discovery.type} !!")
        title.setStyleSheet(f"color: {color}; font-weight: bold; font-family: monospace;")
        layout.addWidget(title)
        
        msg = QLabel(discovery.message)
        msg.setStyleSheet("color: #ccc; font-weight: bold;")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        
        proof = QLabel(f"Geometric Proof: {discovery.geometric_proof}")
        proof.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        proof.setWordWrap(True)
        layout.addWidget(proof)
        
        self.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {color}44;
                border-radius: 4px;
                background-color: #252526;
                padding: 10px;
                margin-bottom: 5px;
            }}
        """)
