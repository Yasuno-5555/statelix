
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame
from PySide6.QtCore import Qt, Signal

class NextActionWidget(QFrame):
    """
    A high-visibility banner for the 'Sole Next Action' after rejection.
    """
    action_clicked = Signal(str) # Emits the description of the action
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.hide()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        header = QLabel("→ Statelix Suggests Next Step:")
        header.setStyleSheet("color: #eb9e34; font-weight: bold; font-size: 11px;")
        layout.addWidget(header)
        
        content_row = QHBoxLayout()
        
        self.action_btn = QPushButton("Stabilize Manifold")
        self.action_btn.setStyleSheet("""
            QPushButton {
                 background-color: #eb9e34;
                 color: #1e1e1e;
                 font-weight: bold;
                 border-radius: 4px;
                 padding: 10px;
                 font-size: 14px;
            }
            QPushButton:hover {
                 background-color: #fbaf44;
            }
        """)
        self.action_btn.clicked.connect(self.on_btn_clicked)
        content_row.addWidget(self.action_btn, stretch=1)
        
        self.desc_label = QLabel("Increase regularization to handle noise.")
        self.desc_label.setStyleSheet("color: #ccc; font-style: italic; margin-left: 10px;")
        self.desc_label.setWordWrap(True)
        content_row.addWidget(self.desc_label, stretch=2)
        
        layout.addLayout(content_row)
        
        self.setStyleSheet("""
            QFrame {
                background-color: #2d2d30;
                border: 2px solid #eb9e34;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
        """)

    def set_action(self, action_data):
        if not action_data:
            self.hide()
            return
        
        self.action_btn.setText(action_data['action'])
        self.desc_label.setText(action_data['desc'])
        self.show()

    def on_btn_clicked(self):
        self.action_clicked.emit(self.desc_label.text())
