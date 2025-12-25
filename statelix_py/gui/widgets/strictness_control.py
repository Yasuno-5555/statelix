
import os
import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, QLabel, QFrame
)
from PySide6.QtCore import Signal, Qt
from statelix_py.diagnostics.presets import GovernanceMode

class StrictnessControlWidget(QWidget):
    mode_changed = Signal(GovernanceMode)
    
    def __init__(self, initial_mode=GovernanceMode.STRICT):
        super().__init__()
        self.current_mode = initial_mode
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.setStyleSheet("""
            StrictnessControlWidget {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        
        header = QLabel("ETHICAL GOVERNANCE")
        header.setStyleSheet("font-weight: bold; font-size: 10px; color: #666; letter-spacing: 1px;")
        layout.addWidget(header)
        
        btn_layout = QHBoxLayout()
        self.group = QButtonGroup(self)
        
        modes = [
            (GovernanceMode.STRICT, "STRICT (0.8)", "#4ec9b0"),
            (GovernanceMode.NORMAL, "NORMAL (0.5)", "#dcdcaa"),
            (GovernanceMode.EXPLORATORY, "EXPLORATORY", "#f44747")
        ]
        
        for mode, label, color in modes:
            rb = QRadioButton(label)
            rb.setStyleSheet(f"QRadioButton {{ color: #aaa; }} QRadioButton::indicator:checked {{ background-color: {color}; }}")
            if mode == self.current_mode:
                rb.setChecked(True)
            self.group.addButton(rb)
            btn_layout.addWidget(rb)
            
            # Map button to mode
            rb._mode = mode
            
        layout.addLayout(btn_layout)
        
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #f44747; font-size: 10px; font-style: italic;")
        layout.addWidget(self.warning_label)
        
        self.group.buttonClicked.connect(self.on_changed)
        self.update_warning()

    def on_changed(self, button):
        new_mode = button._mode
        if new_mode != self.current_mode:
            # Audit log if lowering
            old_level = list(GovernanceMode).index(self.current_mode)
            new_level = list(GovernanceMode).index(new_mode)
            
            if new_level > old_level: # Lowering strictness
                self.log_audit(self.current_mode, new_mode)
                
            self.current_mode = new_mode
            self.update_warning()
            self.mode_changed.emit(new_mode)

    def update_warning(self):
        if self.current_mode == GovernanceMode.STRICT:
            self.warning_label.setText("✓ Production standard enabled.")
            self.warning_label.setStyleSheet("color: #4ec9b0; font-size: 10px;")
        else:
            self.warning_label.setText("⚠ WARNING: Ethical guardrails loosened. Results may be unreliable.")
            self.warning_label.setStyleSheet("color: #f44747; font-size: 10px;")

    def log_audit(self, old_mode, new_mode):
        log_dir = os.path.expanduser("~/.statelix")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "audit.log")
        
        timestamp = datetime.datetime.now().isoformat()
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] GOVERNANCE_DEGRADED: {old_mode.name} -> {new_mode.name}\n")
