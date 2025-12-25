
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QListWidget, QPushButton, QFrame, QScrollArea, QGroupBox
)
from PySide6.QtCore import Qt, Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class MCIGauge(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(5,5,5,5)
        
        self.score_label = QLabel("MCI: --")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888;")
        layout.addWidget(self.score_label)
        
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(20)
        layout.addWidget(self.bar)
        
        self.status_label = QLabel("WAITING")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def set_score(self, score: float):
        percent = int(score * 100)
        self.bar.setValue(percent)
        self.score_label.setText(f"MCI: {score:.2f}")
        
        if score > 0.8:
            color = "#4ec9b0" # Green
            text = "TRUSTWORTHY"
        elif score > 0.5:
            color = "#dcdcaa" # Yellow
            text = "WARNING"
        else:
            color = "#f44747" # Red
            text = "REJECTED"
            
        self.score_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; letter-spacing: 2px;")
        
        # ProgressBar styling
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #252526;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

class DiagnosticsPanel(QWidget):
    suggestion_action = Signal(str) # Emits suggestion text when clicked
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        
        # 1. MCI Judge
        self.mci_gauge = MCIGauge()
        main_layout.addWidget(self.mci_gauge)
        
        # 2. Objections & Suggestions Area
        details_layout = QHBoxLayout()
        
        # Left: Objections
        obj_group = QGroupBox("Objections (Why Statelix complains)")
        obj_layout = QVBoxLayout()
        self.obj_list = QListWidget()
        self.obj_list.setStyleSheet("color: #f44747; border: none; background: transparent;")
        obj_layout.addWidget(self.obj_list)
        obj_group.setLayout(obj_layout)
        details_layout.addWidget(obj_group, stretch=1)
        
        # Right: Suggestions (Actionable)
        sugg_group = QGroupBox("Suggestions (Fix it)")
        self.sugg_layout = QVBoxLayout()
        self.sugg_scroll = QScrollArea()
        self.sugg_scroll.setWidgetResizable(True)
        self.sugg_container = QWidget()
        self.sugg_container.setLayout(self.sugg_layout)
        self.sugg_scroll.setWidget(self.sugg_container)
        
        # Wrap scroll in layout
        sugg_wrap = QVBoxLayout()
        sugg_wrap.addWidget(self.sugg_scroll)
        sugg_group.setLayout(sugg_wrap)
        details_layout.addWidget(sugg_group, stretch=1)
        
        main_layout.addLayout(details_layout)
        
        # 3. Timeline (History)
        hist_group = QGroupBox("Diagnostic History")
        hist_layout = QVBoxLayout()
        self.hist_view = QListWidget()
        self.hist_view.setMaximumHeight(100)
        self.hist_view.setStyleSheet("font-family: monospace;")
        hist_layout.addWidget(self.hist_view)
        hist_group.setLayout(hist_layout)
        main_layout.addWidget(hist_group)
        
        # 4. Integrity Statement
        integrity = QLabel("Statelix does not guarantee correctness. It guarantees refusal to lie.")
        integrity.setAlignment(Qt.AlignmentFlag.AlignCenter)
        integrity.setStyleSheet("color: #555; font-style: italic; font-size: 11px;")
        main_layout.addWidget(integrity)
        
        self.setLayout(main_layout)
        
    def update_diagnostics(self, mci: float, objections: list, suggestions: list, history: list):
        # 1. MCI
        self.mci_gauge.set_score(mci)
        
        # 2. Objections
        self.obj_list.clear()
        if objections:
            for obj in objections:
                self.obj_list.addItem(f"• {obj}")
        else:
            self.obj_list.addItem("None. Model is healthy.")
            
        # 3. Suggestions (Buttons)
        # Clear old buttons
        while self.sugg_layout.count():
            item = self.sugg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if suggestions:
            for sugg in suggestions:
                btn = QPushButton(f"Apply: {sugg}")
                btn.setStyleSheet("text-align: left; padding: 5px;")
                btn.clicked.connect(lambda checked, s=sugg: self.suggestion_action.emit(s))
                self.sugg_layout.addWidget(btn)
        else:
            self.sugg_layout.addWidget(QLabel("(No suggestions)"))
        self.sugg_layout.addStretch()
        
        # 4. History
        self.hist_view.clear()
        for i, item in enumerate(history):
            # item is dict from DiagnosticHistory.get_evolution()
            self.hist_view.addItem(
                f"[Iter {item['iteration']}] MCI: {item['mci']:.2f} | Objections: {item['objections_count']}"
            )
        self.hist_view.scrollToBottom()
