
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QFrame, QScrollArea
)
from PySide6.QtCore import Signal, Qt

class SuggestionCard(QFrame):
    clicked = Signal(str)
    
    def __init__(self, suggestion):
        super().__init__()
        self.suggestion = suggestion
        self.init_ui()
        
    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            SuggestionCard {
                background-color: #252526;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 10px;
            }
            SuggestionCard:hover {
                border-color: #007acc;
                background-color: #2d2d30;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Suggestion Text
        text = QLabel(self.suggestion)
        text.setWordWrap(True)
        text.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(text)
        
        # Predicted Impact (Simulation)
        impact_layout = QHBoxLayout()
        
        # Heuristic impact display
        impacts = self.get_predicted_impact()
        for metric, change in impacts.items():
            color = "#4ec9b0" if change > 0 else "#f44747"
            symbol = "↑" if change > 0 else "↓"
            label = QLabel(f"{metric} {symbol}{abs(change)}")
            label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
            impact_layout.addWidget(label)
        
        impact_layout.addStretch()
        layout.addLayout(impact_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch() # Push buttons to the right

        self.btn_preview = QPushButton("Geometric Preview")
        self.btn_preview.setStyleSheet("""
            QPushButton { background-color: #1e1e1e; border: 1px solid #4ec9b0; border-radius: 4px; padding: 5px; color: #4ec9b0; }
            QPushButton:hover { background-color: #2e2e32; }
        """)
        self.btn_preview.clicked.connect(lambda: self.previewed.emit(self.suggestion))
        btn_layout.addWidget(self.btn_preview)

        self.btn_apply = QPushButton("Apply Strategy")
        self.btn_apply.setStyleSheet("""
            QPushButton { background-color: #3e3e42; border: 1px solid #555; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #4e4e52; }
        """)
        self.btn_apply.clicked.connect(lambda: self.applied.emit(self.suggestion))
        btn_layout.addWidget(self.btn_apply)
        
        layout.addLayout(btn_layout)


    def get_predicted_impact(self):
        # Heuristics for visualization
        s = self.suggestion.lower()
        impacts = {}
        if "interaction" in s:
            impacts["Fit"] = +0.15
        if "regularize" in s or "noise" in s:
            impacts["Topo"] = +0.2
            impacts["Fit"] = -0.05
        if "normalize" in s or "z-score" in s:
            impacts["Geo"] = +0.25
        return impacts

    def mousePressEvent(self, event):
        self.previewed.emit(self.suggestion)

class SuggestionNavigatorWidget(QWidget):
    suggestion_applied = Signal(str)
    suggestion_previewed = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0) # Keep original margins if not specified otherwise
        
        # Non-scrollable area for title
        header = QLabel("Active Refinement Candidates")
        header.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(header)
        
        self.scroll = QScrollArea() # Use self.scroll to match original structure
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;") # Keep original style
        self.scroll.setFrameShape(QFrame.Shape.NoFrame) # Add this from instruction
        
        self.container = QWidget()
        self.card_layout = QVBoxLayout(self.container) # Renamed from cards_layout
        self.card_layout.setSpacing(8) # Keep original spacing
        self.card_layout.addStretch()
        
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
        # self.setLayout(layout) # Not needed if QVBoxLayout(self) is used

    def update_suggestions(self, suggestions):
        # Clear existing
        while self.card_layout.count() > 1: # Use card_layout
            item = self.card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not suggestions:
            label = QLabel("No suggestions needed. Model meets all contracts.")
            label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
            self.card_layout.insertWidget(0, label) # Use card_layout
            return

        for sugg in suggestions:
            card = SuggestionCard(sugg)
            self.cards_layout.insertWidget(self.cards_layout.count()-1, card)
