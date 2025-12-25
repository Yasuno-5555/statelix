
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QLineEdit, QPushButton, QLabel, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot

class AnalystChatWidget(QWidget):
    """
    Chat interface for interacting with the Statelix Analyst.
    """
    def __init__(self):
        super().__init__()
        self.engine = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Chat History
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()
        
        self.scroll.setWidget(self.chat_container)
        layout.addWidget(self.scroll, stretch=1)
        
        # Suggestions Sidebar / Quick Buttons
        btn_row = QHBoxLayout()
        self.btns = []
        for label in ["Why reject?", "How to fix?", "Stats facts", "Assumptions"]:
             btn = QPushButton(label)
             btn.setStyleSheet("font-size: 10px; color: #4ec9b0; border: 1px solid #4ec9b0; border-radius: 10px; padding: 2px 10px;")
             btn.clicked.connect(lambda checked, l=label: self.send_query(l))
             btn_row.addWidget(btn)
             self.btns.append(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        
        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask the Analyst about this model...")
        self.input_field.returnPressed.connect(self.on_send_clicked)
        input_layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("Ask")
        self.send_btn.clicked.connect(self.on_send_clicked)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout)

    def set_context(self, report, summary):
        from statelix_py.diagnostics.analyst_engine import AnalystEngine
        self.engine = AnalystEngine(report, summary)
        self.add_message("Statelix Analyst", "I have finished auditing your model. How can I help you interpret these results?", is_ai=True)

    def add_message(self, sender, text, is_ai=False):
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        
        name = QLabel(sender)
        name.setStyleSheet("font-weight: bold; color: #888; font-size: 10px;")
        bubble_layout.addWidget(name)
        
        content = QLabel(text)
        content.setWordWrap(True)
        content.setStyleSheet("color: #ccc;" if not is_ai else "color: #fff;")
        bubble_layout.addWidget(content)
        
        color = "#2d2d30" if not is_ai else "#3e3e42"
        border = "1px solid #444" if not is_ai else "1px solid #4ec9b0"
        
        bubble.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border: {border};
                border-radius: 8px;
                padding: 5px;
                margin: 5px;
            }}
        """)
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        # Scroll to bottom
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def on_send_clicked(self):
        query = self.input_field.text()
        if query:
            self.send_query(query)
            self.input_field.clear()

    def send_query(self, query):
        if not self.engine:
            return
        self.add_message("You", query, is_ai=False)
        answer = self.engine.answer(query)
        self.add_message("Statelix Analyst", answer, is_ai=True)
