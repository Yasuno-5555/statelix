from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt

class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("結果パネル")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # 1. Summary Metrics
        metrics_frame = QFrame()
        metrics_frame.setFrameShape(QFrame.Shape.StyledPanel)
        metrics_layout = QHBoxLayout()
        
        self.status_label = QLabel("待機中...")
        self.r2_label = QLabel("R^2: -")
        self.mse_label = QLabel("MSE: -")
        
        metrics_layout.addWidget(self.status_label)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.r2_label)
        metrics_layout.addSpacing(20)
        metrics_layout.addWidget(self.mse_label)
        metrics_frame.setLayout(metrics_layout)
        layout.addWidget(metrics_frame)

        # 2. Results Content (Split View: Table/Text vs Graph)
        content_layout = QHBoxLayout()
        
        # Text/Table Result
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("実行結果がここに表示されます...")
        content_layout.addWidget(self.result_text, stretch=1)
        
        # Graph Placeholder
        self.graph_placeholder = QLabel("グラフ表示エリア\n(Matplotlib/Plotly)")
        self.graph_placeholder.setFrameShape(QFrame.Shape.Box)
        self.graph_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px dashed #999;")
        content_layout.addWidget(self.graph_placeholder, stretch=1)
        
        layout.addLayout(content_layout)

        # 3. Step Log
        log_label = QLabel("Step Log:")
        layout.addWidget(log_label)
        
        self.log_view = QTextEdit()
        self.log_view.setMaximumHeight(80)
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.setLayout(layout)

    def display_result(self, result_data: dict):
        success = result_data.get('success', True)
        
        if success:
            self.status_label.setText("成功: True")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("失敗")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        self.r2_label.setText(f"R^2: {result_data.get('r2', 'N/A')}")
        self.mse_label.setText(f"MSE: {result_data.get('mse', 'N/A')}")
        
        summary = result_data.get('summary', '')
        self.result_text.setText(summary)
        
        log_level = "[INFO]" if success else "[ERROR]"
        self.log_view.append(f"{log_level} Analysis completed. Hash: {result_data.get('hash', '???')}")
