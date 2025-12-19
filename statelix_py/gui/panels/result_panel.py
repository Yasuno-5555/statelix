import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QFrame,
    QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

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
        text_layout = QVBoxLayout()
        
        # Tools
        tools = QHBoxLayout()
        tools.addStretch()
        btn_md = QLabel("<a href='#'>Copy Markdown</a>"); btn_md.setOpenExternalLinks(False)
        btn_tex = QLabel("<a href='#'>Copy LaTeX</a>"); btn_tex.setOpenExternalLinks(False)
        
        # Make them look like buttons or just use buttons
        self.btn_copy_md = QPushButton("Copy Markdown")
        self.btn_copy_md.clicked.connect(self.on_copy_markdown)
        self.btn_copy_tex = QPushButton("Copy LaTeX")
        self.btn_copy_tex.clicked.connect(self.on_copy_latex)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_excel = QPushButton("Export Excel")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_report = QPushButton("Generate Report")
        self.btn_report.clicked.connect(self.on_generate_report)
        
        tools.addWidget(self.btn_copy_md)
        tools.addWidget(self.btn_copy_tex)
        tools.addWidget(self.btn_export_csv)
        tools.addWidget(self.btn_export_excel)
        tools.addWidget(self.btn_report)
        
        text_layout.addLayout(tools)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("実行結果がここに表示されます...")
        text_layout.addWidget(self.result_text)
        
        content_layout.addLayout(text_layout, stretch=1)
        
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
        self._last_result = result_data # Store for export
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
        # If there's a table in result, format it
        if 'table' in result_data:
            df = result_data['table']
            if isinstance(df, pd.DataFrame):
                summary += "\n\n" + df.to_string()
        
        self.result_text.setText(summary)
        
        log_level = "[INFO]" if success else "[ERROR]"
        self.log_view.append(f"{log_level} Analysis completed. Hash: {result_data.get('hash', '???')}")

    def on_copy_markdown(self):
        if not hasattr(self, '_last_result'): return
        res = self._last_result
        if 'table' in res and isinstance(res['table'], pd.DataFrame):
            text = res['table'].to_markdown()
        else:
            text = "```\n" + self.result_text.toPlainText() + "\n```"
        
        from PySide6.QtWidgets import QApplication, QMessageBox
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copied", "Result copied as Markdown.")

    def on_copy_latex(self):
        if not hasattr(self, '_last_result'): return
        res = self._last_result
        if 'table' in res and isinstance(res['table'], pd.DataFrame):
            text = res['table'].to_latex()
        else:
            text = "\\begin{verbatim}\n" + self.result_text.toPlainText() + "\n\\end{verbatim}"
        
        from PySide6.QtWidgets import QApplication, QMessageBox
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copied", "Result copied as LaTeX.")

    def on_export_csv(self):
        if not hasattr(self, '_last_result') or 'table' not in self._last_result:
            QMessageBox.warning(self, "Warning", "No table data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            self._last_result['table'].to_csv(path, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {path}")

    def on_export_excel(self):
        if not hasattr(self, '_last_result') or 'table' not in self._last_result:
            QMessageBox.warning(self, "Warning", "No table data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "", "Excel Files (*.xlsx)")
        if path:
            self._last_result['table'].to_excel(path, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {path}")

    def on_generate_report(self):
        from statelix_py.utils.report_generator import ReportGenerator
        from statelix_py.core.data_manager import DataManager
        
        dm = DataManager.instance()
        report = ReportGenerator("Statelix Analysis Report")
        
        # Add descriptive stats if data available
        if dm.df is not None:
            report.add_descriptive_stats(dm.df)
            report.add_correlation_matrix(dm.df)
        
        # Add last result
        if hasattr(self, '_last_result'):
            summary = self._last_result.get('summary', '')
            table = self._last_result.get('table', None)
            report.add_model_result("Analysis Result", summary, table)
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "HTML Files (*.html)")
        if path:
            report.save(path)
            QMessageBox.information(self, "Success", f"Report saved to {path}")
