import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QFrame,
    QPushButton, QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt
from statelix_py.gui.i18n import t

class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel(t("panel.result"))
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #007acc;")
        layout.addWidget(title)

        # 1. Summary Metrics
        metrics_group = QGroupBox(t("panel.result.summary"))
        metrics_layout = QHBoxLayout()
        
        self.status_label = QLabel(t("label.waiting"))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        
        self.r2_label = QLabel(t("result.r2") + ": -")
        self.mse_label = QLabel(t("result.mse") + ": -")
        
        metrics_layout.addWidget(self.status_label)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.r2_label)
        metrics_layout.addSpacing(30)
        metrics_layout.addWidget(self.mse_label)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # 2. Results Content (Split View: Table/Text vs Graph)
        content_layout = QHBoxLayout()
        
        # Text/Table Result
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tools Header
        tools_layout = QHBoxLayout()
        tools_layout.addWidget(QLabel(t("panel.result.tools")))
        tools_layout.addStretch()
        
        self.btn_copy_md = QPushButton(t("btn.copy_md"))
        self.btn_copy_md.clicked.connect(self.on_copy_markdown)
        self.btn_copy_tex = QPushButton(t("btn.copy_tex"))
        self.btn_copy_tex.clicked.connect(self.on_copy_latex)
        
        tools_layout.addWidget(self.btn_copy_md)
        tools_layout.addWidget(self.btn_copy_tex)
        text_layout.addLayout(tools_layout)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText(t("label.waiting"))
        self.result_text.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; background-color: #1a1a1a; color: #dcdcdc; border: 1px solid #333;")
        text_layout.addWidget(self.result_text)
        
        # Action Buttons footer
        footer_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("CSV")
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_excel = QPushButton("Excel")
        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_report = QPushButton(t("btn.generate_report"))
        self.btn_report.clicked.connect(self.on_generate_report)
        self.btn_report.setStyleSheet("background-color: #217346; color: white;")
        
        footer_layout.addWidget(self.btn_export_csv)
        footer_layout.addWidget(self.btn_export_excel)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_report)
        text_layout.addLayout(footer_layout)
        
        content_layout.addWidget(text_widget, stretch=1)
        
        # Graph Placeholder
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        
        self.graph_placeholder = QLabel("📈 Visualization")
        self.graph_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.graph_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_placeholder.setStyleSheet("background-color: #252526; border: 1px dashed #444; color: #777; font-size: 16px;")
        graph_layout.addWidget(self.graph_placeholder)
        
        content_layout.addWidget(graph_widget, stretch=1)
        
        layout.addLayout(content_layout)

        # 3. Step Log
        log_group = QGroupBox(t("panel.result.log"))
        log_layout = QVBoxLayout()
        
        self.log_view = QTextEdit()
        self.log_view.setMaximumHeight(100)
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #111; color: #888; font-size: 9pt; border: none;")
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

    def display_result(self, result_data: dict):
        self._last_result = result_data # Store for export
        success = result_data.get('success', True)
        
        if success:
            self.status_label.setText("✅ " + t("label.success"))
            self.status_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        else:
            self.status_label.setText("❌ " + t("label.failure"))
            self.status_label.setStyleSheet("color: #f44747; font-weight: bold;")
        
        r2 = result_data.get('r2', 'N/A')
        mse = result_data.get('mse', 'N/A')
        
        self.r2_label.setText(f"{t('result.r2')}: {r2}")
        self.mse_label.setText(f"{t('result.mse')}: {mse}")
        
        summary = result_data.get('summary', '')
        # If there's a table in result, format it
        if 'table' in result_data:
            df = result_data['table']
            if isinstance(df, pd.DataFrame):
                summary += "\n\n" + df.to_string()
        
        self.result_text.setText(summary)
        
        log_level = "INFO" if success else "ERROR"
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] [{log_level}] Analysis completed. Hash: {result_data.get('hash', '???')}")

    def on_copy_markdown(self):
        if not hasattr(self, '_last_result'): return
        res = self._last_result
        if 'table' in res and isinstance(res['table'], pd.DataFrame):
            text = res['table'].to_markdown()
        else:
            text = "```\n" + self.result_text.toPlainText() + "\n```"
        
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)
        from statelix_py.gui.components.toast import Toast
        Toast(self.window(), t("toast.analysis_complete")).show_toast()

    def on_copy_latex(self):
        if not hasattr(self, '_last_result'): return
        res = self._last_result
        if 'table' in res and isinstance(res['table'], pd.DataFrame):
            text = res['table'].to_latex()
        else:
            text = "\\begin{verbatim}\n" + self.result_text.toPlainText() + "\n\\end{verbatim}"
        
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)
        from statelix_py.gui.components.toast import Toast
        Toast(self.window(), t("toast.analysis_complete")).show_toast()

    def on_export_csv(self):
        if not hasattr(self, '_last_result') or 'table' not in self._last_result:
            QMessageBox.warning(self, t("menu.language"), "No table data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, t("btn.export_csv"), "", "CSV Files (*.csv)")
        if path:
            self._last_result['table'].to_csv(path, index=False)
            QMessageBox.information(self, t("label.success"), f"Data exported to {path}")

    def on_export_excel(self):
        if not hasattr(self, '_last_result') or 'table' not in self._last_result:
            QMessageBox.warning(self, t("menu.language"), "No table data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, t("btn.export_excel"), "", "Excel Files (*.xlsx)")
        if path:
            self._last_result['table'].to_excel(path, index=False)
            QMessageBox.information(self, t("label.success"), f"Data exported to {path}")

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
        
        path, _ = QFileDialog.getSaveFileName(self, t("btn.generate_report"), "", "HTML Files (*.html)")
        if path:
            report.save(path)
            QMessageBox.information(self, t("label.success"), f"Report saved to {path}")

