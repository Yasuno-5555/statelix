import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QFrame,
    QPushButton, QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt
from statelix_py.gui.i18n import t
from ..widgets.diagnostics_widget import DiagnosticsPanel

class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title_layout = QHBoxLayout()
        title = QLabel(t("panel.result"))
        title.setStyleSheet("font-weight: bold; font-size: 16px; color: #007acc;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        # Governance Label
        gov = QLabel("Statelix Governance Active")
        gov.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        title_layout.addWidget(gov)
        
        self.btn_refusal_report = QPushButton(t("btn.save_refusal_report"))
        self.btn_refusal_report.setVisible(False)
        self.btn_refusal_report.setStyleSheet("background-color: #d32f2f; color: white; padding: 4px 8px; font-weight: bold;")
        self.btn_refusal_report.clicked.connect(self.on_refusal_report)
        title_layout.addWidget(self.btn_refusal_report)
        
        layout.addLayout(title_layout)

        # Content Splitter (Diagnostics Top, Results Bottom? Or Tabs?)
        # User wants "Dashboard". Let's put Diagnostics at the TOP as the "Gatekeeper".
        
        # 1. Diagnostics Judge
        self.diag_panel = DiagnosticsPanel()
        layout.addWidget(self.diag_panel)
        
        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # 2. Results Container (Hidden if rejected)
        self.result_container = QWidget()
        res_layout = QVBoxLayout(self.result_container)
        res_layout.setContentsMargins(0,0,0,0)
        
        # ... (Metrics) ...
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
        res_layout.addWidget(metrics_group)

        # ... (Text/Graph) ...
        content_layout = QHBoxLayout()
        text_widget = QWidget(); text_l = QVBoxLayout(text_widget); text_l.setContentsMargins(0,0,0,0)
        
        # Tools
        tools_layout = QHBoxLayout()
        tools_layout.addWidget(QLabel(t("panel.result.tools")))
        tools_layout.addStretch()
        self.btn_copy_md = QPushButton(t("btn.copy_md")); self.btn_copy_md.clicked.connect(self.on_copy_markdown)
        self.btn_copy_tex = QPushButton(t("btn.copy_tex")); self.btn_copy_tex.clicked.connect(self.on_copy_latex)
        tools_layout.addWidget(self.btn_copy_md); tools_layout.addWidget(self.btn_copy_tex)
        text_l.addLayout(tools_layout)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: 'Consolas', monospace; background-color: #1a1a1a; color: #dcdcdc; border: 1px solid #333;")
        text_l.addWidget(self.result_text)
        
        # Exports
        footer = QHBoxLayout()
        self.btn_export_csv = QPushButton("CSV"); self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_excel = QPushButton("Excel"); self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_report = QPushButton(t("btn.generate_report")); self.btn_report.clicked.connect(self.on_generate_report)
        self.btn_report.setStyleSheet("background-color: #217346; color: white;")
        footer.addWidget(self.btn_export_csv); footer.addWidget(self.btn_export_excel); footer.addStretch(); footer.addWidget(self.btn_report)
        text_l.addLayout(footer)
        
        content_layout.addWidget(text_widget, stretch=1)
        
        # Graph
        graph_widget = QWidget(); graph_l = QVBoxLayout(graph_widget); graph_l.setContentsMargins(0,0,0,0)
        self.graph_placeholder = QLabel("📈 Visualization")
        self.graph_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        self.graph_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_placeholder.setStyleSheet("background-color: #252526; border: 1px dashed #444; color: #777; font-size: 16px;")
        graph_l.addWidget(self.graph_placeholder)
        content_layout.addWidget(graph_widget, stretch=1)
        
        res_layout.addLayout(content_layout)
        layout.addWidget(self.result_container)

        # 3. Log
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
        self._last_result = result_data 
        success = result_data.get('success', True)
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")

        # --- Governance Check ---
        # Did we receive diagnostic data?
        diag = result_data.get('diagnostics', {})
        if not diag:
            # Fallback if no diagnostics (e.g. error before fit)
            # Or legacy model
            mci = 0.0
            objections = ["System Error: Diagnostics missing"] if success else ["Analysis Failed"]
            suggestions = []
            history = []
        else:
            mci = diag.get('mci', 0.0)
            objections = diag.get('objections_list', [])
            suggestions = diag.get('suggestions', [])
            history = diag.get('history', [])

        # Update Judge Panel
        report = result_data.get('report')
        summary = result_data.get('narrative_summary')
        self.diag_panel.update_diagnostics(mci, objections, suggestions, history, report=report, summary=summary)
        
        # If result_data contains raw model/data context, pass for manifold
        if 'model_context' in result_data:
             ctx = result_data['model_context']
             self.diag_panel.set_model_context(ctx['model'], ctx['data'])
        
        # If we have a history object, pass it for richer exploration
        if 'history_obj' in result_data:
            self.diag_panel.set_history_object(result_data['history_obj'])
        
        # VETO LOGIC: If success is False OR MCI < 0.4
        # Note: If ModelRejectedError was caught, `success` might be False but we have diag data.
        # We assume `main_window` catches ModelRejectedError and passes the diag data here.
        
        is_rejected = (not success) or (mci < 0.4 and mci > 0.0) 
        # mci > 0.0 check avoids blocking models that simply don't have diagnostics yet (legacy fallback)
        # But user wants STRICT. For now, trust the 'success' flag primarily, or strict MCI.
        
        if is_rejected:
            self.result_container.setVisible(False)
            self.btn_refusal_report.setVisible(True)
            self.status_label.setText("🛑 RESULT HIDDEN (Governance Veto)")
            self.log_view.append(f"[{timestamp}] [WARNING] Result withheld due to low credibility or failure.")
        else:
            self.result_container.setVisible(True)
            self.btn_refusal_report.setVisible(False)
            self.status_label.setText("✅ " + t("label.success"))
            self.status_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
            
            # Populate fields
            r2 = result_data.get('r2', 'N/A')
            mse = result_data.get('mse', 'N/A')
            self.r2_label.setText(f"{t('result.r2')}: {r2}")
            self.mse_label.setText(f"{t('result.mse')}: {mse}")
            
            summary = result_data.get('summary', '')
            if 'table' in result_data:
                df = result_data['table']
                if isinstance(df, pd.DataFrame):
                    summary += "\n\n" + df.to_string()
            self.result_text.setText(summary)
            
            self.log_view.append(f"[{timestamp}] [INFO] Analysis accepted. MCI: {mci:.2f}")

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

        if path:
            report.save(path)
            QMessageBox.information(self, t("label.success"), f"Report saved to {path}")

    def on_refusal_report(self):
        """Generate and save a Refusal Report."""
        from statelix_py.utils.report_generator import ReportGenerator
        
        if not hasattr(self, '_last_result') or 'diagnostics' not in self._last_result:
            return
            
        diag = self._last_result['diagnostics']
        report = ReportGenerator.create_refusal_report(diag)
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Refusal Report", "Refusal_Report.html", "HTML Files (*.html)")
        if path:
            report.save(path)
            QMessageBox.information(self, "Saved", f"Refusal report saved to {path}")
