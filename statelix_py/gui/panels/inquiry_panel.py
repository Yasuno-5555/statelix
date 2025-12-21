
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTextEdit, QSplitter, QGroupBox, QRadioButton, QButtonGroup,
    QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class InquiryPanel(QWidget):
    """
    Student-Focused "Inquiry Mode" Panel.
    Prioritizes Narrative and Questions over raw models.
    """
    run_inquiry = Signal(dict) # Signal to MainWindow to run logic

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self) # Changed to QV for Banner
        
        # --- 0. Responsibility Banner (Constitution) ---
        banner = QLabel("⚠️ <b>STATELIX DOES NOT PROVIDE ANSWERS.</b><br>The interpretation of these results is <u>your responsibility</u>. AI only provides the scaffolding.")
        banner.setStyleSheet("background-color: #FFF3CD; color: #856404; border: 1px solid #FFEEBA; padding: 10px; font-size: 13px; border-radius: 4px;")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(banner)
        
        # Splitter Container
        splitter_container = QWidget()
        h_layout = QHBoxLayout(splitter_container)
        h_layout.setContentsMargins(0,0,0,0)
        
        # --- Left: The Interrogation Room (Inputs) ---
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        
        # 0.5 Data Load Section (NEW)
        data_group = QGroupBox("📂 0. データを読み込む")
        data_group.setStyleSheet("""
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #4fc3f7;
                font-weight: bold;
            }
        """)
        d_layout = QVBoxLayout()
        
        self.data_status_label = QLabel("📋 データ: 未選択")
        self.data_status_label.setStyleSheet("color: #888888;")
        
        self.btn_load_data = QPushButton("📂 データファイルを開く")
        self.btn_load_data.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)
        self.btn_load_data.clicked.connect(self._on_load_data)
        
        d_layout.addWidget(self.data_status_label)
        d_layout.addWidget(self.btn_load_data)
        data_group.setLayout(d_layout)
        input_layout.addWidget(data_group)
        
        # 1. The Big Question
        question_group = QGroupBox("1. What do you want to know?")
        q_layout = QVBoxLayout()
        self.q_btn_group = QButtonGroup()
        
        self.r_drivers = QRadioButton("🔍 What drives this outcome? (Drivers)")
        self.r_drivers.setChecked(True)
        self.r_causal = QRadioButton("🔬 Did X cause Y? (Causal Check)")
        self.r_whatif = QRadioButton("🔮 What if X changed? (Simulation)")
        
        # Conceptual Tooltips (Metaphors)
        self.r_drivers.setToolTip("<b>Drivers (Association):</b><br>Like checking which friends usually show up at the party together.<br>It doesn't mean one invited the other, just that they are often seen together.")
        self.r_causal.setToolTip("<b>Causal Inference:</b><br>Like checking if pushing a button <i>actually turns on</i> the light,<br>or if you just happened to push it when the sun came up.<br>Requires strict assumptions (parallel worlds).")
        self.r_whatif.setToolTip("<b>Counterfactual Simulation:</b><br>Imagining a parallel world: 'If I had studied 1 hour more, what would my score be?'<br>Based on the patterns found in the data.")
        
        self.q_btn_group.addButton(self.r_drivers, 1)
        self.q_btn_group.addButton(self.r_causal, 2)
        self.q_btn_group.addButton(self.r_whatif, 3)
        
        q_layout.addWidget(self.r_drivers)
        q_layout.addWidget(self.r_causal)
        q_layout.addWidget(self.r_whatif)
        q_layout.addStretch()
        question_group.setLayout(q_layout)
        input_layout.addWidget(question_group)
        
        # 2. Variable context (Outcome / Treatment)
        var_group = QGroupBox("2. Define Context")
        v_layout = QFormLayout()
        
        self.combo_y = QComboBox() # Outcome
        self.combo_x = QComboBox() # Treatment/Driver
        self.combo_z = QComboBox() # Instrument/Control (Aux)
        
        # Tooltips for Variables (IMPROVED)
        self.combo_y.setToolTip("🎯 <b>Outcome (Y)</b>: The result you care about.<br><i>Examples: GDP, Health Score, Test Grades, Sales</i>")
        self.combo_x.setToolTip("📊 <b>Driver/Treatment (X)</b>: The factor you want to test.<br><i>Examples: Policy, Medicine Dose, Study Time, Advertising Budget</i>")
        
        # Improved Z explanation with example
        z_tooltip = """
        <b>🔧 Instrument/Z (操作変数)</b><br><br>
        <b>何これ?</b> XをYに強制的に「押し込む」外部の力。<br><br>
        <b>例: 勉強時間が成績に影響するか？</b><br>
        • Y = 成績<br>
        • X = 勉強時間<br>
        • Z = 「図書館の近さ」(家が近い人は長く勉強する傾向)<br><br>
        <b>なぜ必要?</b> Xが「自己選択」の場合（やる気ある人が勉強する等）、<br>
        Zで「ランダムにXを割り当てた効果」を推定する。<br><br>
        <i>分からなければ空欄でOK → Drivers分析になります</i>
        """
        self.combo_z.setToolTip(z_tooltip)
        
        v_layout.addRow("🎯 Outcome (Y):", self.combo_y)
        v_layout.addRow("📊 Driver/Treatment (X):", self.combo_x)
        
        # Z with inline help button
        z_row = QHBoxLayout()
        z_row.addWidget(self.combo_z, 1)
        self.btn_z_help = QPushButton("❓")
        self.btn_z_help.setFixedWidth(30)
        self.btn_z_help.setToolTip("操作変数の詳しい説明を表示")
        self.btn_z_help.clicked.connect(self._show_z_explanation)
        z_row.addWidget(self.btn_z_help)
        
        v_layout.addRow("🔧 Instrument/Z:", z_row)
        
        var_group.setLayout(v_layout)
        input_layout.addWidget(var_group)
        
        # 3. Simulation Settings (Hidden by default)
        self.sim_group = QGroupBox("3. Simulation Settings")
        s_layout = QFormLayout()
        
        self.combo_change_type = QComboBox()
        self.combo_change_type.addItems(["Increase by %", "Decrease by %", "Set to Value", "Increase by Value"])
        
        self.spin_change_val = QDoubleSpinBox()
        self.spin_change_val.setRange(-1000000, 1000000)
        self.spin_change_val.setValue(10.0)
        
        s_layout.addRow("Change Type:", self.combo_change_type)
        s_layout.addRow("Amount:", self.spin_change_val)
        
        self.sim_group.setLayout(s_layout)
        self.sim_group.hide() # Initially hidden
        input_layout.addWidget(self.sim_group)
        
        # 3. Method hint (Hidden per question)
        self.method_label = QLabel("Method: Auto-Detect")
        self.method_label.setStyleSheet("color: gray; font-style: italic;")
        input_layout.addWidget(self.method_label)
        
        # 4. "Tell Me Why" Button
        self.btn_ask = QPushButton("Tell Me Why")
        self.btn_ask.setStyleSheet("""
            QPushButton {
                background-color: #6C5CE7; 
                color: white; 
                font-size: 16px; 
                font-weight: bold; 
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #5b4cc4; }
        """)
        self.btn_ask.clicked.connect(self.on_ask)
        input_layout.addWidget(self.btn_ask)
        
        # 5. Export Report Button
        self.btn_export = QPushButton("📄 Save Report (HTML)")
        self.btn_export.clicked.connect(self.on_export_report)
        input_layout.addWidget(self.btn_export)
        
        input_layout.addStretch()
        
        # --- Right: The Story (Output) ---
        output_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # A. Narrative (Text First)
        self.narrative_box = QTextEdit()
        self.narrative_box.setReadOnly(True)
        self.narrative_box.setStyleSheet("""
            font-family: 'Segoe UI', sans-serif; 
            font-size: 14px; 
            line-height: 1.5;
            padding: 10px;
        """)
        self.narrative_box.setPlaceholderText("The scaffolding will appear here.\nYou will build the conclusion.")
        
        narrative_container = QGroupBox("The Scaffolding (Facts & Hints)")
        n_layout = QVBoxLayout()
        n_layout.addWidget(self.narrative_box)
        narrative_container.setLayout(n_layout)
        
        # B. Assumption Visualizer (Diagnostics)
        self.viz_container = QGroupBox("Reality Check (Assumptions)")
        v_layout_2 = QVBoxLayout()
        
        self.figure = plt.figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        v_layout_2.addWidget(self.canvas)
        self.viz_container.setLayout(v_layout_2)
        
        output_splitter.addWidget(narrative_container)
        output_splitter.addWidget(self.viz_container)
        output_splitter.setSizes([400, 300]) # Text bigger
        
        # Split Main
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(input_container)
        splitter.addWidget(output_splitter)
        splitter.setSizes([300, 700])
        
        h_layout.addWidget(splitter)
        main_layout.addWidget(splitter_container)
        
        # Connections
        self.q_btn_group.buttonClicked.connect(self.update_ui_state)

    def on_export_report(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import datetime
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Report", "Statelix_Report.html", "HTML Files (*.html)")
        if not filename: return
        
        try:
            content = self.narrative_box.toHtml()
            # Inject Instructions and Empty Conclusion
            report_html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                    .warning {{ background-color: #fff3cd; color: #856404; padding: 15px; border: 1px solid #ffeeba; border-left: 5px solid #ffc107; margin-bottom: 20px; }}
                    .conclusion {{ border: 2px dashed #bdc3c7; padding: 20px; background-color: #f9f9f9; color: #7f8c8d; min-height: 150px; }}
                </style>
            </head>
            <body>
                <h1>Statelix Analysis Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                
                <div class="warning">
                    <strong>⚠️ NOTE:</strong> This document contains analysis scaffolding provided by AI. 
                    The final interpretation and conclusions are the responsibility of the author.
                </div>
                
                <h2>1. Analysis Scaffolding (Facts & Hints)</h2>
                {content}
                
                <h2>2. Conclusion (To Be Written by Author)</h2>
                <div class="conclusion">
                    <p><b>[INSTRUCTIONS]</b></p>
                    <p>Based on the scaffolding above, write your conclusion here. Consider:</p>
                    <ul>
                        <li>Does the evidence support your initial hypothesis?</li>
                        <li>Are there alternative explanations (confounders)?</li>
                        <li>What are the limitations of this analysis?</li>
                    </ul>
                    <br><br>
                    <i>(Delete this text and write your own conclusion.)</i>
                </div>
            </body>
            </html>
            """
            
            with open(filename, "w", encoding='utf-8') as f:
                f.write(report_html)
                
            QMessageBox.information(self, "Success", f"Report template saved to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
    
    def _on_load_data(self):
        """Open file dialog to load data directly from Inquiry Mode."""
        from PySide6.QtWidgets import QFileDialog
        import pandas as pd
        import os
        
        filters = (
            "All Supported Files (*.csv *.xlsx *.xls *.json *.parquet);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;"
            "JSON Files (*.json);;"
            "Parquet Files (*.parquet);;"
            "All Files (*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", filters
        )
        
        if not file_path:
            return
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            df = None
            
            if ext == '.csv':
                # Try multiple encodings
                for enc in ['utf-8', 'cp932', 'shift_jis', 'latin1']:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        break
                    except (UnicodeDecodeError, ValueError):
                        continue
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            elif ext in ['.parquet', '.pq']:
                df = pd.read_parquet(file_path)
            
            if df is not None:
                # Update DataManager
                from statelix_py.core.data_manager import DataManager
                dm = DataManager.instance()
                dm.set_data(df, file_path)
                
                # Update columns in this panel
                self.update_columns(df)
                
                # Update status
                filename = os.path.basename(file_path)
                self.data_status_label.setText(f"✅ データ: {filename} ({df.shape[0]}行 x {df.shape[1]}列)")
                self.data_status_label.setStyleSheet("color: #81c784;")
                
                QMessageBox.information(self, "Success", f"データを読み込みました: {filename}")
            else:
                raise Exception("ファイル形式を読み込めませんでした。")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"データ読み込みエラー:\n{str(e)}")
    
    def _show_z_explanation(self):
        """Show detailed explanation dialog for Instrument/Z variable."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("🔧 操作変数 (Instrument/Z) とは？")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        explanation = """
        <h2>🔧 操作変数 (Instrument Variable / IV) とは？</h2>
        
        <h3>📖 簡単な説明</h3>
        <p>XがYに与える「本当の因果効果」を知りたいとき、<br>
        Xに「外から」影響を与える変数Zを使って推定します。</p>
        
        <h3>🎯 なぜ必要？</h3>
        <p>多くの場合、Xは「自己選択」の結果です：</p>
        <ul>
            <li>やる気のある人が多く勉強する（X=勉強時間）</li>
            <li>健康な人が運動する（X=運動量）</li>
            <li>優秀な企業がR&Dに投資する（X=研究開発費）</li>
        </ul>
        <p>この場合、XとYの関係は「因果」ではなく「相関」かもしれません。</p>
        
        <h3>🔬 Zの条件（重要！）</h3>
        <ol>
            <li><b>関連性</b>: ZはXに影響を与える</li>
            <li><b>排他性</b>: ZはYに「直接」影響しない（Xを通じてのみ）</li>
        </ol>
        
        <h3>💡 具体例</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>研究課題</th><th>Y</th><th>X</th><th>Z</th></tr>
            <tr><td>教育の効果</td><td>収入</td><td>教育年数</td><td>家から学校への距離</td></tr>
            <tr><td>兵役の効果</td><td>将来収入</td><td>兵役経験</td><td>くじ引き番号</td></tr>
            <tr><td>政策の効果</td><td>健康</td><td>治療受診</td><td>地域の医療キャパ</td></tr>
        </table>
        
        <h3>⚠️ 注意点</h3>
        <p>良いZを見つけるのは難しいです。<br>
        分からなければ空欄にして、まずは「Drivers」分析から始めましょう。</p>
        """
        
        label = QLabel(explanation)
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 11pt; line-height: 1.5;")
        layout.addWidget(label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        
        dialog.exec()
        
    def update_columns(self, df):
        if df is None: return
        cols = list(df.columns)
        for c in [self.combo_y, self.combo_x, self.combo_z]:
            c.clear()
            c.addItems(cols)
        
        # Update data status if available
        if hasattr(self, 'data_status_label'):
            self.data_status_label.setText(f"✅ データ: {df.shape[0]}行 x {df.shape[1]}列")
            self.data_status_label.setStyleSheet("color: #81c784;")
            
    def update_ui_state(self):
        mid = self.q_btn_group.checkedId()
        if mid == 1: # Drivers
            self.method_label.setText("Method: OLS/Lasso (Association)")
            self.combo_z.setEnabled(False)
            self.sim_group.hide()
        elif mid == 2: # Causal
            self.method_label.setText("Method: IV / DiD / RDD (Inference)")
            self.combo_z.setEnabled(True)
            self.sim_group.hide()
        elif mid == 3: # WhatIf
            self.method_label.setText("Method: Counterfactual Simulation")
            self.combo_z.setEnabled(True)
            self.sim_group.show()

    def on_ask(self):
        # Prepare Inquiry Request
        mid = self.q_btn_group.checkedId()
        
        params = {
            "mode": "Inquiry", # Validates inside worker
            "question_id": mid,
            "y": self.combo_y.currentText(),
            "x": self.combo_x.currentText(),
            "z": self.combo_z.currentText()
        }
        
        if mid == 3:
            params["sim_type"] = self.combo_change_type.currentText()
            params["sim_val"] = self.spin_change_val.value()
            
        self.run_inquiry.emit(params)

    def set_narrative(self, text):
        # Render Markdown-like text
        html = text.replace("\n", "<br>")
        html = html.replace("### ", "<h3>").replace("## ", "<h2>")
        html = html.replace("**", "<b>").replace("__", "<b>")
        # Caveats
        if "> [!WARNING]" in html:
             html = html.replace("> [!WARNING]", "<div style='background-color: #FFF3CD; padding: 10px; border-left: 5px solid #FFC107;'><b>⚠️ CAUTION</b>")
             html = html.replace("> -", "<li>")
             html += "</div>"
             
        self.narrative_box.setHtml(html)

    def plot_assumption(self, fig_func):
        # Helper to plot assumptions
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        fig_func(ax)
        self.canvas.draw()

