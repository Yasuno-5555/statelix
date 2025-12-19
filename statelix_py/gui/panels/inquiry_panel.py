
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
        main_layout = QHBoxLayout(self)
        
        # --- Left: The Interrogation Room (Inputs) ---
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        
        # 1. The Big Question
        question_group = QGroupBox("1. What do you want to know?")
        q_layout = QVBoxLayout()
        self.q_btn_group = QButtonGroup()
        
        self.r_drivers = QRadioButton("🔍 What drives this outcome? (Drivers)")
        self.r_drivers.setChecked(True)
        self.r_causal = QRadioButton("🔬 Did X cause Y? (Causal Check)")
        self.r_whatif = QRadioButton("🔮 What if X changed? (Simulation)")
        
        self.r_drivers.setToolTip("Finds variables strongly associated with the outcome.\nUses regularization (Lasso/Ridge) to screen predictors.")
        self.r_causal.setToolTip("Tests for a causal relationship using IV, DiD, or RDD.\nChecks checks critical assumptions like parallel trends.")
        self.r_whatif.setToolTip("Simulates a counterfactual world.\ne.g., 'What would GDP be if Logic didn't exist?'")
        
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
        
        v_layout.addRow("Outcome (Y):", self.combo_y)
        v_layout.addRow("Driver/Treatment (X):", self.combo_x)
        v_layout.addRow("Instrument/Control (Z):", self.combo_z)
        
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
        self.narrative_box.setPlaceholderText("The story will appear here...")
        
        narrative_container = QGroupBox("The Story")
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
        
        main_layout.addWidget(splitter)
        
        # Connections
        self.q_btn_group.buttonClicked.connect(self.update_ui_state)
        
    def update_columns(self, df):
        if df is None: return
        cols = list(df.columns)
        for c in [self.combo_y, self.combo_x, self.combo_z]:
            c.clear()
            c.addItems(cols)
            
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
