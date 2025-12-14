from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QCheckBox, 
    QFrame, QHBoxLayout, QPushButton, QFormLayout, QLineEdit,
    QListWidget, QListWidgetItem, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import pyqtSignal

class ModelPanel(QWidget):
    run_requested = pyqtSignal(dict) # Signal to trigger execution

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("モデルパネル")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # 1. Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("モデル:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "OLS (最小二乗法)", 
            "K-Means (クラスタリング)", 
            "ANOVA (分散分析)", 
            "AR Model (時系列)",
            "Logistic Regression",
            "Poisson Regression",
            "Negative Binomial",
            "Gamma Regression",
            "Probit Regression",
            "Ridge Regression",
            "Cox PH (生存時間解析)"
        ])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # 2. Parameters (Dynamic Grid)
        self.param_frame = QFrame()
        param_layout = QFormLayout()
        
        # Params Widgets
        self.spin_k = QSpinBox() # K-Means
        self.spin_k.setRange(2, 50); self.spin_k.setValue(3)
        
        self.spin_p = QSpinBox() # AR
        self.spin_p.setRange(1, 20); self.spin_p.setValue(1)

        self.spin_iter = QSpinBox() # GLM / All
        self.spin_iter.setRange(10, 1000); self.spin_iter.setValue(50)
        
        self.spin_alpha = QDoubleSpinBox() # Ridge
        self.spin_alpha.setRange(0.001, 1000.0); self.spin_alpha.setValue(1.0)
        
        # Labels for form
        self.row_k = (QLabel("Clusters (k):"), self.spin_k)
        self.row_p = (QLabel("Order (p):"), self.spin_p)
        self.row_iter = (QLabel("Max Iter:"), self.spin_iter)
        self.row_alpha = (QLabel("Alpha (λ):"), self.spin_alpha)

        # Add to layout
        param_layout.addRow(*self.row_k)
        param_layout.addRow(*self.row_p)
        param_layout.addRow(*self.row_iter)
        param_layout.addRow(*self.row_alpha)
        
        self.param_frame.setLayout(param_layout)
        layout.addWidget(self.param_frame)

        # 3. Variable Selection
        var_layout = QHBoxLayout()
        
        # Target (y) / Time
        self.target_layout = QVBoxLayout()
        self.label_target = QLabel("目的変数 (y) / Time")
        self.target_layout.addWidget(self.label_target)
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(80)
        self.target_layout.addWidget(self.target_list)
        var_layout.addLayout(self.target_layout)
        
        # Features (x) / Status
        self.feature_layout = QVBoxLayout()
        self.label_feature = QLabel("説明変数 (x) / Status")
        self.feature_layout.addWidget(self.label_feature)
        self.feature_list = QListWidget()
        self.feature_list.setMaximumHeight(80)
        self.feature_layout.addWidget(self.feature_list)
        var_layout.addLayout(self.feature_layout)
        
        layout.addLayout(var_layout)
        
        # Third List (Special for Cox mostly) - Optional
        # For simplicity, reuse target/feature logic:
        # Cox: Target=Time, Feature=Covariates. Where is Status?
        # Let's add specific Status box? 
        # Or Just use "Target" as Time, "Features" as X, and a separate "Status" list?
        # Let's add a "Group/Status" list that is hidden usually.
        
        self.status_layout = QVBoxLayout()
        self.label_status = QLabel("イベント/グループ")
        self.status_layout.addWidget(self.label_status)
        self.status_list = QListWidget()
        self.status_list.setMaximumHeight(80)
        self.status_layout.addWidget(self.status_list)
        var_layout.addLayout(self.status_layout)

        # 4. Actions
        action_layout = QHBoxLayout()
        self.run_btn = QPushButton("実行")
        self.run_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self.on_run)
        
        action_layout.addWidget(self.run_btn)
        layout.addLayout(action_layout)

        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize state
        self.on_model_changed(0)

    def on_model_changed(self, index):
        model = self.model_combo.currentText()
        
        # Hide all params first
        self.row_k[0].setVisible(False); self.row_k[1].setVisible(False)
        self.row_p[0].setVisible(False); self.row_p[1].setVisible(False)
        self.row_iter[0].setVisible(False); self.row_iter[1].setVisible(False)
        self.row_alpha[0].setVisible(False); self.row_alpha[1].setVisible(False)

        # Variable lists visibility
        self.label_target.setVisible(True); self.target_list.setVisible(True)
        self.label_feature.setVisible(True); self.feature_list.setVisible(True)
        self.label_status.setVisible(False); self.status_list.setVisible(False)
        
        self.feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Configuration Logic
        if "K-Means" in model:
            self.row_k[0].setVisible(True); self.row_k[1].setVisible(True)
            self.row_iter[0].setVisible(True); self.row_iter[1].setVisible(True)
            self.label_target.setVisible(False); self.target_list.setVisible(False)
            
        elif "AR Model" in model:
            self.row_p[0].setVisible(True); self.row_p[1].setVisible(True)
            self.label_feature.setVisible(False); self.feature_list.setVisible(False)
            
        elif "ANOVA" in model:
            self.label_feature.setText("グループ変数 (Group)")
            self.feature_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            
        elif "Cox PH" in model:
            self.label_target.setText("時間 (Time)")
            self.label_status.setVisible(True); self.status_list.setVisible(True)
            self.label_status.setText("イベント (1=Event, 0=Censor)")
            self.row_iter[0].setVisible(True); self.row_iter[1].setVisible(True)
            
        elif "Ridge" in model:
            self.row_alpha[0].setVisible(True); self.row_alpha[1].setVisible(True)
            
        elif "Regression" in model: # GLMs
            self.row_iter[0].setVisible(True); self.row_iter[1].setVisible(True)

        if "ANOVA" not in model:
             self.label_feature.setText("説明変数 (x)")

    def update_columns(self, df):
        if df is None: return
        cols = list(df.columns)
        self.target_list.clear() # Target / Time
        self.feature_list.clear() # X / Group
        self.status_list.clear() # Status
        
        self.target_list.addItems(cols)
        self.feature_list.addItems(cols)
        self.status_list.addItems(cols)

    def on_run(self):
        # Gather params
        target_items = self.target_list.selectedItems()
        target = target_items[0].text() if target_items else None
        
        feature_items = self.feature_list.selectedItems()
        features = [item.text() for item in feature_items]
        
        status_items = self.status_list.selectedItems()
        status = status_items[0].text() if status_items else None
        
        params = {
            "model": self.model_combo.currentText(),
            "target": target,
            "features": features,
            "status": status,
            "k": self.spin_k.value(),
            "p": self.spin_p.value(),
            "max_iter": self.spin_iter.value(),
            "alpha": self.spin_alpha.value()
        }
        self.run_requested.emit(params)
