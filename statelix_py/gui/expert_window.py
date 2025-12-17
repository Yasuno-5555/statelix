from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget
)
from PySide6.QtCore import Qt

from statelix_py.gui.panels.data_panel import DataPanel
from statelix_py.gui.panels.model_panel import ModelPanel
from statelix_py.gui.panels.result_panel import ResultPanel
from statelix_py.gui.panels.plot_panel import PlotPanel
from statelix_py.gui.panels.exploratory_panel import ExploratoryPanel
from statelix_py.gui.panels.variable_inspector import VariableInspector

class ExpertWindow(QMainWindow):
    def __init__(self, parent_main_window=None):
        super().__init__()
        self.parent_mw = parent_main_window # Ref to main window for status/toasts
        self.setWindowFlags(Qt.WindowType.Widget) # Embeddable
        # self.setAttribute(Qt.WidgetAttribute.WA_StaticContents) # Maybe?
        
        # Force background just in case
        # self.setStyleSheet("background-color: #1e1e1e;") 
        self.init_ui()

    def init_ui(self):
        # 1. Central Area (Results/Plots)
        self.output_tabs = QTabWidget()
        self.result_panel = ResultPanel()
        self.plot_panel = PlotPanel()
        self.exploratory_panel = ExploratoryPanel()
        
        self.output_tabs.addTab(self.result_panel, "Result")
        self.output_tabs.addTab(self.plot_panel, "Plots")
        self.output_tabs.addTab(self.exploratory_panel, "EDA")
        
        self.setCentralWidget(self.output_tabs)
        
        # 2. Docks
        # Data Dock (Left)
        self.dock_data = QDockWidget("Data Manager", self)
        self.dock_data.setWidget(DataPanel())
        self.data_panel = self.dock_data.widget() # Accessor
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_data)
        
        # Inspector Dock (Left, below Data)
        self.dock_inspector = QDockWidget("Variable Inspector", self)
        self.dock_inspector.setWidget(VariableInspector())
        self.inspector_panel = self.dock_inspector.widget()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_inspector)
        
        # Model Dock (Right)
        self.dock_model = QDockWidget("Model Configuration", self)
        self.dock_model.setWidget(ModelPanel())
        self.model_panel = self.dock_model.widget()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_model)
        
        # 3. Connections
        self.model_panel.run_requested.connect(self.run_analysis_bridge)
        
        # Data -> Model
        self.data_panel.data_loaded.connect(self.model_panel.update_columns)
        # Data -> Inspector
        self.data_panel.data_loaded.connect(self.inspector_panel.set_data)
        # Data -> EDA
        self.data_panel.data_loaded.connect(self.exploratory_panel.on_data_loaded)
        
        # Connect Toast signals if possible? 
        # For now, rely on direct calls if needed, or pass parent.

    def run_analysis_bridge(self, params):
        # Forward to parent MainWindow which holds the Worker
        if self.parent_mw:
            self.parent_mw.run_analysis(params)
            
    def add_wasm_plugins(self, plugins):
        self.model_panel.add_wasm_plugins(plugins)
