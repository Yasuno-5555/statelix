
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QListWidget, QPushButton, QFrame, QScrollArea, QGroupBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal, Slot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from .timeline_widget import TimelineWidget
from .objection_tree import ObjectionTreeWidget
from .suggestion_navigator import SuggestionNavigatorWidget
from .contract_view import ContractViewWidget
from .strictness_control import StrictnessControlWidget
from ...diagnostics.presets import GovernanceMode

class MCIGauge(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(5,5,5,5)
        
        self.score_label = QLabel("MCI: --")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888;")
        layout.addWidget(self.score_label)
        
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(20)
        layout.addWidget(self.bar)
        
        self.status_label = QLabel("WAITING")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def set_score(self, score: float):
        percent = int(score * 100)
        self.bar.setValue(percent)
        self.score_label.setText(f"MCI: {score:.2f}")
        
        if score > 0.8:
            color = "#4ec9b0" # Green
            text = "TRUSTWORTHY"
        elif score > 0.5:
            color = "#dcdcaa" # Yellow
            text = "WARNING"
        else:
            color = "#f44747" # Red
            text = "REJECTED"
            
        self.score_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; letter-spacing: 2px;")
        
        # ProgressBar styling
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #252526;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

class DiagnosticsPanel(QWidget):
    suggestion_action = Signal(str) # Emits suggestion text when clicked
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # 0. Governance Control (Header)
        self.strict_ctrl = StrictnessControlWidget()
        main_layout.addWidget(self.strict_ctrl)
        
        # 1. MCI Judge (Visual Gauge)
        self.mci_gauge = MCIGauge()
        main_layout.addWidget(self.mci_gauge)
        
        # 2. Main Diagnostic Workspace (Tabs)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #333; }")
        
        # Tab A: Timeline
        self.timeline = TimelineWidget()
        self.timeline.iteration_selected.connect(self.on_iteration_selected)
        self.tabs.addTab(self.timeline, "Timeline")
        
        # Tab B: Objections
        self.obj_tree = ObjectionTreeWidget()
        self.tabs.addTab(self.obj_tree, "Objections")
        
        # Tab C: Suggestions
        self.sugg_nav = SuggestionNavigatorWidget()
        self.sugg_nav.suggestion_applied.connect(self.suggestion_action.emit)
        self.sugg_nav.suggestion_previewed.connect(self.on_suggestion_previewed)
        self.tabs.addTab(self.sugg_nav, "Suggestions")
        
        # Tab D: Contract
        self.contract_view = ContractViewWidget()
        self.tabs.addTab(self.contract_view, "Contract")
        
        # Tab E: Manifold
        from .manifold_widget import CausalManifoldWidget
        self.manifold_view = CausalManifoldWidget()
        self.manifold_view.point_selected.connect(self.on_manifold_point_selected)
        self.tabs.addTab(self.manifold_view, "Manifold")
        
        # Tab F: CIT Discovery
        from .cit_log_widget import CITDiscoveryWidget
        self.cit_log = CITDiscoveryWidget()
        self.cit_tab_idx = self.tabs.addTab(self.cit_log, "CIT Discovery")
        self.tabs.setTabVisible(self.cit_tab_idx, False)
        
        # Tab G: Interview
        from .analyst_chat_widget import AnalystChatWidget
        self.analyst_chat = AnalystChatWidget()
        self.tabs.addTab(self.analyst_chat, "Interview")
        
        main_layout.addWidget(self.tabs, stretch=1)
        
        # Connect strictness toggle to CIT visibility
        self.strict_ctrl.group.buttonClicked.connect(self.on_strictness_changed)
        
        # 4. Integrity Statement
        integrity = QLabel("Statelix does not guarantee correctness. It guarantees refusal to lie.")
        integrity.setAlignment(Qt.AlignmentFlag.AlignCenter)
        integrity.setStyleSheet("color: #555; font-style: italic; font-size: 11px;")
        main_layout.addWidget(integrity)
        
        from .next_action_widget import NextActionWidget
        self.next_action_banner = NextActionWidget()
        self.next_action_banner.action_clicked.connect(self.suggestion_action.emit)
        main_layout.addWidget(self.next_action_banner)
        
        self.setLayout(main_layout)
        
    def update_diagnostics(self, mci: float, objections: list, suggestions: list, history: list, report=None, summary=None):
        self._full_history_obj = None # Will be set if available
        if report: 
             self._current_report = report
             # Show the 'Next Best Action' if it exists
             from ..critic import ModelCritic
             critic = ModelCritic()
             action = critic.get_sole_next_action(report)
             self.next_action_banner.set_action(action)
        else:
             self.next_action_banner.hide()
             
        if summary: self._current_summary = summary
        
        # 1. MCI
        self.mci_gauge.set_score(mci)
        
        # 2. Objections
        self.obj_tree.update_objections(objections)
            
        # 3. Suggestions
        self.sugg_nav.update_suggestions(suggestions)
        
        # 4. Timeline
        if hasattr(self, '_current_history') and self._current_history:
            self.timeline.update_history(self._current_history)
            
        # 5. Manifold (If model/data available)
        if hasattr(self, '_current_model_data'):
            from ...diagnostics.causal_manifold import CausalManifold
            engine = CausalManifold(self._current_model_data['model'], self._current_model_data['data'])
            points = engine.compute_manifold()
            quivers = engine.get_quivers()
            self.manifold_view.update_manifold(points, quivers)
            
            # 6. CIT Discovery (Only if exploratory)
            if self.strict_ctrl.current_mode == GovernanceMode.EXPLORATORY:
                from ...diagnostics.cit_detector import CITDetector
                # Prepare combined metrics
                cit_metrics = {
                    'p_value': getattr(self._current_model_data['model'], 'p_value_', None),
                    'topology_score': mci, # Simplified fallback
                    'r2': getattr(self._current_model_data['model'], 'r2_score_', 0.0),
                    'mci': mci
                }
                cit_engine = CITDetector(cit_metrics, points)
                discoveries = cit_engine.detect()
                self.cit_log.update_discoveries(discoveries)
                
            # 7. Analyst Interview
            # We need the full report object for the engine
            if hasattr(self, '_current_report') and hasattr(self, '_current_summary'):
                self.analyst_chat.set_context(self._current_report, self._current_summary)

    def set_history_object(self, history_obj):
        """Called by result panel to provide full history context."""
        self._current_history = history_obj
        self.timeline.update_history(history_obj)
        
        # Update contract based on latest
        if history_obj.history:
              latest = history_obj.history[-1]
              self.contract_view.update_status(latest.mci)

    @Slot(int)
    def on_iteration_selected(self, iteration):
        """Handle timeline click – show historical state."""
        if not hasattr(self, '_current_history') or not self._current_history:
            return
            
        data = self._current_history.get_score_at(iteration)
        if data:
            # Update gauges/trees for that specific moment
            self.mci_gauge.set_score(data['mci'])
            self.obj_tree.set_historical_objections(iteration, data['messages'])
            self.sugg_nav.update_suggestions(data['suggestions'])
            # Highlight iteration in tab title or status
            self.tabs.setTabText(1, f"Objections (Iter {iteration})")

    @Slot()
    def on_strictness_changed(self):
        """Toggle CIT tab visibility based on mode."""
        is_exploratory = self.strict_ctrl.current_mode == GovernanceMode.EXPLORATORY
        self.tabs.setTabVisible(self.cit_tab_idx, is_exploratory)
        if is_exploratory:
             # Force update if we have data
             self.update_diagnostics(self.mci_gauge.bar.value()/100.0, [], [], [])

    @Slot(dict)
    def on_manifold_point_selected(self, params):
        """Handle manifold point click - suggest adoption."""
        param_str = ", ".join([f"{k}={v:.4f}" for k,v in params.items()])
        print(f"[Manifold] User selected refinement: {param_str}")
    @Slot(str)
    def on_suggestion_previewed(self, suggestion_text):
        """Overlay a 'What-If' manifold based on suggestion."""
        if not hasattr(self, '_current_model_data'):
            return
            
        from ...diagnostics.causal_manifold import CausalManifold
        engine = CausalManifold(self._current_model_data['model'], self._current_model_data['data'])
        
        # 1. Propose refinement
        refinement = engine.propose_refinement(suggestion_text)
        if refinement:
            # 2. Compute hypothetical manifold
            hypo_points = engine.compute_hypothetical_manifold(refinement)
            # 3. Overlay on widget
            self.manifold_view.update_hypothetical_manifold(hypo_points)
            # 4. Switch to Manifold tab to show result
            self.tabs.setCurrentIndex(4) # Manifold is Tab E
        
