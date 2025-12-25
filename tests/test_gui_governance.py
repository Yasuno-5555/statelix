
import sys
import os
import pytest
from PySide6.QtWidgets import QApplication

# Ensure we can import statelix_py
sys.path.append(os.path.abspath("statelix_py"))

try:
    from statelix_py.gui.panels.result_panel import ResultPanel
except ImportError:
    sys.path.append(os.getcwd())
    from statelix_py.gui.panels.result_panel import ResultPanel

# Needed for PySide6 widgets
app = QApplication.instance() or QApplication(sys.argv)

def test_gui_governance():
    print("\n--- Testing GUI Governance Logic ---")
    
    panel = ResultPanel()
    panel.show() # Must show to test visibility
    
    # 1. Test Acceptance
    print("[Case 1] High MCI -> Show Results")
    good_result = {
        'success': True,
        'r2': 0.95,
        'summary': 'Good Model',
        'diagnostics': {
            'mci': 0.95,
            'objections_list': [],
            'history': []
        }
    }
    panel.display_result(good_result)
    assert panel.result_container.isVisible(), "Result container should be visible for good model"
    assert "MCI: 0.95" in panel.diag_panel.mci_gauge.score_label.text()
    
    # 2. Test Rejection (Veto)
    print("[Case 2] Low MCI -> Hide Results")
    bad_result = {
        'success': True, # Technically succeeded fitting, but bad quality
        'r2': 0.1,
        'summary': 'Garbage Model',
        'diagnostics': {
            'mci': 0.2, # < 0.4 Threshold
            'objections_list': ['Terrible fit'],
            'history': []
        }
    }
    panel.display_result(bad_result)
    assert not panel.result_container.isVisible(), "Result container MUST be hidden for low MCI"
    assert "REJECTED" in panel.diag_panel.mci_gauge.status_label.text()
    assert "Garbage Model" not in panel.result_text.toPlainText(), "Should not display text if hidden (though widget holds it, container hides it)"
    
    # 3. Test Failure (ModelRejectedError case)
    print("[Case 3] Failure Flag -> Hide Results")
    fail_result = {
        'success': False,
        'diagnostics': {
            'mci': 0.5, # MCI might be okayish but something else failed
            'objections_list': ['Topology Collapse'],
            'history': []
        }
    }
    panel.display_result(fail_result)
    assert not panel.result_container.isVisible(), "Result container should be hidden on failure"
    
    print("\nSUCCESS: GUI Governance Logic verified.")

def test_new_widgets():
    print("\n--- Testing New Diagnostic Widgets ---")
    from statelix_py.gui.widgets.diagnostics_widget import DiagnosticsPanel
    from statelix_py.diagnostics.history import DiagnosticHistory
    from statelix_py.diagnostics.critic import DiagnosticReport, MCIScore
    from statelix_py.diagnostics.presets import GovernanceMode
    import os

    panel = DiagnosticsPanel()
    
    # 1. Test Timeline & History
    print("[Widget Test] Timeline & History")
    history = DiagnosticHistory()
    # Add a couple of iterations
    history.add(DiagnosticReport(MCIScore(0.9, 0.9, 0.9, 0.9, ""), ["Msg 1"]))
    history.add(DiagnosticReport(MCIScore(0.5, 0.5, 0.5, 0.5, ""), ["Msg 2"]))
    
    panel.set_history_object(history)
    assert panel.timeline.ax.has_data(), "Timeline should have data points"
    
    # 2. Test Objection Tree
    print("[Widget Test] Objection Tree")
    panel.obj_tree.update_objections(["Model Fit: Too low", "Stability: Fluctuating"])
    assert panel.obj_tree.topLevelItemCount() > 0, "Objection tree should have categories"
    
    # 3. Test Strictness Control & Audit Log
    print("[Widget Test] Strictness Control & Audit")
    log_path = os.path.expanduser("~/.statelix/audit.log")
    if os.path.exists(log_path): os.remove(log_path)
    
    # Find the radio button for NORMAL and click it
    found = False
    for i in range(panel.strict_ctrl.group.buttons().__len__()):
        btn = panel.strict_ctrl.group.buttons()[i]
        if "NORMAL" in btn.text():
            btn.click()
            found = True
            break
    
    assert found, "NORMAL radio button not found"
    assert panel.strict_ctrl.current_mode == GovernanceMode.NORMAL
    assert os.path.exists(log_path), "Audit log MUST be created when strictness is lowered"
    
    with open(log_path, "r") as f:
        log_content = f.read()
        assert "GOVERNANCE_DEGRADED" in log_content
        assert "STRICT -> NORMAL" in log_content

    print("SUCCESS: New widget logic verified.")

if __name__ == "__main__":
    test_gui_governance()
    test_new_widgets()
