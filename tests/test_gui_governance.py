
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

if __name__ == "__main__":
    test_gui_governance()
