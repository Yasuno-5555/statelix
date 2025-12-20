import sys
import os
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statelix_py.inquiry.narrative import Storyteller
from statelix_py.gui.panels.inquiry_panel import InquiryPanel

def test_storyteller():
    print("--- Testing Storyteller (Student Mode) ---")
    
    # Mock Linear Model (OLS-like)
    class MockAdapter:
        def get_coefficients(self):
            return {"Education": 0.5, "Age": -0.2, "Constant": 1.0}
        def get_metrics(self):
            return {"r2": 0.75, "effect": 0.5}
        def get_assumptions(self):
            return [] # No causal assumptions for OLS usually, or ignored

    # Mock Storyteller with mocked adapter
    story = Storyteller("mock_model") 
    story.adapter = MockAdapter() # Inject mock
    
    narrative = story.explain()
    print(narrative)
    
    # Verification Checks
    assert "### 1. Analysis Facts" in narrative, "Missing Fact Section"
    assert "### 2. Interpretation Hints" in narrative, "Missing Hint Section"
    assert "### 3. Conclusion" in narrative, "Missing Conclusion Section"
    assert "Education" in narrative
    assert "positive (+)" in narrative or "positive" in narrative
    print(">>> Storyteller structure verified.")

def test_inquiry_panel_export():
    print("\n--- Testing Inquiry Panel Export ---")
    app = QApplication.instance() or QApplication(sys.argv)
    panel = InquiryPanel()
    
    # Set some narrative
    panel.set_narrative("<h1>Test Analysis</h1><p>Facts...</p>")
    
    # We can't easily click the button without blocking, 
    # but we can verify the text content exists in the box.
    html = panel.narrative_box.toHtml()
    assert "Facts" in html
    print(">>> Inquiry Panel narrative set correctly.")

if __name__ == "__main__":
    try:
        test_storyteller()
        test_inquiry_panel_export()
        print("\nALL TESTS PASSED.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
