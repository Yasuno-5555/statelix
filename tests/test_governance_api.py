
import sys
import os
import numpy as np
import pytest

# Ensure we can import statelix_py
sys.path.append(os.path.abspath("statelix_py"))

try:
    from statelix_py import fit_and_judge, StatelixOLS, GovernanceMode, ModelRejectedError
    from statelix_py.utils.report_generator import ReportGenerator
except ImportError:
    sys.path.append(os.getcwd())
    from statelix_py import fit_and_judge, StatelixOLS, GovernanceMode, ModelRejectedError
    from statelix_py.utils.report_generator import ReportGenerator

def test_governance_api():
    print("\n--- Testing Phase 7: Governance Consolidation ---")
    
    # Random Data (Noise)
    X = np.random.rand(100, 1)
    y = np.random.normal(0, 10, 100)
    
    # 1. Test One-Line API (Strict - Should Fail)
    print("\n[Case 1] fit_and_judge (Strict Mode)")
    try:
        fit_and_judge(StatelixOLS, X, y, mode=GovernanceMode.STRICT)
        assert False, "Should have rejected noise in STRICT mode"
    except ModelRejectedError as e:
        print(f"Caught expected rejection: {e}")
        assert "MCI" in str(e)

    # 2. Test One-Line API (Exploratory - Should Pass)
    print("\n[Case 2] fit_and_judge (Exploratory Mode)")
    model = fit_and_judge(StatelixOLS, X, y, mode=GovernanceMode.EXPLORATORY)
    print(f"Model accepted in Exploratory mode. MCI: {model.mci}")
    assert model.mci < 0.8, "Noise should have low MCI even if accepted"
    
    # 3. Refusal Report Generation
    print("\n[Case 3] Refusal Report Generation")
    try:
        fit_and_judge(StatelixOLS, X, y, mode=GovernanceMode.STRICT, save_report_on_refusal=True)
    except ModelRejectedError:
        pass
        
    assert os.path.exists("refusal_report.html")
    print("Refusal report file verified.")
    with open("refusal_report.html") as f:
        content = f.read()
        assert "Statelix Refusal Report" in content
        assert "Integrity Statement" in content
        
    # Clean up
    if os.path.exists("refusal_report.html"):
        os.remove("refusal_report.html")
        
    print("\nSUCCESS: Governance API verified.")

if __name__ == "__main__":
    test_governance_api()
