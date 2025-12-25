
import pytest
from statelix_py.diagnostics.analyst_engine import AnalystEngine
from statelix_py.diagnostics.critic import DiagnosticReport, MCIScore

def test_analyst_grounding():
    mci = MCIScore(0.3, 0.4, 0.2, 0.3, "Fit Fail")
    report = DiagnosticReport(mci, ["Poor fit", "Topo Collapse"], ["Add interactions"])
    summary = "### 1. Analysis Facts\n- X1: Positive\n### 2. Interpretation Hints\n- Noise risk\n### 3. Conclusion\nNone"
    
    engine = AnalystEngine(report, summary)
    
    # Check MCI explanation
    ans_mci = engine.answer("Why was it rejected?")
    assert "0.30" in ans_mci
    assert "Rejected" in ans_mci
    assert "Topo Collapse" in ans_mci
    
    # Check Suggestions
    ans_fix = engine.answer("How to fix this?")
    assert "Add interactions" in ans_fix
    
    # Check Facts
    ans_facts = engine.answer("What are the facts?")
    assert "Analysis Facts" in ans_facts
    assert "X1: Positive" in ans_facts
    
    # Check Fallback
    ans_rand = engine.answer("Tell me a joke")
    assert "I can explain" in ans_rand

if __name__ == "__main__":
    test_analyst_grounding()
    print("Analyst Engine tests PASSED.")
