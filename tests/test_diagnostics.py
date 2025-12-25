
import sys
import os

# Ensure we can import statelix_py
sys.path.append(os.path.abspath("statelix_py"))


try:
    # If running from root, 'statelix_py' is a top-level package
    from statelix_py.diagnostics.critic import ModelCritic, CriticMode
except ImportError:
    # If that fails, try appending the current directory explicitly
    # But wait, looking at the layout: statelix_py/ is a folder.
    # If we append os.path.abspath("."), we can do "import statelix_py..."
    # The previous attempt did that but failed. Let's try simpler relative import or check path.
    import statelix_py.diagnostics.critic as critic_module
    ModelCritic = critic_module.ModelCritic
    CriticMode = critic_module.CriticMode

def test_diagnostics():
    print("\n--- Testing Human-Centric Diagnostics ---")
    
    # Scene 1: Good Model
    print("\n[Scenario 1] High Quality Model")
    good_metrics = {
        'r2': 0.98,
        'mean_structure': 5.0,
        'std_structure': 0.1, # Low variance -> High Stability
        'invariant_ratio': 1.0
    }
    critic = ModelCritic(mode=CriticMode.DIAGNOSTIC)
    report = critic.critique(good_metrics)
    print(report)
    
    assert report.mci.total_score > 0.9, "Good model should have high MCI"
    assert len(report.messages) == 0, "Good model should have no objections"
    
    # Scene 2: Bad Model (Poor Fit + Unstable Topology)
    print("\n[Scenario 2] Broken Model")
    bad_metrics = {
        'r2': 0.4,
        'mean_structure': 5.0,
        'std_structure': 2.0, # High variance -> Low Stability
        'invariant_ratio': 0.6
    }
    
    # Use Suggestive Mode
    critic_sugg = ModelCritic(mode=CriticMode.SUGGESTIVE)
    report_bad = critic_sugg.critique(bad_metrics)
    print(report_bad)
    
    assert report_bad.mci.total_score < 0.6, "Bad model should have low MCI"
    assert len(report_bad.messages) > 0, "Bad model must have objections"
    assert len(report_bad.suggestions) > 0, "Suggestive mode must provide helpful tips"
    
    print("\nSUCCESS: Diagnostics layer verified.")

if __name__ == "__main__":
    test_diagnostics()
