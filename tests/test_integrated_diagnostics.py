
import sys
import os
import numpy as np
import pytest

# Ensure we can import statelix_py
sys.path.append(os.path.abspath("statelix_py"))

try:
    from statelix_py.models.linear import StatelixOLS
    from statelix_py.models.bayes import BayesianLogisticRegression
    from statelix_py.diagnostics.critic import ModelRejectedError
except ImportError:
    # Adjust imports if necessary (though env should be set)
    sys.path.append(os.getcwd())
    from statelix_py.models.linear import StatelixOLS
    from statelix_py.models.bayes import BayesianLogisticRegression
    from statelix_py.diagnostics.critic import ModelRejectedError

def test_integrated_diagnostics():
    print("\n--- Testing Phase 5: Integrated Diagnostics & Veto Power ---")
    
    # 1. Linear Regression with Good Data
    print("\n[Scenario 1] Linear Regression (Good Data)")
    X = np.random.rand(100, 1)
    y = 3 * X[:, 0] + np.random.normal(0, 0.1, 100) # Strong linear relationship
    
    fit_model = StatelixOLS(strict_threshold=0.5)
    fit_model.fit(X, y)
    
    print(f"MCI: {fit_model.mci}")
    assert fit_model.mci > 0.8, "Good OLS should have high MCI"
    assert len(fit_model.objections) == 0, "Good OLS should have no objections"
    
    # 2. Linear Regression with Bad Data (Pure Noise)
    print("\n[Scenario 2] Linear Regression (Strict Rejection)")
    X_noise = np.random.rand(100, 1)
    y_noise = np.random.normal(0, 10, 100) # No relationship
    
    strict_model = StatelixOLS(strict_threshold=0.8) # Very strict
    
    try:
        strict_model.fit(X_noise, y_noise)
        assert False, "Should have rejected the model due to low R2"
    except ModelRejectedError as e:
        print(f"Caught expected rejection: {e}")
        
    # 3. Bayesian Model Contract
    print("\n[Scenario 3] Bayesian Model (Contract Check)")
    # Simple binary data
    X_bin = np.random.normal(0, 1, (100, 2))
    logits = X_bin[:, 0] + X_bin[:, 1]
    probs = 1 / (1 + np.exp(-logits))
    y_bin = (np.random.rand(100) < probs).astype(float)
    
    # Using small samples for speed in test
    bayes_model = BayesianLogisticRegression(n_samples=50, warmup=10, strict_threshold=0.0)
    bayes_model.fit(X_bin, y_bin)
    
    print(f"Bayes MCI: {bayes_model.mci}")
    
    # Verify Unified Interface
    assert hasattr(bayes_model, 'mci')
    assert hasattr(bayes_model, 'objections')
    assert hasattr(bayes_model, 'suggestions')
    assert bayes_model.mci is not None
    
    print("\nSUCCESS: Integrated Diagnostics & Veto Power verified.")

    # 4. History & Stagnation Detection
    print("\n[Scenario 4] History & Stagnation Check")
    # Simulate a loop
    hist_model = StatelixOLS(strict_threshold=0.0)
    
    # Iteration 1: Bad
    hist_model.fit(np.random.rand(100, 1), np.random.normal(0, 10, 100))
    # Iteration 2: Same Bad
    hist_model.fit(np.random.rand(100, 1), np.random.normal(0, 10, 100))
    # Iteration 3: Same Bad
    hist_model.fit(np.random.rand(100, 1), np.random.normal(0, 10, 100))
    
    evolution = hist_model.history.get_evolution()
    print(f"History Length: {len(evolution)}")
    assert len(evolution) == 3
    
    is_stagnated = hist_model.history.detect_stagnation(window=3, threshold=0.1)
    print(f"Stagnated? {is_stagnated}")
    assert is_stagnated, "Should detect stagnation"
    
    print("\nSUCCESS: Diagnostic History verified.")


if __name__ == "__main__":
    test_integrated_diagnostics()
