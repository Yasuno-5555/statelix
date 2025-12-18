import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statelix_py.models.causal import PropensityScoreMatching
import time

def test_psm_logistic():
    print("Testing PSM with new Logistic backend...")
    np.random.seed(42)
    n = 10000
    p = 10
    X = np.random.randn(n, p)
    # True coefs
    beta = np.random.randn(p) * 0.5
    z = X @ beta
    prob = 1 / (1 + np.exp(-z))
    D = (np.random.rand(n) < prob).astype(float)
    
    psm = PropensityScoreMatching()
    
    start_time = time.time()
    # Estimate propensity scores using Logistic Regression (which now uses WeightedSolver)
    ps_result = psm._estimate_propensity(D, X)
    end_time = time.time()
    
    print(f"Estimation took {end_time - start_time:.4f} seconds for N={n}, P={p}")
    print("First 5 Coefficients:", ps_result.coef[:5])
    
    # Simple check for correlation
    est_prob = ps_result.scores
    correlation = np.corrcoef(prob, est_prob)[0, 1]
    print(f"Correlation with true probs: {correlation:.4f}")
    
    if correlation > 0.95:
        print("[PASS] PSM Logistic works significantly well.")
    else:
        print(f"[FAIL] PSM Logistic correlation too low: {correlation}")

if __name__ == "__main__":
    test_psm_logistic()
