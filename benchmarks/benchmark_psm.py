import time
import numpy as np
import pandas as pd
from statelix_py.models.causal import PropensityScoreMatching
try:
    import statelix_core
    HAS_CPP = True
    print("Statelix Core C++ module loaded.")
except ImportError:
    HAS_CPP = False
    print("Statelix Core C++ module NOT loaded.")

def generate_data(n=50000):
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(n, 10))
    # True propensity score
    true_ps = 1 / (1 + np.exp(-(0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2])))
    D = np.random.binomial(1, true_ps)
    # Outcome
    Y = 1.0 + 2.0 * D + 0.5 * X[:, 0] + np.random.normal(0, 1, size=n)
    return Y, D, X

def run_benchmark():
    print(f"Generating data...")
    Y, D, X = generate_data(n=50000)
    print(f"Data generated: N={len(Y)}")
    
    # Python Baseline (simulated by disabling cpp usage if possible, or just note it)
    # Note: The python implementation in causal.py attempts to use C++ if available.
    # To test pure python, we'd need to force use_cpp=False.
    
    if HAS_CPP:
        print("\n--- Running with C++ Backend (OpenMP expected) ---")
        start = time.time()
        psm = PropensityScoreMatching(method='nearest_neighbor', caliper=0.2)
        psm.fit(Y, D, X, use_cpp=True)
        end = time.time()
        print(f"C++ Time: {end - start:.4f} seconds")
        print(f"ATT: {psm.att:.4f} (True ~2.0)")
    
    print("\n--- Running with Pure Python (Approximation) ---")
    start = time.time()
    # Force pure python by setting use_cpp=False
    psm_py = PropensityScoreMatching(method='nearest_neighbor', caliper=0.2)
    psm_py.fit(Y, D, X, use_cpp=False)
    end = time.time()
    print(f"Python Time: {end - start:.4f} seconds")
    print(f"ATT: {psm_py.att:.4f} (True ~2.0)")

if __name__ == "__main__":
    run_benchmark()
