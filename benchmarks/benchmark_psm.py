import time
import numpy as np
import pandas as pd
import sys
import os

# Ensure local .so files in benchmarks/ can be imported
sys.path.append(os.path.join(os.path.dirname(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from benchmarks.utils import BenchmarkTimer, BenchmarkResult, print_comparison_table, generate_synthetic_data

# Import Statelix PSM (try/except for local dev fallback)
try:
    import benchmarks.statelix_psm as statelix_psm
    STATELIX_AVAILABLE = True
except ImportError:
    try:
        import statelix_psm
        STATELIX_AVAILABLE = True
    except ImportError:
        print("WARNING: statelix_psm module not found. Statelix benchmark will be skipped.")
        STATELIX_AVAILABLE = False

def run_sklearn_psm(X, D, Y):
    # 1. Estimate Propensity Score
    with BenchmarkTimer("Sklearn PS Estimation") as t_ps:
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X, D)
        ps_scores = lr.predict_proba(X)[:, 1]
    
    # 2. Match (Nearest Neighbor)
    with BenchmarkTimer("Sklearn Matching") as t_match:
        treated_mask = (D == 1)
        control_mask = (D == 0)
        
        X_treated = ps_scores[treated_mask].reshape(-1, 1)
        X_control = ps_scores[control_mask].reshape(-1, 1)
        
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_control)
        distances, indices = nn.kneighbors(X_treated)
        
        # Calculate ATT
        y_treated = Y[treated_mask]
        y_matched_control = Y[control_mask][indices.flatten()]
        att = np.mean(y_treated - y_matched_control)
        
    return t_ps.duration + t_match.duration, att

def run_statelix_psm(X, D, Y):
    if not STATELIX_AVAILABLE:
        return 0, 0

    # 1. Estimate Propensity Score
    psm = statelix_psm.PropensityScoreMatching()
    
    with BenchmarkTimer("Statelix PS Estimation") as t_ps:
        # Statelix expects Eigen vectors, pybind11 handles numpy conversion
        ps_result = psm.estimate_propensity(D, X)
    
    # 2. Match
    with BenchmarkTimer("Statelix Matching") as t_match:
        match_result = psm.match(Y, D, X, ps_result)
        att = match_result.att
        
    return t_ps.duration + t_match.duration, att

def main():
    results = []
    
    # Define scenarios (Reduced sizes for Docker limits)
    scenarios = [
        {"n_samples": 1000, "n_features": 10},
        {"n_samples": 5000, "n_features": 10},
        {"n_samples": 10000, "n_features": 20},
    ]
    
    print(f"Running Loop: {len(scenarios)} scenarios...")
    
    for sc in scenarios:
        n = sc['n_samples']
        p = sc['n_features']
        print(f"\n--- Scenario: N={n}, P={p} ---")
        
        X, D, Y = generate_synthetic_data(n_samples=n, n_features=p, task='causal')
        print("Data generated.")
        
        # Scikit-learn
        print("Running Sklearn...")
        sk_time, sk_att = run_sklearn_psm(X, D, Y)
        results.append(BenchmarkResult("Scikit-learn", "PSM (LogReg+NN)", n, p, sk_time, sk_att))
        
        # Statelix
        if STATELIX_AVAILABLE:
            print("Running Statelix...")
            st_time, st_att = run_statelix_psm(X, D, Y)
            results.append(BenchmarkResult("Statelix", "PSM (C++ Optimized)", n, p, st_time, st_att))
            print(f"ATT Comparison: Sklearn={sk_att:.4f}, Statelix={st_att:.4f}")
        
    print_comparison_table(results)

if __name__ == "__main__":
    main()
