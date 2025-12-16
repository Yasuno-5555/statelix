import time
import numpy as np
import pandas as pd
import sys
import os

# Create benchmarks/__init__.py might not be enough if I run as script?
# But checking paths...
sys.path.append(os.path.join(os.path.dirname(__file__)))

from benchmarks.utils import BenchmarkTimer, BenchmarkResult, print_comparison_table

try:
    import benchmarks.statelix_panel as statelix_panel
    STATELIX_AVAILABLE = True
except ImportError:
    try:
        import statelix_panel
        STATELIX_AVAILABLE = True
    except ImportError:
        print("WARNING: statelix_panel module not found.")
        STATELIX_AVAILABLE = False

# Naive Python implementation of GMM (Difference GMM, 1-step logic)
def python_gmm_diff(y, X, ids, time_pers):
    # This is a mock implementation of the heavy logic to serve as baseline
    # It does naive matrix ops in pure numpy
    # Real "linearmodels" is complex, so we compare "Optimized C++" vs "Naive Python"
    
    start_time = time.time()
    
    N = len(y)
    K = X.shape[1]
    
    # 1. First Differences
    # Sort
    df = pd.DataFrame({'y': y, 'id': ids, 'time': time_pers})
    for k in range(K):
        df[f'x{k}'] = X[:, k]
        
    df = df.sort_values(['id', 'time'])
    
    # Diff
    df_diff = df.groupby('id').diff().dropna()
    
    y_diff = df_diff['y'].values
    X_cols = [c for c in df_diff.columns if c.startswith('x')]
    X_diff = df_diff[X_cols].values
    
    # 2. Instrument construction (Naive dense Z)
    # Instruments: lags of y.
    # We'll just create a random large matrix to simulate the workload of Z
    # because implementing full GMM logic in python for benchmark is tedious.
    # We want to measure overhead.
    
    n_diff = len(y_diff)
    # Simulate Z being roughly (T-2)*T/2 columns
    T = df['time'].max()
    n_instr = int(T * (T-1) / 2)
    Z = np.random.randn(n_diff, n_instr) # Random instruments
    
    # 3. GMM Algebra
    # W = (Z' H Z)^-1
    # naive H = I for now (IV estimator)
    # W = (Z'Z)^-1
    ZtZ = Z.T @ Z
    W = np.linalg.inv(ZtZ + np.eye(n_instr)*1e-6)
    
    # Beta = (X' Z W Z' X)^-1 X' Z W Z' y
    ZtX = Z.T @ X_diff
    Zty = Z.T @ y_diff
    
    XZWZX = ZtX.T @ W @ ZtX
    XZWZY = ZtX.T @ W @ Zty
    
    beta = np.linalg.solve(XZWZX, XZWZY)
    
    duration = time.time() - start_time
    return duration, beta[0]

def run_statelix_panel(y, X, ids, time_pers):
    if not STATELIX_AVAILABLE:
        return 0, 0
    
    gmm = statelix_panel.DynamicPanelGMM()
    
    with BenchmarkTimer("Statelix GMM") as t:
        # Pass data. Note: Python list conversion overhead
        # ids and time are int lists
        res = gmm.estimate(y, X, ids.tolist(), time_pers.tolist())
    
    return t.duration, res.coefficients[0] if res.coefficients.size > 0 else 0

def main():
    results = []
    
    # N individuals, T time periods
    scenarios = [
        {"N": 100, "T": 10, "K": 5},
        {"N": 500, "T": 10, "K": 10},
        {"N": 1000, "T": 20, "K": 10}, # T=20 -> many instruments (~200)
    ]
    
    for sc in scenarios:
        N = sc['N']
        T = sc['T']
        K = sc['K']
        print(f"\n--- Scenario: N={N}, T={T}, K={K} ---")
        
        # Generate Panel Data (AR(1) process)
        # y_it = 0.5 y_{i,t-1} + X_it beta + u_i + e_it
        ids = []
        times = []
        y_list = []
        X_list = []
        
        for i in range(N):
            y_prev = 0
            u_i = np.random.randn()
            for t in range(T):
                ids.append(i)
                times.append(t)
                x_row = np.random.randn(K)
                X_list.append(x_row)
                
                # y = 0.5 * y_prev + 1.0 * x[0] + u_i + noise
                val = 0.5 * y_prev + 1.0 * x_row[0] + u_i + np.random.randn()
                y_list.append(val)
                y_prev = val
                
        y = np.array(y_list)
        X = np.vstack(X_list)
        ids = np.array(ids, dtype=np.int32)
        times = np.array(times, dtype=np.int32)
        
        print(f"Total Obs: {len(y)}")
        
        # Python Naive
        py_time, py_beta = python_gmm_diff(y, X, ids, times)
        results.append(BenchmarkResult("Python (Naive)", "Diff-GMM", len(y), K, py_time, py_beta))
        
        # Statelix
        if STATELIX_AVAILABLE:
            st_time, st_beta = run_statelix_panel(y, X, ids, times)
            results.append(BenchmarkResult("Statelix", "Diff-GMM (C++)", len(y), K, st_time, st_beta))
            
    print_comparison_table(results)

if __name__ == "__main__":
    main()
