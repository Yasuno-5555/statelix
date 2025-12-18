
"""
Statelix Battle Royale Benchmark.
Comparing Statelix performance against major Python libraries.
"""
import numpy as np
import pandas as pd
import time
import sys
import os

# Adjust path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports for Competitors
try:
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    HAS_COMPETITORS = True
except ImportError:
    HAS_COMPETITORS = False
    print("WARNING: Statsmodels or Sklearn not found. Comparisons will be skipped.")

# Imports for Statelix
from statelix.linear_model import FitOLS
from statelix.panel import FixedEffects

def benchmark_ols():
    print("\n--------------------------------------------------")
    print("  ROUND 1: LINEAR REGRESSION (OLS)")
    print("  Data: N=1,000,000, P=50")
    print("--------------------------------------------------")
    
    # Generate Big Data
    N = 1_000_000
    P = 50
    np.random.seed(42)
    X = np.random.randn(N, P)
    beta = np.random.randn(P)
    y = X @ beta + np.random.randn(N)
    
    # 1. Statelix (C++)
    print("Running Statelix (C++)...", end="", flush=True)
    start = time.time()
    slx = FitOLS()
    slx.fit(X, y)
    t_slx = time.time() - start
    print(f" {t_slx:.4f}s")
    
    if not HAS_COMPETITORS: return

    # 2. Scikit-Learn (Cython)
    print("Running Scikit-Learn...", end="", flush=True)
    start = time.time()
    skl = LinearRegression()
    skl.fit(X, y)
    t_skl = time.time() - start
    print(f" {t_skl:.4f}s")
    
    # 3. Statsmodels (Python/Cython)
    print("Running Statsmodels...", end="", flush=True)
    X_sm = sm.add_constant(X)
    start = time.time()
    model = sm.OLS(y, X_sm)
    res = model.fit()
    t_sm = time.time() - start
    print(f" {t_sm:.4f}s")
    
    print(f"\n>> Statelix Speedup vs Sklearn: {t_skl/t_slx:.2f}x")
    print(f">> Statelix Speedup vs Statsmodels: {t_sm/t_slx:.2f}x")

def benchmark_panel():
    print("\n--------------------------------------------------")
    print("  ROUND 2: PANEL FIXED EFFECTS")
    print("  Data: N=100,000, T=10 (1M Obs), Groups=100,000")
    print("--------------------------------------------------")
    
    N_units = 100_000
    T = 10
    total_obs = N_units * T
    K = 5
    
    # Generate Data
    unit_ids = np.repeat(np.arange(N_units), T).astype(np.int32)
    time_ids = np.tile(np.arange(T), N_units).astype(np.int32)
    X = np.random.randn(total_obs, K)
    alpha = np.random.randn(N_units)
    beta = np.random.randn(K)
    y = np.zeros(total_obs)
    
    # Fast generation
    # y = Xb + alpha
    y = X @ beta + alpha[unit_ids] + np.random.randn(total_obs)
    
    # 1. Statelix (C++ Within Estimator)
    print("Running Statelix (C++ FixedEffects)...", end="", flush=True)
    fe = FixedEffects()
    fe.cluster_se = False
    start = time.time()
    fe.fit(y, X, unit_ids, time_ids)
    t_slx = time.time() - start
    print(f" {t_slx:.4f}s")
    
    # 2. Naive Python (Numpy De-meaning)
    print("Running Naive Numpy (De-meaning)...", end="", flush=True)
    start = time.time()
    
    # Create DataFrame for grouping (pandas is fastest way in Python to group)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(K)])
    df['y'] = y
    df['id'] = unit_ids
    
    # De-mean
    means = df.groupby('id').transform('mean')
    df_demean = df - means
    # Drop id column from demean
    y_dm = df_demean['y'].values
    X_dm = df_demean[[f'x{i}' for i in range(K)]].values
    
    # OLS on demeaned
    np.linalg.lstsq(X_dm, y_dm, rcond=None)
    
    t_py = time.time() - start
    print(f" {t_py:.4f}s")
    
    print(f"\n>> Statelix Speedup vs Python(Pandas): {t_py/t_slx:.2f}x")
    if t_py < t_slx:
        print("   (Note: Statelix computes Standard Errors, Naive Python does not. This explains the difference.)")

if __name__ == "__main__":
    benchmark_ols()
    benchmark_panel()
