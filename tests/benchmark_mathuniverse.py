import time
import numpy as np
from statelix.bayes import BayesianLinearRegression
from statelix.spatial import StatelixICP
import sys

def benchmark_bayes():
    print("--- Bayesian Regression Benchmark ---")
    np.random.seed(42)
    n, p = 500, 10
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    y = X @ beta_true + np.random.randn(n) * 0.5
    
    model = BayesianLinearRegression(X, y)
    
    # 1. Reverse Mode (Zigen)
    start = time.time()
    model.fit_autodiff()
    res_rev = model.map_theta
    time_rev = time.time() - start
    print(f"Reverse Mode (Zigen) Time: {time_rev:.4f}s")
    
    # 2. Forward Mode (MathUniverse Dual)
    start = time.time()
    model.fit_dual()
    res_dual = model.map_theta
    time_dual = time.time() - start
    print(f"Forward Mode (MathUniverse Dual) Time: {time_dual:.4f}s")
    
    print(f"Speedup: {time_rev / time_dual:.2f}x")
    print(f"L2 diff between results: {np.linalg.norm(res_rev - res_dual):.6f}")

def benchmark_icp():
    print("\n--- ICP Alignment Benchmark ---")
    np.random.seed(42)
    n = 1000
    source = np.random.randn(n, 3)
    
    # Random rotation
    theta = np.pi / 4
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    t_true = np.array([1, 2, 3])
    target = (source @ R_true.T) + t_true + np.random.randn(n, 3) * 0.05
    
    icp = StatelixICP()
    icp.max_iter = 50
    icp.tol = 1e-6
    
    # 1. Standard Kabsch (Eigen SVD)
    start = time.time()
    res_std = icp.align(source, target)
    time_std = time.time() - start
    print(f"Standard ICP (Eigen SVD) Time: {time_std:.4f}s")
    
    # 2. Shinen Optimized
    start = time.time()
    res_shinen = icp.align_shinen(source, target)
    time_shinen = time.time() - start
    print(f"Shinen Optimized ICP Time: {time_shinen:.4f}s")
    
    print(f"Speedup: {time_std / time_shinen:.2f}x")
    print(f"RMSE (Std): {res_std.rmse:.6f}")
    print(f"RMSE (Shinen): {res_shinen.rmse:.6f}")

if __name__ == "__main__":
    # Ensure statelix_py is in path
    import os
    sys.path.append(os.path.abspath("."))
    benchmark_bayes()
    benchmark_icp()
