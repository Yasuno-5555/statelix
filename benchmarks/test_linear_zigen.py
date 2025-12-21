
import numpy as np
import time
from statelix.linear_model import FitOLS
from sklearn.linear_model import LinearRegression as SklearnLR

def test_linear_gemm():
    print("=== Testing Linear Regression with Zigen GEMM ===")
    
    # Large Data (Should trigger Zigen path if p >= 50 && n >= 100)
    # N=2000, P=100
    n_large = 2000
    p_large = 100
    
    print(f"\n--- Large Data Test (N={n_large}, P={p_large}) ---")
    np.random.seed(42)
    X_large = np.random.randn(n_large, p_large)
    true_beta_large = np.random.randn(p_large + 1) # +1 for intercept
    y_large = X_large @ true_beta_large[1:] + true_beta_large[0] + np.random.randn(n_large) * 0.1
    
    model = FitOLS()
    
    # Warmup
    model.fit(X_large[:100], y_large[:100])
    
    start = time.time()
    model.fit(X_large, y_large)
    end = time.time()
    
    print(f"Statelix Time: {end - start:.4f}s")
    print(f"R-squared: {model.r_squared:.4f}")
    
    coef_statelix = model.coef
    
    # Verify correctness against sklearn
    sk_model = SklearnLR()
    start_sk = time.time()
    sk_model.fit(X_large, y_large)
    end_sk = time.time()
    print(f"Sklearn Time:  {end_sk - start_sk:.4f}s")
    
    coef_sk = sk_model.coef_
    
    mse_coef = np.mean((coef_statelix - coef_sk)**2)
    print(f"Coef MSE vs Sklearn: {mse_coef:.8f}")
    
    if mse_coef < 1e-10:
        print("✅ Large Data Correctness Passed")
    else:
        print("❌ Large Data Correctness Failed")

    # Small Data (Fallback to Eigen)
    # N=50, P=10
    n_small = 50
    p_small = 10
    print(f"\n--- Small Data Test (N={n_small}, P={p_small}) ---")
    
    X_small = np.random.randn(n_small, p_small)
    true_beta_small = np.random.randn(p_small + 1)
    y_small = X_small @ true_beta_small[1:] + true_beta_small[0] + np.random.randn(n_small) * 0.1
    
    model.fit(X_small, y_small)
    
    coef_statelix_small = model.coef
    
    sk_model.fit(X_small, y_small)
    coef_sk_small = sk_model.coef_
    
    mse_coef_small = np.mean((coef_statelix_small - coef_sk_small)**2)
    print(f"Coef MSE vs Sklearn: {mse_coef_small:.8f}")

    if mse_coef_small < 1e-10:
        print("✅ Small Data Correctness Passed")
    else:
        print("❌ Small Data Correctness Failed")

if __name__ == "__main__":
    test_linear_gemm()
