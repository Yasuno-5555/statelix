
import numpy as np
import statelix
from statelix.bayes import BayesianLinearRegression
import time

def test_map_zigen():
    print("=== Testing MAP Estimation with Zigen Autodiff ===")
    
    # 1. Generate Synthetic Data
    np.random.seed(42)
    n_samples = 500
    n_features = 3
    
    # True parameters
    true_beta = np.array([2.0, -1.5, 0.5])
    true_sigma = 1.2
    
    X = np.random.randn(n_samples, n_features)
    # Add intercept column? BayesianLinearRegression assumes X contains all features.
    # Usually users add intercept. Let's add it or just assume centered.
    # The implementation uses simple X * beta.
    
    y = X @ true_beta + np.random.randn(n_samples) * true_sigma
    
    # 2. Initialize Model
    model = BayesianLinearRegression(X, y)
    
    # Configure Priors (match generation somewhat)
    model.prior_beta_std = 10.0
    model.prior_sigma_scale = 5.0
    
    # 3. Fit using Legacy method (Verification Baseline)
    print("\n--- Running Legacy fit() ---")
    start_time = time.time()
    try:
        model.fit() # This uses the simple GD implemented in C++
        legacy_theta = model.map_theta
        legacy_time = time.time() - start_time
        print(f"Legacy Fit Time: {legacy_time:.4f}s")
        print(f"Legacy Theta: {legacy_theta}")
    except Exception as e:
        print(f"Legacy fit failed: {e}")
        legacy_theta = None

    # Reset model? map_theta is overwritten.
    
    # 4. Fit using Zigen Autodiff
    print("\n--- Running Zigen fit_autodiff() ---")
    start_time = time.time()
    try:
        model.fit_autodiff()
        zigen_theta = model.map_theta
        zigen_time = time.time() - start_time
        print(f"Zigen Fit Time: {zigen_time:.4f}s")
        print(f"Zigen Theta: {zigen_theta}")
        
    except AttributeError:
        print("Error: fit_autodiff method not found. Ensure bindings are updated.")
        return
    except Exception as e:
        print(f"Zigen fit failed: {e}")
        return

    # 5. Compare Results
    # Theta structure: [beta_0, beta_1, beta_2, log_sigma]
    
    print("\n--- Comparison ---")
    print(f"{'Parameter':<10} {'True':<10} {'Legacy':<10} {'Zigen':<10}")
    
    # Betas
    for i in range(n_features):
        l_val = legacy_theta[i] if legacy_theta is not None else float('nan')
        z_val = zigen_theta[i]
        t_val = true_beta[i]
        print(f"Beta_{i:<5} {t_val:<10.4f} {l_val:<10.4f} {z_val:<10.4f}")
        
    # Sigma
    true_log_sigma = np.log(true_sigma)
    l_log_sigma = legacy_theta[n_features] if legacy_theta is not None else float('nan')
    z_log_sigma = zigen_theta[n_features]
    print(f"LogSigma  {true_log_sigma:<10.4f} {l_log_sigma:<10.4f} {z_log_sigma:<10.4f}")
    
    # Assertions
    mse_beta = np.mean((zigen_theta[:n_features] - true_beta)**2)
    print(f"\nZigen Beta MSE: {mse_beta:.6f}")
    
    if mse_beta < 0.05:
        print("✅ SUCCESS: Zigen MAP estimation is accurate.")
    else:
        print("❌ FAILURE: Zigen MAP estimation logic might be wrong.")

if __name__ == "__main__":
    test_map_zigen()
