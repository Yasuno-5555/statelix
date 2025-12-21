
import numpy as np
import time
from statelix.time_series import GARCH_Model, GARCHType, GARCHDist

def generate_garch_data(n=1000, omega=0.1, alpha=0.1, beta=0.8):
    np.random.seed(42)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    
    # Init
    sigma2[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.random.normal(0, np.sqrt(sigma2[0]))
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
        
    return returns

def test_garch_optimization():
    print("Generating GARCH(1,1) data...")
    returns = generate_garch_data(n=1000)
    
    print("\n--- Testing Zigen GARCH(1,1) Optimization ---")
    model = GARCH_Model(p=1, q=1)
    
    start_time = time.time()
    result = model.fit(returns)
    end_time = time.time()
    
    print(f"Converged: {result.converged}")
    print(f"Log-Likelihood: {result.log_likelihood:.4f}")
    print(f"Time: {end_time - start_time:.4f}s")
    
    print("\nParameters:")
    print(f"Intercept (mu): {result.mu:.4f}")
    # Python bindings expose vector params directly
    print(f"Omega: {result.omega:.4f}")
    if len(result.alpha) > 0: print(f"Alpha: {result.alpha[0]:.4f}")
    if len(result.beta) > 0: print(f"Beta: {result.beta[0]:.4f}")
    
    # Checks
    assert result.converged, "Optimization failed to converge"
    assert result.omega > 0, "Omega must be positive"
    assert result.alpha[0] >= 0, "Alpha must be positive"
    assert result.beta[0] >= 0, "Beta must be positive"
    
    # Check if close to true values (0.1, 0.1, 0.8)
    # GARCH estimation has variance, but should be reasonably close with N=1000
    print(f"Omega Error: {abs(result.omega - 0.1):.4f}")
    print(f"Alpha Error: {abs(result.alpha[0] - 0.1):.4f}")
    print(f"Beta Error: {abs(result.beta[0] - 0.8):.4f}")
    
    mse = (result.omega - 0.1)**2 + (result.alpha[0] - 0.1)**2 + (result.beta[0] - 0.8)**2
    print(f"Parameter MSE: {mse:.4f}")
    
    if mse < 0.01:
        print("\nSUCCESS: Parameters matched ground truth within tolerance.")
    else:
        print("\nWARNING: Parameters deviated from ground truth (expected with finite sample).")

if __name__ == "__main__":
    test_garch_optimization()
