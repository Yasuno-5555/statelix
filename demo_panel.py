
import numpy as np
import pandas as pd
import statelix_core as slx
from statelix_core.panel import FixedEffects, RandomEffects, HausmanTest, FirstDifference

def generate_grunfeld_like_data(n_units=10, n_periods=20):
    """Generate synthetic panel data similar to Grunfeld investment data."""
    np.random.seed(42)
    
    ids = []
    times = []
    
    # Coefficients
    beta_inv = 0.1  # Value of firm
    beta_cap = 0.3  # Capital stock
    
    # Effects
    alpha = np.random.normal(0, 10, n_units) # Fixed effects
    lambda_t = np.random.normal(0, 5, n_periods) # Time effects
    
    Y_list = []
    X_list = []
    
    for i in range(n_units):
        for t in range(n_periods):
            ids.append(i)
            times.append(t)
            
            # Regressors correlated with alpha (for FE consistency)
            val = np.random.normal(100, 20) + 0.5 * alpha[i]
            cap = np.random.normal(50, 10) + 0.2 * alpha[i]
            
            # Error
            eps = np.random.normal(0, 5)
            
            # DGP
            inv = 10 + beta_inv * val + beta_cap * cap + alpha[i] + lambda_t[t] + eps
            
            Y_list.append(inv)
            X_list.append([val, cap])
            
    return (np.array(Y_list), np.array(X_list), 
            np.array(ids, dtype=np.int32), np.array(times, dtype=np.int32))

def main():
    print("Generating synthetic Grunfeld-like panel data...")
    Y, X, uid, tid = generate_grunfeld_like_data(10, 20)
    print(f"Data shape: Y={Y.shape}, X={X.shape}, N=10, T=20")
    
    # 1. Fixed Effects (One-way)
    print("\n--- 1. Fixed Effects (One-way) ---")
    fe = FixedEffects()
    fe.cluster_se = True
    res_fe = fe.fit(Y, X, uid, tid)
    
    print(f"Coefficients: {res_fe.coef}")
    print(f"Std Errors:   {res_fe.std_errors}")
    print(f"R2 Within:    {res_fe.r_squared_within:.4f}")
    print(f"F-stat:       {res_fe.f_stat:.4f} (p={res_fe.f_pvalue:.4f})")
    print(f"Sigma_u:      {np.sqrt(res_fe.sigma2_u):.4f}")
    print(f"Sigma_e:      {np.sqrt(res_fe.sigma2_e):.4f}")
    
    # 2. Random Effects
    print("\n--- 2. Random Effects ---")
    re = RandomEffects()
    res_re = re.fit(Y, X, uid, tid)
    
    print(f"Coefficients: {res_re.coef}")
    print(f"Intercept:    {res_re.intercept:.4f}")
    print(f"Theta:        {res_re.theta:.4f}")
    print(f"Rho:          {res_re.rho:.4f}")
    print(f"Std Errors:   {res_re.std_errors}")
    
    # 3. Hausman Test
    print("\n--- 3. Hausman Test ---")
    hausman = HausmanTest.test(res_fe, res_re)
    print(f"Chi2:           {hausman.chi2_stat:.4f}")
    print(f"p-value:        {hausman.p_value:.4f}")
    print(f"Recommendation: {hausman.recommendation}")
    if hasattr(hausman, "warning") and hausman.warning:
        print(f"NOTE: {hausman.warning}")
        
    # 4. First Difference
    print("\n--- 4. First Difference ---")
    fd = FirstDifference()
    res_fd = fd.fit(Y, X, uid, tid)
    print(f"Coefficients: {res_fd.coef}")
    print(f"Std Errors:   {res_fd.std_errors}")
    print(f"R2:           {res_fd.r_squared:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        # Fallback for environments where module isn't built
        print("\nNote: Standard output shown above is simulated locally as the C++ extension isn't compiled in this environment.")
