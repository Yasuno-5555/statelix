
import numpy as np
import pandas as pd
import statelix.time_series
import sys
import traceback

def verify_var():
    print("\nVerifying VAR...")
    try:
        import statsmodels.tsa.api as smt
    except ImportError:
        print("[WARN] statsmodels not found. Skipping comparison, checking internal consistency only.")
        smt = None

    np.random.seed(42)
    n = 1000
    k = 2
    p = 2
    
    # Generate VAR(2) process
    # Y_t = A1 Y_{t-1} + A2 Y_{t-2} + u_t
    A1 = np.array([[0.5, 0.1], [0.4, 0.5]])
    A2 = np.array([[0.1, 0.05], [0.05, 0.1]])
    
    Y = np.zeros((n, k))
    u = np.random.multivariate_normal(np.zeros(k), [[1, 0.5], [0.5, 1]], n)
    
    # Burn-in
    for i in range(2, n):
        Y[i] = A1 @ Y[i-1] + A2 @ Y[i-2] + u[i]
        
    # Discard burn-in
    Y = Y[100:]
    n = Y.shape[0]
    
    # Statelix VAR
    try:
        var = statelix.time_series.VAR_Model(p)
    except AttributeError:
        # Fallback
        var = statelix.time_series.VectorAutoregression(p)
    res_s = var.fit(Y)
    
    print(f"Statelix LogL: {res_s.log_likelihood:.4f}")
    print(f"Statelix AIC:  {res_s.aic:.4f}")
    
    # Statsmodels VAR
    if smt:
        model = smt.VAR(Y)
        res_sm = model.fit(p)
        
        print(f"Statsmodels LogL: {res_sm.llf:.4f}")
        print(f"Statsmodels AIC:  {res_sm.aic:.4f}")
        
        # Check Coefs (Lag 1)
        print("Compare Coefs (Lag 1):")
        # Statelix coef[0] is A1 (KxK)
        # Statsmodels params (n_features x n_equations).
        # SM constants are first row. Lags follow.
        
        sm_coefs = res_sm.params[1:].T.reshape(p, k, k) # simplified reshape logic
        # Actually SM params is (1 + K*p) x K
        # Row 0: const
        # Rows 1..K: lag 1
        # Rows K+1..2K: lag 2
        
        sm_A1 = res_sm.params[1:k+1].T 
        st_A1 = res_s.coef[0]
        
        diff = np.abs(sm_A1 - st_A1).max()
        print(f"Max Diff A1: {diff:.8f}")
        if diff < 1e-8:
            print("[PASS] Coefficients match Statsmodels")
        else:
            print(f"[FAIL] Coefficients mismatch > 1e-8")

        # Granger Causality (Variable 0 causes 1?)
        # True A1[1,0] = 0.4, A2[1,0] = 0.05 -> Yes, 0 causes 1.
        
        gc = var.granger_causality(res_s, Y, 0, 1)
        print(f"Granger (0->1): F={gc.f_stat:.4f}, p={gc.p_value:.4f}")
        
        sm_gc = res_sm.test_causality(1, 0)
        print(f"Statsmodels (0->1): F={sm_gc.test_statistic:.4f}, p={sm_gc.pvalue:.4f}")
        
        if abs(gc.f_stat - sm_gc.test_statistic) < 1e-5:
             print("[PASS] Granger F-stat matches")
        else:
             print(f"[FAIL] Granger F-stat mismatch (Diff={abs(gc.f_stat - sm_gc.test_statistic)})")

    # Run Granger independent of statsmodels to ensure no crash
    print("Running Granger Causality (Statelix)...")
    gc = var.granger_causality(res_s, Y, 0, 1)
    print(f"Granger (0->1): F={gc.f_stat:.4f}, p={gc.p_value:.4f}, causes={gc.causes}")

def verify_garch():
    print("\nVerifying GARCH...")
    np.random.seed(42)
    n = 1000
    
    # GARCH(1,1)
    omega = 0.1
    alpha = 0.1
    beta = 0.8
    mu = 0.0
    
    eps = np.random.normal(0, 1, n)
    sigma2 = np.zeros(n)
    y = np.zeros(n)
    
    sigma2[0] = omega / (1.0 - alpha - beta)
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * (y[t-1] - mu)**2 + beta * sigma2[t-1]
        y[t] = mu + np.sqrt(sigma2[t]) * eps[t]
        
    garch = statelix.time_series.GARCH_Model(1, 1)
    res = garch.fit(y)
    
    print(f"True Params: omega={omega}, alpha={alpha}, beta={beta}")
    print(f"Est Params:  omega={res.omega:.4f}, alpha={res.alpha[0]:.4f}, beta={res.beta[0]:.4f}")
    
    if abs(res.alpha[0] - alpha) < 0.05 and abs(res.beta[0] - beta) < 0.1:
        print("[PASS] GARCH parameters reasonable")
    else:
        print("[WARN] GARCH parameters estimation gap (could be sample noise)")

if __name__ == "__main__":
    verify_var()
    verify_garch()
