
import numpy as np
import pandas as pd
import statelix.causal
import sys
import traceback

def verify_iv():
    print("Verifying IV/2SLS...")
    np.random.seed(42)
    n = 10000
    
    # Instruments
    z = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n) # Second instrument for overidentification
    
    # Error terms (correlated)
    u = np.random.normal(0, 1, n)
    rho = 0.5
    e = rho * u + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
    
    # Endogenous variable X
    # x = 0.5*z + 0.3*z2 + u
    x = 0.5 * z + 0.3 * z2 + u
    
    # Outcome Y
    # y = 1.0 * x + e
    true_beta = 1.0
    y = true_beta * x + e
    
    # OLS (Biased)
    # X_ols = [1, x]
    X_ols = np.column_stack([np.ones(n), x])
    beta_ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ y
    print(f"True Beta: {true_beta}")
    print(f"OLS Beta: {beta_ols[1]:.4f} (Should be biased biased, likely > 1.0)")
    
    # IV Estimation
    # fit(Y, X_endog, Z) 
    # Intercept is handled by class option fit_intercept=True (default)
    
    iv = statelix.causal.TwoStageLeastSquares()
    
    # Reshape for C++ binding (needs matrix/vector)
    Y_vec = y
    X_endog = x.reshape(-1, 1)
    Z = np.column_stack([z, z2])
    
    try:
        res = iv.fit(Y_vec, X_endog, Z)
        
        print(f"IV Beta: {res.coef[0]:.4f} (Intercept), {res.coef[1]:.4f} (Slope)")
        print(f"Standard Errors: {res.std_errors}")
        print(f"First Stage F: {res.first_stage_f:.2f} (p={res.first_stage_f_pvalue:.4f})")
        print(f"Sargan J: {res.sargan_stat:.2f} (p={res.sargan_pvalue:.4f})")
        
        # Checks
        slope = res.coef[1]
        if abs(slope - true_beta) < 0.05:
            print("[PASS] IV Coeff matches True Beta")
        else:
            print(f"[FAIL] IV Coeff {slope} too far from {true_beta}")
        
        # Sargan should pass (p > 0.05) as instruments are valid
        if res.sargan_pvalue > 0.05:
            print("[PASS] Sargan Test (Valid Instruments not rejected)")
        else:
            print(f"[FAIL] Sargan Test rejected (p={res.sargan_pvalue})")
            
        # Weak IV
        if res.first_stage_f > 10:
             print("[PASS] First Stage F > 10")
        else:
             print("[FAIL] First Stage F too low")
             
    except Exception as e:
        print(f"[FAIL] Exception during IV fit: {e}")
        traceback.print_exc()

def verify_gmm():
    print("\nVerifying GMM...")
    np.random.seed(42)
    n = 2000
    
    # Instruments
    z = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    u = np.random.normal(0, 1, n)
    rho = 0.5
    # Homoskedastic errors
    e = rho*u + np.sqrt(1-rho**2)*np.random.normal(0, 1, n)
    x = 0.5*z + 0.3*z2 + u
    true_beta = 1.0
    y = true_beta * x + e
    
    Y_vec = y
    X_endog = x.reshape(-1, 1)
    # Empty X_exog
    X_exog = np.zeros((n, 0))
    Z = np.column_stack([z, z2])
    
    gmm = statelix.causal.LinearGMM()
    
    try:
        # 1. 2SLS Mode
        print("1. Testing GMM (mode='2sls')...")
        res_2sls = gmm.fit(Y_vec, X_endog, X_exog, Z, "2sls")
        print(f"   GMM(2sls) Coef: {res_2sls.coef[1]:.5f}")
        
        # Compare with actual 2SLS
        iv = statelix.causal.TwoStageLeastSquares()
        res_iv = iv.fit(Y_vec, X_endog, X_exog, Z)
        print(f"   IV(2sls)  Coef: {res_iv.coef[1]:.5f}")
        
        diff = abs(res_2sls.coef[1] - res_iv.coef[1])
        if diff < 1e-9:
             print("[PASS] GMM(2sls) matches IV exactly")
        else:
             print(f"[FAIL] GMM(2sls) mismatch! Diff={diff}")
             
        # 2. Optimal Mode
        print("2. Testing GMM (mode='optimal')...")
        res_opt = gmm.fit(Y_vec, X_endog, X_exog, Z, "optimal")
        print(f"   GMM(opt)  Coef: {res_opt.coef[1]:.5f}")
        print(f"   J-Stat: {res_opt.j_stat:.4f} (p={res_opt.j_pvalue:.4f})")
        
        # Should be close to IV (since homoskedastic)
        diff_opt = abs(res_opt.coef[1] - res_iv.coef[1])
        if diff_opt < 5e-3:
             print(f"[PASS] GMM(opt) aligns with IV (Diff={diff_opt:.5f})")
        else:
             print(f"[WARN] GMM(opt) deviates from IV (Diff={diff_opt:.5f}) - might be sampling noise or difference in efficiency")

    except Exception as e:
        print(f"[FAIL] Exception during GMM fit: {e}")
        traceback.print_exc()

def verify_did():
    print("\nVerifying DiD/TWFE...")
    np.random.seed(42)
    
    # 1. Basic 2x2 DID
    print("1. Testing Basic 2x2 DID...")
    n = 1000
    treated_group = np.random.choice([0, 1], n)
    post_period = np.random.choice([0, 1], n)
    
    # Truth
    alpha = 2.0
    beta_treat = 1.0 # Group difference
    beta_post = 0.5  # Time trend
    tau = 3.0        # Treatment effect
    
    Y = alpha + beta_treat*treated_group + beta_post*post_period + tau*(treated_group * post_period) + np.random.normal(0, 1, n)
    
    did = statelix.causal.DifferenceInDifferences()
    # Cast to integer (C++ expects VectorXi)
    res_did = did.fit(Y, treated_group.astype(np.int32), post_period.astype(np.int32))
    
    print(f"   True ATT: {tau}")
    print(f"   Est ATT:  {res_did.att:.4f} (SE={res_did.att_std_error:.4f})")
    
    if abs(res_did.att - tau) < 0.2:
        print("[PASS] Basic DID Estimate is correct")
    else:
        print(f"[FAIL] Basic DID Estimate incorrect (Diff={res_did.att - tau})")
        
    # 2. TWFE
    print("2. Testing TWFE (Panel)...")
    n_units = 200
    n_periods = 20
    total_obs = n_units * n_periods
    
    unit_id = np.repeat(np.arange(n_units), n_periods)
    time_id = np.tile(np.arange(n_periods), n_units)
    
    unit_fe = np.random.normal(0, 2, n_units)
    time_fe = np.random.normal(0, 1, n_periods)
    
    # Treatment: Staggered
    # Half units treated starting at t=5
    treatment_start = np.random.choice([5, 100], n_units) # 100 = never treated
    D = np.zeros(total_obs, dtype=np.int32)
    for i in range(total_obs):
        u = unit_id[i]
        t = time_id[i]
        if t >= treatment_start[u]:
            D[i] = 1
            
    # Effect
    tau_twfe = 2.5
    Y_panel = np.zeros(total_obs)
    for i in range(total_obs):
        u = unit_id[i]
        t = time_id[i]
        Y_panel[i] = unit_fe[u] + time_fe[t] + tau_twfe * D[i] + np.random.normal(0, 1)
        
    twfe = statelix.causal.TwoWayFixedEffects()
    res_twfe = twfe.fit(Y_panel, D, unit_id.astype(np.int32), time_id.astype(np.int32))
    
    print(f"   True Delta: {tau_twfe}")
    print(f"   Est Delta:  {res_twfe.delta:.4f} (SE={res_twfe.delta_std_error:.4f})")
    
    if abs(res_twfe.delta - tau_twfe) < 0.2:
        print("[PASS] TWFE Estimate is correct")
    else:
        print(f"[FAIL] TWFE Estimate incorrect (Diff={res_twfe.delta - tau_twfe})")

def verify_rdd():
    print("\nVerifying RDD...")
    np.random.seed(42)
    
    # 1. Sharp RDD
    print("1. Sharp RDD...")
    n = 2000
    X = np.random.uniform(-10, 10, n)
    cutoff = 0.0
    treated = (X >= cutoff).astype(int)
    
    # Outcome: Y = 1 + 2*X + 3*Treated + e
    tau_true = 3.0
    Y = 1.0 + 2.0 * X + tau_true * treated + np.random.normal(0, 2, n)
    
    rdd = statelix.causal.SharpRDD()
    rdd.bandwidth = 5.0 # Fixed bandwidth for stability
    res_sharp = rdd.fit(Y, X, cutoff)
    
    print(f"   True Tau: {tau_true}")
    print(f"   Est Tau:  {res_sharp.tau:.4f} (SE={res_sharp.tau_se:.4f})")
    print(f"   Bandwidth: {res_sharp.bandwidth:.4f}, N_eff: {res_sharp.n_left + res_sharp.n_right}")
    
    if abs(res_sharp.tau - tau_true) < 0.6:
        print("[PASS] Sharp RDD Estimate is correct")
    else:
        print(f"[FAIL] Sharp RDD Estimate incorrect (Diff={res_sharp.tau - tau_true})")

    # 2. Fuzzy RDD
    print("2. Fuzzy RDD...")
    # First stage: Prob(D=1) = 0.2 + 0.5*Above + 0.01*X
    prob_d = 0.2 + 0.5 * (X >= cutoff) + 0.01 * X
    prob_d = np.clip(prob_d, 0, 1)
    D_fuzzy = np.random.binomial(1, prob_d)
    
    # Outcome: Y = 1 + X + 4*D + e
    tau_fuzzy = 4.0
    Y_fuzzy = 1.0 + X + tau_fuzzy * D_fuzzy + np.random.normal(0, 2, n)
    
    frdd = statelix.causal.FuzzyRDD()
    frdd.bandwidth = 5.0
    res_fuzzy = frdd.fit(Y_fuzzy, D_fuzzy.astype(np.float64), X, cutoff)
    
    print(f"   True Tau: {tau_fuzzy}")
    print(f"   Est Tau:  {res_fuzzy.tau:.4f} (SE={res_fuzzy.tau_se:.4f})")
    print(f"   First Stage Jump: {res_fuzzy.first_stage_jump:.4f}")
    
    if abs(res_fuzzy.tau - tau_fuzzy) < 0.5: # Higher variance for fuzzy
        print("[PASS] Fuzzy RDD Estimate is correct")
    else:
        print(f"[FAIL] Fuzzy RDD Estimate incorrect (Diff={res_fuzzy.tau - tau_fuzzy})")

if __name__ == "__main__":
    try:
        verify_iv()
        verify_gmm()
        verify_did()
        verify_rdd()
        print("\nVerification script finished.")
    except Exception as e:
        print(f"Top-level error: {e}")
        traceback.print_exc()
