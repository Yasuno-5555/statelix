import numpy as np
import pandas as pd
from statelix.panel import FixedEffects, FirstDifference, RandomEffects
import time
import traceback

def verify_fe_vs_lsdv():
    try:
        print("Verifying Panel Fixed Effects (WeightedSolver) vs LSDV...", flush=True)
        np.random.seed(42)
        N = 100
        T = 10
        n_obs = N * T
        K = 3
        
        # Generate Data
        unit_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)
        
        X = np.random.randn(n_obs, K)
        # Unit effects
        alpha = np.random.randn(N)
        # True beta
        beta = np.array([1.5, -0.5, 2.0])
        
        y = np.zeros(n_obs)
        for i in range(n_obs):
            uid = unit_ids[i]
            y[i] = X[i] @ beta + alpha[uid] + np.random.randn() * 0.5

        print("Data Generated.", flush=True)

        # Statelix expects int32 for IDs
        unit_ids_int = unit_ids.astype(np.int32)
        time_ids_int = time_ids.astype(np.int32)

        # Check FD First (Debug)
        print("\nVerifying First Difference...", flush=True)
        fd = FirstDifference()
        try:
            res_fd = fd.fit(y, X, unit_ids_int, time_ids_int)
            print("FD Coefs:", res_fd.coef)
            err_fd = np.linalg.norm(res_fd.coef - beta)
            print(f"FD Error vs True Beta: {err_fd:.6f}")
            if err_fd < 0.2:
                print("[PASS] FD coefficients are reasonable.")
            else:
                print("[FAIL] FD coefficients are not reasonable.")
        except Exception as e:
            print(f"[FAIL] FD crashed: {e}")
            traceback.print_exc() # Added for more detailed crash info

        # 1. Run Statelix FE
        fe = FixedEffects()
        fe.cluster_se = False # Disable clustering for valid Matrix comparison with RE (Hausman)
        print("Initialized FE object.", flush=True)
        
        start = time.time()
        print("Calling fit...", flush=True)
        res_fe = fe.fit(y, X, unit_ids_int, time_ids_int)
        print("Fit returned.", flush=True)
        print(f"FE Time: {time.time() - start:.4f}s")
        print("FE Coefs:", res_fe.coef)
        
        # 2. Run OLS with Dummies (LSDV) using NumPy
        dummies = pd.get_dummies(unit_ids).values
        X_lsdv = np.hstack([X, dummies])
        
        # lstsq
        beta_lsdv_full, _, _, _ = np.linalg.lstsq(X_lsdv, y, rcond=None)
        beta_lsdv = beta_lsdv_full[:K]
        
        print("LSDV Coefs:", beta_lsdv)
        
        # Verify match
        err = np.linalg.norm(res_fe.coef - beta_lsdv)
        print(f"Error vs LSDV Beta: {err:.10f}")
        
        if err < 1e-8:
            print("[PASS] FE coefficients match LSDV perfectly.")
        else:
            print("[FAIL] FE coefficients mismatch.")
            
        # Check RE
        print("\nVerifying Random Effects...")
        re = RandomEffects()
        try:
            res_re = re.fit(y, X, unit_ids_int, time_ids_int)
            print("RE Coefs:", res_re.coef)
            err_re = np.linalg.norm(res_re.coef - beta)
            print(f"RE Error vs True Beta: {err_re:.6f}")
            if err_re < 0.2:
                print("[PASS] RE coefficients are reasonable.")
            else:
                 print("[FAIL] RE coefficients diverged.")
        except Exception as e:
            print(f"[FAIL] RE crashed: {e}")
            traceback.print_exc()

        # Check Hausman Test
        print("\nVerifying Hausman Test...")
        from statelix.panel import HausmanTest
        try:
            # We already have res_fe and res_re
            ht = HausmanTest.compare(res_fe, res_re)
            print(f"Hausman Chi2: {ht.chi2_stat:.4f} p-val: {ht.p_value:.4f}")
            print(f"Result: {ht.recommendation}")
            
            # Since X and alpha are independent, we expect p-value > 0.05 (Fail to reject H0)
            if ht.p_value > 0.05:
                print("[PASS] Hausman correctly recommends RE (independent alpha/X).")
            else:
                print("[WARN] Hausman rejected H0 (p < 0.05). False positive or small sample noise?")
                
        except Exception as e:
            print(f"[FAIL] Hausman crashed: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"CRASH: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    verify_fe_vs_lsdv()
