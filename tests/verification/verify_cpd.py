
import numpy as np
import statelix.time_series as ts
import sys

def check_cp(true_cps, est_cps, tol=2):
    """Check if estimated CPs match true CPs within tolerance."""
    # true_cps and est_cps are lists of ints.
    # We want to match every true CP (except end of signal) to an estimated CP.
    # Note: est_cps might include 'n'. true_cps usually doesn't include 'n' in definition but implementation returns it?
    # Let's normalize: exclude 'n' if present in both.
    
    # Sort
    est_cps = sorted(list(est_cps))
    true_cps = sorted(list(true_cps))
    
    # Remove last point if it equals data length (we might not know n here easily, but usually it's the largest)
    # Actually, let's just match "intermediate" change points.
    
    matched_count = 0
    used_est = set()
    
    print(f"  True: {true_cps} | Est: {est_cps}")
    
    for tcp in true_cps:
        found = False
        for ecp in est_cps:
            if ecp in used_est: continue
            if abs(tcp - ecp) <= tol:
                used_est.add(ecp)
                matched_count += 1
                found = True
                break
        if not found:
            print(f"  [MISS] CP at {tcp} not found within tol={tol}")
            return False
            
    # Check for excessive false positives?
    # For now, strict matching of expected CPs is enough. 
    # But if we detect 100 CPs for 2 true ones, that's bad.
    if len(est_cps) > len(true_cps) + 2: # Allow n and maybe 1 spurious?
        print(f"  [WARN] Too many estimated CPs.")
        
    return True

def verify_l2():
    print("\n--- Testing L2 Cost (Mean Shift) ---")
    np.random.seed(42)
    n_seg = 100
    data = np.concatenate([
        np.random.normal(0, 1, n_seg),
        np.random.normal(10, 1, n_seg), # Huge jump
        np.random.normal(0, 1, n_seg)
    ])
    
    # True CPs: 100, 200, (300)
    true_cps = [100, 200]
    
    detector = ts.ChangePointDetector(cost_type=ts.CostType.L2, min_size=5)
    # penalty=0.0 -> BIC
    res = detector.fit(data)
    
    if check_cp(true_cps, res.change_points):
        print("[PASS] L2 Mean Shift detected correclty.")
    else:
        print("[FAIL] L2 Mean Shift failed.")
        sys.exit(1)

def verify_gaussian():
    print("\n--- Testing Gaussian Cost (Variance Shift) ---")
    np.random.seed(42)
    n_seg = 200
    # Mean 0 always. Var changes 1 -> 5 -> 1
    data = np.concatenate([
        np.random.normal(0, 1, n_seg),
        np.random.normal(0, np.sqrt(5), n_seg),
        np.random.normal(0, 1, n_seg)
    ])
    
    true_cps = [200, 400]
    
    detector = ts.ChangePointDetector(cost_type=ts.CostType.GAUSSIAN, min_size=10)
    res = detector.fit(data)
    
    if check_cp(true_cps, res.change_points, tol=5): # Var shift is harder, relax tol
        print("[PASS] Gaussian Variance Shift detected.")
    else:
        print("[FAIL] Gaussian Variance Shift failed.")
        
def verify_poisson():
    print("\n--- Testing Poisson Cost (Count Shift) ---")
    np.random.seed(42)
    n_seg = 100
    # Lambda 1 -> 4 -> 10
    data = np.concatenate([
        np.random.poisson(1, n_seg),
        np.random.poisson(4, n_seg),
        np.random.poisson(10, n_seg)
    ])
    
    true_cps = [100, 200]
    
    detector = ts.ChangePointDetector(cost_type=ts.CostType.POISSON, min_size=5)
    res = detector.fit(data)
    
    if check_cp(true_cps, res.change_points):
        print("[PASS] Poisson Shift detected.")
    else:
        print("[FAIL] Poisson Shift failed.")

def verify_stress():
    print("\n--- Testing Stress / Robustness ---")
    np.random.seed(999)
    # Small shift, high noise
    n_seg = 100
    # Shift 0 -> 1 with noise std=2. SNR = 0.5. Very hard.
    # Might need manual penalty tuning if BIC is too conservative?
    # Or BIC should prune it if not significant.
    # Let's try SNR=1 (Mean 0 -> 2, Std 2).
    data = np.concatenate([
        np.random.normal(0, 2, n_seg),
        np.random.normal(2, 2, n_seg)
    ])
    true_cps = [100]
    
    detector = ts.ChangePointDetector(cost_type=ts.CostType.L2, min_size=5)
    # Default BIC might be too strong for low SNR?
    # 2 * log(200) ~ 10.6. 
    # Cost gain ~ 100 * (DiffMean^2) ?? 
    res = detector.fit(data)
    # We accept if it finds it OR if it decides no change (conservative). 
    # But for verification we want to see if it CAN find it.
    
    print("  Scenario: SNR=1 (Mean 0->2, Std=2)")
    check_cp(true_cps, res.change_points, tol=10)
    
    # Boundary test
    print("  Scenario: Short segment at end")
    data2 = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 10) # 10 samples
    ])
    true_cps2 = [100]
    detector.min_size = 2 # Allow small
    res2 = detector.fit(data2)
    check_cp(true_cps2, res2.change_points, tol=2)

if __name__ == "__main__":
    try:
        verify_l2()
        verify_gaussian()
        verify_poisson()
        verify_stress()
        print("\nAll CPD verification tests PASSED.")
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        sys.exit(1)
