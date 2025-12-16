"""
PSM Full Suite Benchmark: Statelix vs Manual sklearn Implementation

Fair comparison of COMPLETE causal inference workflow:
  - Propensity score estimation
  - Matching with caliper
  - Balance diagnostics
  - ATT/ATE estimation with SE
  - AIPW (doubly robust)

This shows the TRUE value of Statelix as a complete tool vs rolling your own.

Run: python benchmarks/psm_fullsuite_benchmark.py
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
import sys

sys.path.insert(0, '.')

# =============================================================================
# Data Generation (same as before)
# =============================================================================

def generate_psm_data(n=100_000, k=5, treatment_effect=0.5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, k)
    ps_linear = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    ps_true = 1 / (1 + np.exp(-ps_linear))
    treatment = (np.random.random(n) < ps_true).astype(float)
    y0 = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
    y1 = y0 + treatment_effect + 0.1 * X[:, 0]
    y = np.where(treatment > 0.5, y1, y0)
    true_att = np.mean(y1[treatment > 0.5] - y0[treatment > 0.5])
    return y, treatment, X, true_att


# =============================================================================
# Statelix Full Suite
# =============================================================================

def benchmark_statelix_fullsuite(y, treatment, X):
    """Complete Statelix workflow: PS + matching + balance + ATT/ATE/ATC + AIPW"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("causal", "statelix_py/models/causal.py")
    causal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(causal)
    
    tracemalloc.start()
    start = time.perf_counter()
    
    # 1. Create estimator
    psm = causal.PropensityScoreMatching(caliper=0.2, n_neighbors=1)
    
    # 2. Fit (includes PS estimation + matching + ATT/ATE/ATC)
    psm.fit(y, treatment, X)
    
    # 3. Balance diagnostics
    balance = psm.balance_summary()
    
    # 4. AIPW for doubly robust estimation
    dr = causal.DoublyRobust(trim=0.01)
    dr.fit(y, treatment, X, psm.propensity_scores)
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'method': 'Statelix Full Suite',
        'time_sec': elapsed,
        'att': psm.att,
        'ate': psm.ate,
        'att_aipw': dr.result_.att,
        'memory_mb': peak / 1024 / 1024,
        'code_lines': 8  # Lines of user code
    }


# =============================================================================
# Manual sklearn Implementation (same features)
# =============================================================================

def benchmark_sklearn_manual(y, treatment, X):
    """Manually implementing same features with sklearn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return {'method': 'sklearn Manual', 'error': 'sklearn not installed'}
    
    n = len(y)
    k = X.shape[1]
    
    tracemalloc.start()
    start = time.perf_counter()
    
    # =========================================================================
    # 1. Propensity Score Estimation (sklearn LogisticRegression)
    # =========================================================================
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X, treatment)
    ps_scores = lr.predict_proba(X)[:, 1]
    
    # =========================================================================
    # 2. Matching with Caliper (manual implementation)
    # =========================================================================
    caliper = 0.2 * np.std(ps_scores)
    
    treated_idx = np.where(treatment > 0.5)[0]
    control_idx = np.where(treatment <= 0.5)[0]
    
    # Build KNN on controls
    control_ps = ps_scores[control_idx].reshape(-1, 1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_ps)
    
    # Query treated
    treated_ps = ps_scores[treated_idx].reshape(-1, 1)
    distances, indices = nn.kneighbors(treated_ps)
    
    # Apply caliper filter
    valid_mask = distances.ravel() <= caliper
    matched_treated = treated_idx[valid_mask]
    matched_control = control_idx[indices.ravel()[valid_mask]]
    
    # =========================================================================
    # 3. Balance Diagnostics (manual implementation)
    # =========================================================================
    def compute_std_diff(X, t_idx, c_idx):
        std_diffs = []
        for j in range(X.shape[1]):
            x_t = X[t_idx, j]
            x_c = X[c_idx, j]
            pooled_sd = np.sqrt((np.var(x_t) + np.var(x_c)) / 2)
            std_diffs.append((np.mean(x_t) - np.mean(x_c)) / pooled_sd if pooled_sd > 1e-10 else 0)
        return np.array(std_diffs)
    
    std_diff_before = compute_std_diff(X, treated_idx, control_idx)
    std_diff_after = compute_std_diff(X, matched_treated, matched_control)
    
    # =========================================================================
    # 4. ATT Estimation with SE (manual implementation)
    # =========================================================================
    diffs = y[matched_treated] - y[matched_control]
    att = np.mean(diffs)
    att_se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    
    # =========================================================================
    # 5. ATE Estimation (manual - need to also match controls to treated)
    # =========================================================================
    # Build KNN on treated
    treated_ps_for_atc = ps_scores[treated_idx].reshape(-1, 1)
    nn_atc = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn_atc.fit(treated_ps_for_atc)
    
    control_ps_for_atc = ps_scores[control_idx].reshape(-1, 1)
    distances_atc, indices_atc = nn_atc.kneighbors(control_ps_for_atc)
    
    valid_atc = distances_atc.ravel() <= caliper
    matched_control_atc = control_idx[valid_atc]
    matched_treated_atc = treated_idx[indices_atc.ravel()[valid_atc]]
    
    diffs_atc = y[matched_treated_atc] - y[matched_control_atc]
    atc = np.mean(diffs_atc)
    
    p_treated = len(treated_idx) / n
    ate = p_treated * att + (1 - p_treated) * atc
    
    # =========================================================================
    # 6. AIPW / Doubly Robust (manual implementation)
    # =========================================================================
    # Outcome models via OLS
    from numpy.linalg import lstsq
    
    X_aug = np.hstack([np.ones((n, 1)), X])
    
    # mu0: E[Y|D=0, X]
    X0 = X_aug[treatment <= 0.5]
    y0 = y[treatment <= 0.5]
    beta0, _, _, _ = lstsq(X0, y0, rcond=None)
    mu0 = X_aug @ beta0
    
    # mu1: E[Y|D=1, X]
    X1 = X_aug[treatment > 0.5]
    y1_data = y[treatment > 0.5]
    beta1, _, _, _ = lstsq(X1, y1_data, rcond=None)
    mu1 = X_aug @ beta1
    
    # AIPW ATT
    trim = 0.01
    valid = (ps_scores >= trim) & (ps_scores <= 1 - trim)
    
    psi_sum = 0
    n_t = 0
    for i in range(n):
        if not valid[i]:
            continue
        e = ps_scores[i]
        if treatment[i] > 0.5:
            psi_sum += y[i] - mu0[i]
            n_t += 1
        else:
            psi_sum -= e / (1 - e) * (y[i] - mu0[i])
    
    att_aipw = psi_sum / n_t if n_t > 0 else np.nan
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'method': 'sklearn Manual',
        'time_sec': elapsed,
        'att': att,
        'ate': ate,
        'att_aipw': att_aipw,
        'memory_mb': peak / 1024 / 1024,
        'code_lines': 85  # Lines of implementation code above
    }


# =============================================================================
# Main Benchmark
# =============================================================================

def run_fullsuite_benchmark(sample_sizes=[10_000, 50_000, 100_000]):
    results = []
    
    for n in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Full Suite Benchmark: n = {n:,}")
        print('='*60)
        
        y, treatment, X, true_att = generate_psm_data(n=n)
        print(f"True ATT: {true_att:.4f}")
        
        # Statelix
        print("\n[1/2] Statelix Full Suite...")
        try:
            res = benchmark_statelix_fullsuite(y, treatment, X)
            res['n'] = n
            res['true_att'] = true_att
            results.append(res)
            print(f"  Time: {res['time_sec']:.3f}s")
            print(f"  ATT: {res['att']:.4f}, ATE: {res['ate']:.4f}, AIPW: {res['att_aipw']:.4f}")
            print(f"  Memory: {res['memory_mb']:.1f}MB")
            print(f"  Code lines: {res['code_lines']}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
        
        # sklearn Manual
        print("\n[2/2] sklearn Manual Implementation...")
        try:
            res = benchmark_sklearn_manual(y, treatment, X)
            if 'error' not in res:
                res['n'] = n
                res['true_att'] = true_att
                results.append(res)
                print(f"  Time: {res['time_sec']:.3f}s")
                print(f"  ATT: {res['att']:.4f}, ATE: {res['ate']:.4f}, AIPW: {res['att_aipw']:.4f}")
                print(f"  Memory: {res['memory_mb']:.1f}MB")
                print(f"  Code lines: {res['code_lines']}")
            else:
                print(f"  Skipped: {res['error']}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)


def print_comparison(df):
    print("\n" + "="*80)
    print("FULL SUITE COMPARISON SUMMARY")
    print("="*80)
    
    for n in df['n'].unique():
        statelix = df[(df['n'] == n) & (df['method'] == 'Statelix Full Suite')]
        sklearn = df[(df['n'] == n) & (df['method'] == 'sklearn Manual')]
        
        if len(statelix) > 0 and len(sklearn) > 0:
            s_time = statelix['time_sec'].values[0]
            sk_time = sklearn['time_sec'].values[0]
            
            s_lines = statelix['code_lines'].values[0]
            sk_lines = sklearn['code_lines'].values[0]
            
            print(f"\nn = {n:,}:")
            print(f"  Statelix:       {s_time:.3f}s ({s_lines} lines of user code)")
            print(f"  sklearn Manual: {sk_time:.3f}s ({sk_lines} lines of implementation)")
            
            if s_time < sk_time:
                print(f"  → Statelix is {sk_time/s_time:.1f}x FASTER")
            else:
                print(f"  → sklearn is {s_time/sk_time:.1f}x faster")
            
            print(f"  → Statelix needs {sk_lines - s_lines} fewer lines ({sk_lines/s_lines:.0f}x less code)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        sizes = [10_000]
    else:
        sizes = [10_000, 50_000, 100_000]
    
    print("="*80)
    print("PSM FULL SUITE BENCHMARK")
    print("Comparing COMPLETE causal inference workflow")
    print("="*80)
    
    df = run_fullsuite_benchmark(sample_sizes=sizes)
    print_comparison(df)
    
    # Save
    df.to_csv('benchmarks/psm_fullsuite_results.csv', index=False)
    print("\nResults saved to benchmarks/psm_fullsuite_results.csv")
    print("Done!")
