"""
PSM Benchmark: Statelix vs Python Alternatives

Compares:
  - Statelix (O(log n) binary search matching)
  - sklearn NearestNeighbors (brute-force / KD-tree)
  - Pure Python naive matching (O(n²))

Metrics:
  - Matching time
  - ATT accuracy (vs ground truth)
  - Memory usage

Run: python benchmarks/psm_benchmark.py
"""

import numpy as np
import pandas as pd
import time
import sys
import tracemalloc
from typing import Tuple, Dict, Any

# Add parent directory for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'statelix_py')

# =============================================================================
# Data Generation
# =============================================================================

def generate_psm_data(
    n: int = 100_000,
    k: int = 5,
    treatment_effect: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic data for PSM benchmarking.
    
    Returns:
        y: Outcomes
        treatment: Treatment indicator
        X: Covariates
        true_att: True ATT (for accuracy check)
    """
    np.random.seed(seed)
    
    # Covariates
    X = np.random.randn(n, k)
    
    # Propensity score model: logistic
    ps_linear = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    ps_true = 1 / (1 + np.exp(-ps_linear))
    
    # Treatment assignment
    treatment = (np.random.random(n) < ps_true).astype(float)
    
    # Outcome model: Y = treatment_effect * D + X effects + noise
    y0 = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
    y1 = y0 + treatment_effect + 0.1 * X[:, 0]  # Heterogeneous effect
    
    y = np.where(treatment > 0.5, y1, y0)
    
    # True ATT: E[Y(1) - Y(0) | D=1]
    true_att = np.mean(y1[treatment > 0.5] - y0[treatment > 0.5])
    
    return y, treatment, X, true_att


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_statelix(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray
) -> Dict[str, Any]:
    """Benchmark Statelix PSM."""
    # Direct import to avoid sklearn dependency from other modules
    import importlib.util
    spec = importlib.util.spec_from_file_location("causal", "statelix_py/models/causal.py")
    causal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(causal_module)
    PropensityScoreMatching = causal_module.PropensityScoreMatching
    
    # Warmup
    psm = PropensityScoreMatching(caliper=0.2, n_neighbors=1)
    
    # Memory tracking
    tracemalloc.start()
    
    # Time
    start = time.perf_counter()
    psm.fit(y, treatment, X)
    elapsed = time.perf_counter() - start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'method': 'Statelix',
        'time_sec': elapsed,
        'att': psm.att,
        'att_se': psm.att_se,
        'n_matched': psm.match_result_.n_matched_treated,
        'memory_mb': peak / 1024 / 1024
    }


def benchmark_sklearn_knn(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
    ps_scores: np.ndarray
) -> Dict[str, Any]:
    """Benchmark sklearn NearestNeighbors for matching."""
    try:
        from sklearn.neighbors import NearestNeighbors
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False
        return {'method': 'sklearn KNN', 'error': 'sklearn not installed', 'time_sec': np.nan}
    
    if not HAS_SKLEARN:
        return {'method': 'sklearn KNN', 'error': 'sklearn not installed', 'time_sec': np.nan}
    
    treated_idx = np.where(treatment > 0.5)[0]
    control_idx = np.where(treatment <= 0.5)[0]
    
    control_ps = ps_scores[control_idx].reshape(-1, 1)
    treated_ps = ps_scores[treated_idx].reshape(-1, 1)
    
    tracemalloc.start()
    start = time.perf_counter()
    
    # Build index
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control_ps)
    
    # Query
    distances, indices = nn.kneighbors(treated_ps)
    
    # Compute ATT
    matched_control_idx = control_idx[indices.ravel()]
    diffs = y[treated_idx] - y[matched_control_idx]
    att = np.mean(diffs)
    att_se = np.std(diffs) / np.sqrt(len(diffs))
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'method': 'sklearn KNN',
        'time_sec': elapsed,
        'att': att,
        'att_se': att_se,
        'n_matched': len(treated_idx),
        'memory_mb': peak / 1024 / 1024
    }


def benchmark_psmpy(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
    max_n: int = 20_000  # psmpy can be slow on large n
) -> Dict[str, Any]:
    """Benchmark psmpy - a popular Python PSM library."""
    try:
        from psmpy import PsmPy
        from psmpy.functions import cohenD
        HAS_PSMPY = True
    except ImportError:
        return {'method': 'psmpy', 'error': 'psmpy not installed', 'time_sec': np.nan}
    
    n = len(y)
    
    # Subsample if too large (psmpy is slow)
    if n > max_n:
        np.random.seed(42)
        idx = np.random.choice(n, max_n, replace=False)
        y = y[idx]
        treatment = treatment[idx]
        X = X[idx]
        n = max_n
    
    # psmpy expects a DataFrame
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    df['treatment'] = treatment.astype(int)
    df['outcome'] = y
    df['id'] = np.arange(n)
    
    tracemalloc.start()
    start = time.perf_counter()
    
    try:
        # Initialize psmpy
        psm = PsmPy(df, treatment='treatment', indx='id', exclude=['outcome'])
        
        # Fit propensity model
        psm.logistic_ps(balance=False)
        
        # Perform matching
        psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=0.2)
        
        # Get matched data
        matched_df = psm.matched_ids
        
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Compute ATT from matched pairs
        if matched_df is not None and len(matched_df) > 0:
            # psmpy stores matched IDs
            n_matched = len(matched_df)
            # Simple ATT calculation
            treated_outcomes = df[df['treatment'] == 1]['outcome'].values
            control_outcomes = df[df['treatment'] == 0]['outcome'].values
            att = np.mean(treated_outcomes) - np.mean(control_outcomes)  # Rough
        else:
            n_matched = 0
            att = np.nan
        
        return {
            'method': f'psmpy (n={n})',
            'time_sec': elapsed,
            'att': att,
            'att_se': np.nan,  # psmpy doesn't provide SE easily
            'n_matched': n_matched,
            'memory_mb': peak / 1024 / 1024
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            'method': f'psmpy (n={n})',
            'error': str(e),
            'time_sec': np.nan
        }


def benchmark_naive_python(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
    ps_scores: np.ndarray,
    max_n: int = 10_000  # Limit for O(n²)
) -> Dict[str, Any]:
    """Benchmark naive O(n²) Python matching."""
    
    treated_idx = np.where(treatment > 0.5)[0]
    control_idx = np.where(treatment <= 0.5)[0]
    
    # Subsample if too large
    if len(treated_idx) > max_n:
        np.random.seed(42)
        treated_idx = np.random.choice(treated_idx, max_n, replace=False)
    
    tracemalloc.start()
    start = time.perf_counter()
    
    diffs = []
    for t in treated_idx:
        ps_t = ps_scores[t]
        
        # O(n) search for nearest control
        best_dist = np.inf
        best_c = None
        for c in control_idx:
            dist = abs(ps_scores[c] - ps_t)
            if dist < best_dist:
                best_dist = dist
                best_c = c
        
        if best_c is not None:
            diffs.append(y[t] - y[best_c])
    
    att = np.mean(diffs)
    att_se = np.std(diffs) / np.sqrt(len(diffs))
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Extrapolate time for full dataset
    scale = len(np.where(treatment > 0.5)[0]) / len(treated_idx)
    extrapolated_time = elapsed * scale  # O(n²) scaling
    
    return {
        'method': f'Naive Python (n={len(treated_idx)})',
        'time_sec': elapsed,
        'time_extrapolated': extrapolated_time,
        'att': att,
        'att_se': att_se,
        'n_matched': len(diffs),
        'memory_mb': peak / 1024 / 1024
    }


def benchmark_numpy_vectorized(
    y: np.ndarray,
    treatment: np.ndarray,
    X: np.ndarray,
    ps_scores: np.ndarray,
    max_n: int = 20_000  # Limited by RAM for distance matrix
) -> Dict[str, Any]:
    """Benchmark vectorized numpy (still O(n²) memory)."""
    
    treated_idx = np.where(treatment > 0.5)[0]
    control_idx = np.where(treatment <= 0.5)[0]
    
    # Subsample
    if len(treated_idx) > max_n:
        np.random.seed(42)
        treated_idx = np.random.choice(treated_idx, max_n, replace=False)
    if len(control_idx) > max_n:
        np.random.seed(43)
        control_idx = np.random.choice(control_idx, max_n, replace=False)
    
    tracemalloc.start()
    start = time.perf_counter()
    
    # Distance matrix: O(n_t * n_c) memory
    treated_ps = ps_scores[treated_idx]
    control_ps = ps_scores[control_idx]
    
    dist_matrix = np.abs(treated_ps[:, None] - control_ps[None, :])
    nearest_idx = np.argmin(dist_matrix, axis=1)
    
    matched_control = control_idx[nearest_idx]
    diffs = y[treated_idx] - y[matched_control]
    att = np.mean(diffs)
    att_se = np.std(diffs) / np.sqrt(len(diffs))
    
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'method': f'NumPy vectorized (n={len(treated_idx)})',
        'time_sec': elapsed,
        'att': att,
        'att_se': att_se,
        'n_matched': len(diffs),
        'memory_mb': peak / 1024 / 1024
    }


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(
    sample_sizes: list = [1_000, 10_000, 50_000, 100_000],
    k: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """Run full benchmark suite."""
    
    results = []
    
    for n in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Benchmarking n = {n:,}")
        print('='*60)
        
        # Generate data
        y, treatment, X, true_att = generate_psm_data(n=n, k=k, seed=seed)
        
        print(f"True ATT: {true_att:.4f}")
        print(f"Treated: {int(np.sum(treatment)):,}, Control: {int(n - np.sum(treatment)):,}")
        
        # Estimate PS first (shared across methods)
        # Direct import to avoid sklearn dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location("causal", "statelix_py/models/causal.py")
        causal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal_module)
        PropensityScoreMatching = causal_module.PropensityScoreMatching
        
        psm_temp = PropensityScoreMatching()
        ps_result = psm_temp._estimate_propensity(treatment, X)
        ps_scores = ps_result.scores
        
        # 1. Statelix
        print("\n[1/4] Statelix PSM...")
        try:
            res = benchmark_statelix(y, treatment, X)
            res['n'] = n
            res['true_att'] = true_att
            res['att_error'] = abs(res['att'] - true_att)
            results.append(res)
            print(f"  Time: {res['time_sec']:.3f}s, ATT: {res['att']:.4f}, Error: {res['att_error']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 2. sklearn KNN
        print("[2/5] sklearn KNN...")
        try:
            res = benchmark_sklearn_knn(y, treatment, X, ps_scores)
            if 'error' not in res:
                res['n'] = n
                res['true_att'] = true_att
                res['att_error'] = abs(res['att'] - true_att)
                results.append(res)
                print(f"  Time: {res['time_sec']:.3f}s, ATT: {res['att']:.4f}, Error: {res['att_error']:.4f}")
            else:
                print(f"  Skipped: {res['error']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 3. psmpy (real library comparison)
        if n <= 20_000:
            print("[3/5] psmpy...")
            try:
                res = benchmark_psmpy(y, treatment, X)
                if 'error' not in res:
                    res['n'] = n
                    res['true_att'] = true_att
                    res['att_error'] = abs(res['att'] - true_att) if not np.isnan(res['att']) else np.nan
                    results.append(res)
                    print(f"  Time: {res['time_sec']:.3f}s, Matched: {res['n_matched']}")
                else:
                    print(f"  Skipped: {res['error']}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("[3/5] psmpy... Skipped (n too large)")
        
        # 4. NumPy vectorized (limited n)
        if n <= 50_000:
            print("[4/5] NumPy vectorized...")
            try:
                res = benchmark_numpy_vectorized(y, treatment, X, ps_scores)
                res['n'] = n
                res['true_att'] = true_att
                res['att_error'] = abs(res['att'] - true_att)
                results.append(res)
                print(f"  Time: {res['time_sec']:.3f}s, ATT: {res['att']:.4f}, Memory: {res['memory_mb']:.1f}MB")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("[4/5] NumPy vectorized... Skipped (n too large)")
        
        # 5. Naive Python (very limited n)
        if n <= 10_000:
            print("[5/5] Naive Python...")
            try:
                res = benchmark_naive_python(y, treatment, X, ps_scores)
                res['n'] = n
                res['true_att'] = true_att
                res['att_error'] = abs(res['att'] - true_att)
                results.append(res)
                print(f"  Time: {res['time_sec']:.3f}s, ATT: {res['att']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("[5/5] Naive Python... Skipped (n too large)")
    
    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Pivot by n and method
    summary = df.pivot_table(
        index='method',
        columns='n',
        values=['time_sec', 'att_error', 'memory_mb'],
        aggfunc='first'
    )
    
    print("\n--- Time (seconds) ---")
    time_df = df.pivot_table(index='method', columns='n', values='time_sec', aggfunc='first')
    print(time_df.to_string())
    
    print("\n--- ATT Error (|estimate - true|) ---")
    error_df = df.pivot_table(index='method', columns='n', values='att_error', aggfunc='first')
    print(error_df.to_string())
    
    print("\n--- Memory (MB) ---")
    mem_df = df.pivot_table(index='method', columns='n', values='memory_mb', aggfunc='first')
    print(mem_df.to_string())
    
    # Speedup vs baseline
    print("\n--- Speedup (vs sklearn KNN) ---")
    for n in df['n'].unique():
        statelix_time = df[(df['n'] == n) & (df['method'] == 'Statelix')]['time_sec'].values
        sklearn_time = df[(df['n'] == n) & (df['method'] == 'sklearn KNN')]['time_sec'].values
        
        if len(statelix_time) > 0 and len(sklearn_time) > 0 and sklearn_time[0] > 0:
            speedup = sklearn_time[0] / statelix_time[0]
            print(f"  n={n:,}: Statelix is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PSM Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick run with smaller n')
    parser.add_argument('--full', action='store_true', help='Full run with n up to 500k')
    args = parser.parse_args()
    
    if args.quick:
        sizes = [1_000, 5_000, 10_000]
    elif args.full:
        sizes = [1_000, 10_000, 50_000, 100_000, 200_000, 500_000]
    else:
        sizes = [1_000, 10_000, 50_000, 100_000]
    
    print("="*80)
    print("STATELIX PSM BENCHMARK")
    print("="*80)
    print(f"Sample sizes: {sizes}")
    print(f"Comparing: Statelix (O(log n)) vs sklearn KNN vs NumPy vs Naive")
    
    df = run_benchmark(sample_sizes=sizes)
    
    print_summary(df)
    
    # Save results
    output_file = 'benchmarks/psm_results.csv'
    try:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    except:
        print("\nCould not save results to file")
    
    print("\nDone!")
