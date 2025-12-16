import time
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, List

class BenchmarkTimer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"[{self.name}] Finished in {self.duration:.4f} seconds")

class BenchmarkResult:
    def __init__(self, library: str, algorithm: str, n_samples: int, n_features: int, time_sec: float, metric: float = None):
        self.library = library
        self.algorithm = algorithm
        self.n_samples = n_samples
        self.n_features = n_features
        self.time_sec = time_sec
        self.metric = metric

def print_comparison_table(results: List[BenchmarkResult]):
    """Prints a comparison table of benchmark results."""
    df = pd.DataFrame([vars(r) for r in results])
    
    # Calculate speedup relative to the slowest
    if not df.empty:
        df['speedup'] = df.groupby(['n_samples', 'algorithm'])['time_sec'].transform(lambda x: x.max() / x)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("="*80 + "\n")
    return df

def generate_synthetic_data(n_samples=10000, n_features=20, task='regression'):
    """Generates synthetic data for benchmarking."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    if task == 'regression':
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        return X, y
    elif task == 'classification':
        true_coef = np.random.randn(n_features)
        logits = X @ true_coef
        probs = 1 / (1 + np.exp(-logits))
        y = (np.random.rand(n_samples) < probs).astype(int)
        return X, y
    elif task == 'causal':
        # Treatment assignment based on covariates
        true_coef = np.random.randn(n_features)
        ps = 1 / (1 + np.exp(-(X @ true_coef)))
        D = (np.random.rand(n_samples) < ps).astype(int)
        
        # Outcome depends on X and D
        y = X @ np.random.randn(n_features) + 2.0 * D + np.random.randn(n_samples)
        return X, D, y
    
    return X, None
