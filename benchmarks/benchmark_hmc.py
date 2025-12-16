import time
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from benchmarks.utils import BenchmarkTimer, BenchmarkResult, print_comparison_table

try:
    import benchmarks.statelix_hmc as statelix_hmc
    STATELIX_AVAILABLE = True
except ImportError:
    try:
        import statelix_hmc
        STATELIX_AVAILABLE = True
    except ImportError:
        print("WARNING: statelix_hmc module not found.")
        STATELIX_AVAILABLE = False

# HMC needs a model. Since we use EfficientObjective trampoline:
if STATELIX_AVAILABLE:
    class GaussianObjective(statelix_hmc.EfficientObjective):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.mu = np.zeros(dim)
            self.cov_inv = np.eye(dim)
            
        def value_and_gradient(self, x):
            # log_prob = -0.5 * (x-mu)' cov_inv (x-mu)
            # objective = -log_prob = 0.5 * ...
            # grad objective = cov_inv * (x-mu)
            diff = x - self.mu
            # Important: return double, VectorXd
            val = 0.5 * np.dot(diff, np.dot(self.cov_inv, diff))
            grad = np.dot(self.cov_inv, diff)
            return val, grad
            
        def dimension(self):
            return self.dim

def run_hmc_benchmark(dim, n_samples):
    if not STATELIX_AVAILABLE:
        return 0, 0
    
    config = statelix_hmc.HMCConfig()
    config.n_samples = n_samples
    config.warmup = int(n_samples / 2)
    config.step_size = 0.1
    config.n_leapfrog = 10
    config.seed = 42
    
    hmc = statelix_hmc.HamiltonianMonteCarlo(config)
    
    model = GaussianObjective(dim)
    theta0 = np.random.randn(dim)
    
    with BenchmarkTimer("Statelix HMC") as t:
        result = hmc.sample(model, theta0)
        
    # Validation: Check mean is close to 0
    samples = np.array(result.samples)
    mean_est = np.mean(samples, axis=0)
    print(f"  Estimated Mean (First 3 dims): {mean_est[:3]}")
    print(f"  Acceptance Rate: {result.acceptance_rate:.2f}")
    
    return t.duration, result.acceptance_rate

def main():
    results = []
    
    scenarios = [
        {"dim": 2, "n_samples": 2000},
        {"dim": 10, "n_samples": 2000},
        {"dim": 50, "n_samples": 2000},
    ]
    
    print(f"Running HMC Benchmark...")
    
    for sc in scenarios:
        dim = sc['dim']
        n = sc['n_samples']
        print(f"\n--- Scenario: Dim={dim}, Samples={n} ---")
        
        if STATELIX_AVAILABLE:
            t, acc = run_hmc_benchmark(dim, n)
            results.append(BenchmarkResult("Statelix HMC", f"Gaussian (D={dim})", n, dim, t, acc))
        else:
            print("Skipped (Module not found)")
            
    # No comparison vs other lib for now, just validation/speed check
    print_comparison_table(results)

if __name__ == "__main__":
    main()
