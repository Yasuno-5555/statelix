
import sys
import os
import numpy as np

# Ensure we can import statelix_py
sys.path.append(os.path.abspath("statelix_py"))

try:
    from statelix_py import topology_bayes
except ImportError:
    # Try importing directly if inside statelix_py
    import topology_bayes

def test_posterior_persistence():
    print("\n--- Testing Bayesian x Topology Fusion ---")
    
    # 1. Robust Case: Samples have similar "shape" (e.g., peak at index 2)
    # n_samples=10, n_features=5
    robust_samples = []
    for _ in range(10):
        # Base signal: [0, 1, 5, 1, 0]
        # Noise: random small variations
        noise = np.random.normal(0, 0.1, 5)
        signal = np.array([0, 1, 5, 1, 0]) + noise
        robust_samples.append(signal)
        
    robust_samples = np.array(robust_samples)
    
    pp_robust = topology_bayes.PosteriorPersistence(robust_samples)
    res_robust = pp_robust.summary()
    
    assert res_robust["robust"], "Should be robust"
    
    # 2. Unstable Case: Samples fluctuate wildly
    unstable_samples = []
    for _ in range(10):
        # Random big spikes
        signal = np.random.uniform(0, 10, 5)
        unstable_samples.append(signal)
        
    unstable_samples = np.array(unstable_samples)
    
    pp_unstable = topology_bayes.PosteriorPersistence(unstable_samples)
    res_unstable = pp_unstable.summary()
    
    assert not res_unstable["robust"], "Should NOT be robust"
    
    print("SUCCESS: Posterior Persistence verified.")

if __name__ == "__main__":
    test_posterior_persistence()
