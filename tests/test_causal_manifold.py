
import numpy as np
import pytest
from statelix_py.causal.rdd import RDD
from statelix_py.diagnostics.causal_manifold import CausalManifold

def test_rdd_manifold_generation():
    # 1. Create fake RDD data with a 'cliff'
    # Outcome has a jump at 0, but is very sensitive to bandwidth
    np.random.seed(42)
    X = np.linspace(-1, 1, 100)
    Y = 2 * (X >= 0) + 0.5 * X + np.random.normal(0, 0.1, 100)
    
    # Add some noise/outliers far from cutoff to cause bandwidth sensitivity
    Y[X > 0.8] += 5.0 
    
    rdd = RDD(cutoff=0.0, bandwidth=0.5)
    rdd.fit(Y, X)
    
    data = {'Y': Y, 'RunVar': X}
    engine = CausalManifold(rdd, data)
    
    points = engine.compute_manifold(n_steps=10)
    assert len(points) > 0
    assert all(p.estimate is not None for p in points)
    
    quivers = engine.get_quivers()
    # Should detect some sensitivity due to simulated outliers
    print(f"Detected {len(quivers)} instability arrows.")
    
    # Check if bandwidth sweep works
    bws = [p.params['bandwidth'] for p in points]
    assert min(bws) < 0.5
    assert max(bws) > 0.5

if __name__ == "__main__":
    test_rdd_manifold_generation()
