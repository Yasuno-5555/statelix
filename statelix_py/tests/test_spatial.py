
import pytest
import numpy as np
import pandas as pd
from statelix_py.models.spatial import SpatialRegression, SpatialWeights

# Only run if C++ backend is available
try:
    import statelix_core
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_spatial_basics():
    np.random.seed(42)
    n = 50
    k = 2
    
    # 1. Generate Coordinates & Weights
    coords = np.random.rand(n, 2)
    W = SpatialWeights.knn(coords, k=5)
    
    # Check row-standardization
    row_sums = np.sum(W, axis=1)
    assert np.allclose(row_sums, 1.0)
    
    # 2. Generate Data (SAR process)
    # y = rho*W*y + X*beta + e
    # y = (I - rho*W)^-1 (X*beta + e)
    X = np.random.randn(n, k)
    beta = np.array([1.5, -0.8])
    rho = 0.6
    
    I = np.eye(n)
    inv_transform = np.linalg.inv(I - rho * W)
    
    e = np.random.randn(n) * 0.5
    y = inv_transform @ (X @ beta + e)
    
    # 3. Fit SAR Model
    sar = SpatialRegression(model='SAR')
    sar.fit(y, X, W)
    
    res = sar.result_
    
    # Check estimates
    assert abs(res.rho - rho) < 0.2  # Allow some noise
    assert np.allclose(res.coef[['x0', 'x1']], beta, atol=0.3)
    
    # Check effects
    assert res.direct_effects is not None
    # Total effect for x0 should be approx beta[0] / (1-rho)
    expected_total = beta[0] / (1.0 - rho)
    assert abs(res.total_effects['x0'] - expected_total) < 0.5

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_spatial_sem():
    np.random.seed(42)
    n = 50
    k = 2
    
    coords = np.random.rand(n, 2)
    W = SpatialWeights.distance(coords, bandwidth=0.5)
    
    # SEM Process
    # y = X*beta + u
    # u = lambda*W*u + e  => u = (I - lambda*W)^-1 e
    X = np.random.randn(n, k)
    beta = np.array([2.0, 1.0])
    lambda_ = 0.5
    
    I = np.eye(n)
    u = np.linalg.inv(I - lambda_ * W) @ np.random.randn(n) * 0.5
    y = X @ beta + u
    
    sem = SpatialRegression(model='SEM')
    sem.fit(y, X, W)
    
    res = sem.result_
    
    assert abs(res.lambda_ - lambda_) < 0.3
    assert np.allclose(res.coef[['x0', 'x1']], beta, atol=0.3)
