import numpy as np
import pytest
from statelix_py.models import StatelixOLS, StatelixHNSW

def test_ols_synthetic():
    """
    Test OLS on exact linear data: y = 2x + 1
    """
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([3, 5, 7, 9], dtype=np.float64) # 2*x + 1
    
    model = StatelixOLS(fit_intercept=True)
    model.fit(X, y)
    
    assert np.isclose(model.coef_[0], 2.0)
    assert np.isclose(model.intercept_, 1.0)
    
    pred = model.predict(np.array([[5]]))
    assert np.isclose(pred[0], 11.0)

def test_hnsw_exact_match():
    """
    Test HNSW recall on a small dataset where it should be perfect.
    """
    # 5 points on a line
    X = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0]
    ], dtype=np.float64)
    
    model = StatelixHNSW(M=16, ef_construction=100, n_neighbors=1)
    model.fit(X)
    
    # Query with the same points
    # Nearest neighbor of 1.0 should be index 0
    indices = model.transform(X)
    
    expected = np.arange(5).reshape(-1, 1)
    # HNSW returns 'self' as the 1st neighbor commonly if included.
    np.testing.assert_array_equal(indices, expected)

def test_hnsw_input_validation():
    """
    Ensure HNSW accepts float64 list and converts or raises if invalid.
    """
    X = [[1, 2], [3, 4]]
    model = StatelixHNSW()
    # Should work (internal conversion)
    model.fit(X)
    assert model._index.size == 2
