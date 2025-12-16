
import pytest
import numpy as np
import pandas as pd
from statelix_py.stats.resampling import Resampler, bootstrap, block_bootstrap

# Check if C++ backend is available
try:
    import statelix_core
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_bootstrap_mean_se():
    np.random.seed(42)
    # Generate data from N(10, 4) -> std error of mean should be 2 / sqrt(n)
    n = 100
    sigma = 2.0
    data = np.random.normal(10, sigma, n)
    
    expected_se = sigma / np.sqrt(n)
    
    # Statistic: Mean
    def calc_mean(x):
        return np.array([np.mean(x)])
        
    res = bootstrap(data, calc_mean, n_reps=2000, seed=123)
    
    # Check if bootstrap SE is close to theoretical SE
    # Allow 10% margin due to sampling noise
    assert abs(res.std_error[0] - expected_se) < 0.1 * expected_se
    
    # Check bias is small
    assert abs(res.bias[0]) < 0.1

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_jackknife_bias():
    # Estimator: Sample Variance (biased by factor (n-1)/n)
    # Jackknife can correct this bias for O(1/n) biases
    data = np.array([1, 2, 3, 4, 5], dtype=float)
    n = len(data)
    
    # Biased estimator: Population variance formula
    def biased_var(x):
        return np.array([np.var(x)]) # Defaults to ddof=0
        
    rs = Resampler()
    jk_res = rs.jackknife(data, biased_var)
    
    # Bias estimate should be approx -sigma^2/n
    # True variance = 2.5 (population), Sample var (biased) = 2.0
    # Bias = -0.5
    
    assert jk_res.bias[0] < -0.1 # Should detect negative bias
    
    # Bias corrected estimate = Original - Bias
    corrected = jk_res.original_stat[0] - jk_res.bias[0]
    unbiased_var = np.var(data, ddof=1)
    
    assert abs(corrected - unbiased_var) < 1e-10

@pytest.mark.skipif(not HAS_CPP, reason="Requires compiled statelix_core")
def test_block_bootstrap():
    # Time series data 0, 1, 2, ...
    data = np.arange(20).reshape(-1, 1)
    
    def get_first_lag_corr(x):
        if len(x) < 2: return np.array([0.0])
        return np.array([np.corrcoef(x[:-1,0], x[1:,0])[0,1]])
        
    # Standard bootstrap destroys autocorrelation -> corr should drop
    # Block bootstrap preserves it -> corr should stay high
    
    rs = Resampler(seed=42)
    
    # Block size 5 (large enough to capture lag-1)
    res_block = rs.block_bootstrap(data, get_first_lag_corr, block_size=5, n_reps=100)
    
    mean_corr_block = np.mean(res_block.resampled_stats)
    assert mean_corr_block > 0.8 # Should remain high
    
    # Standard bootstrap for comparison (manual via Resampler)
    res_iid = rs.bootstrap(data, get_first_lag_corr, n_reps=100)
    mean_corr_iid = np.mean(res_iid.resampled_stats)
    
    assert mean_corr_iid < 0.2 # Should be destroyed (scrambled order)
