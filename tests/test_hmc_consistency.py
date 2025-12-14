import numpy as np
import pytest
from statelix_py.models import StatelixHMC

def test_hmc_gaussian_consistency():
    """
    Test that HMC correctly samples from a 2D correlated Gaussian.
    Target: mu = [0, 0], Sigma = [[1, 0.8], [0.8, 1]]
    """
    # Target precision matrix (inverse covariance)
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    Precision = np.linalg.inv(Sigma)
    
    def log_prob(theta):
        # log_p = -0.5 * theta.T @ Precision @ Θ
        # grad = -Precision @ theta
        lp = -0.5 * theta @ Precision @ theta
        grad = -Precision @ theta
        return lp, grad

    # High sample count for statistical checks
    hmc = StatelixHMC(n_samples=2000, warmup=500, step_size=0.1, 
                      target_accept=0.8, seed=42)
    
    theta0 = np.array([0.0, 0.0])
    res = hmc.sample(log_prob, theta0)
    
    samples = res.samples # (n_samples, 2)
    
    # 1. Check Diagnostics
    assert res.acceptance_rate > 0.6, f"Acceptance rate too low: {res.acceptance_rate}"
    assert np.all(res.ess > 100), f"ESS too low: {res.ess}"
    
    # 2. Check Moments
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)
    
    # Tolerances using standard error of mean (~1/sqrt(N))
    # Mean should be 0. 2000 samples => stderr ~ 1/sqrt(2000) ~ 0.02
    # assert abs(mean) < 3 * stderr
    assert np.all(np.abs(sample_mean) < 0.1), f"Mean bias: {sample_mean}"
    
    # Covariance elements should be close to Sigma
    # Error roughly 0.05-0.1 for 2000 samples
    diff_cov = np.abs(sample_cov - Sigma)
    assert np.all(diff_cov < 0.15), f"Covariance mismatch:\n{sample_cov}\nvs\n{Sigma}"

def test_hmc_divergence_handling():
    """
    Test that HMC handles difficult geometries without crashing, 
    potentially reporting divergences.
    (Simple case: just ensure it runs on a funnel-like shape or similar, 
    but for now just a basic check that it returns).
    """
    # Simple 1D normal
    def log_prob(x):
        return -0.5 * x[0]**2, np.array([-x[0]])
        
    hmc = StatelixHMC(n_samples=100, warmup=10, seed=1)
    res = hmc.sample(log_prob, np.array([10.0]))
    assert len(res.samples) == 100
