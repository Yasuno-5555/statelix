import pytest
import numpy as np
import statelix_core as sc

class TestBayes:
    def setup_method(self):
        np.random.seed(42)

    def test_hmc_simple_gaussian(self):
        # Target: 2D Gaussian with mean [1, 2] and covariance [[1, 0.5], [0.5, 1]]
        # Log Prob: -0.5 * (x - mu)^T Sigma^-1 (x - mu)
        
        mu = np.array([1.0, 2.0])
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        Sigma_inv = np.linalg.inv(Sigma)
        
        def log_prob_func(x):
            diff = x - mu
            # log probability (ignoring constants)
            logp = -0.5 * (diff @ Sigma_inv @ diff)
            # Gradient: -Sigma^-1 (x - mu)
            grad = -Sigma_inv @ diff
            return logp, grad
        
        config = sc.bayes.HMCConfig()
        config.n_samples = 1000
        config.warmup = 500
        config.step_size = 0.1
        config.n_leapfrog = 10
        config.seed = 42
        
        theta0 = np.zeros(2)
        
        # This uses the C++ HMC with Python callback
        result = sc.hmc_sample(log_prob_func, theta0, config)
        
        samples = result.samples
        # Check posterior mean recovery
        est_mean = np.mean(samples, axis=0)
        assert np.allclose(est_mean, mu, atol=0.2)
        
        # Check acceptance rate
        assert config.target_accept * 0.5 < result.acceptance_rate < 1.0

    def test_hmc_logistic_native(self):
        # Logistic regression using purely native C++ objective (no Python overhead)
        n = 200
        k = 2
        X = np.random.randn(n, k)
        true_beta = np.array([0.5, -0.5])
        logits = X @ true_beta
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = np.random.binomial(1, probs).astype(float)
        
        config = sc.bayes.HMCConfig()
        config.n_samples = 500
        config.warmup = 200
        config.step_size = 0.05
        config.n_leapfrog = 5
        
        # hmc_sample_logistic avoids Python GIL for gradients
        # Use VectorXd for y? bindings expect VectorXd
        y_vec = y # numpy array converts to VectorXd automatically in pybind11
        
        result = sc.hmc_sample_logistic(X, y_vec, config, prior_std=10.0)
        
        est_beta = result.mean
        assert np.allclose(est_beta, true_beta, atol=0.3)
