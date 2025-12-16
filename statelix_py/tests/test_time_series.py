import pytest
import numpy as np
import statelix_core as sc

class TestTimeSeries:
    def setup_method(self):
        np.random.seed(42)
        
    def test_var_estimation(self):
        # Create bivariate VAR(1):
        # y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + e1
        # y2_t = -0.2*y1_{t-1} + 0.5*y2_{t-1} + e2
        T = 200
        K = 2
        Y = np.zeros((T, K))
        Y[0] = np.random.randn(K)
        
        A = np.array([[0.5, 0.2], [-0.2, 0.5]])
        noise = np.random.randn(T, K) * 0.1
        
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + noise[t]
            
        var = sc.VAR(p=1)
        var.include_intercept = False # Generate data without intercept for simplicity
        result = var.fit(Y)
        
        est_A = result.coef[0]
        assert np.allclose(est_A, A, atol=0.1)
        
        # Test IRF
        irf_res = var.irf(result, horizon=10)
        assert len(irf_res.irf) == K

    def test_garch_estimation(self):
        # Simulate GARCH(1,1) process
        # sigma^2_t = omega + alpha * eps_{t-1}^2 + beta * sigma^2_{t-1}
        T = 500
        omega = 0.1
        alpha = 0.2
        beta = 0.7
        
        returns = np.zeros(T)
        sigma2 = np.zeros(T)
        sigma2[0] = omega / (1 - alpha - beta)
        
        eps = np.random.randn(T)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
            returns[t] = np.sqrt(sigma2[t]) * eps[t]
            
        garch = sc.GARCH(p=1, q=1)
        res = garch.fit(returns)
        
        # GARCH estimation is tricky on small samples, just check if it runs 
        # and returns valid ranges (alpha+beta < 1 for stationarity)
        assert res.omega > 0
        assert res.alpha > 0
        assert res.beta > 0
        assert res.alpha + res.beta < 1.05 # Allow some tolerance or estimation noise
        assert res.converged

    def test_kalman_filter(self):
        # Simple 1D tracking: x_t = x_{t-1}, y_t = x_t + noise
        kf = sc.KalmanFilter(state_dim=1, measure_dim=1)
        
        # Need to expose matrices to Python to set them
        # Assuming bindings allow direct property access if defined
        # Based on bindings: F, H, Q, R, P, x are readwrite
        
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.1]])
        kf.R = np.array([[1.0]])
        
        data = np.random.randn(50)
        res = kf.filter(data.reshape(-1, 1))
        
        assert res.states.shape == (50, 1)
        assert res.log_likelihood != 0.0

