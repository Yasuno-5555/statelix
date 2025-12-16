import pytest
import numpy as np
import statelix_core as sc

class TestLinearModel:
    def setup_method(self):
        # Create a simple synthetic dataset
        # y = 2x1 + 3x2 + 5 + noise
        self.n_samples = 100
        self.n_features = 2
        np.random.seed(42)
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.true_coef = np.array([2.0, 3.0])
        self.intercept = 5.0
        self.noise = np.random.randn(self.n_samples) * 0.1
        self.y = self.X @ self.true_coef + self.intercept + self.noise

    def test_ols_full(self):
        # Test full OLS fitting
        result = sc.fit_ols_full(self.X, self.y, fit_intercept=True)
        
        # Check coefficients
        assert np.allclose(result.coef, self.true_coef, atol=0.2)
        assert np.isclose(result.intercept, self.intercept, atol=0.2)
        
        # Check R-squared (should be high for low noise)
        assert result.r_squared > 0.95
        
        # Check dimensions
        assert result.n_obs == self.n_samples
        assert result.n_params == self.n_features + 1

    def test_ols_no_intercept(self):
        # Test OLS without intercept
        result = sc.fit_ols_full(self.X, self.y, fit_intercept=False)
        assert result.intercept == 0.0
        # Coefficients might be biased because we forced intercept=0 on data with intercept=5
        assert result.n_params == self.n_features

    def test_ols_prediction(self):
        result = sc.fit_ols_full(self.X, self.y, fit_intercept=True)
        
        X_new = np.array([[1.0, 1.0], [0.0, 0.0]])
        preds = sc.predict_ols(result, X_new, fit_intercept=True)
        
        expected_0 = 1.0 * 2.0 + 1.0 * 3.0 + 5.0
        expected_1 = 5.0
        
        assert np.isclose(preds[0], expected_0, atol=0.2)
        assert np.isclose(preds[1], expected_1, atol=0.2)

    def test_ridge_regression(self):
        model = sc.RidgeRegression()
        model.alpha = 1.0
        model.fit(self.X, self.y)
        
        # Ridge should be close to OLS for small alpha and well-conditioned problem
        assert np.allclose(model.coef, self.true_coef, atol=0.5)

    def test_elastic_net(self):
        model = sc.ElasticNet()
        model.alpha = 0.5
        model.l1_ratio = 0.5
        model.fit(self.X, self.y)
        
        assert len(model.coef) == self.n_features
        # Just check it runs and produces finite results
        assert np.all(np.isfinite(model.coef))
