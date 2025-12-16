import pytest
import numpy as np
import statelix_core as sc

class TestPanel:
    def setup_method(self):
        np.random.seed(42)
        
    def test_fixed_effects(self):
        # Y_it = X_it * beta + alpha_i + u_it
        n_units = 50
        n_periods = 10
        N = n_units * n_periods
        
        unit_ids = np.repeat(np.arange(n_units), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_units)
        
        X = np.random.randn(N, 1)
        alpha = np.random.randn(n_units)
        alpha_expanded = alpha[unit_ids]
        
        true_beta = 3.5
        u = np.random.randn(N) * 0.5
        
        Y = X.flatten() * true_beta + alpha_expanded + u
        
        fe = sc.panel.FixedEffects()
        # Ensure input types match bindings (VectorXd, MatrixXd, VectorXi, VectorXi)
        result = fe.fit(Y, X, unit_ids.astype(np.int32), time_ids.astype(np.int32))
        
        assert np.isclose(result.coef[0], true_beta, atol=0.2)
        assert result.n_units == n_units
        assert result.n_periods == n_periods

    def test_random_effects(self):
        # Similar setup but assuming alpha_i is random noise uncorrel with X
        n_units = 50
        n_periods = 10
        N = n_units * n_periods
        
        unit_ids = np.repeat(np.arange(n_units), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_units)
        
        X = np.random.randn(N, 1)
        alpha = np.random.randn(n_units) # Random effect
        alpha_expanded = alpha[unit_ids]
        
        true_beta = 2.0
        u = np.random.randn(N) * 0.5
        
        Y = X.flatten() * true_beta + alpha_expanded + u
        
        re = sc.panel.RandomEffects()
        result = re.fit(Y, X, unit_ids.astype(np.int32), time_ids.astype(np.int32))
        
        # RE should be consistent here too
        assert np.isclose(result.coef[0], true_beta, atol=0.2)
