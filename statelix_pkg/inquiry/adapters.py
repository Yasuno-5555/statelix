
import numpy as np
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    """
    Abstract adapter for Statelix models to unify interactions.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def get_metrics(self) -> dict:
        """Returns dictionary with 'aic', 'bic', 'r2', 'log_likelihood'."""
        pass

    @abstractmethod
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Returns point predictions."""
        pass
    
    @abstractmethod
    def get_coefficients(self) -> dict:
        """Returns dictionary of {feature_index: value}."""
        pass

    def simulate(self, X_new: np.ndarray, n_draws: int = 1000) -> np.ndarray:
        """
        Returns prediction samples (n_samples, n_draws).
        Default implementation uses simple parametric bootstrap if coef/se available,
        otherwise returns point estimate repeated.
        """
        preds = self.predict(X_new)
        # Rudimentary fallback: return replicas
        return np.tile(preds[:, None], (1, n_draws))

class LinearAdapter(BaseAdapter):
    """Adapter for OLS and Linear Models with C++/Python Fallback."""
    
    def __init__(self, model):
        self.model = model
        self.use_fallback = False
        
        # Determine if we are wrapping a C++ result or a Mock result
        # C++ Result usually has explicit C++ type
        # Mock usually Python dict or object
        pass

    def get_metrics(self):
        # 1. Try Standard Attributes (C++ or Statsmodels)
        res = self.model
        metrics = {}
        
        try:
             # C++ Binding often exposes properties directly
             if hasattr(res, 'aic'): metrics['aic'] = res.aic
             if hasattr(res, 'bic'): metrics['bic'] = res.bic
             if hasattr(res, 'r_squared'): metrics['r2'] = res.r_squared
             if hasattr(res, 'rsquared'): metrics['r2'] = res.rsquared # Statsmodels/Mock compatibility
             if hasattr(res, 'log_likelihood'): metrics['log_likelihood'] = res.log_likelihood
        except:
             pass
             
        # 2. Heuristic Calculation if missing (for Mock)
        if 'r2' not in metrics and hasattr(res, 'coef_'):
             # We can't easily compute R2 without data.
             pass
             
        return metrics

    def predict(self, X_new):
        if hasattr(self.model, 'predict'):
            try:
                return self.model.predict(X_new)
            except:
                pass
        
        # Fallback: X @ coef
        if hasattr(self.model, 'coef_'):
            coef = getattr(self.model, 'coef_')
            intercept = getattr(self.model, 'intercept_', 0.0)
            return X_new @ coef + intercept
            
        return np.zeros(X_new.shape[0])

    def get_coefficients(self):
        if hasattr(self.model, 'coef_'):
            coefs = getattr(self.model, 'coef_')
            return {i: c for i, c in enumerate(coefs)}
        return {}

# --- HELPER: Safe Import & Factory ---
class StatelixLinearFactory:
    """Safely returns C++ OLS or Python Mock OLS."""
    @staticmethod
    def get_ols():
        try:
            from statelix.linear_model import FitOLS
            return FitOLS
        except ImportError:
            return MockOLS

class MockOLS:
    """Pure Python fallback for OLS."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        self.r_squared = 0.0
        self.aic = 0.0
        self.bic = 0.0
        
    def fit(self, X, y):
        # Simple Numpy OLS
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
            
            # Simple Stats
            y_pred = X_aug @ beta
            resid = y - y_pred
            sse = np.sum(resid**2)
            sst = np.sum((y - np.mean(y))**2)
            self.r_squared = 1 - sse/sst if sst > 1e-9 else 0.0
            
            # Mock AIC
            n = len(y)
            k = len(beta)
            self.aic = n * np.log(sse/n) + 2*k
        except:
            self.intercept_ = 0.0
            self.coef_ = np.zeros(X.shape[1])
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

class BayesAdapter(BaseAdapter):
    """Adapter for BayesianLinearRegression"""
    def __init__(self, model_obj, chain_result=None):
        # model_obj is BayesianLinearRegression instance
        # chain_result is HMCResult (optional)
        self.model = model_obj
        self.chain = chain_result

    def get_metrics(self):
        # Bayes Comparison uses WAIC or LOO usually.
        # For now, if we have chain, we can compute DIC/WAIC approx.
        # If simply MAP fitted, usage is like Linear.
        # We assume HMC run for now for "Bayes Experience".
        metrics = {}
        if self.chain is not None:
             # Calculate simple metrics from mean posterior
             pass
        return {} # TODO: Implement WAIC

    def predict(self, X_new):
        # Use mean of coefficients
        if self.chain is not None:
            mean_theta = self.chain.mean
            # Last param is log_sigma, rest are beta
            beta = mean_theta[:-1] 
            return X_new @ beta
        elif hasattr(self.model, 'map_theta') and self.model.map_theta.size > 0:
             beta = self.model.map_theta[:-1]
             return X_new @ beta
        return np.zeros(X_new.shape[0])

    def simulate(self, X_new, n_draws=1000):
        # Ideally use chain samples
        if self.chain is not None:
            samples = self.chain.samples # (n_stored, dim)
            # Use up to n_draws
            n_stored = samples.shape[0]
            indices = np.random.choice(n_stored, n_draws)
            
            # Prediction matrix: (n_obs, n_draws)
            # X: (n_obs, k)
            # Betas: (n_draws, k)
            betas = samples[indices, :-1] # Exclude log_sigma
            return X_new @ betas.T
        return super().simulate(X_new, n_draws)

    def get_coefficients(self):
        # Return posterior means
        if self.chain is not None:
            mean = self.chain.mean[:-1]
            return {i: c for i, c in enumerate(mean)}
        elif hasattr(self.model, 'map_theta') and self.model.map_theta.size > 0:
            mean = self.model.map_theta[:-1]
            return {i: c for i, c in enumerate(mean)}
        return {}

# Causal Adapter using BaseCausalModel
from statelix.causal.core import BaseCausalModel

class CausalAdapter(BaseAdapter):
    """Adapter for Statelix Causal Models."""
    def __init__(self, model):
        if not isinstance(model, BaseCausalModel):
            raise TypeError("Model must be an instance of BaseCausalModel")
        self.model = model
        
    def get_coefficients(self):
        # Return params_ if available as dict
        if self.model.params_ is not None:
             return {i: c for i, c in enumerate(self.model.params_)}
        return {0: self.model.effect_} if self.model.effect_ is not None else {}
        
    def get_metrics(self):
        metrics = {}
        if self.model.p_value_ is not None:
            metrics['p_value'] = self.model.p_value_
        if self.model.effect_ is not None:
             # Store effect/std_error specific to Causal
            metrics['effect'] = self.model.effect_
            metrics['std_error'] = self.model.std_error_
        return metrics 
        
    def get_assumptions(self):
        return self.model.assumptions

    def predict(self, X):
        return self.model.predict(X)

