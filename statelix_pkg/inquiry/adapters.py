
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
    """Adapter for OLS and Linear Models"""
    def get_metrics(self):
        # Assuming statelix.linear_model.OLSResult has these properties or we compute them
        # Actually OLSResult usually has AIC/BIC.
        # If model is the *Fitted Model Object* or *Result Object*?
        # User sets `fit()` then passes model.
        # Statelix OLS: model.fit() returns Result.
        # So we likely wrap the Result object or Model that holds result.
        # Let's assume we wrap the Result object for now, or Model that has result.
        # Statelix API: OLS.fit() returns OLSResult.
        # So we adapt the RESULT.
        res = self.model
        
        # Check if attributes exist, else compute roughly
        metrics = {}
        if hasattr(res, 'aic'): metrics['aic'] = res.aic
        if hasattr(res, 'bic'): metrics['bic'] = res.bic
        if hasattr(res, 'rsquared'): metrics['r2'] = res.rsquared
        if hasattr(res, 'log_likelihood'): metrics['log_likelihood'] = res.log_likelihood
        return metrics

    def predict(self, X_new):
        # Statelix OLSResult usually has .predict(X)
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_new)
        # Fallback: X @ coef
        if hasattr(self.model, 'coef_'):
            coef = getattr(self.model, 'coef_')
            # Handle intercept? Usually X_new must match training X logic.
            # Assuming X_new matches.
            return X_new @ coef
        return np.zeros(X_new.shape[0])

    def get_coefficients(self):
        if hasattr(self.model, 'coef_'):
            coefs = getattr(self.model, 'coef_')
            # Return as dict with indices
            return {i: c for i, c in enumerate(coefs)}
        return {}

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

