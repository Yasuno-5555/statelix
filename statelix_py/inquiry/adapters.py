
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

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
        """
        preds = self.predict(X_new)
        return np.tile(preds[:, None], (1, n_draws))

class LinearAdapter(BaseAdapter):
    """Adapter for OLS and Linear Models with C++/Python Fallback."""
    
    def __init__(self, model):
        self.model = model
        self.use_fallback = False

    def get_metrics(self):
        res = self.model
        metrics = {}
        try:
             if hasattr(res, 'aic'): metrics['aic'] = res.aic
             if hasattr(res, 'bic'): metrics['bic'] = res.bic
             if hasattr(res, 'r_squared'): metrics['r2'] = res.r_squared
             if hasattr(res, 'rsquared'): metrics['r2'] = res.rsquared
             if hasattr(res, 'log_likelihood'): metrics['log_likelihood'] = res.log_likelihood
        except:
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
        elif hasattr(self.model, 'params'):
            params = self.model.params
            if hasattr(params, 'to_dict'): # pandas series
                return params.to_dict()
            return {i: c for i, c in enumerate(params)}
        return {}

class DiscreteAdapter(BaseAdapter):
    """Adapter for Discrete Choice Models (Ordered/Multinomial)."""
    
    def get_coefficients(self):
        # Ordered model/MNLogit usually has params
        if hasattr(self.model, 'result_'):
            res = self.model.result_
            if hasattr(res, 'coef'):
                 if hasattr(res.coef, 'to_dict'):
                     return res.coef.to_dict()
                 # Handle generic pandas Series
                 if hasattr(res.coef, 'keys'):
                      return res.coef
        return {}
        
    def get_metrics(self):
        metrics = {}
        if hasattr(self.model, 'result_'):
            res = self.model.result_
            metrics['aic'] = res.aic
            metrics['bic'] = res.bic
            metrics['pseudo_r2'] = res.pseudo_r2
        return metrics
        
    def predict(self, X_new):
        return self.model.predict(X_new)

class SEMAdapter(BaseAdapter):
    """Adapter for Path/Mediation Analysis."""
    
    def get_coefficients(self):
        # Return path coefficients as a dict with string keys
        coefs = {}
        if hasattr(self.model, 'results_'): # PathAnalysis
            for r in self.model.results_:
                key = f"{r.source} -> {r.target}"
                coefs[key] = r.coef
        elif hasattr(self.model, 'result_'): # Mediation
            res = self.model.result_
            coefs['Direct Effect'] = res.direct_effect
            coefs['Indirect Effect'] = res.indirect_effect
            coefs['Total Effect'] = res.total_effect
        return coefs

    def get_metrics(self):
        metrics = {}
        if hasattr(self.model, 'result_'): # Mediation
             res = self.model.result_
             metrics['proportion_mediated'] = res.proportion_mediated
             metrics['p_value_indirect'] = res.p_value_indirect
        return metrics
        
    def predict(self, X_new):
        # SEM typically doesn't predict Y directly in one step for new data easily
        # without full graph traversal. Return zeros or implement later.
        return np.zeros(len(X_new))


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
        
    def fit(self, X, y):
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        except:
            self.intercept_ = 0.0
            self.coef_ = np.zeros(X.shape[1])
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

class BayesAdapter(BaseAdapter):
    """Adapter for Bayes."""
    def get_metrics(self): return {}
    def predict(self, X): return np.zeros(len(X))
    def get_coefficients(self): return {}

# Safe Import for Causal
try:
    from statelix.causal import BaseCausalModel
except ImportError:
    # Fallback if specific class not found
    class BaseCausalModel: pass

class CausalAdapter(BaseAdapter):
    """Adapter for Causal."""
    def __init__(self, model):
        self.model = model
    def get_coefficients(self):
        if hasattr(self.model, 'params_') and self.model.params_ is not None:
             return {i: c for i, c in enumerate(self.model.params_)}
        return {0: self.model.effect_} if hasattr(self.model, 'effect_') and self.model.effect_ is not None else {}
    def get_metrics(self):
        metrics = {}
        if hasattr(self.model, 'p_value_') and self.model.p_value_ is not None: 
            metrics['p_value'] = self.model.p_value_
        if hasattr(self.model, 'effect_') and self.model.effect_ is not None: 
            metrics['effect'] = self.model.effect_
        return metrics 
    def predict(self, X): return self.model.predict(X)
    def get_assumptions(self): 
        return getattr(self.model, 'assumptions', [])
