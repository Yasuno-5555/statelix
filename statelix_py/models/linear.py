import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from ..core import FitOLS
try:
    import statelix.accelerator as acc
    HAS_ACCELERATOR = True
except ImportError:
    HAS_ACCELERATOR = False

class StatelixOLS(BaseEstimator, RegressorMixin):
    """
    Ordinary Least Squares Linear Regression.
    Powered by Statelix C++ Core (Eigen3).
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.model_ = None
        
    def fit(self, X, y):
        """
        Fit linear model.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        gram = None
        # Attempt GPU Acceleration if problem is large enough
        if HAS_ACCELERATOR and X.shape[0] > 10000:
            if acc.is_available():
                try:
                    # Construct augmented matrix [1, X] because FitOLS enforces intercept
                    # Note: If self.fit_intercept is False, logic differs, but current FitOLS hardcodes True.
                    # We assume True for now to match C++.
                    N, K = X.shape
                    X_aug = np.empty((N, K + 1), dtype=np.float64)
                    X_aug[:, 0] = 1.0
                    X_aug[:, 1:] = X
                    
                    weights = np.ones(N, dtype=np.float64)
                    
                    gram = acc.weighted_gram_matrix(X_aug, weights)
                except Exception:
                    # Fallback to CPU silently on any GPU error
                    gram = None

        self.model_ = FitOLS()
        self.model_.fit(X, y, gram=gram)
        
        self.coef_ = np.array(self.model_.coef_)
        self.intercept_ = self.model_.intercept_
        return self
        
    def predict(self, X):
        """
        Predict using the linear model.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
            
        X = np.ascontiguousarray(X, dtype=np.float64)
        pred = self.model_.predict(X)
        return np.array(pred)
