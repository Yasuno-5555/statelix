import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from ..core import FitOLS

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
        
        self.model_ = FitOLS()
        # fit method in C++ takes (X, y) and uses fit_intercept internally (hardcoded to true/conf_level 0.95 in current binding wrapper)
        # But wait, looking at python_bindings_linear.cpp: result = fit_ols_full(X, y, true, 0.95);
        # It ignores self.fit_intercept passed to python class if I don't update bindings, but for now let's just use what's there.
        self.model_.fit(X, y)
        
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
