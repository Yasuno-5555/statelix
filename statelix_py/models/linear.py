import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from ..core import fit_ols_full, predict_ols

class StatelixOLS(BaseEstimator, RegressorMixin):
    """
    Ordinary Least Squares Linear Regression.
    Powered by Statelix C++ Core (Eigen3).
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.result_ = None
        
    def fit(self, X, y):
        """
        Fit linear model.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        self.result_ = fit_ols_full(X, y, self.fit_intercept)
        
        self.coef_ = np.array(self.result_.coef)
        self.intercept_ = self.result_.intercept
        return self
        
    def predict(self, X):
        """
        Predict using the linear model.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted")
            
        X = np.ascontiguousarray(X, dtype=np.float64)
        pred = predict_ols(self.result_, X, self.fit_intercept)
        return np.array(pred)
