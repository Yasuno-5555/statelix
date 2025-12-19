
import numpy as np
import pandas as pd
from scipy import stats
from .core import BaseCausalModel

class RDD(BaseCausalModel):
    """
    Regression Discontinuity Design (Sharp RDD).
    
    Estimates Local Average Treatment Effect (LATE) at the Cutoff.
    Uses Local Linear Regression within a bandwidth.
    
    Model: Y = beta0 + beta1*(X-c) + tau*Treat + beta2*(X-c)*Treat + e
    where Treat = I(X >= c).
    Effect (tau) is the jump at c.
    
    Args:
        cutoff (float): The threshold value.
        bandwidth (float): The width of the window around cutoff (h).
        kernel (str): 'rectangular' (default) or 'triangular'.
    """
    
    def __init__(self, cutoff=0.0, bandwidth=1.0, kernel='rectangular'):
        super().__init__()
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.assumptions = [
            f"Continuity: Potential outcomes change smoothly at cutoff {cutoff}.",
            "No Manipulation: Agents cannot precisely manipulate the running variable."
        ]
        
    def fit(self, Y, RunVar, Exog=None):
        """
        Fit Sharp RDD.
        
        Args:
            Y (array-like): Outcome.
            RunVar (array-like): Running variable (forcing variable).
            Exog (array-like): Additional controls.
        """
        Y = np.asarray(Y).flatten()
        RunVar = np.asarray(RunVar).flatten()
        n_total = len(Y)
        
        # 1. Filter Data by Bandwidth (Rectangular Kernel)
        # |X - c| <= h
        dist = np.abs(RunVar - self.cutoff)
        mask = dist <= self.bandwidth
        
        Y_sub = Y[mask]
        X_sub = RunVar[mask]
        
        if Exog is not None:
             Exog = np.asarray(Exog)
             if Exog.ndim == 1: Exog = Exog.reshape(-1, 1)
             Exog_sub = Exog[mask]
        else:
             Exog_sub = None
             
        n_sub = len(Y_sub)
        self.n_effective_ = n_sub
        
        if n_sub < 10:
             raise ValueError("Bandwidth too small: insufficient data points.")
             
        # 2. Construct Design Matrix (Local Linear)
        # Centered X: Xc = X - c
        Xc = X_sub - self.cutoff
        Treat = (X_sub >= self.cutoff).astype(float)
        Interaction = Xc * Treat
        
        # Design: [Treat, Xc, Interaction, Exog, Intercept]
        # Effect is Treat coef.
        
        X_list = [Treat.reshape(-1, 1), Xc.reshape(-1, 1), Interaction.reshape(-1, 1)]
        
        if Exog_sub is not None:
             X_list.append(Exog_sub)
             
        X_list.append(np.ones((n_sub, 1))) # Intercept
        
        X_mat = np.hstack(X_list)
        
        # Weights (Triangular Kernel)
        # K(u) = 1 - |u| for |u|<=1
        # u = (X - c)/h
        weights = np.ones(n_sub)
        if self.kernel == 'triangular':
             u = (X_sub - self.cutoff) / self.bandwidth
             weights = 1 - np.abs(u)
             # WLS
             W = np.diag(weights)
             XtW = X_mat.T @ W
             XtWX = XtW @ X_mat
             XtWy = XtW @ Y_sub
             beta = np.linalg.pinv(XtWX) @ XtWy
             
             # Residuals
             y_pred = X_mat @ beta
             residuals = Y_sub - y_pred
             # Sigma2 (Weighted?)
             # Simple approx: Sum( w*r^2 ) / (Sum(w) - k) or similar
             # Using unweighted residuals variance for robust SE?
             # Let's use simple WLS variance formula: (X'WX)^-1 (X'W Sigma W X) (X'WX)^-1
             # Simplify: sigma2 * (X'WX)^-1
             sse = np.sum(weights * residuals**2)
             sigma2 = sse / (np.sum(weights) - X_mat.shape[1])
             cov_matrix = sigma2 * np.linalg.pinv(XtWX)
             
        else: # Rectangular (OLS on subset)
             beta = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T @ Y_sub
             residuals = Y_sub - X_mat @ beta
             sse = np.sum(residuals**2)
             sigma2 = sse / (n_sub - X_mat.shape[1])
             cov_matrix = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)
             
        self.effect_ = beta[0] # Treat
        self.params_ = beta
        self.std_error_ = np.sqrt(cov_matrix[0, 0])
        
        # Inference
        t_stat = self.effect_ / self.std_error_
        self.p_value_ = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_sub - X_mat.shape[1]))
        
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, df=n_sub - X_mat.shape[1])
        self.ci_ = (self.effect_ - t_crit * self.std_error_, self.effect_ + t_crit * self.std_error_)
        
        # Add bandwidth info to assumptions for Storyteller
        self.assumptions.append(f"Bandwidth: {self.bandwidth:.4f}, Kernel: {self.kernel}")
        self.assumptions.append(f"Effective Sample Size: {n_sub} (Total: {n_total})")
        
        self._is_fitted = True
        return self
        
    def predict(self, X):
         # Similar to DiD, requires specific structure.
         # Assume X is valid design matrix.
         if not self._is_fitted: raise RuntimeError("Not fitted")
         X = np.asarray(X)
         if X.shape[1] == len(self.params_):
              return X @ self.params_
         elif X.shape[1] == len(self.params_) - 1:
              return np.hstack([X, np.ones((len(X),1))]) @ self.params_
         raise ValueError("Shape mismatch")
