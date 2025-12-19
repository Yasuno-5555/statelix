
import numpy as np
import pandas as pd
from scipy import stats
from .core import BaseCausalModel

class DiffInDiff(BaseCausalModel):
    """
    Difference-in-Differences (DiD) Estimator (2x2).
    
    Estimates Average Treatment Effect on Treated (ATT).
    Model: Y = beta0 + beta1*Group + beta2*Post + delta*(Group*Post) + gamma*X + e
    
    Effect (delta) is the coefficient of the interaction term.
    
    Args:
        fit_intercept (bool): Always True for DiD usually.
    """
    
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.assumptions = [
            "Parallel Trends: In absence of treatment, the average difference between groups would remain constant over time.",
            "SUTVA: No spillover effects between units."
        ]
        
    def fit(self, Y, Group, Time, Exog=None):
        """
        Fit 2x2 DiD.
        
        Args:
            Y (array-like): Outcome.
            Group (array-like): Treatment Group indicator (1=Treated, 0=Control).
            Time (array-like): Time indicator (1=Post, 0=Pre).
            Exog (array-like): Controls.
        """
        Y = np.asarray(Y).flatten()
        n = len(Y)
        
        Group = np.asarray(Group).flatten()
        Time = np.asarray(Time).flatten()
        
        # Interaction Term (The Treatment Effect)
        Interaction = Group * Time
        
        # Design Matrix
        # X = [Interaction, Group, Time, Exog, Intercept]
        X_list = [Interaction.reshape(-1, 1), Group.reshape(-1, 1), Time.reshape(-1, 1)]
        
        if Exog is not None:
            Exog = np.asarray(Exog)
            if Exog.ndim == 1: Exog = Exog.reshape(-1, 1)
            X_list.append(Exog)
            
        if self.fit_intercept:
            X_list.append(np.ones((n, 1)))
            
        X_mat = np.hstack(X_list)
        
        # OLS
        # beta = (X'X)^-1 X'Y
        XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
        beta = XtX_inv @ X_mat.T @ Y
        
        self.effect_ = beta[0] # Interaction term is first
        self.params_ = beta
        
        # Standard Errors (Simple OLS, not clustered yet - warning?)
        residuals = Y - X_mat @ beta
        sigma2 = np.sum(residuals**2) / (n - X_mat.shape[1])
        cov_matrix = sigma2 * XtX_inv
        
        self.std_error_ = np.sqrt(cov_matrix[0, 0])
        
        # T-test
        t_stat = self.effect_ / self.std_error_
        self.p_value_ = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - X_mat.shape[1]))
        
        # CI
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, df=n - X_mat.shape[1])
        self.ci_ = (self.effect_ - t_crit * self.std_error_, self.effect_ + t_crit * self.std_error_)
        
        self._is_fitted = True
        return self
        
    def predict(self, X):
        """
        Predict Y given structural inputs [Interaction, Group, Time, Exog].
        Note: The user must provide the interaction term manually if calling predict directly,
        or we assume X matches fit structure.
        Ideally X should be [Group, Time, Exog], and we construct Interaction.
        But BaseCausalModel predict(X) usually takes raw features.
        
        For DiD, predicting counterfactuals:
        - Treatment Off: Set Interaction=0.
        
        For simplicity, we assume X follows fit signature (Group, Time, Exog).
        Wait. BaseCausalModel.predict(X). X is one argument.
        If X is dataframe with columns?
        For now, let's assume X is raw design matrix OR dictionary.
        Let's assume X contains [Group, Time, Exog].
        """
        # Simplistic implementation assuming X is properly formatted matching params structure?
        # That's dangerous.
        # Let's rely on user passing the correct columns, including Interaction if they computed it.
        # OR:
        # If we stick to "Inference Only", we could raise NotImplementedError.
        # BUT WhatIf needs prediction.
        # Let's assume X is the full Design Matrix matching params.
        
        if not self._is_fitted:
            raise RuntimeError("Not fitted")
            
        # Very simple dot product
        # Ensure constant is added if needed
        # We can't easily auto-add constant because we don't know X structure perfectly here.
        # Fallback: Assume X matches params length exactly.
        X = np.asarray(X)
        if hasattr(self, 'params_') and len(self.params_) == X.shape[1]:
             return X @ self.params_
        elif hasattr(self, 'params_') and len(self.params_) == X.shape[1] + 1 and self.fit_intercept:
             # Assume intercept missing
             return np.hstack([X, np.ones((len(X), 1))]) @ self.params_
        else:
             raise ValueError(f"Input X shape {X.shape} does not match params shape {self.params_.shape}")
