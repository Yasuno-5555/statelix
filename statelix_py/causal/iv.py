
import numpy as np
import pandas as pd
from scipy import stats
from .core import BaseCausalModel

class IV2SLS(BaseCausalModel):
    """
    Instrumental Variables Estimator (2SLS).
    
    Estimates the effect of Endogenous variable X on Y using Instrument Z.
    
    Model:
        1st Stage: X = gamma*Z + pi*W + u
        2nd Stage: Y = beta*X_hat + delta*W + v
        
    Args:
        fit_intercept (bool): Whether to include intercept.
    """
    
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.assumptions = [
            "Relevance: Cov(Z, X) != 0 (Instrument must correlate with Endogenous)",
            "Exclusion: Cov(Z, v) = 0 (Instrument affects Y only through X)",
            "Exogeneity: Cov(W, v) = 0 (Controls are exogenous)"
        ]
        self.first_stage_f_stat_ = None
        self.r_squared_ = None
        
    def fit(self, Y, Endog, Instruments, Exog=None):
        """
        Fit 2SLS.
        
        Args:
            Y (array-like): Dependent variable (n_samples,).
            Endog (array-like): Endogenous variable (n_samples, 1 or more).
            Instruments (array-like): Instrumental variables (n_samples, k).
            Exog (array-like, optional): Exogenous control variables (n_samples, m).
        """
        # Data Prep
        Y = np.asarray(Y).flatten()
        n = len(Y)
        
        Endog = np.asarray(Endog)
        if Endog.ndim == 1: Endog = Endog.reshape(-1, 1)
            
        Instruments = np.asarray(Instruments)
        if Instruments.ndim == 1: Instruments = Instruments.reshape(-1, 1)
        
        # Prepare Design Matrix for 1st Stage
        # RHS_1st = [Instruments, Exog, Intercept]
        # Endog is approximated by RHS_1st
        
        if Exog is not None:
            Exog = np.asarray(Exog)
            if Exog.ndim == 1: Exog = Exog.reshape(-1, 1)
            
        # Construct X_first (Instruments + Exog)
        X_first_list = [Instruments]
        if Exog is not None:
            X_first_list.append(Exog)
            
        if self.fit_intercept:
            X_first_list.append(np.ones((n, 1)))
            
        X_first = np.hstack(X_first_list)
        
        # First Stage Regression: Endog ~ X_first
        # beta_1st = (X_first' X_first)^-1 X_first' Endog
        
        XtX_inv_1st = np.linalg.pinv(X_first.T @ X_first)
        beta_1st = XtX_inv_1st @ X_first.T @ Endog
        
        Endog_hat = X_first @ beta_1st
        
        # Diagnostics: Weak Instruments (F-stat of Instruments in 1st stage)
        # We test joint significance of Instruments on Endog (controlling for Exog)
        # Simplified: F-test of regression Endog ~ Instruments + Exog vs Null model (~ Exog)
        # BUT here we need Partial F-test of Instruments.
        # SSR_restricted (only Exog) vs SSR_unrestricted (Instruments + Exog)
        
        # Calc F-stat for first endogenous variable (assuming 1 for narrative simplicity)
        resid_1st = Endog - Endog_hat
        ssr_unrestricted = np.sum(resid_1st**2)
        
        # Restricted model (Exog only)
        X_rest_list = []
        if Exog is not None:
             X_rest_list.append(Exog)
        if self.fit_intercept:
             X_rest_list.append(np.ones((n, 1)))
             
        if not X_rest_list:
            # Null model (Mean only if intercept, else 0)
             # If no intercept and no controls, restricted model is y=0
             ssr_restricted = np.sum(Endog**2)
             df_rest_params = 0
        else:
             X_rest = np.hstack(X_rest_list)
             beta_rest = np.linalg.pinv(X_rest.T @ X_rest) @ X_rest.T @ Endog
             resid_rest = Endog - X_rest @ beta_rest
             ssr_restricted = np.sum(resid_rest**2)
             df_rest_params = X_rest.shape[1]
             
        df_unrest_params = X_first.shape[1]
        n_instruments = Instruments.shape[1] # This is q in F formula
        df_denom = n - df_unrest_params
        
        # F = ((SSR_r - SSR_ur) / q) / (SSR_ur / (n - k))
        f_stat = ((ssr_restricted - ssr_unrestricted) / n_instruments) / (ssr_unrestricted / df_denom)
        self.first_stage_f_stat_ = f_stat
        
        if f_stat < 10:
             self.assumptions.append("[WARNING] Weak Instruments Detected (F < 10). Results may be biased.")
        else:
             self.assumptions.append("Instruments appear strong (F > 10).")

        # Second Stage: Y ~ Endog_hat + Exog
        # Construct Design Matrix for 2nd Stage
        # RHS_2nd = [Endog_hat, Exog, Intercept]
        
        X_second_list = [Endog_hat]
        if Exog is not None:
            X_second_list.append(Exog)
        if self.fit_intercept:
            X_second_list.append(np.ones((n, 1)))
            
        X_second = np.hstack(X_second_list)
        
        XtX_inv_2nd = np.linalg.pinv(X_second.T @ X_second)
        beta_2nd = XtX_inv_2nd @ X_second.T @ Y
        
        # Extract Causal Effect (Coefficient of Endog_hat)
        # Assuming Endog is first column(s)
        self.effect_ = beta_2nd[0] 
        self.params_ = beta_2nd
        
        # Standard Errors
        # CRITICAL: Must use Original Endog for residuals, not Endog_hat
        X_structural_list = [Endog]
        if Exog is not None:
            X_structural_list.append(Exog)
        if self.fit_intercept:
            X_structural_list.append(np.ones((n, 1)))
        
        X_structural = np.hstack(X_structural_list)
        
        residuals_structural = Y - X_structural @ beta_2nd
        sigma2 = np.sum(residuals_structural**2) / (n - X_second.shape[1])
        
        # Var-Cov = sigma2 * (X_hat' X_hat)^-1  <-- Standard 2SLS Variance
        # Note: X_hat is used in inverse part, but sigma2 uses structural residuals
        
        cov_matrix = sigma2 * XtX_inv_2nd
        
        self.std_error_ = np.sqrt(cov_matrix[0, 0])
        
        # T-test
        t_stat = self.effect_ / self.std_error_
        self.p_value_ = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - X_second.shape[1]))
        
        # Confidence Interval
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, df=n - X_second.shape[1])
        self.ci_ = (self.effect_ - t_crit * self.std_error_, self.effect_ + t_crit * self.std_error_)
        
        # R2 (Structural)
        sst = np.sum((Y - np.mean(Y))**2)
        sse = np.sum(residuals_structural**2)
        self.r_squared_ = 1 - sse/sst
        
        self.r_squared_ = 1 - sse/sst
        
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict Structural Y given X (Endog or Endog+Exog).
        X must match the structure of [Endog, Exog] provided in fit.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted")
            
        X = np.asarray(X)
        if self.fit_intercept:
             # Check if X already has intercept? User usually provides raw features.
             # We should append constant.
             # BUT params_ includes intercept at end.
             X_aug = np.hstack([X, np.ones((len(X), 1))])
        else:
             X_aug = X
             
        # params_ = [Endog_coefs, Exog_coefs, Intercept]
        # X provided by user usually contains [Endog, Exog columns].
        
        return X_aug @ self.params_
