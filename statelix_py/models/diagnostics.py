"""
Statelix Regression Diagnostics Module

Provides:
  - vif: Variance Inflation Factor (multicollinearity)
  - durbin_watson: Autocorrelation test
  - breusch_pagan: Heteroscedasticity test
  - white_test: White's test for heteroscedasticity
  - cooks_distance: Outlier detection
  - leverage: Influence diagnostics
  - RegressionDiagnostics: Comprehensive diagnostic class
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, List
from scipy import stats


@dataclass
class DiagnosticResult:
    """Generic diagnostic result."""
    name: str
    statistic: float
    p_value: Optional[float] = None
    conclusion: Optional[str] = None
    
    def __repr__(self) -> str:
        result = f"{self.name}: {self.statistic:.4f}"
        if self.p_value is not None:
            sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
            result += f" (p={self.p_value:.4g} {sig})"
        if self.conclusion:
            result += f"\n  → {self.conclusion}"
        return result


def vif(X: Union[np.ndarray, pd.DataFrame], add_constant: bool = False) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each predictor.
    
    VIF > 5-10 indicates problematic multicollinearity.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
    add_constant : bool
        Whether to add a constant column (intercept).
        
    Returns
    -------
    DataFrame with VIF for each predictor.
    
    Examples
    --------
    >>> vif_result = vif(X)
    >>> print(vif_result)
         Variable    VIF
    0          x1   1.23
    1          x2   5.67
    """
    if isinstance(X, pd.DataFrame):
        names = X.columns.tolist()
        X = X.values
    else:
        X = np.asarray(X)
        names = [f"x{i}" for i in range(X.shape[1])]
    
    if add_constant:
        X = np.column_stack([np.ones(X.shape[0]), X])
        names = ["const"] + names
    
    n_features = X.shape[1]
    vif_values = []
    
    for i in range(n_features):
        # Regress x_i on all other predictors
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        if X_others.shape[1] == 0:
            vif_values.append(1.0)
            continue
        
        # OLS: y_i = X_others @ beta
        # R² = 1 - SS_res / SS_tot
        # VIF = 1 / (1 - R²)
        
        # Add constant to X_others for regression
        X_aug = np.column_stack([np.ones(X_others.shape[0]), X_others])
        
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X_aug, y_i, rcond=None)
            y_pred = X_aug @ beta
            ss_res = np.sum((y_i - y_pred) ** 2)
            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
            
            if ss_tot == 0:
                vif_i = np.inf
            else:
                r_squared = 1 - ss_res / ss_tot
                if r_squared >= 1:
                    vif_i = np.inf
                else:
                    vif_i = 1 / (1 - r_squared)
        except:
            vif_i = np.inf
            
        vif_values.append(vif_i)
    
    return pd.DataFrame({
        'Variable': names,
        'VIF': vif_values
    })


def durbin_watson(residuals: np.ndarray) -> DiagnosticResult:
    """
    Durbin-Watson test for autocorrelation in residuals.
    
    Values near 2 indicate no autocorrelation.
    Values < 2 indicate positive autocorrelation.
    Values > 2 indicate negative autocorrelation.
    
    Parameters
    ----------
    residuals : array-like
        Regression residuals.
        
    Returns
    -------
    DiagnosticResult
    
    Examples
    --------
    >>> dw = durbin_watson(residuals)
    >>> print(dw)
    Durbin-Watson: 1.95
      → No significant autocorrelation
    """
    residuals = np.asarray(residuals).flatten()
    residuals = residuals[~np.isnan(residuals)]
    
    diff = np.diff(residuals)
    dw = np.sum(diff ** 2) / np.sum(residuals ** 2)
    
    # Interpretation
    if dw < 1.5:
        conclusion = "Positive autocorrelation detected"
    elif dw > 2.5:
        conclusion = "Negative autocorrelation detected"
    else:
        conclusion = "No significant autocorrelation"
    
    return DiagnosticResult(
        name="Durbin-Watson",
        statistic=dw,
        conclusion=conclusion
    )


def breusch_pagan(
    residuals: np.ndarray,
    X: np.ndarray
) -> DiagnosticResult:
    """
    Breusch-Pagan test for heteroscedasticity.
    
    H0: Homoscedasticity (constant variance)
    H1: Heteroscedasticity (variance depends on X)
    
    Parameters
    ----------
    residuals : array-like
        OLS residuals.
    X : array-like
        Predictor matrix used in original regression.
        
    Returns
    -------
    DiagnosticResult
    
    Examples
    --------
    >>> bp = breusch_pagan(residuals, X)
    >>> if bp.p_value < 0.05:
    ...     print("Heteroscedasticity detected!")
    """
    residuals = np.asarray(residuals).flatten()
    X = np.asarray(X)
    
    n = len(residuals)
    
    # Step 1: Squared standardized residuals
    sigma2 = np.sum(residuals ** 2) / n
    u2 = residuals ** 2 / sigma2
    
    # Step 2: Regress u² on X
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, u2, rcond=None)
        u2_pred = X_aug @ beta
    except:
        return DiagnosticResult(
            name="Breusch-Pagan",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Computation failed"
        )
    
    # Step 3: LM statistic = n * R² (from auxiliary regression)
    ss_res = np.sum((u2 - u2_pred) ** 2)
    ss_tot = np.sum((u2 - np.mean(u2)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    k = X.shape[1]
    lm = n * r2
    p_value = 1 - stats.chi2.cdf(lm, k)
    
    conclusion = "Heteroscedasticity detected" if p_value < 0.05 else "Homoscedasticity (no heteroscedasticity)"
    
    return DiagnosticResult(
        name="Breusch-Pagan",
        statistic=lm,
        p_value=p_value,
        conclusion=conclusion
    )


def white_test(
    residuals: np.ndarray,
    X: np.ndarray
) -> DiagnosticResult:
    """
    White's test for heteroscedasticity.
    
    More general than Breusch-Pagan, includes squared terms and cross-products.
    
    Parameters
    ----------
    residuals : array-like
        OLS residuals.
    X : array-like
        Predictor matrix used in original regression.
        
    Returns
    -------
    DiagnosticResult
    """
    residuals = np.asarray(residuals).flatten()
    X = np.asarray(X)
    
    n, k = X.shape
    
    # Squared residuals
    e2 = residuals ** 2
    
    # Create terms: X, X², and cross-products
    terms = [np.ones(n)]  # constant
    
    # Original variables
    for j in range(k):
        terms.append(X[:, j])
    
    # Squared terms
    for j in range(k):
        terms.append(X[:, j] ** 2)
    
    # Cross-products (for small k)
    if k <= 10:
        for i in range(k):
            for j in range(i + 1, k):
                terms.append(X[:, i] * X[:, j])
    
    Z = np.column_stack(terms)
    
    # Regress e² on Z
    try:
        beta, _, _, _ = np.linalg.lstsq(Z, e2, rcond=None)
        e2_pred = Z @ beta
    except:
        return DiagnosticResult(
            name="White Test",
            statistic=np.nan,
            p_value=np.nan,
            conclusion="Computation failed"
        )
    
    # R² and LM statistic
    ss_res = np.sum((e2 - e2_pred) ** 2)
    ss_tot = np.sum((e2 - np.mean(e2)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    lm = n * r2
    df = Z.shape[1] - 1  # exclude constant
    p_value = 1 - stats.chi2.cdf(lm, df)
    
    conclusion = "Heteroscedasticity detected" if p_value < 0.05 else "No heteroscedasticity detected"
    
    return DiagnosticResult(
        name="White Test",
        statistic=lm,
        p_value=p_value,
        conclusion=conclusion
    )


def cooks_distance(
    X: np.ndarray,
    y: np.ndarray,
    residuals: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Cook's Distance for each observation.
    
    Measures influence of each observation on the regression.
    Values > 4/n are typically considered influential.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
    y : array-like of shape (n_samples,)
        Response variable.
    residuals : array-like, optional
        Pre-computed residuals. If None, computed internally.
        
    Returns
    -------
    cooks_d : ndarray
        Cook's distance for each observation.
    threshold : float
        Recommended threshold (4/n).
        
    Examples
    --------
    >>> d, threshold = cooks_distance(X, y)
    >>> influential = d > threshold
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n, k = X.shape
    
    # Add constant if not present
    if not np.allclose(X[:, 0], 1):
        X = np.column_stack([np.ones(n), X])
        k += 1
    
    # Hat matrix: H = X @ (X'X)^-1 @ X'
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except:
        XtX_inv = np.linalg.pinv(X.T @ X)
    
    H = X @ XtX_inv @ X.T
    h = np.diag(H)  # leverage values
    
    # Residuals
    if residuals is None:
        beta = XtX_inv @ X.T @ y
        y_pred = X @ beta
        residuals = y - y_pred
    
    # MSE
    mse = np.sum(residuals ** 2) / (n - k)
    
    # Cook's Distance
    # D_i = (e_i² / (k * MSE)) * (h_i / (1 - h_i)²)
    cooks_d = (residuals ** 2 / (k * mse)) * (h / (1 - h) ** 2)
    
    threshold = 4 / n
    
    return cooks_d, threshold


def leverage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate leverage (hat values) for each observation.
    
    High leverage points have unusual X values.
    Values > 2(k+1)/n are typically considered high leverage.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
        
    Returns
    -------
    h : ndarray
        Leverage (hat) values.
    threshold : float
        Recommended threshold 2(k+1)/n.
        
    Examples
    --------
    >>> h, threshold = leverage(X)
    >>> high_leverage = h > threshold
    """
    X = np.asarray(X)
    n, k = X.shape
    
    # Add constant if not present
    if not np.allclose(X[:, 0], 1):
        X = np.column_stack([np.ones(n), X])
        k += 1
    
    # Hat matrix diagonal
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except:
        XtX_inv = np.linalg.pinv(X.T @ X)
    
    H = X @ XtX_inv @ X.T
    h = np.diag(H)
    
    threshold = 2 * k / n
    
    return h, threshold


class RegressionDiagnostics:
    """
    Comprehensive regression diagnostics.
    
    Computes all diagnostics in one pass for efficiency.
    
    Examples
    --------
    >>> diag = RegressionDiagnostics(X, y, residuals)
    >>> print(diag.summary())
    >>> diag.plot_diagnostics()  # if matplotlib available
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residuals: Optional[np.ndarray] = None
    ):
        """
        Initialize diagnostics.
        
        Parameters
        ----------
        X : array-like
            Predictor matrix.
        y : array-like
            Response variable.
        residuals : array-like, optional
            Pre-computed residuals.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y).flatten()
        
        n = len(self.y)
        
        # Add constant if needed
        if self.X.shape[1] > 0 and not np.allclose(self.X[:, 0], 1):
            self.X_aug = np.column_stack([np.ones(n), self.X])
        else:
            self.X_aug = self.X
        
        # Compute residuals if not provided
        if residuals is not None:
            self.residuals = np.asarray(residuals).flatten()
        else:
            try:
                beta = np.linalg.lstsq(self.X_aug, self.y, rcond=None)[0]
                self.residuals = self.y - self.X_aug @ beta
            except:
                self.residuals = np.zeros(n)
        
        # Pre-compute
        self._vif = None
        self._dw = None
        self._bp = None
        self._white = None
        self._cooks = None
        self._leverage = None
    
    @property
    def vif_values(self) -> pd.DataFrame:
        """VIF for each predictor."""
        if self._vif is None:
            self._vif = vif(self.X)
        return self._vif
    
    @property
    def durbin_watson_stat(self) -> DiagnosticResult:
        """Durbin-Watson statistic."""
        if self._dw is None:
            self._dw = durbin_watson(self.residuals)
        return self._dw
    
    @property
    def breusch_pagan_stat(self) -> DiagnosticResult:
        """Breusch-Pagan test."""
        if self._bp is None:
            self._bp = breusch_pagan(self.residuals, self.X)
        return self._bp
    
    @property
    def white_stat(self) -> DiagnosticResult:
        """White test."""
        if self._white is None:
            self._white = white_test(self.residuals, self.X)
        return self._white
    
    @property
    def cooks_d(self) -> Tuple[np.ndarray, float]:
        """Cook's distance."""
        if self._cooks is None:
            self._cooks = cooks_distance(self.X, self.y, self.residuals)
        return self._cooks
    
    @property
    def leverage_values(self) -> Tuple[np.ndarray, float]:
        """Leverage (hat) values."""
        if self._leverage is None:
            self._leverage = leverage(self.X)
        return self._leverage
    
    def summary(self) -> pd.DataFrame:
        """
        Return summary of all diagnostics.
        """
        results = []
        
        # VIF summary
        vif_max = self.vif_values['VIF'].max()
        vif_issue = vif_max > 5
        results.append({
            'Diagnostic': 'VIF (max)',
            'Statistic': vif_max,
            'Threshold': 5.0,
            'Issue': 'Yes' if vif_issue else 'No'
        })
        
        # Durbin-Watson
        dw = self.durbin_watson_stat
        dw_issue = dw.statistic < 1.5 or dw.statistic > 2.5
        results.append({
            'Diagnostic': 'Durbin-Watson',
            'Statistic': dw.statistic,
            'Threshold': '1.5-2.5',
            'Issue': 'Yes' if dw_issue else 'No'
        })
        
        # Breusch-Pagan
        bp = self.breusch_pagan_stat
        bp_issue = bp.p_value is not None and bp.p_value < 0.05
        results.append({
            'Diagnostic': 'Breusch-Pagan',
            'Statistic': bp.statistic,
            'Threshold': f'p>{0.05}',
            'Issue': 'Yes' if bp_issue else 'No'
        })
        
        # White test
        wt = self.white_stat
        wt_issue = wt.p_value is not None and wt.p_value < 0.05
        results.append({
            'Diagnostic': 'White Test',
            'Statistic': wt.statistic,
            'Threshold': f'p>{0.05}',
            'Issue': 'Yes' if wt_issue else 'No'
        })
        
        # Cook's distance
        cooks, cooks_thresh = self.cooks_d
        n_influential = np.sum(cooks > cooks_thresh)
        results.append({
            'Diagnostic': "Cook's D (# influential)",
            'Statistic': n_influential,
            'Threshold': f'>{cooks_thresh:.4f}',
            'Issue': 'Yes' if n_influential > 0 else 'No'
        })
        
        # Leverage
        lev, lev_thresh = self.leverage_values
        n_high_lev = np.sum(lev > lev_thresh)
        results.append({
            'Diagnostic': 'High Leverage (#)',
            'Statistic': n_high_lev,
            'Threshold': f'>{lev_thresh:.4f}',
            'Issue': 'Yes' if n_high_lev > 0 else 'No'
        })
        
        return pd.DataFrame(results)
    
    def influential_observations(self) -> pd.DataFrame:
        """
        Identify influential observations.
        
        Returns DataFrame with indices of problematic observations.
        """
        cooks, cooks_thresh = self.cooks_d
        lev, lev_thresh = self.leverage_values
        
        influential_cooks = np.where(cooks > cooks_thresh)[0]
        influential_lev = np.where(lev > lev_thresh)[0]
        
        # Standardized residuals
        std_res = self.residuals / np.std(self.residuals)
        outliers = np.where(np.abs(std_res) > 2)[0]
        
        all_influential = set(influential_cooks) | set(influential_lev) | set(outliers)
        
        if not all_influential:
            return pd.DataFrame(columns=['Index', 'Cooks_D', 'Leverage', 'Std_Residual', 'Issues'])
        
        rows = []
        for idx in sorted(all_influential):
            issues = []
            if idx in influential_cooks:
                issues.append('High Cook\'s D')
            if idx in influential_lev:
                issues.append('High Leverage')
            if idx in outliers:
                issues.append('Outlier')
            
            rows.append({
                'Index': idx,
                'Cooks_D': cooks[idx],
                'Leverage': lev[idx],
                'Std_Residual': std_res[idx],
                'Issues': ', '.join(issues)
            })
        
        return pd.DataFrame(rows)
