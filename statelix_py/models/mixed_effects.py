"""
Statelix Mixed Effects Models

Provides:
  - LinearMixedModel: LMM via Restricted Maximum Likelihood (REML)
  
Wrapper around statsmodels.mixedlm if available, with valid fallback (or error).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Any

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class MixedEffectResult:
    """Result from Linear Mixed Model."""
    coef: pd.Series
    std_errors: pd.Series
    z_values: pd.Series
    p_values: pd.Series
    random_effects: Dict[Any, pd.Series]
    fitted_values: np.ndarray
    residuals: np.ndarray
    aic: float
    bic: float
    converged: bool
    
    def summary(self) -> pd.DataFrame:
        """Return basic coefficient summary."""
        return pd.DataFrame({
            'Coef': self.coef,
            'StdErr': self.std_errors,
            'z': self.z_values,
            'P>|z|': self.p_values
        })


class LinearMixedModel:
    """
    Linear Mixed Effects Model.
    
    Models: y = Xβ + Zγ + ε
    where γ ~ N(0, G) and ε ~ N(0, R)
    
    Parameters
    ----------
    formula : str
        Formula string (e.g., 'y ~ x1 + x2').
    groups : str
        Name of grouping variable.
    re_formula : str, optional
        Formula for random effects (e.g., '1 + x1' for random slope).
        Default is '1' (random intercept only).
    """
    
    def __init__(
        self,
        formula: Optional[str] = None,
        groups: Optional[str] = None,
        re_formula: str = "1"
    ):
        self.formula = formula
        self.groups = groups
        self.re_formula = re_formula
        self.result_: Optional[MixedEffectResult] = None
        self.model_ = None
        self.sm_result_ = None
        
    def fit(
        self,
        data: pd.DataFrame,
        formula: Optional[str] = None,
        groups: Optional[str] = None,
        re_formula: Optional[str] = None,
        method: str = "lbfgs",
        reml: bool = True
    ) -> 'LinearMixedModel':
        """
        Fit Linear Mixed Model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset containing all variables.
        formula : str, optional
            Override init formula.
        groups : str, optional
            Override init groups.
        re_formula : str, optional
            Override init re_formula.
        method : str
            Optimization method ('lbfgs', 'powell', 'cg', etc.).
        reml : bool
            If True, use REML. If False, use ML.
            
        Returns
        -------
        self
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for LinearMixedModel.")
            
        formula = formula or self.formula
        groups = groups or self.groups
        re_formula = re_formula or self.re_formula
        
        if formula is None or groups is None:
            raise ValueError("Formula and Groups must be specified.")
            
        # Create model using statsmodels formula interface
        self.model_ = smf.mixedlm(
            formula,
            data,
            groups=data[groups],
            re_formula=re_formula
        )
        
        # Fit
        # full_output=True creates a RegressionResultsWrapper
        self.sm_result_ = self.model_.fit(
            method=method, 
            reml=reml,
            full_output=True
        )
        
        # Extract results
        self.result_ = MixedEffectResult(
            coef=self.sm_result_.params,
            std_errors=self.sm_result_.bse,
            z_values=self.sm_result_.tvalues,  # SM calls them tvalues but they are z-scores asymptotically
            p_values=self.sm_result_.pvalues,
            random_effects={k: v for k, v in self.sm_result_.random_effects.items()},
            fitted_values=self.sm_result_.fittedvalues,
            residuals=self.sm_result_.resid,
            aic=self.sm_result_.aic,
            bic=self.sm_result_.bic,
            converged=self.sm_result_.converged
        )
        
        return self
        
    def predict(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict new values.
        
        Note: Prediction with new groups often uses only fixed effects
        unless group mapping is provided. Statsmodels behavior applies.
        """
        if self.sm_result_ is None:
            raise RuntimeError("Model not fitted")
            
        return self.sm_result_.predict(exog=data)

