"""
Statelix Discrete Choice Models

Provides:
  - OrderedModel: Ordered Logit/Probit for ordinal data (e.g., surveys).
  - MultinomialLogit: For nominal categorical data.
  
Wraps statsmodels implementations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Literal

try:
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel as SMOrderedModel
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class DiscreteResult:
    """Result from Discrete Choice Model."""
    coef: pd.Series
    std_errors: pd.Series
    p_values: pd.Series
    aic: float
    bic: float
    pseudo_r2: float
    log_likelihood: float
    conf_int: pd.DataFrame
    
    def summary(self) -> pd.DataFrame:
        """Return summary dataframe."""
        df = pd.DataFrame({
            'Coef': self.coef,
            'StdErr': self.std_errors,
            'P-Value': self.p_values,
            'Lower 95%': self.conf_int.iloc[:, 0],
            'Upper 95%': self.conf_int.iloc[:, 1]
        })
        return df


class OrderedModel:
    """
    Ordered Logit/Probit Model.
    
    Used for ordinal dependent variables (e.g., Likert scales 1-5).
    
    Parameters
    ----------
    dist : {'logit', 'probit'}, default='logit'
        The distribution for the latent variable.
    """
    
    def __init__(self, dist: Literal['logit', 'probit'] = 'logit'):
        self.dist = dist
        self.result_: Optional[DiscreteResult] = None
        self.model_ = None
        self.sm_result_ = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        method: str = 'bfgs'
    ) -> 'OrderedModel':
        """
        Fit Ordered Model.
        
        Parameters
        ----------
        X : array-like
            Predictors.
        y : array-like
            Ordinal target variable.
        method : str
            Optimization method.
            
        Returns
        -------
        self
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for OrderedModel.")
            
        # Ensure pandas for better handling
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            
        # Statsmodels OrderedModel expects endog (y) then exog (X)
        try:
            self.model_ = SMOrderedModel(y, X, distr=self.dist)
            self.sm_result_ = self.model_.fit(method=method, disp=False)
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {str(e)}")
            
        self.result_ = DiscreteResult(
            coef=self.sm_result_.params,
            std_errors=self.sm_result_.bse,
            p_values=self.sm_result_.pvalues,
            aic=self.sm_result_.aic,
            bic=self.sm_result_.bic,
            pseudo_r2=self.sm_result_.prsquared,
            log_likelihood=self.sm_result_.llf,
            conf_int=self.sm_result_.conf_int()
        )
        
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Predict probabilities for each category.
        
        Returns
        -------
        DataFrame of probabilities (rows=samples, cols=categories)
        """
        if self.sm_result_ is None:
            raise RuntimeError("Model not fitted")
        
        return self.sm_result_.predict(X)


class MultinomialLogit:
    """
    Multinomial Logit Model (MNLogit).
    
    Used for nominal categorical dependent variables (e.g., Brand A vs B vs C).
    """
    
    def __init__(self):
        self.result_: Optional[DiscreteResult] = None
        self.model_ = None
        self.sm_result_ = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        method: str = 'bfgs'
    ) -> 'MultinomialLogit':
        """
        Fit Multinomial Logit.
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for MultinomialLogit.")
        
        # Add constant to X usually required for MNLogit
        if isinstance(X, pd.DataFrame):
            X_aug = sm.add_constant(X)
        else:
            X_aug = sm.add_constant(pd.DataFrame(X))
            
        try:
            self.model_ = sm.MNLogit(y, X_aug)
            self.sm_result_ = self.model_.fit(method=method, disp=False)
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {str(e)}")
            
        self.result_ = DiscreteResult(
            coef=self.sm_result_.params, # Matrix for MNLogit
            std_errors=self.sm_result_.bse,
            p_values=self.sm_result_.pvalues,
            aic=self.sm_result_.aic,
            bic=self.sm_result_.bic,
            pseudo_r2=self.sm_result_.prsquared,
            log_likelihood=self.sm_result_.llf,
            conf_int=self.sm_result_.conf_int()
        )
        
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """Predict class probabilities."""
        if self.sm_result_ is None:
            raise RuntimeError("Model not fitted")
            
        if isinstance(X, pd.DataFrame):
            X_aug = sm.add_constant(X, has_constant='add')
        else:
            X_aug = sm.add_constant(pd.DataFrame(X), has_constant='add')
            
        return self.sm_result_.predict(X_aug)

