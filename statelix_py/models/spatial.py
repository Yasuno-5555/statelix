"""
Statelix Spatial Econometrics Module

Provides:
  - SpatialRegression: Wrapper for C++ backend implementation of SAR/SEM/SDM.
  - SpatialWeights: Utilities for creating weight matrices (KNN, Distance).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import statelix.spatial as _cpp_spatial
    _HAS_CPP_CORE = True
except ImportError:
    try:
        import statelix_core
        _cpp_spatial = statelix_core.spatial
        _HAS_CPP_CORE = True
    except (ImportError, AttributeError):
        _HAS_CPP_CORE = False

class SpatialModelType(Enum):
    SAR = "SAR"
    SEM = "SEM"
    SDM = "SDM"

@dataclass
class SpatialResultWrapper:
    """Result from Spatial Regression."""
    coef: pd.Series
    std_errors: pd.Series
    p_values: pd.Series
    rho: float
    lambda_: float
    model: str
    log_likelihood: float
    aic: float
    r2: float
    direct_effects: Optional[pd.Series] = None
    indirect_effects: Optional[pd.Series] = None
    total_effects: Optional[pd.Series] = None
    
    def summary(self) -> pd.DataFrame:
        """Return summary dataframe."""
        df = pd.DataFrame({
            'Coef': self.coef,
            'StdErr': self.std_errors,
            'P-Value': self.p_values
        })
        return df

class SpatialRegression:
    """
    Spatial Regression Model (SAR/SEM/SDM).
    
    Parameters
    ----------
    model : {'SAR', 'SEM', 'SDM'}, default='SAR'
        Model type.
    max_iter : int, default=100
        Maximum iterations for optimization.
    tol : float, default=1e-6
        Convergence tolerance.
    """
    
    def __init__(
        self,
        model: str = 'SAR',
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        self.model = model
        self.max_iter = max_iter
        self.tol = tol
        self.result_: Optional[SpatialResultWrapper] = None
        
    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        W: Union[np.ndarray, pd.DataFrame]
    ) -> 'SpatialRegression':
        """
        Fit spatial model.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        X : array-like of shape (n_samples, n_features)
            Training data.
        W : array-like of shape (n_samples, n_samples)
            Spatial weights matrix. Should be row-standardized.
            
        Returns
        -------
        self
        """
        if not _HAS_CPP_CORE:
            raise RuntimeError("statelix_core C++ module not found.")
            
        # Convert inputs
        y_arr = np.asarray(y, dtype=np.float64)
        X_arr = np.asarray(X, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        
        # Helper for feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
            
        if self.model == 'SDM':
            # Add spatial lag names
            feature_names = feature_names + [f"W_{name}" for name in feature_names]
            
        # Add intercept name if implicit (Note: C++ currently assumes X includes intercept if desired, 
        # or handles it. The current impl assumes X contains all regressors.)
        # Usually user passes X with constant.
        
        # Setup C++ model
        model_enum = getattr(_cpp_spatial.SpatialModel, self.model)
        sr_cpp = _cpp_spatial.SpatialRegression()
        sr_cpp.model = model_enum
        sr_cpp.max_iter = self.max_iter
        
        # Fit
        try:
            res_cpp = sr_cpp.fit(y_arr, X_arr, W_arr)
        except Exception as e:
            raise RuntimeError(f"Spatial optimization failed: {str(e)}")
            
        # Wrap results
        # Coef names: Rho/Lambda first, then Beta
        coef_names = []
        if self.model in ['SAR', 'SDM']:
            coef_names.append('rho')
        elif self.model == 'SEM':
            coef_names.append('lambda')
            
        coef_names.extend(feature_names)
        
        # Effects
        direct = None
        indirect = None
        total = None
        
        if self.model in ['SAR', 'SDM']:
            # Effects correspond to X columns (k)
            # direct/indirect/total in C++ are size k
            x_names = feature_names[:X_arr.shape[1]] # Original X features
            direct = pd.Series(res_cpp.direct_effects, index=x_names)
            indirect = pd.Series(res_cpp.indirect_effects, index=x_names)
            total = pd.Series(res_cpp.total_effects, index=x_names)
            
        self.result_ = SpatialResultWrapper(
            coef=pd.Series(res_cpp.coef, index=coef_names),
            std_errors=pd.Series(res_cpp.std_errors, index=coef_names),
            p_values=pd.Series(res_cpp.p_values, index=coef_names),
            rho=res_cpp.rho,
            lambda_=getattr(res_cpp, 'lambda'),
            model=self.model,
            log_likelihood=res_cpp.log_likelihood,
            aic=res_cpp.aic,
            r2=res_cpp.pseudo_r_squared,
            direct_effects=direct,
            indirect_effects=indirect,
            total_effects=total
        )
        
        return self

class SpatialWeights:
    """Utilities for spatial weights."""
    
    @staticmethod
    def knn(coords: np.ndarray, k: int = 5) -> np.ndarray:
        """Create K-Nearest Neighbors weights (row-standardized)."""
        if not _HAS_CPP_CORE:
            raise RuntimeError("statelix_core required")
        # Ensure array
        coords_arr = np.asarray(coords, dtype=np.float64)
        return _cpp_spatial.SpatialWeights.knn_weights(coords_arr, k)
        
    @staticmethod
    def distance(coords: np.ndarray, bandwidth: float = -1) -> np.ndarray:
        """Create Inverse Distance weights (row-standardized)."""
        if not _HAS_CPP_CORE:
            raise RuntimeError("statelix_core required")
        coords_arr = np.asarray(coords, dtype=np.float64)
        return _cpp_spatial.SpatialWeights.inverse_distance_weights(coords_arr, bandwidth)
