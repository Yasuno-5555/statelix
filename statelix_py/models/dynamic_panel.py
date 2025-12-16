
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Union

try:
    from statelix_py.core import statelix_core
    _HAS_CPP = True
except ImportError:
    import statelix_core # fallback or error
    _HAS_CPP = True

@dataclass
class DynamicPanelResult:
    coefficients: np.ndarray
    std_errors: np.ndarray
    sargan_test: float
    n_obs: int

    @property
    def summary(self):
        df = pd.DataFrame({
            'Coef': self.coefficients,
            'StdErr': self.std_errors,
            't': self.coefficients / (self.std_errors + 1e-10)
        })
        return df

class ArellanoBond:
    """
    Arellano-Bond (Difference GMM) Estimator for Dynamic Panel Data.
    
    Model:
        y_{it} = rho * y_{i,t-1} + X_{it} * beta + alpha_i + e_{it}
    
    Estimation uses First Differences to remove alpha_i, and lagged levels as instruments.
    """
    def __init__(self, two_step: bool = True):
        self.two_step = two_step
        self.result_: Optional[DynamicPanelResult] = None
        
    def fit(self, data: pd.DataFrame, y_col: str, x_cols: List[str], id_col: str, time_col: str):
        """
        Fit the model using C++ backend.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Long-format panel data.
        y_col : str
            Dependent variable name.
        x_cols : List[str]
            Exogenous regressor names.
        id_col : str
            Individual identifier column name.
        time_col : str
            Time period column name.
        """
        if not _HAS_CPP:
             raise RuntimeError("Statelix Core C++ module is required for GMM.")
             
        # Extract arrays
        # Ensure data is numeric
        y = data[y_col].values.astype(float)
        X = data[x_cols].values.astype(float)
        
        # ID and Time must be integer encoded for C++
        # We'll use factorize to map to 0..N-1 if not already
        ids, unique_ids = pd.factorize(data[id_col])
        times, unique_times = pd.factorize(data[time_col], sort=True)
        # Verify times are integers reflecting order? 
        # C++ assumes integer time periods for lag calculation (t-1).
        # We should ideally pass real integer times if they exist, or mapped indices.
        # If 'times' are just indices 0..T-1, it works.
        
        # Instantiate C++ class
        gmm = statelix_core.panel.DynamicPanelGMM()
        gmm.two_step = self.two_step
        
        # Estimate
        res_cpp = gmm.estimate(y, X, ids.astype(np.int32), times.astype(np.int32))
        
        self.result_ = DynamicPanelResult(
            coefficients=np.array(res_cpp.coefficients),
            std_errors=np.array(res_cpp.std_errors),
            sargan_test=res_cpp.sargan_test,
            n_obs=res_cpp.n_obs
        )
        
        return self

    @property
    def coef_(self):
        return self.result_.coefficients if self.result_ else None
        
