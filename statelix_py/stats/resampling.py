"""
Statelix Resampling Module

Provides robust bootstrap, block bootstrap, and jackknife methods backed by C++.
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Union, List, Tuple
from dataclasses import dataclass

try:
    import statelix_core
    _cpp_core = statelix_core
    _HAS_CPP = True
except ImportError:
    try:
        from ..core import statelix_core
        _cpp_core = statelix_core
        _HAS_CPP = True
    except ImportError:
        _HAS_CPP = False

@dataclass
class ResamplingResult:
    """Container for resampling results."""
    original_stat: np.ndarray
    resampled_stats: np.ndarray
    std_error: np.ndarray
    bias: np.ndarray
    conf_int: Optional[np.ndarray] = None
    
    def summary(self) -> pd.DataFrame:
        """Return summary of estimates."""
        # Check if 1D or multi-dim
        n_params = self.original_stat.size
        
        data = {
            'Estimate': self.original_stat.flatten(),
            'StdError': self.std_error.flatten(),
            'Bias': self.bias.flatten()
        }
        
        if self.conf_int is not None:
            data['CI_Lower'] = self.conf_int[:, 0].flatten()
            data['CI_Upper'] = self.conf_int[:, 1].flatten()
            
        return pd.DataFrame(data)

class Resampler:
    """
    Resampling engine backed by C++ statelix_core.
    """
    
    def __init__(self, seed: int = 42, parallel: bool = True, n_jobs: int = -1):
        if not _HAS_CPP:
            raise RuntimeError("statelix_core required for resampling.")
        
        self.cpp_resampler = _cpp_core.Resampler(seed, parallel, n_jobs)
        
    def bootstrap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        statistic: Callable[[np.ndarray], np.ndarray],
        n_reps: int = 1000,
        ci_alpha: float = 0.05
    ) -> ResamplingResult:
        """
        Perform I.I.D. Bootstrap.
        
        Parameters
        ----------
        data : array-like
            Input data (rows are samples).
        statistic : function
            Function mapping data -> statistic (1D array).
        n_reps : int
            Number of bootstrap replications.
        ci_alpha : float
            Alpha level for confidence intervals (e.g. 0.05 for 95%).
            
        Returns
        -------
        ResamplingResult
        """
        data_arr = np.asarray(data, dtype=np.float64)
        
        # Original statistic
        orig_stat = np.asarray(statistic(data_arr), dtype=np.float64)
        
        # Wrapper for C++ callback
        # C++ passes Eigen matrix, pybind11 converts to numpy array
        # Return must be numpy array (vector) for Eigen conversion
        def func_wrapper(x):
            return np.asarray(statistic(x), dtype=np.float64)
            
        boot_dist = self.cpp_resampler.bootstrap(data_arr, func_wrapper, n_reps)
        
        # Compute metrics
        mean_stat = np.mean(boot_dist, axis=0)
        bias = mean_stat - orig_stat
        std_err = np.std(boot_dist, axis=0, ddof=1)
        
        # Basic Percentile CI
        # alpha/2 and 1-alpha/2 quantiles
        lower = np.percentile(boot_dist, 100 * ci_alpha / 2, axis=0)
        upper = np.percentile(boot_dist, 100 * (1 - ci_alpha / 2), axis=0)
        conf_int = np.column_stack([lower, upper])
        
        return ResamplingResult(
            original_stat=orig_stat,
            resampled_stats=boot_dist,
            std_error=std_err,
            bias=bias,
            conf_int=conf_int
        )
        
    def block_bootstrap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        statistic: Callable[[np.ndarray], np.ndarray],
        block_size: int,
        n_reps: int = 1000,
        ci_alpha: float = 0.05
    ) -> ResamplingResult:
        """
        Perform Moving Block Bootstrap for Time Series.
        Preserves local dependency structure.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        orig_stat = np.asarray(statistic(data_arr), dtype=np.float64)
        
        def func_wrapper(x):
            return np.asarray(statistic(x), dtype=np.float64)
            
        boot_dist = self.cpp_resampler.block_bootstrap(data_arr, func_wrapper, block_size, n_reps)
        
        mean_stat = np.mean(boot_dist, axis=0)
        bias = mean_stat - orig_stat
        std_err = np.std(boot_dist, axis=0, ddof=1)
        
        lower = np.percentile(boot_dist, 100 * ci_alpha / 2, axis=0)
        upper = np.percentile(boot_dist, 100 * (1 - ci_alpha / 2), axis=0)
        conf_int = np.column_stack([lower, upper])
        
        return ResamplingResult(
            original_stat=orig_stat,
            resampled_stats=boot_dist,
            std_error=std_err,
            bias=bias,
            conf_int=conf_int
        )

    def jackknife(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        statistic: Callable[[np.ndarray], np.ndarray]
    ) -> ResamplingResult:
        """
        Perform Delete-1 Jackknife.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        orig_stat = np.asarray(statistic(data_arr), dtype=np.float64)
        
        def func_wrapper(x):
            return np.asarray(statistic(x), dtype=np.float64)
            
        jack_dist = self.cpp_resampler.jackknife(data_arr, func_wrapper)
        
        n = data_arr.shape[0]
        mean_jack = np.mean(jack_dist, axis=0)
        
        # Jackknife bias: (n-1)*(mean_jack - orig)
        bias = (n - 1) * (mean_jack - orig_stat)
        
        # Jackknife SE
        diff = jack_dist - mean_jack
        var_jack = np.sum(diff**2, axis=0) * (n - 1) / n
        std_err = np.sqrt(var_jack)
        
        return ResamplingResult(
            original_stat=orig_stat,
            resampled_stats=jack_dist,
            std_error=std_err,
            bias=bias,
            conf_int=None # Jackknife doesn't give CI directly
        )

# Functional API
def bootstrap(data, statistic, n_reps=1000, seed=42):
    rs = Resampler(seed=seed)
    return rs.bootstrap(data, statistic, n_reps)

def block_bootstrap(data, statistic, block_size, n_reps=1000, seed=42):
    rs = Resampler(seed=seed)
    return rs.block_bootstrap(data, statistic, block_size, n_reps)
