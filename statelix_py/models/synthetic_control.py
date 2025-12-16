"""
Statelix Synthetic Control Module

Provides:
  - SyntheticControl: Wrapper for C++ backend implementation of Abadie et al. (2010)
  - Visualization tools for Synthetic Control analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple
from dataclasses import dataclass

try:
    import statelix_core
    _cpp_causal = statelix_core.causal
    _HAS_CPP_CORE = True
except ImportError:
    try:
        from ..core import statelix_core
        _cpp_causal = statelix_core.causal
        _HAS_CPP_CORE = True
    except ImportError:
        _HAS_CPP_CORE = False

@dataclass
class SyntheticControlResult:
    """Result from Synthetic Control estimation."""
    weights: np.ndarray
    gaps: np.ndarray
    y_synthetic: np.ndarray
    att: float
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    pre_treatment_fit: float
    selected_donors: List[int]
    predictor_balance: np.ndarray
    
    # Metadata
    treated_idx: int
    donor_indices: List[int]
    treatment_period: int
    unit_names: Optional[List[str]] = None
    
    def summary(self) -> pd.DataFrame:
        """Return summary metrics."""
        return pd.DataFrame({
            'Metric': ['ATT', 'Pre-RMSPE', 'Post-RMSPE', 'RMSPE Ratio', 'Pre-treatment R2'],
            'Value': [self.att, self.pre_rmspe, self.post_rmspe, self.rmspe_ratio, self.pre_treatment_fit]
        })
    
    def donor_weights(self) -> pd.DataFrame:
        """Return weights of selected donors."""
        if self.unit_names:
            names = [self.unit_names[i] for i in self.donor_indices]
        else:
            names = [f"Donor_{i}" for i in self.donor_indices]
            
        df = pd.DataFrame({
            'Unit': names,
            'Weight': self.weights
        })
        return df[df['Weight'] > 1e-4].sort_values('Weight', ascending=False)


class SyntheticControl:
    """
    Synthetic Control Method Estimator.
    
    Parameters
    ----------
    max_iter : int, default=1000
        Maximum iterations for weight optimization
    tol : float, default=1e-8
        Convergence tolerance
    normalize : bool, default=True
        Whether to normalize variables before optimization
    v_penalty : float, default=0.0
        Ridge penalty for V matrix optimization
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        normalize: bool = True,
        v_penalty: float = 0.0
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.v_penalty = v_penalty
        
        self.result_: Optional[SyntheticControlResult] = None
        self.placebo_result_ = None
        
    def fit(
        self,
        data: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treated_unit: Union[str, int],
        treatment_period: int,
        predictors: Optional[List[str]] = None
    ) -> 'SyntheticControl':
        """
        Fit Synthetic Control model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Long-format data containing unit, time, and outcome columns
        unit_col : str
            Name of unit column
        time_col : str
            Name of time column
        outcome_col : str
            Name of outcome column
        treated_unit : str or int
            ID of the treated unit
        treatment_period : int
            Time period first treated (must match values in time_col)
        predictors : list, optional
            List of predictor columns. If None, uses pre-treatment outcomes.
            
        Returns
        -------
        self
        """
        if not _HAS_CPP_CORE:
            raise RuntimeError("statelix_core C++ module not found. Synthetic Control requires compiled backend.")
            
        # 1. Pivot data to wide format (Time x Units)
        df_pivot = data.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()
        
        if treated_unit not in df_pivot.columns:
            raise ValueError(f"Treated unit {treated_unit} not found in data")
            
        # 2. Get matrix Y
        units = list(df_pivot.columns)
        treated_idx = units.index(treated_unit)
        
        Y = np.ascontiguousarray(df_pivot.values, dtype=np.float64)
        
        # 3. Handle predictors if provided (Time x Units for now, or aggregated)
        # Note: current C++ implementation expects X as (K, N)
        # If predictors are time-varying, we usually take pre-treatment means
        X = np.array([], dtype=np.float64).reshape(0, Y.shape[1])
        
        if predictors:
            # Aggregate predictors for pre-treatment period
            pre_mask = data[time_col] < treatment_period
            df_pre = data[pre_mask]
            
            X_list = []
            for unit in units:
                unit_data = df_pre[df_pre[unit_col] == unit]
                if unit_data.empty:
                    raise ValueError(f"No pre-treatment data for unit {unit}")
                # Mean of predictors
                X_list.append(unit_data[predictors].mean().values)
            
            X = np.column_stack(X_list) # (K, N)
            
        # 4. Determine integer treatment index (0-based)
        times = df_pivot.index.tolist()
        if treatment_period not in times:
            raise ValueError(f"Treatment period {treatment_period} not found in time index")
        
        t_idx = times.index(treatment_period)
        
        # 5. Call C++ backend
        sc_cpp = _cpp_causal.SyntheticControl()
        sc_cpp.max_iter = self.max_iter
        sc_cpp.tol = self.tol
        sc_cpp.v_penalty = self.v_penalty
        sc_cpp.normalize = self.normalize
        
        # Determine strict donor indices (all except treated)
        donor_indices = [i for i in range(len(units)) if i != treated_idx]
        
        try:
            res_cpp = sc_cpp.fit(Y, X, treated_idx, t_idx)
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")
            
        # 6. Wrap result
        self.result_ = SyntheticControlResult(
            weights=res_cpp.weights,
            gaps=res_cpp.gaps,
            y_synthetic=res_cpp.y_synthetic,
            att=res_cpp.att,
            pre_rmspe=res_cpp.pre_rmspe,
            post_rmspe=res_cpp.post_rmspe,
            rmspe_ratio=res_cpp.rmspe_ratio,
            pre_treatment_fit=res_cpp.pre_treatment_fit,
            selected_donors=res_cpp.selected_donors,
            predictor_balance=res_cpp.predictor_balance,
            # Metadata
            treated_idx=treated_idx,
            donor_indices=donor_indices,
            treatment_period=t_idx,
            unit_names=units
        )
        
        self.data_context_ = {
            'times': times,
            'units': units,
            'Y': Y,
            'X': X,
            'treated_unit': treated_unit
        }
        
        return self

    def placebo_test(self, n_samples: int = None) -> pd.DataFrame:
        """
        Run in-space placebo test (permutation inference).
        
        Returns DataFrame with placebo gaps and RMSPE ratios.
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() first")
            
        ctx = self.data_context_
        sc_cpp = _cpp_causal.SyntheticControl()
        sc_cpp.max_iter = self.max_iter
        
        # Run placebo test in C++
        # Note: robust C++ implementation handles failures
        placebo_res = sc_cpp.placebo_test(
            ctx['Y'], 
            self.result_.treated_idx, 
            self.result_.treatment_period
        )
        
        self.placebo_result_ = placebo_res
        
        return pd.DataFrame({
            'p_value': [placebo_res.p_value],
            'treated_rank': [placebo_res.treated_rank],
            'n_placebos': [len(placebo_res.rmspe_ratios)]
        })

    def plot(self, show_placebo: bool = True):
        """
        Plot Synthetic Control results using Matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plotting")
            
        if self.result_ is None:
            raise RuntimeError("Call fit() first")
            
        ctx = self.data_context_
        times = ctx['times']
        treated_name = ctx['treated_unit']
        
        # Plot 1: Trends
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Actual
        ax1.plot(times, ctx['Y'][:, self.result_.treated_idx], 'k-', lw=2, label=f'Actual ({treated_name})')
        # Synthetic
        ax1.plot(times, self.result_.y_synthetic, 'r--', lw=2, label='Synthetic Control')
        
        # Treatment line
        t_start = times[self.result_.treatment_period]
        ax1.axvline(x=t_start, color='gray', linestyle=':', alpha=0.6)
        
        ax1.set_title("Synthetic Control: Outcomes")
        ax1.set_ylabel("Outcome")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gaps
        if show_placebo and self.placebo_result_:
            # Plot background placebos
            for gap in self.placebo_result_.placebo_gaps:
                ax2.plot(times, gap, color='gray', alpha=0.2, lw=1)
                
        # Main gap
        ax2.plot(times, self.result_.gaps, 'r-', lw=2, label=f'Gap ({treated_name})')
        ax2.axhline(0, color='k', lw=1)
        ax2.axvline(x=t_start, color='gray', linestyle=':', alpha=0.6)
        
        ax2.set_title("Treatment Effect (Gaps)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Gap (Actual - Synthetic)")
        if self.placebo_result_:
            ax2.text(0.02, 0.95, f"p-value: {self.placebo_result_.p_value:.3f}", 
                     transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
