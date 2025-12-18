"""
Statelix Survival Analysis Module

Provides:
  - KaplanMeier: Kaplan-Meier estimator for survival curves
  - LogRankTest: Log-rank test for comparing survival curves
  
Pure NumPy implementation to avoid heavy dependencies (like lifelines)
unless necessary, though lifelines is recommended for full features.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from scipy import stats


@dataclass
class SurvivalResult:
    """Result from Survival Estimation."""
    times: np.ndarray
    survival_prob: np.ndarray
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None
    median_survival_time: Optional[float] = None
    
    def summary(self) -> pd.DataFrame:
        """Return survival table."""
        df = pd.DataFrame({
            'Time': self.times,
            'Survival': self.survival_prob
        })
        if self.confidence_interval:
            df['CI_Lower'] = self.confidence_interval[0]
            df['CI_Upper'] = self.confidence_interval[1]
        return df


class KaplanMeier:
    """
    Kaplan-Meier Estimator for Survival Function.
    
    S(t) = Π (1 - d_i / n_i)
    
    Examples
    --------
    >>> km = KaplanMeier()
    >>> km.fit(durations, event_observed)
    >>> print(km.median_survival_time_)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.result_: Optional[SurvivalResult] = None
        self.median_survival_time_: Optional[float] = None
        
    def fit(
        self,
        durations: Union[np.ndarray, pd.Series],
        event_observed: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'KaplanMeier':
        """
        Fit Kaplan-Meier estimator.
        
        Parameters
        ----------
        durations : array-like
            Time to event or censoring.
        event_observed : array-like, optional
            1 if event occurred, 0 if censored.
            Default: assumes all events observed.
            
        Returns
        -------
        self
        """
        durations = np.asarray(durations).flatten()
        n = len(durations)
        
        if event_observed is None:
            event_observed = np.ones(n, dtype=int)
        else:
            event_observed = np.asarray(event_observed).flatten()
            
        # Sort by duration
        # We need to handle ties: deaths come before censorships at same time?
        # Standard: usually event=1 processed before or after?
        # Actually just grouping by time t_i:
        # d_i = number of events at t_i
        # n_i = number at risk just before t_i
        
        df = pd.DataFrame({
            'time': durations,
            'event': event_observed
        }).sort_values('time')
        
        unique_times = df['time'].unique()
        unique_times.sort()
        
        survival = 1.0
        survival_curve = []
        n_at_risk = n
        
        # Greenwood's variance sum
        var_sum = 0.0
        ci_lower = []
        ci_upper = []
        z_crit = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        times_out = []
        
        current_idx = 0
        
        for t in unique_times:
            # Stats at this time point
            in_time = df[df['time'] == t]
            d_i = in_time['event'].sum()     # deaths
            n_event_time = len(in_time)      # total samples at this time (death + censor)
            
            # n_i is number at risk at START of this time interval
            # Effectively, subtract previous dropouts
            # But here we iterate sorted unique times
            
            if n_at_risk <= 0:
                break
                
            # Update survival
            # S(t) = S(t-1) * (1 - d_i / n_i)
            term = 1.0 - d_i / n_at_risk
            survival *= term
            
            # Variance (Greenwood)
            # Var(S(t)) = S(t)^2 * Σ d_j / (n_j * (n_j - d_j))
            if n_at_risk - d_i > 0:
                var_sum += d_i / (n_at_risk * (n_at_risk - d_i))
            
            se_s = survival * np.sqrt(var_sum)
            
            # Log-log CI handling (better for 0-1 bound) is standard, 
            # but simple linear CI: S +/- z*SE
            # Let's use linear for simplicity or log-log
            
            cil = max(0.0, survival - z_crit * se_s)
            ciu = min(1.0, survival + z_crit * se_s)
            
            times_out.append(t)
            survival_curve.append(survival)
            ci_lower.append(cil)
            ci_upper.append(ciu)
            
            # Subtract those who had event OR were censored at this time from risk set for NEXT time
            n_at_risk -= n_event_time
            
        # Median survival time
        surv_arr = np.array(survival_curve)
        times_arr = np.array(times_out)
        
        # Find first time where S(t) <= 0.5
        below_05 = np.where(surv_arr <= 0.5)[0]
        if len(below_05) > 0:
            median_time = times_arr[below_05[0]]
            # Could interpolate
        else:
            median_time = np.inf
            
        self.median_survival_time_ = median_time
            
        self.result_ = SurvivalResult(
            times=times_arr,
            survival_prob=surv_arr,
            confidence_interval=(np.array(ci_lower), np.array(ci_upper)),
            median_survival_time=median_time
        )
        
        return self


class LogRankTest:
    """
    Log-Rank Test.
    
    Compares survival distributions of two samples.
    Null hypothesis: No difference between populations.
    
    Examples
    --------
    >>> result = LogRankTest.test(t1, e1, t2, e2)
    >>> print(result.p_value)
    """
    
    @staticmethod
    def test(
        times_a: np.ndarray,
        events_a: np.ndarray,
        times_b: np.ndarray,
        events_b: np.ndarray
    ):
        """
        Perform Log-Rank test.
        
        Returns
        -------
        TestResult object
        """
        times_a = np.asarray(times_a)
        events_a = np.asarray(events_a)
        times_b = np.asarray(times_b)
        events_b = np.asarray(events_b)
        
        # Combine data
        df = pd.DataFrame({
            'time': np.concatenate([times_a, times_b]),
            'event': np.concatenate([events_a, events_b]),
            'group': np.concatenate([np.zeros(len(times_a)), np.ones(len(times_b))])
        }).sort_values('time')
        
        unique_times = df[df['event'] == 1]['time'].unique()
        unique_times.sort()
        
        observed_a = 0.0
        expected_a = 0.0
        
        # Initial risk sets
        n_risk_a = len(times_a)
        n_risk_b = len(times_b)
        
        # We need to process all unique event times
        for t in unique_times:
            # Events at this time
            events_at_t = df[df['time'] == t]
            d_total = events_at_t['event'].sum()
            
            # Events in A at t
            d_a = events_at_t[(events_at_t['event'] == 1) & (events_at_t['group'] == 0)].shape[0]
            
            # Risk sets at t
            # Need strict count of those still at risk (>= t)
            # Correct approach:
            # n_risk_a = sum(times_a >= t)
            # n_risk_b = sum(times_b >= t) 
            # This is O(N^2) if naive loop.
            # But since we sorted locally, let's just use boolean sum for clarity/correctness first.
            
            n_a = np.sum(times_a >= t)
            n_b = np.sum(times_b >= t)
            n_total = n_a + n_b
            
            if n_total == 0:
                break
                
            # Expected events in A
            # E_a = d_total * (n_a / n_total)
            e_a = d_total * (n_a / n_total)
            
            observed_a += d_a
            expected_a += e_a
            
            # Variance contribution
            # V = E_a * (n_b / n_total) * ((n_total - d_total) / (n_total - 1))
            if n_total > 1:
                var_t = e_a * (n_b / n_total) * ((n_total - d_total) / (n_total - 1))
                # Add to variance... needed for denominator
                # Wait, generic log rank uses (O-E)^2 / V ? No, usually Z = (O-E)/sqrt(V)
        
        # Re-loop for variance or optimize?
        # Let's do a single pass properly.
        
        numerator = 0.0
        var_sum = 0.0
        
        for t in unique_times:
            n_a = np.sum(times_a >= t)
            n_b = np.sum(times_b >= t)
            n_total = n_a + n_b
            
            if n_total == 0: continue
            
            # True events at t
            d_a = np.sum((times_a == t) & (events_a == 1))
            d_b = np.sum((times_b == t) & (events_b == 1))
            d_total = d_a + d_b
            
            if d_total == 0: continue
            
            e_a = d_total * (n_a / n_total)
            numerator += (d_a - e_a)
            
            if n_total > 1:
                var_t = (d_total * (n_a / n_total) * (n_b / n_total) * 
                         ((n_total - d_total) / (n_total - 1)))
                var_sum += var_t
                
        z = numerator / np.sqrt(var_sum) if var_sum > 0 else 0
        chi2 = z ** 2
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        # Construct simple result object
        from .hypothesis_tests import TestResult
        return TestResult(
            test_name="Log-Rank Test",
            statistic=chi2,
            p_value=p_value,
            alternative="two-sided",
            n=len(times_a) + len(times_b),
            df=1
        )
