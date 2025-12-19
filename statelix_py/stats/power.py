"""
Power Analysis Module
Provides sample size and power calculations for common statistical tests.
"""
import numpy as np
from scipy import stats
from typing import Optional, Dict, Any

class PowerAnalysis:
    """Power analysis for various statistical tests."""
    
    @staticmethod
    def ttest_power(effect_size: float, n: int, alpha: float = 0.05,
                    alternative: str = 'two-sided') -> float:
        """
        Calculate power for a one-sample or two-sample t-test.
        
        Parameters:
            effect_size: Cohen's d
            n: Sample size per group
            alpha: Significance level
            alternative: 'two-sided', 'larger', 'smaller'
        """
        df = 2 * n - 2  # Two-sample t-test
        nc = effect_size * np.sqrt(n / 2)  # Non-centrality parameter
        
        if alternative == 'two-sided':
            crit = stats.t.ppf(1 - alpha / 2, df)
            power = 1 - stats.nct.cdf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)
        else:
            crit = stats.t.ppf(1 - alpha, df)
            power = 1 - stats.nct.cdf(crit, df, nc)
        
        return power
    
    @staticmethod
    def ttest_sample_size(effect_size: float, power: float = 0.8, 
                          alpha: float = 0.05) -> int:
        """
        Calculate required sample size for a t-test.
        
        Uses iterative search.
        """
        for n in range(5, 10000):
            if PowerAnalysis.ttest_power(effect_size, n, alpha) >= power:
                return n
        return 10000
    
    @staticmethod
    def anova_power(effect_size: float, n_groups: int, n_per_group: int,
                    alpha: float = 0.05) -> float:
        """
        Calculate power for one-way ANOVA.
        
        Parameters:
            effect_size: Cohen's f
            n_groups: Number of groups
            n_per_group: Sample size per group
        """
        df1 = n_groups - 1
        df2 = n_groups * (n_per_group - 1)
        nc = effect_size ** 2 * n_groups * n_per_group
        
        crit = stats.f.ppf(1 - alpha, df1, df2)
        power = 1 - stats.ncf.cdf(crit, df1, df2, nc)
        
        return power
    
    @staticmethod
    def proportion_power(p1: float, p2: float, n: int, alpha: float = 0.05) -> float:
        """
        Calculate power for a two-proportion z-test.
        """
        p_pool = (p1 + p2) / 2
        se = np.sqrt(2 * p_pool * (1 - p_pool) / n)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_effect = abs(p1 - p2) / se
        
        power = 1 - stats.norm.cdf(z_crit - z_effect) + stats.norm.cdf(-z_crit - z_effect)
        return power
    
    @staticmethod
    def power_curve(effect_sizes: np.ndarray, n: int, test: str = 'ttest',
                    alpha: float = 0.05) -> np.ndarray:
        """
        Generate power curve data for visualization.
        """
        powers = []
        for es in effect_sizes:
            if test == 'ttest':
                powers.append(PowerAnalysis.ttest_power(es, n, alpha))
            elif test == 'anova':
                powers.append(PowerAnalysis.anova_power(es, 3, n, alpha))
        return np.array(powers)
