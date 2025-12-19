"""
Statistical Tests Module
Provides wrappers for common statistical tests.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

def shapiro_wilk(data: np.ndarray) -> Dict[str, Any]:
    """Shapiro-Wilk test for normality."""
    stat, p = stats.shapiro(data)
    return {'test': 'Shapiro-Wilk', 'statistic': stat, 'p_value': p, 'normal': p > 0.05}

def kolmogorov_smirnov(data: np.ndarray) -> Dict[str, Any]:
    """Kolmogorov-Smirnov test for normality (against standard normal)."""
    # Standardize data first
    data_std = (data - np.mean(data)) / np.std(data)
    stat, p = stats.kstest(data_std, 'norm')
    return {'test': 'Kolmogorov-Smirnov', 'statistic': stat, 'p_value': p, 'normal': p > 0.05}

def levene_test(*groups) -> Dict[str, Any]:
    """Levene's test for equality of variances."""
    stat, p = stats.levene(*groups)
    return {'test': 'Levene', 'statistic': stat, 'p_value': p, 'equal_var': p > 0.05}

def bartlett_test(*groups) -> Dict[str, Any]:
    """Bartlett's test for equality of variances."""
    stat, p = stats.bartlett(*groups)
    return {'test': 'Bartlett', 'statistic': stat, 'p_value': p, 'equal_var': p > 0.05}

def mann_whitney(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return {'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p, 'significant': p < 0.05}

def kruskal_wallis(*groups) -> Dict[str, Any]:
    """Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA)."""
    stat, p = stats.kruskal(*groups)
    return {'test': 'Kruskal-Wallis', 'statistic': stat, 'p_value': p, 'significant': p < 0.05}

def run_all_normality_tests(data: np.ndarray) -> Dict[str, Dict]:
    """Run all normality tests on data."""
    return {
        'shapiro_wilk': shapiro_wilk(data),
        'ks_test': kolmogorov_smirnov(data)
    }

def format_test_result(result: Dict[str, Any]) -> str:
    """Format test result as a readable string."""
    lines = [f"Test: {result['test']}"]
    lines.append(f"Statistic: {result['statistic']:.4f}")
    lines.append(f"P-Value: {result['p_value']:.4f}")
    
    if 'normal' in result:
        lines.append(f"Conclusion: {'Likely Normal' if result['normal'] else 'Not Normal'} (α=0.05)")
    elif 'equal_var' in result:
        lines.append(f"Conclusion: {'Equal Variances' if result['equal_var'] else 'Unequal Variances'} (α=0.05)")
    elif 'significant' in result:
        lines.append(f"Conclusion: {'Significant Difference' if result['significant'] else 'No Significant Difference'} (α=0.05)")
    
    return "\n".join(lines)
