"""
Statelix ANOVA Module

Provides:
  - OneWayANOVA: One-way analysis of variance
  - TwoWayANOVA: Two-way analysis of variance with interaction
  - TukeyHSD: Tukey's Honest Significant Difference post-hoc test
  - ANCOVA: Analysis of covariance
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict
from scipy import stats
from itertools import combinations


@dataclass
class ANOVAResult:
    """ANOVA test result."""
    source: List[str]
    ss: List[float]         # Sum of squares
    df: List[int]           # Degrees of freedom
    ms: List[float]         # Mean squares
    f_stat: List[float]     # F-statistic
    p_value: List[float]    # P-values
    eta_sq: Optional[List[float]] = None  # Effect size (η²)
    
    def summary(self) -> pd.DataFrame:
        """Return ANOVA table as DataFrame."""
        data = {
            'Source': self.source,
            'SS': self.ss,
            'df': self.df,
            'MS': self.ms,
            'F': self.f_stat,
            'P-Value': self.p_value
        }
        if self.eta_sq:
            data['η²'] = self.eta_sq
        return pd.DataFrame(data)
    
    def __repr__(self) -> str:
        return self.summary().to_string()


@dataclass
class PostHocResult:
    """Post-hoc test result."""
    comparison: List[str]
    mean_diff: List[float]
    se: List[float]
    p_value: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    significant: List[bool]
    
    def summary(self) -> pd.DataFrame:
        """Return post-hoc results as DataFrame."""
        return pd.DataFrame({
            'Comparison': self.comparison,
            'Mean Diff': self.mean_diff,
            'SE': self.se,
            'P-Value': self.p_value,
            '95% CI Lower': self.ci_lower,
            '95% CI Upper': self.ci_upper,
            'Significant': self.significant
        })
    
    def __repr__(self) -> str:
        return self.summary().to_string()


class OneWayANOVA:
    """
    One-Way Analysis of Variance.
    
    Tests whether the means of multiple groups are equal.
    
    Examples
    --------
    >>> anova = OneWayANOVA()
    >>> result = anova.fit(y, groups)
    >>> print(result.summary())
    
    >>> # Or with separate group arrays
    >>> result = anova.fit_groups(group1, group2, group3)
    """
    
    def fit(
        self,
        y: np.ndarray,
        groups: np.ndarray
    ) -> ANOVAResult:
        """
        Fit one-way ANOVA.
        
        Parameters
        ----------
        y : array-like
            Response variable.
        groups : array-like
            Group labels for each observation.
            
        Returns
        -------
        ANOVAResult
        """
        y = np.asarray(y).flatten()
        groups = np.asarray(groups).flatten()
        
        # Get unique groups
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        n = len(y)
        
        # Group data
        group_data = [y[groups == g] for g in unique_groups]
        
        return self._compute_anova(group_data, n, k)
    
    def fit_groups(self, *groups: np.ndarray) -> ANOVAResult:
        """
        Fit one-way ANOVA from separate group arrays.
        
        Parameters
        ----------
        *groups : array-like
            Variable number of group arrays.
            
        Returns
        -------
        ANOVAResult
        """
        group_data = [np.asarray(g).flatten() for g in groups]
        group_data = [g[~np.isnan(g)] for g in group_data]
        
        n = sum(len(g) for g in group_data)
        k = len(group_data)
        
        return self._compute_anova(group_data, n, k)
    
    def _compute_anova(
        self,
        group_data: List[np.ndarray],
        n: int,
        k: int
    ) -> ANOVAResult:
        """Compute ANOVA statistics."""
        # Grand mean
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        
        # Group means
        group_means = [np.mean(g) for g in group_data]
        group_ns = [len(g) for g in group_data]
        
        # Sum of squares
        # SS_between = Σ n_j * (mean_j - grand_mean)²
        ss_between = sum(
            n_j * (m_j - grand_mean) ** 2
            for n_j, m_j in zip(group_ns, group_means)
        )
        
        # SS_within = Σ Σ (y_ij - mean_j)²
        ss_within = sum(
            np.sum((g - m) ** 2)
            for g, m in zip(group_data, group_means)
        )
        
        ss_total = ss_between + ss_within
        
        # Degrees of freedom
        df_between = k - 1
        df_within = n - k
        df_total = n - 1
        
        # Mean squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # F-statistic
        f_stat = ms_between / ms_within if ms_within > 0 else np.inf
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
        
        # Effect size (η²)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        return ANOVAResult(
            source=['Between Groups', 'Within Groups', 'Total'],
            ss=[ss_between, ss_within, ss_total],
            df=[df_between, df_within, df_total],
            ms=[ms_between, ms_within, np.nan],
            f_stat=[f_stat, np.nan, np.nan],
            p_value=[p_value, np.nan, np.nan],
            eta_sq=[eta_sq, np.nan, np.nan]
        )


class TwoWayANOVA:
    """
    Two-Way Analysis of Variance.
    
    Tests main effects and interaction of two factors.
    
    Examples
    --------
    >>> anova = TwoWayANOVA()
    >>> result = anova.fit(y, factor_a, factor_b)
    >>> print(result.summary())
    """
    
    def fit(
        self,
        y: np.ndarray,
        factor_a: np.ndarray,
        factor_b: np.ndarray
    ) -> ANOVAResult:
        """
        Fit two-way ANOVA.
        
        Parameters
        ----------
        y : array-like
            Response variable.
        factor_a : array-like
            First factor labels.
        factor_b : array-like
            Second factor labels.
            
        Returns
        -------
        ANOVAResult
        """
        y = np.asarray(y).flatten()
        factor_a = np.asarray(factor_a).flatten()
        factor_b = np.asarray(factor_b).flatten()
        
        n = len(y)
        grand_mean = np.mean(y)
        
        # Unique levels
        levels_a = np.unique(factor_a)
        levels_b = np.unique(factor_b)
        a = len(levels_a)
        b = len(levels_b)
        
        # Marginal means
        means_a = {lev: np.mean(y[factor_a == lev]) for lev in levels_a}
        means_b = {lev: np.mean(y[factor_b == lev]) for lev in levels_b}
        
        # Cell means
        cell_means = {}
        cell_ns = {}
        for la in levels_a:
            for lb in levels_b:
                mask = (factor_a == la) & (factor_b == lb)
                if np.any(mask):
                    cell_means[(la, lb)] = np.mean(y[mask])
                    cell_ns[(la, lb)] = np.sum(mask)
        
        # Sum of squares
        # SS_A (main effect of A)
        ss_a = sum(
            np.sum(factor_a == lev) * (means_a[lev] - grand_mean) ** 2
            for lev in levels_a
        )
        
        # SS_B (main effect of B)
        ss_b = sum(
            np.sum(factor_b == lev) * (means_b[lev] - grand_mean) ** 2
            for lev in levels_b
        )
        
        # SS_AB (interaction)
        ss_ab = 0
        for la in levels_a:
            for lb in levels_b:
                if (la, lb) in cell_means:
                    expected = means_a[la] + means_b[lb] - grand_mean
                    ss_ab += cell_ns[(la, lb)] * (cell_means[(la, lb)] - expected) ** 2
        
        # SS_within (error)
        ss_within = 0
        for la in levels_a:
            for lb in levels_b:
                mask = (factor_a == la) & (factor_b == lb)
                if np.any(mask):
                    ss_within += np.sum((y[mask] - cell_means[(la, lb)]) ** 2)
        
        ss_total = np.sum((y - grand_mean) ** 2)
        
        # Degrees of freedom
        df_a = a - 1
        df_b = b - 1
        df_ab = (a - 1) * (b - 1)
        df_within = n - a * b
        df_total = n - 1
        
        # Mean squares
        ms_a = ss_a / df_a if df_a > 0 else 0
        ms_b = ss_b / df_b if df_b > 0 else 0
        ms_ab = ss_ab / df_ab if df_ab > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # F-statistics
        f_a = ms_a / ms_within if ms_within > 0 else np.inf
        f_b = ms_b / ms_within if ms_within > 0 else np.inf
        f_ab = ms_ab / ms_within if ms_within > 0 else np.inf
        
        # P-values
        p_a = 1 - stats.f.cdf(f_a, df_a, df_within) if df_within > 0 else np.nan
        p_b = 1 - stats.f.cdf(f_b, df_b, df_within) if df_within > 0 else np.nan
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_within) if df_within > 0 else np.nan
        
        # Effect sizes (partial η²)
        eta_a = ss_a / (ss_a + ss_within) if (ss_a + ss_within) > 0 else 0
        eta_b = ss_b / (ss_b + ss_within) if (ss_b + ss_within) > 0 else 0
        eta_ab = ss_ab / (ss_ab + ss_within) if (ss_ab + ss_within) > 0 else 0
        
        return ANOVAResult(
            source=['Factor A', 'Factor B', 'A × B', 'Within', 'Total'],
            ss=[ss_a, ss_b, ss_ab, ss_within, ss_total],
            df=[df_a, df_b, df_ab, df_within, df_total],
            ms=[ms_a, ms_b, ms_ab, ms_within, np.nan],
            f_stat=[f_a, f_b, f_ab, np.nan, np.nan],
            p_value=[p_a, p_b, p_ab, np.nan, np.nan],
            eta_sq=[eta_a, eta_b, eta_ab, np.nan, np.nan]
        )


class TukeyHSD:
    """
    Tukey's Honest Significant Difference test.
    
    Post-hoc test for pairwise comparisons after ANOVA.
    
    Examples
    --------
    >>> tukey = TukeyHSD()
    >>> result = tukey.fit(y, groups)
    >>> print(result.summary())
    """
    
    def fit(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        alpha: float = 0.05
    ) -> PostHocResult:
        """
        Perform Tukey HSD test.
        
        Parameters
        ----------
        y : array-like
            Response variable.
        groups : array-like
            Group labels.
        alpha : float
            Significance level.
            
        Returns
        -------
        PostHocResult
        """
        y = np.asarray(y).flatten()
        groups = np.asarray(groups).flatten()
        
        # Get unique groups
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        n = len(y)
        
        # Group statistics
        group_data = {g: y[groups == g] for g in unique_groups}
        group_means = {g: np.mean(d) for g, d in group_data.items()}
        group_ns = {g: len(d) for g, d in group_data.items()}
        
        # MS_within (pooled variance)
        ss_within = sum(
            np.sum((d - group_means[g]) ** 2)
            for g, d in group_data.items()
        )
        df_within = n - k
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # Pairwise comparisons
        comparisons = []
        mean_diffs = []
        ses = []
        p_values = []
        ci_lowers = []
        ci_uppers = []
        
        for g1, g2 in combinations(unique_groups, 2):
            # Mean difference
            diff = group_means[g1] - group_means[g2]
            
            # Standard error
            se = np.sqrt(ms_within * (1/group_ns[g1] + 1/group_ns[g2]) / 2)
            
            # q statistic
            q = abs(diff) / se if se > 0 else np.inf
            
            # P-value using studentized range distribution
            # Approximation using Tukey-Kramer
            from scipy.stats import studentized_range
            try:
                p = 1 - studentized_range.cdf(q * np.sqrt(2), k, df_within)
            except:
                # Fallback: use conservative Bonferroni
                p = min(1.0, stats.t.sf(abs(diff) / se, df_within) * 2 * k * (k-1) / 2) if se > 0 else 0
            
            # Critical value for CI
            try:
                q_crit = studentized_range.ppf(1 - alpha, k, df_within)
            except:
                q_crit = stats.t.ppf(1 - alpha / (k * (k-1)), df_within) * np.sqrt(2)
            
            ci_lower = diff - q_crit * se / np.sqrt(2)
            ci_upper = diff + q_crit * se / np.sqrt(2)
            
            comparisons.append(f"{g1} - {g2}")
            mean_diffs.append(diff)
            ses.append(se)
            p_values.append(p)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
        
        return PostHocResult(
            comparison=comparisons,
            mean_diff=mean_diffs,
            se=ses,
            p_value=p_values,
            ci_lower=ci_lowers,
            ci_upper=ci_uppers,
            significant=[p < alpha for p in p_values]
        )
    
    def fit_groups(self, *groups: np.ndarray, alpha: float = 0.05) -> PostHocResult:
        """
        Perform Tukey HSD from separate group arrays.
        
        Parameters
        ----------
        *groups : array-like
            Variable number of group arrays.
        alpha : float
            Significance level.
            
        Returns
        -------
        PostHocResult
        """
        group_data = [np.asarray(g).flatten() for g in groups]
        group_data = [g[~np.isnan(g)] for g in group_data]
        
        # Create combined arrays
        y = np.concatenate(group_data)
        group_labels = np.concatenate([
            np.full(len(g), i) for i, g in enumerate(group_data)
        ])
        
        return self.fit(y, group_labels, alpha)


class ANCOVA:
    """
    Analysis of Covariance.
    
    Combines ANOVA with regression to control for covariates.
    
    Examples
    --------
    >>> ancova = ANCOVA()
    >>> result = ancova.fit(y, groups, covariates)
    >>> print(result.summary())
    """
    
    def fit(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        covariates: np.ndarray
    ) -> Dict:
        """
        Fit ANCOVA model.
        
        Parameters
        ----------
        y : array-like
            Response variable.
        groups : array-like
            Group labels.
        covariates : array-like
            Covariate(s) to control for.
            
        Returns
        -------
        dict with ANOVA table and adjusted means
        """
        y = np.asarray(y).flatten()
        groups = np.asarray(groups).flatten()
        covariates = np.asarray(covariates)
        
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        
        n = len(y)
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        p = covariates.shape[1]  # number of covariates
        
        # Create dummy variables for groups
        dummies = np.zeros((n, k - 1))
        for i, g in enumerate(unique_groups[1:]):
            dummies[:, i] = (groups == g).astype(float)
        
        # Full model: y ~ dummies + covariates
        X_full = np.column_stack([np.ones(n), dummies, covariates])
        
        # Reduced model: y ~ covariates only
        X_reduced = np.column_stack([np.ones(n), covariates])
        
        # Fit models
        try:
            beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
            y_pred_full = X_full @ beta_full
            ss_res_full = np.sum((y - y_pred_full) ** 2)
            
            beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
            y_pred_reduced = X_reduced @ beta_reduced
            ss_res_reduced = np.sum((y - y_pred_reduced) ** 2)
        except:
            return {'error': 'Model fitting failed'}
        
        # F-test for group effect
        df1 = k - 1  # groups df
        df2 = n - k - p  # error df
        
        ms_groups = (ss_res_reduced - ss_res_full) / df1
        ms_error = ss_res_full / df2 if df2 > 0 else 0
        
        f_stat = ms_groups / ms_error if ms_error > 0 else np.inf
        p_value = 1 - stats.f.cdf(f_stat, df1, df2) if df2 > 0 else np.nan
        
        # Adjusted means
        covariate_means = np.mean(covariates, axis=0)
        adjusted_means = {}
        
        for i, g in enumerate(unique_groups):
            # Predicted y at mean covariate values for this group
            if i == 0:
                x_g = np.concatenate([[1], np.zeros(k-1), covariate_means])
            else:
                dummy = np.zeros(k - 1)
                dummy[i - 1] = 1
                x_g = np.concatenate([[1], dummy, covariate_means])
            adjusted_means[g] = x_g @ beta_full
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_groups': df1,
            'df_error': df2,
            'adjusted_means': adjusted_means,
            'ss_groups': ss_res_reduced - ss_res_full,
            'ss_error': ss_res_full,
            'coefficients': beta_full
        }


# Convenience functions
def one_way_anova(y: np.ndarray, groups: np.ndarray) -> ANOVAResult:
    """One-way ANOVA convenience function."""
    return OneWayANOVA().fit(y, groups)

def two_way_anova(y: np.ndarray, factor_a: np.ndarray, factor_b: np.ndarray) -> ANOVAResult:
    """Two-way ANOVA convenience function."""
    return TwoWayANOVA().fit(y, factor_a, factor_b)

def tukey_hsd(y: np.ndarray, groups: np.ndarray, alpha: float = 0.05) -> PostHocResult:
    """Tukey HSD convenience function."""
    return TukeyHSD().fit(y, groups, alpha)
