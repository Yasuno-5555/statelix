"""
Statelix Hypothesis Testing Module

Provides:
  - TTest: One-sample, two-sample, and paired t-tests
  - ChiSquaredTest: Independence and goodness-of-fit tests
  - MannWhitneyU: Non-parametric two-group comparison
  - WilcoxonTest: Signed-rank test for paired samples
  - KruskalWallis: Non-parametric ANOVA alternative
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
from scipy import stats


@dataclass
class TestResult:
    """Generic hypothesis test result."""
    test_name: str
    statistic: float
    p_value: float
    alternative: str
    n: int
    df: Optional[float] = None
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    
    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"{self.test_name}\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.p_value:.4g} {sig}\n"
            f"  N: {self.n}"
            + (f", df: {self.df:.1f}" if self.df else "")
            + (f"\n  Effect size: {self.effect_size:.4f}" if self.effect_size else "")
            + (f"\n  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]" if self.ci_lower is not None else "")
        )
    
    def summary(self) -> pd.DataFrame:
        """Return summary as DataFrame."""
        data = {
            'Test': [self.test_name],
            'Statistic': [self.statistic],
            'P-Value': [self.p_value],
            'N': [self.n],
        }
        if self.df is not None:
            data['df'] = [self.df]
        if self.effect_size is not None:
            data['Effect Size'] = [self.effect_size]
        return pd.DataFrame(data)


class TTest:
    """
    T-Test for comparing means.
    
    Examples
    --------
    >>> # One-sample t-test
    >>> result = TTest.one_sample(data, mu=0)
    
    >>> # Two-sample t-test (independent)
    >>> result = TTest.two_sample(group1, group2)
    
    >>> # Paired t-test
    >>> result = TTest.paired(before, after)
    """
    
    @staticmethod
    def one_sample(
        x: np.ndarray,
        mu: float = 0,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        confidence: float = 0.95
    ) -> TestResult:
        """
        One-sample t-test.
        
        Parameters
        ----------
        x : array-like
            Sample data.
        mu : float
            Hypothesized population mean.
        alternative : {'two-sided', 'less', 'greater'}
            Alternative hypothesis.
        confidence : float
            Confidence level for CI.
            
        Returns
        -------
        TestResult
        """
        x = np.asarray(x).flatten()
        x = x[~np.isnan(x)]
        n = len(x)
        
        result = stats.ttest_1samp(x, mu, alternative=alternative)
        
        # Effect size (Cohen's d)
        d = (np.mean(x) - mu) / np.std(x, ddof=1)
        
        # Confidence interval
        se = stats.sem(x)
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        mean = np.mean(x)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        return TestResult(
            test_name="One-Sample T-Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative=alternative,
            n=n,
            df=n - 1,
            effect_size=d,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
    
    @staticmethod
    def two_sample(
        x: np.ndarray,
        y: np.ndarray,
        equal_var: bool = True,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        confidence: float = 0.95
    ) -> TestResult:
        """
        Two-sample t-test (independent samples).
        
        Parameters
        ----------
        x, y : array-like
            Sample data from two groups.
        equal_var : bool
            If True, use Student's t-test (equal variance).
            If False, use Welch's t-test (unequal variance).
        alternative : {'two-sided', 'less', 'greater'}
            Alternative hypothesis.
        confidence : float
            Confidence level for CI.
            
        Returns
        -------
        TestResult
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        result = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
        
        n1, n2 = len(x), len(y)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
        d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
        
        # Degrees of freedom
        if equal_var:
            df = n1 + n2 - 2
        else:
            # Welch-Satterthwaite
            v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
            df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        # CI for difference in means
        diff = np.mean(x) - np.mean(y)
        se = np.sqrt(np.var(x, ddof=1)/n1 + np.var(y, ddof=1)/n2)
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
        
        test_name = "Two-Sample T-Test" if equal_var else "Welch's T-Test"
        
        return TestResult(
            test_name=test_name,
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative=alternative,
            n=n1 + n2,
            df=df,
            effect_size=d,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
    
    @staticmethod
    def paired(
        x: np.ndarray,
        y: np.ndarray,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
        confidence: float = 0.95
    ) -> TestResult:
        """
        Paired t-test (dependent samples).
        
        Parameters
        ----------
        x, y : array-like
            Paired sample data.
        alternative : {'two-sided', 'less', 'greater'}
            Alternative hypothesis.
        confidence : float
            Confidence level for CI.
            
        Returns
        -------
        TestResult
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # Remove pairs with NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        result = stats.ttest_rel(x, y, alternative=alternative)
        
        n = len(x)
        diff = x - y
        
        # Effect size (Cohen's d for paired)
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        # CI for mean difference
        se = stats.sem(diff)
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        mean_diff = np.mean(diff)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        return TestResult(
            test_name="Paired T-Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative=alternative,
            n=n,
            df=n - 1,
            effect_size=d,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )


class ChiSquaredTest:
    """
    Chi-squared tests.
    
    Examples
    --------
    >>> # Independence test (contingency table)
    >>> result = ChiSquaredTest.independence(table)
    
    >>> # Goodness-of-fit test
    >>> result = ChiSquaredTest.goodness_of_fit(observed, expected)
    """
    
    @staticmethod
    def independence(
        table: Union[np.ndarray, pd.DataFrame],
        correction: bool = True
    ) -> TestResult:
        """
        Chi-squared test of independence.
        
        Parameters
        ----------
        table : array-like
            Contingency table (2D array).
        correction : bool
            Apply Yates' correction for 2x2 tables.
            
        Returns
        -------
        TestResult
        """
        table = np.asarray(table)
        
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=correction)
        
        # Cramér's V effect size
        n = table.sum()
        min_dim = min(table.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return TestResult(
            test_name="Chi-Squared Independence Test",
            statistic=chi2,
            p_value=p,
            alternative="two-sided",
            n=int(n),
            df=dof,
            effect_size=v  # Cramér's V
        )
    
    @staticmethod
    def goodness_of_fit(
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None
    ) -> TestResult:
        """
        Chi-squared goodness-of-fit test.
        
        Parameters
        ----------
        observed : array-like
            Observed frequencies.
        expected : array-like, optional
            Expected frequencies. If None, assumes uniform distribution.
            
        Returns
        -------
        TestResult
        """
        observed = np.asarray(observed).flatten()
        
        if expected is None:
            expected = np.full_like(observed, observed.sum() / len(observed), dtype=float)
        else:
            expected = np.asarray(expected).flatten()
        
        result = stats.chisquare(observed, f_exp=expected)
        
        return TestResult(
            test_name="Chi-Squared Goodness-of-Fit Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative="two-sided",
            n=int(observed.sum()),
            df=len(observed) - 1
        )


class MannWhitneyU:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).
    Non-parametric alternative to two-sample t-test.
    
    Example
    -------
    >>> result = MannWhitneyU.test(group1, group2)
    """
    
    @staticmethod
    def test(
        x: np.ndarray,
        y: np.ndarray,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
    ) -> TestResult:
        """
        Perform Mann-Whitney U test.
        
        Parameters
        ----------
        x, y : array-like
            Sample data from two groups.
        alternative : {'two-sided', 'less', 'greater'}
            Alternative hypothesis.
            
        Returns
        -------
        TestResult
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        result = stats.mannwhitneyu(x, y, alternative=alternative)
        
        n1, n2 = len(x), len(y)
        
        # Rank-biserial correlation as effect size
        # r = 1 - (2*U) / (n1*n2)
        r = 1 - (2 * result.statistic) / (n1 * n2)
        
        return TestResult(
            test_name="Mann-Whitney U Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative=alternative,
            n=n1 + n2,
            effect_size=r  # Rank-biserial correlation
        )


class WilcoxonTest:
    """
    Wilcoxon signed-rank test.
    Non-parametric alternative to paired t-test.
    
    Example
    -------
    >>> result = WilcoxonTest.test(before, after)
    """
    
    @staticmethod
    def test(
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided'
    ) -> TestResult:
        """
        Perform Wilcoxon signed-rank test.
        
        Parameters
        ----------
        x : array-like
            First sample or differences (if y is None).
        y : array-like, optional
            Second sample for paired comparison.
        alternative : {'two-sided', 'less', 'greater'}
            Alternative hypothesis.
            
        Returns
        -------
        TestResult
        """
        x = np.asarray(x).flatten()
        
        if y is not None:
            y = np.asarray(y).flatten()
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            diff = x - y
        else:
            diff = x[~np.isnan(x)]
        
        # Remove zeros
        diff = diff[diff != 0]
        n = len(diff)
        
        if n < 10:
            # Use exact method for small samples
            result = stats.wilcoxon(diff, alternative=alternative, method='exact')
        else:
            result = stats.wilcoxon(diff, alternative=alternative, method='approx')
        
        # Matched-pairs rank-biserial correlation as effect size
        # r = (W+ - W-) / sum(ranks)
        # Approximation: r = 1 - (2*W / (n*(n+1)))
        r = 1 - (2 * result.statistic) / (n * (n + 1) / 2) if n > 0 else 0
        
        return TestResult(
            test_name="Wilcoxon Signed-Rank Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative=alternative,
            n=n,
            effect_size=r
        )


class KruskalWallis:
    """
    Kruskal-Wallis H test.
    Non-parametric alternative to one-way ANOVA.
    
    Example
    -------
    >>> result = KruskalWallis.test(group1, group2, group3)
    """
    
    @staticmethod
    def test(*groups: np.ndarray) -> TestResult:
        """
        Perform Kruskal-Wallis H test.
        
        Parameters
        ----------
        *groups : array-like
            Variable number of groups to compare.
            
        Returns
        -------
        TestResult
        """
        # Clean data
        cleaned = []
        for g in groups:
            arr = np.asarray(g).flatten()
            arr = arr[~np.isnan(arr)]
            cleaned.append(arr)
        
        result = stats.kruskal(*cleaned)
        
        total_n = sum(len(g) for g in cleaned)
        k = len(cleaned)
        
        # Epsilon-squared effect size
        # ε² = H / (n - 1)
        eps_sq = result.statistic / (total_n - 1) if total_n > 1 else 0
        
        return TestResult(
            test_name="Kruskal-Wallis H Test",
            statistic=result.statistic,
            p_value=result.pvalue,
            alternative="two-sided",
            n=total_n,
            df=k - 1,
            effect_size=eps_sq  # Epsilon-squared
        )


# Convenience aliases
t_test_one_sample = TTest.one_sample
t_test_two_sample = TTest.two_sample
t_test_paired = TTest.paired
chi2_independence = ChiSquaredTest.independence
chi2_goodness_of_fit = ChiSquaredTest.goodness_of_fit
mann_whitney_u = MannWhitneyU.test
wilcoxon = WilcoxonTest.test
kruskal_wallis = KruskalWallis.test
