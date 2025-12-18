"""
Statelix Structural Equation Modeling (SEM) Module

Provides:
  - PathAnalysis: Basic path analysis for observed variables.
  - MediationAnalysis: Baron-Kenny and Bootstrap mediation analysis.
  
Simplifies causal chain analysis: X -> M -> Y
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple

# For path analysis, we can mostly use iterated OLS
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class PathResult:
    """Result of a single path coefficient."""
    source: str
    target: str
    coef: float
    std_err: float
    p_value: float
    
    def __repr__(self) -> str:
        sig = "*" if self.p_value < 0.05 else ""
        return f"{self.source} -> {self.target}: {self.coef:.4f}{sig} (p={self.p_value:.4g})"


@dataclass
class MediationResult:
    """Result of Mediation Analysis."""
    direct_effect: float
    indirect_effect: float
    total_effect: float
    proportion_mediated: float
    p_value_indirect: float  # Sobel or Bootstrap p-value
    paths: Dict[str, PathResult]
    
    def __repr__(self):
        return (
            "Mediation Analysis\n"
            f"  Total Effect:    {self.total_effect:.4f}\n"
            f"  Direct Effect:   {self.direct_effect:.4f}\n"
            f"  Indirect Effect: {self.indirect_effect:.4f} (p={self.p_value_indirect:.4g})\n"
            f"  Prop. Mediated:  {self.proportion_mediated:.1%}"
        )


class PathAnalysis:
    """
    Simple Path Analysis for observed variables.
    
    Estimates a set of regression equations to determine path coefficients.
    
    Example
    -------
    >>> model = PathAnalysis()
    >>> model.add_path('Y ~ X + M')
    >>> model.add_path('M ~ X')
    >>> model.fit(data)
    """
    
    def __init__(self):
        self.equations: List[str] = []
        self.results_: List[PathResult] = []
        self.models_ = {}
        
    def add_path(self, formula: str):
        """Add a regression path (e.g., 'Y ~ X + M')."""
        self.equations.append(formula)
        
    def fit(self, data: pd.DataFrame) -> 'PathAnalysis':
        """Fit all path equations."""
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels required")
            
        self.results_ = []
        self.models_ = {}
        
        for eq in self.equations:
            # Parse simple formula to get target (LHS)
            target = eq.split('~')[0].strip()
            
            # Fit OLS
            model = smf.ols(eq, data=data)
            res = model.fit()
            self.models_[target] = res
            
            # Extract coefficients (excluding Intercept usually for path diagram, but keeping results)
            for name, coef in res.params.items():
                if name == 'Intercept': continue
                
                self.results_.append(PathResult(
                    source=name,
                    target=target,
                    coef=coef,
                    std_err=res.bse[name],
                    p_value=res.pvalues[name]
                ))
                
        return self
        
    def summary(self) -> pd.DataFrame:
        """Return table of all path coefficients."""
        return pd.DataFrame([
            {
                'Source': r.source, 
                'Target': r.target, 
                'Coef': r.coef, 
                'StdErr': r.std_err, 
                'P-Value': r.p_value
            }
            for r in self.results_
        ])


class MediationAnalysis:
    """
    Mediation Analysis (X -> M -> Y).
    
    Decomposes effect of X on Y into:
      1. Indirect (Mediation): X -> M -> Y (a * b)
      2. Direct: X -> Y (c')
     
    Total Effect (c) = Direct (c') + Indirect (a*b)
    """
    
    def __init__(self, x: str, m: str, y: str):
        self.x = x
        self.m = m
        self.y = y
        self.result_: Optional[MediationResult] = None
        
    def fit(
        self, 
        data: pd.DataFrame, 
        covariates: Optional[List[str]] = None,
        bootstrap: int = 1000,
        seed: int = 42
    ) -> 'MediationAnalysis':
        """
        Fit mediation model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data.
        covariates : list of str, optional
            Control variables included in all equations.
        bootstrap : int
            Number of bootstrap samples for indirect effect CI/p-value.
            If 0, uses Sobel test (normal approximation).
            
        Returns
        -------
        self
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels required")
            
        cov_str = (" + " + " + ".join(covariates)) if covariates else ""
        
        # Path a: M ~ X (+ cov)
        f_a = f"{self.m} ~ {self.x}{cov_str}"
        model_a = smf.ols(f_a, data=data).fit()
        a = model_a.params[self.x]
        sa = model_a.bse[self.x]
        
        # Path b and c': Y ~ M + X (+ cov)
        f_bc = f"{self.y} ~ {self.m} + {self.x}{cov_str}"
        model_bc = smf.ols(f_bc, data=data).fit()
        b = model_bc.params[self.m]
        sb = model_bc.bse[self.m]
        c_prime = model_bc.params[self.x] # Direct effect
        
        # Path c (Total Effect): Y ~ X (+ cov)
        f_c = f"{self.y} ~ {self.x}{cov_str}"
        model_c = smf.ols(f_c, data=data).fit()
        c = model_c.params[self.x] # Total effect
        
        # Indirect Effect
        indirect = a * b
        
        # Significance of Indirect Effect
        p_indirect = np.nan
        
        if bootstrap > 0:
            # Bootstrap
            np.random.seed(seed)
            boot_indirect = []
            n = len(data)
            
            for _ in range(bootstrap):
                # Resample
                idx = np.random.choice(n, n, replace=True)
                sample = data.iloc[idx]
                
                # Re-estimate (simplified for speed? No, full statsmodels might be slow loop)
                # For pure speed we would use linalg directly, but let's stick to robustness
                try:
                    ma = smf.ols(f_a, data=sample).fit()
                    mbc = smf.ols(f_bc, data=sample).fit()
                    boot_indirect.append(ma.params[self.x] * mbc.params[self.m])
                except:
                    pass
            
            boot_indirect = np.array(boot_indirect)
            # Two-sided p-value from bootstrap
            # Simplest: 2 * min(P(I > 0), P(I < 0))
            p_gt_0 = np.mean(boot_indirect > 0)
            p_lt_0 = np.mean(boot_indirect < 0)
            p_indirect = 2 * min(p_gt_0, p_lt_0)
            
        else:
            # Sobel Test
            # SE_ab = sqrt(b^2*sa^2 + a^2*sb^2)
            se_ab = np.sqrt(b**2 * sa**2 + a**2 * sb**2)
            z = indirect / se_ab if se_ab > 0 else 0
            from scipy import stats
            p_indirect = 2 * (1 - stats.norm.cdf(abs(z)))
            
        prop = indirect / c if abs(c) > 1e-9 else 0
        
        # Construct Paths
        paths = {
            'a': PathResult(self.x, self.m, a, sa, model_a.pvalues[self.x]),
            'b': PathResult(self.m, self.y, b, sb, model_bc.pvalues[self.m]),
            'c_prime': PathResult(self.x, self.y, c_prime, model_bc.bse[self.x], model_bc.pvalues[self.x]),
            'c': PathResult(self.x, self.y, c, model_c.bse[self.x], model_c.pvalues[self.x])
        }
        
        self.result_ = MediationResult(
            direct_effect=c_prime,
            indirect_effect=indirect,
            total_effect=c,
            proportion_mediated=prop,
            p_value_indirect=p_indirect,
            paths=paths
        )
        
        return self

