"""
Statelix Descriptive Statistics Module

Provides:
  - DataProfile: Comprehensive summary of a DataFrame.
  - correlation_matrix: Correlation matrix with p-values.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class VariableProfile:
    """Summary of a single variable."""
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    top_values: Optional[dict] = None  # for categorical
    
    def to_dict(self) -> dict:
        d = {
            'Variable': self.name,
            'Type': self.dtype,
            'Count': self.count,
            'Missing': self.missing,
            'Missing (%)': self.missing_pct,
            'Unique': self.unique
        }
        if self.mean is not None:
            d.update({
                'Mean': self.mean,
                'Std': self.std,
                'Min': self.min,
                'Median': self.median,
                'Max': self.max,
                'Skew': self.skew,
                'Kurtosis': self.kurtosis
            })
        if self.top_values:
            top_str = ", ".join([f"{k} ({v})" for k, v in self.top_values.items()])
            d['Top Values'] = top_str
        return d


class DataProfile:
    """
    Comprehensive data profiling tool.
    Similar to R's summary() or pandas-profiling (simplified).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.summary_table_: Optional[pd.DataFrame] = None
        self.profiles_: Dict[str, VariableProfile] = {}
        
    def analyze(self) -> 'DataProfile':
        """
        Run analysis on the DataFrame.
        """
        rows = []
        
        for col in self.df.columns:
            series = self.df[col]
            n = len(series)
            missing = series.isna().sum()
            missing_pct = (missing / n) * 100
            unique = series.nunique()
            dtype = str(series.dtype)
            
            prof = VariableProfile(
                name=col,
                dtype=dtype,
                count=n - missing,
                missing=missing,
                missing_pct=missing_pct,
                unique=unique
            )
            
            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna()
                if not clean_s.empty:
                    prof.mean = clean_s.mean()
                    prof.std = clean_s.std()
                    prof.min = clean_s.min()
                    prof.max = clean_s.max()
                    prof.median = clean_s.median()
                    prof.skew = clean_s.skew()
                    prof.kurtosis = clean_s.kurtosis()
            else:
                # Categorical
                if not series.empty:
                    top = series.value_counts().head(3).to_dict()
                    prof.top_values = top
            
            self.profiles_[col] = prof
            rows.append(prof.to_dict())
            
        self.summary_table_ = pd.DataFrame(rows)
        return self
    
    def summary(self) -> pd.DataFrame:
        """Return summary table."""
        if self.summary_table_ is None:
            self.analyze()
        return self.summary_table_
    
    def __repr__(self) -> str:
        return self.summary().to_string()


def correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute correlation matrix with significance stars.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    method : {'pearson', 'spearman', 'kendall'}
        Correlation method.
        
    Returns
    -------
    DataFrame with formatted correlations (e.g., "0.85***")
    """
    # Select numeric columns
    numeric_df = data.select_dtypes(include=[np.number])
    cols = numeric_df.columns
    n = len(cols)
    
    corr_mat = pd.DataFrame(index=cols, columns=cols)
    
    for i in range(n):
        for j in range(i, n):
            c1, c2 = cols[i], cols[j]
            x = numeric_df[c1]
            y = numeric_df[c2]
            
            # Remove NaNs
            mask = ~(x.isna() | y.isna())
            x_clean, y_clean = x[mask], y[mask]
            
            if len(x_clean) < 2:
                corr_mat.loc[c1, c2] = np.nan
                corr_mat.loc[c2, c1] = np.nan
                continue
                
            if method == 'pearson':
                r, p = stats.pearsonr(x_clean, y_clean)
            elif method == 'spearman':
                r, p = stats.spearmanr(x_clean, y_clean)
            elif method == 'kendall':
                r, p = stats.kendalltau(x_clean, y_clean)
            else:
                raise ValueError(f"Unknown method {method}")
            
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            val_str = f"{r:.3f}{sig}"
            
            corr_mat.loc[c1, c2] = val_str
            corr_mat.loc[c2, c1] = val_str
    
    return corr_mat
