"""
Statelix Multiple Comparison Correction Module

Provides:
  - bonferroni: Bonferroni correction
  - holm: Holm-Bonferroni correction
  - fdr: False Discovery Rate (Benjamini-Hochberg) correction
  
These methods adjust p-values to control the family-wise error rate (FWER)
or the false discovery rate (FDR) when performing multiple hypothesis tests.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple

def bonferroni(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bonferroni correction.
    
    Adjusted p = min(1, p * m)
    where m is the number of tests.
    
    Parameters
    ----------
    p_values : array-like
        List or array of unadjusted p-values.
    alpha : float
        Significance level.
        
    Returns
    -------
    reject : boolean array (True if null hypothesis rejected)
    corrected_p : array of corrected p-values
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    corrected_p = np.minimum(1.0, p_values * m)
    reject = corrected_p < alpha
    
    return reject, corrected_p

def holm(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni correction (step-down).
    
    More powerful than Bonferroni, still controls FWER.
    
    Parameters
    ----------
    p_values : array-like
        List or array of unadjusted p-values.
    alpha : float
        Significance level.
        
    Returns
    -------
    reject : boolean array
    corrected_p : array of corrected p-values
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    # Sort p-values
    indices = np.argsort(p_values)
    sorted_p = p_values[indices]
    
    # Calculate corrected p-values
    # adj_p_i = min(1, max(adj_p_{i-1}, (m - i + 1) * p_i))
    # But usually just check rejection step-down
    
    corrected_p = np.empty(m)
    current_max = 0.0
    
    # Calculate corrections in sorted order
    for i in range(m):
        # Rank k = i + 1 (1-based)
        # correction factor = m - i
        factor = m - i
        adj_p = sorted_p[i] * factor
        
        # Enforce monotonicity
        adj_p = max(adj_p, current_max)
        adj_p = min(1.0, adj_p)
        current_max = adj_p
        
        corrected_p[indices[i]] = adj_p
        
    reject = corrected_p < alpha
    return reject, corrected_p

def fdr(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    method: str = 'bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    False Discovery Rate correction (Benjamini-Hochberg).
    
    Controls expected proportion of false positives among rejected hypotheses.
    Less conservative than FWER methods.
    
    Parameters
    ----------
    p_values : array-like
        List or array of unadjusted p-values.
    alpha : float
        Target FDR level.
    method : str
        'bh' for Benjamini-Hochberg (independent or positive dependent)
        'by' for Benjamini-Yekutieli (general dependency) - not implemented yet
        
    Returns
    -------
    reject : boolean array
    corrected_p : array of corrected p-values (q-values)
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    
    # Sort
    indices = np.argsort(p_values)
    sorted_p = p_values[indices]
    
    # Critical values for rejection: P_i <= (i/m) * alpha
    # Q-values (corrected p):
    # q_i = min(1, min_{j>=i} (m/j) * p_j)
    
    corrected_p = np.empty(m)
    
    # Calculate q-values from largest to smallest
    min_q = 1.0
    
    for i in range(m - 1, -1, -1):
        # Rank k = i + 1
        rank = i + 1
        factor = m / rank
        q = sorted_p[i] * factor
        
        # Enforce monotonicity (step-up)
        # q_i <= q_{i+1} -> q_{i, adjusted} = min(q_i, q_{i+1, adjusted})
        q = min(q, min_q)
        q = min(1.0, q)
        min_q = q
        
        corrected_p[indices[i]] = q
        
    reject = corrected_p < alpha
    return reject, corrected_p
