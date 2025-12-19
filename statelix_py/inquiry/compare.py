
import pandas as pd
import numpy as np
from .adapters import BaseAdapter, LinearAdapter, BayesAdapter

def compare_models(models: list):
    """
    Compare multiple models using Information Criteria.
    
    Args:
        models: List of (ModelObject, "Name") tuples.
                ModelObject should be a Result object or wrapped adapter.
                
    Returns:
        Pandas DataFrame sorted by AIC (asc).
    """
    results = []
    
    for model_obj, name in models:
        # Auto-detect adapter
        # This detection logic is rudimentary. Ideally we check types.
        # Check if it has 'aic' attribute directly (Result object)
        if hasattr(model_obj, 'aic') or hasattr(model_obj, 'params'):
             adapter = LinearAdapter(model_obj)
        elif hasattr(model_obj, 'map_theta'): # Bayes
             adapter = BayesAdapter(model_obj)
        elif isinstance(model_obj, BaseAdapter):
             adapter = model_obj
        else:
             # Fallback
             adapter = LinearAdapter(model_obj)
             
        metrics = adapter.get_metrics()
        
        row = {
            'Model': name,
            'AIC': metrics.get('aic', np.nan),
            'BIC': metrics.get('bic', np.nan),
            'LogLik': metrics.get('log_likelihood', np.nan),
            'R2': metrics.get('r2', np.nan)
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Compute Delta AIC and Weights if AIC exists
    if not df['AIC'].isna().all():
        min_aic = df['AIC'].min()
        df['d_AIC'] = df['AIC'] - min_aic
        # Akaike Weights: exp(-0.5 * dAIC) / sum
        weights = np.exp(-0.5 * df['d_AIC'])
        df['Weight'] = weights / weights.sum()
        
        # Sort
        df = df.sort_values('AIC')
        
    return df
