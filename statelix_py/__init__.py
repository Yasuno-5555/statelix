
# Statelix Python Package
__version__ = '2.4.0'

# Core - Unified Space
from .diagnostics.presets import GovernanceMode
from .diagnostics.critic import ModelRejectedError

def analyze(df, target, *, mode=None, gui=True):
    """
    Statelix unified entry point.
    
    Drop your data. Get structural truth.
    
    Parameters
    ----------
    df : pd.DataFrame
        Your data.
    target : str
        Column name to analyze.
    mode : GovernanceMode, optional
        Strictness level. Default: STRICT.
    gui : bool
        Launch interactive manifold GUI. Default: True.
    
    Returns
    -------
    result : dict
        Structural analysis result with stability metrics.
    
    Example
    -------
    >>> import statelix
    >>> result = statelix.analyze(df, 'outcome')
    """
    import numpy as np
    import pandas as pd
    
    if mode is None:
        mode = GovernanceMode.STRICT
    
    # Identify features (all numeric except target)
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if not features:
        raise ValueError(f"No numeric features found. Target: {target}")
    
    # Limit features if n < p
    n_rows = len(df)
    max_features = max(1, n_rows - 2)
    if len(features) > max_features:
        features = features[:max_features]
    
    # Get target and feature matrix
    y = df[target].values.astype(float)
    X = df[features].values.astype(float)
    
    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Build Causal Space
    from .core import cpp_binding
    ols_result = cpp_binding.fit_ols_full(X_clean, y_clean)
    
    result = {
        'target': target,
        'features': features,
        'n_obs': len(y_clean),
        'n_features': len(features),
        'r_squared': ols_result.r_squared,
        'coefficients': dict(zip(features, ols_result.coef_.tolist())),
        'stability': 'stable' if ols_result.r_squared > 0.1 else 'unstable',
        'mode': mode.name,
    }
    
    # Launch GUI if requested
    if gui:
        from .gui.app import run_app
        from .core.data_manager import DataManager
        dm = DataManager.instance()
        dm.set_data(df, "")
        run_app(result_to_show=result)
    
    return result


# Legacy alias
quickstart = analyze


# Accelerator
try:
    import statelix.accelerator as accelerator
except ImportError:
    accelerator = None


__all__ = [
    'analyze',
    'quickstart',  # legacy
    'GovernanceMode',
    'ModelRejectedError',
    'accelerator',
]
