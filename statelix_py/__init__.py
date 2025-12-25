
# Statelix Python Package
__version__ = '0.2.0'

# Core Models
from .models.linear import StatelixOLS
from .models.bayes import BayesianLogisticRegression

# Governance
from .facade import fit_and_judge
from .diagnostics.presets import GovernanceMode
from .diagnostics.critic import ModelRejectedError

def quickstart(df, target_col, features=None):
    """
    Magic entry point for new users. 
    Selects a model, runs diagnostics, and launches the GUI.
    """
    import pandas as pd
    from .models.linear import StatelixOLS
    from .facade import fit_and_judge
    
    if features is None:
        features = [c for c in df.columns if c != target_col]
        
    model = StatelixOLS()
    print("Statelix Quickstart: Selecting OLS as baseline model...")
    
    # Run fit_and_judge in STRICT mode by default
    result = fit_and_judge(model, df[target_col], df[features], mode=GovernanceMode.STRICT)
    
    # Launch GUI
    from .gui.app import run_app
    print("Statelix: Launching Intelligent Diagnostics GUI...")
    run_app(result_to_show=result)
    return result

# Accelerator
try:
    import statelix.accelerator as accelerator
except ImportError:
    accelerator = None

__all__ = [
    'StatelixOLS', 
    'BayesianLogisticRegression',
    'fit_and_judge',
    'quickstart',
    'GovernanceMode',
    'ModelRejectedError',
    'accelerator'
]
