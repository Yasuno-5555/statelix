
# Statelix Python Package
__version__ = '0.2.0'

# Core Models
from .models.linear import StatelixOLS
from .models.bayes import BayesianLogisticRegression

# Governance
from .facade import fit_and_judge
from .diagnostics.presets import GovernanceMode
from .diagnostics.critic import ModelRejectedError

# Accelerator
try:
    import statelix.accelerator as accelerator
except ImportError:
    accelerator = None

__all__ = [
    'StatelixOLS', 
    'BayesianLogisticRegression',
    'fit_and_judge',
    'GovernanceMode',
    'ModelRejectedError',
    'accelerator'
]
