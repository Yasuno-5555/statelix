
# Statelix Python Package
__version__ = '0.2.0'

# Core Models
from statelix.models.linear import StatelixOLS

# Accelerator
try:
    import statelix.accelerator as accelerator
except ImportError:
    accelerator = None

# Stats
# from statelix.stats import ... 

__all__ = ['StatelixOLS', 'accelerator']
