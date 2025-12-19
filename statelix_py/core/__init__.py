from statelix.linear_model import FitOLS
from statelix.time_series import search
from statelix.bayes import hmc_sample
import statelix.bayes as bayes
import statelix.graph as graph

# Fallback init
causal = None
statelix_core = None
try:
    # Attempt import but ignore failure
    # from .statelix_core import causal
    pass
    pass
except ImportError:
    pass

try:
    import statelix.accelerator as accelerator
except ImportError:
    accelerator = None
