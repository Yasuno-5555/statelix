from .linear import StatelixOLS
from .hnsw import StatelixHNSW
from .bayes import StatelixHMC, BayesianLogisticRegression
from .causal import (
    PropensityScoreMatching,
    InverseProbabilityWeighting,
    DoublyRobust,
    DifferenceInDifferences,
    # Aliases for backward compatibility
    StatelixPSM,
    StatelixDID,
)
from .synthetic_control import SyntheticControl, SyntheticControlResult
from .spatial import SpatialRegression, SpatialWeights
from .graph import StatelixGraph

__all__ = [
    # Linear models
    "StatelixOLS", 
    # Search
    "StatelixHNSW", 
    # Bayesian
    "StatelixHMC", "BayesianLogisticRegression",
    # Causal inference
    "PropensityScoreMatching", "InverseProbabilityWeighting", "DoublyRobust",
    "DifferenceInDifferences",
    # Aliases
    "StatelixPSM", "StatelixDID",
    "SyntheticControl", "SyntheticControlResult",
    "SpatialRegression", "SpatialWeights",
    # Graph
    "StatelixGraph"
]
