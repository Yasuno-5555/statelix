
from enum import Enum, auto

class GovernanceMode(Enum):
    """
    Defines the strictness level of Statelix's quality control.
    """
    STRICT = "STRICT"           # Threshold 0.8. Production standard.
    NORMAL = "NORMAL"           # Threshold 0.5. Draft standard.
    EXPLORATORY = "EXPLORATORY" # Threshold 0.0. No guarantees.

class GovernancePreset:
    """
    Configuration based on mode.
    """
    
    @staticmethod
    def get_threshold(mode: GovernanceMode) -> float:
        if mode == GovernanceMode.STRICT:
            return 0.8
        elif mode == GovernanceMode.NORMAL:
            return 0.5
        elif mode == GovernanceMode.EXPLORATORY:
            return 0.0
        else:
            raise ValueError(f"Unknown GovernanceMode: {mode}")

    @staticmethod
    def default() -> GovernanceMode:
        return GovernanceMode.STRICT
