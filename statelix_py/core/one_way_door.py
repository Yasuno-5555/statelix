"""
One-Way Door Analysis: Point of No Return Detection

Detects operations that fundamentally change the nature of analysis,
making certain conclusions or methods no longer valid.

"今押すと、もう因果推論じゃありません"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum

try:
    from .assumption_path import AssumptionState, AssumptionPath
except ImportError:
    from statelix_py.core.assumption_path import AssumptionState, AssumptionPath


class DoorType(Enum):
    """Types of one-way doors in statistical analysis."""
    CAUSAL_TO_CORRELATIONAL = "causal_to_correlational"
    PARAMETRIC_TO_NONPARAMETRIC = "parametric_to_nonparametric"
    IDENTIFIED_TO_UNIDENTIFIED = "identified_to_unidentified"
    ESTIMABLE_TO_BOUNDS_ONLY = "estimable_to_bounds_only"
    PREDICTION_TO_DESCRIPTION = "prediction_to_description"


@dataclass
class OneWayDoor:
    """
    A detected point of no return in the analysis.
    """
    door_type: DoorType
    trigger_assumption: str
    threshold_value: float
    current_value: float
    
    # Human-readable descriptions
    before_state: str
    after_state: str
    warning_message: str
    
    # How close are we?
    distance_to_door: float  # 0 = at the door, 1 = far away
    
    @property
    def is_crossed(self) -> bool:
        """Has this door already been crossed?"""
        return self.current_value < self.threshold_value
    
    @property
    def is_imminent(self) -> bool:
        """Are we about to cross?"""
        return not self.is_crossed and self.distance_to_door < 0.2
    
    def __repr__(self) -> str:
        status = "CROSSED" if self.is_crossed else ("IMMINENT" if self.is_imminent else "safe")
        return f"OneWayDoor({self.door_type.value}, {status})"


class OneWayDoorAnalyzer:
    """
    Analyzes assumption states for one-way doors.
    
    Detects when relaxing assumptions crosses a threshold that
    fundamentally changes what kind of claims can be made.
    
    Example:
        >>> analyzer = OneWayDoorAnalyzer()
        >>> doors = analyzer.analyze(state)
        >>> for door in doors:
        ...     if door.is_imminent:
        ...         print(f"⚠ {door.warning_message}")
    """
    
    # Thresholds for each door type
    DOOR_THRESHOLDS = {
        'exogeneity': (0.3, DoorType.CAUSAL_TO_CORRELATIONAL),
        'linearity': (0.2, DoorType.PARAMETRIC_TO_NONPARAMETRIC),
        'independence': (0.25, DoorType.IDENTIFIED_TO_UNIDENTIFIED),
        'normality': (0.15, DoorType.ESTIMABLE_TO_BOUNDS_ONLY),
    }
    
    DOOR_DESCRIPTIONS = {
        DoorType.CAUSAL_TO_CORRELATIONAL: {
            'before': "因果的解釈が可能",
            'after': "相関的記述のみ",
            'warning': "今押すと、もう因果推論じゃありません。外生性の仮定が弱すぎます。"
        },
        DoorType.PARAMETRIC_TO_NONPARAMETRIC: {
            'before': "パラメトリック推定",
            'after': "ノンパラメトリック推定",
            'warning': "線形性を捨てると、推定値の解釈が根本的に変わります。"
        },
        DoorType.IDENTIFIED_TO_UNIDENTIFIED: {
            'before': "識別されたモデル",
            'after': "識別不能（集合推定のみ）",
            'warning': "独立性がこれ以上緩むと、点推定は不可能になります。"
        },
        DoorType.ESTIMABLE_TO_BOUNDS_ONLY: {
            'before': "分布推定可能",
            'after': "区間推定のみ",
            'warning': "正規性なしでは、信頼区間の意味が変わります。"
        },
        DoorType.PREDICTION_TO_DESCRIPTION: {
            'before': "予測可能",
            'after': "記述のみ",
            'warning': "この構造では、新しいデータへの予測は意味を持ちません。"
        }
    }
    
    def analyze(self, state: AssumptionState) -> List[OneWayDoor]:
        """
        Analyze an assumption state for one-way doors.
        
        Args:
            state: Current assumption state
        
        Returns:
            List of detected one-way doors
        """
        doors = []
        state_vec = state.to_vector()
        dims = ['linearity', 'independence', 'stationarity', 
                'normality', 'homoscedasticity', 'exogeneity']
        
        for dim, (threshold, door_type) in self.DOOR_THRESHOLDS.items():
            idx = dims.index(dim)
            current = state_vec[idx]
            
            desc = self.DOOR_DESCRIPTIONS[door_type]
            distance = max(0, current - threshold) / max(threshold, 0.01)
            
            door = OneWayDoor(
                door_type=door_type,
                trigger_assumption=dim,
                threshold_value=threshold,
                current_value=current,
                before_state=desc['before'],
                after_state=desc['after'],
                warning_message=desc['warning'],
                distance_to_door=min(1.0, distance)
            )
            doors.append(door)
        
        return doors
    
    def analyze_path(self, path: AssumptionPath) -> Dict[float, List[OneWayDoor]]:
        """
        Analyze entire path for door crossings.
        
        Returns:
            Dict mapping t values to doors crossed at that point
        """
        crossings = {}
        prev_crossed = set()
        
        for point in path.points:
            doors = self.analyze(point.state)
            newly_crossed = []
            
            for door in doors:
                door_key = door.door_type.value
                if door.is_crossed and door_key not in prev_crossed:
                    newly_crossed.append(door)
                    prev_crossed.add(door_key)
            
            if newly_crossed:
                crossings[point.t] = newly_crossed
        
        return crossings
    
    def get_warnings(self, state: AssumptionState) -> List[str]:
        """
        Get warning messages for current state.
        """
        warnings = []
        for door in self.analyze(state):
            if door.is_crossed:
                warnings.append(f"🚫 {door.warning_message}")
            elif door.is_imminent:
                warnings.append(f"⚠ 警告: {door.trigger_assumption} があと少しで限界です。")
        return warnings


def check_one_way_doors(state: AssumptionState) -> List[OneWayDoor]:
    """Convenience function to check for one-way doors."""
    return OneWayDoorAnalyzer().analyze(state)
