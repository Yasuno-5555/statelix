"""
Cliff Learner: Learning from Assumption Breakdowns

Collects cliff data across analyses and learns patterns.
Can predict "あなた、またここで死にますよ" before it happens.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import json
import os
import warnings

try:
    from .assumption_path import CliffPoint, AssumptionState
except ImportError:
    from statelix_py.core.assumption_path import CliffPoint, AssumptionState


# =============================================================================
# Cliff Record
# =============================================================================

@dataclass
class CliffRecord:
    """
    A recorded cliff with context for learning.
    """
    cliff: CliffPoint
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis metadata
    data_size: int = 0
    n_features: int = 0
    model_type: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            't': self.cliff.t,
            'broken_assumption': self.cliff.broken_assumption,
            'curvature': self.cliff.curvature,
            'severity': self.cliff.severity,
            'state': self.cliff.state.to_vector().tolist(),
            'context': self.context,
            'data_size': self.data_size,
            'n_features': self.n_features,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CliffRecord':
        """Create from dictionary."""
        state = AssumptionState.from_vector(np.array(d.get('state', [0.5]*6)))
        cliff = CliffPoint(
            t=d['t'],
            state=state,
            curvature=d.get('curvature', 0.0),
            broken_assumption=d['broken_assumption'],
            estimate_before=0.0,  # Not stored
            estimate_after=0.0
        )
        return cls(
            cliff=cliff,
            context=d.get('context', {}),
            data_size=d.get('data_size', 0),
            n_features=d.get('n_features', 0),
            model_type=d.get('model_type', ''),
            timestamp=d.get('timestamp', '')
        )


# =============================================================================
# Cliff Learner
# =============================================================================

class CliffLearner:
    """
    Learns patterns in assumption breakdowns.
    
    Over many analyses, builds a model of:
    - Which assumptions are fragile together
    - What data characteristics trigger cliffs
    - Where in assumption space cliffs cluster
    
    Then provides proactive warnings:
    "あなた、またここで死にますよ"
    
    Usage:
        >>> learner = CliffLearner()
        >>> learner.load("cliff_history.json")  # Load past experience
        >>> 
        >>> # After analysis
        >>> for cliff in path.cliffs:
        ...     learner.record_cliff(cliff, {'model': 'OLS', 'n': 1000})
        >>> 
        >>> # Before new analysis
        >>> risks = learner.predict_cliff_risk(proposed_state)
        >>> if risks['linearity'] > 0.7:
        ...     print("⚠ あなた、またここで死にますよ")
    """
    
    def __init__(self, history_path: Optional[str] = None):
        """
        Initialize the learner.
        
        Args:
            history_path: Optional path to persist cliff history
        """
        self.records: List[CliffRecord] = []
        self.history_path = history_path
        
        # Cached statistics
        self._assumption_counts: Dict[str, int] = defaultdict(int)
        self._pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._location_clusters: List[float] = []
    
    def record_cliff(
        self,
        cliff: CliffPoint,
        context: Optional[Dict[str, Any]] = None,
        data_size: int = 0,
        n_features: int = 0,
        model_type: str = ""
    ) -> None:
        """
        Record a cliff for learning.
        
        Args:
            cliff: The detected cliff point
            context: Optional analysis context
            data_size: Number of observations
            n_features: Number of features
            model_type: Type of model
        """
        from datetime import datetime
        
        record = CliffRecord(
            cliff=cliff,
            context=context or {},
            data_size=data_size,
            n_features=n_features,
            model_type=model_type,
            timestamp=datetime.now().isoformat()
        )
        
        self.records.append(record)
        
        # Update statistics
        self._assumption_counts[cliff.broken_assumption] += 1
        self._location_clusters.append(cliff.t)
        
        # Auto-save if path configured
        if self.history_path:
            self.save(self.history_path)
    
    def record_from_path(self, path: 'AssumptionPath', context: Optional[Dict] = None) -> int:
        """
        Record all cliffs from a path analysis.
        
        Args:
            path: AssumptionPath with detected cliffs
            context: Optional analysis context
        
        Returns:
            Number of cliffs recorded
        """
        count = 0
        for cliff in path.cliffs:
            self.record_cliff(cliff, context)
            count += 1
        return count
    
    def predict_cliff_risk(self, state: AssumptionState) -> Dict[str, float]:
        """
        Predict cliff probability for each assumption dimension.
        
        Based on historical patterns, estimates which assumptions
        are likely to cause problems at the given state.
        
        Args:
            state: Proposed assumption state
        
        Returns:
            Dict mapping assumption names to risk scores [0, 1]
        """
        if not self.records:
            # No history - return uniform low risk
            return {dim: 0.1 for dim in [
                'linearity', 'independence', 'stationarity',
                'normality', 'homoscedasticity', 'exogeneity'
            ]}
        
        risks = {}
        total_cliffs = len(self.records)
        state_vec = state.to_vector()
        
        dimensions = ['linearity', 'independence', 'stationarity',
                     'normality', 'homoscedasticity', 'exogeneity']
        
        for i, dim in enumerate(dimensions):
            # Base risk from frequency
            freq_risk = self._assumption_counts.get(dim, 0) / max(total_cliffs, 1)
            
            # Position risk - are we in a dangerous region?
            position_risk = self._compute_position_risk(dim, state_vec[i])
            
            # Combine risks
            risks[dim] = min(1.0, freq_risk * 0.5 + position_risk * 0.5)
        
        return risks
    
    def _compute_position_risk(self, dim: str, value: float) -> float:
        """
        Compute risk based on where we are in assumption space.
        
        Looks at where past cliffs occurred for this dimension.
        """
        relevant_records = [r for r in self.records 
                          if r.cliff.broken_assumption == dim]
        
        if not relevant_records:
            return 0.1
        
        # Get values where cliffs occurred
        dim_idx = ['linearity', 'independence', 'stationarity',
                   'normality', 'homoscedasticity', 'exogeneity'].index(dim)
        
        cliff_values = []
        for r in relevant_records:
            cliff_values.append(r.cliff.state.to_vector()[dim_idx])
        
        cliff_values = np.array(cliff_values)
        
        # Risk increases when close to historical cliff locations
        distances = np.abs(cliff_values - value)
        if len(distances) > 0:
            min_dist = np.min(distances)
            # Risk is inverse of distance (closer = higher risk)
            return 1.0 / (1.0 + 5 * min_dist)
        
        return 0.1
    
    def fragile_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Find assumption pairs that tend to break together.
        
        Returns:
            List of (assumption1, assumption2, co-occurrence_score) tuples
        """
        # Build co-occurrence from sequential cliffs
        from itertools import combinations
        
        pair_counts = defaultdict(int)
        
        # Group records by analysis (crude: same minute)
        by_analysis = defaultdict(list)
        for r in self.records:
            # Use first 16 chars of timestamp as group key
            key = r.timestamp[:16] if r.timestamp else str(id(r))
            by_analysis[key].append(r)
        
        # Count pairs within same analysis
        for analysis_records in by_analysis.values():
            assumptions = [r.cliff.broken_assumption for r in analysis_records]
            unique = list(set(assumptions))
            for a1, a2 in combinations(sorted(unique), 2):
                pair_counts[(a1, a2)] += 1
        
        # Convert to list with scores
        total_analyses = len(by_analysis)
        if total_analyses == 0:
            return []
        
        pairs = []
        for (a1, a2), count in pair_counts.items():
            score = count / total_analyses
            if score > 0.1:  # Only report significant pairs
                pairs.append((a1, a2, score))
        
        return sorted(pairs, key=lambda x: -x[2])
    
    def warning_message(self, state: AssumptionState) -> Optional[str]:
        """
        Generate a warning message if high risk detected.
        
        Returns:
            Warning message or None if low risk
        """
        risks = self.predict_cliff_risk(state)
        high_risks = [(dim, risk) for dim, risk in risks.items() if risk > 0.5]
        
        if not high_risks:
            return None
        
        # Sort by risk
        high_risks.sort(key=lambda x: -x[1])
        
        if high_risks[0][1] > 0.7:
            return f"⚠ あなた、またここで死にますよ。{high_risks[0][0]} の仮定が危険です。"
        else:
            dims = ", ".join([d for d, _ in high_risks[:2]])
            return f"⚠ 注意: {dims} で問題が起きやすい傾向があります。"
    
    def statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        """
        if not self.records:
            return {'total_cliffs': 0, 'top_assumptions': [], 'fragile_pairs': []}
        
        # Top broken assumptions
        sorted_assumptions = sorted(
            self._assumption_counts.items(),
            key=lambda x: -x[1]
        )
        
        return {
            'total_cliffs': len(self.records),
            'top_assumptions': sorted_assumptions[:3],
            'fragile_pairs': self.fragile_pairs()[:3],
            'cliff_location_mean': np.mean(self._location_clusters) if self._location_clusters else 0.5,
            'cliff_location_std': np.std(self._location_clusters) if len(self._location_clusters) > 1 else 0.0
        }
    
    def save(self, path: str) -> None:
        """Save cliff history to JSON file."""
        data = {
            'records': [r.to_dict() for r in self.records],
            'assumption_counts': dict(self._assumption_counts),
            'location_clusters': self._location_clusters
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> bool:
        """
        Load cliff history from JSON file.
        
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.records = [CliffRecord.from_dict(r) for r in data.get('records', [])]
            self._assumption_counts = defaultdict(int, data.get('assumption_counts', {}))
            self._location_clusters = data.get('location_clusters', [])
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to load cliff history: {e}")
            return False
    
    def __repr__(self) -> str:
        return f"CliffLearner(records={len(self.records)})"


# =============================================================================
# Global Learner Instance
# =============================================================================

_global_learner: Optional[CliffLearner] = None

def get_cliff_learner() -> CliffLearner:
    """Get or create the global cliff learner instance."""
    global _global_learner
    if _global_learner is None:
        _global_learner = CliffLearner()
    return _global_learner

def set_cliff_learner(learner: CliffLearner) -> None:
    """Set the global cliff learner instance."""
    global _global_learner
    _global_learner = learner
