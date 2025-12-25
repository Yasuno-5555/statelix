
from typing import List, Dict, Any, Optional
from .critic import DiagnosticReport

class DiagnosticHistory:
    """
    Tracks the evolution of model diagnostics over time (iterations).
    Used for detecting improvement or stagnation in autonomous loops.
    """
    
    def __init__(self):
        self.history: List[DiagnosticReport] = []
        
    def add(self, report: DiagnosticReport):
        self.history.append(report)
        
    def get_evolution(self) -> List[Dict[str, Any]]:
        """Returns a time-series of key metrics."""
        return [
            {
                "iteration": i,
                "mci": r.mci.total_score,
                "fit_score": r.mci.fit_score,
                "topo_score": r.mci.topology_score,
                "geo_score": r.mci.geometry_score,
                "objections_count": len(r.messages)
            }
            for i, r in enumerate(self.history)
        ]
        
    def detect_stagnation(self, window: int = 3, threshold: float = 0.01) -> bool:
        """
        Check if MCI has stopped improving over the last `window` iterations.
        """
        if len(self.history) < window:
            return False
            
        recent_scores = [r.mci.total_score for r in self.history[-window:]]
        # Check if max change is within threshold
        max_change = max(recent_scores) - min(recent_scores)
        
        return max_change < threshold

    def get_stagnation_points(self, window: int = 3, threshold: float = 0.01) -> List[int]:
        """Returns indices of iterations where stagnation was detected."""
        points = []
        for i in range(window, len(self.history) + 1):
            subset = self.history[:i]
            # Create a temporary history to use detect_stagnation logic
            temp = DiagnosticHistory()
            temp.history = subset
            if temp.detect_stagnation(window, threshold):
                points.append(i-1)
        return points

    def get_score_history(self, metric: str) -> List[float]:
        """Returns time-series for a specific MCI metric."""
        if metric == 'mci':
            return [r.mci.total_score for r in self.history]
        elif metric == 'fit':
            return [r.mci.fit_score for r in self.history]
        elif metric == 'topo':
            return [r.mci.topology_score for r in self.history]
        elif metric == 'geo':
            return [r.mci.geometry_score for r in self.history]
        return []

    def get_score_at(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Returns the full report data at a specific iteration."""
        if 0 <= iteration < len(self.history):
            r = self.history[iteration]
            return {
                "iteration": iteration,
                "mci": r.mci.total_score,
                "fit_score": r.mci.fit_score,
                "topo_score": r.mci.topology_score,
                "geo_score": r.mci.geometry_score,
                "messages": r.messages,
                "suggestions": r.suggestions
            }
        return None
