
from typing import List, Dict, Any
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
