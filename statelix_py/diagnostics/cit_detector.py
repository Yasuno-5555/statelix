
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Discovery:
    type: str          # e.g., "Topological Fragility", "Spurious Correlation"
    message: str
    severity: str      # "Warning" or "Critical"
    geometric_proof: str # Explanation of the math reason

class CITDetector:
    """
    Counter-Intuitive Truth (CIT) Detector.
    Challenging 'Success' by analyzing geometric consistency.
    """
    
    def __init__(self, metrics: Dict[str, Any], manifold_points: List[Any] = None):
        self.metrics = metrics
        self.manifold_points = manifold_points or []
        self.discoveries: List[Discovery] = []

    def detect(self) -> List[Discovery]:
        self.discoveries = []
        
        # 1. Topological Fragility (High significance, low topology)
        p_val = self.metrics.get('p_value', 1.0)
        topo_score = self.metrics.get('topology_score', 1.0)
        
        if p_val < 0.05 and topo_score < 0.4:
            self.discoveries.append(Discovery(
                type="Topological Fragility",
                message="This correlation is statistically significant but mathematically fragile.",
                severity="Critical",
                geometric_proof=f"p={p_val:.4f} suggests effect, but Topology={topo_score:.2f} indicates the structure collapses under minor perturbation."
            ))
            
        # 2. Manifold Cliff (High R2, high gradient)
        r2 = self.metrics.get('r2', 0.0)
        if r2 > 0.8 and self.manifold_points:
            # Check for steep gradients in manifold
            estimates = [p.estimate for p in self.manifold_points]
            if len(estimates) > 2:
                max_change = np.max(np.abs(np.diff(estimates)))
                avg_estimate = np.mean(estimates)
                if max_change > abs(avg_estimate) * 0.5:
                    self.discoveries.append(Discovery(
                        type="Overfit Manifold",
                        message="High explanatory power detected near a geometric cliff.",
                        severity="Warning",
                        geometric_proof=f"R²={r2:.2f} is high, but the manifold shows extreme sensitivity (instability gradient > 50%). Likely matching noise/leakage."
                    ))

        # 3. Paradoxical Convergence (MCI low but Fit high)
        mci = self.metrics.get('mci', 0.0)
        if r2 > 0.7 and mci < 0.4:
             self.discoveries.append(Discovery(
                type="Paradoxical Convergence",
                message="Model explains data while violating fundamental credibility contracts.",
                severity="Warning",
                geometric_proof="The 'Fit' is a mask. Underlying stability conditions are systematically violated."
            ))
            
        return self.discoveries
