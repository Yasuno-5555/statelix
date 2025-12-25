
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MCIScore:
    total_score: float
    fit_score: float
    topology_score: float
    geometry_score: float
    summary: str

class ModelCredibilityIndex:
    """
    Calculates the Model Credibility Index (MCI), a unified trust score (0.0 - 1.0)
    combining fit quality, topological stability, and geometric robustness.
    """
    
    def calculate(
        self, 
        fit_metrics: Dict[str, float], 
        topology_metrics: Dict[str, float], 
        geometry_metrics: Dict[str, float]
    ) -> MCIScore:
        """
        :param fit_metrics: e.g., {'r2': 0.95, 'rmse': 0.1}
        :param topology_metrics: e.g., {'mean_structure': 4.0, 'std_structure': 0.1}
        :param geometry_metrics: e.g., {'invariant_ratio': 0.9}
        """
        
        # 1. Fit Score (0-1)
        # Use R2 if available, else derive from RMSE/Likelihood roughly
        fit_score = fit_metrics.get('r2', 0.5) 
        if fit_score < 0: fit_score = 0.0 # R2 can be negative
        
        # 2. Topology Score (0-1)
        # Higher score = Lower Variance (Higher Stability)
        # Assume if std > 10% of mean, it's unstable
        mean_topo = topology_metrics.get('mean_structure', 1.0)
        std_topo = topology_metrics.get('std_structure', 0.0)
        
        cv = std_topo / (abs(mean_topo) + 1e-9) # Coefficient of variation
        # Map CV to score: CV=0 -> 1.0, CV=0.1 -> 0.8, CV=0.5 -> 0.0
        # sigmoid-like decay
        topo_score = max(0.0, 1.0 - (cv * 2.0)) # linear penalty for now
        
        # 3. Geometry Score (0-1)
        # Simply the ratio of invariant features or average invariant score
        geo_score = geometry_metrics.get('invariant_ratio', 0.5)
        
        # Optimized Weights (can be tuned)
        # Fit is important, but a high Fit with low Stability is "Overfitting"
        w_fit = 0.4
        w_topo = 0.3
        w_geo = 0.3
        
        total_score = (fit_score * w_fit) + (topo_score * w_topo) + (geo_score * w_geo)
        
        summary = f"MCI: {total_score:.2f} (Fit: {fit_score:.2f}, Topo: {topo_score:.2f}, Geo: {geo_score:.2f})"
        
        return MCIScore(
            total_score=total_score,
            fit_score=fit_score,
            topology_score=topo_score,
            geometry_score=geo_score,
            summary=summary
        )
