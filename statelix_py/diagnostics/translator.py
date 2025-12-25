
from typing import Dict, List

class ReasonTranslator:
    """
    Translates mathematical signals and metrics into human-readable diagnostic messages.
    """
    
    def translate(self, metrics: Dict[str, float]) -> List[str]:
        messages = []
        
        # --- Fit Checks ---
        if metrics.get('r2', 1.0) < 0.3:
            messages.append("Model Fit: Extremely poor fit (R² < 0.3). The model fails to explain the data variance.")
        elif metrics.get('r2', 1.0) < 0.6:
            messages.append("Model Fit: Weak fit. Consider adding more features or non-linear terms.")
            
        # --- Topology Checks (Keirin) ---
        # CV > 10%
        mean_topo = metrics.get('mean_structure', 1.0)
        std_topo = metrics.get('std_structure', 0.0)
        cv = std_topo / (abs(mean_topo) + 1e-9)
        
        if cv > 0.1:
            messages.append("Stability: Topological features are fluctuating significantly. The underlying manifold might be unstable or data is too noisy.")
            
        if metrics.get('topology_jump', False): # Boolean signal
             messages.append("Stability: Detected a sudden 'Structure Collapse'. The model may have degenerated to a trivial solution.")

        # --- Geometry Checks (Shinen) ---
        if metrics.get('invariant_ratio', 1.0) < 0.8:
            messages.append("Robustness: Key features change value when the coordinate system is rotated. Ensure units are standardized or use invariant features.")
            
        return messages
