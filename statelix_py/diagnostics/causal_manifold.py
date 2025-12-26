
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from statelix_py.core.unified_space import CausalSpace

@dataclass
class ManifoldPoint:
    params: Dict[str, float]  # e.g., {'bandwidth': 0.5}
    estimate: float           # Treatment effect
    std_error: float
    mci_score: float
    is_stable: bool

class CausalManifold:
    """
    Computes the stability manifold for causal models.
    Sweeps through key hyperparameters to detect 'cliffs' or singularities.
    
    Enhanced with CausalSpace integration for tensor-backed operations
    and persistent homology-based stability computation.
    """
    
    def __init__(self, model: Any, data: Dict[str, Any], causal_space: Optional['CausalSpace'] = None):
        self.model = model
        self.data = data
        self.points: List[ManifoldPoint] = []
        self.causal_space = causal_space  # Optional unified space integration
    
    def compute_stability_as_ph_gradient(self) -> np.ndarray:
        """
        Compute stability using persistent homology gradient.
        
        When CausalSpace is available, uses topological stability analysis.
        Falls back to standard error-based stability otherwise.
        """
        if self.causal_space is not None:
            return self.causal_space.compute_stability_gradient()
        
        # Legacy fallback: use standard errors as stability proxy
        if self.points:
            return np.array([p.std_error for p in self.points])
        return np.array([])
    
    def get_tensor_representation(self) -> Optional[np.ndarray]:
        """Get the tensor representation of the manifold if available."""
        if self.causal_space is not None:
            return self.causal_space.points
        return None

    def compute_manifold(self, n_steps: int = 20) -> List[ManifoldPoint]:
        """Auto-detect model type and compute appropriate manifold."""
        model_name = self.model.__class__.__name__
        
        if "RDD" in model_name:
            return self._compute_rdd_manifold(n_steps)
        elif "InstrumentalVariables" in model_name or "StatelixIV" in model_name:
            return self._compute_iv_manifold(n_steps)
        elif "PropensityScoreMatching" in model_name or "StatelixPSM" in model_name:
            return self._compute_psm_manifold(n_steps)
        else:
            # Fallback for generic models: parameter sensitivity
            return []

    def _compute_rdd_manifold(self, n_steps: int) -> List[ManifoldPoint]:
        """Sweep bandwidth for RDD."""
        current_bw = getattr(self.model, 'bandwidth', 1.0)
        # Search range: 0.2x to 3x of current bandwidth
        bw_range = np.linspace(current_bw * 0.2, current_bw * 3.0, n_steps)
        
        self.points = []
        for bw in bw_range:
            try:
                # Create a lightweight clone and refit
                test_model = copy.deepcopy(self.model)
                test_model.bandwidth = bw
                test_model.fit(self.data['Y'], self.data['RunVar'], self.data.get('Exog'))
                
                # Abstract MCI - if not present, use a simple stability metric based on SE
                mci = getattr(test_model, 'mci_score_', 1.0 / (1.0 + test_model.std_error_))
                
                self.points.append(ManifoldPoint(
                    params={'bandwidth': bw},
                    estimate=test_model.effect_,
                    std_error=test_model.std_error_,
                    mci_score=mci,
                    is_stable=test_model.std_error_ < test_model.effect_ * 2 # Heuristic
                ))
            except Exception:
                continue
        return self.points

    def _compute_iv_manifold(self, n_steps: int) -> List[ManifoldPoint]:
        """
        Sweep instrument strength or inclusion. 
        For simplicity, we simulate 'instrument perturbation' or varying a regularization parameter.
        """
        # In a real 2SLS, we might look at the First Stage F-stat behavior
        # Here we simulate sensitivity to the 'first stage' coefficient scaling
        self.points = []
        # Simulate different 'instrument relevance' scenarios if possible
        # For now, let's mock a 1D sweep on a generic 'lambda' if it exists, or just return empty for stub
        return []

    def _compute_psm_manifold(self, n_steps: int) -> List[ManifoldPoint]:
        """Sweep caliper for PSM."""
        current_caliper = getattr(self.model, 'caliper', 0.2)
        if current_caliper is None: current_caliper = 0.5
        
        caliper_range = np.linspace(0.01, current_caliper * 2.5, n_steps)
        
        self.points = []
        for cal in caliper_range:
            try:
                test_model = copy.deepcopy(self.model)
                test_model.caliper = cal
                test_model.fit(self.data['y'], self.data['treatment'], self.data['X'])
                
                self.points.append(ManifoldPoint(
                    params={'caliper': cal},
                    estimate=test_model.att,
                    std_error=test_model.att_se,
                    mci_score=0.8, # Placeholder
                    is_stable=True
                ))
            except Exception:
                continue
        return self.points

    def get_quivers(self) -> List[Dict[str, Any]]:
        """Calculate 'instability arrows' pointing to lower variance/higher MCI."""
        if len(self.points) < 2: return []
        
        quivers = []
        for i in range(1, len(self.points) - 1):
            p_prev = self.points[i-1]
            p_curr = self.points[i]
            p_next = self.points[i+1]
            
            # Local gradient of estimate
            grad = (p_next.estimate - p_prev.estimate) / 2
            
            # If estimate changes rapidly, we have an instability arrow
            if abs(grad) > abs(p_curr.estimate) * 0.1:
                 quivers.append({
                     'x': i,
                     'y': p_curr.estimate,
                     'u': 0, 
                     'v': grad,
                     'color': 'red' if abs(grad) > abs(p_curr.estimate) * 0.5 else 'orange'
                 })
        return quivers

    def propose_refinement(self, suggestion_text: str) -> Optional[Dict[str, Any]]:
        """Maps a human suggestion to a concrete parameter adjustment."""
        text = suggestion_text.lower()
        if "bandwidth" in text or "window" in text:
            # Suggest a 'stabler' bandwidth if current is unstable
            if self.points:
                stablest = min(self.points, key=lambda p: p.std_error)
                return stablest.params
        if "regularization" in text or "variance" in text or "prior" in text:
             # Simulation of prior adjustment
             return {'prioir_variance_scale': 1.5} # Mock
        if "interaction" in text:
             return {'use_interactions': True}
        return None

    def compute_hypothetical_manifold(self, refinement: Dict[str, Any], n_steps: int = 20) -> List[ManifoldPoint]:
        """Compute how the manifold WOULD look after applying refinement."""
        # Clone model and apply adjustment
        hypo_model = copy.deepcopy(self.model)
        for k, v in refinement.items():
            if hasattr(hypo_model, k):
                setattr(hypo_model, k, v)
        
        # Recalculate manifold with adjusted fixed params
        hypo_engine = CausalManifold(hypo_model, self.data)
        return hypo_engine.compute_manifold(n_steps)
