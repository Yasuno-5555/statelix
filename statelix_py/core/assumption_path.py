"""
Assumption Path: Model Is a Path

Statistical models are not points - they are paths through assumption space.
This module traces how model estimates change as assumptions are relaxed,
detecting "cliffs" where assumptions break down.

Architecture:
- AssumptionState: A point in assumption space [0,1]^d
- AssumptionPath: A trajectory through assumption space
- PathPoint: Estimate + diagnostics at each path position
- CliffPoint: Detected instability (assumption breakdown)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, TYPE_CHECKING
from enum import Enum
import warnings

if TYPE_CHECKING:
    from statelix_py.core.unified_space import CausalSpace, PersistenceDiagram


# =============================================================================
# Assumption Dimensions
# =============================================================================

class AssumptionDimension(Enum):
    """Named dimensions in assumption space."""
    LINEARITY = "linearity"         # 0=nonlinear, 1=linear
    INDEPENDENCE = "independence"   # 0=correlated, 1=iid
    STATIONARITY = "stationarity"   # 0=nonstationary, 1=stationary
    NORMALITY = "normality"         # 0=heavy-tailed, 1=gaussian
    HOMOSCEDASTICITY = "homoscedasticity"  # 0=heteroscedastic, 1=homoscedastic
    EXOGENEITY = "exogeneity"       # 0=endogenous, 1=exogenous


@dataclass
class AssumptionState:
    """
    A point in assumption space.
    
    Each dimension represents how "strong" an assumption is held:
    - 0.0 = assumption fully relaxed (most general model)
    - 1.0 = assumption fully enforced (classical model)
    
    The space is [0,1]^d where d = number of assumption dimensions.
    """
    linearity: float = 1.0
    independence: float = 1.0
    stationarity: float = 1.0
    normality: float = 1.0
    homoscedasticity: float = 1.0
    exogeneity: float = 1.0
    
    def __post_init__(self):
        # Clamp all values to [0, 1]
        self.linearity = np.clip(self.linearity, 0.0, 1.0)
        self.independence = np.clip(self.independence, 0.0, 1.0)
        self.stationarity = np.clip(self.stationarity, 0.0, 1.0)
        self.normality = np.clip(self.normality, 0.0, 1.0)
        self.homoscedasticity = np.clip(self.homoscedasticity, 0.0, 1.0)
        self.exogeneity = np.clip(self.exogeneity, 0.0, 1.0)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for geometric operations."""
        return np.array([
            self.linearity,
            self.independence,
            self.stationarity,
            self.normality,
            self.homoscedasticity,
            self.exogeneity
        ])
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'AssumptionState':
        """Create from numpy array."""
        return cls(
            linearity=float(v[0]),
            independence=float(v[1]),
            stationarity=float(v[2]),
            normality=float(v[3]),
            homoscedasticity=float(v[4]) if len(v) > 4 else 1.0,
            exogeneity=float(v[5]) if len(v) > 5 else 1.0
        )
    
    @classmethod
    def classical(cls) -> 'AssumptionState':
        """Create a classical (all assumptions enforced) state."""
        return cls(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    
    @classmethod
    def fully_relaxed(cls) -> 'AssumptionState':
        """Create a fully relaxed (no assumptions) state."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def distance(self, other: 'AssumptionState') -> float:
        """Euclidean distance in assumption space."""
        return np.linalg.norm(self.to_vector() - other.to_vector())
    
    def interpolate(self, other: 'AssumptionState', t: float) -> 'AssumptionState':
        """Linear interpolation: self * (1-t) + other * t"""
        v = self.to_vector() * (1 - t) + other.to_vector() * t
        return AssumptionState.from_vector(v)
    
    def relaxation_order(self) -> List[str]:
        """
        Order of how relaxed each assumption is.
        Returns dimension names from most relaxed to most strict.
        """
        dims = [
            ('linearity', self.linearity),
            ('independence', self.independence),
            ('stationarity', self.stationarity),
            ('normality', self.normality),
            ('homoscedasticity', self.homoscedasticity),
            ('exogeneity', self.exogeneity),
        ]
        return [name for name, _ in sorted(dims, key=lambda x: x[1])]
    
    def __repr__(self) -> str:
        return (f"AssumptionState(lin={self.linearity:.2f}, ind={self.independence:.2f}, "
                f"stat={self.stationarity:.2f}, norm={self.normality:.2f})")


# =============================================================================
# Path Points and Cliffs
# =============================================================================

@dataclass
class PathPoint:
    """
    A single point along the assumption path.
    
    Contains the assumption state, the resulting estimate, and diagnostics.
    """
    state: AssumptionState
    estimate: float
    std_error: float
    t: float  # Path parameter [0, 1]
    
    # Diagnostics
    r_squared: Optional[float] = None
    mci_score: Optional[float] = None
    log_likelihood: Optional[float] = None
    
    # Curvature at this point (computed later)
    curvature: Optional[float] = None
    
    # Is this point stable?
    is_stable: bool = True
    
    def __repr__(self) -> str:
        return f"PathPoint(t={self.t:.2f}, est={self.estimate:.4f}±{self.std_error:.4f})"


@dataclass
class CliffPoint:
    """
    A detected cliff (assumption breakdown point).
    
    A cliff occurs where small changes in assumptions cause
    large changes in estimates - indicating model fragility.
    """
    t: float  # Path parameter where cliff occurs
    state: AssumptionState
    curvature: float  # How sharp the cliff is
    broken_assumption: str  # Which assumption dimension broke
    
    # Estimates before and after the cliff
    estimate_before: float
    estimate_after: float
    
    @property
    def severity(self) -> float:
        """How severe is this cliff? (0=mild, 1=catastrophic)"""
        jump = abs(self.estimate_after - self.estimate_before)
        return np.clip(jump / (abs(self.estimate_before) + 1e-8), 0, 1)
    
    def __repr__(self) -> str:
        return f"Cliff(t={self.t:.2f}, {self.broken_assumption}, severity={self.severity:.2f})"


# =============================================================================
# Assumption Path
# =============================================================================

class AssumptionPath:
    """
    A path through assumption space.
    
    Traces how model estimates evolve as assumptions are relaxed,
    detecting instabilities and breakdown points.
    
    This is the core of "Model Is a Path" - the idea that a statistical
    model is not a point, but a trajectory through the space of assumptions.
    
    Example:
        >>> path = AssumptionPath()
        >>> path.trace(model, data, steps=20)
        >>> cliffs = path.detect_cliffs()
        >>> print(f"Model breaks at: {cliffs}")
    """
    
    def __init__(
        self,
        start: Optional[AssumptionState] = None,
        end: Optional[AssumptionState] = None,
        relaxation_sequence: Optional[List[str]] = None
    ):
        """
        Initialize an assumption path.
        
        Args:
            start: Starting assumption state (default: classical)
            end: Ending assumption state (default: fully relaxed)
            relaxation_sequence: Order to relax assumptions (e.g., ['linearity', 'normality'])
        """
        self.start = start or AssumptionState.classical()
        self.end = end or AssumptionState.fully_relaxed()
        self.relaxation_sequence = relaxation_sequence
        
        self.points: List[PathPoint] = []
        self.cliffs: List[CliffPoint] = []
    
    def trace(
        self,
        model: Any,
        data: Dict[str, Any],
        steps: int = 20,
        model_factory: Optional[Callable[[AssumptionState], Any]] = None
    ) -> List[PathPoint]:
        """
        Trace the path through assumption space.
        
        For each point along the path, fits a model with the corresponding
        assumption configuration and records the estimate.
        
        Args:
            model: Base model to trace
            data: Data dictionary with 'X', 'y' keys
            steps: Number of points along the path
            model_factory: Optional function to create model for each assumption state
        
        Returns:
            List of PathPoints along the path
        """
        self.points = []
        
        t_values = np.linspace(0, 1, steps)
        
        for i, t in enumerate(t_values):
            # Interpolate assumption state
            state = self.start.interpolate(self.end, t)
            
            try:
                # Get estimate for this assumption state
                if model_factory is not None:
                    current_model = model_factory(state)
                else:
                    current_model = self._configure_model(model, state)
                
                # Fit and extract results
                estimate, std_error, diagnostics = self._fit_and_extract(
                    current_model, state, data
                )
                
                point = PathPoint(
                    state=state,
                    estimate=estimate,
                    std_error=std_error,
                    t=t,
                    r_squared=diagnostics.get('r_squared'),
                    mci_score=diagnostics.get('mci_score'),
                    log_likelihood=diagnostics.get('log_likelihood'),
                    is_stable=std_error < abs(estimate) * 2 if estimate != 0 else True
                )
                self.points.append(point)
                
            except Exception as e:
                # Record failed point (model couldn't fit)
                warnings.warn(f"Failed at t={t:.2f}: {e}")
                self.points.append(PathPoint(
                    state=state,
                    estimate=np.nan,
                    std_error=np.nan,
                    t=t,
                    is_stable=False
                ))
        
        # Compute curvatures
        self._compute_curvatures()
        
        return self.points
    
    def _configure_model(self, model: Any, state: AssumptionState) -> Any:
        """
        Configure a model based on assumption state.
        
        This is a default implementation - can be overridden by subclasses
        or bypassed via model_factory.
        """
        import copy
        configured = copy.deepcopy(model)
        
        # Apply assumption-based configuration
        # (These are heuristics - real implementation would be model-specific)
        
        # Linearity: switch to polynomial/nonparametric
        if hasattr(configured, 'polynomial_degree'):
            configured.polynomial_degree = int(1 + (1 - state.linearity) * 3)
        
        # Normality: switch to robust estimator
        if hasattr(configured, 'robust'):
            configured.robust = state.normality < 0.5
        
        # Heteroscedasticity: use HC standard errors
        if hasattr(configured, 'heteroscedasticity_robust'):
            configured.heteroscedasticity_robust = state.homoscedasticity < 0.5
        
        return configured
    
    def _fit_and_extract(
        self, 
        model: Any, 
        state: AssumptionState,
        data: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Fit the model and extract estimate, std error, and diagnostics.
        
        Returns:
            (estimate, std_error, diagnostics_dict)
        """
        X = data.get('X')
        y = data.get('y')
        
        # Try to fit
        if hasattr(model, 'fit'):
            model.fit(X, y)
        
        # Extract estimate
        estimate = 0.0
        std_error = 0.0
        diagnostics = {}
        
        if hasattr(model, 'coef_'):
            coef = model.coef_
            estimate = coef[0] if hasattr(coef, '__len__') else coef
        elif hasattr(model, 'effect_'):
            estimate = model.effect_
        elif hasattr(model, 'params'):
            estimate = model.params[0] if hasattr(model.params, '__len__') else model.params
        
        if hasattr(model, 'std_error_'):
            std_error = model.std_error_
        elif hasattr(model, 'bse'):
            std_error = model.bse[0] if hasattr(model.bse, '__len__') else model.bse
        else:
            std_error = abs(estimate) * 0.1  # Fallback
        
        if hasattr(model, 'rsquared'):
            diagnostics['r_squared'] = model.rsquared
        elif hasattr(model, 'r_squared_'):
            diagnostics['r_squared'] = model.r_squared_
        
        if hasattr(model, 'mci_score_'):
            diagnostics['mci_score'] = model.mci_score_
        
        if hasattr(model, 'llf'):
            diagnostics['log_likelihood'] = model.llf
        
        return estimate, std_error, diagnostics
    
    def _compute_curvatures(self):
        """Compute curvature at each point along the path."""
        if len(self.points) < 3:
            return
        
        estimates = np.array([p.estimate for p in self.points])
        t_vals = np.array([p.t for p in self.points])
        
        # Handle NaNs
        valid = ~np.isnan(estimates)
        if valid.sum() < 3:
            return
        
        # Compute second derivative (curvature proxy)
        for i in range(1, len(self.points) - 1):
            if valid[i-1] and valid[i] and valid[i+1]:
                dt = t_vals[i+1] - t_vals[i-1]
                if dt > 0:
                    # Second derivative via finite differences
                    d2 = (estimates[i+1] - 2*estimates[i] + estimates[i-1]) / (dt**2)
                    self.points[i].curvature = abs(d2)
    
    def compute_curvature(self) -> np.ndarray:
        """
        Return curvature values along the path.
        
        High curvature = rapid change = potential assumption breakdown.
        """
        return np.array([p.curvature if p.curvature is not None else 0.0 
                        for p in self.points])
    
    def detect_cliffs(self, threshold: float = 0.5) -> List[CliffPoint]:
        """
        Detect cliffs (assumption breakdown points) along the path.
        
        A cliff is where:
        1. Curvature exceeds threshold
        2. Estimate changes significantly
        
        Args:
            threshold: Curvature threshold (relative to max curvature)
        
        Returns:
            List of detected CliffPoints
        """
        self.cliffs = []
        
        if len(self.points) < 3:
            return self.cliffs
        
        curvatures = self.compute_curvature()
        max_curv = curvatures.max() if curvatures.max() > 0 else 1.0
        
        for i in range(1, len(self.points) - 1):
            p = self.points[i]
            
            # Check curvature threshold
            if p.curvature is not None and p.curvature / max_curv > threshold:
                # Determine which assumption changed most
                if i > 0:
                    prev = self.points[i-1].state.to_vector()
                    curr = p.state.to_vector()
                    dim_changes = np.abs(curr - prev)
                    broken_idx = np.argmax(dim_changes)
                    dim_names = ['linearity', 'independence', 'stationarity', 
                                'normality', 'homoscedasticity', 'exogeneity']
                    broken_assumption = dim_names[broken_idx]
                else:
                    broken_assumption = 'unknown'
                
                cliff = CliffPoint(
                    t=p.t,
                    state=p.state,
                    curvature=p.curvature,
                    broken_assumption=broken_assumption,
                    estimate_before=self.points[i-1].estimate if i > 0 else np.nan,
                    estimate_after=self.points[i+1].estimate if i < len(self.points)-1 else np.nan
                )
                self.cliffs.append(cliff)
        
        return self.cliffs
    
    def to_causal_space(self) -> 'CausalSpace':
        """
        Convert the path to a CausalSpace for topological analysis.
        
        Each PathPoint becomes a point in geometric space, enabling:
        - Persistence homology of the path
        - Detection of topological features (loops, voids)
        - Rotor-invariant analysis
        
        Returns:
            CausalSpace with path embedded
        """
        try:
            from statelix_py.core.unified_space import CausalSpace
        except ImportError:
            # Fallback for relative import when running from within package
            from .unified_space import CausalSpace
        
        if not self.points:
            return CausalSpace()
        
        # Create feature matrix from path points
        # Columns: assumption dims + estimate + std_error
        n_points = len(self.points)
        n_dims = 6 + 2  # 6 assumption dims + estimate + std_error
        
        feature_matrix = np.zeros((n_points, n_dims))
        for i, p in enumerate(self.points):
            if np.isnan(p.estimate):
                continue
            feature_matrix[i, :6] = p.state.to_vector()
            feature_matrix[i, 6] = p.estimate
            feature_matrix[i, 7] = p.std_error
        
        # Create adjacency (path edges: i -> i+1)
        adjacency = np.zeros((n_points, n_points))
        for i in range(n_points - 1):
            adjacency[i, i+1] = 1
        
        node_names = [f"t={p.t:.2f}" for p in self.points]
        
        return CausalSpace(
            adjacency=adjacency,
            feature_matrix=feature_matrix,
            node_names=node_names
        )
    
    def path_length(self) -> float:
        """
        Compute the total length of the path in estimate space.
        
        Longer paths indicate more sensitivity to assumptions.
        """
        if len(self.points) < 2:
            return 0.0
        
        estimates = np.array([p.estimate for p in self.points if not np.isnan(p.estimate)])
        if len(estimates) < 2:
            return 0.0
        
        return np.sum(np.abs(np.diff(estimates)))
    
    def stability_score(self) -> float:
        """
        Overall stability score for the path.
        
        Returns:
            Score in [0, 1] where 1 = perfectly stable, 0 = completely unstable
        """
        if not self.points:
            return 1.0
        
        # Factors:
        # 1. Number of cliffs detected
        cliff_penalty = len(self.cliffs) / max(len(self.points), 1)
        
        # 2. Path length variability
        estimates = np.array([p.estimate for p in self.points if not np.isnan(p.estimate)])
        if len(estimates) > 1:
            cv = estimates.std() / (np.abs(estimates).mean() + 1e-8)
            variability_penalty = np.clip(cv, 0, 1)
        else:
            variability_penalty = 0
        
        # 3. Ratio of stable points
        stable_ratio = sum(1 for p in self.points if p.is_stable) / max(len(self.points), 1)
        
        score = stable_ratio * (1 - cliff_penalty * 0.5) * (1 - variability_penalty * 0.3)
        return np.clip(score, 0, 1)
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the path analysis."""
        return {
            'n_points': len(self.points),
            'n_cliffs': len(self.cliffs),
            'path_length': self.path_length(),
            'stability_score': self.stability_score(),
            'most_fragile_assumption': (
                self.cliffs[0].broken_assumption if self.cliffs else None
            ),
            'start_estimate': self.points[0].estimate if self.points else np.nan,
            'end_estimate': self.points[-1].estimate if self.points else np.nan,
        }
    
    def __repr__(self) -> str:
        return (f"AssumptionPath(points={len(self.points)}, "
                f"cliffs={len(self.cliffs)}, "
                f"stability={self.stability_score():.2f})")


# =============================================================================
# Convenience Functions
# =============================================================================

def trace_linearity(model: Any, data: Dict[str, Any], steps: int = 20) -> AssumptionPath:
    """
    Trace a path that relaxes only the linearity assumption.
    """
    start = AssumptionState.classical()
    end = AssumptionState(
        linearity=0.0,
        independence=1.0,
        stationarity=1.0,
        normality=1.0,
        homoscedasticity=1.0,
        exogeneity=1.0
    )
    path = AssumptionPath(start=start, end=end)
    path.trace(model, data, steps=steps)
    return path


def trace_normality(model: Any, data: Dict[str, Any], steps: int = 20) -> AssumptionPath:
    """
    Trace a path that relaxes only the normality assumption.
    """
    start = AssumptionState.classical()
    end = AssumptionState(
        linearity=1.0,
        independence=1.0,
        stationarity=1.0,
        normality=0.0,
        homoscedasticity=1.0,
        exogeneity=1.0
    )
    path = AssumptionPath(start=start, end=end)
    path.trace(model, data, steps=steps)
    return path


def trace_full_relaxation(model: Any, data: Dict[str, Any], steps: int = 20) -> AssumptionPath:
    """
    Trace a path from classical to fully relaxed assumptions.
    """
    path = AssumptionPath()
    path.trace(model, data, steps=steps)
    return path
