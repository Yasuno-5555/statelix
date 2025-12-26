"""
Assumption Tube: Truth Is a Tube

A single path through assumption space is not enough.
Multiple paths, generated via bootstrap/noise, form a "tube" that captures
the thickness and robustness of our conclusions.

Key insight: How thick is the tube = how stable is the truth.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, TYPE_CHECKING
from enum import Enum
import warnings
import copy

if TYPE_CHECKING:
    from statelix_py.core.assumption_path import AssumptionPath, AssumptionState, PathPoint, CliffPoint


# =============================================================================
# Tube Metrics
# =============================================================================

@dataclass
class TubeMetrics:
    """
    Robustness metrics computed from an assumption tube.
    """
    robustness_radius: float        # Mean distance from center to boundary
    truth_thickness: float          # Minimum tube diameter
    truth_thickness_location: float # Path parameter t where thickness is minimal
    total_variance: float           # Total variance across all paths
    brittleness_index: Dict[str, float] = field(default_factory=dict)
    self_intersection_count: int = 0
    
    def stability_rating(self) -> str:
        """Human-readable stability rating."""
        if self.robustness_radius > 0.5 and self.truth_thickness > 0.3:
            return "ROBUST"
        elif self.robustness_radius > 0.2 or self.truth_thickness > 0.1:
            return "MODERATE"
        else:
            return "FRAGILE"
    
    def most_brittle_assumption(self) -> Optional[str]:
        """Which assumption is most dangerous?"""
        if not self.brittleness_index:
            return None
        return max(self.brittleness_index.items(), key=lambda x: x[1])[0]
    
    def __repr__(self) -> str:
        return (f"TubeMetrics(radius={self.robustness_radius:.3f}, "
                f"thickness={self.truth_thickness:.3f}, "
                f"rating={self.stability_rating()})")


@dataclass
class TubeCrossSection:
    """
    A cross-section of the tube at a specific t value.
    """
    t: float
    estimates: np.ndarray  # Estimates from all paths at this t
    center: float          # Central estimate (median)
    radius: float          # Distance from center to boundary
    diameter: float        # Full width (2 * radius-ish, but accounts for asymmetry)
    std: float             # Standard deviation
    
    @property
    def is_thin(self) -> bool:
        """Is this cross-section dangerously thin?"""
        return self.diameter < 0.1 * abs(self.center) if self.center != 0 else self.diameter < 0.01


# =============================================================================
# Assumption Tube
# =============================================================================

class AssumptionTube:
    """
    A collection of assumption paths forming a "tube" in estimate space.
    
    The tube captures the robustness of conclusions:
    - Thick tube = robust, conclusions survive perturbation
    - Thin tube = fragile, small changes flip conclusions
    - Self-intersecting tube = qualitative instability
    
    Generation methods:
    - Bootstrap: resample data, trace paths
    - Noise injection: perturb assumption values
    - Multiple starts: vary initial conditions
    
    Example:
        >>> tube = AssumptionTube(base_path)
        >>> tube.generate_bootstrap_paths(data, n_samples=100)
        >>> metrics = tube.compute_metrics()
        >>> print(f"Robustness: {metrics.robustness_radius:.3f}")
        >>> print(f"Most brittle: {metrics.most_brittle_assumption()}")
    """
    
    def __init__(self, base_path: Optional['AssumptionPath'] = None):
        """
        Initialize a tube from a base path.
        
        Args:
            base_path: The central path around which the tube is built
        """
        self.base_path = base_path
        self.paths: List['AssumptionPath'] = []
        if base_path is not None:
            self.paths.append(base_path)
        
        self._metrics_cache: Optional[TubeMetrics] = None
        self._cross_sections_cache: Optional[List[TubeCrossSection]] = None
    
    def generate_bootstrap_paths(
        self,
        model: Any,
        data: Dict[str, Any],
        n_samples: int = 50,
        steps: int = 20,
        model_factory: Optional[Callable] = None,
        random_state: Optional[int] = None
    ) -> 'AssumptionTube':
        """
        Generate paths from bootstrap resamples of the data.
        
        Each bootstrap sample creates a different path through assumption space,
        revealing how data uncertainty propagates to conclusions.
        
        Args:
            model: Base model for path tracing
            data: Original data dict with 'X' and 'y'
            n_samples: Number of bootstrap samples
            steps: Steps per path
            model_factory: Optional model factory for path tracing
            random_state: Random seed for reproducibility
        
        Returns:
            self for chaining
        """
        try:
            from .assumption_path import AssumptionPath
        except ImportError:
            from statelix_py.core.assumption_path import AssumptionPath
        
        if random_state is not None:
            np.random.seed(random_state)
        
        X = data.get('X')
        y = data.get('y')
        
        if X is None or y is None:
            warnings.warn("Data must contain 'X' and 'y' for bootstrap")
            return self
        
        n = len(y)
        
        for i in range(n_samples):
            # Bootstrap resample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx] if hasattr(X, '__getitem__') else X
            y_boot = y[idx] if hasattr(y, '__getitem__') else y
            
            boot_data = {'X': X_boot, 'y': y_boot}
            
            try:
                path = AssumptionPath(
                    start=self.base_path.start if self.base_path else None,
                    end=self.base_path.end if self.base_path else None
                )
                path.trace(model, boot_data, steps=steps, model_factory=model_factory)
                self.paths.append(path)
            except Exception as e:
                warnings.warn(f"Bootstrap path {i} failed: {e}")
                continue
        
        self._invalidate_cache()
        return self
    
    def generate_noise_paths(
        self,
        model: Any,
        data: Dict[str, Any],
        noise_scale: float = 0.05,
        n_samples: int = 50,
        steps: int = 20,
        model_factory: Optional[Callable] = None,
        random_state: Optional[int] = None
    ) -> 'AssumptionTube':
        """
        Generate paths with noise injected into assumption values.
        
        This tests how robust conclusions are to slightly different
        interpretations of "how linear" or "how independent" the data is.
        
        Args:
            model: Base model
            data: Data dict
            noise_scale: Standard deviation of noise added to assumptions
            n_samples: Number of noisy paths
            steps: Steps per path
            model_factory: Optional model factory
            random_state: Random seed
        
        Returns:
            self for chaining
        """
        try:
            from .assumption_path import AssumptionPath, AssumptionState
        except ImportError:
            from statelix_py.core.assumption_path import AssumptionPath, AssumptionState
        
        if random_state is not None:
            np.random.seed(random_state)
        
        base_start = self.base_path.start if self.base_path else AssumptionState.classical()
        base_end = self.base_path.end if self.base_path else AssumptionState.fully_relaxed()
        
        for i in range(n_samples):
            # Add noise to start and end states
            noise_start = np.random.normal(0, noise_scale, 6)
            noise_end = np.random.normal(0, noise_scale, 6)
            
            noisy_start = AssumptionState.from_vector(
                np.clip(base_start.to_vector() + noise_start, 0, 1)
            )
            noisy_end = AssumptionState.from_vector(
                np.clip(base_end.to_vector() + noise_end, 0, 1)
            )
            
            try:
                path = AssumptionPath(start=noisy_start, end=noisy_end)
                path.trace(model, data, steps=steps, model_factory=model_factory)
                self.paths.append(path)
            except Exception as e:
                warnings.warn(f"Noise path {i} failed: {e}")
                continue
        
        self._invalidate_cache()
        return self
    
    def compute_cross_sections(self) -> List[TubeCrossSection]:
        """
        Compute cross-sections of the tube at each t value.
        
        Returns:
            List of TubeCrossSection objects
        """
        if self._cross_sections_cache is not None:
            return self._cross_sections_cache
        
        if not self.paths or len(self.paths) < 2:
            return []
        
        # Get common t values from base path
        if self.base_path and self.base_path.points:
            t_values = [p.t for p in self.base_path.points]
        else:
            t_values = [p.t for p in self.paths[0].points]
        
        cross_sections = []
        
        for t_idx, t in enumerate(t_values):
            estimates = []
            
            for path in self.paths:
                if t_idx < len(path.points):
                    est = path.points[t_idx].estimate
                    if not np.isnan(est):
                        estimates.append(est)
            
            if len(estimates) < 2:
                continue
            
            estimates = np.array(estimates)
            center = np.median(estimates)
            
            # Compute radius as mean absolute deviation from center
            radius = np.mean(np.abs(estimates - center))
            
            # Diameter accounts for asymmetry (uses percentiles)
            q05, q95 = np.percentile(estimates, [5, 95])
            diameter = q95 - q05
            
            cross_sections.append(TubeCrossSection(
                t=t,
                estimates=estimates,
                center=center,
                radius=radius,
                diameter=diameter,
                std=np.std(estimates)
            ))
        
        self._cross_sections_cache = cross_sections
        return cross_sections
    
    def compute_metrics(self) -> TubeMetrics:
        """
        Compute all robustness metrics for the tube.
        
        Returns:
            TubeMetrics object with all computed metrics
        """
        if self._metrics_cache is not None:
            return self._metrics_cache
        
        cross_sections = self.compute_cross_sections()
        
        if not cross_sections:
            return TubeMetrics(
                robustness_radius=0.0,
                truth_thickness=0.0,
                truth_thickness_location=0.0,
                total_variance=0.0
            )
        
        # Robustness Radius: mean distance from center to boundary
        radii = [cs.radius for cs in cross_sections]
        robustness_radius = np.mean(radii) if radii else 0.0
        
        # Truth Thickness: minimum diameter (where tube is thinnest)
        diameters = [cs.diameter for cs in cross_sections]
        min_diameter_idx = np.argmin(diameters)
        truth_thickness = diameters[min_diameter_idx]
        truth_thickness_location = cross_sections[min_diameter_idx].t
        
        # Total variance
        all_estimates = np.concatenate([cs.estimates for cs in cross_sections])
        total_variance = np.var(all_estimates)
        
        # Brittleness Index: per-assumption sensitivity
        brittleness = self._compute_brittleness_index()
        
        # Self-intersections
        intersection_count = self._count_self_intersections()
        
        self._metrics_cache = TubeMetrics(
            robustness_radius=robustness_radius,
            truth_thickness=truth_thickness,
            truth_thickness_location=truth_thickness_location,
            total_variance=total_variance,
            brittleness_index=brittleness,
            self_intersection_count=intersection_count
        )
        
        return self._metrics_cache
    
    def _compute_brittleness_index(self) -> Dict[str, float]:
        """
        Compute per-assumption brittleness.
        
        For each assumption dimension, measures how much variance in estimates
        is explained by variation in that assumption value.
        """
        if len(self.paths) < 3:
            return {}
        
        dimensions = ['linearity', 'independence', 'stationarity', 
                     'normality', 'homoscedasticity', 'exogeneity']
        brittleness = {}
        
        # Collect all (assumption_value, estimate) pairs for each dimension
        for dim in dimensions:
            dim_values = []
            estimates = []
            
            for path in self.paths:
                for point in path.points:
                    if np.isnan(point.estimate):
                        continue
                    dim_val = getattr(point.state, dim, 0.5)
                    dim_values.append(dim_val)
                    estimates.append(point.estimate)
            
            if len(estimates) < 3:
                brittleness[dim] = 0.0
                continue
            
            dim_values = np.array(dim_values)
            estimates = np.array(estimates)
            
            # Brittleness = correlation between assumption value and estimate variance
            # Higher correlation = more brittle (estimate depends on assumption)
            try:
                # Use absolute correlation as brittleness
                corr = np.abs(np.corrcoef(dim_values, estimates)[0, 1])
                brittleness[dim] = corr if not np.isnan(corr) else 0.0
            except:
                brittleness[dim] = 0.0
        
        return brittleness
    
    def _count_self_intersections(self) -> int:
        """
        Count how many times paths cross each other.
        
        Crossings indicate qualitative instability - small changes
        can flip which estimate is larger.
        """
        if len(self.paths) < 2:
            return 0
        
        cross_sections = self.compute_cross_sections()
        if len(cross_sections) < 2:
            return 0
        
        crossings = 0
        
        # For each pair of paths, count sign changes in their difference
        for i in range(len(self.paths)):
            for j in range(i + 1, min(i + 5, len(self.paths))):  # Limit comparisons
                path_i = self.paths[i]
                path_j = self.paths[j]
                
                min_len = min(len(path_i.points), len(path_j.points))
                if min_len < 2:
                    continue
                
                diffs = []
                for k in range(min_len):
                    est_i = path_i.points[k].estimate
                    est_j = path_j.points[k].estimate
                    if not np.isnan(est_i) and not np.isnan(est_j):
                        diffs.append(est_i - est_j)
                
                if len(diffs) < 2:
                    continue
                
                # Count sign changes
                signs = np.sign(diffs)
                crossings += np.sum(np.abs(np.diff(signs)) == 2)
        
        return crossings
    
    def robustness_radius(self) -> float:
        """Get robustness radius metric."""
        return self.compute_metrics().robustness_radius
    
    def truth_thickness(self) -> float:
        """Get truth thickness metric."""
        return self.compute_metrics().truth_thickness
    
    def assumption_brittleness_index(self) -> Dict[str, float]:
        """Get per-assumption brittleness scores."""
        return self.compute_metrics().brittleness_index
    
    def self_intersection_points(self) -> int:
        """Get count of path crossings."""
        return self.compute_metrics().self_intersection_count
    
    def get_envelope(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the tube envelope for visualization.
        
        Returns:
            (t_values, lower_bound, upper_bound)
        """
        cross_sections = self.compute_cross_sections()
        
        if not cross_sections:
            return np.array([]), np.array([]), np.array([])
        
        t_vals = np.array([cs.t for cs in cross_sections])
        centers = np.array([cs.center for cs in cross_sections])
        radii = np.array([cs.radius for cs in cross_sections])
        
        return t_vals, centers - radii, centers + radii
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the tube analysis."""
        metrics = self.compute_metrics()
        
        return {
            'n_paths': len(self.paths),
            'robustness_radius': metrics.robustness_radius,
            'truth_thickness': metrics.truth_thickness,
            'thickness_location': metrics.truth_thickness_location,
            'stability_rating': metrics.stability_rating(),
            'most_brittle': metrics.most_brittle_assumption(),
            'self_intersections': metrics.self_intersection_count,
            'brittleness_index': metrics.brittleness_index
        }
    
    def _invalidate_cache(self):
        """Invalidate cached computations."""
        self._metrics_cache = None
        self._cross_sections_cache = None
    
    def __repr__(self) -> str:
        if self._metrics_cache:
            return f"AssumptionTube(paths={len(self.paths)}, {self._metrics_cache.stability_rating()})"
        return f"AssumptionTube(paths={len(self.paths)})"


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_robustness_tube(
    model: Any,
    data: Dict[str, Any],
    n_bootstrap: int = 30,
    n_noise: int = 20,
    noise_scale: float = 0.05,
    steps: int = 15,
    model_factory: Optional[Callable] = None,
    random_state: Optional[int] = None
) -> AssumptionTube:
    """
    Generate a complete robustness tube with bootstrap and noise paths.
    
    Args:
        model: Base model to analyze
        data: Data dict with 'X' and 'y'
        n_bootstrap: Number of bootstrap paths
        n_noise: Number of noise-injected paths
        noise_scale: Noise magnitude for assumption perturbation
        steps: Steps per path
        model_factory: Optional model factory
        random_state: Random seed
    
    Returns:
        AssumptionTube with all paths generated and metrics computed
    
    Example:
        >>> tube = generate_robustness_tube(model, {'X': X, 'y': y})
        >>> print(f"Verdict: {tube.compute_metrics().stability_rating()}")
    """
    try:
        from .assumption_path import AssumptionPath
    except ImportError:
        from statelix_py.core.assumption_path import AssumptionPath
    
    # Generate base path
    base_path = AssumptionPath()
    base_path.trace(model, data, steps=steps, model_factory=model_factory)
    base_path.detect_cliffs()
    
    # Create tube
    tube = AssumptionTube(base_path)
    
    # Generate paths
    tube.generate_bootstrap_paths(
        model, data, 
        n_samples=n_bootstrap, 
        steps=steps,
        model_factory=model_factory,
        random_state=random_state
    )
    
    tube.generate_noise_paths(
        model, data,
        noise_scale=noise_scale,
        n_samples=n_noise,
        steps=steps,
        model_factory=model_factory,
        random_state=random_state + 1000 if random_state else None
    )
    
    return tube
