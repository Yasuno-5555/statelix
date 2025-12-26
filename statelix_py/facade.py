
from typing import Type, Any, Dict, Optional, Callable
from sklearn.base import BaseEstimator
from .diagnostics.presets import GovernanceMode
from .diagnostics.critic import ModelRejectedError
from .utils.report_generator import ReportGenerator

def fit_and_judge(model_cls: Type[BaseEstimator], X, y, 
                  mode: GovernanceMode = GovernanceMode.STRICT, 
                  save_report_on_refusal: bool = False,
                  **kwargs) -> BaseEstimator:
    """
    One-Line API for governed modeling.
    Instantiates, fits, and validates the model.
    Raises ModelRejectedError if the model fails governance checks.
    
    Args:
        model_cls: The model class (e.g. StatelixOLS)
        X: Features
        y: Target
        mode: Governance Strictness (default: STRICT)
        save_report_on_refusal: If True, saves an HTML report when rejected.
        **kwargs: Parameters for model instantiation
        
    Returns:
        Fitted model instance if accepted.
    """
    # Create model
    try:
        model = model_cls(mode=mode, **kwargs)
    except TypeError:
        # Fallback for models that don't support 'mode' (legacy/external)
        # But for Statelix models we expect it.
        model = model_cls(**kwargs)
        
    try:
        model.fit(X, y)
        return model
    except ModelRejectedError as e:
        if save_report_on_refusal:
            report = ReportGenerator.create_refusal_report(e.diagnostics)
            report.save("refusal_report.html")
            print("Refusal report saved to 'refusal_report.html'")
        raise e


def trace_assumptions(
    model: BaseEstimator,
    data: Dict[str, Any],
    steps: int = 20,
    model_factory: Optional[Callable] = None,
    detect_cliffs: bool = True,
    cliff_threshold: float = 0.3
):
    """
    Model Is a Path: Trace assumptions and find where they break.
    
    This function traces a path through assumption space, showing how
    the model estimate changes as classical assumptions are relaxed.
    
    Args:
        model: The base model to analyze
        data: Data dictionary with 'X' and 'y' keys
        steps: Number of points along the path (default: 20)
        model_factory: Optional function to create model for each assumption state.
                      Signature: (AssumptionState) -> model_instance
        detect_cliffs: Whether to detect assumption breakdown points
        cliff_threshold: Threshold for cliff detection (0-1)
    
    Returns:
        AssumptionPath: Path object with analysis results
        
    Example:
        >>> from statelix import trace_assumptions
        >>> from statelix.models import StatelixOLS
        >>> 
        >>> model = StatelixOLS()
        >>> path = trace_assumptions(model, {'X': X, 'y': y})
        >>> 
        >>> print(f"Stability: {path.stability_score():.2f}")
        >>> print(f"Cliffs detected: {len(path.cliffs)}")
        >>> for cliff in path.cliffs:
        ...     print(f"  - {cliff.broken_assumption} breaks at t={cliff.t:.2f}")
    
    The path object can be:
    - Visualized with PathWidget in the GUI
    - Converted to CausalSpace for topological analysis
    - Summarized for reporting
    """
    from .core.assumption_path import AssumptionPath
    
    path = AssumptionPath()
    path.trace(model, data, steps=steps, model_factory=model_factory)
    
    if detect_cliffs:
        path.detect_cliffs(threshold=cliff_threshold)
    
    return path


def explore_model_path(
    model: BaseEstimator,
    X,
    y,
    show_gui: bool = True
):
    """
    Convenience function to explore a model's assumption path.
    
    Traces the path and optionally shows interactive visualization.
    
    Args:
        model: The model to analyze
        X: Feature matrix
        y: Target vector
        show_gui: Whether to show the interactive GUI (default: True)
    
    Returns:
        AssumptionPath or (AssumptionPath, widget) if show_gui=True
    """
    data = {'X': X, 'y': y}
    path = trace_assumptions(model, data)
    
    if show_gui:
        try:
            from .gui.widgets.path_widget import AssumptionPathWidget
            from PySide6.QtWidgets import QApplication
            import sys
            
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            widget = AssumptionPathWidget()
            widget.set_path(path)
            widget.show()
            
            return path, widget
        except ImportError:
            print("GUI not available. Returning path only.")
            return path
    
    return path


def analyze_robustness(
    model: BaseEstimator,
    X,
    y,
    n_bootstrap: int = 30,
    n_noise: int = 20,
    noise_scale: float = 0.05,
    detect_cliffs: bool = True,
    random_state: Optional[int] = None
):
    """
    Truth Is a Tube: Comprehensive robustness analysis.
    
    Generates multiple paths through assumption space via bootstrap
    and noise injection, measuring how thick/thin the "tube of truth" is.
    
    Args:
        model: Base model to analyze
        X: Feature matrix
        y: Target vector
        n_bootstrap: Number of bootstrap resamples
        n_noise: Number of noise-perturbed paths
        noise_scale: Magnitude of assumption noise
        detect_cliffs: Whether to detect and record cliffs
        random_state: Random seed for reproducibility
    
    Returns:
        AssumptionTube: Tube object with robustness metrics
        
    Example:
        >>> tube = analyze_robustness(model, X, y)
        >>> metrics = tube.compute_metrics()
        >>> print(f"Robustness: {metrics.robustness_radius:.3f}")
        >>> print(f"Truth Thickness: {metrics.truth_thickness:.3f}")
        >>> print(f"Verdict: {metrics.stability_rating()}")
        >>> print(f"Most brittle: {metrics.most_brittle_assumption()}")
    
    Interpretation:
    - Robustness Radius > 0.5: Conclusions survive data perturbation
    - Truth Thickness > 0.3: Conclusions are thick, not razor-thin
    - FRAGILE rating: Small changes flip conclusions - be worried
    """
    from .core.assumption_tube import generate_robustness_tube
    from .core.cliff_learner import get_cliff_learner
    
    data = {'X': X, 'y': y}
    
    tube = generate_robustness_tube(
        model, data,
        n_bootstrap=n_bootstrap,
        n_noise=n_noise,
        noise_scale=noise_scale,
        random_state=random_state
    )
    
    # Record cliffs for learning
    if detect_cliffs:
        learner = get_cliff_learner()
        for path in tube.paths:
            path.detect_cliffs()
            for cliff in path.cliffs:
                learner.record_cliff(cliff, context={
                    'model': model.__class__.__name__,
                    'n': len(y) if hasattr(y, '__len__') else 0
                })
    
    return tube


def get_cliff_warning(state=None):
    """
    Get proactive warning based on learned cliff patterns.
    
    Args:
        state: Optional AssumptionState to check (default: classical)
    
    Returns:
        Warning message or None if low risk
    """
    from .core.assumption_path import AssumptionState
    from .core.cliff_learner import get_cliff_learner
    
    if state is None:
        state = AssumptionState.classical()
    
    learner = get_cliff_learner()
    return learner.warning_message(state)
