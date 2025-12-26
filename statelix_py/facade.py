
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

