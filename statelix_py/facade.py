
from typing import Type, Any, Dict
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
