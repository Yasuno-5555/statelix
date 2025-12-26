
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


def governance_report(
    model: Optional[BaseEstimator] = None,
    X = None,
    y = None,
    tube: Optional['AssumptionTube'] = None,
    path: Optional['AssumptionPath'] = None,
    **context
):
    """
    Generate comprehensive governance report.
    
    Combines:
    - One-Way Door Analysis: irreversible assumption relaxations
    - Claim Budget: maximum claim strength allowed
    - The Last Honest Sentence: the one thing you can truthfully say
    
    Args:
        model: Optional model for analysis
        X: Optional feature matrix
        y: Optional target
        tube: Pre-computed AssumptionTube
        path: Pre-computed AssumptionPath
        **context: Additional context (effect_name, target_name, etc.)
    
    Returns:
        GovernanceReport with verdict, honest sentence, and warnings
        
    Example:
        >>> report = governance_report(model, X, y)
        >>> 
        >>> print(f"Verdict: {report.verdict}")
        >>> print(f"Honest Sentence: {report.honest_sentence.sentence}")
        >>> print(report.to_markdown())
    """
    from .core.governance_report import GovernanceReportGenerator
    
    # Generate tube if needed
    if tube is None and model is not None and X is not None and y is not None:
        tube = analyze_robustness(model, X, y, detect_cliffs=True)
        if tube.base_path:
            path = tube.base_path
    
    gen = GovernanceReportGenerator()
    return gen.generate(tube=tube, path=path, context=context)


def get_honest_sentence(
    tube: Optional['AssumptionTube'] = None,
    path: Optional['AssumptionPath'] = None,
    **context
) -> str:
    """
    Get the single honest sentence for your analysis.
    
    This is the ONE thing you can truthfully claim.
    
    Args:
        tube: Pre-computed AssumptionTube
        path: Pre-computed AssumptionPath  
        **context: Additional context
    
    Returns:
        The honest sentence as a string
    """
    from .core.honest_sentence import generate_honest_sentence
    honest = generate_honest_sentence(tube, path, **context)
    return honest.sentence


def simulate_truth_collapse(
    model: Optional[BaseEstimator] = None,
    X = None,
    y = None,
    tube: Optional['AssumptionTube'] = None,
    destruction_order: Optional[list] = None
):
    """
    Truth Collapse Simulator: How does your conclusion die?
    
    Systematically destroys assumptions to reveal:
    - The order in which truth collapses
    - When claims become unsustainable  
    - The archetype of your analysis's death
    
    Args:
        model: Optional model for analysis
        X: Optional feature matrix
        y: Optional target
        tube: Pre-computed AssumptionTube
        destruction_order: Custom assumption destruction order
    
    Returns:
        CollapseReport with schedule, archetype, and timeline
        
    Example:
        >>> report = simulate_truth_collapse(model, X, y)
        >>> 
        >>> print(f"{report.schedule.archetype.icon} {report.schedule.archetype.description}")
        >>> print(f"Death at stage: {report.schedule.death_stage}")
        >>> print(f"Critical assumption: {report.schedule.critical_assumption}")
    
    Archetypes:
    - 🗼 Glass Tower: Strong initially, shatters instantly
    - ☁️ Marshmallow: Never strong, survives but says nothing  
    - 🔩 Steel Rod: Survives long, thin claims
    - 🐯 Paper Tiger: Looks strong, tube is paper-thin
    """
    from .core.collapse_simulator import TruthCollapseSimulator
    
    # Generate tube if needed
    if tube is None and model is not None and X is not None and y is not None:
        tube = analyze_robustness(model, X, y, detect_cliffs=True)
    
    sim = TruthCollapseSimulator(destruction_order=destruction_order)
    return sim.simulate(tube=tube)


def compile_claim_text(
    governance_report: Optional['GovernanceReport'] = None,
    dialect: str = "academic",
    effect_name: str = "the effect",
    target_name: str = "the outcome",
    model: Optional[BaseEstimator] = None,
    X = None,
    y = None
):
    """
    Claim Language Compiler: Project claim space to language.
    
    Not a text generator - a controlled projection that respects
    mathematical constraints on what can be said.
    
    Args:
        governance_report: Pre-computed GovernanceReport
        dialect: "academic", "referee", "policy_averse", "exploratory"
        effect_name: Name of the effect
        target_name: Name of the outcome
        model: Optional model (for auto-generating report)
        X: Optional feature matrix
        y: Optional target
    
    Returns:
        CompiledClaim with text (or rejection reason)
        
    Example:
        >>> claim = compile_claim_text(report, dialect="referee")
        >>> 
        >>> if claim.is_valid:
        ...     print(claim.text)
        >>> else:
        ...     print(f"Cannot claim: {claim.rejection_reason}")
    
    Dialects:
    - academic: Conservative, full caveats
    - referee: Ultra-safe for peer review
    - policy_averse: No policy implications
    - exploratory: For notebooks, less formal
    """
    from .core.claim_compiler import ClaimCompiler, Dialect, compile_claim
    
    dialect_map = {
        "academic": Dialect.ACADEMIC_CONSERVATIVE,
        "referee": Dialect.REFEREE_SAFE,
        "policy_averse": Dialect.POLICY_AVERSE,
        "exploratory": Dialect.EXPLORATORY_NOTEBOOK,
    }
    
    d = dialect_map.get(dialect.lower(), Dialect.ACADEMIC_CONSERVATIVE)
    
    # Generate report if needed
    if governance_report is None and model is not None and X is not None and y is not None:
        governance_report = governance_report_func(model, X, y)
    
    if governance_report is None:
        # Minimal report
        governance_report = governance_report_func()
    
    return compile_claim(governance_report, d, effect_name, target_name)


def simulate_reviewer_attack(claim, effect_name: str = "the effect"):
    """
    Reviewer Attack Simulator: How will your claims be destroyed?
    
    Simulates hostile reviewer questioning.
    
    Args:
        claim: CompiledClaim to attack
        effect_name: Name of the effect
    
    Returns:
        AttackReport with survival probability and vulnerabilities
    """
    from .core.claim_compiler import ClaimIR
    from .core.reviewer_attack import ReviewerAttackSimulator
    
    ir = claim.ir if hasattr(claim, 'ir') else ClaimIR(
        strength=claim.strength if hasattr(claim, 'strength') else None,
        robustness_score=0.5,
        nature=None,
        scope=None,
        effect_name=effect_name,
        target_name="outcome"
    )
    
    sim = ReviewerAttackSimulator()
    return sim.attack(claim, ir)


# Alias for governance_report to avoid collision
governance_report_func = governance_report


def evaluate_responsibility(
    claim_text: str = "",
    effect_name: str = "the effect",
    target_name: str = "the outcome",
    robustness_score: float = 0.5,
    context: Optional[str] = None
):
    """
    Responsibility Sink: Where claims fall before they reach the world.
    
    Evaluates whether a claim should be made based on impact,
    not just whether it CAN be made.
    
    Args:
        claim_text: The text being evaluated
        effect_name: Name of the effect
        target_name: Name of the outcome
        robustness_score: Robustness of the analysis
        context: Additional context
    
    Returns:
        ResponsibilityBudget with verdict
        
    Example:
        >>> budget = evaluate_responsibility(
        ...     effect_name="criminal prediction",
        ...     target_name="individual"
        ... )
        >>> 
        >>> if not budget.can_proceed:
        ...     print(f"Cannot claim: {budget.rejection_reason}")
    
    Impact Classes:
    - academic_only: Low bar
    - policy_triggering: Requires 0.6+ robustness
    - individual_targeting: Requires 0.95+ (usually rejected)
    """
    from .core.claim_compiler import ClaimIR, ClaimNature, ClaimScope
    from .core.claim_budget import ClaimStrength
    from .core.responsibility_sink import ResponsibilitySink
    
    # Create minimal IR
    strength = ClaimStrength.MODERATE if robustness_score > 0.4 else ClaimStrength.WEAK
    
    ir = ClaimIR(
        strength=strength,
        robustness_score=robustness_score,
        nature=ClaimNature.ASSOCIATIVE,
        scope=ClaimScope.SAMPLE_ONLY,
        effect_name=effect_name,
        target_name=target_name
    )
    
    sink = ResponsibilitySink()
    return sink.evaluate(ir, context=context)


def generate_responsible_claim(
    governance_rep = None,
    effect_name: str = "the effect",
    target_name: str = "the outcome",
    model: Optional[BaseEstimator] = None,
    X = None,
    y = None
):
    """
    Generate the Last Responsible Sentence.
    
    Minimizes harm while preserving truth.
    Adds ethical disclaimers where necessary.
    
    Args:
        governance_rep: GovernanceReport
        effect_name: Name of the effect
        target_name: Name of the outcome
        model: Optional model
        X: Optional features
        y: Optional target
    
    Returns:
        ResponsibleClaim with final text and modifications
    """
    from .core.claim_compiler import compile_claim, Dialect
    from .core.responsibility_sink import ResponsibilitySink
    
    # Get governance report
    if governance_rep is None:
        if model is not None and X is not None and y is not None:
            governance_rep = governance_report_func(model, X, y)
        else:
            governance_rep = governance_report_func()
    
    # Compile claim
    claim = compile_claim(governance_rep, Dialect.ACADEMIC_CONSERVATIVE, effect_name, target_name)
    
    # Generate responsible version
    from .core.claim_compiler import ClaimIR
    ir = ClaimIR.from_governance_report(governance_rep, effect_name, target_name)
    
    sink = ResponsibilitySink()
    return sink.generate_responsible_claim(ir, claim)
