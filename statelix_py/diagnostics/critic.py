

import enum
from typing import Dict, Any, List, Optional
from .mci import ModelCredibilityIndex, MCIScore
from .translator import ReasonTranslator
from .presets import GovernanceMode, GovernancePreset

class CriticMode(enum.Enum):
    DIAGNOSTIC = "Diagnostic"
    SUGGESTIVE = "Suggestive"
    AUTONOMOUS = "Autonomous"

class DiagnosticReport:
    def __init__(self, mci: MCIScore, messages: List[str], suggestions: List[str] = None):
        self.mci = mci
        self.messages = messages
        self.suggestions = suggestions or []

    def __str__(self):
        lines = [f"--- Statelix Diagnostic Report ---"]
        lines.append(f"Credibility Score: {self.mci.total_score:.2f} / 1.00")
        lines.append(f"  > Fit: {self.mci.fit_score:.2f}")
        lines.append(f"  > Topology: {self.mci.topology_score:.2f}")
        lines.append(f"  > Geometry: {self.mci.geometry_score:.2f}")
        
        if self.messages:
            lines.append("\nObjections:")
            for msg in self.messages:
                lines.append(f"  - {msg}")
                
        if self.suggestions:
            lines.append("\nSuggestions:")
            for sugg in self.suggestions:
                lines.append(f"  * {sugg}")
                
        return "\n".join(lines)

class ModelRejectedError(Exception):
    """Raised when the model fails to meet the minimum credibility standards of Statelix."""
    def __init__(self, message, diagnostics: Dict[str, Any] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

class ModelCritic:
    """
    The 'Critic' engine that audits models and provides human-centric feedback.
    """
    
    def __init__(self, mode: CriticMode = CriticMode.DIAGNOSTIC, 
                 governance_mode: GovernanceMode = GovernanceMode.STRICT):
        self.mode = mode
        self.strict_threshold = GovernancePreset.get_threshold(governance_mode)
        self.mci_calc = ModelCredibilityIndex()
        self.translator = ReasonTranslator()
        
    def critique(self, metrics: Dict[str, Any]) -> DiagnosticReport:
        """
        Audits the model based on provided metrics.
        """
        
        # 1. Calc MCI
        fit_keys = ['r2', 'rmse', 'log_likelihood']
        topo_keys = ['mean_structure', 'std_structure', 'topology_jump']
        geo_keys = ['invariant_ratio']
        
        fit_m = {k: v for k, v in metrics.items() if k in fit_keys}
        topo_m = {k: v for k, v in metrics.items() if k in topo_keys}
        geo_m = {k: v for k, v in metrics.items() if k in geo_keys}
        
        mci = self.mci_calc.calculate(fit_m, topo_m, geo_m)
        messages = self.translator.translate(metrics)
        
        # Prepare diagnostics dict for potential rejection
        diag_data = {
            'mci': mci.total_score,
            'objections_list': messages,
            'suggestions': [], # Calculated later if allowed, or empty
            'history': []
        }
        
        # Check Veto Power (STRICT MODE)
        if self.strict_threshold > 0:
            if mci.total_score < self.strict_threshold:
                raise ModelRejectedError(
                    f"Model rejected by Statelix (MCI {mci.total_score:.2f} < {self.strict_threshold}).\n"
                    f"Reasons: {messages}",
                    diagnostics=diag_data
                )
            # Special Checks: Topology Collapse is fatal
            if metrics.get('topology_jump', False):
                diag_data['objections_list'].append("Topology Collapse detected")
                raise ModelRejectedError(
                    "Model rejected: Structural instability (Topology Collapse) detected.",
                    diagnostics=diag_data
                )
        
        suggestions = []
        if self.mode in [CriticMode.SUGGESTIVE, CriticMode.AUTONOMOUS]:
            # Generate suggestions based on messages/metrics
            if mci.fit_score < 0.6:
                suggestions.append("Try adding interaction terms (e.g. X1*X2).")
            if mci.topology_score < 0.5:
                suggestions.append("Data might be too noisy. Try smoothing or increasing regularization (prioir variance).")
            if mci.geometry_score < 0.8:
                suggestions.append("Normalize your input features (Z-score) to improve geometric robustness.")

        return DiagnosticReport(mci, messages, suggestions)

    def categorize_objection(self, message: str) -> str:
        """Categorizes a message for tree view display."""
        msg_l = message.lower()
        if "fit" in msg_l or "r²" in msg_l or "variance" in msg_l:
            return "Fit Problems"
        if "stability" in msg_l or "topo" in msg_l or "fluct" in msg_l:
            return "Topology Issues"
        if "robust" in msg_l or "invariant" in msg_l or "geomet" in msg_l:
            return "Geometry Warnings"
        if "rejected" in msg_l or "collapse" in msg_l or "fatal" in msg_l:
            return "Critical Errors"
        return "Miscellaneous"

    def get_sole_next_action(self, report: DiagnosticReport) -> Optional[Dict[str, str]]:
        """
        Distills all suggestions into exactly ONE prioritized 'Next Best Action'.
        Returns {'action': str, 'desc': str}
        """
        if not report.suggestions:
            return None
            
        # Priority: Topology > Geometry > Fit
        suggs = report.suggestions
        
        # 1. Topology (The most dangerous)
        for s in suggs:
            if any(w in s.lower() for w in ["regularization", "noise", "bandwidth"]):
                return {"action": "Stabilize Manifold", "desc": s}
                
        # 2. Geometry
        for s in suggs:
            if any(w in s.lower() for w in ["normali", "z-score", "invariant", "re-scale"]):
                return {"action": "Apply Normalization", "desc": s}
                
        # 3. Fit
        for s in suggs:
            if any(w in s.lower() for w in ["interaction", "non-linear", "terms"]):
                return {"action": "Complexity Boost", "desc": s}
                
        # Fallback to the first one
        return {"action": "Refine Model", "desc": suggs[0]}
