
from ..diagnostics.critic import ModelCritic, CriticMode, ModelRejectedError
from ..diagnostics.history import DiagnosticHistory
from ..diagnostics.presets import GovernanceMode

class DiagnosticAwareMixin:
    """
    Mixin to imbue specific estimators with Human-Centric Diagnostics.
    Ensures all Statelix models return a uniform Diagnostic Contract.
    """
    
    def _init_diagnostics(self, governance_mode: GovernanceMode = GovernanceMode.STRICT):
        self._critic = ModelCritic(mode=CriticMode.SUGGESTIVE, governance_mode=governance_mode)
        self.diagnostic_report_ = None
        self.diagnostic_history_ = DiagnosticHistory()
        
    @property
    def mci(self):
        """Model Credibility Index score (0.0 - 1.0)."""
        if self.diagnostic_report_ is None: return None
        return self.diagnostic_report_.mci.total_score
        
    @property
    def objections(self):
        """List of diagnostic objections (reasons why MCI is low)."""
        if self.diagnostic_report_ is None: return []
        return self.diagnostic_report_.messages
        
    @property
    def suggestions(self):
        """List of suggested fixes."""
        if self.diagnostic_report_ is None: return []
        return self.diagnostic_report_.suggestions
        
    @property
    def history(self):
        """Diagnostic history object."""
        return self.diagnostic_history_
        
    def _run_diagnostics(self, metrics: dict):
        """
        Runs the ModelCritic and stores the report.
        Raises ModelRejectedError if strict standards are not met.
        """
        report = self._critic.critique(metrics)
        self.diagnostic_report_ = report
        self.diagnostic_history_.add(report)
