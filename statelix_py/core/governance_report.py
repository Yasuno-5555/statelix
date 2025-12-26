"""
Governance Report: Unified Analysis of What Can Be Claimed

Combines One-Way Door, Claim Budget, and Honest Sentence into
a single comprehensive governance report.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from .assumption_path import AssumptionPath, AssumptionState
    from .assumption_tube import AssumptionTube, TubeMetrics
    from .one_way_door import OneWayDoorAnalyzer, OneWayDoor
    from .claim_budget import ClaimBudgetCalculator, ClaimBudget, ClaimStrength
    from .honest_sentence import HonestSentenceGenerator, HonestSentence
except ImportError:
    from statelix_py.core.assumption_path import AssumptionPath, AssumptionState
    from statelix_py.core.assumption_tube import AssumptionTube, TubeMetrics
    from statelix_py.core.one_way_door import OneWayDoorAnalyzer, OneWayDoor
    from statelix_py.core.claim_budget import ClaimBudgetCalculator, ClaimBudget, ClaimStrength
    from statelix_py.core.honest_sentence import HonestSentenceGenerator, HonestSentence


@dataclass
class GovernanceReport:
    """
    Comprehensive governance report combining all constraint systems.
    
    This is the final verdict on what can and cannot be claimed.
    """
    # Metadata
    generated_at: str
    analysis_id: str
    
    # One-Way Doors
    doors: List[OneWayDoor]
    doors_crossed: List[OneWayDoor]
    doors_imminent: List[OneWayDoor]
    
    # Claim Budget
    budget: ClaimBudget
    
    # The Last Honest Sentence
    honest_sentence: HonestSentence
    
    # Overall verdict
    verdict: str  # "GREEN", "YELLOW", "RED"
    verdict_explanation: str
    
    # Warnings collected
    all_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'generated_at': self.generated_at,
            'analysis_id': self.analysis_id,
            'verdict': self.verdict,
            'verdict_explanation': self.verdict_explanation,
            'max_claim_strength': self.budget.max_strength.value,
            'honest_sentence': self.honest_sentence.sentence,
            'forbidden_claims': self.honest_sentence.forbidden_claims,
            'doors_crossed': [d.door_type.value for d in self.doors_crossed],
            'doors_imminent': [d.door_type.value for d in self.doors_imminent],
            'warnings': self.all_warnings
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        verdict_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(self.verdict, "⚪")
        
        md = f"""# Governance Report

**Generated:** {self.generated_at}

## Verdict: {verdict_emoji} {self.verdict}

{self.verdict_explanation}

---

## The Last Honest Sentence

> **{self.honest_sentence.sentence}**

Claim Strength: `{self.honest_sentence.strength.value.upper()}`

### Forbidden Claims

{"- " + chr(10) + "- ".join(self.honest_sentence.forbidden_claims[:5]) if self.honest_sentence.forbidden_claims else "(none)"}

---

## Claim Budget

- **Maximum Strength:** {self.budget.max_strength.value}
- **Robustness Score:** {self.budget.robustness_score:.2f}
- **Causal Claims:** {"✓" if self.budget.causal_allowed else "✗"}
- **Predictive Claims:** {"✓" if self.budget.predictive_allowed else "✗"}
- **Generalizable:** {"✓" if self.budget.generalizable else "✗"}

---

## One-Way Doors

"""
        if self.doors_crossed:
            md += "### 🚫 Crossed (Cannot Return)\n\n"
            for d in self.doors_crossed:
                md += f"- **{d.door_type.value}**: {d.warning_message}\n"
            md += "\n"
        
        if self.doors_imminent:
            md += "### ⚠ Imminent\n\n"
            for d in self.doors_imminent:
                md += f"- **{d.trigger_assumption}**: Distance {d.distance_to_door:.2f}\n"
            md += "\n"
        
        if self.all_warnings:
            md += "---\n\n## Warnings\n\n"
            for w in self.all_warnings:
                md += f"- {w}\n"
        
        return md
    
    def __repr__(self) -> str:
        return f"GovernanceReport({self.verdict}, max={self.budget.max_strength.value})"


class GovernanceReportGenerator:
    """
    Generates comprehensive governance reports.
    
    Combines One-Way Door analysis, Claim Budget calculation,
    and Honest Sentence generation into unified report.
    
    Example:
        >>> gen = GovernanceReportGenerator()
        >>> report = gen.generate(tube=tube, path=path)
        >>> 
        >>> print(report.verdict)
        >>> print(report.honest_sentence.sentence)
        >>> print(report.to_markdown())
    """
    
    def __init__(self):
        self.door_analyzer = OneWayDoorAnalyzer()
        self.budget_calculator = ClaimBudgetCalculator()
        self.sentence_generator = HonestSentenceGenerator()
    
    def generate(
        self,
        tube: Optional[AssumptionTube] = None,
        path: Optional[AssumptionPath] = None,
        state: Optional[AssumptionState] = None,
        context: Optional[Dict[str, Any]] = None,
        analysis_id: Optional[str] = None
    ) -> GovernanceReport:
        """
        Generate comprehensive governance report.
        
        Args:
            tube: AssumptionTube from robustness analysis
            path: AssumptionPath from path tracing
            state: Current AssumptionState (inferred if not provided)
            context: Additional context for sentence generation
            analysis_id: Optional ID for the analysis
        
        Returns:
            GovernanceReport with all governance information
        """
        context = context or {}
        
        # Determine current state
        if state is None and path and path.points:
            state = path.points[-1].state  # Use final state
        elif state is None:
            state = AssumptionState.classical()
        
        # One-Way Door Analysis
        doors = self.door_analyzer.analyze(state)
        doors_crossed = [d for d in doors if d.is_crossed]
        doors_imminent = [d for d in doors if d.is_imminent]
        
        # Claim Budget
        if tube is not None:
            budget = self.budget_calculator.from_tube(tube)
        else:
            # Minimal budget without tube
            budget = ClaimBudget(
                max_strength=ClaimStrength.WEAK,
                robustness_score=0.3,
                causal_allowed=False,
                predictive_allowed=False,
                generalizable=False,
                warnings=["No robustness tube available"],
                budget_remaining=0.5
            )
        
        # The Last Honest Sentence
        honest = self.sentence_generator.generate(
            tube=tube, 
            path=path, 
            budget=budget,
            context=context
        )
        
        # Compute overall verdict
        verdict, explanation = self._compute_verdict(
            doors_crossed, doors_imminent, budget, path
        )
        
        # Collect all warnings
        all_warnings = budget.warnings.copy()
        for door in doors_crossed:
            all_warnings.append(door.warning_message)
        for door in doors_imminent:
            all_warnings.append(f"接近中: {door.trigger_assumption} ({door.distance_to_door:.0%} まで)")
        
        if path and path.cliffs:
            for cliff in path.cliffs[:3]:
                all_warnings.append(f"崖検出: {cliff.broken_assumption} at t={cliff.t:.2f}")
        
        return GovernanceReport(
            generated_at=datetime.now().isoformat(),
            analysis_id=analysis_id or f"gov_{id(tube) % 10000:04d}",
            doors=doors,
            doors_crossed=doors_crossed,
            doors_imminent=doors_imminent,
            budget=budget,
            honest_sentence=honest,
            verdict=verdict,
            verdict_explanation=explanation,
            all_warnings=all_warnings
        )
    
    def _compute_verdict(
        self,
        crossed: List[OneWayDoor],
        imminent: List[OneWayDoor],
        budget: ClaimBudget,
        path: Optional[AssumptionPath]
    ) -> tuple:
        """Compute overall verdict."""
        # Count issues
        n_crossed = len(crossed)
        n_imminent = len(imminent)
        n_cliffs = len(path.cliffs) if path else 0
        
        strength = budget.max_strength
        robustness = budget.robustness_score
        
        # RED: Major issues
        if (n_crossed >= 2 or 
            strength == ClaimStrength.NONE or
            robustness < 0.1):
            return "RED", (
                "重大な問題があります。複数の一方通行ドアを超えたか、"
                "頑健性が極めて低いです。主張は避けてください。"
            )
        
        # YELLOW: Some concerns
        if (n_crossed >= 1 or 
            n_imminent >= 2 or
            n_cliffs >= 3 or
            strength in [ClaimStrength.MINIMAL, ClaimStrength.WEAK] or
            robustness < 0.4):
            return "YELLOW", (
                "注意が必要です。いくつかの制約に達しているか、"
                "頑健性に懸念があります。主張を控えめにしてください。"
            )
        
        # GREEN: Generally okay
        return "GREEN", (
            "分析は概ね健全です。推奨された最大主張強度の範囲内で"
            "結論を述べることができます。"
        )


def generate_governance_report(
    tube: Optional[AssumptionTube] = None,
    path: Optional[AssumptionPath] = None,
    **context
) -> GovernanceReport:
    """Convenience function to generate governance report."""
    return GovernanceReportGenerator().generate(tube, path, context=context)
