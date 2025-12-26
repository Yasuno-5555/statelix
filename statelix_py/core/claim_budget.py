"""
Claim Budget: Maximum Assertable Strength

Calculates the upper bound on how strong a claim can be made,
based on robustness metrics. Write stronger claims = red warning.

"この論文で主張できる強さの上限"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from .assumption_tube import TubeMetrics, AssumptionTube
except ImportError:
    from statelix_py.core.assumption_tube import TubeMetrics, AssumptionTube


class ClaimStrength(Enum):
    """Levels of claim strength."""
    DEFINITIVE = "definitive"       # "X causes Y"
    STRONG = "strong"               # "X strongly associated with Y"  
    MODERATE = "moderate"           # "Evidence suggests X relates to Y"
    WEAK = "weak"                   # "Under these conditions, X correlates with Y"
    MINIMAL = "minimal"             # "We observe a pattern"
    NONE = "none"                   # Cannot make claims


@dataclass
class ClaimBudget:
    """
    The budget of claims available for this analysis.
    """
    max_strength: ClaimStrength
    robustness_score: float  # 0-1, higher = more robust
    
    # Per-dimension claim limits
    causal_allowed: bool
    predictive_allowed: bool
    generalizable: bool
    
    # Specific warnings
    warnings: List[str]
    
    # How much "claim currency" is left
    budget_remaining: float  # 0-1, depleted by each claim
    
    def can_claim(self, strength: ClaimStrength) -> bool:
        """Check if a claim of given strength is allowed."""
        strength_order = [
            ClaimStrength.NONE,
            ClaimStrength.MINIMAL,
            ClaimStrength.WEAK,
            ClaimStrength.MODERATE,
            ClaimStrength.STRONG,
            ClaimStrength.DEFINITIVE
        ]
        max_idx = strength_order.index(self.max_strength)
        claim_idx = strength_order.index(strength)
        return claim_idx <= max_idx
    
    def claim_cost(self, strength: ClaimStrength) -> float:
        """How much budget does this claim cost?"""
        costs = {
            ClaimStrength.NONE: 0,
            ClaimStrength.MINIMAL: 0.1,
            ClaimStrength.WEAK: 0.2,
            ClaimStrength.MODERATE: 0.4,
            ClaimStrength.STRONG: 0.7,
            ClaimStrength.DEFINITIVE: 1.0
        }
        return costs.get(strength, 0.5)
    
    def __repr__(self) -> str:
        return f"ClaimBudget(max={self.max_strength.value}, remaining={self.budget_remaining:.2f})"


# Claim strength phrases for language checking
CLAIM_PHRASES = {
    ClaimStrength.DEFINITIVE: [
        "causes", "proves", "demonstrates conclusively", 
        "definitively shows", "establishes", "confirms",
        "を引き起こす", "証明する", "確定的に示す"
    ],
    ClaimStrength.STRONG: [
        "strongly suggests", "clearly indicates", "robust evidence",
        "significantly affects", "strong association",
        "強く示唆", "明確に示す", "有意に影響"
    ],
    ClaimStrength.MODERATE: [
        "suggests", "indicates", "appears to", "evidence for",
        "associated with", "related to",
        "示唆する", "関連している", "〜と思われる"
    ],
    ClaimStrength.WEAK: [
        "may", "might", "could", "possible", "tentative",
        "under these conditions", "in this sample",
        "かもしれない", "可能性がある", "この条件下では"
    ],
    ClaimStrength.MINIMAL: [
        "we observe", "the data show", "pattern", "trend",
        "観察される", "データは示す", "傾向"
    ]
}


class ClaimBudgetCalculator:
    """
    Calculates claim budget from robustness metrics.
    
    Maps TubeMetrics to the maximum strength of claims
    that can be honestly made about the results.
    
    Example:
        >>> calc = ClaimBudgetCalculator()
        >>> budget = calc.from_tube_metrics(metrics)
        >>> 
        >>> if not budget.can_claim(ClaimStrength.STRONG):
        ...     print("Cannot make strong claims with this data")
        >>> 
        >>> # Check a sentence
        >>> violations = calc.check_text("This proves X causes Y", budget)
    """
    
    def from_tube_metrics(self, metrics: TubeMetrics) -> ClaimBudget:
        """
        Calculate claim budget from tube metrics.
        
        Args:
            metrics: TubeMetrics from robustness analysis
        
        Returns:
            ClaimBudget with allowed claim strengths
        """
        # Core robustness score
        robustness = metrics.robustness_radius
        thickness = metrics.truth_thickness
        intersections = metrics.self_intersection_count
        
        # Combined score
        combined = (robustness * 0.5 + thickness * 0.3 + 
                   (1 / (1 + intersections)) * 0.2)
        
        # Map to maximum strength
        if combined > 0.8 and intersections == 0:
            max_strength = ClaimStrength.DEFINITIVE
        elif combined > 0.6:
            max_strength = ClaimStrength.STRONG
        elif combined > 0.4:
            max_strength = ClaimStrength.MODERATE
        elif combined > 0.2:
            max_strength = ClaimStrength.WEAK
        elif combined > 0.05:
            max_strength = ClaimStrength.MINIMAL
        else:
            max_strength = ClaimStrength.NONE
        
        # Determine allowed claim types
        causal = robustness > 0.5 and intersections < 3
        predictive = thickness > 0.2
        generalizable = robustness > 0.4 and metrics.total_variance < 0.5
        
        # Generate warnings
        warnings = []
        if robustness < 0.3:
            warnings.append("低頑健性: データ摂動に敏感です。強い主張は避けてください。")
        if thickness < 0.1:
            warnings.append("真実が薄い: 結論が極めて脆弱な箇所があります。")
        if intersections > 5:
            warnings.append("経路が交差: 小さな変化で結論が反転する可能性があります。")
        
        most_brittle = metrics.most_brittle_assumption()
        if most_brittle:
            warnings.append(f"最も脆い仮定: {most_brittle}")
        
        return ClaimBudget(
            max_strength=max_strength,
            robustness_score=combined,
            causal_allowed=causal,
            predictive_allowed=predictive,
            generalizable=generalizable,
            warnings=warnings,
            budget_remaining=1.0
        )
    
    def from_tube(self, tube: AssumptionTube) -> ClaimBudget:
        """Calculate claim budget from an AssumptionTube."""
        return self.from_tube_metrics(tube.compute_metrics())
    
    def check_text(
        self, 
        text: str, 
        budget: ClaimBudget
    ) -> List[Tuple[str, ClaimStrength, bool]]:
        """
        Check text for claims that exceed the budget.
        
        Args:
            text: Text to check (e.g., conclusion section)
            budget: Current claim budget
        
        Returns:
            List of (phrase, detected_strength, is_violation) tuples
        """
        text_lower = text.lower()
        findings = []
        
        for strength, phrases in CLAIM_PHRASES.items():
            for phrase in phrases:
                if phrase.lower() in text_lower:
                    is_violation = not budget.can_claim(strength)
                    findings.append((phrase, strength, is_violation))
        
        return findings
    
    def sanitize_text(
        self,
        text: str,
        budget: ClaimBudget
    ) -> Tuple[str, List[str]]:
        """
        Suggest sanitized version of text.
        
        Returns:
            (sanitized_text, list_of_changes)
        """
        changes = []
        result = text
        
        violations = self.check_text(text, budget)
        
        # Replacement suggestions
        replacements = {
            "causes": "is associated with",
            "proves": "suggests",
            "demonstrates conclusively": "indicates",
            "definitively shows": "appears to show",
            "strongly suggests": "may suggest",
            "clearly indicates": "possibly indicates",
            "証明する": "示唆する",
            "確定的に示す": "〜と思われる",
            "引き起こす": "関連している可能性がある"
        }
        
        for phrase, strength, is_violation in violations:
            if is_violation and phrase.lower() in replacements:
                replacement = replacements[phrase.lower()]
                result = result.replace(phrase, replacement)
                changes.append(f"'{phrase}' → '{replacement}'")
        
        return result, changes


def calculate_claim_budget(tube: AssumptionTube) -> ClaimBudget:
    """Convenience function to calculate claim budget."""
    return ClaimBudgetCalculator().from_tube(tube)
