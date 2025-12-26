"""
The Last Honest Sentence: The One Claim You Can Make

Generates the single, maximally honest conclusion that can be
drawn from the analysis. Anything more triggers a warning.

"現状で唯一言っていい結論文を一文だけ生成"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from .assumption_tube import TubeMetrics, AssumptionTube
    from .assumption_path import AssumptionPath
    from .claim_budget import ClaimBudget, ClaimStrength
    from .one_way_door import OneWayDoor, DoorType
except ImportError:
    from statelix_py.core.assumption_tube import TubeMetrics, AssumptionTube
    from statelix_py.core.assumption_path import AssumptionPath
    from statelix_py.core.claim_budget import ClaimBudget, ClaimStrength
    from statelix_py.core.one_way_door import OneWayDoor, DoorType


@dataclass
class HonestSentence:
    """
    The last honest sentence - the one thing you can truthfully say.
    """
    sentence: str
    strength: ClaimStrength
    
    # What you CANNOT say
    forbidden_claims: List[str]
    
    # Why this is the limit
    limiting_factors: List[str]
    
    # Confidence in this sentence
    confidence: float  # 0-1
    
    # Extended version (if user insists on more)
    extended_version: Optional[str] = None
    extended_warning: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"HonestSentence({self.strength.value}): {self.sentence[:50]}..."


class HonestSentenceGenerator:
    """
    Generates the maximally honest conclusion from analysis results.
    
    This is the nuclear option: instead of letting researchers
    overstate findings, we generate the ONE sentence they can say.
    
    Example:
        >>> gen = HonestSentenceGenerator()
        >>> honest = gen.generate(tube, path, context)
        >>> 
        >>> print(honest.sentence)
        >>> print("Cannot say:", honest.forbidden_claims)
    """
    
    # Templates by claim strength
    TEMPLATES = {
        ClaimStrength.DEFINITIVE: [
            "Under the maintained assumptions, {effect_desc} with high confidence ({ci}).",
            "The data provide robust evidence that {effect_desc}.",
        ],
        ClaimStrength.STRONG: [
            "The analysis suggests {effect_desc}, though sensitivity to {brittle} warrants caution.",
            "We find evidence consistent with {effect_desc} (robustness: {robust:.0%}).",
        ],
        ClaimStrength.MODERATE: [
            "Under {conditions}, {effect_desc}, but conclusions are sensitive to {brittle}.",
            "The observed pattern suggests {effect_desc}, though {caveats}.",
        ],
        ClaimStrength.WEAK: [
            "In this sample, under {conditions}, we observe {effect_desc}.",
            "The data are consistent with {effect_desc}, but generalization is limited.",
        ],
        ClaimStrength.MINIMAL: [
            "We observe {effect_desc} in the present data.",
            "The analysis reveals {effect_desc}, with substantial uncertainty.",
        ],
        ClaimStrength.NONE: [
            "The available evidence does not support reliable conclusions.",
            "Methodological limitations prevent substantive claims.",
        ]
    }
    
    FORBIDDEN_TEMPLATES = {
        ClaimStrength.DEFINITIVE: [],
        ClaimStrength.STRONG: [
            "This proves...", "We definitively show...", "This establishes..."
        ],
        ClaimStrength.MODERATE: [
            "Strong evidence indicates...", "Clearly, ...", "Robust findings show..."
        ],
        ClaimStrength.WEAK: [
            "The evidence suggests...", "Our findings indicate...", 
            "This study demonstrates..."
        ],
        ClaimStrength.MINIMAL: [
            "We find that...", "Results show...", "The analysis reveals..."
        ],
        ClaimStrength.NONE: [
            "We observe...", "The data suggest...", "Patterns indicate..."
        ]
    }
    
    def generate(
        self,
        tube: Optional[AssumptionTube] = None,
        path: Optional[AssumptionPath] = None,
        budget: Optional[ClaimBudget] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> HonestSentence:
        """
        Generate the honest sentence.
        
        Args:
            tube: AssumptionTube from robustness analysis
            path: AssumptionPath from path tracing
            budget: Pre-computed ClaimBudget
            context: Additional context (effect_name, target_name, etc.)
        
        Returns:
            HonestSentence with the one allowed claim
        """
        context = context or {}
        
        # Compute or use provided budget
        if budget is None and tube is not None:
            from .claim_budget import ClaimBudgetCalculator
            budget = ClaimBudgetCalculator().from_tube(tube)
        elif budget is None:
            # Minimal budget if no info
            budget = ClaimBudget(
                max_strength=ClaimStrength.MINIMAL,
                robustness_score=0.3,
                causal_allowed=False,
                predictive_allowed=False,
                generalizable=False,
                warnings=["No robustness analysis available"],
                budget_remaining=0.5
            )
        
        # Extract metrics
        if tube is not None:
            metrics = tube.compute_metrics()
            robust = metrics.robustness_radius
            brittle = metrics.most_brittle_assumption() or "assumption violations"
        else:
            robust = budget.robustness_score
            brittle = "various assumptions"
        
        # Build template variables
        effect_name = context.get('effect_name', 'the estimated effect')
        target_name = context.get('target_name', 'the outcome')
        effect_value = context.get('effect_value')
        ci = context.get('confidence_interval', 'see supplementary')
        
        if effect_value is not None:
            effect_desc = f"{effect_name} is approximately {effect_value:.3f}"
        else:
            effect_desc = f"a relationship exists between {effect_name} and {target_name}"
        
        # Build conditions description
        conditions = self._describe_conditions(budget, path)
        caveats = self._describe_caveats(budget, tube)
        
        # Select template
        templates = self.TEMPLATES[budget.max_strength]
        template = templates[0]  # Use first template
        
        # Fill template
        sentence = template.format(
            effect_desc=effect_desc,
            effect_name=effect_name,
            target_name=target_name,
            ci=ci,
            brittle=brittle,
            robust=robust,
            conditions=conditions,
            caveats=caveats
        )
        
        # Collect forbidden claims
        forbidden = []
        strength_order = [
            ClaimStrength.MINIMAL, ClaimStrength.WEAK, ClaimStrength.MODERATE,
            ClaimStrength.STRONG, ClaimStrength.DEFINITIVE
        ]
        
        for strength in strength_order:
            if not budget.can_claim(strength):
                forbidden.extend(self.FORBIDDEN_TEMPLATES.get(strength, []))
        
        # Add specific forbidden claims based on analysis
        if not budget.causal_allowed:
            forbidden.extend(["causes", "causal effect", "因果効果"])
        if not budget.generalizable:
            forbidden.extend(["in general", "generally", "一般に"])
        if not budget.predictive_allowed:
            forbidden.extend(["predicts", "will", "予測する"])
        
        # Limiting factors
        limiting = budget.warnings.copy()
        if robust < 0.5:
            limiting.append(f"Robustness radius: {robust:.2f}")
        
        # Confidence
        confidence = min(0.9, robust + 0.1)
        
        # Extended version (grudgingly provided)
        extended = self._generate_extended(sentence, budget, context)
        extended_warning = (
            "⚠ これ以上の主張は、現在の分析では正当化されません。" 
            if extended else None
        )
        
        return HonestSentence(
            sentence=sentence,
            strength=budget.max_strength,
            forbidden_claims=forbidden[:10],  # Limit list
            limiting_factors=limiting,
            confidence=confidence,
            extended_version=extended,
            extended_warning=extended_warning
        )
    
    def _describe_conditions(
        self, 
        budget: ClaimBudget, 
        path: Optional[AssumptionPath]
    ) -> str:
        """Describe the conditions under which the claim holds."""
        conditions = []
        
        if not budget.causal_allowed:
            conditions.append("assuming association only")
        if not budget.generalizable:
            conditions.append("for this sample")
        
        if path and path.cliffs:
            cliff_assumptions = [c.broken_assumption for c in path.cliffs[:2]]
            if cliff_assumptions:
                conditions.append(f"excluding extreme {', '.join(cliff_assumptions)}")
        
        if not conditions:
            return "standard assumptions"
        
        return "; ".join(conditions)
    
    def _describe_caveats(
        self,
        budget: ClaimBudget,
        tube: Optional[AssumptionTube]
    ) -> str:
        """Describe caveats and limitations."""
        caveats = []
        
        if budget.robustness_score < 0.5:
            caveats.append("robustness is limited")
        
        if tube:
            metrics = tube.compute_metrics()
            if metrics.self_intersection_count > 0:
                caveats.append("some instability observed")
            if metrics.truth_thickness < 0.2:
                caveats.append("conclusions are thin at some points")
        
        if not budget.predictive_allowed:
            caveats.append("prediction is not advised")
        
        return "; ".join(caveats) if caveats else "see supplementary analysis"
    
    def _generate_extended(
        self,
        base_sentence: str,
        budget: ClaimBudget,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate extended version if user insists."""
        if budget.max_strength in [ClaimStrength.STRONG, ClaimStrength.DEFINITIVE]:
            return None  # No need for extension
        
        effect_name = context.get('effect_name', 'the effect')
        
        extension = (
            f"{base_sentence} Additional analysis would be required to "
            f"strengthen claims about {effect_name}. Specifically, "
            f"addressing {', '.join(budget.warnings[:2]) if budget.warnings else 'identified limitations'} "
            f"could improve robustness."
        )
        
        return extension


def generate_honest_sentence(
    tube: Optional[AssumptionTube] = None,
    path: Optional[AssumptionPath] = None,
    **context
) -> HonestSentence:
    """Convenience function to generate the honest sentence."""
    return HonestSentenceGenerator().generate(tube, path, context=context)
