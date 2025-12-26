"""
Responsibility Sink: Where Claims Fall Before They Reach the World

Not just what CAN be said, but what SHOULD be said.
The ethical completion of Statelix magic.

"言った結果、誰が傷つくか / 何が動くか"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum 

try:
    from .claim_compiler import ClaimIR, CompiledClaim, ClaimNature, ClaimScope, Dialect
    from .claim_budget import ClaimBudget, ClaimStrength
    from .assumption_tube import TubeMetrics
except ImportError:
    from statelix_py.core.claim_compiler import ClaimIR, CompiledClaim, ClaimNature, ClaimScope, Dialect
    from statelix_py.core.claim_budget import ClaimBudget, ClaimStrength
    from statelix_py.core.assumption_tube import TubeMetrics


# =============================================================================
# Impact Classes
# =============================================================================

class ImpactClass(Enum):
    """Classification of claim impact on the world."""
    ACADEMIC_ONLY = "academic_only"           # Stays in journals
    POLICY_TRIGGERING = "policy_triggering"   # May influence policy
    MARKET_SENSITIVE = "market_sensitive"     # May move markets
    NORM_SHAPING = "norm_shaping"             # May change social norms
    INDIVIDUAL_TARGETING = "individual_targeting"  # About individuals/groups
    
    @property
    def risk_multiplier(self) -> float:
        """Higher impact = higher required robustness."""
        multipliers = {
            ImpactClass.ACADEMIC_ONLY: 1.0,
            ImpactClass.POLICY_TRIGGERING: 2.0,
            ImpactClass.MARKET_SENSITIVE: 2.5,
            ImpactClass.NORM_SHAPING: 3.0,
            ImpactClass.INDIVIDUAL_TARGETING: 5.0,  # Highest bar
        }
        return multipliers.get(self, 1.0)
    
    @property
    def description(self) -> str:
        descriptions = {
            ImpactClass.ACADEMIC_ONLY: "学術発表のみ - 影響は限定的",
            ImpactClass.POLICY_TRIGGERING: "政策に影響しうる - 高い責任",
            ImpactClass.MARKET_SENSITIVE: "市場に影響しうる - 慎重な発言が必要",
            ImpactClass.NORM_SHAPING: "社会規範を形成しうる - 極めて慎重に",
            ImpactClass.INDIVIDUAL_TARGETING: "個人・集団への言及 - 原則拒否",
        }
        return descriptions.get(self, "不明")


# =============================================================================
# Ethical Cliff Detection
# =============================================================================

class EthicalConcern(Enum):
    """Types of ethical concerns."""
    GROUP_ESSENTIALISM = "group_essentialism"      # Essentializing groups
    DISCRIMINATION_RISK = "discrimination_risk"    # May enable discrimination
    MISUSE_VULNERABILITY = "misuse_vulnerability"  # Easy to misuse
    CONSENT_VIOLATION = "consent_violation"        # Analysis without consent
    POWER_ASYMMETRY = "power_asymmetry"           # Benefits powerful over weak
    IRREVERSIBLE_HARM = "irreversible_harm"       # Cannot undo damage


@dataclass
class EthicalCliff:
    """
    A detected ethical cliff - true but shouldn't be said.
    """
    concern: EthicalConcern
    severity: float  # 0-1
    description: str
    affected_groups: List[str] = field(default_factory=list)
    mitigation_possible: bool = True
    mitigation_strategy: Optional[str] = None
    
    @property
    def is_fatal(self) -> bool:
        """Does this cliff require RED judgment?"""
        return self.severity > 0.7 or not self.mitigation_possible


# =============================================================================
# Responsibility Budget
# =============================================================================

@dataclass
class ResponsibilityBudget:
    """
    Separate from Claim Budget - about impact, not certainty.
    
    Higher impact requires exponentially higher robustness.
    """
    impact_class: ImpactClass
    required_robustness: float  # What robustness is needed
    actual_robustness: float    # What we have
    
    ethical_cliffs: List[EthicalCliff] = field(default_factory=list)
    
    # Verdict
    is_responsible: bool = True
    rejection_reason: Optional[str] = None
    
    @property
    def responsibility_gap(self) -> float:
        """Gap between required and actual robustness."""
        return max(0, self.required_robustness - self.actual_robustness)
    
    @property
    def can_proceed(self) -> bool:
        """Can we ethically proceed with this claim?"""
        if not self.is_responsible:
            return False
        if any(c.is_fatal for c in self.ethical_cliffs):
            return False
        return self.responsibility_gap <= 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'impact_class': self.impact_class.value,
            'required_robustness': self.required_robustness,
            'actual_robustness': self.actual_robustness,
            'responsibility_gap': self.responsibility_gap,
            'can_proceed': self.can_proceed,
            'ethical_cliffs': len(self.ethical_cliffs),
            'fatal_cliffs': sum(1 for c in self.ethical_cliffs if c.is_fatal)
        }


# =============================================================================
# Responsibility Sink
# =============================================================================

class ResponsibilitySink:
    """
    Where claims fall before they reach the world.
    
    Simulates the impact of a claim and determines if it should be made.
    
    Example:
        >>> sink = ResponsibilitySink()
        >>> budget = sink.evaluate(ir, tube_metrics)
        >>> 
        >>> if not budget.can_proceed:
        ...     print("この主張は責任ある形で行えません")
        >>> 
        >>> responsible_claim = sink.generate_responsible_claim(ir, claim)
    """
    
    # Keywords triggering impact classification
    IMPACT_TRIGGERS = {
        ImpactClass.POLICY_TRIGGERING: [
            'policy', 'regulation', 'law', 'government', 'intervention',
            '政策', '規制', '法律', '政府', '介入'
        ],
        ImpactClass.MARKET_SENSITIVE: [
            'stock', 'market', 'price', 'investment', 'economic',
            '株', '市場', '価格', '投資', '経済'
        ],
        ImpactClass.NORM_SHAPING: [
            'culture', 'society', 'gender', 'race', 'norm', 'stereotype',
            '文化', '社会', 'ジェンダー', '人種', '規範', 'ステレオタイプ'
        ],
        ImpactClass.INDIVIDUAL_TARGETING: [
            'individual', 'person', 'patient', 'student', 'employee',
            'criminal', 'predict who', 'identify',
            '個人', '患者', '生徒', '従業員', '犯罪者', '誰を予測'
        ]
    }
    
    # Ethical concern triggers
    ETHICAL_TRIGGERS = {
        EthicalConcern.GROUP_ESSENTIALISM: [
            'inherent', 'natural', 'biological', 'innate', 'essential',
            '本質的', '生来', '生物学的', '先天的'
        ],
        EthicalConcern.DISCRIMINATION_RISK: [
            'predict', 'classify', 'identify', 'screen', 'select',
            '予測', '分類', '識別', 'スクリーニング', '選別'
        ],
        EthicalConcern.POWER_ASYMMETRY: [
            'employer', 'government', 'institution', 'authority',
            '雇用者', '政府', '機関', '権威'
        ]
    }
    
    def evaluate(
        self,
        ir: ClaimIR,
        metrics: Optional[TubeMetrics] = None,
        context: Optional[str] = None
    ) -> ResponsibilityBudget:
        """
        Evaluate the responsibility of making a claim.
        
        Args:
            ir: ClaimIR to evaluate
            metrics: Tube metrics for robustness
            context: Optional additional context
        
        Returns:
            ResponsibilityBudget with verdict
        """
        # Classify impact
        impact_class = self._classify_impact(ir, context)
        
        # Calculate required robustness
        base_required = 0.3
        required = min(0.95, base_required * impact_class.risk_multiplier)
        
        # Get actual robustness
        actual = ir.robustness_score
        if metrics:
            actual = max(actual, metrics.robustness_radius)
        
        # Detect ethical cliffs
        ethical_cliffs = self._detect_ethical_cliffs(ir, context)
        
        # Determine if responsible
        is_responsible = True
        rejection_reason = None
        
        if actual < required - 0.1:
            is_responsible = False
            rejection_reason = f"頑健性が不足: 必要 {required:.2f}, 実際 {actual:.2f}"
        
        fatal_cliffs = [c for c in ethical_cliffs if c.is_fatal]
        if fatal_cliffs:
            is_responsible = False
            rejection_reason = f"倫理的崖: {fatal_cliffs[0].description}"
        
        # Special case: individual targeting with any causal claim
        if (impact_class == ImpactClass.INDIVIDUAL_TARGETING and 
            ir.nature in [ClaimNature.CAUSAL, ClaimNature.PRESCRIPTIVE]):
            is_responsible = False
            rejection_reason = "個人への因果的主張は原則拒否"
        
        return ResponsibilityBudget(
            impact_class=impact_class,
            required_robustness=required,
            actual_robustness=actual,
            ethical_cliffs=ethical_cliffs,
            is_responsible=is_responsible,
            rejection_reason=rejection_reason
        )
    
    def _classify_impact(
        self,
        ir: ClaimIR,
        context: Optional[str]
    ) -> ImpactClass:
        """Classify the impact of a claim."""
        text = f"{ir.effect_name} {ir.target_name} {context or ''}"
        text_lower = text.lower()
        
        # Check each impact class
        for impact_class, triggers in self.IMPACT_TRIGGERS.items():
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    return impact_class
        
        # Check claim nature
        if ir.nature == ClaimNature.PRESCRIPTIVE:
            return ImpactClass.POLICY_TRIGGERING
        
        return ImpactClass.ACADEMIC_ONLY
    
    def _detect_ethical_cliffs(
        self,
        ir: ClaimIR,
        context: Optional[str]
    ) -> List[EthicalCliff]:
        """Detect ethical concerns in a claim."""
        cliffs = []
        text = f"{ir.effect_name} {ir.target_name} {context or ''}"
        text_lower = text.lower()
        
        for concern, triggers in self.ETHICAL_TRIGGERS.items():
            triggered = any(t.lower() in text_lower for t in triggers)
            if triggered:
                severity = 0.5
                if concern == EthicalConcern.GROUP_ESSENTIALISM:
                    severity = 0.8
                if concern == EthicalConcern.DISCRIMINATION_RISK:
                    severity = 0.6
                
                cliff = EthicalCliff(
                    concern=concern,
                    severity=severity,
                    description=self._get_concern_description(concern),
                    mitigation_possible=severity < 0.8,
                    mitigation_strategy=self._get_mitigation(concern)
                )
                cliffs.append(cliff)
        
        return cliffs
    
    def _get_concern_description(self, concern: EthicalConcern) -> str:
        descriptions = {
            EthicalConcern.GROUP_ESSENTIALISM: "集団を本質化する危険があります",
            EthicalConcern.DISCRIMINATION_RISK: "差別的利用が可能な形式です",
            EthicalConcern.MISUSE_VULNERABILITY: "誤用に対して脆弱です",
            EthicalConcern.CONSENT_VIOLATION: "同意なき分析の可能性",
            EthicalConcern.POWER_ASYMMETRY: "権力非対称を強化する可能性",
            EthicalConcern.IRREVERSIBLE_HARM: "取り消せない害を与える可能性",
        }
        return descriptions.get(concern, "倫理的懸念")
    
    def _get_mitigation(self, concern: EthicalConcern) -> Optional[str]:
        mitigations = {
            EthicalConcern.GROUP_ESSENTIALISM: 
                "個人差・文脈依存性を明示的に強調してください",
            EthicalConcern.DISCRIMINATION_RISK:
                "予測モデルの使用制限を明記してください",
            EthicalConcern.POWER_ASYMMETRY:
                "被分析者への開示と同意を確認してください",
        }
        return mitigations.get(concern)
    
    def generate_responsible_claim(
        self,
        ir: ClaimIR,
        claim: CompiledClaim,
        budget: Optional[ResponsibilityBudget] = None
    ) -> 'ResponsibleClaim':
        """
        Generate the last responsible sentence.
        
        Minimizes harm while preserving truth.
        """
        if budget is None:
            budget = self.evaluate(ir)
        
        if not budget.can_proceed:
            return ResponsibleClaim(
                text="この分析結果の公表は、倫理的観点から推奨されません。",
                is_cleared=False,
                original_claim=claim.text,
                modifications=["Complete rejection due to ethical concerns"],
                budget=budget
            )
        
        # Build responsible text
        disclaimers = []
        modifications = []
        
        # Add scope limitation
        if ir.scope != ClaimScope.SAMPLE_ONLY:
            disclaimers.append("本分析は、特定の条件下で観測された統計的構造を示すに留まります")
            modifications.append("Scope limited to sample")
        
        # Add non-essentialism disclaimer
        if any(c.concern == EthicalConcern.GROUP_ESSENTIALISM for c in budget.ethical_cliffs):
            disclaimers.append("これは集団や個人の本質的特性を示すものではありません")
            modifications.append("Non-essentialism disclaimer added")
        
        # Add misuse warning
        if any(c.concern == EthicalConcern.DISCRIMINATION_RISK for c in budget.ethical_cliffs):
            disclaimers.append("個人の選別や差別的利用を正当化する根拠にはなりません")
            modifications.append("Anti-discrimination disclaimer added")
        
        # Build final text
        base_text = claim.text if claim.is_valid else f"{ir.effect_name} と {ir.target_name} の関係について"
        
        if disclaimers:
            final_text = f"{base_text}\n\n【重要】\n" + "\n".join(f"・{d}" for d in disclaimers)
        else:
            final_text = base_text
        
        return ResponsibleClaim(
            text=final_text,
            is_cleared=True,
            original_claim=claim.text,
            modifications=modifications,
            budget=budget,
            disclaimers=disclaimers
        )


# =============================================================================
# Responsible Claim
# =============================================================================

@dataclass
class ResponsibleClaim:
    """
    The last responsible sentence - minimizes harm.
    """
    text: str
    is_cleared: bool
    original_claim: str
    modifications: List[str]
    budget: ResponsibilityBudget
    disclaimers: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        status = "✓ 公表可" if self.is_cleared else "✗ 公表推奨せず"
        
        md = f"""# Responsibility Report

**Status:** {status}

## Impact Classification

- **Class:** {self.budget.impact_class.value}
- **Description:** {self.budget.impact_class.description}

## Robustness Check

| Required | Actual | Gap |
|----------|--------|-----|
| {self.budget.required_robustness:.2f} | {self.budget.actual_robustness:.2f} | {self.budget.responsibility_gap:.2f} |

"""
        if self.budget.ethical_cliffs:
            md += "## Ethical Cliffs\n\n"
            for cliff in self.budget.ethical_cliffs:
                fatal = "💀" if cliff.is_fatal else "⚠️"
                md += f"- {fatal} **{cliff.concern.value}**: {cliff.description}\n"
                if cliff.mitigation_strategy:
                    md += f"  - 対策: {cliff.mitigation_strategy}\n"
            md += "\n"
        
        if self.is_cleared:
            md += f"""## The Last Responsible Sentence

> {self.text}

"""
        else:
            md += f"""## Rejection

{self.budget.rejection_reason}

"""
        
        return md


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_responsibility(
    ir: ClaimIR,
    metrics: Optional[TubeMetrics] = None,
    context: Optional[str] = None
) -> ResponsibilityBudget:
    """Convenience function to evaluate responsibility."""
    return ResponsibilitySink().evaluate(ir, metrics, context)


def generate_responsible_claim(
    ir: ClaimIR,
    claim: CompiledClaim,
    metrics: Optional[TubeMetrics] = None
) -> ResponsibleClaim:
    """Convenience function to generate responsible claim."""
    sink = ResponsibilitySink()
    budget = sink.evaluate(ir, metrics)
    return sink.generate_responsible_claim(ir, claim, budget)
