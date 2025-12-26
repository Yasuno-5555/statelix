"""
Reviewer Attack Simulator: How Will Your Claims Be Destroyed?

Simulates hostile reviewer questioning to reveal where claims are vulnerable.

"この文、外生性を仮定してますよね？"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

try:
    from .claim_compiler import ClaimIR, CompiledClaim, ClaimNature, ClaimScope
    from .claim_budget import ClaimStrength
except ImportError:
    from statelix_py.core.claim_compiler import ClaimIR, CompiledClaim, ClaimNature, ClaimScope
    from statelix_py.core.claim_budget import ClaimStrength


class ReviewerPersona(Enum):
    """Types of hostile reviewers."""
    CAUSAL_SKEPTIC = "causal_skeptic"
    METHODOLOGY_PEDANT = "methodology_pedant"
    SCOPE_CHALLENGER = "scope_challenger"
    REPLICATION_ADVOCATE = "replication_advocate"
    POLICY_CRITIC = "policy_critic"


@dataclass
class ReviewerAttack:
    """
    A simulated reviewer attack on a claim.
    """
    persona: ReviewerPersona
    attack_text: str
    target_aspect: str  # Which part of the claim is attacked
    severity: float  # 0-1, how damaging this attack is
    requires_assumption: Optional[str] = None
    suggested_defense: Optional[str] = None
    
    @property
    def is_fatal(self) -> bool:
        """Is this attack likely fatal to the claim?"""
        return self.severity > 0.7


@dataclass
class AttackReport:
    """
    Full report of simulated reviewer attacks.
    """
    claim: CompiledClaim
    attacks: List[ReviewerAttack]
    survival_probability: float  # 0-1
    weakest_point: Optional[str] = None
    
    def fatal_attacks(self) -> List[ReviewerAttack]:
        return [a for a in self.attacks if a.is_fatal]
    
    def to_markdown(self) -> str:
        md = f"""# Reviewer Attack Simulation

**Survival Probability:** {self.survival_probability:.0%}
**Weakest Point:** {self.weakest_point or 'Unknown'}

---

## Attacks

"""
        for i, attack in enumerate(self.attacks, 1):
            fatal = "💀" if attack.is_fatal else "⚠️"
            md += f"### {i}. {attack.persona.value} {fatal}\n\n"
            md += f"> {attack.attack_text}\n\n"
            md += f"- **Target:** {attack.target_aspect}\n"
            md += f"- **Severity:** {attack.severity:.0%}\n"
            if attack.suggested_defense:
                md += f"- **Defense:** {attack.suggested_defense}\n"
            md += "\n"
        
        return md


class ReviewerAttackSimulator:
    """
    Simulates hostile reviewer questioning.
    
    Example:
        >>> sim = ReviewerAttackSimulator()
        >>> report = sim.attack(compiled_claim, ir)
        >>> 
        >>> for attack in report.fatal_attacks():
        ...     print(f"FATAL: {attack.attack_text}")
    """
    
    ATTACKS = {
        ReviewerPersona.CAUSAL_SKEPTIC: [
            ("causal", "この因果的主張を支持する識別戦略は何ですか？", "causal_claim", 0.8, "exogeneity"),
            ("causal", "逆因果の可能性を排除できますか？", "causal_direction", 0.7, None),
            ("causes", "「因果」という言葉は強すぎませんか？", "word_choice", 0.6, None),
            ("effect", "これは因果効果ですか、それとも相関ですか？", "interpretation", 0.75, "independence"),
        ],
        ReviewerPersona.METHODOLOGY_PEDANT: [
            ("linear", "線形性の仮定は満たされていますか？", "linearity", 0.5, "linearity"),
            ("normal", "残差の正規性を検定しましたか？", "normality", 0.4, "normality"),
            ("robust", "頑健性チェックは行いましたか？", "robustness", 0.6, None),
            ("standard", "標準誤差は適切に計算されていますか？", "inference", 0.5, "homoscedasticity"),
        ],
        ReviewerPersona.SCOPE_CHALLENGER: [
            ("general", "この結果は一般化できますか？", "generalization", 0.6, None),
            ("population", "対象母集団は何ですか？", "scope", 0.5, None),
            ("sample", "サンプル選択バイアスはありませんか？", "selection", 0.7, None),
            ("context", "他の文脈でも成り立ちますか？", "external_validity", 0.55, None),
        ],
        ReviewerPersona.REPLICATION_ADVOCATE: [
            ("data", "データとコードは公開されますか？", "reproducibility", 0.3, None),
            ("result", "事前登録はされていますか？", "preregistration", 0.4, None),
            ("significant", "p-hackingの可能性はありませんか？", "multiple_testing", 0.65, None),
        ],
        ReviewerPersona.POLICY_CRITIC: [
            ("policy", "政策含意を述べるには証拠が不十分では？", "policy_leap", 0.8, None),
            ("should", "「べき」という表現は研究の範囲を超えています。", "normative", 0.75, None),
            ("recommend", "推奨を行う根拠は何ですか？", "prescription", 0.7, None),
            ("implement", "実装を示唆するのは時期尚早では？", "intervention", 0.65, None),
        ],
    }
    
    def attack(self, claim: CompiledClaim, ir: ClaimIR) -> AttackReport:
        """
        Simulate attacks on a compiled claim.
        
        Args:
            claim: The compiled claim to attack
            ir: The claim IR
        
        Returns:
            AttackReport with simulated attacks
        """
        attacks = []
        text_lower = claim.text.lower()
        
        for persona, attack_templates in self.ATTACKS.items():
            for trigger, question, target, base_severity, assumption in attack_templates:
                # Check if attack is relevant
                if trigger.lower() in text_lower or self._is_concept_present(ir, trigger):
                    # Adjust severity based on IR
                    severity = self._adjust_severity(base_severity, ir, assumption)
                    
                    defense = self._suggest_defense(ir, assumption, target)
                    
                    attack = ReviewerAttack(
                        persona=persona,
                        attack_text=question,
                        target_aspect=target,
                        severity=severity,
                        requires_assumption=assumption,
                        suggested_defense=defense
                    )
                    attacks.append(attack)
        
        # Calculate survival
        if attacks:
            max_severity = max(a.severity for a in attacks)
            avg_severity = sum(a.severity for a in attacks) / len(attacks)
            survival = 1.0 - (max_severity * 0.6 + avg_severity * 0.4)
        else:
            survival = 0.95
        
        # Find weakest point
        weakest = None
        if attacks:
            worst = max(attacks, key=lambda a: a.severity)
            weakest = worst.target_aspect
        
        return AttackReport(
            claim=claim,
            attacks=attacks,
            survival_probability=max(0, survival),
            weakest_point=weakest
        )
    
    def _is_concept_present(self, ir: ClaimIR, concept: str) -> bool:
        """Check if a concept is implied by the IR."""
        concept_map = {
            'causal': ir.nature == ClaimNature.CAUSAL,
            'policy': ir.nature == ClaimNature.PRESCRIPTIVE,
            'general': ir.scope == ClaimScope.POPULATION_GENERAL,
            'linear': 'linearity' not in ir.assumptions_required,
        }
        return concept_map.get(concept.lower(), False)
    
    def _adjust_severity(
        self, 
        base: float, 
        ir: ClaimIR, 
        assumption: Optional[str]
    ) -> float:
        """Adjust attack severity based on IR strength."""
        severity = base
        
        # Stronger claims are harder to defend
        strength_modifier = {
            ClaimStrength.NONE: -0.3,
            ClaimStrength.MINIMAL: -0.2,
            ClaimStrength.WEAK: -0.1,
            ClaimStrength.MODERATE: 0.0,
            ClaimStrength.STRONG: 0.1,
            ClaimStrength.DEFINITIVE: 0.2,
        }
        severity += strength_modifier.get(ir.strength, 0)
        
        # If assumption is in forbidden list, severity increases
        if assumption and assumption in ir.forbidden_concepts:
            severity += 0.15
        
        return min(1.0, max(0.0, severity))
    
    def _suggest_defense(
        self,
        ir: ClaimIR,
        assumption: Optional[str],
        target: str
    ) -> Optional[str]:
        """Suggest a defense for this attack."""
        if ir.robustness_score > 0.7:
            return "頑健性分析により、結果は安定していることを示せます。"
        
        if target == "scope":
            return "サンプルの限界を明示的に認めることで対応できます。"
        
        if target == "causal_claim":
            if ir.nature != ClaimNature.CAUSAL:
                return "因果的解釈を避け、相関としてのみ解釈してください。"
            return "識別戦略の詳細な説明を追加してください。"
        
        return None


def simulate_reviewer_attacks(
    claim: CompiledClaim,
    ir: ClaimIR
) -> AttackReport:
    """Convenience function to simulate attacks."""
    return ReviewerAttackSimulator().attack(claim, ir)
