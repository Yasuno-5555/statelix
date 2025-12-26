"""
Claim Language Compiler: Controlling What Can Be Said

Not a text generator - a projection from claim space to language.
Compiles mathematical constraints into permissible statements.

"人間が言っていいことを制限する数学"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from copy import deepcopy

try:
    from .claim_budget import ClaimBudget, ClaimStrength
    from .collapse_simulator import CollapseReport, CollapseArchetype
    from .governance_report import GovernanceReport
except ImportError:
    from statelix_py.core.claim_budget import ClaimBudget, ClaimStrength
    from statelix_py.core.collapse_simulator import CollapseReport, CollapseArchetype
    from statelix_py.core.governance_report import GovernanceReport


# =============================================================================
# Claim IR (Intermediate Representation)
# =============================================================================

class ClaimScope(Enum):
    """Scope of claim applicability."""
    SAMPLE_ONLY = "sample_only"
    POPULATION_CONDITIONAL = "population_conditional"
    POPULATION_GENERAL = "population_general"
    UNIVERSAL = "universal"


class ClaimNature(Enum):
    """Nature of the claim."""
    DESCRIPTIVE = "descriptive"
    ASSOCIATIVE = "associative"
    PREDICTIVE = "predictive"
    CAUSAL = "causal"
    PRESCRIPTIVE = "prescriptive"  # Policy implications


@dataclass
class ClaimIR:
    """
    Claim Intermediate Representation - the mathematical claim object.
    
    This is NOT text. This is the claim as a mathematical object
    that gets compiled to text under constraints.
    """
    # Core strength
    strength: ClaimStrength
    robustness_score: float  # 0-1
    
    # Claim properties
    nature: ClaimNature
    scope: ClaimScope
    
    # Content
    effect_name: str
    target_name: str
    effect_magnitude: Optional[float] = None
    effect_direction: Optional[str] = None  # "positive", "negative", "unclear"
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Conditions
    conditions: List[str] = field(default_factory=list)
    assumptions_required: List[str] = field(default_factory=list)
    
    # Forbidden flags
    forbidden_words: Set[str] = field(default_factory=set)
    forbidden_concepts: Set[str] = field(default_factory=set)
    
    # Collapse info
    is_collapsed: bool = False
    collapse_archetype: Optional[CollapseArchetype] = None
    
    def can_claim(self, nature: ClaimNature) -> bool:
        """Check if this nature of claim is allowed."""
        nature_rank = {
            ClaimNature.DESCRIPTIVE: 1,
            ClaimNature.ASSOCIATIVE: 2,
            ClaimNature.PREDICTIVE: 3,
            ClaimNature.CAUSAL: 4,
            ClaimNature.PRESCRIPTIVE: 5
        }
        current_allowed = {
            ClaimStrength.NONE: 0,
            ClaimStrength.MINIMAL: 1,
            ClaimStrength.WEAK: 2,
            ClaimStrength.MODERATE: 3,
            ClaimStrength.STRONG: 4,
            ClaimStrength.DEFINITIVE: 5
        }
        return nature_rank.get(nature, 0) <= current_allowed.get(self.strength, 0)
    
    @classmethod
    def from_governance_report(
        cls,
        report: GovernanceReport,
        effect_name: str = "the effect",
        target_name: str = "the outcome"
    ) -> 'ClaimIR':
        """Create ClaimIR from governance report."""
        budget = report.budget
        
        # Determine nature
        if budget.causal_allowed:
            nature = ClaimNature.CAUSAL
        elif budget.predictive_allowed:
            nature = ClaimNature.PREDICTIVE
        else:
            nature = ClaimNature.ASSOCIATIVE
        
        # Determine scope
        if budget.generalizable:
            scope = ClaimScope.POPULATION_GENERAL
        else:
            scope = ClaimScope.SAMPLE_ONLY
        
        # Build forbidden set
        forbidden_words = set()
        if not budget.causal_allowed:
            forbidden_words.update(['causes', 'causal', 'effect of', 'impact of', 'leads to'])
        if not budget.predictive_allowed:
            forbidden_words.update(['predicts', 'will', 'forecast'])
        if not budget.generalizable:
            forbidden_words.update(['generally', 'in general', 'universally'])
        
        forbidden_concepts = set()
        if budget.max_strength in [ClaimStrength.WEAK, ClaimStrength.MINIMAL]:
            forbidden_concepts.add('policy_implication')
        
        return cls(
            strength=budget.max_strength,
            robustness_score=budget.robustness_score,
            nature=nature,
            scope=scope,
            effect_name=effect_name,
            target_name=target_name,
            conditions=["under maintained assumptions"],
            assumptions_required=[],
            forbidden_words=forbidden_words,
            forbidden_concepts=forbidden_concepts
        )


# =============================================================================
# Dialect Profiles
# =============================================================================

class Dialect(Enum):
    """Output dialect profiles."""
    ACADEMIC_CONSERVATIVE = "academic_conservative"
    REFEREE_SAFE = "referee_safe"
    POLICY_AVERSE = "policy_averse"
    EXPLORATORY_NOTEBOOK = "exploratory_notebook"
    PRESS_RELEASE = "press_release"  # Only for STRONG+
    SILENT = "silent"  # No output


@dataclass
class DialectProfile:
    """
    Defines how a dialect shapes language.
    
    Same meaning, different vocabulary.
    """
    dialect: Dialect
    min_strength: ClaimStrength  # Minimum required to use this dialect
    
    # Vocabulary transformations
    strength_words: Dict[ClaimStrength, List[str]]
    hedging_phrases: List[str]
    forbidden_additions: Set[str]
    
    # Structure
    requires_conditions: bool
    requires_caveats: bool
    max_claim_sentences: int


# Built-in dialect profiles
DIALECT_PROFILES = {
    Dialect.ACADEMIC_CONSERVATIVE: DialectProfile(
        dialect=Dialect.ACADEMIC_CONSERVATIVE,
        min_strength=ClaimStrength.MINIMAL,
        strength_words={
            ClaimStrength.DEFINITIVE: ["demonstrates", "establishes", "confirms"],
            ClaimStrength.STRONG: ["provides evidence", "suggests strongly"],
            ClaimStrength.MODERATE: ["indicates", "is consistent with"],
            ClaimStrength.WEAK: ["may", "appears to", "is potentially"],
            ClaimStrength.MINIMAL: ["we observe", "the data show"],
        },
        hedging_phrases=[
            "under the maintained assumptions",
            "subject to the conditions outlined",
            "with the caveats discussed",
        ],
        forbidden_additions={'proves', 'definitively', 'undoubtedly'},
        requires_conditions=True,
        requires_caveats=True,
        max_claim_sentences=3
    ),
    
    Dialect.REFEREE_SAFE: DialectProfile(
        dialect=Dialect.REFEREE_SAFE,
        min_strength=ClaimStrength.WEAK,
        strength_words={
            ClaimStrength.DEFINITIVE: ["the evidence supports"],
            ClaimStrength.STRONG: ["is consistent with"],
            ClaimStrength.MODERATE: ["may be associated with"],
            ClaimStrength.WEAK: ["patterns suggest"],
        },
        hedging_phrases=[
            "pending replication",
            "within the limitations of this design",
            "further research is needed",
        ],
        forbidden_additions={'proves', 'causes', 'definitively', 'policy'},
        requires_conditions=True,
        requires_caveats=True,
        max_claim_sentences=2
    ),
    
    Dialect.POLICY_AVERSE: DialectProfile(
        dialect=Dialect.POLICY_AVERSE,
        min_strength=ClaimStrength.MODERATE,
        strength_words={
            ClaimStrength.DEFINITIVE: ["research indicates"],
            ClaimStrength.STRONG: ["findings suggest"],
            ClaimStrength.MODERATE: ["results are consistent with"],
        },
        hedging_phrases=[
            "this research does not constitute policy advice",
            "implications for practice require further investigation",
        ],
        forbidden_additions={
            'should', 'must', 'policy', 'implement', 'recommend',
            'intervention', 'action', 'べき', '政策'
        },
        requires_conditions=True,
        requires_caveats=True,
        max_claim_sentences=2
    ),
    
    Dialect.EXPLORATORY_NOTEBOOK: DialectProfile(
        dialect=Dialect.EXPLORATORY_NOTEBOOK,
        min_strength=ClaimStrength.MINIMAL,
        strength_words={
            ClaimStrength.DEFINITIVE: ["strong evidence"],
            ClaimStrength.STRONG: ["good evidence"],
            ClaimStrength.MODERATE: ["some evidence"],
            ClaimStrength.WEAK: ["tentative pattern"],
            ClaimStrength.MINIMAL: ["observation"],
        },
        hedging_phrases=["exploratory analysis", "preliminary"],
        forbidden_additions={'proves', 'definitively'},
        requires_conditions=False,
        requires_caveats=False,
        max_claim_sentences=5
    ),
    
    Dialect.PRESS_RELEASE: DialectProfile(
        dialect=Dialect.PRESS_RELEASE,
        min_strength=ClaimStrength.STRONG,  # Only for robust findings
        strength_words={
            ClaimStrength.DEFINITIVE: ["researchers find", "study shows"],
            ClaimStrength.STRONG: ["new research suggests"],
        },
        hedging_phrases=["according to the study"],
        forbidden_additions={'proves', 'cure', 'breakthrough'},
        requires_conditions=False,
        requires_caveats=True,
        max_claim_sentences=3
    ),
}


# =============================================================================
# Forbidden Gradient
# =============================================================================

@dataclass
class ForbiddenGradient:
    """
    Continuous constraint on language - not binary blacklist.
    
    Each word has a minimum robustness threshold.
    """
    word: str
    min_robustness: float  # 0-1, required robustness to use this word
    min_strength: ClaimStrength
    category: str  # 'causal', 'certainty', 'scope', 'policy'
    
    def is_allowed(self, ir: ClaimIR) -> bool:
        """Check if this word is allowed given the ClaimIR."""
        if ir.robustness_score < self.min_robustness:
            return False
        strength_rank = {
            ClaimStrength.NONE: 0, ClaimStrength.MINIMAL: 1,
            ClaimStrength.WEAK: 2, ClaimStrength.MODERATE: 3,
            ClaimStrength.STRONG: 4, ClaimStrength.DEFINITIVE: 5
        }
        return strength_rank.get(ir.strength, 0) >= strength_rank.get(self.min_strength, 0)


# Default forbidden gradient
DEFAULT_FORBIDDEN_GRADIENT = [
    # Causal language
    ForbiddenGradient("causes", 0.9, ClaimStrength.DEFINITIVE, "causal"),
    ForbiddenGradient("causal effect", 0.8, ClaimStrength.STRONG, "causal"),
    ForbiddenGradient("leads to", 0.7, ClaimStrength.STRONG, "causal"),
    ForbiddenGradient("affects", 0.6, ClaimStrength.MODERATE, "causal"),
    ForbiddenGradient("influences", 0.5, ClaimStrength.MODERATE, "causal"),
    ForbiddenGradient("associated with", 0.2, ClaimStrength.WEAK, "causal"),
    ForbiddenGradient("correlated with", 0.1, ClaimStrength.MINIMAL, "causal"),
    
    # Certainty language
    ForbiddenGradient("proves", 1.0, ClaimStrength.DEFINITIVE, "certainty"),
    ForbiddenGradient("demonstrates", 0.8, ClaimStrength.STRONG, "certainty"),
    ForbiddenGradient("shows", 0.5, ClaimStrength.MODERATE, "certainty"),
    ForbiddenGradient("suggests", 0.3, ClaimStrength.WEAK, "certainty"),
    ForbiddenGradient("may indicate", 0.2, ClaimStrength.WEAK, "certainty"),
    ForbiddenGradient("is consistent with", 0.1, ClaimStrength.MINIMAL, "certainty"),
    
    # Scope language
    ForbiddenGradient("universally", 0.95, ClaimStrength.DEFINITIVE, "scope"),
    ForbiddenGradient("in general", 0.7, ClaimStrength.STRONG, "scope"),
    ForbiddenGradient("typically", 0.5, ClaimStrength.MODERATE, "scope"),
    ForbiddenGradient("in this context", 0.2, ClaimStrength.WEAK, "scope"),
    ForbiddenGradient("in this sample", 0.0, ClaimStrength.MINIMAL, "scope"),
    
    # Policy language
    ForbiddenGradient("should implement", 1.0, ClaimStrength.DEFINITIVE, "policy"),
    ForbiddenGradient("policy implication", 0.8, ClaimStrength.STRONG, "policy"),
    ForbiddenGradient("may inform", 0.5, ClaimStrength.MODERATE, "policy"),
    ForbiddenGradient("further research needed", 0.0, ClaimStrength.MINIMAL, "policy"),
]


class ForbiddenGradientChecker:
    """
    Checks text against the forbidden gradient.
    """
    
    def __init__(self, gradient: Optional[List[ForbiddenGradient]] = None):
        self.gradient = gradient or DEFAULT_FORBIDDEN_GRADIENT
    
    def check_text(self, text: str, ir: ClaimIR) -> List[Tuple[str, bool, str]]:
        """
        Check text for gradient violations.
        
        Returns:
            List of (word, is_violation, category)
        """
        text_lower = text.lower()
        results = []
        
        for fg in self.gradient:
            if fg.word.lower() in text_lower:
                is_violation = not fg.is_allowed(ir)
                results.append((fg.word, is_violation, fg.category))
        
        return results
    
    def get_allowed_words(self, ir: ClaimIR, category: str) -> List[str]:
        """Get words allowed for this IR in a category."""
        return [fg.word for fg in self.gradient 
                if fg.category == category and fg.is_allowed(ir)]
    
    def get_strongest_allowed(self, ir: ClaimIR, category: str) -> Optional[str]:
        """Get the strongest allowed word in a category."""
        allowed = [fg for fg in self.gradient 
                   if fg.category == category and fg.is_allowed(ir)]
        if not allowed:
            return None
        # Return highest robustness threshold that's still allowed
        return max(allowed, key=lambda fg: fg.min_robustness).word


# =============================================================================
# Claim Compiler
# =============================================================================

@dataclass
class CompiledClaim:
    """
    The output of claim compilation - controlled text.
    """
    text: str
    dialect: Dialect
    ir: ClaimIR
    
    # Validation
    gradient_violations: List[str] = field(default_factory=list)
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    
    # Metadata
    words_used: List[str] = field(default_factory=list)
    hedges_used: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "✓" if self.is_valid else f"✗ ({self.rejection_reason})"
        return f"CompiledClaim({self.dialect.value}, {status})"


class ClaimCompiler:
    """
    Compiles ClaimIR to controlled text under dialect constraints.
    
    Not a text generator - a projection from claim space to language.
    
    Example:
        >>> compiler = ClaimCompiler()
        >>> ir = ClaimIR.from_governance_report(report)
        >>> claim = compiler.compile(ir, dialect=Dialect.ACADEMIC_CONSERVATIVE)
        >>> 
        >>> if claim.is_valid:
        ...     print(claim.text)
        >>> else:
        ...     print(f"Cannot generate: {claim.rejection_reason}")
    """
    
    def __init__(self):
        self.gradient_checker = ForbiddenGradientChecker()
        self.profiles = DIALECT_PROFILES
    
    def compile(
        self,
        ir: ClaimIR,
        dialect: Dialect = Dialect.ACADEMIC_CONSERVATIVE,
        collapse_report: Optional[CollapseReport] = None
    ) -> CompiledClaim:
        """
        Compile ClaimIR to text.
        
        Args:
            ir: The claim intermediate representation
            dialect: Output dialect
            collapse_report: Optional collapse info
        
        Returns:
            CompiledClaim with text (or rejection)
        """
        # Check for SILENT mode
        if dialect == Dialect.SILENT:
            return CompiledClaim(
                text="",
                dialect=dialect,
                ir=ir,
                is_valid=False,
                rejection_reason="Silent mode - no output permitted"
            )
        
        # Check strength requirement
        profile = self.profiles.get(dialect)
        if profile is None:
            return CompiledClaim(
                text="",
                dialect=dialect,
                ir=ir,
                is_valid=False,
                rejection_reason=f"Unknown dialect: {dialect.value}"
            )
        
        if not self._meets_min_strength(ir.strength, profile.min_strength):
            return CompiledClaim(
                text="",
                dialect=dialect,
                ir=ir,
                is_valid=False,
                rejection_reason=f"Insufficient strength for {dialect.value}. Required: {profile.min_strength.value}"
            )
        
        # Check for collapse
        if ir.is_collapsed and collapse_report:
            if collapse_report.schedule.final_strength == ClaimStrength.NONE:
                return CompiledClaim(
                    text="この段階では、結論文を生成できません。",
                    dialect=dialect,
                    ir=ir,
                    is_valid=False,
                    rejection_reason="Collapsed beyond claimable state"
                )
        
        # Generate text
        text, words_used, hedges = self._generate_text(ir, profile)
        
        # Check gradient violations
        violations = self.gradient_checker.check_text(text, ir)
        violation_words = [v[0] for v in violations if v[1]]
        
        # If violations, try to sanitize
        if violation_words:
            text = self._sanitize_text(text, ir, violation_words)
            # Re-check
            violations = self.gradient_checker.check_text(text, ir)
            violation_words = [v[0] for v in violations if v[1]]
        
        return CompiledClaim(
            text=text,
            dialect=dialect,
            ir=ir,
            gradient_violations=violation_words,
            is_valid=len(violation_words) == 0,
            rejection_reason="Gradient violations" if violation_words else None,
            words_used=words_used,
            hedges_used=hedges
        )
    
    def _meets_min_strength(
        self, 
        strength: ClaimStrength, 
        required: ClaimStrength
    ) -> bool:
        """Check if strength meets minimum."""
        ranks = {
            ClaimStrength.NONE: 0, ClaimStrength.MINIMAL: 1,
            ClaimStrength.WEAK: 2, ClaimStrength.MODERATE: 3,
            ClaimStrength.STRONG: 4, ClaimStrength.DEFINITIVE: 5
        }
        return ranks.get(strength, 0) >= ranks.get(required, 0)
    
    def _generate_text(
        self,
        ir: ClaimIR,
        profile: DialectProfile
    ) -> Tuple[str, List[str], List[str]]:
        """Generate text from IR and profile."""
        words_used = []
        hedges_used = []
        
        # Get strength word
        strength_words = profile.strength_words.get(ir.strength, [""])
        verb = strength_words[0] if strength_words else "is observed"
        words_used.append(verb)
        
        # Get scope qualifier
        scope_word = self.gradient_checker.get_strongest_allowed(ir, "scope")
        if scope_word and ir.scope != ClaimScope.SAMPLE_ONLY:
            scope_clause = f" {scope_word},"
        else:
            scope_clause = " In this analysis,"
        
        # Build effect description
        if ir.effect_magnitude is not None:
            effect_desc = f"{ir.effect_name} {verb} {ir.effect_direction or 'related to'} {ir.target_name} (magnitude: {ir.effect_magnitude:.3f})"
        else:
            effect_desc = f"{ir.effect_name} {verb} {ir.target_name}"
        
        # Add hedging if required
        hedge_clause = ""
        if profile.requires_caveats and profile.hedging_phrases:
            hedge = profile.hedging_phrases[0]
            hedge_clause = f", {hedge}"
            hedges_used.append(hedge)
        
        # Build conditions
        condition_clause = ""
        if profile.requires_conditions and ir.conditions:
            condition_clause = f" ({', '.join(ir.conditions[:2])})"
        
        # Assemble
        text = f"{scope_clause} {effect_desc}{condition_clause}{hedge_clause}."
        
        return text.strip(), words_used, hedges_used
    
    def _sanitize_text(
        self,
        text: str,
        ir: ClaimIR,
        violations: List[str]
    ) -> str:
        """Replace violating words with allowed alternatives."""
        result = text
        
        for word in violations:
            # Find category
            category = None
            for fg in self.gradient_checker.gradient:
                if fg.word.lower() == word.lower():
                    category = fg.category
                    break
            
            if category:
                # Get strongest allowed in same category
                replacement = self.gradient_checker.get_strongest_allowed(ir, category)
                if replacement and replacement != word:
                    # Case-preserving replacement
                    result = result.replace(word, replacement)
                    result = result.replace(word.capitalize(), replacement.capitalize())
        
        return result
    
    def compile_all_dialects(self, ir: ClaimIR) -> Dict[Dialect, CompiledClaim]:
        """Compile to all dialects, showing which succeed."""
        results = {}
        for dialect in Dialect:
            if dialect != Dialect.SILENT:
                results[dialect] = self.compile(ir, dialect)
        return results


# =============================================================================
# Silent Mode
# =============================================================================

class SilentMode:
    """
    The final guardian - when nothing can be said.
    
    RED judgment → no output at all.
    """
    
    SILENT_MESSAGE = "この分析は、結論を語る資格を満たしていません。"
    SILENT_MESSAGE_EN = "This analysis does not meet the requirements to state conclusions."
    
    @classmethod
    def should_silence(cls, governance_report: GovernanceReport) -> bool:
        """Check if silent mode should be activated."""
        return governance_report.verdict == "RED"
    
    @classmethod
    def get_silent_output(cls, lang: str = "ja") -> str:
        """Get the silent mode message."""
        if lang == "en":
            return cls.SILENT_MESSAGE_EN
        return cls.SILENT_MESSAGE
    
    @classmethod
    def enforce(
        cls,
        governance_report: GovernanceReport,
        ir: ClaimIR
    ) -> Optional[CompiledClaim]:
        """
        Enforce silent mode if necessary.
        
        Returns:
            CompiledClaim with silent message, or None if not silenced
        """
        if cls.should_silence(governance_report):
            return CompiledClaim(
                text=cls.SILENT_MESSAGE,
                dialect=Dialect.SILENT,
                ir=ir,
                is_valid=False,
                rejection_reason="RED judgment - silent mode enforced"
            )
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_claim(
    governance_report: GovernanceReport,
    dialect: Dialect = Dialect.ACADEMIC_CONSERVATIVE,
    effect_name: str = "the effect",
    target_name: str = "the outcome",
    collapse_report: Optional[CollapseReport] = None
) -> CompiledClaim:
    """
    Compile a claim from governance report.
    
    Example:
        >>> claim = compile_claim(report, dialect=Dialect.REFEREE_SAFE)
        >>> if claim.is_valid:
        ...     print(claim.text)
    """
    # Check silent mode first
    ir = ClaimIR.from_governance_report(governance_report, effect_name, target_name)
    
    silent = SilentMode.enforce(governance_report, ir)
    if silent:
        return silent
    
    compiler = ClaimCompiler()
    return compiler.compile(ir, dialect, collapse_report)
