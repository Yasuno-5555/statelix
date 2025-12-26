"""
Tests for Claim Language Compiler
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.claim_compiler import (
    ClaimIR, ClaimScope, ClaimNature,
    Dialect, DialectProfile, DIALECT_PROFILES,
    ForbiddenGradient, ForbiddenGradientChecker, DEFAULT_FORBIDDEN_GRADIENT,
    ClaimCompiler, CompiledClaim,
    SilentMode,
    compile_claim
)
from core.reviewer_attack import (
    ReviewerAttackSimulator, ReviewerPersona, ReviewerAttack, AttackReport,
    simulate_reviewer_attacks
)
from core.claim_budget import ClaimBudget, ClaimStrength
from core.governance_report import GovernanceReport
from core.assumption_path import AssumptionState


# =============================================================================
# ClaimIR Tests
# =============================================================================

class TestClaimIR:
    """Tests for ClaimIR."""
    
    def test_creation(self):
        """Test basic creation."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="treatment",
            target_name="outcome"
        )
        assert ir.strength == ClaimStrength.MODERATE
        assert ir.nature == ClaimNature.ASSOCIATIVE
    
    def test_can_claim_hierarchy(self):
        """Test claim nature hierarchy."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        assert ir.can_claim(ClaimNature.DESCRIPTIVE)
        assert ir.can_claim(ClaimNature.ASSOCIATIVE)
        assert ir.can_claim(ClaimNature.PREDICTIVE)
        assert not ir.can_claim(ClaimNature.CAUSAL)
        assert not ir.can_claim(ClaimNature.PRESCRIPTIVE)
    
    def test_from_governance_report(self):
        """Test creation from governance report."""
        # Create minimal governance report
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            causal_allowed=False,
            predictive_allowed=True,
            generalizable=False,
            warnings=[],
            budget_remaining=1.0
        )
        
        # Mock governance report
        class MockReport:
            def __init__(self):
                self.budget = budget
                self.verdict = "YELLOW"
        
        ir = ClaimIR.from_governance_report(MockReport(), effect_name="X")
        
        assert ir.strength == ClaimStrength.MODERATE
        assert 'causes' in ir.forbidden_words


# =============================================================================
# DialectProfile Tests
# =============================================================================

class TestDialectProfile:
    """Tests for dialect profiles."""
    
    def test_all_profiles_exist(self):
        """Test that all dialect profiles are defined."""
        assert Dialect.ACADEMIC_CONSERVATIVE in DIALECT_PROFILES
        assert Dialect.REFEREE_SAFE in DIALECT_PROFILES
        assert Dialect.POLICY_AVERSE in DIALECT_PROFILES
    
    def test_min_strength_requirements(self):
        """Test minimum strength requirements."""
        assert DIALECT_PROFILES[Dialect.PRESS_RELEASE].min_strength == ClaimStrength.STRONG
        assert DIALECT_PROFILES[Dialect.ACADEMIC_CONSERVATIVE].min_strength == ClaimStrength.MINIMAL


# =============================================================================
# ForbiddenGradient Tests
# =============================================================================

class TestForbiddenGradient:
    """Tests for ForbiddenGradient."""
    
    def test_is_allowed(self):
        """Test word allowance checking."""
        fg = ForbiddenGradient("causes", 0.9, ClaimStrength.DEFINITIVE, "causal")
        
        strong_ir = ClaimIR(
            strength=ClaimStrength.DEFINITIVE,
            robustness_score=0.95,
            nature=ClaimNature.CAUSAL,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="X",
            target_name="Y"
        )
        assert fg.is_allowed(strong_ir)
        
        weak_ir = ClaimIR(
            strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        assert not fg.is_allowed(weak_ir)
    
    def test_gradient_checker(self):
        """Test gradient checker."""
        checker = ForbiddenGradientChecker()
        
        ir = ClaimIR(
            strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        text = "X causes Y and proves Z"
        results = checker.check_text(text, ir)
        
        # Should find "causes" and "proves" as violations
        violations = [r for r in results if r[1]]
        assert len(violations) >= 1
    
    def test_get_allowed_words(self):
        """Test getting allowed words."""
        checker = ForbiddenGradientChecker()
        
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        allowed = checker.get_allowed_words(ir, "certainty")
        assert len(allowed) > 0
        assert "proves" not in allowed


# =============================================================================
# ClaimCompiler Tests
# =============================================================================

class TestClaimCompiler:
    """Tests for ClaimCompiler."""
    
    def test_basic_compilation(self):
        """Test basic claim compilation."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="treatment effect",
            target_name="outcome"
        )
        
        compiler = ClaimCompiler()
        claim = compiler.compile(ir, Dialect.ACADEMIC_CONSERVATIVE)
        
        assert claim is not None
        assert len(claim.text) > 0
    
    def test_insufficient_strength_rejection(self):
        """Test rejection when strength insufficient for dialect."""
        ir = ClaimIR(
            strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        compiler = ClaimCompiler()
        claim = compiler.compile(ir, Dialect.PRESS_RELEASE)
        
        assert not claim.is_valid
        assert "Insufficient strength" in claim.rejection_reason
    
    def test_silent_mode(self):
        """Test silent mode rejection."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        compiler = ClaimCompiler()
        claim = compiler.compile(ir, Dialect.SILENT)
        
        assert not claim.is_valid
        assert claim.text == ""
    
    def test_all_dialects_compilation(self):
        """Test compiling to all dialects."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        compiler = ClaimCompiler()
        results = compiler.compile_all_dialects(ir)
        
        assert len(results) >= 4  # All non-silent dialects


# =============================================================================
# SilentMode Tests
# =============================================================================

class TestSilentMode:
    """Tests for SilentMode."""
    
    def test_should_silence_red(self):
        """Test silencing on RED verdict."""
        class MockReport:
            verdict = "RED"
        
        assert SilentMode.should_silence(MockReport())
    
    def test_should_not_silence_green(self):
        """Test not silencing on GREEN verdict."""
        class MockReport:
            verdict = "GREEN"
        
        assert not SilentMode.should_silence(MockReport())
    
    def test_silent_message(self):
        """Test silent message content."""
        msg_ja = SilentMode.get_silent_output("ja")
        msg_en = SilentMode.get_silent_output("en")
        
        assert "資格" in msg_ja or "満たして" in msg_ja
        assert "does not meet" in msg_en


# =============================================================================
# ReviewerAttackSimulator Tests
# =============================================================================

class TestReviewerAttackSimulator:
    """Tests for ReviewerAttackSimulator."""
    
    def test_attack_causal_claim(self):
        """Test attacks on causal claims."""
        ir = ClaimIR(
            strength=ClaimStrength.STRONG,
            robustness_score=0.7,
            nature=ClaimNature.CAUSAL,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="X",
            target_name="Y"
        )
        
        claim = CompiledClaim(
            text="X causes Y with strong effect",
            dialect=Dialect.ACADEMIC_CONSERVATIVE,
            ir=ir
        )
        
        sim = ReviewerAttackSimulator()
        report = sim.attack(claim, ir)
        
        assert len(report.attacks) > 0
        # Should have causal skeptic attacks
        causal_attacks = [a for a in report.attacks 
                         if a.persona == ReviewerPersona.CAUSAL_SKEPTIC]
        assert len(causal_attacks) > 0
    
    def test_survival_probability(self):
        """Test survival probability calculation."""
        ir = ClaimIR(
            strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        claim = CompiledClaim(
            text="X may be associated with Y",
            dialect=Dialect.ACADEMIC_CONSERVATIVE,
            ir=ir
        )
        
        sim = ReviewerAttackSimulator()
        report = sim.attack(claim, ir)
        
        assert 0 <= report.survival_probability <= 1
    
    def test_markdown_report(self):
        """Test markdown report generation."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        claim = CompiledClaim(
            text="X is associated with Y",
            dialect=Dialect.ACADEMIC_CONSERVATIVE,
            ir=ir
        )
        
        report = simulate_reviewer_attacks(claim, ir)
        md = report.to_markdown()
        
        assert "Reviewer Attack" in md


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full claim compilation pipeline."""
        # Create mock governance report
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            causal_allowed=False,
            predictive_allowed=True,
            generalizable=False,
            warnings=["Low robustness"],
            budget_remaining=1.0
        )
        
        class MockReport:
            def __init__(self):
                self.budget = budget
                self.verdict = "YELLOW"
        
        claim = compile_claim(
            MockReport(),
            dialect=Dialect.REFEREE_SAFE,
            effect_name="treatment"
        )
        
        assert claim is not None
    
    def test_silent_enforcement(self):
        """Test silent mode enforcement in compile_claim."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.NONE,
            robustness_score=0.05,
            causal_allowed=False,
            predictive_allowed=False,
            generalizable=False,
            warnings=["Critical failure"],
            budget_remaining=0
        )
        
        class MockReport:
            def __init__(self):
                self.budget = budget
                self.verdict = "RED"
        
        claim = compile_claim(MockReport())
        
        assert not claim.is_valid
        assert claim.dialect == Dialect.SILENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
