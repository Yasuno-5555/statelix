"""
Tests for Responsibility Sink
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.responsibility_sink import (
    ResponsibilitySink,
    ImpactClass,
    EthicalConcern,
    EthicalCliff,
    ResponsibilityBudget,
    ResponsibleClaim,
    evaluate_responsibility,
    generate_responsible_claim
)
from core.claim_compiler import ClaimIR, CompiledClaim, ClaimNature, ClaimScope, Dialect
from core.claim_budget import ClaimStrength
from core.assumption_tube import TubeMetrics


# =============================================================================
# ImpactClass Tests
# =============================================================================

class TestImpactClass:
    """Tests for ImpactClass."""
    
    def test_risk_multipliers(self):
        """Test risk multipliers are increasing."""
        assert ImpactClass.ACADEMIC_ONLY.risk_multiplier == 1.0
        assert ImpactClass.POLICY_TRIGGERING.risk_multiplier > 1.0
        assert ImpactClass.INDIVIDUAL_TARGETING.risk_multiplier >= 5.0
    
    def test_descriptions(self):
        """Test descriptions exist."""
        for ic in ImpactClass:
            assert len(ic.description) > 0


# =============================================================================
# EthicalCliff Tests
# =============================================================================

class TestEthicalCliff:
    """Tests for EthicalCliff."""
    
    def test_is_fatal(self):
        """Test fatal cliff detection."""
        fatal = EthicalCliff(
            concern=EthicalConcern.GROUP_ESSENTIALISM,
            severity=0.85,
            description="Test",
            mitigation_possible=False
        )
        assert fatal.is_fatal
        
        non_fatal = EthicalCliff(
            concern=EthicalConcern.DISCRIMINATION_RISK,
            severity=0.5,
            description="Test",
            mitigation_possible=True
        )
        assert not non_fatal.is_fatal


# =============================================================================
# ResponsibilityBudget Tests
# =============================================================================

class TestResponsibilityBudget:
    """Tests for ResponsibilityBudget."""
    
    def test_responsibility_gap(self):
        """Test gap calculation."""
        budget = ResponsibilityBudget(
            impact_class=ImpactClass.POLICY_TRIGGERING,
            required_robustness=0.6,
            actual_robustness=0.4
        )
        assert abs(budget.responsibility_gap - 0.2) < 0.01
    
    def test_can_proceed(self):
        """Test can_proceed logic."""
        # Can proceed with small gap
        good = ResponsibilityBudget(
            impact_class=ImpactClass.ACADEMIC_ONLY,
            required_robustness=0.3,
            actual_robustness=0.35,
            is_responsible=True
        )
        assert good.can_proceed
        
        # Cannot proceed with large gap
        bad = ResponsibilityBudget(
            impact_class=ImpactClass.POLICY_TRIGGERING,
            required_robustness=0.8,
            actual_robustness=0.3,
            is_responsible=True
        )
        assert not bad.can_proceed
    
    def test_cannot_proceed_with_fatal_cliff(self):
        """Test that fatal cliffs block proceeding."""
        budget = ResponsibilityBudget(
            impact_class=ImpactClass.ACADEMIC_ONLY,
            required_robustness=0.3,
            actual_robustness=0.5,
            is_responsible=True,
            ethical_cliffs=[
                EthicalCliff(
                    concern=EthicalConcern.GROUP_ESSENTIALISM,
                    severity=0.9,
                    description="Fatal",
                    mitigation_possible=False
                )
            ]
        )
        assert not budget.can_proceed


# =============================================================================
# ResponsibilitySink Tests
# =============================================================================

class TestResponsibilitySink:
    """Tests for ResponsibilitySink."""
    
    def test_academic_classification(self):
        """Test academic-only classification."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="correlation coefficient",
            target_name="variables"
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir)
        
        assert budget.impact_class == ImpactClass.ACADEMIC_ONLY
    
    def test_policy_classification(self):
        """Test policy-triggering classification."""
        ir = ClaimIR(
            strength=ClaimStrength.STRONG,
            robustness_score=0.7,
            nature=ClaimNature.PRESCRIPTIVE,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="government intervention",
            target_name="outcome"
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir)
        
        assert budget.impact_class == ImpactClass.POLICY_TRIGGERING
    
    def test_individual_classification(self):
        """Test individual-targeting classification."""
        ir = ClaimIR(
            strength=ClaimStrength.STRONG,
            robustness_score=0.8,
            nature=ClaimNature.CAUSAL,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="criminal prediction",
            target_name="individual risk"
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir)
        
        assert budget.impact_class == ImpactClass.INDIVIDUAL_TARGETING
        assert not budget.is_responsible  # Should be rejected
    
    def test_ethical_cliff_detection(self):
        """Test ethical cliff detection."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="inherent ability",
            target_name="group outcome"
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir, context="biological essential")
        
        assert len(budget.ethical_cliffs) > 0
        assert any(c.concern == EthicalConcern.GROUP_ESSENTIALISM 
                  for c in budget.ethical_cliffs)
    
    def test_required_robustness_scaling(self):
        """Test that required robustness scales with impact."""
        sink = ResponsibilitySink()
        
        academic_ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        policy_ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.PRESCRIPTIVE,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="policy intervention",
            target_name="outcome"
        )
        
        academic_budget = sink.evaluate(academic_ir)
        policy_budget = sink.evaluate(policy_ir)
        
        assert policy_budget.required_robustness > academic_budget.required_robustness


# =============================================================================
# ResponsibleClaim Tests
# =============================================================================

class TestResponsibleClaim:
    """Tests for ResponsibleClaim generation."""
    
    def test_generate_responsible_claim(self):
        """Test responsible claim generation."""
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
            ir=ir,
            is_valid=True
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir)
        responsible = sink.generate_responsible_claim(ir, claim, budget)
        
        assert responsible.is_cleared
        assert len(responsible.text) > 0
    
    def test_rejected_claim(self):
        """Test rejection of irresponsible claims."""
        ir = ClaimIR(
            strength=ClaimStrength.STRONG,
            robustness_score=0.8,
            nature=ClaimNature.CAUSAL,
            scope=ClaimScope.POPULATION_GENERAL,
            effect_name="predict criminal",
            target_name="individual"
        )
        
        claim = CompiledClaim(
            text="We can predict criminals",
            dialect=Dialect.ACADEMIC_CONSERVATIVE,
            ir=ir,
            is_valid=True
        )
        
        sink = ResponsibilitySink()
        budget = sink.evaluate(ir)
        responsible = sink.generate_responsible_claim(ir, claim, budget)
        
        assert not responsible.is_cleared
    
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
            ir=ir,
            is_valid=True
        )
        
        responsible = generate_responsible_claim(ir, claim)
        md = responsible.to_markdown()
        
        assert "Responsibility Report" in md
        assert "Impact Classification" in md


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_evaluate_responsibility(self):
        """Test evaluate_responsibility function."""
        ir = ClaimIR(
            strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            nature=ClaimNature.ASSOCIATIVE,
            scope=ClaimScope.SAMPLE_ONLY,
            effect_name="X",
            target_name="Y"
        )
        
        budget = evaluate_responsibility(ir)
        assert budget is not None
        assert budget.impact_class is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
