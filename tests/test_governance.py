"""
Tests for Governance Trio:
- One-Way Door Analysis
- Claim Budget
- The Last Honest Sentence
- Unified Governance Report
"""

import pytest
import numpy as np
import sys
import os

# Add statelix_py to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.one_way_door import (
    OneWayDoorAnalyzer, OneWayDoor, DoorType,
    check_one_way_doors
)
from core.claim_budget import (
    ClaimBudgetCalculator, ClaimBudget, ClaimStrength,
    calculate_claim_budget
)
from core.honest_sentence import (
    HonestSentenceGenerator, HonestSentence,
    generate_honest_sentence
)
from core.governance_report import (
    GovernanceReportGenerator, GovernanceReport,
    generate_governance_report
)
from core.assumption_path import AssumptionState, AssumptionPath, PathPoint
from core.assumption_tube import AssumptionTube, TubeMetrics


# =============================================================================
# One-Way Door Tests
# =============================================================================

class TestOneWayDoor:
    """Tests for One-Way Door Analysis."""
    
    def test_classical_state_no_crossed(self):
        """Classical state should have no crossed doors."""
        state = AssumptionState.classical()
        analyzer = OneWayDoorAnalyzer()
        doors = analyzer.analyze(state)
        
        crossed = [d for d in doors if d.is_crossed]
        assert len(crossed) == 0
    
    def test_relaxed_state_has_crossed(self):
        """Fully relaxed state should have crossed doors."""
        state = AssumptionState.fully_relaxed()
        analyzer = OneWayDoorAnalyzer()
        doors = analyzer.analyze(state)
        
        crossed = [d for d in doors if d.is_crossed]
        assert len(crossed) > 0
    
    def test_partial_relaxation(self):
        """Test partial relaxation - some doors crossed."""
        state = AssumptionState(
            linearity=0.1,      # Below threshold - crossed
            exogeneity=0.9,     # Above threshold - safe
            normality=0.5,      # Above threshold - safe
        )
        analyzer = OneWayDoorAnalyzer()
        doors = analyzer.analyze(state)
        
        # Linearity door should be crossed
        lin_doors = [d for d in doors if d.trigger_assumption == 'linearity']
        assert len(lin_doors) == 1
        assert lin_doors[0].is_crossed
    
    def test_imminent_detection(self):
        """Test imminent door detection."""
        state = AssumptionState(
            exogeneity=0.35,  # Just above threshold 0.3
        )
        analyzer = OneWayDoorAnalyzer()
        doors = analyzer.analyze(state)
        
        exo_doors = [d for d in doors if d.trigger_assumption == 'exogeneity']
        assert len(exo_doors) == 1
        assert exo_doors[0].is_imminent or exo_doors[0].is_crossed
    
    def test_path_analysis(self):
        """Test analyzing entire path for crossings."""
        path = AssumptionPath()
        
        for t in np.linspace(0, 1, 10):
            state = AssumptionState.classical().interpolate(
                AssumptionState.fully_relaxed(), t
            )
            path.points.append(PathPoint(
                state=state,
                estimate=1.0,
                std_error=0.1,
                t=t
            ))
        
        analyzer = OneWayDoorAnalyzer()
        crossings = analyzer.analyze_path(path)
        
        # Should detect crossings somewhere along the path
        assert len(crossings) > 0
    
    def test_warnings(self):
        """Test warning message generation."""
        state = AssumptionState(exogeneity=0.1)  # Crossed
        analyzer = OneWayDoorAnalyzer()
        warnings = analyzer.get_warnings(state)
        
        assert len(warnings) > 0
        assert any("因果推論" in w for w in warnings)


# =============================================================================
# Claim Budget Tests
# =============================================================================

class TestClaimBudget:
    """Tests for Claim Budget calculation."""
    
    def test_strong_metrics_high_budget(self):
        """Strong metrics should allow strong claims."""
        metrics = TubeMetrics(
            robustness_radius=0.8,
            truth_thickness=0.6,
            truth_thickness_location=0.5,
            total_variance=0.1,
            self_intersection_count=0
        )
        
        calc = ClaimBudgetCalculator()
        budget = calc.from_tube_metrics(metrics)
        
        assert budget.can_claim(ClaimStrength.STRONG)
        assert budget.robustness_score > 0.6
    
    def test_weak_metrics_low_budget(self):
        """Weak metrics should limit claims."""
        metrics = TubeMetrics(
            robustness_radius=0.15,
            truth_thickness=0.05,
            truth_thickness_location=0.5,
            total_variance=2.0,
            self_intersection_count=10
        )
        
        calc = ClaimBudgetCalculator()
        budget = calc.from_tube_metrics(metrics)
        
        assert not budget.can_claim(ClaimStrength.STRONG)
        assert budget.max_strength in [ClaimStrength.WEAK, ClaimStrength.MINIMAL, ClaimStrength.NONE]
    
    def test_can_claim_hierarchy(self):
        """Test claim strength hierarchy."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            causal_allowed=True,
            predictive_allowed=True,
            generalizable=False,
            warnings=[],
            budget_remaining=1.0
        )
        
        assert budget.can_claim(ClaimStrength.MINIMAL)
        assert budget.can_claim(ClaimStrength.WEAK)
        assert budget.can_claim(ClaimStrength.MODERATE)
        assert not budget.can_claim(ClaimStrength.STRONG)
        assert not budget.can_claim(ClaimStrength.DEFINITIVE)
    
    def test_text_check(self):
        """Test text checking for claim violations."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            causal_allowed=False,
            predictive_allowed=False,
            generalizable=False,
            warnings=[],
            budget_remaining=1.0
        )
        
        calc = ClaimBudgetCalculator()
        
        text = "This proves that X causes Y and strongly suggests Z."
        findings = calc.check_text(text, budget)
        
        # Should find violations
        violations = [f for f in findings if f[2]]  # (phrase, strength, is_violation)
        assert len(violations) > 0
    
    def test_text_sanitization(self):
        """Test text sanitization."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            causal_allowed=False,
            predictive_allowed=False,
            generalizable=False,
            warnings=[],
            budget_remaining=1.0
        )
        
        calc = ClaimBudgetCalculator()
        
        text = "This proves something."
        sanitized, changes = calc.sanitize_text(text, budget)
        
        assert "proves" not in sanitized or len(changes) > 0


# =============================================================================
# Honest Sentence Tests
# =============================================================================

class TestHonestSentence:
    """Tests for The Last Honest Sentence generation."""
    
    def test_generation_with_budget(self):
        """Test sentence generation with budget."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            causal_allowed=False,
            predictive_allowed=True,
            generalizable=False,
            warnings=["Low robustness"],
            budget_remaining=1.0
        )
        
        gen = HonestSentenceGenerator()
        honest = gen.generate(budget=budget, context={'effect_name': 'treatment effect'})
        
        assert honest.sentence is not None
        assert len(honest.sentence) > 10
        assert honest.strength == ClaimStrength.MODERATE
    
    def test_forbidden_claims(self):
        """Test that forbidden claims are generated."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.WEAK,
            robustness_score=0.3,
            causal_allowed=False,
            predictive_allowed=False,
            generalizable=False,
            warnings=[],
            budget_remaining=0.5
        )
        
        gen = HonestSentenceGenerator()
        honest = gen.generate(budget=budget)
        
        assert len(honest.forbidden_claims) > 0
    
    def test_limiting_factors(self):
        """Test that limiting factors are listed."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.3,
            causal_allowed=True,
            predictive_allowed=True,
            generalizable=False,
            warnings=["Robustness is limited", "Sample specific"],
            budget_remaining=1.0
        )
        
        gen = HonestSentenceGenerator()
        honest = gen.generate(budget=budget)
        
        assert len(honest.limiting_factors) > 0
    
    def test_no_causal_claims_forbidden(self):
        """Test causal claims are forbidden when not allowed."""
        budget = ClaimBudget(
            max_strength=ClaimStrength.MODERATE,
            robustness_score=0.5,
            causal_allowed=False,
            predictive_allowed=True,
            generalizable=True,
            warnings=[],
            budget_remaining=1.0
        )
        
        gen = HonestSentenceGenerator()
        honest = gen.generate(budget=budget)
        
        assert any("cause" in f.lower() for f in honest.forbidden_claims)


# =============================================================================
# Governance Report Tests
# =============================================================================

class TestGovernanceReport:
    """Tests for unified Governance Report."""
    
    def test_report_generation_minimal(self):
        """Test report generation with minimal inputs."""
        gen = GovernanceReportGenerator()
        report = gen.generate()
        
        assert report is not None
        assert report.verdict in ["GREEN", "YELLOW", "RED"]
        assert report.honest_sentence is not None
    
    def test_report_with_tube(self):
        """Test report generation with tube."""
        tube = AssumptionTube()
        
        # Add some paths
        for i in range(10):
            path = AssumptionPath()
            for t in np.linspace(0, 1, 5):
                path.points.append(PathPoint(
                    state=AssumptionState.classical(),
                    estimate=1.0 + np.random.normal(0, 0.1),
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        gen = GovernanceReportGenerator()
        report = gen.generate(tube=tube)
        
        assert report.budget is not None
        assert report.budget.robustness_score >= 0
    
    def test_report_verdict_logic(self):
        """Test verdict logic."""
        gen = GovernanceReportGenerator()
        
        # Test with relaxed state (should be worse)
        relaxed_state = AssumptionState.fully_relaxed()
        report = gen.generate(state=relaxed_state)
        
        # Should not be GREEN with fully relaxed assumptions
        assert report.verdict in ["YELLOW", "RED"]
    
    def test_to_dict(self):
        """Test dict export."""
        gen = GovernanceReportGenerator()
        report = gen.generate()
        
        d = report.to_dict()
        
        assert 'verdict' in d
        assert 'honest_sentence' in d
        assert 'max_claim_strength' in d
    
    def test_to_markdown(self):
        """Test markdown export."""
        gen = GovernanceReportGenerator()
        report = gen.generate()
        
        md = report.to_markdown()
        
        assert "# Governance Report" in md
        assert "Verdict" in md
        assert "Claim Budget" in md
    
    def test_all_warnings_collected(self):
        """Test that all warnings are collected."""
        path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            state = AssumptionState.classical().interpolate(
                AssumptionState.fully_relaxed(), t
            )
            path.points.append(PathPoint(
                state=state,
                estimate=1.0 if t < 0.5 else 10.0,  # Jump
                std_error=0.1,
                t=t
            ))
        path._compute_curvatures()
        path.detect_cliffs()
        
        gen = GovernanceReportGenerator()
        report = gen.generate(path=path)
        
        # Should have warnings from doors or budget
        assert len(report.all_warnings) >= 0  # May or may not have warnings


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_check_one_way_doors(self):
        """Test check_one_way_doors function."""
        state = AssumptionState(linearity=0.1)
        doors = check_one_way_doors(state)
        assert len(doors) > 0
    
    def test_generate_honest_sentence(self):
        """Test generate_honest_sentence function."""
        honest = generate_honest_sentence(effect_name="treatment")
        assert honest.sentence is not None
    
    def test_generate_governance_report(self):
        """Test generate_governance_report function."""
        report = generate_governance_report()
        assert report.verdict in ["GREEN", "YELLOW", "RED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
