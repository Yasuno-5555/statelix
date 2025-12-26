"""
Tests for Truth Collapse Simulator
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.collapse_simulator import (
    TruthCollapseSimulator,
    CollapseStage,
    CollapseSchedule,
    CollapseArchetype,
    CollapseReport,
    simulate_collapse
)
from core.assumption_path import AssumptionPath, AssumptionState, PathPoint
from core.assumption_tube import AssumptionTube, TubeMetrics
from core.claim_budget import ClaimStrength


# =============================================================================
# CollapseArchetype Tests
# =============================================================================

class TestCollapseArchetype:
    """Tests for CollapseArchetype enum."""
    
    def test_icons(self):
        """Test archetype icons."""
        assert CollapseArchetype.GLASS_TOWER.icon == "🗼"
        assert CollapseArchetype.MARSHMALLOW.icon == "☁️"
        assert CollapseArchetype.STEEL_ROD.icon == "🔩"
        assert CollapseArchetype.PAPER_TIGER.icon == "🐯"
    
    def test_descriptions(self):
        """Test archetype descriptions."""
        assert "即死" in CollapseArchetype.GLASS_TOWER.description
        assert "曖昧" in CollapseArchetype.MARSHMALLOW.description


# =============================================================================
# CollapseStage Tests
# =============================================================================

class TestCollapseStage:
    """Tests for CollapseStage."""
    
    def test_is_alive(self):
        """Test is_alive property."""
        alive_stage = CollapseStage(
            stage_index=0,
            destroyed_assumption='normality',
            state_before=AssumptionState.classical(),
            state_after=AssumptionState(normality=0),
            robustness_radius=0.5,
            truth_thickness=0.3,
            claim_strength=ClaimStrength.MODERATE,
            honest_sentence="Test"
        )
        assert alive_stage.is_alive
        
        dead_stage = CollapseStage(
            stage_index=1,
            destroyed_assumption='exogeneity',
            state_before=AssumptionState.classical(),
            state_after=AssumptionState.fully_relaxed(),
            robustness_radius=0.05,
            truth_thickness=0.01,
            claim_strength=ClaimStrength.NONE,
            honest_sentence="Cannot claim",
            is_terminal=True
        )
        assert not dead_stage.is_alive


# =============================================================================
# TruthCollapseSimulator Tests
# =============================================================================

class TestTruthCollapseSimulator:
    """Tests for TruthCollapseSimulator."""
    
    def test_simulate_minimal(self):
        """Test simulation with minimal inputs."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        assert report is not None
        assert report.schedule is not None
        assert len(report.schedule.stages) > 0
    
    def test_simulate_with_tube(self):
        """Test simulation with tube."""
        tube = AssumptionTube()
        
        # Add some paths
        for i in range(5):
            path = AssumptionPath()
            for t in np.linspace(0, 1, 5):
                path.points.append(PathPoint(
                    state=AssumptionState.classical(),
                    estimate=1.0 + np.random.normal(0, 0.1),
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        sim = TruthCollapseSimulator()
        report = sim.simulate(tube=tube)
        
        assert report is not None
        assert report.initial_metrics is not None
    
    def test_destruction_order(self):
        """Test custom destruction order."""
        custom_order = ['linearity', 'exogeneity', 'normality']
        sim = TruthCollapseSimulator(destruction_order=custom_order)
        report = sim.simulate()
        
        # First destroyed should be linearity
        if report.schedule.stages:
            assert report.schedule.stages[0].destroyed_assumption == 'linearity'
    
    def test_death_detection(self):
        """Test that death stage is detected."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        # Death stage might or might not exist depending on initial state
        schedule = report.schedule
        assert schedule.death_stage is None or schedule.death_stage >= 0
    
    def test_archetype_classification(self):
        """Test archetype classification."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        assert report.schedule.archetype in list(CollapseArchetype)
    
    def test_irreversibility_computed(self):
        """Test irreversibility is computed."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        assert 0 <= report.schedule.irreversibility_index <= 1


# =============================================================================
# CollapseReport Tests
# =============================================================================

class TestCollapseReport:
    """Tests for CollapseReport."""
    
    def test_to_dict(self):
        """Test dict export."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        d = report.to_dict()
        
        assert 'archetype' in d
        assert 'death_stage' in d
        assert 'trajectory' in d
        assert 'initial_sentence' in d
    
    def test_to_markdown(self):
        """Test markdown export."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        md = report.to_markdown()
        
        assert "# Truth Collapse Report" in md
        assert "Archetype" in md
        assert "Timeline" in md
    
    def test_strength_trajectory(self):
        """Test trajectory extraction."""
        sim = TruthCollapseSimulator()
        report = sim.simulate()
        
        trajectory = report.schedule.strength_trajectory()
        assert isinstance(trajectory, list)
        assert len(trajectory) == len(report.schedule.stages)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for convenience function."""
    
    def test_simulate_collapse(self):
        """Test simulate_collapse function."""
        report = simulate_collapse()
        
        assert report is not None
        assert report.schedule.archetype is not None
    
    def test_simulate_collapse_custom_order(self):
        """Test simulate_collapse with custom order."""
        report = simulate_collapse(destruction_order=['exogeneity', 'linearity'])
        
        assert len(report.schedule.stages) >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with other modules."""
    
    def test_full_pipeline(self):
        """Test full collapse pipeline."""
        # Create tube with varying robustness
        tube = AssumptionTube()
        
        for i in range(10):
            path = AssumptionPath()
            for t in np.linspace(0, 1, 10):
                state = AssumptionState.classical().interpolate(
                    AssumptionState.fully_relaxed(), t
                )
                path.points.append(PathPoint(
                    state=state,
                    estimate=1.0 + np.random.normal(0, 0.2),
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        # Simulate collapse
        report = simulate_collapse(tube=tube)
        
        # Verify rich output
        assert report.initial_sentence is not None
        assert report.final_sentence is not None
        assert len(report.schedule.stages) > 0
        
        # Check markdown generation doesn't error
        md = report.to_markdown()
        assert len(md) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
