"""
Tests for Truth Is a Tube

Tests:
1. AssumptionTube creation and path generation
2. TubeMetrics computation
3. Cross-section analysis
4. Self-intersection detection
5. CliffLearner recording and prediction
"""

import pytest
import numpy as np
import sys
import os
import tempfile

# Add statelix_py to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.assumption_tube import (
    AssumptionTube,
    TubeMetrics,
    TubeCrossSection,
    generate_robustness_tube
)
from core.cliff_learner import (
    CliffLearner,
    CliffRecord,
    get_cliff_learner
)
from core.assumption_path import (
    AssumptionPath,
    AssumptionState,
    PathPoint,
    CliffPoint
)


# =============================================================================
# TubeMetrics Tests
# =============================================================================

class TestTubeMetrics:
    """Tests for TubeMetrics class."""
    
    def test_stability_rating_robust(self):
        """Test robust stability rating."""
        metrics = TubeMetrics(
            robustness_radius=0.6,
            truth_thickness=0.4,
            truth_thickness_location=0.5,
            total_variance=0.1
        )
        assert metrics.stability_rating() == "ROBUST"
    
    def test_stability_rating_moderate(self):
        """Test moderate stability rating."""
        metrics = TubeMetrics(
            robustness_radius=0.3,
            truth_thickness=0.05,
            truth_thickness_location=0.5,
            total_variance=0.5
        )
        assert metrics.stability_rating() == "MODERATE"
    
    def test_stability_rating_fragile(self):
        """Test fragile stability rating."""
        metrics = TubeMetrics(
            robustness_radius=0.1,
            truth_thickness=0.01,
            truth_thickness_location=0.5,
            total_variance=1.0
        )
        assert metrics.stability_rating() == "FRAGILE"
    
    def test_most_brittle_assumption(self):
        """Test finding most brittle assumption."""
        metrics = TubeMetrics(
            robustness_radius=0.5,
            truth_thickness=0.3,
            truth_thickness_location=0.5,
            total_variance=0.2,
            brittleness_index={
                'linearity': 0.3,
                'normality': 0.7,
                'independence': 0.2
            }
        )
        assert metrics.most_brittle_assumption() == 'normality'


# =============================================================================
# TubeCrossSection Tests
# =============================================================================

class TestTubeCrossSection:
    """Tests for TubeCrossSection class."""
    
    def test_is_thin(self):
        """Test thin cross-section detection."""
        thin_cs = TubeCrossSection(
            t=0.5,
            estimates=np.array([1.0, 1.01, 0.99]),
            center=1.0,
            radius=0.01,
            diameter=0.02,
            std=0.01
        )
        assert thin_cs.is_thin
        
        thick_cs = TubeCrossSection(
            t=0.5,
            estimates=np.array([0.5, 1.0, 1.5]),
            center=1.0,
            radius=0.5,
            diameter=1.0,
            std=0.5
        )
        assert not thick_cs.is_thin


# =============================================================================
# AssumptionTube Tests
# =============================================================================

class TestAssumptionTube:
    """Tests for AssumptionTube class."""
    
    def test_empty_tube(self):
        """Test empty tube creation."""
        tube = AssumptionTube()
        assert len(tube.paths) == 0
    
    def test_tube_with_base_path(self):
        """Test tube with base path."""
        path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=1.0 + t * 0.1,
                std_error=0.1,
                t=t
            ))
        
        tube = AssumptionTube(path)
        assert len(tube.paths) == 1
        assert tube.base_path == path
    
    def test_manual_path_addition(self):
        """Test manually adding paths to tube."""
        tube = AssumptionTube()
        
        for i in range(5):
            path = AssumptionPath()
            for t in np.linspace(0, 1, 5):
                path.points.append(PathPoint(
                    state=AssumptionState.classical(),
                    estimate=1.0 + t * 0.1 + np.random.normal(0, 0.05),
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        assert len(tube.paths) == 5
    
    def test_cross_sections(self):
        """Test cross-section computation."""
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
        
        cross_sections = tube.compute_cross_sections()
        
        assert len(cross_sections) == 5
        for cs in cross_sections:
            assert len(cs.estimates) <= 10
            assert cs.radius >= 0
            assert cs.diameter >= 0
    
    def test_metrics_computation(self):
        """Test tube metrics computation."""
        tube = AssumptionTube()
        
        # Add paths with varying estimates
        for i in range(20):
            path = AssumptionPath()
            offset = np.random.normal(0, 0.2)
            for j, t in enumerate(np.linspace(0, 1, 10)):
                path.points.append(PathPoint(
                    state=AssumptionState.classical().interpolate(
                        AssumptionState.fully_relaxed(), t
                    ),
                    estimate=1.0 + t * 0.5 + offset,
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        metrics = tube.compute_metrics()
        
        assert metrics.robustness_radius >= 0
        assert metrics.truth_thickness >= 0
        assert 0 <= metrics.truth_thickness_location <= 1
        assert metrics.total_variance >= 0
    
    def test_envelope(self):
        """Test envelope extraction."""
        tube = AssumptionTube()
        
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
        
        t_vals, lower, upper = tube.get_envelope()
        
        assert len(t_vals) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        assert np.all(lower <= upper)
    
    def test_summary(self):
        """Test summary generation."""
        tube = AssumptionTube()
        
        for i in range(5):
            path = AssumptionPath()
            for t in np.linspace(0, 1, 5):
                path.points.append(PathPoint(
                    state=AssumptionState.classical(),
                    estimate=1.0,
                    std_error=0.1,
                    t=t
                ))
            tube.paths.append(path)
        
        summary = tube.summary()
        
        assert 'n_paths' in summary
        assert 'robustness_radius' in summary
        assert 'truth_thickness' in summary
        assert 'stability_rating' in summary
        assert summary['n_paths'] == 5


class TestTubeGeneration:
    """Tests for tube path generation."""
    
    def test_bootstrap_with_mock_model(self):
        """Test bootstrap path generation."""
        class MockModel:
            coef_ = [1.0]
            std_error_ = 0.1
            def fit(self, X, y):
                self.coef_ = [np.mean(y)]
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        data = {'X': X, 'y': y}
        
        base_path = AssumptionPath()
        base_path.trace(MockModel(), data, steps=5)
        
        tube = AssumptionTube(base_path)
        tube.generate_bootstrap_paths(MockModel(), data, n_samples=5, steps=5, random_state=42)
        
        # Should have base + bootstrap paths
        assert len(tube.paths) >= 2
    
    def test_noise_paths(self):
        """Test noise path generation."""
        class MockModel:
            coef_ = [1.0]
            std_error_ = 0.1
            def fit(self, X, y): pass
        
        np.random.seed(42)
        data = {'X': np.random.randn(50, 2), 'y': np.random.randn(50)}
        
        base_path = AssumptionPath()
        base_path.trace(MockModel(), data, steps=5)
        
        tube = AssumptionTube(base_path)
        tube.generate_noise_paths(MockModel(), data, n_samples=5, steps=5, random_state=42)
        
        assert len(tube.paths) >= 2


# =============================================================================
# CliffLearner Tests
# =============================================================================

class TestCliffLearner:
    """Tests for CliffLearner class."""
    
    def test_empty_learner(self):
        """Test empty learner."""
        learner = CliffLearner()
        assert len(learner.records) == 0
    
    def test_record_cliff(self):
        """Test recording a cliff."""
        learner = CliffLearner()
        
        cliff = CliffPoint(
            t=0.5,
            state=AssumptionState(linearity=0.3),
            curvature=10.0,
            broken_assumption='linearity',
            estimate_before=1.0,
            estimate_after=5.0
        )
        
        learner.record_cliff(cliff, context={'model': 'OLS'})
        
        assert len(learner.records) == 1
        assert learner.records[0].cliff.broken_assumption == 'linearity'
    
    def test_predict_cliff_risk(self):
        """Test cliff risk prediction."""
        learner = CliffLearner()
        
        # Record some cliffs
        for i in range(5):
            cliff = CliffPoint(
                t=0.5,
                state=AssumptionState(linearity=0.3),
                curvature=10.0,
                broken_assumption='linearity',
                estimate_before=1.0,
                estimate_after=5.0
            )
            learner.record_cliff(cliff)
        
        # Predict risk for similar state
        state = AssumptionState(linearity=0.3)
        risks = learner.predict_cliff_risk(state)
        
        assert 'linearity' in risks
        assert risks['linearity'] > 0.1  # Should have elevated risk
    
    def test_fragile_pairs(self):
        """Test fragile pair detection."""
        learner = CliffLearner()
        
        # Simulate linearity and normality breaking together
        from datetime import datetime
        timestamp = datetime.now().isoformat()[:16]
        
        for i in range(3):
            r1 = CliffRecord(
                cliff=CliffPoint(
                    t=0.4, state=AssumptionState(), curvature=5.0,
                    broken_assumption='linearity',
                    estimate_before=1.0, estimate_after=2.0
                ),
                timestamp=timestamp
            )
            r2 = CliffRecord(
                cliff=CliffPoint(
                    t=0.5, state=AssumptionState(), curvature=5.0,
                    broken_assumption='normality',
                    estimate_before=1.0, estimate_after=2.0
                ),
                timestamp=timestamp
            )
            learner.records.extend([r1, r2])
        
        pairs = learner.fragile_pairs()
        # Should detect linearity-normality pair
        assert len(pairs) >= 1 or len(learner.records) < 3
    
    def test_warning_message(self):
        """Test warning message generation."""
        learner = CliffLearner()
        
        # Record many linearity cliffs
        for i in range(20):
            cliff = CliffPoint(
                t=0.5,
                state=AssumptionState(linearity=0.5),
                curvature=10.0,
                broken_assumption='linearity',
                estimate_before=1.0,
                estimate_after=5.0
            )
            learner.record_cliff(cliff)
        
        # Request warning for risky state
        state = AssumptionState(linearity=0.5)
        warning = learner.warning_message(state)
        
        # Should get a warning
        assert warning is not None or len(learner.records) < 10
    
    def test_save_and_load(self):
        """Test persistence."""
        learner = CliffLearner()
        
        cliff = CliffPoint(
            t=0.5,
            state=AssumptionState(linearity=0.3),
            curvature=10.0,
            broken_assumption='linearity',
            estimate_before=1.0,
            estimate_after=5.0
        )
        learner.record_cliff(cliff, context={'test': True})
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            learner.save(path)
            
            # Load into new learner
            new_learner = CliffLearner()
            success = new_learner.load(path)
            
            assert success
            assert len(new_learner.records) == 1
            assert new_learner.records[0].cliff.broken_assumption == 'linearity'
        finally:
            os.unlink(path)
    
    def test_statistics(self):
        """Test statistics generation."""
        learner = CliffLearner()
        
        for i in range(5):
            cliff = CliffPoint(
                t=0.5, state=AssumptionState(), curvature=5.0,
                broken_assumption='linearity' if i < 3 else 'normality',
                estimate_before=1.0, estimate_after=2.0
            )
            learner.record_cliff(cliff)
        
        stats = learner.statistics()
        
        assert stats['total_cliffs'] == 5
        assert 'top_assumptions' in stats


class TestCliffRecord:
    """Tests for CliffRecord serialization."""
    
    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        cliff = CliffPoint(
            t=0.5,
            state=AssumptionState(linearity=0.3, normality=0.7),
            curvature=10.0,
            broken_assumption='linearity',
            estimate_before=1.0,
            estimate_after=5.0
        )
        
        record = CliffRecord(
            cliff=cliff,
            context={'model': 'OLS'},
            data_size=1000,
            model_type='StatelixOLS'
        )
        
        d = record.to_dict()
        restored = CliffRecord.from_dict(d)
        
        assert restored.cliff.broken_assumption == 'linearity'
        assert restored.cliff.t == 0.5
        assert restored.data_size == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
