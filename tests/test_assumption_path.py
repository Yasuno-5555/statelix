"""
Tests for Assumption Path - Model Is a Path

Tests:
1. AssumptionState creation and operations
2. Path interpolation
3. Curvature computation
4. Cliff detection
5. CausalSpace integration
"""

import pytest
import numpy as np
import sys
import os

# Add statelix_py to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.assumption_path import (
    AssumptionState,
    AssumptionPath,
    PathPoint,
    CliffPoint,
    trace_linearity,
    trace_normality,
    trace_full_relaxation,
)


# =============================================================================
# AssumptionState Tests
# =============================================================================

class TestAssumptionState:
    """Tests for AssumptionState class."""
    
    def test_default_state(self):
        """Test default state creation."""
        state = AssumptionState()
        assert state.linearity == 1.0
        assert state.independence == 1.0
        assert state.normality == 1.0
    
    def test_classical_state(self):
        """Test classical (all enforced) state."""
        state = AssumptionState.classical()
        vec = state.to_vector()
        assert np.allclose(vec, np.ones(6))
    
    def test_fully_relaxed_state(self):
        """Test fully relaxed state."""
        state = AssumptionState.fully_relaxed()
        vec = state.to_vector()
        assert np.allclose(vec, np.zeros(6))
    
    def test_to_vector(self):
        """Test vector conversion."""
        state = AssumptionState(linearity=0.5, independence=0.3)
        vec = state.to_vector()
        assert vec[0] == 0.5
        assert vec[1] == 0.3
        assert len(vec) == 6
    
    def test_from_vector(self):
        """Test creating from vector."""
        vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        state = AssumptionState.from_vector(vec)
        assert state.linearity == pytest.approx(0.1)
        assert state.normality == pytest.approx(0.4)
    
    def test_value_clamping(self):
        """Test that values are clamped to [0, 1]."""
        state = AssumptionState(linearity=1.5, independence=-0.2)
        assert state.linearity == 1.0
        assert state.independence == 0.0
    
    def test_distance(self):
        """Test distance computation."""
        s1 = AssumptionState.classical()
        s2 = AssumptionState.fully_relaxed()
        dist = s1.distance(s2)
        assert dist == pytest.approx(np.sqrt(6))  # sqrt(1^2 * 6)
    
    def test_interpolate(self):
        """Test interpolation between states."""
        start = AssumptionState.classical()
        end = AssumptionState.fully_relaxed()
        
        mid = start.interpolate(end, 0.5)
        assert mid.linearity == pytest.approx(0.5)
        assert mid.normality == pytest.approx(0.5)
        
        # At t=0, should be start
        at_start = start.interpolate(end, 0.0)
        assert at_start.linearity == pytest.approx(1.0)
        
        # At t=1, should be end
        at_end = start.interpolate(end, 1.0)
        assert at_end.linearity == pytest.approx(0.0)
    
    def test_relaxation_order(self):
        """Test getting the order of relaxed assumptions."""
        state = AssumptionState(
            linearity=0.2,    # Most relaxed
            independence=0.8,
            stationarity=0.5,
            normality=1.0,    # Least relaxed
        )
        order = state.relaxation_order()
        assert order[0] == 'linearity'  # Most relaxed first


# =============================================================================
# PathPoint Tests
# =============================================================================

class TestPathPoint:
    """Tests for PathPoint class."""
    
    def test_creation(self):
        """Test path point creation."""
        state = AssumptionState.classical()
        point = PathPoint(
            state=state,
            estimate=1.5,
            std_error=0.2,
            t=0.0
        )
        assert point.estimate == 1.5
        assert point.t == 0.0
        assert point.is_stable  # Default
    
    def test_repr(self):
        """Test string representation."""
        point = PathPoint(
            state=AssumptionState.classical(),
            estimate=1.234,
            std_error=0.05,
            t=0.5
        )
        repr_str = repr(point)
        assert "0.50" in repr_str
        assert "1.2340" in repr_str


# =============================================================================
# CliffPoint Tests
# =============================================================================

class TestCliffPoint:
    """Tests for CliffPoint class."""
    
    def test_severity(self):
        """Test cliff severity computation."""
        cliff = CliffPoint(
            t=0.3,
            state=AssumptionState(),
            curvature=10.0,
            broken_assumption='linearity',
            estimate_before=1.0,
            estimate_after=3.0
        )
        # Jump of 2.0 on estimate of 1.0 = 200% change
        assert cliff.severity > 0.5
    
    def test_mild_cliff(self):
        """Test mild cliff detection."""
        cliff = CliffPoint(
            t=0.3,
            state=AssumptionState(),
            curvature=1.0,
            broken_assumption='normality',
            estimate_before=10.0,
            estimate_after=10.1
        )
        assert cliff.severity < 0.1


# =============================================================================
# AssumptionPath Tests
# =============================================================================

class TestAssumptionPath:
    """Tests for AssumptionPath class."""
    
    def test_empty_path(self):
        """Test empty path creation."""
        path = AssumptionPath()
        assert len(path.points) == 0
        assert path.stability_score() == 1.0
    
    def test_path_with_mock_model(self):
        """Test path tracing with a mock model."""
        
        class MockModel:
            def __init__(self):
                self.coef_ = [0.0]
                self.std_error_ = 0.1
            
            def fit(self, X, y):
                # Simulate estimate changing with "linearity" assumption
                self.coef_ = [1.0]
        
        path = AssumptionPath()
        data = {'X': np.random.randn(100, 3), 'y': np.random.randn(100)}
        
        path.trace(MockModel(), data, steps=5)
        
        assert len(path.points) == 5
        assert path.points[0].t == 0.0
        assert path.points[-1].t == 1.0
    
    def test_path_with_model_factory(self):
        """Test path tracing with custom model factory."""
        estimates_by_linearity = []
        
        def model_factory(state: AssumptionState):
            class ConfiguredModel:
                def __init__(self, lin):
                    self.lin = lin
                    self.coef_ = [0.0]
                    self.std_error_ = 0.1
                
                def fit(self, X, y):
                    # Estimate depends on linearity assumption
                    self.coef_ = [1.0 + (1 - self.lin) * 5.0]  # Nonlinear finds larger effect
                    estimates_by_linearity.append(self.coef_[0])
            
            return ConfiguredModel(state.linearity)
        
        path = AssumptionPath()
        data = {'X': np.random.randn(100, 3), 'y': np.random.randn(100)}
        
        path.trace(None, data, steps=5, model_factory=model_factory)
        
        assert len(path.points) == 5
        # Estimate should increase as linearity relaxes
        assert path.points[-1].estimate > path.points[0].estimate
    
    def test_curvature_computation(self):
        """Test curvature computation along path."""
        path = AssumptionPath()
        
        # Manually add points with known pattern
        for i, t in enumerate(np.linspace(0, 1, 10)):
            state = AssumptionState.classical().interpolate(
                AssumptionState.fully_relaxed(), t
            )
            # Quadratic pattern -> constant curvature
            estimate = t ** 2
            path.points.append(PathPoint(
                state=state,
                estimate=estimate,
                std_error=0.1,
                t=t
            ))
        
        path._compute_curvatures()
        curvatures = path.compute_curvature()
        
        # Middle points should have non-zero curvature
        assert curvatures[4] > 0
    
    def test_cliff_detection(self):
        """Test cliff detection with synthetic data."""
        path = AssumptionPath()
        
        # Create a path with a clear cliff at t=0.5
        # Use 21 points for finer resolution
        for i, t in enumerate(np.linspace(0, 1, 21)):
            state = AssumptionState.classical().interpolate(
                AssumptionState.fully_relaxed(), t
            )
            # Sharp step function cliff at t=0.5
            if t < 0.48:
                estimate = 1.0
            elif t > 0.52:
                estimate = 10.0  # Large jump
            else:
                # Transition zone - linear interpolation
                estimate = 1.0 + (t - 0.48) / 0.04 * 9.0
            
            path.points.append(PathPoint(
                state=state,
                estimate=estimate,
                std_error=0.1,
                t=t
            ))
        
        path._compute_curvatures()
        curvatures = path.compute_curvature()
        
        # Verify curvatures were computed
        assert curvatures.max() > 0, "Curvatures should be non-zero"
        
        cliffs = path.detect_cliffs(threshold=0.1)  # Lower threshold
        
        # Should detect at least one cliff around t=0.5
        assert len(cliffs) >= 1, f"Expected cliffs, got none. Max curvature: {curvatures.max()}"
        assert any(0.4 < c.t < 0.6 for c in cliffs), f"Cliff not at expected location. Cliffs: {cliffs}"
    
    def test_path_length(self):
        """Test path length computation."""
        path = AssumptionPath()
        
        # Path with constant estimate -> length 0
        for t in np.linspace(0, 1, 5):
            path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=1.0,
                std_error=0.1,
                t=t
            ))
        
        assert path.path_length() == pytest.approx(0.0)
        
        # Path with increasing estimate
        path.points = []
        for i, t in enumerate(np.linspace(0, 1, 5)):
            path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=float(i),
                std_error=0.1,
                t=t
            ))
        
        assert path.path_length() == pytest.approx(4.0)  # 0->1->2->3->4
    
    def test_stability_score(self):
        """Test stability score computation."""
        # Stable path
        stable_path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            stable_path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=1.0,
                std_error=0.01,
                t=t,
                is_stable=True
            ))
        
        assert stable_path.stability_score() > 0.8
        
        # Unstable path with cliffs
        unstable_path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            unstable_path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=np.random.uniform(0, 10),
                std_error=5.0,
                t=t,
                is_stable=False
            ))
        unstable_path.cliffs = [CliffPoint(
            t=0.5, state=AssumptionState(), curvature=10.0,
            broken_assumption='test', estimate_before=0, estimate_after=10
        )]
        
        assert unstable_path.stability_score() < 0.5
    
    def test_summary(self):
        """Test summary generation."""
        path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            path.points.append(PathPoint(
                state=AssumptionState.classical(),
                estimate=1.0 + t,
                std_error=0.1,
                t=t
            ))
        
        summary = path.summary()
        assert 'n_points' in summary
        assert 'stability_score' in summary
        assert summary['n_points'] == 5


# =============================================================================
# CausalSpace Integration Tests
# =============================================================================

class TestCausalSpaceIntegration:
    """Tests for AssumptionPath -> CausalSpace conversion."""
    
    def test_to_causal_space(self):
        """Test converting path to CausalSpace."""
        # Import CausalSpace for verification
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))
        from core.unified_space import CausalSpace
        
        path = AssumptionPath()
        for t in np.linspace(0, 1, 5):
            state = AssumptionState.classical().interpolate(
                AssumptionState.fully_relaxed(), t
            )
            path.points.append(PathPoint(
                state=state,
                estimate=1.0 + t,
                std_error=0.1,
                t=t
            ))
        
        space = path.to_causal_space()
        
        assert space.n_nodes == 5
        assert space.adjacency.sum() == 4  # 4 edges in a path of 5 nodes
        assert space.feature_matrix.shape == (5, 8)  # 6 dims + estimate + std_error
    
    def test_empty_path_to_causal_space(self):
        """Test empty path conversion."""
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))
        
        path = AssumptionPath()
        space = path.to_causal_space()
        assert space.n_nodes == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience tracing functions."""
    
    def test_trace_linearity(self):
        """Test tracing linearity dimension only."""
        class MockModel:
            coef_ = [1.0]
            std_error_ = 0.1
            def fit(self, X, y): pass
        
        data = {'X': np.random.randn(50, 2), 'y': np.random.randn(50)}
        path = trace_linearity(MockModel(), data, steps=5)
        
        assert len(path.points) == 5
        # Only linearity should change
        assert path.start.independence == 1.0
        assert path.end.independence == 1.0
        assert path.end.linearity == 0.0
    
    def test_trace_normality(self):
        """Test tracing normality dimension only."""
        class MockModel:
            coef_ = [1.0]
            std_error_ = 0.1
            def fit(self, X, y): pass
        
        data = {'X': np.random.randn(50, 2), 'y': np.random.randn(50)}
        path = trace_normality(MockModel(), data, steps=5)
        
        assert path.end.normality == 0.0
        assert path.end.linearity == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
