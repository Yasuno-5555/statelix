"""
Tests for the Unified Causal Space and Feature Synthesizer.

Tests:
1. CausalSpace DAG embedding
2. Rotor invariance of topology
3. Persistence diagram computation
4. Feature synthesis
5. Integration with CausalManifold
"""

import pytest
import numpy as np
import sys
import os

# Add statelix_py to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'statelix_py')))

from core.unified_space import (
    CausalSpace, 
    RotorTransform, 
    PersistenceDiagram, 
    PersistencePair,
    TopologicalHole
)
from core.feature_synthesizer import FeatureSynthesizer, SynthesizedFeature


# =============================================================================
# CausalSpace Tests
# =============================================================================

class TestCausalSpace:
    """Tests for CausalSpace class."""
    
    def test_empty_space(self):
        """Test creating an empty CausalSpace."""
        space = CausalSpace()
        assert space.n_nodes == 0
        assert space.points.shape == (0, 3)  # Default dim is 3
    
    def test_simple_dag_embedding(self):
        """Test embedding a simple 3-node chain: A -> B -> C."""
        adj = np.array([
            [0, 1, 0],  # A -> B
            [0, 0, 1],  # B -> C  
            [0, 0, 0]   # C
        ])
        
        space = CausalSpace(adjacency=adj, node_names=['A', 'B', 'C'])
        
        assert space.n_nodes == 3
        assert space.points.shape[0] == 3
        assert space.embedding_dim >= 2  # Should auto-detect
    
    def test_distance_matrix(self):
        """Test distance matrix computation."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        dist = space.distance_matrix()
        
        assert dist.shape == (3, 3)
        assert np.allclose(np.diag(dist), 0)  # Self-distance is 0
        assert np.allclose(dist, dist.T)  # Symmetric
    
    def test_persistence_diagram(self):
        """Test persistence diagram computation."""
        adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        pd = space.topological_filter()
        
        assert isinstance(pd, PersistenceDiagram)
        assert pd.structure_score() >= 0


class TestRotorTransform:
    """Tests for rotor transformations."""
    
    def test_rotor_creation(self):
        """Test creating a rotor."""
        rotor = RotorTransform(angle=np.pi/4, plane=(0, 1), dim=3)
        assert rotor.angle == np.pi/4
        assert rotor.plane == (0, 1)
        assert rotor.dim == 3
    
    def test_rotor_apply(self):
        """Test applying rotor to points."""
        rotor = RotorTransform(angle=np.pi/2, plane=(0, 1), dim=3)
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        rotated = rotor.apply(points)
        
        # [1,0,0] rotated 90 deg in xy plane -> [0,1,0]
        assert rotated.shape == points.shape
        assert np.allclose(rotated[0], [0, 1, 0], atol=1e-6)
        # [0,0,1] should be unchanged
        assert np.allclose(rotated[2], [0, 0, 1], atol=1e-6)
    
    def test_random_rotor(self):
        """Test random rotor creation."""
        rotor = RotorTransform.random(dim=5)
        assert rotor.dim == 5
        assert 0 <= rotor.plane[0] < 5
        assert 0 <= rotor.plane[1] < 5


class TestRotorInvariance:
    """Tests for topological invariance under rotations."""
    
    def test_rotor_preserves_topology(self):
        """Test that rotor rotation preserves persistence diagram structure."""
        adj = np.array([
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        original_pd = space.topological_filter()
        original_score = original_pd.structure_score()
        
        # Apply random rotor
        rotor = RotorTransform.random(dim=space.embedding_dim)
        rotated_space = space.apply_rotor(rotor)
        rotated_pd = rotated_space.topological_filter()
        rotated_score = rotated_pd.structure_score()
        
        # Distance-based persistence should be invariant to rotation
        assert np.isclose(original_score, rotated_score, atol=1e-6)
    
    def test_verify_rotor_invariance(self):
        """Test the verify_rotor_invariance method."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        rotor = RotorTransform(angle=np.pi/3, plane=(0, 1), dim=space.embedding_dim)
        
        assert space.verify_rotor_invariance(rotor)


class TestStabilityGradient:
    """Tests for stability gradient computation."""
    
    def test_stability_gradient(self):
        """Test stability gradient computation."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        gradient = space.compute_stability_gradient()
        
        assert len(gradient) == 3
        assert all(g >= 0 for g in gradient)  # Stability should be non-negative


# =============================================================================
# FeatureSynthesizer Tests
# =============================================================================

class TestFeatureSynthesizer:
    """Tests for FeatureSynthesizer class."""
    
    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        X = np.random.randn(100, 3)
        synthesizer = FeatureSynthesizer(max_polynomial_degree=2)
        
        X_aug, features = synthesizer.synthesize(X)
        
        # Should have more columns
        assert X_aug.shape[1] >= X.shape[1]
        
        # Check for polynomial features
        poly_features = [f for f in features if f.source_type == 'polynomial']
        assert len(poly_features) > 0
    
    def test_interaction_features(self):
        """Test interaction feature generation."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * X[:, 1] + np.random.randn(100) * 0.1  # y depends on interaction
        
        synthesizer = FeatureSynthesizer(max_interactions=5)
        X_aug, features = synthesizer.synthesize(X, y=y)
        
        interaction_features = [f for f in features if f.source_type == 'interaction']
        # Should detect the x0*x1 interaction
        assert len(interaction_features) > 0
    
    def test_orthogonal_features(self):
        """Test orthogonal complement feature generation."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # y has unexplained component
        hidden = np.random.randn(100)
        y = X[:, 0] + hidden
        
        synthesizer = FeatureSynthesizer(orthogonal_threshold=0.05)
        X_aug, features = synthesizer.synthesize(X, y=y)
        
        orth_features = [f for f in features if f.source_type == 'orthogonal']
        # Should capture unexplained variance
        assert len(orth_features) > 0
    
    def test_suggest_features(self):
        """Test feature suggestion."""
        X = np.random.randn(50, 3)
        y = X[:, 0] ** 2 + X[:, 1] * X[:, 2]
        
        synthesizer = FeatureSynthesizer()
        suggestions = synthesizer.suggest_features(X, y, top_k=3)
        
        assert len(suggestions) <= 3
        for s in suggestions:
            assert 'name' in s
            assert 'type' in s
            assert 'confidence' in s


class TestFeatureSynthesizerOrthogonality:
    """Tests for orthogonality of synthesized features."""
    
    def test_polynomial_not_too_correlated(self):
        """Test that polynomial features are not too correlated with originals."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        synthesizer = FeatureSynthesizer(max_polynomial_degree=2)
        X_aug, features = synthesizer.synthesize(X)
        
        for f in features:
            if f.source_type == 'polynomial':
                # Check correlation with parent
                parent_idx = 0 if f.parent_features[0] == 'x0' else 1
                corr = np.abs(np.corrcoef(X[:, parent_idx], f.values)[0, 1])
                assert corr < 0.95  # Should not be too correlated


# =============================================================================
# Integration Tests
# =============================================================================

class TestCausalManifoldIntegration:
    """Tests for CausalManifold integration."""
    
    def test_causal_manifold_with_causal_space(self):
        """Test CausalManifold with CausalSpace integration."""
        from diagnostics.causal_manifold import CausalManifold
        
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        space = CausalSpace(adjacency=adj)
        
        # Mock model and data
        class MockModel:
            pass
        
        manifold = CausalManifold(MockModel(), {}, causal_space=space)
        
        # Should be able to compute PH-based stability
        gradient = manifold.compute_stability_as_ph_gradient()
        assert len(gradient) == 3
    
    def test_causal_manifold_backward_compat(self):
        """Test that CausalManifold works without CausalSpace (backward compat)."""
        from diagnostics.causal_manifold import CausalManifold
        
        class MockModel:
            pass
        
        # No causal_space argument - should still work
        manifold = CausalManifold(MockModel(), {})
        
        # compute_stability_as_ph_gradient should return empty array
        gradient = manifold.compute_stability_as_ph_gradient()
        assert len(gradient) == 0


# =============================================================================
# PersistenceDiagram Tests
# =============================================================================

class TestPersistenceDiagram:
    """Tests for PersistenceDiagram class."""
    
    def test_empty_diagram(self):
        """Test empty persistence diagram."""
        pd = PersistenceDiagram()
        assert pd.total_persistence() == 0
        assert pd.structure_score() == 0
    
    def test_betti_numbers(self):
        """Test Betti number computation at filtration values."""
        pairs = [
            PersistencePair(birth=0.0, death=0.5, dimension=0),
            PersistencePair(birth=0.0, death=1.0, dimension=0),
            PersistencePair(birth=0.2, death=0.8, dimension=1)
        ]
        pd = PersistenceDiagram(pairs=pairs)
        
        # At epsilon=0.3, both H0 pairs are alive, and 1 H1 pair
        betti = pd.betti_at(0.3)
        assert betti.get(0, 0) == 2
        assert betti.get(1, 0) == 1
    
    def test_significant_pairs(self):
        """Test filtering significant pairs."""
        pairs = [
            PersistencePair(birth=0.0, death=0.1, dimension=0),  # Short-lived
            PersistencePair(birth=0.0, death=1.0, dimension=0),  # Long-lived
        ]
        pd = PersistenceDiagram(pairs=pairs)
        
        significant = pd.significant_pairs(threshold=0.5)
        assert len(significant) == 1
        assert significant[0].lifetime == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
