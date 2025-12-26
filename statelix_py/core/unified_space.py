"""
Unified Causal Space: Integrated Mathematical Computation Space

This module unifies causal (Risan), topological (Keirin), and geometric (Shinen)
representations into a single tensor-backed computational space.

Architecture:
- DAG nodes → geometric points on manifold
- Topology → persistence barcodes on the same space
- Rotors → coordinate transforms preserving topology
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import warnings


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PersistencePair:
    """A single persistence pair (birth, death) from persistent homology."""
    birth: float
    death: float
    dimension: int = 0  # Homology dimension (0=components, 1=loops, 2=voids)
    
    @property
    def lifetime(self) -> float:
        return self.death - self.birth
    
    def __repr__(self) -> str:
        return f"H{self.dimension}({self.birth:.3f}, {self.death:.3f})"


@dataclass
class PersistenceDiagram:
    """Collection of persistence pairs with analysis methods."""
    pairs: List[PersistencePair] = field(default_factory=list)
    
    def total_persistence(self) -> float:
        """L1 norm of lifetimes."""
        return sum(p.lifetime for p in self.pairs)
    
    def structure_score(self) -> float:
        """L2 norm of lifetimes (Wasserstein-like)."""
        return np.sqrt(sum(p.lifetime ** 2 for p in self.pairs))
    
    def betti_at(self, epsilon: float) -> Dict[int, int]:
        """Betti numbers at filtration value epsilon."""
        betti = {}
        for p in self.pairs:
            if p.birth <= epsilon < p.death:
                betti[p.dimension] = betti.get(p.dimension, 0) + 1
        return betti
    
    def significant_pairs(self, threshold: float = 0.1) -> List[PersistencePair]:
        """Filter pairs with lifetime above threshold."""
        return [p for p in self.pairs if p.lifetime > threshold]


@dataclass  
class TopologicalHole:
    """Represents a detected topological hole with associated direction vector."""
    dimension: int
    birth: float
    death: float
    direction: np.ndarray  # Direction vector in feature space
    
    @property
    def persistence(self) -> float:
        return self.death - self.birth


class RotorTransform:
    """
    Geometric Algebra rotor for coordinate transformations.
    Wraps MathUniverse::Shinen concepts in a Python-friendly interface.
    """
    
    def __init__(self, angle: float, plane: Tuple[int, int], dim: int):
        """
        Create a rotor for rotation in a given plane.
        
        Args:
            angle: Rotation angle in radians
            plane: Tuple of axis indices (i, j) defining the rotation plane
            dim: Total dimension of the space
        """
        self.angle = angle
        self.plane = plane
        self.dim = dim
        self._matrix = self._build_rotation_matrix()
    
    def _build_rotation_matrix(self) -> np.ndarray:
        """Build the rotation matrix for this rotor."""
        R = np.eye(self.dim)
        i, j = self.plane
        c, s = np.cos(self.angle), np.sin(self.angle)
        R[i, i] = c
        R[i, j] = -s
        R[j, i] = s
        R[j, j] = c
        return R
    
    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the rotor to a point cloud (n, dim)."""
        return points @ self._matrix.T
    
    @classmethod
    def random(cls, dim: int, max_angle: float = np.pi/4) -> 'RotorTransform':
        """Create a random rotor for testing invariance."""
        i, j = np.random.choice(dim, 2, replace=False)
        angle = np.random.uniform(-max_angle, max_angle)
        return cls(angle, (int(i), int(j)), dim)


# =============================================================================
# CausalSpace: The Unified Representation
# =============================================================================

class CausalSpace:
    """
    Unified representation of causal structure as a geometric manifold.
    
    This class embeds a DAG (causal graph) as points in a high-dimensional space,
    enabling unified operations:
    - Topological analysis via persistent homology (Keirin)
    - Geometric transformations via rotors (Shinen)  
    - Causal structure preservation (Risan)
    
    The key insight: causal relationships become geometric distances,
    allowing topology to detect causal structure anomalies.
    """
    
    def __init__(
        self,
        adjacency: Optional[np.ndarray] = None,
        feature_matrix: Optional[np.ndarray] = None,
        node_names: Optional[List[str]] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize a CausalSpace.
        
        Args:
            adjacency: (n, n) adjacency matrix of the DAG. A[i,j]=1 means i→j.
            feature_matrix: (n, d) feature matrix for each node.
            node_names: Optional names for each node.
            embedding_dim: Embedding dimension. Auto-detected if None.
        """
        self.adjacency = adjacency
        self.feature_matrix = feature_matrix
        self.node_names = node_names or []
        
        # Compute embedding
        if adjacency is not None:
            self.n_nodes = adjacency.shape[0]
            self.embedding_dim = embedding_dim or self._auto_embedding_dim()
            self.points = self._embed_dag()
        else:
            self.n_nodes = 0
            self.embedding_dim = embedding_dim or 3
            self.points = np.array([]).reshape(0, self.embedding_dim)
        
        # Caches
        self._persistence_cache: Optional[PersistenceDiagram] = None
        self._distance_matrix_cache: Optional[np.ndarray] = None
    
    def _auto_embedding_dim(self) -> int:
        """Automatically determine embedding dimension."""
        if self.feature_matrix is not None:
            return max(3, self.feature_matrix.shape[1])
        # Rule of thumb: sqrt(n) but at least 3
        return max(3, int(np.ceil(np.sqrt(self.n_nodes))))
    
    def _embed_dag(self) -> np.ndarray:
        """
        Embed DAG nodes as points in geometric space.
        
        Uses spectral embedding of the graph Laplacian, combined with
        causal order to preserve directed structure.
        """
        if self.n_nodes == 0:
            return np.zeros((0, self.embedding_dim))
        
        # Compute graph Laplacian
        A = self.adjacency + self.adjacency.T  # Symmetrize for embedding
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Spectral embedding
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Use smallest non-zero eigenvalues (Fiedler vectors)
            idx = np.argsort(eigenvalues)[1:self.embedding_dim + 1]
            base_embedding = eigenvectors[:, idx]
            
            # Pad if we don't have enough eigenvectors
            if base_embedding.shape[1] < self.embedding_dim:
                pad = np.zeros((self.n_nodes, self.embedding_dim - base_embedding.shape[1]))
                base_embedding = np.hstack([base_embedding, pad])
        except np.linalg.LinAlgError:
            # Fallback: random embedding
            base_embedding = np.random.randn(self.n_nodes, self.embedding_dim)
        
        # Add causal order information
        causal_order = self._compute_causal_order()
        base_embedding[:, 0] += causal_order * 0.1  # Shift by causal depth
        
        # Incorporate feature matrix if available
        if self.feature_matrix is not None:
            # Project features onto embedding dimensions
            n_feat = min(self.feature_matrix.shape[1], self.embedding_dim)
            normalized_features = self.feature_matrix[:, :n_feat]
            if normalized_features.std() > 0:
                normalized_features = (normalized_features - normalized_features.mean(axis=0)) / (normalized_features.std(axis=0) + 1e-8)
            base_embedding[:, :n_feat] += normalized_features * 0.5
        
        return base_embedding
    
    def _compute_causal_order(self) -> np.ndarray:
        """Compute topological sort depth for each node."""
        order = np.zeros(self.n_nodes)
        in_degree = self.adjacency.sum(axis=0)
        
        # BFS-style topological ordering
        queue = list(np.where(in_degree == 0)[0])
        depth = 0
        while queue:
            next_queue = []
            for node in queue:
                order[node] = depth
                for child in np.where(self.adjacency[node] > 0)[0]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue
            depth += 1
        
        return order
    
    # -------------------------------------------------------------------------
    # Topological Analysis (Keirin)
    # -------------------------------------------------------------------------
    
    def distance_matrix(self) -> np.ndarray:
        """Compute pairwise distances between embedded points."""
        if self._distance_matrix_cache is not None:
            return self._distance_matrix_cache
        
        if self.n_nodes == 0:
            return np.array([]).reshape(0, 0)
        
        diff = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        self._distance_matrix_cache = np.sqrt((diff ** 2).sum(axis=2))
        return self._distance_matrix_cache
    
    def topological_filter(self, max_radius: Optional[float] = None) -> PersistenceDiagram:
        """
        Compute persistence diagram using Vietoris-Rips filtration.
        
        This is a simplified implementation for H0 (connected components).
        For full persistent homology, use external libraries.
        """
        if self._persistence_cache is not None:
            return self._persistence_cache
        
        dist = self.distance_matrix()
        if dist.size == 0:
            return PersistenceDiagram()
        
        max_radius = max_radius or dist.max() * 1.1
        
        # Union-Find for H0 persistence (connected components)
        pairs = self._compute_h0_persistence(dist, max_radius)
        
        self._persistence_cache = PersistenceDiagram(pairs=pairs)
        return self._persistence_cache
    
    def _compute_h0_persistence(self, dist: np.ndarray, max_radius: float) -> List[PersistencePair]:
        """Compute H0 persistence via Union-Find."""
        n = dist.shape[0]
        parent = list(range(n))
        birth = [0.0] * n  # All components born at 0
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        pairs = []
        
        # Sort edges by distance
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= max_radius:
                    edges.append((dist[i, j], i, j))
        edges.sort()
        
        for d, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                # Merge: younger component dies
                if birth[ri] < birth[rj]:
                    ri, rj = rj, ri
                parent[rj] = ri
                pairs.append(PersistencePair(birth=birth[rj], death=d, dimension=0))
        
        # Remaining components persist to infinity (represented as max_radius)
        roots = set(find(i) for i in range(n))
        for r in roots:
            if len(roots) > 1:  # Don't add if only one component
                pairs.append(PersistencePair(birth=birth[r], death=max_radius, dimension=0))
                break  # Only one "infinite" component
        
        return pairs
    
    # -------------------------------------------------------------------------
    # Geometric Transformations (Shinen)
    # -------------------------------------------------------------------------
    
    def apply_rotor(self, rotor: RotorTransform) -> 'CausalSpace':
        """
        Apply a rotor transformation to the embedded points.
        
        This creates a new CausalSpace with rotated coordinates.
        Crucially, the topology should be preserved.
        """
        new_space = CausalSpace(
            adjacency=self.adjacency.copy() if self.adjacency is not None else None,
            feature_matrix=self.feature_matrix.copy() if self.feature_matrix is not None else None,
            node_names=self.node_names.copy(),
            embedding_dim=self.embedding_dim
        )
        new_space.points = rotor.apply(self.points)
        return new_space
    
    def verify_rotor_invariance(self, rotor: RotorTransform, tolerance: float = 1e-6) -> bool:
        """
        Verify that applying a rotor preserves topological structure.
        
        Returns True if the persistence diagram is unchanged (up to tolerance).
        """
        original_pd = self.topological_filter()
        rotated_space = self.apply_rotor(rotor)
        rotated_pd = rotated_space.topological_filter()
        
        # Compare persistence diagrams
        return abs(original_pd.structure_score() - rotated_pd.structure_score()) < tolerance
    
    # -------------------------------------------------------------------------
    # Stability Analysis
    # -------------------------------------------------------------------------
    
    def compute_stability_gradient(self, epsilon: float = 0.01) -> np.ndarray:
        """
        Compute the stability gradient as the derivative of persistent homology.
        
        For each point, measures how sensitive the topology is to moving that point.
        
        Returns:
            (n_nodes,) array of stability scores (lower = more stable)
        """
        if self.n_nodes == 0:
            return np.array([])
        
        base_score = self.topological_filter().structure_score()
        gradients = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            # Perturb point i in all directions
            max_change = 0.0
            for d in range(self.embedding_dim):
                perturbed = self.points.copy()
                perturbed[i, d] += epsilon
                
                # Create temporary space for perturbed computation
                temp_space = CausalSpace(
                    adjacency=self.adjacency,
                    embedding_dim=self.embedding_dim
                )
                temp_space.points = perturbed
                temp_space.n_nodes = self.n_nodes
                
                new_score = temp_space.topological_filter().structure_score()
                change = abs(new_score - base_score) / epsilon
                max_change = max(max_change, change)
            
            gradients[i] = max_change
        
        return gradients
    
    def detect_topological_holes(self, threshold: float = 0.1) -> List[TopologicalHole]:
        """
        Detect significant topological holes and compute their direction vectors.
        
        The direction vector points toward where adding a feature could "fill" the hole.
        """
        pd = self.topological_filter()
        holes = []
        
        for pair in pd.significant_pairs(threshold):
            # Find the points involved in this persistence pair
            # (Simplified: use the gradient as the direction)
            direction = np.zeros(self.embedding_dim)
            direction[0] = pair.lifetime  # Placeholder
            
            holes.append(TopologicalHole(
                dimension=pair.dimension,
                birth=pair.birth,
                death=pair.death,
                direction=direction
            ))
        
        return holes
    
    # -------------------------------------------------------------------------
    # Causal Structure (Risan)
    # -------------------------------------------------------------------------
    
    def suggest_rewiring(self) -> List[Dict[str, Any]]:
        """
        Suggest causal graph rewiring based on geometric proximity.
        
        If two causally unconnected nodes are geometrically close,
        there may be a missing edge or confounding.
        """
        if self.adjacency is None or self.n_nodes < 2:
            return []
        
        dist = self.distance_matrix()
        suggestions = []
        
        median_dist = np.median(dist[dist > 0])
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # Check if geometrically close but causally disconnected
                if self.adjacency[i, j] == 0 and self.adjacency[j, i] == 0:
                    if dist[i, j] < median_dist * 0.5:
                        suggestions.append({
                            'action': 'ADD_EDGE',
                            'source': self.node_names[i] if self.node_names else str(i),
                            'target': self.node_names[j] if self.node_names else str(j),
                            'confidence': 1.0 - (dist[i, j] / median_dist)
                        })
        
        return suggestions
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    def invalidate_cache(self):
        """Invalidate all caches after modification."""
        self._persistence_cache = None
        self._distance_matrix_cache = None
    
    def __repr__(self) -> str:
        return f"CausalSpace(n_nodes={self.n_nodes}, dim={self.embedding_dim})"


# =============================================================================
# Integration with Existing CausalManifold
# =============================================================================

def enhance_causal_manifold(manifold_class):
    """
    Decorator to enhance existing CausalManifold with unified space capabilities.
    """
    original_init = manifold_class.__init__
    
    def new_init(self, model, data, causal_space: Optional[CausalSpace] = None):
        original_init(self, model, data)
        self.causal_space = causal_space
    
    def compute_stability_as_ph_gradient(self) -> np.ndarray:
        """Compute stability using persistent homology gradient."""
        if self.causal_space is not None:
            return self.causal_space.compute_stability_gradient()
        # Legacy fallback
        return np.array([p.std_error for p in self.points]) if self.points else np.array([])
    
    manifold_class.__init__ = new_init
    manifold_class.compute_stability_as_ph_gradient = compute_stability_as_ph_gradient
    return manifold_class
