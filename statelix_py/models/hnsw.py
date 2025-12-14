import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..core import search

class StatelixHNSW(BaseEstimator, TransformerMixin):
    """
    Approximate Nearest Neighbor Search using HNSW.
    Compatible with scikit-learn API.
    """
    def __init__(self, M=16, ef_construction=200, ef_search=50, 
                 distance='L2', n_neighbors=5, n_jobs=1, seed=42):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs # Placeholder for future parallelization
        self.seed = seed
        self._index = None
    
    def fit(self, X, y=None):
        """
        Build the HNSW index.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must be convertible to float64 (or float32 if optimized).
        """
        # Ensure contiguous array
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        # Configure
        config = search.HNSWConfig()
        config.M = self.M
        config.ef_construction = self.ef_construction
        config.ef_search = self.ef_search
        config.seed = self.seed
        
        if self.distance.upper() == 'L2':
            config.distance = search.Distance.L2
        elif self.distance.upper() == 'COSINE':
            config.distance = search.Distance.COSINE
        elif self.distance.upper() == 'IP':
            config.distance = search.Distance.INNER_PRODUCT
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
            
        self._index = search.HNSW(config)
        self._index.build(X)
        return self
        
    def transform(self, X):
        """
        Find k-nearest neighbors for query points X.
        
        Returns
        -------
        indices : array of shape (n_samples, n_neighbors)
        """
        if self._index is None:
            raise RuntimeError("Index not fitted")
            
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        results = self._index.query_batch(X, self.n_neighbors)
        
        # Extract indices
        indices = np.array([r.indices for r in results], dtype=np.int32)
        return indices
        
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """
        Finds the K-neighbors of a point.
        """
        if self._index is None:
            raise RuntimeError("Index not fitted")
            
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        results = self._index.query_batch(X, k)
        
        indices = np.array([r.indices for r in results], dtype=np.int32)
        if return_distance:
            dists = np.array([r.distances for r in results], dtype=np.float64)
            return dists, indices
        return indices
