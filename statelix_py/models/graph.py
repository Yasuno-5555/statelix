import numpy as np
import pandas as pd
import scipy.sparse
from ..core import graph

class StatelixGraph:
    """
    Graph Analysis Utilities.
    Handles node ID mapping automatically.
    """
    def __init__(self):
        self.node_map_ = {} # name -> int
        self.rev_node_map_ = {} # int -> name
        self.adj_ = None # scipy.sparse.csr_matrix
        self.n_nodes_ = 0

    def fit(self, source, target, directed=False):
        """
        Build graph from edge list.
        source, target: Arrays of node IDs (strings or ints)
        """
        # 1. Create ID Map
        # Use pandas for efficient unique/factorize
        s_series = pd.Series(source)
        t_series = pd.Series(target)
        
        uniques = pd.unique(pd.concat([s_series, t_series]))
        self.n_nodes_ = len(uniques)
        
        self.node_map_ = {val: i for i, val in enumerate(uniques)}
        self.rev_node_map_ = {i: val for i, val in enumerate(uniques)}
        
        # 2. Map to Ints
        s_ids = s_series.map(self.node_map_).values.astype(np.int32)
        t_ids = t_series.map(self.node_map_).values.astype(np.int32)
        
        # 3. Build Sparse Matrix
        data = np.ones(len(s_ids), dtype=np.float64)
        self.adj_ = scipy.sparse.csr_matrix(
            (data, (s_ids, t_ids)), 
            shape=(self.n_nodes_, self.n_nodes_)
        )
        
        if not directed:
            self.adj_ = self.adj_ + self.adj_.T
            
        return self

    def louvain(self, resolution=1.0, seed=42):
        """
        Run Louvain Community Detection.
        Returns: DataFrame [Node, Community]
        """
        if self.adj_ is None: raise RuntimeError("Graph not fitted")
        
        eng = graph.Louvain()
        eng.resolution = resolution
        eng.seed = seed
        
        # Pass scipy sparse directly (pybind11 casts to Eigen::Sparse)
        res = eng.fit(self.adj_)
        
        # Map IDs back
        return pd.DataFrame({
            "Node": [self.rev_node_map_[i] for i in range(self.n_nodes_)],
            "Community": res.labels
        })

    def pagerank(self, damping=0.85):
        """
        Run PageRank.
        Returns: DataFrame [Node, Score]
        """
        if self.adj_ is None: raise RuntimeError("Graph not fitted")

        eng = graph.PageRank()
        eng.damping = damping
        
        res = eng.compute(self.adj_)
        
        return pd.DataFrame({
            "Node": [self.rev_node_map_[i] for i in range(self.n_nodes_)],
            "Score": res.scores
        }).sort_values("Score", ascending=False)
