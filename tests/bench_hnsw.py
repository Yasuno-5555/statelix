
import time
import numpy as np
import sys
import os

# Try to import statelix_core
try:
    import statelix_core
    from statelix_core import search
except ImportError:
    print("Error: statelix_core module not found. Please build and install the package first.")
    print("If you are in dev environment, make sure build/ is in PYTHONPATH.")
    sys.exit(1)

def run_benchmark():
    print("=== Statelix HNSW Benchmark ===")
    
    # Parameters
    N = 50000       # Database size
    D = 64          # Dimension
    Q = 1000        # Number of queries
    K = 10          # Top-K
    
    print(f"Generating data: N={N}, D={D}...")
    np.random.seed(42)
    data = np.random.randn(N, D).astype(np.float64)
    queries = np.random.randn(Q, D).astype(np.float64)
    
    # Configuration
    config = search.HNSWConfig()
    config.M = 16
    config.ef_construction = 200
    config.ef_search = 100
    config.distance = search.Distance.L2
    
    # Build Index
    print(f"Building index (M={config.M}, ef_c={config.ef_construction})...")
    start_time = time.time()
    index = search.HNSW(config)
    index.build(data)
    build_time = time.time() - start_time
    print(f"Build finished in {build_time:.4f}s ({N/build_time:.1f} items/s)")
    
    # Search
    print(f"Running search (ef_s={config.ef_search}, k={K})...")
    start_time = time.time()
    results = index.query_batch(queries, K)
    total_time = time.time() - start_time
    
    qps = Q / total_time
    print(f"Search finished in {total_time:.4f}s")
    print(f"QPS: {qps:.1f}")
    
    # Validation (Recall)
    print("Verifying recall (vs Scipy Brute Force)...")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(data)
        _, ground_truth = tree.query(queries, k=K)
        
        correct = 0
        total_neighbors = Q * K
        
        for i in range(Q):
            hnsw_indices = set(results[i].indices)
            gt_indices = set(ground_truth[i])
            correct += len(hnsw_indices.intersection(gt_indices))
            
        recall = correct / total_neighbors
        print(f"Recall@{K}: {recall:.4f}")
        
    except ImportError:
        print("Scipy not found. Skipping exact recall verification.")
        print("Sanity check: First query result distances:")
        print(results[0].distances)

if __name__ == "__main__":
    run_benchmark()
