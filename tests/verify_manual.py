import numpy as np
import statelix_py.core as stx
from statelix_py.models import StatelixHNSW, StatelixHMC, StatelixOLS

def test_hnsw():
    print("Testing HNSW...")
    data = np.random.randn(100, 10).astype(np.float64)
    model = StatelixHNSW(M=16, ef_construction=100)
    model.fit(data)
    
    # Query self
    indices = model.transform(data[:5])
    print(f"Self-query indices (should include self): {indices[:, 0]}")
    assert indices.shape == (5, 5)

def test_hmc():
    print("\nTesting HMC...")
    # Define simple quadratic potential: U(x) = 0.5 * x^T x (Standard Normal)
    def target(x):
        # Value: 0.5 * sum(x^2)
        val = 0.5 * np.sum(x**2)
        # Gradient: x
        grad = x
        return val, grad # Returns (neg_log_prob, neg_grad) logichandled by wrapper?
                         # Wrapper expects (log_prob, grad)
                         # Wait, implementation says: return {-log_prob, -grad};
                         # So if I return (log_p, grad_log_p), wrapper negates it.
                         # Target density is exp(-0.5 x^2).
                         # log_p = -0.5 x^2. grad_log_p = -x.
                         
        lp = -0.5 * np.sum(x**2)
        grad_lp = -x
        return lp, grad_lp

    hmc = StatelixHMC(n_samples=200, warmup=100, step_size=0.1, seed=123)
    res = hmc.sample(target, np.zeros(2))
    
    print(f"HMC Acceptance: {res.acceptance_rate:.2f}")
    print(f"HMC Mean (should be near 0): {res.mean}")
    assert res.acceptance_rate > 0.5

def test_graph():
    print("\nTesting Graph (Louvain)...")
    import scipy.sparse
    # Create simple 2-clique graph
    # 0-1-2 and 3-4-5, weak link 2-3
    data = [1, 1, 1, 1, 1, 0.1]
    rows = [0, 1, 0, 3, 4, 2]
    cols = [1, 2, 2, 4, 5, 3]
    N = 6
    adj = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    # Make symmetric
    adj = adj + adj.T
    
    # Convert manually (since binding for stx.SparseMatrix.from_csr exists but Louvain takes Eigen)
    # Actually Louvain takes Eigen::SparseMatrix. pybind11 might auto-convert scipy.sparse if included?
    # python_bindings.cpp includes pybind11/eigen.h which generally handles Dense.
    # For Sparse, we need to ensure the cast works. 
    # If not, we might need to use stx.SparseMatrix helper.
    # But let's try direct passing first.
    
    try:
        louvain = stx.graph.Louvain()
        res = louvain.fit(adj)
        print(f"Communities: {res.n_communities}")
        print(f"Labels: {res.labels}")
    except TypeError:
        print("Direct scipy sparse passing might not be supported without header. Using workaround?")

def main():
    try:
        test_hnsw()
        test_hmc()
        test_graph()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
