import os
import sys
import time
import numpy as np

# Ensure statelix is in path
sys.path.append(os.path.abspath("."))
from statelix import mathuniverse

def benchmark_sokudo():
    print("=== Sokudo: Parallel Sampling Benchmark ===")
    n_samples = 10_000_000
    print(f"Generating {n_samples:,} standard normal samples...")
    
    # Sokudo Parallel (Zigen Backend)
    start = time.time()
    s_samples = mathuniverse.sokudo.generate_normal(n_samples, seed=42)
    sokudo_time = time.time() - start
    print(f"Sokudo (Parallel OpenMP) Time: {sokudo_time:.4f}s")
    
    # NumPy (Reference)
    start = time.time()
    np_samples = np.random.randn(n_samples)
    numpy_time = time.time() - start
    print(f"NumPy Time: {numpy_time:.4f}s")
    
    speedup = numpy_time / sokudo_time
    print(f"Speedup vs NumPy: {speedup:.2f}x")
    
    # Statistical check
    print(f"Sokudo Mean: {np.mean(s_samples):.6f} (Expected: 0)")
    print(f"Sokudo Std:  {np.std(s_samples):.6f} (Expected: 1)")

def test_ryoshi_quantum():
    print("\n=== Ryoshi: Quantum-Inspired Optimization Check ===")
    print("Initializing 2-qubit state...")
    qs = mathuniverse.ryoshi.QuantumState(2)
    
    # Create Bell State |phi+> = (|00> + |11>) / sqrt(2)
    print("Applying H(0) and CNOT(0, 1)...")
    qs.H(0)
    qs.CNOT(0, 1)
    
    print("State Vector (Measured):")
    qs.print_state()
    
    outcomes = []
    for _ in range(10):
        outcomes.append(qs.measure())
    
    print(f"Measurement Outcomes (10 trials): {outcomes}")
    # Outcomes should be 0 (bin 00) or 3 (bin 11) for a Bell state
    valid = all(o in [0, 3] for o in outcomes)
    print(f"Entanglement Verification: {'PASSED' if valid else 'FAILED'}")

if __name__ == "__main__":
    benchmark_sokudo()
    test_ryoshi_quantum()
