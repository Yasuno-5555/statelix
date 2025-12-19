import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from statelix_py.models.linear import StatelixOLS

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
    # Pre-compile JIT function
    # Use float64 for fair comparison with C++ double
    jax.config.update("jax_enable_x64", True)
except ImportError:
    HAS_JAX = False
    print("WARNING: JAX not found. Install via 'pip install jax jaxlib'")

def generate_data(n_samples, n_features):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    intercept = 2.5
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_coef + intercept + noise
    return X, y

if HAS_JAX:
    @jax.jit
    def jax_ols_fit(X, y):
        # OLS: (X'X)^-1 X'y
        # Add intercept column
        intercept = jnp.ones((X.shape[0], 1))
        X_aug = jnp.hstack([intercept, X])
        
        # Solve Normal Equations
        # This matches the Cholesky/LDLT approach in complexity roughly (solving A x = b)
        # jnp.linalg.solve is generally fast for this
        XTX = X_aug.T @ X_aug
        XTy = X_aug.T @ y
        beta = jnp.linalg.solve(XTX, XTy)
        return beta
else:
    def jax_ols_fit(X, y):
        pass

def run_benchmark():
    n_features = 20
    sample_sizes = [10_000, 100_000, 1_000_000, 3_000_000]
    
    print(f"{'N_Samples':<15} | {'Backend':<10} | {'Time (s)':<10} | {'Speedup (vs CPP)':<15}")
    print("-" * 60)

    for n in sample_sizes:
        print(f"Generating data for N={n}...")
        X_np, y_np = generate_data(n, n_features)
        
        # --- Statelix C++ ---
        start_cpp = time.perf_counter()
        model_cpp = StatelixOLS()
        model_cpp.fit(X_np, y_np)
        end_cpp = time.perf_counter()
        time_cpp = end_cpp - start_cpp
        
        print(f"{n:<15} | {'Statelix':<10} | {time_cpp:<10.5f} | {'1.0x':<15}")
        
        # --- JAX ---
        if HAS_JAX:
            # Transfer to device (include this in time? usually yes for end-to-end)
            # But let's measure pure computation separately if we want to be generous
            # Realistically, data starts on CPU.
            
            # WARMUP (JIT Compilation overhead)
            # We run once on small data or dummy data to compile
            if n == sample_sizes[0]:
                print("(Warming up JIT...)")
                _X = jnp.array(X_np[:100], dtype=jnp.float64)
                _y = jnp.array(y_np[:100], dtype=jnp.float64)
                _ = jax_ols_fit(_X, _y).block_until_ready()
            
            start_jax = time.perf_counter()
            X_jax = jnp.array(X_np, dtype=jnp.float64) # Transfer
            y_jax = jnp.array(y_np, dtype=jnp.float64) # Transfer
            
            beta_jax = jax_ols_fit(X_jax, y_jax)
            beta_jax.block_until_ready() # Wait for GPU/TPU
            
            end_jax = time.perf_counter()
            time_jax = end_jax - start_jax
            
            speedup = time_cpp / time_jax
            print(f"{n:<15} | {'JAX':<10} | {time_jax:<10.5f} | {f'{speedup:.2f}x':<15}")
            
            # Platform check
            if n == sample_sizes[0]:
                print(f"JAX Platform: {X_jax.devices()}")

        print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
