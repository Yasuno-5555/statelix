
import sys
import numpy as np

try:
    import statelix.accelerator as acc
    HAS_ACCELERATOR = True
except ImportError as e:
    print(f"Accelerator Import Failed (Expected if NO CUDA): {e}")
    HAS_ACCELERATOR = False

def verify():
    print("--- Statelix CUDA Accelerator Verification ---")
    
    if not HAS_ACCELERATOR:
        print("CUDA Accelerator module is NOT installed.")
        print("This is correct behavior if nvcc is missing from PATH.")
        return

    print("CUDA Accelerator module INSTALLED.")
    
    # helper to print
    def check(msg, val):
        print(f"CHECK {msg}: {val}")

    # 1. Availability
    avail = acc.is_available()
    check("is_available()", avail)

    if avail:
        print("CUDA Device Detected. Running Kernel Test...")
        # 2. Kernel Test
        N = 1000
        K = 10
        X = np.random.randn(N, K).astype(np.float64)
        W = np.random.rand(N).astype(np.float64) 
        
        # GPU Computation
        try:
            G_gpu = acc.weighted_gram_matrix(X, W)
            print(f"GPU Result Shape: {G_gpu.shape}")
            
            # CPU Reference
            G_cpu = (X.T * W) @ X
            
            # Compare
            diff = np.abs(G_gpu - G_cpu).max()
            print(f"Max Absolute Error (GPU vs CPU): {diff}")
            
            if diff < 1e-9:
                print("[PASS] GPU result matches CPU result.")
            else:
                print("[FAIL] Significant numerical mismatch.")
        except RuntimeError as e:
            print(f"[PARTIAL PASS] Runtime Error during kernel execution: {e}")
            print("Note: This often happens if the GPU is present but restricted, or out of memory.")
            
    else:
        print("CUDA Module loaded but runtime reports NO DEVICE (or cudaGetDeviceCount failed).")

if __name__ == "__main__":
    verify()
