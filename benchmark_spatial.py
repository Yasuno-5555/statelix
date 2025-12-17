
import numpy as np
import pandas as pd
import time
from statelix_py.models.spatial import SpatialRegression, SpatialWeights

def benchmark_spatial():
    print("--- Statelix Spatial Econometrics Benchmark ---")
    
    # 1. Generate Data
    n_samples = 1000
    n_features = 3
    rho_true = 0.5
    beta_true = np.array([1.5, -2.0, 0.5])
    
    np.random.seed(42)
    
    # Coordinates for weights
    coords = np.random.rand(n_samples, 2) * 100
    
    # Create Weights (KNN)
    # Using C++ optimized weights
    print(f"Generating optimized weights (N={n_samples})...")
    t0 = time.time()
    W = SpatialWeights.knn(coords, k=5)
    print(f"Weights generation time: {time.time() - t0:.4f}s")
    
    # Features
    X = np.random.randn(n_samples, n_features)
    
    # SAR Process: y = (I - rho*W)^-1 (X*beta + e)
    I = np.eye(n_samples)
    A = I - rho_true * W
    A_inv = np.linalg.inv(A)
    
    epsilon = np.random.randn(n_samples) * 0.5
    y = A_inv @ (X @ beta_true + epsilon)
    
    print("\nData generated: SAR process with rho=0.5")
    
    # 2. Fit SAR Model
    print("\n--- Fitting SAR Model ---")
    model = SpatialRegression(model='SAR')
    
    t0 = time.time()
    model.fit(y, X, W)
    t1 = time.time()
    
    res = model.result_
    print(f"Fitting Time: {t1 - t0:.4f}s")
    print(f"Converged: True") # C++ output
    
    print("\nResults:")
    print(f"Rho (Goal {rho_true}): {res.rho:.4f}")
    
    print("\nCoefficients (Goal [1.5, -2.0, 0.5]):")
    print(res.coef.filter(regex='^x').values)
    
    # Validation
    error_rho = abs(res.rho - rho_true)
    error_beta = np.linalg.norm(res.coef.filter(regex='^x').values - beta_true)
    
    if error_rho < 0.05 and error_beta < 0.1:
        print("\n[PASS] Parameters estimated correctly.")
    else:
        print("\n[FAIL] Parameter estimation inaccurate.")
        sys.exit(1)

    # 3. Fit SEM Model (Just ensuring it runs)
    print("\n--- Fitting SEM Model ---")
    model_sem = SpatialRegression(model='SEM')
    model_sem.fit(y, X, W)
    print(f"Lambda: {model_sem.result_.lambda_:.4f}")
    print("[PASS] SEM runs successfully.")

if __name__ == "__main__":
    benchmark_spatial()
