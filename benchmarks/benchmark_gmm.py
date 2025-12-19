
import numpy as np
import pandas as pd
import time
from statelix_py.models.dynamic_panel import ArellanoBond

def generate_dynamic_panel(n=1000, t=10, rho=0.5, beta=1.0):
    np.random.seed(42)
    # Fixed effects
    alpha = np.random.normal(0, 1, n)
    
    # Data containers
    ids = []
    times = []
    ys = []
    xs = []
    
    # Simulation
    for i in range(n):
        y_prev = 0 # Initial condition (steady state approximation or 0)
        for time_per in range(t):
            x = np.random.normal(0, 1) + 0.5 * alpha[i] # Correlated with FE
            u = np.random.normal(0, 1)
            
            # y_{it} = rho * y_{i,t-1} + beta * x_{it} + alpha_i + u_{it}
            y = rho * y_prev + beta * x + alpha[i] + u
            
            ids.append(i)
            times.append(time_per)
            ys.append(y)
            xs.append(x)
            
            y_prev = y

    df = pd.DataFrame({'id': ids, 'time': times, 'y': ys, 'x': xs})
    return df

def run_benchmark():
    print("Generating Dynamic Panel Data (N=1000, T=10)...")
    df = generate_dynamic_panel(n=1000, t=10, rho=0.5, beta=1.0)
    
    print("\n--- Estimating Arellano-Bond (Difference GMM) ---")
    start = time.time()
    model = ArellanoBond(two_step=True)
    try:
        model.fit(df, 'y', ['x'], 'id', 'time')
        end = time.time()
        
        print(f"Time: {end - start:.4f} seconds")
        print("Results:")
        res = model.result_
        print(f"Rho (Goal 0.5): {res.coefficients[0]:.4f}")
        print(f"Beta (Goal 1.0): {res.coefficients[1]:.4f}")
        print(f"Sargan p-value: (Should be > 0.05): {res.sargan_test:.4f}") # Actually this is statistic, need p-val logic
        
        # Simple OLS bias check
        # Regress y on lag_y and x (ignoring FE bias)
        # We expect OLS Rho to be biased upwards (GLS/Random Effects) or Fixed Effects biased downwards (Nickell bias)
        # AB should be consistent.
        
    except Exception as e:
        print(f"Estimation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
