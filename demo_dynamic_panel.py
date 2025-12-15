
import numpy as np
import pandas as pd
import statelix_core as slx
from statelix_core.panel import DynamicPanelGMM, GMMType

def generate_dynamic_panel_data(n_units=100, n_periods=10, gamma=0.5, beta=1.0):
    """Generate dynamic panel data: y_it = gamma*y_i,t-1 + beta*x_it + alpha_i + eps_it"""
    np.random.seed(42)
    
    ids = []
    times = []
    Y_list = []
    X_list = []
    
    alpha = np.random.normal(0, 1, n_units)
    
    for i in range(n_units):
        y_prev = 0 # Initial condition (steady state approx 0)
        
        for t in range(n_periods):
            ids.append(i)
            times.append(t)
            
            x = np.random.normal(0, 1) # Exogenous
            eps = np.random.normal(0, 1)
            
            y = gamma * y_prev + beta * x + alpha[i] + eps
            
            Y_list.append(y)
            X_list.append([x]) # X includes only exogenous here
            
            y_prev = y
            
    return (np.array(Y_list), np.array(X_list), 
            np.array(ids, dtype=np.int32), np.array(times, dtype=np.int32))

def main():
    print("Generating synthetic Dynamic Panel data (N=100, T=10)...")
    print("True parameters: gamma=0.5, beta=1.0")
    
    Y, X, uid, tid = generate_dynamic_panel_data(100, 10, 0.5, 1.0)
    
    # 1. Arellano-Bond (Difference GMM)
    print("\n--- 1. Arellano-Bond (Difference GMM) ---")
    ab = DynamicPanelGMM()
    ab.type = GMMType.DIFFERENCE
    ab.two_step = True
    result_ab = ab.fit(Y, X, uid, tid)
    
    print(f"Gamma (lagged Y): {result_ab.gamma:.4f} (SE: {result_ab.gamma_se:.4f})")
    print(f"Beta:             {result_ab.beta[0]:.4f} (SE: {result_ab.beta_se[0]:.4f})")
    print(f"Sargan Test:      {result_ab.sargan_stat:.4f} (p={result_ab.sargan_pvalue:.4f})")
    print(f"Hansen J Test:    {result_ab.hansen_stat:.4f} (p={result_ab.hansen_pvalue:.4f})")
    print(f"AR(1) Test:       {result_ab.ar1_stat:.4f} (p={result_ab.ar1_pvalue:.4f})")
    print(f"AR(2) Test:       {result_ab.ar2_stat:.4f} (p={result_ab.ar2_pvalue:.4f})")
    
    # 2. Blundell-Bond (System GMM)
    print("\n--- 2. Blundell-Bond (System GMM) ---")
    bb = DynamicPanelGMM()
    bb.type = GMMType.SYSTEM
    bb.two_step = True
    result_bb = bb.fit(Y, X, uid, tid)
    
    print(f"Gamma (lagged Y): {result_bb.gamma:.4f} (SE: {result_bb.gamma_se:.4f})")
    print(f"Beta:             {result_bb.beta[0]:.4f} (SE: {result_bb.beta_se[0]:.4f})")
    print(f"Hansen J Test:    {result_bb.hansen_stat:.4f} (p={result_bb.hansen_pvalue:.4f})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: C++ extension required for execution.")
