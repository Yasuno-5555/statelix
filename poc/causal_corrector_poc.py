import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Ensure statelix is in path
sys.path.append(os.path.abspath("."))
from statelix import mathuniverse

def generate_non_linear_data(n=200):
    """
    X -> Y
    Y -> Z (Linear)
    X -> Z (Non-linear, e.g., square or sine)
    """
    print(f"Generating synthetic non-linear data (N={n})...")
    np.random.seed(42)
    X = np.random.randn(n)
    Y = 0.5 * X + np.random.randn(n) * 0.1
    # Hidden non-linearity: X**2
    Z = 0.3 * Y + 1.0 * (X**2) + np.random.randn(n) * 0.05
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

def run_feedback_loop(data):
    print("\n=== Starting Causal-Topological Feedback Loop ===")
    
    current_features = ['Y'] # Intentionally missing X (and its non-linear form)
    target = 'Z'
    
    for iteration in range(1, 4):
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current Model Features: {current_features}")
        
        X_matrix = data[current_features].values
        z_vec = data[target].values
        
        model = LinearRegression()
        model.fit(X_matrix, z_vec)
        residuals = z_vec - model.predict(X_matrix)
        r2 = model.score(X_matrix, z_vec)
        
        print(f"Linear Model R2: {r2:.4f}")
        
        # Keirin Diagnostic
        persistence = mathuniverse.keirin.Persistence()
        res_sorted = np.sort(residuals)
        for i, val in enumerate(res_sorted):
            persistence.add_simplex([i], float(abs(val)))
            if i > 0:
                persistence.add_simplex([i-1, i], float(abs(res_sorted[i] - res_sorted[i-1])))
        
        persistence.compute_homology()
        score = persistence.structure_score
        
        if score < 0.2:
            print(f"Topological Cleanliness Achieved! (Score: {score:.4f})")
            break
        else:
            print(f"High Structure Detected in Residuals (Score: {score:.4f})")
            
            # Automated Suggestion Logic (Simulated for PoC)
            if iteration == 1:
                print("Suggestion: Missing primary causal factor detected. Adding 'X'.")
                current_features.append('X')
            elif iteration == 2:
                print("Suggestion: Persistence persists. Non-linearity suspected for 'X'. Adding 'X^2'.")
                data['X_sq'] = data['X']**2
                current_features.append('X_sq')
            else:
                print("Max iterations reached. Structural limit found.")
                break

if __name__ == "__main__":
    df = generate_non_linear_data()
    run_feedback_loop(df)
    print("\nFeedback Loop PoC Complete.")
