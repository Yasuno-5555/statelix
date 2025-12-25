
import sys
import os

sys.path.append(os.path.abspath("statelix_py"))

try:
    import mathuniverse
except ImportError:
    from statelix import mathuniverse

print("MathUniverse imported successfully.")

def test_ryoshi_early_stopping():
    print("\n--- Testing Ryoshi Causal Early Stopping ---")
    
    # Initialize search with high temperature
    search = mathuniverse.ryoshi.CausalSearch(initial_temp=10.0)
    
    stopping_triggered = False
    
    # Simulate a search process
    # Scores increase initially then plateau
    scores = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.88, 0.89, 0.895, 0.898, 0.899, 0.9, 0.9, 0.9]
    
    for i, score in enumerate(scores):
        search.step(score)
        stop = search.check_early_stopping(tolerance=0.01)
        print(f"Step {i}: Score={score:.3f}, Temp={search.temperature:.3f}, Stop={stop}")
        
        if stop:
            stopping_triggered = True
            print(f"Early stopping triggered at step {i}")
            break
            
    assert stopping_triggered, "Early stopping should have been triggered"
    print("SUCCESS: Causal early stopping verified.")

if __name__ == "__main__":
    test_ryoshi_early_stopping()
