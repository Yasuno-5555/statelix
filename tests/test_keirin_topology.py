
import sys
import os

# Ensure we can import the built extensions
# The extensions seem to be in statelix_py
sys.path.append(os.path.abspath("statelix_py"))

try:
    import mathuniverse
except ImportError:
    # If not found directly, try finding where it is
    print("Could not import mathuniverse directly. Checking common paths...")
    # It might be registered as statelix.mathuniverse if the user installed it or specific path mapping
    try:
        from statelix import mathuniverse
    except ImportError:
        print("Could not import statelix.mathuniverse either.")
        raise

print("MathUniverse imported successfully.")

def test_keirin_collapse():
    print("\n--- Testing Keirin Topology Collapse Alert ---")
    p = mathuniverse.keirin.Persistence()
    
    # 1. Normal State: Stable high-dimensional loops or simple noise
    # We simulate a sequence of diagrams where the structure score is relatively stable
    print("Simulating stable phase...")
    for i in range(5):
        # Clear complex for new frame (conceptually, we just make a new object or clear it)
        # The binding doesn't expose clear(), so we make a new Persistence object or just use the history?
        # Our Persistence object stores history but re-computing on the same object appends to history.
        # But we need to change the complex data.
        # The 'complex' attribute is exposed as read-write, so we can clear it.
        
        p.complex = [] 
        # Add some noise (small gaps)
        p.add_simplex([0], 0.1)
        p.add_simplex([1], 0.15)
        p.add_simplex([0, 1], 0.2) # Small lifetime
        
        # Add a "feature"
        p.add_simplex([2], 0.1)
        p.add_simplex([3], 0.1)
        p.add_simplex([2, 3], 0.8) # Lifetime 0.7
        
        p.compute_homology()
        is_jump = p.detect_jump(threshold=0.5)
        print(f"Frame {i}: Score={p.structure_score:.4f}, Jump={is_jump}")
        assert not is_jump, f"False positive jump at stable frame {i}"

    # 2. Collapse Event: The feature disappears (e.g., over-smoothing or data loss)
    print("Simulating collapse event...")
    p.complex = []
    # Only noise
    p.add_simplex([0], 0.1)
    p.add_simplex([1], 0.15)
    p.add_simplex([0, 1], 0.2)
    # No large feature
    
    p.compute_homology()
    is_jump = p.detect_jump(threshold=0.5)
    print(f"Collapse Frame: Score={p.structure_score:.4f}, Jump={is_jump}")
    
    if is_jump:
        print("SUCCESS: Topology collapse detected!")
    else:
        print("FAILURE: Topology collapse NOT detected!")
        # Debug info
        print(f"History: {p.history_scores}")

if __name__ == "__main__":
    test_keirin_collapse()
