
import sys
import os

sys.path.append(os.path.abspath("statelix_py"))

try:
    import mathuniverse
except ImportError:
    from statelix import mathuniverse

print("MathUniverse imported successfully.")

def test_risan_rewiring():
    print("\n--- Testing Risan Self-Healing Graph ---")
    
    g = mathuniverse.risan.Graph()
    
    # Create nodes
    n1 = mathuniverse.risan.Node("X")
    n2 = mathuniverse.risan.Node("Y")
    n3 = mathuniverse.risan.Node("Z")
    
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    
    # Create a loop/cycle X <-> Y which implies ambiguity or feedback
    n1.connect(n2) # X -> Y
    n2.connect(n1) # Y -> X
    
    # X -> Z
    n1.connect(n3)
    
    # Get suggestions
    suggestions = g.suggest_rewiring()
    print(f"Suggestions found: {len(suggestions)}")
    
    found_fix = False
    for s in suggestions:
        print(s)
        if s.action == "REMOVE" and (
            (s.source == "X" and s.target == "Y") or 
            (s.source == "Y" and s.target == "X")
        ):
            found_fix = True
            
    assert found_fix, "Should suggest removing one edge of the 2-cycle"
    print("SUCCESS: Self-healing rewiring verified.")

if __name__ == "__main__":
    test_risan_rewiring()
