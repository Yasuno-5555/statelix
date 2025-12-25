import sys
import os
sys.path.append(os.path.abspath("."))

from statelix import mathuniverse

def test_risan():
    print("--- Testing Risan (Discrete) ---")
    node_a = mathuniverse.risan.Node("Node A")
    node_b = mathuniverse.risan.Node("Node B")
    
    print(f"Created {node_a.data} and {node_b.data}")
    
    node_a.connect(node_b)
    print(f"Connected {node_a.data} -> {node_b.data}")
    
    edges = node_a.edges
    print(f"Node A edges: {edges}")
    assert "Node B" in edges
    print("Risan test passed!")

def test_keirin():
    print("\n--- Testing Keirin (Topological) ---")
    persistence = mathuniverse.keirin.Persistence()
    
    # Simple triangle (V0, V1, V2)
    persistence.add_simplex([0], 0.0)
    persistence.add_simplex([1], 0.0)
    persistence.add_simplex([2], 0.0)
    persistence.add_simplex([0, 1], 1.0)
    persistence.add_simplex([1, 2], 1.0)
    persistence.add_simplex([0, 2], 1.5)
    
    print(f"Added {len(persistence.complex)} simplices to the complex.")
    
    print("Computing homology...")
    persistence.compute_homology()
    print("Keirin test passed!")

if __name__ == "__main__":
    try:
        test_risan()
        test_keirin()
        print("\nAll MathUniverse skeleton bindings verified successfully!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
