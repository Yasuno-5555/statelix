
import sys
import os
import math

sys.path.append(os.path.abspath("statelix_py"))

try:
    import mathuniverse
except ImportError:
    from statelix import mathuniverse

print("MathUniverse imported successfully.")

def test_shinen_invariants():
    print("\n--- Testing Shinen Invariant Feature Ranking ---")
    
    # Define features
    # 1. Scalar (Invariant)
    v1 = mathuniverse.shinen.MultiVector.scalar(5.0)
    
    # 2. Vector in XY plane (Rotates)
    v2 = mathuniverse.shinen.MultiVector.vector(1.0, 0.0, 0.0)
    
    # 3. Vector along Z axis (Invariant under XY rotation)
    v3 = mathuniverse.shinen.MultiVector.vector(0.0, 0.0, 1.0)
    
    # 4. Pseudoscalar (Invariant under rotation in 3D? Orientation might flip but magnitude stays? 
    # Actually rotor rotation R I ~R = I R ~R = I if R commutes.
    # Rotors in G3 (even subalg) commute with pseudoscalar if they are even?
    # Let's test it.
    v4 = mathuniverse.shinen.MultiVector.pseudoscalar(1.0)
    
    names = ["Scalar", "Vector_XY", "Vector_Z", "Pseudoscalar"]
    vectors = [v1, v2, v3, v4]
    
    scores = mathuniverse.shinen.InvariantSensors.rank_invariants(names, vectors)
    
    print("Invariant Scores (Higher is better):")
    for s in scores:
        print(s)
        
    # Validation logic
    score_map = {s.feature_name: s.score for s in scores}
    
    # Scalar should be perfect (score ~ 1.0)
    assert score_map["Scalar"] > 0.99, "Scalar should be invariant"
    
    # Vector in XY plane should change significantly (score < 1.0)
    # Rotating e1 by 90 deg -> e2. Diff is |e1 - e2| = sqrt(2).
    # Score = 1 / (1 + sqrt(2)) ~= 1 / 2.414 ~= 0.41
    assert score_map["Vector_XY"] < 0.9, "XY Vector should vary under XY rotation"
    
    # Vector along Z should be invariant under XY rotation
    assert score_map["Vector_Z"] > 0.99, "Z Vector should be invariant under XY rotation"

    print("SUCCESS: Geometric invariant ranking verified.")

if __name__ == "__main__":
    test_shinen_invariants()
