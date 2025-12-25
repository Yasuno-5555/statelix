
import pytest
from statelix_py.diagnostics.cit_detector import CITDetector, Discovery

def test_cit_fragility_detection():
    # Scenario: Low p-value (significant) but very low topology score
    metrics = {
        'p_value': 0.001,
        'topology_score': 0.2,
        'r2': 0.85
    }
    
    detector = CITDetector(metrics)
    discoveries = detector.detect()
    
    assert len(discoveries) > 0
    types = [d.type for d in discoveries]
    assert "Topological Fragility" in types

def test_cit_overfit_detection():
    # Scenario: High R2 but manifold has a cliff
    class MockPoint:
        def __init__(self, est): self.estimate = est
        
    points = [MockPoint(1.0), MockPoint(1.1), MockPoint(5.0), MockPoint(5.2)]
    metrics = {
        'r2': 0.95,
        'p_value': 0.01,
        'topology_score': 0.8
    }
    
    detector = CITDetector(metrics, points)
    discoveries = detector.detect()
    
    types = [d.type for d in discoveries]
    assert "Overfit Manifold" in types

if __name__ == "__main__":
    test_cit_fragility_detection()
    test_cit_overfit_detection()
    print("CIT Detector tests PASSED.")
