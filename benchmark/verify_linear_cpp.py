
import sys
import numpy as np
from statelix.inquiry.adapters import StatelixLinearFactory, MockOLS

def test_integration():
    print("Testing StatelixLinearFactory...")
    OLSClass = StatelixLinearFactory.get_ols()
    print(f"Got OLS Class: {OLSClass}")
    
    if OLSClass is MockOLS:
        print("FAIL: Still using MockOLS fallback.")
        sys.exit(1)
        
    print("SUCCESS: Using C++ OLS Implementation.")
    
    # Functional Test
    print("\nRunning Functional Test (Fit)...")
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([3.0, 5.0, 7.0]) # y = 1 + 1*x1 + 0*x2 approx? 1+2=3, 2+3=5. actually y=2x1 + 1. x2 is correlated.
    
    # Try instantiation and fit
    try:
        model = OLSClass()
        print(f"Model instantiated: {model}")
        model.fit(X, y)
        print("Model fitted.")
        
        if hasattr(model, 'coef_'):
            print(f"Coefficients: {model.coef_}")
        if hasattr(model, 'r_squared'):
            print(f"R-squared: {model.r_squared}")
            
    except Exception as e:
        print(f"FAIL: Runtime error with C++ model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_integration()
