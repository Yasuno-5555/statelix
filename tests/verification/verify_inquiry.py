
import numpy as np
import pandas as pd
import statelix.inquiry
from statelix.inquiry.narrative import Storyteller

# Mock OLS for verification if C++ extension is missing
class MockOLSResult:
    def __init__(self, coef, r2, aic, bic):
        self.coef = coef
        self.r_squared = r2
        self.aic = aic
        self.bic = bic
        self.p_values = np.zeros_like(coef) # Dummy
    
    def predict(self, X):
        return X @ self.coef

class MockOLS:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.result = None
    
    def fit(self, y, X):
        # Naive implementation for testing Inquiry logic
        coef = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        y_pred = X @ coef
        resid = y - y_pred
        sse = np.sum(resid**2)
        n = len(y)
        k = len(coef)
        
        r2 = 1.0 - sse / np.sum((y - np.mean(y))**2)
        aic = 2*k + n*np.log(sse/n)
        bic = k*np.log(n) + n*np.log(sse/n)
        
        self.result = MockOLSResult(coef, r2, aic, bic)
        return self.result
    
    def predict(self, X):
        if self.result:
            return self.result.predict(X)
        return None
    
    # Simulate attribute access for Adapter that might check 'coef_' on model directly
    @property
    def coef_(self):
        return self.result.coef if self.result else None
        
    @property
    def aic(self):
        return self.result.aic if self.result else None

    @property
    def bic(self):
        return self.result.bic if self.result else None

    @property
    def rsquared(self):
        return self.result.r_squared if self.result else None

def verify_inquiry():
    print("--- Verifying Inquiry Extension (with Mock OLS) ---")
    
    # 1. Generate Data
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 2))
    beta = np.array([2.0, -1.0])
    y = X @ beta + np.random.normal(0, 1, 100)
    features = ['GDP', 'InterestRate']
    
    # 2. Train Models
    # OLS (Mock)
    ols = MockOLS(fit_intercept=False)
    ols.fit(y, X)
    
    # Bayes (Mock or skip)
    # bayes = statelix.bayes.BayesianLinearRegression(X, y)
    # bayes.fit() 
    
    # 3. Compare
    print("\n[Model Comparison]")
    # Only one model for now unless we mock Bayes too
    ranking = statelix.inquiry.compare_models([(ols, "Mock OLS")])
    print(ranking)
    
    # 4. Narrative
    print("\n[Narrative Generation - OLS]")
    story = Storyteller(ols, feature_names=features)
    print(story.explain())
    
    # 5. Counterfactual
    print("\n[Counterfactual - What If GDP + 1.0?]")
    wi = statelix.inquiry.WhatIf(ols, feature_names=features)
    # Increase GDP (col 0) by 1.0
    res = wi.simulate(X.mean(axis=0), {'GDP': lambda x: x + 1.0})
    
    print(f"Baseline Y: {res['baseline'][0]:.4f}")
    print(f"Scenario Y: {res['scenario'][0]:.4f}")
    print(f"Delta:      {res['delta'][0]:.4f}")
    print(f"Expected Delta (Beta[0]=2.0): {2.0:.4f}")
    
    if np.isclose(res['delta'][0], 2.0, atol=0.1):
        print("[PASS] Counterfactual Delta correct.")
    else:
        print("[WARN] Counterfactual Delta mismatch.")

if __name__ == "__main__":
    verify_inquiry()
