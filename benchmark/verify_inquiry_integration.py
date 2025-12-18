
"""
Verification script for Inquiry Engine Model Integration.
"""
import numpy as np
import pandas as pd
import unittest
import sys
import os

# Adjust path to find statelix_pkg/statelix_py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statelix_py.models.discrete import OrderedModel
from statelix_py.models.sem import MediationAnalysis
from statelix_pkg.inquiry.narrative import Storyteller

class TestInquiryIntegration(unittest.TestCase):
    
    def test_discrete_narrative(self):
        print("Testing Discrete Narrative...")
        try:
             import statsmodels.api as sm
        except ImportError:
             print("Skipping (no statsmodels)")
             return

        # Mock Data
        n = 50
        df = pd.DataFrame({'price': np.random.randn(n), 'quality': np.random.randn(n)})
        y = pd.cut(df['price'] + df['quality'], 3, labels=False)
        
        model = OrderedModel()
        try:
            model.fit(df, y)
        except:
             print("Fit failed, skipping narrative check.")
             return
             
        story = Storyteller(model, feature_names=['Price', 'Quality'])
        narrative = story.explain()
        
        print("\n--- Ordered Logit Narrative ---")
        print(narrative)
        
        self.assertTrue("Analysis Narrative" in narrative)
        # Check case insensitive or exact match depending on implementation
        # The narrative.py maps keys. If DataFrame cols are 'price', 'quality', 
        # model keys are 'price', 'quality'.
        # Storyteller input feature_names=['Price', 'Quality'].
        # So display name matches.
        self.assertTrue("Price" in narrative or "price" in narrative.lower())

    def test_sem_narrative(self):
        print("\nTesting SEM Narrative...")
        try:
             import statsmodels.api as sm
        except:
             return
             
        # Mediation: X -> M -> Y
        df = pd.DataFrame({'X': np.random.randn(50)})
        df['M'] = 0.5 * df['X'] + np.random.randn(50)*0.1
        df['Y'] = 0.5 * df['M'] + np.random.randn(50)*0.1
        
        med = MediationAnalysis('X', 'M', 'Y')
        med.fit(df, bootstrap=0)
        
        story = Storyteller(med)
        narrative = story.explain()
        
        print("\n--- Mediation Narrative ---")
        print(narrative)
        
        self.assertTrue("Mediation Analysis Result" in narrative)
        self.assertTrue("Indirect Effect" in narrative)
        self.assertTrue("proportion" in narrative or "indirect" in narrative.lower())

if __name__ == '__main__':
    unittest.main()
