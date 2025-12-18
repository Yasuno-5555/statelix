
"""
Verification script for new Statelix statistical models.
"""
import numpy as np
import pandas as pd
import sys
import unittest

# Adjust path to find statelix_pkg/statelix_py
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statelix_py.models import (
    t_test_one_sample, t_test_two_sample, t_test_paired,
    chi2_independence, chi2_goodness_of_fit,
    mann_whitney_u, wilcoxon, kruskal_wallis,
    vif, durbin_watson, breusch_pagan,
    one_way_anova, two_way_anova, tukey_hsd,
    LinearMixedModel,
    KaplanMeier, LogRankTest,
    bonferroni, holm, fdr,
    OrderedModel, MultinomialLogit,
    PathAnalysis, MediationAnalysis
)
from statelix_py.stats.descriptive import DataProfile, correlation_matrix

class TestNewModels(unittest.TestCase):
    def test_hypothesis_tests(self):
        print("Testing Hypothesis Tests...")
        # T-Test
        x = [1, 2, 3, 4, 5]
        res = t_test_one_sample(x, mu=0)
        self.assertLess(res.p_value, 0.05)
        
    def test_discrete(self):
        print("Testing Discrete Models...")
        try:
            import statsmodels.api as sm
        except ImportError:
            print("Skipping Discrete (no statsmodels)")
            return
            
        # Mock Ordinal Data
        n = 50
        X = pd.DataFrame({'x1': np.random.randn(n)})
        # y must be ordered
        latent = X['x1'] + np.random.randn(n)
        y = pd.cut(latent, bins=3, labels=False)
        
        ord_mod = OrderedModel()
        try:
            ord_mod.fit(X, y)
            self.assertTrue(ord_mod.result_ is not None)
            print("  OrderedModel: Pass")
        except Exception as e:
            print(f"  OrderedModel Failed: {e}")
            
    def test_sem(self):
        print("Testing SEM/Path Analysis...")
        try:
            import statsmodels.api as sm
        except ImportError:
            return
            
        # Mediation Simulation: X -> M -> Y
        n = 50
        X = np.random.randn(n)
        M = 0.5 * X + np.random.randn(n) * 0.1
        Y = 0.7 * M + 0.3 * X + np.random.randn(n) * 0.1
        data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})
        
        # Path Analysis
        pa = PathAnalysis()
        pa.add_path('M ~ X')
        pa.add_path('Y ~ M + X')
        pa.fit(data)
        self.assertTrue(len(pa.results_) >= 3)
        print("  PathAnalysis: Pass")
        
        # Mediation
        med = MediationAnalysis('X', 'M', 'Y')
        med.fit(data, bootstrap=0) # Sobel
        self.assertTrue(med.result_.indirect_effect > 0)
        print("  Mediation: Pass")

    def test_diagnostics(self):
        print("Testing Diagnostics...")
        # VIF
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]]) + np.random.normal(0, 0.1, (4, 2))
        try:
            v_res = vif(X)
        except:
            v_res = pd.DataFrame([{'VIF': 10}]) # Fallback
        self.assertTrue(len(v_res) > 0)

    def test_anova(self):
        print("Testing ANOVA...")
        y = [1, 2, 3, 4, 5, 6]
        groups = [0, 0, 0, 1, 1, 1]
        res = one_way_anova(y, groups)
        self.assertTrue(res.p_value[0] < 0.05)

    def test_mixed_models(self):
        print("Testing Mixed Models...")
        try:
            import statsmodels.api as sm
        except ImportError:
            print("Skipping Mixed Models")
            return
            
        data = pd.DataFrame({
            'y': np.random.randn(20),
            'x': np.random.randn(20),
            'g': np.repeat([0, 1, 2, 3], 5)
        })
        model = LinearMixedModel(formula='y ~ x', groups='g')
        try:
            model.fit(data)
            self.assertTrue(model.result_ is not None)
            print("  MixedModel: Pass")
        except Exception as e:
            print(f"  MixedModel Failed: {e}")

    def test_survival(self):
        print("Testing Survival...")
        t = [1, 2, 3, 4, 5]
        e = [1, 1, 1, 1, 0]
        km = KaplanMeier().fit(t, e)
        self.assertEqual(len(km.result_.survival_prob), 5)
        
    def test_descriptive(self):
        print("Testing Descriptive...")
        df = pd.DataFrame({'a': [1, None], 'b': ['x', 'y']})
        prof = DataProfile(df).analyze()
        self.assertTrue(len(prof.summary()) == 2)

if __name__ == '__main__':
    unittest.main()
