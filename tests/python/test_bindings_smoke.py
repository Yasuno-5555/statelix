
import sys
import unittest
import numpy as np
try:
    # Try importing from package structure first
    try:
        from statelix_py.core import statelix_core as sc
    except ImportError:
        import statelix_core as sc
except ImportError as e:
    print(f"CRITICAL: Could not import statelix_core: {e}")
    # Print sys.path for debugging
    print("sys.path:", sys.path)
    import os
    # debug directory
    print("CWD contents:", os.listdir("."))
    if os.path.exists("statelix_py/core"):
        print("statelix_py/core contents:", os.listdir("statelix_py/core"))
    sys.exit(1)

class TestBindingsSmoke(unittest.TestCase):
    def setUp(self):
        print(f"Testing bindings for: {self._testMethodName}")

    def test_stats_module(self):
        """Test basic statistical tests and diagnostics."""
        print("Checking stats module...")
        
        # 1. Condition Number
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        res = sc.stats.condition_number(X)
        self.assertAlmostEqual(res.condition_number, 1.0)
        print("  - condition_number: OK")
        
        # 2. Durbin-Watson
        resid = np.random.randn(100)
        dw = sc.stats.tests.durbin_watson(resid, 100, 2)
        print(f"  - durbin_watson: {dw.dw_statistic:.4f} (OK)")
        
        # 3. Breusch-Godfrey (Newly implemented)
        X_mat = np.random.randn(100, 2)
        bg = sc.stats.tests.breusch_godfrey(X_mat, resid, 1)
        print(f"  - breusch_godfrey: p={bg.lm_pvalue:.4f} (OK)")

    def test_panel_module(self):
        """Test Dynamic Panel GMM bindings."""
        print("Checking panel module...")
        
        if not hasattr(sc.panel, "DynamicPanelGMM"):
            self.fail("DynamicPanelGMM not exposed in sc.panel")
            
        gmm = sc.panel.DynamicPanelGMM()
        gmm.two_step = True
        gmm.max_lags = 2
        
        # Dummy data for estimation smoke test
        # 10 individuals, 10 periods
        ids = []
        times = []
        y = []
        X = []
        
        for i in range(10):
            for t in range(10):
                ids.append(i)
                times.append(t)
                y.append(np.random.randn())
                X.append([np.random.randn(), np.random.randn()])
                
        y_vec = np.array(y)
        X_mat = np.array(X)
        ids_vec = np.array(ids, dtype=np.int32)
        times_vec = np.array(times, dtype=np.int32)
        
        try:
            res = gmm.estimate(y_vec, X_mat, ids_vec, times_vec)
            print(f"  - DynamicPanelGMM.estimate: Coefs={res.coefficients.flatten()} (OK)")
        except Exception as e:
            self.fail(f"DynamicPanelGMM estimation failed: {e}")

    def test_bayes_module(self):
        """Test HMC bindings."""
        print("Checking bayes module...")
        
        if not hasattr(sc.bayes, "HMCConfig"):
            self.fail("HMCConfig not exposed in sc.bayes")
            
        config = sc.bayes.HMCConfig()
        config.n_samples = 10
        config.n_warmup = 5
        config.step_size = 0.1
        
        # Mock target density: LogNormal(0, 1) -> log p(x) = -0.5 * x^2
        # Gradient: -x
        def log_prob(x):
            return -0.5 * np.sum(x**2), -x
            
        init_param = np.array([0.5])
        
        try:
            res = sc.bayes.hmc_sample(init_param, log_prob, config)
            print(f"  - hmc_sample: Samples={len(res.samples)} (OK)")
            self.assertTrue(len(res.samples) > 0)
        except Exception as e:
            self.fail(f"HMC sampling failed: {e}")

if __name__ == '__main__':
    unittest.main()
