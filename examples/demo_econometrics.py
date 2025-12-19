import numpy as np
import pandas as pd
try:
    import statelix_core as stx
    print("Statelix Core loaded successfully!")
except ImportError:
    print("Error: Could not import statelix_core. Please ensure it is built.")
    print("Run: pip install -e .")
    # Mocking for demo purpose if not built
    class Mock:
        pass
    stx = Mock()
    stx.causal = Mock()
    stx.panel = Mock()
    stx.spatial = Mock()
    stx.QuantileRegression = lambda: None

def demo_synthetic_control():
    print("\n=== Synthetic Control Demo ===")
    # Generate mock data: 20 units, 40 periods
    # Unit 0 is treated at period 30
    np.random.seed(42)
    Y = np.random.randn(20, 40)
    # Effect
    Y[0, 30:] += 5.0
    
    if hasattr(stx.causal, 'SyntheticControl'):
        sc = stx.causal.SyntheticControl()
        print("Fitting Synthetic Control...")
        # treated_idx=0, treatment_period=30
        try:
            result = sc.fit(Y, 0, 30)
            print(f"ATT: {result.att:.4f}")
            print(f"Pre-RMSPE: {result.pre_rmspe:.4f}")
            print(f"Post-RMSPE: {result.post_rmspe:.4f}")
            
            # Placebo
            print("Running Placebo Test...")
            placebo = sc.placebo_test(Y, 0, 30)
            print(f"Placebo p-value: {placebo.p_value:.4f}")
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        print("SyntheticControl not available in build.")

def demo_dynamic_panel():
    print("\n=== Dynamic Panel GMM Demo ===")
    # N=100, T=10
    N, T = 100, 10
    unit_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)
    
    # y_it = 0.5 * y_i,t-1 + x_it + u_i + e_it
    X = np.random.randn(N*T, 1)
    Y = np.zeros(N*T)
    
    if hasattr(stx.panel, 'DynamicPanelGMM'):
        gmm = stx.panel.DynamicPanelGMM()
        gmm.type = stx.panel.GMMType.SYSTEM
        print("Fitting System GMM...")
        try:
            result = gmm.fit(Y, X, unit_id, time_id)
            print(f"Gamma (Lag coef): {result.gamma:.4f}")
            print(f"Beta: {result.beta[0]:.4f}")
            print(f"Hansen J p-value: {result.hansen_pvalue:.4f}")
            print(f"AR(2) p-value: {result.ar2_pvalue:.4f}")
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        print("DynamicPanelGMM not available in build.")

def demo_spatial():
    print("\n=== Spatial Econometrics Demo ===")
    n = 50
    coords = np.random.rand(n, 2)
    X = np.random.randn(n, 2)
    y = np.random.randn(n)
    
    if hasattr(stx.spatial, 'SpatialRegression'):
        # Weights
        W = stx.spatial.SpatialWeights.knn_weights(coords, 5)
        
        sr = stx.spatial.SpatialRegression()
        print("Running LM Tests...")
        try:
            lm = sr.lm_tests(y, X, W)
            print(f"Recommendation: {lm.recommendation}")
            
            sr.model = stx.spatial.SpatialModel.SAR
            print("Fitting SAR Model...")
            result = sr.fit(y, X, W)
            print(f"Rho (Spatial Lag): {result.rho:.4f}")
            print(f"Direct Effects: {result.direct_effects}")
            print(f"Indirect Effects: {result.indirect_effects}")
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        print("SpatialRegression not available in build.")

def demo_quantile():
    print("\n=== Quantile Regression Demo ===")
    n = 100
    X = np.random.randn(n, 2)
    y = X[:, 0] + np.random.randn(n) # Heteroskedasticity?
    
    if hasattr(stx, 'QuantileRegression'):
        qr = stx.QuantileRegression()
        tau = 0.5
        print(f"Fitting Median Regression (tau={tau})...")
        try:
            result = qr.fit(y, X, tau)
            print(f"Coefficients: {result.coef}")
            print(f"Pseudo R2: {result.pseudo_r_squared:.4f}")
            
            print("Checking Quantile Process...")
            proc = qr.quantile_process(y, X, [0.1, 0.5, 0.9])
            print(f"Wald Test p-value: {proc.wald_pvalue:.4f}")
        except Exception as e:
            print(f"Execution failed: {e}")
    else:
        print("QuantileRegression not available in build.")

if __name__ == "__main__":
    demo_synthetic_control()
    demo_dynamic_panel()
    demo_spatial()
    demo_quantile()
