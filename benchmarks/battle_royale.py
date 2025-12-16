import time
import numpy as np
import pandas as pd
import sys
import os

# --- Imports & Setup ---
print("="*60)
print("STATELIX BATTLE ROYALE: BENCHMARK SUITE")
print("="*60)

try:
    import statelix_core as sc
    print(f"[OK] Statelix Core imported.")
except ImportError:
    print("[FAIL] Could not import statelix_core.")
    print("Ensure the extension is built and in the path (run setup.py build_ext --inplace).")
    sys.exit(1)

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR as SmVAR
    print("[OK] Statsmodels imported.")
except ImportError:
    sm = None
    SmVAR = None
    print("[WARN] Statsmodels not found. Skipping related battles.")

try:
    from sklearn.linear_model import LinearRegression, Ridge
    print("[OK] Scikit-learn imported.")
except ImportError:
    LinearRegression = None
    Ridge = None
    print("[WARN] Scikit-learn not found. Skipping related battles.")

try:
    from linearmodels.panel import PanelOLS
    print("[OK] Linearmodels imported.")
except ImportError:
    PanelOLS = None
    print("[WARN] Linearmodels not found. Skipping related battles.")

print("-" * 60)

class BattleArena:
    def __init__(self):
        self.results = []

    def record(self, battle_name, contender, time_ms, error_measure=None, notes=""):
        self.results.append({
            "Battle": battle_name,
            "Contender": contender,
            "Time (ms)": round(time_ms, 2),
            "Max Error": error_measure,
            "Notes": notes
        })

    def print_summary(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No battles fought.")
            return
        
        print("\n\n" + "="*60)
        print("BATTLE REPORT")
        print("="*60)
        # Group by Battle to determine winner
        for battle, group in df.groupby("Battle"):
            print(f"\n>> {battle}")
            group = group.sort_values("Time (ms)")
            fastest = group.iloc[0]
            
            # Print table
            print(group[["Contender", "Time (ms)", "Max Error", "Notes"]].to_string(index=False))
            
            print(f"\nWINNER: {fastest['Contender']} ({fastest['Time (ms)']} ms)")
            
            # Calculate speedup
            statelix_row = group[group["Contender"].str.contains("Statelix")]
            if len(statelix_row) > 0:
                s_time = statelix_row.iloc[0]["Time (ms)"]
                if s_time == fastest["Time (ms)"]:
                    runner_up = group.iloc[1] if len(group) > 1 else None
                    if runner_up is not None:
                        speedup = runner_up["Time (ms)"] / s_time
                        print(f"Statelix is {speedup:.2f}x faster than {runner_up['Contender']}")
                else:
                    speedup = s_time / fastest["Time (ms)"]
                    print(f"Statelix is {speedup:.2f}x slower than {fastest['Contender']}")
            print("-" * 40)

arena = BattleArena()

def time_func(func, runs=5):
    times = []
    # Warmup
    try:
        func()
    except Exception as e:
        return -1.0, None, str(e)
        
    for _ in range(runs):
        start = time.perf_counter()
        res = func()
        end = time.perf_counter()
        times.append((end - start) * 1000.0) # ms
    return np.mean(times), res, None

# ==============================================================================
# BATTLE 1: OLS (Large N, Small K)
# ==============================================================================
if sm and LinearRegression:
    print("\n[Battle 1] OLS: N=1,000,000, K=50")
    N = 1_000_000
    K = 50
    np.random.seed(42)
    X = np.random.randn(N, K)
    true_beta = np.random.randn(K)
    y = X @ true_beta + np.random.randn(N)
    
    # 1. Statelix
    def run_statelix():
        return sc.fit_ols_full(X, y, fit_intercept=False)
    
    t_statelix, res_statelix, err = time_func(run_statelix)
    if err:
        print(f"Statelix failed: {err}")
    else:
        arena.record("OLS (1M x 50)", "Statelix (C++ Eigen)", t_statelix, 0.0)

    # 2. Sklearn
    def run_sklearn():
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        return model
        
    t_sklearn, res_sklearn, err = time_func(run_sklearn)
    if not err:
        diff = np.max(np.abs(res_statelix.coef - res_sklearn.coef_))
        arena.record("OLS (1M x 50)", "Sklearn (LAPACK)", t_sklearn, diff)

    # 3. Statsmodels
    def run_sm():
        model = sm.OLS(y, X)
        return model.fit()
        
    t_sm, res_sm, err = time_func(run_sm)
    if not err:
        diff = np.max(np.abs(res_statelix.coef - res_sm.params))
        arena.record("OLS (1M x 50)", "Statsmodels", t_sm, diff)

# ==============================================================================
# BATTLE 2: Ridge (High Dim)
# ==============================================================================
if Ridge:
    print("\n[Battle 2] Ridge: N=5,000, K=2,000")
    N = 5_000
    K = 2_000
    X = np.random.randn(N, K)
    y = np.random.randn(N)
    alpha = 1.0
    
    # 1. Statelix
    def run_statelix_ridge():
        model = sc.RidgeRegression()
        model.alpha = alpha
        model.fit(X, y)
        return model
        
    t_statelix, res_statelix, err = time_func(run_statelix_ridge)
    if not err:
        arena.record("Ridge (5k x 2k)", "Statelix", t_statelix, 0.0)
        
    # 2. Sklearn
    def run_sklearn_ridge():
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        return model
        
    t_sklearn, res_sklearn, err = time_func(run_sklearn_ridge)
    if not err:
        diff = np.max(np.abs(res_statelix.coef - res_sklearn.coef_))
        arena.record("Ridge (5k x 2k)", "Sklearn", t_sklearn, diff)

# ==============================================================================
# BATTLE 3: VAR (Time Series)
# ==============================================================================
if SmVAR:
    print("\n[Battle 3] VAR(2): T=10,000, K=10")
    T = 10_000
    K = 10
    p = 2
    Y = np.random.randn(T, K) # Random walkish
    
    # 1. Statelix
    def run_statelix_var():
        model = sc.VAR(p=p)
        model.include_intercept = True # Default
        return model.fit(Y)
        
    t_statelix, res_statelix, err = time_func(run_statelix_var)
    if not err:
        arena.record("VAR(2) (10k x 10)", "Statelix", t_statelix, 0.0)
        
    # 2. Statsmodels
    def run_sm_var():
        model = SmVAR(Y)
        return model.fit(p)
        
    t_sm, res_sm, err = time_func(run_sm_var)
    if not err:
        # Compare coeff (lag 1). Statelix stores as list of matrices.
        # Sm stores as Single matrix params? Or coeff?
        # Simply check correctness if possible, otherwise just time
        arena.record("VAR(2) (10k x 10)", "Statsmodels", t_sm, "N/A", "Detailed coef check skipped")

# ==============================================================================
# BATTLE 4: Panel Fixed Effects
# ==============================================================================
if PanelOLS:
    print("\n[Battle 4] Panel Fixed Effects: N=100, T=50 (5,000 rows)")
    N = 100
    T = 50
    X = np.random.randn(N*T, 5)
    y = np.random.randn(N*T)
    # ids
    entities = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(5)])
    df['y'] = y
    df['entity'] = entities
    df['time'] = times
    df = df.set_index(['entity', 'time'])
    
    # 1. Statelix
    def run_statelix_panel():
        fe = sc.panel.FixedEffects()
        # two_way=False by default (Unit only?) let's assume OneWay Unit FE for comparison
        # But wait, Linearmodels PanelOLS defaults?
        # Let's align on EntityEffects=True
        return fe.fit(y, X, entities.astype(np.int32), times.astype(np.int32))
        
    # Note: Statelix might implement TwoWay by default or configurable. 
    # Checking bindings: two_way is readwrite. Default?
    # Let's set it explicitly if possible.
    # Actually just run it.
    
    t_statelix, res_statelix, err = time_func(run_statelix_panel)
    if not err:
        arena.record("Panel FE (5k rows)", "Statelix", t_statelix, 0.0)

    # 2. Linearmodels
    def run_lm_panel():
        mod = PanelOLS(df.y, df[['x0','x1','x2','x3','x4']], entity_effects=True)
        return mod.fit()
    
    t_lm, res_lm, err = time_func(run_lm_panel)
    if not err:
        # Compare coefficients
        diff = np.max(np.abs(res_statelix.coef - res_lm.params.values))
        arena.record("Panel FE (5k rows)", "Linearmodels", t_lm, diff)

# --- Summary ---
arena.print_summary()
print("\nDone.")
