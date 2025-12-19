
import numpy as np
import pandas as pd
from statelix.causal import IV2SLS, DiffInDiff, RDD
from statelix.inquiry.narrative import Storyteller

def verify_causal_inquiry():
    print("--- Verifying Causal Inference Integration ---")
    
    # 1. Generate IV Data (Simulated)
    print("\n--- IV2SLS ---")
    np.random.seed(314)
    n = 200
    U = np.random.normal(0, 1, n)
    W = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)
    X = 0.8 * Z + 0.5 * W + 0.5 * U + np.random.normal(0, 0.5, n)
    Y = 2.0 * X - 1.0 * W + 1.0 * U + np.random.normal(0, 0.5, n)
    
    iv_model = IV2SLS(fit_intercept=True)
    iv_model.fit(Y, Endog=X, Instruments=Z, Exog=W)
    
    print(iv_model.summary())
    if abs(iv_model.effect_ - 2.0) < 0.2:
        print("[PASS] IV estimate is close to true effect (2.0).")
    else:
        print("[WARN] IV estimate deviation.")
        
    # Storyteller (IV)
    story_iv = Storyteller(iv_model, feature_names=["Treatment(X)", "Control(W)", "Intercept"])
    print(story_iv.explain())
    
    if "causal impact" in story_iv.explain(): print("[PASS] Found 'causal impact'.")

    # 2. Difference in Differences (DiD)
    print("\n--- DiffInDiff ---")
    # Y = 1.0 + 2.0*Group + 1.5*Time + 5.0*(Group*Time) + noise
    # ATT = 5.0
    Group = np.random.randint(0, 2, n)
    Time = np.random.randint(0, 2, n)
    DiD_Interaction = Group * Time
    
    Y_did = 1.0 + 2.0 * Group + 1.5 * Time + 5.0 * DiD_Interaction + np.random.normal(0, 0.5, n)
    
    did_model = DiffInDiff()
    did_model.fit(Y_did, Group=Group, Time=Time)
    
    print(did_model.summary())
    if abs(did_model.effect_ - 5.0) < 0.2:
        print("[PASS] DiD estimate close to 5.0.")
    else:
        print(f"[WARN] DiD estimate deviation: {did_model.effect_}")
        
    story_did = Storyteller(did_model, feature_names=["ATT(Interaction)", "Group", "Time", "Intercept"])
    print(story_did.explain())
    if "Parallel Trends" in story_did.explain(): print("[PASS] Found Parallel Trends assumption.")

    # 3. RDD
    print("\n--- RDD ---")
    # Y = 10 + 2*RunVar + 3*Treat + noise
    # 3.0 is jump at cutoff 0
    RunVar = np.random.uniform(-2, 2, 500)
    Cutoff = 0.0
    Treat = (RunVar >= Cutoff).astype(float)
    Y_rdd = 10 + 2 * RunVar + 3.0 * Treat + np.random.normal(0, 0.5, 500)
    
    rdd_model = RDD(cutoff=Cutoff, bandwidth=1.0, kernel='rectangular')
    rdd_model.fit(Y_rdd, RunVar=RunVar)
    
    print(rdd_model.summary())
    if abs(rdd_model.effect_ - 3.0) < 0.2:
        print("[PASS] RDD estimate close to 3.0.")
    
    story_rdd = Storyteller(rdd_model, feature_names=["Jump(Trend)", "RunVar", "RunVar*Treat", "Intercept"])
    print(story_rdd.explain())
    if "Bandwidth" in story_rdd.explain(): print("[PASS] Found Bandwidth info in narrative.")

if __name__ == "__main__":
    verify_causal_inquiry()
