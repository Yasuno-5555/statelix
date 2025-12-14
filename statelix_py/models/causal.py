import numpy as np
import pandas as pd
from ..core import causal

class StatelixIV:
    """
    Instrumental Variable Estimation (2SLS).
    """
    def __init__(self, fit_intercept=True, robust_se=True):
        self.fit_intercept = fit_intercept
        self.robust_se = robust_se
        self.result_ = None
        
    def fit(self, X_endog, y, Z, X_exog=None):
        """
        Fit 2SLS.
        X_endog: Endogenous variables (cause)
        y: Outcome
        Z: Instruments
        X_exog: Exogenous controls (optional)
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        X_endog = np.ascontiguousarray(X_endog, dtype=np.float64)
        if X_endog.ndim == 1: X_endog = X_endog.reshape(-1, 1)
        
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        if Z.ndim == 1: Z = Z.reshape(-1, 1)
        
        if X_exog is None:
            X_exog = np.empty((len(y), 0), dtype=np.float64)
        else:
            X_exog = np.ascontiguousarray(X_exog, dtype=np.float64)
            
        model = causal.TwoStageLeastSquares()
        model.fit_intercept = self.fit_intercept
        model.robust_se = self.robust_se
        
        self.result_ = model.fit(y, X_endog, X_exog, Z)
        return self

class StatelixDID:
    """
    Difference-in-Differences Estimator.
    """
    def __init__(self, robust_se=True):
        self.robust_se = robust_se
        self.result_ = None
        
    def fit(self, y, treated, post):
        """
        Fit DID.
        y: Outcome
        treated: Binary mask (1=Treatment Group, 0=Control)
        post: Binary mask (1=Post-Treatment Period, 0=Pre)
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        treated = np.ascontiguousarray(treated, dtype=np.int32)
        post = np.ascontiguousarray(post, dtype=np.int32)
        
        model = causal.DifferenceInDifferences()
        model.robust_se = self.robust_se
        
        self.result_ = model.fit(y, treated, post)
        return self

class StatelixPSM:
    """
    Propensity Score Matching (PSM) Estimator.
    
    Architecture:
    - Estimator: ATT (Average Treatment eff. on Treated)
    - Score Model: Logistic Regression (Statelix Core)
    - Matcher: HNSW Index (Statelix Core)
    """
    def __init__(self, caliper=0.2, n_bootstrap=50, seed=42):
        self.caliper = caliper
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        
        self.result_ = None
        self.score_model_ = None
        self.matcher_ = None
        self.matched_df_ = None # For debugging/plotting

    def fit(self, y, T, X):
        """
        Fit PSM and estimate ATT.
        y: Outcome array (Shape: N)
        T: Treatment binary array (Shape: N)
        X: Covariate matrix (Shape: N, M)
        """
        import numpy as np
        import pandas as pd
        from ..core import cpp_binding as core
        from ..core import search
        
        # Ensure types for C++ bindings
        y = np.ascontiguousarray(y, dtype=np.float64)
        T = np.ascontiguousarray(T, dtype=np.float64) # LogReg expects double often
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        # 1. Propensity Score Estimation
        # Use Statelix C++ Logistic Regression
        self.score_model_ = core.LogisticRegression()
        self.score_model_.max_iter = 1000
        self.score_model_.fit(X, T)
        
        # Predict probs (returns vector of P(T=1))
        scores = self.score_model_.predict_prob(X)
        
        # 2. Separate Groups
        treated_mask = (T == 1)
        control_mask = (T == 0)
        
        treated_idx = np.where(treated_mask)[0]
        control_idx = np.where(control_mask)[0]
        
        # 3. Matching via HNSW
        # We match Treated units to Control units based on Score.
        
        # Build Index on CONTROL scores
        control_scores = scores[control_mask].reshape(-1, 1).astype(np.float64)
        control_scores = np.ascontiguousarray(control_scores)
        
        config = search.HNSWConfig()
        config.M = 16
        config.ef_construction = 100
        config.seed = self.seed
        config.distance = search.HNSWConfig.Distance.L2 # |a-b|
        
        self.matcher_ = search.HNSW(config)
        self.matcher_.build(data=control_scores) # keyword arg 'data'
        
        # Query for TREATED units
        treated_scores = scores[treated_mask].reshape(-1, 1).astype(np.float64)
        treated_scores = np.ascontiguousarray(treated_scores)
        
        # Find 1-NN
        match_res = self.matcher_.query_batch(treated_scores, k=1)
        
        # 4. Process Matches & Diagnostics
        matched_indices_local = match_res.indices.flatten() # Index within Control subset
        distances = match_res.distances.flatten() # Score differences
        
        # Recover global indices of matched controls
        matched_control_global_idx = control_idx[matched_indices_local]
        
        # Caliper Filtering
        # Caliper is usually defined in terms of std dev of the logit of propensity score.
        # For simplicity here, we use std dev of raw probability score.
        score_std = np.std(scores)
        threshold = self.caliper * score_std
        
        valid_match_mask = (distances <= threshold)
        n_treated_total = len(treated_idx)
        n_matched = np.sum(valid_match_mask)
        unmatched_ratio = 1.0 - (n_matched / n_treated_total) if n_treated_total > 0 else 0.0
        
        if n_matched == 0:
            # Avoid crash, just warn or return empty
            # raise RuntimeError("No matches found within caliper. loosen caliper.")
            att = np.nan
            se_boot = np.nan
        else:
            # Extract Outcomes for valid matches
            valid_treated_idx = treated_idx[valid_match_mask]
            valid_control_idx = matched_control_global_idx[valid_match_mask]
            
            y_treated_matched = y[valid_treated_idx]
            y_control_matched = y[valid_control_idx]
            
            # 5. Estimate ATT
            diffs = y_treated_matched - y_control_matched
            att = np.mean(diffs)
            
            # 6. Bootstrap SE (Simple: Resampling the differences)
            # This assumes the matching structure is fixed (Conditional SE).
            # Full bootstrap would require re-matching.
            boot_means = []
            rng = np.random.RandomState(self.seed)
            for _ in range(self.n_bootstrap):
                resample = rng.choice(diffs, size=n_matched, replace=True)
                boot_means.append(np.mean(resample))
            
            se_boot = np.std(boot_means)
        
        # 7. Construct Result Schema
        self.result_ = {
            "ATT": att,
            "SE": se_boot,
            "n_treated": int(n_treated_total),
            "n_matched": int(n_matched),
            "unmatched_ratio": unmatched_ratio,
            "score_summary": {
                "treated_mean": float(np.mean(scores[treated_mask])),
                "control_mean": float(np.mean(scores[control_mask])),
                "overlap_std": float(score_std),
                "caliper_threshold": float(threshold)
            }
        }
        
        return self

    @property
    def summary(self):
        return self.result_
