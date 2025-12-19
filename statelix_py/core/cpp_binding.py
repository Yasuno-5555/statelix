import numpy as np
try:
    from statelix.linear_model import FitOLS
except ImportError:
    pass

# --- OLS Adapter for Legacy GUI ---
def fit_ols_full(X, y, fit_intercept=True, conf_level=0.95):
    """
    Fit OLS (Full) using the new FitOLS class.
    Returns the FitOLS object which now has properties like .r_squared, .coef, etc.
    """
    model = FitOLS()
    # Note: fit takes (X, y, fit_intercept, conf_level, gram)
    model.fit(
        np.asarray(X, dtype=np.float64), 
        np.asarray(y, dtype=np.float64),
        fit_intercept,
        conf_level
    )
    return model

def predict_ols(result, X_new, fit_intercept=True):
    # result is expected to be the model object (FitOLS) in new paradigm
    # or the OLSResult result member?
    # GUI logic: Not strictly defined. If GUI passes the result object back here?
    # Usually GUI calls result.predict or something.
    # But checking cpp_binding usage in main_window.py:
    # It doesn't use predict_ols.
    # But `cpp_binding.py` has it.
    if hasattr(result, 'predict'):
        return result.predict(np.asarray(X_new, dtype=np.float64), fit_intercept)
    return None

def predict_with_interval(result, X_new, fit_intercept=True, conf_level=0.95):
    # Not implemented in FitOLS directly?
    # FitOLS has raw OLSResult inside.
    # C++ has `predict_with_interval(OLSResult, ...)`
    # But it is NOT exposed in binding as standalone function.
    # And FitOLS class didn't expose it.
    # So this feature is currently broken in new Binding.
    raise NotImplementedError("Prediction Interval not yet ported to new binding.")

# --- Stubbing other legacy functions to prevent import errors ---
# If these are used, they will crash at runtime, but at least import is safe.
def fit_kmeans(*args, **kwargs): raise NotImplementedError("KMeans moved to statelix.cluster")
def f_oneway(*args, **kwargs): raise NotImplementedError("ANOVA moved to statelix.stats")
def fit_ar(*args, **kwargs): raise NotImplementedError("AR moved to statelix.time_series")

