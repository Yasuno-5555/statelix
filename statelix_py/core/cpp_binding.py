import numpy as np
import sys
import os

# Try to import the compiled C++ module
# It is expected to be in the parent directory or installed in site-packages
try:
    from .. import statelix_core as _core
except ImportError:
    try:
        import statelix_core as _core
    except ImportError:
        _core = None
        print("Warning: statelix_core C++ module not found. Some features will be unavailable.")

def _check_core():
    if _core is None:
        raise ImportError("statelix_core module is not loaded. Please build the C++ extension.")

# --- OLS ---
def fit_ols(X, y, fit_intercept=True, conf_level=0.95):
    """
    Fit Ordinary Least Squares model.
    """
    _check_core()
    return _core.fit_ols_full(np.asarray(X), np.asarray(y), fit_intercept, conf_level)

def predict_ols(result, X_new, fit_intercept=True):
    """
    Predict using fitted OLS model.
    """
    _check_core()
    return _core.predict_ols(result, np.asarray(X_new), fit_intercept)

def predict_with_interval(result, X_new, fit_intercept=True, conf_level=0.95):
    _check_core()
    return _core.predict_with_interval(result, np.asarray(X_new), fit_intercept, conf_level)

# --- KMeans ---
def fit_kmeans(X, k, max_iter=300, tol=1e-4, random_state=42):
    """
    Perform K-Means clustering.
    params:
        X: data matrix (n_samples, n_features)
        k: number of clusters
    """
    _check_core()
    return _core.fit_kmeans(np.asarray(X), k, max_iter, tol, random_state)

# --- ANOVA ---
def f_oneway(data, groups):
    """
    One-way ANOVA.
    data: Flat array of values
    groups: Array of group labels (integers)
    """
    _check_core()
    return _core.f_oneway(np.asarray(data), np.asarray(groups, dtype=np.int32))

# --- Time Series ---
def fit_ar(series, p):
    """
    Fit Autoregressive AR(p) model.
    """
    _check_core()
    return _core.fit_ar(np.asarray(series), p)


# --- GLM Models ---
def fit_logistic(X, y, max_iter=50):
    _check_core()
    model = _core.LogisticRegression()
    model.max_iter = max_iter
    return model.fit(np.asarray(X), np.asarray(y))

def fit_poisson(X, y, max_iter=50):
    _check_core()
    model = _core.PoissonRegression()
    model.max_iter = max_iter
    return model.fit(np.asarray(X), np.asarray(y))

def fit_negbin(X, y):
    _check_core()
    model = _core.NegBinRegression()
    return model.fit(np.asarray(X), np.asarray(y))

def fit_gamma(X, y):
    _check_core()
    model = _core.GammaRegression()
    return model.fit(np.asarray(X), np.asarray(y))

def fit_probit(X, y):
    _check_core()
    model = _core.ProbitRegression()
    return model.fit(np.asarray(X), np.asarray(y))

# --- Regularized ---
def fit_ridge(X, y, alpha=1.0):
    _check_core()
    model = _core.RidgeRegression()
    model.alpha = alpha
    return model.fit(np.asarray(X), np.asarray(y))

def fit_elastic_net(X, y, alpha=1.0, l1_ratio=0.5):
    _check_core()
    model = _core.ElasticNet()
    model.alpha = alpha
    model.l1_ratio = l1_ratio
    return model.fit(np.asarray(X), np.asarray(y))

# --- Time Series ---
def compute_dtw(s1, s2):
    _check_core()
    model = _core.DTW()
    return model.compute(np.asarray(s1), np.asarray(s2))

# --- Search ---
class KDTree:
    def __init__(self, data):
        _check_core()
        self.model = _core.KDTree()
        self.model.fit(np.asarray(data, dtype=float))
    
    def query(self, point, k=1):
        return self.model.query(np.asarray(point, dtype=float), k)

# --- Survival ---
def fit_cox_ph(X, time, status):
    _check_core()
    model = _core.CoxPH()
    return model.fit(np.asarray(X), np.asarray(time), np.asarray(status, dtype=np.int32))
