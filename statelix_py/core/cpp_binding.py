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

def detect_change_points(data, penalty=0.0):
    _check_core()
    model = _core.ChangePointDetector()
    if penalty > 0:
        model.penalty = penalty
    return model.fit_pelt(np.asarray(data, dtype=float))

# --- Search ---
class KDTree:
    def __init__(self, data):
        _check_core()
        self.model = _core.KDTree()
        self.model.fit(np.asarray(data, dtype=float))
    
    def query(self, point, k=1):
        return self.model.query(np.asarray(point, dtype=float), k)

# --- State Space ---
class KalmanFilter:
    def __init__(self, state_dim, measure_dim):
        _check_core()
        self.model = _core.KalmanFilter(state_dim, measure_dim)
        
    def filter(self, measurements):
        # We need to expose matrix setters for F, H, Q, R etc. 
        # For now, we assume user sets them via property access which pybind11 exposes directly on the inner model.
        # But this wrapper is shielding the inner model.
        # Let's Expose the inner model directly or proxy attributes.
        pass
        # Better: Return the inner model directly in a factory? 
        # Or Just use the inner model directly from statelix_core in user code.
        # But here we are unifying.
        
    # Simplified wrapper access to inner C++ object attributes is tricky without __getattr__
    def get_model(self):
        return self.model

# --- Spatial ---
def align_icp(source, target, max_iter=50):
    _check_core()
    model = _core.ICP()
    model.max_iter = max_iter
    return model.align(np.asarray(source, dtype=float), np.asarray(target, dtype=float))

# --- Signal ---
def perform_wavelet_transform(signal, level=0):
    _check_core()
    model = _core.WaveletTransform()
    # model.type = ... (default Haar)
    return model.transform(np.asarray(signal, dtype=float), level)

def inverse_wavelet_transform(coeffs, n_original):
    _check_core()
    model = _core.WaveletTransform()
    return model.inverse(np.asarray(coeffs, dtype=float), n_original)

# --- ML ---
def fit_gbdt(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0):
    _check_core()
    model = _core.GradientBoostingRegressor()
    model.n_estimators = n_estimators
    model.learning_rate = learning_rate
    model.max_depth = max_depth
    model.subsample = subsample
    model.max_depth = max_depth
    model.subsample = subsample
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model # Return model to allow predict

def fit_fm(X, y, n_factors=8, max_iter=100, reg_w=0.0, reg_v=0.0, task="Regression"):
    _check_core()
    model = _core.FactorizationMachine()
    model.n_factors = n_factors
    model.max_iter = max_iter
    model.reg_w = reg_w
    model.reg_v = reg_v
    
    if task == "Classification":
        model.task = _core.FMTask.Classification
    else:
        model.task = _core.FMTask.Regression
        
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model

# --- Optimization ---
class PyObjective(_core.DifferentiableFunction):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        # We need to return (val, grad)
        # x is a numpy array (copy usually unless configured)
        # This function matches the C++ trampoline expectation
        return self.func(x)

def minimize_lbfgs(func, x0, max_iter=100, m=10, epsilon=1e-5):
    """
    Minimize function using L-BFGS.
    func: Callable(x) -> (value, gradient_vector)
    x0: Initial guess (numpy array)
    """
    _check_core()
    objective = PyObjective(func)
    solver = _core.LBFGS()
    solver.max_iter = max_iter
    solver.m = m
    solver.epsilon = epsilon
    
    # Run C++ optimization which calls back into Python
    result = solver.minimize(objective, np.asarray(x0, dtype=float))
    
    # Run C++ optimization which calls back into Python
    result = solver.minimize(objective, np.asarray(x0, dtype=float))
    
    return result

# --- Bayes: MCMC ---
class PyLogProb(_core.LogProbFunction):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, x):
        # x is numpy array
        return float(self.func(x))

def run_mcmc(log_prob_func, x0, n_samples=1000, burn_in=100, step_size=0.5):
    """
    Run Metropolis-Hastings MCMC.
    log_prob_func: Callable(x) -> log_probability (float)
    x0: Initial state (numpy array)
    """
    _check_core()
    log_prob = PyLogProb(log_prob_func)
    sampler = _core.MetropolisHastings()
    sampler.n_samples = n_samples
    sampler.burn_in = burn_in
    sampler.step_size = step_size
    
    result = sampler.sample(log_prob, np.asarray(x0, dtype=float))
    return result

# --- Survival ---
def fit_cox_ph(X, time, status):
    _check_core()
    model = _core.CoxPH()
    return model.fit(np.asarray(X), np.asarray(time), np.asarray(status, dtype=np.int32))
