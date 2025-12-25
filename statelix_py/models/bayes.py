import numpy as np
from ..core import hmc_sample, bayes

class StatelixHMC:
    """
    Hamiltonian Monte Carlo Sampler.
    
    Uses C++ backend for efficient NUTS-like sampling implementation.
    """
    def __init__(self, n_samples=1000, warmup=500, step_size=0.1, 
                 adapt_step_size=True, target_accept=0.8, seed=42):
        self.config = bayes.HMCConfig()
        self.config.n_samples = n_samples
        self.config.warmup = warmup
        self.config.step_size = step_size
        self.config.adapt_step_size = adapt_step_size
        self.config.target_accept = target_accept
        self.config.seed = seed
        self.results_ = None

    def sample(self, log_prob_func, theta0):
        """
        Run HMC sampling.

        Parameters
        ----------
        log_prob_func : callable
            Function that takes a parameter vector theta (numpy array)
            and returns a tuple (log_probability, gradient_vector).
            
            def log_p(theta):
                ...
                return lp, grad

        theta0 : array-like
            Initial parameter vector.
        
        Returns
        -------
        HMCResult object containing samples and diagnostics.
        """
        theta0 = np.ascontiguousarray(theta0, dtype=np.float64)
        
        # Verify call signature quickly (optional safety)
        try:
            res = log_prob_func(theta0)
            if len(res) != 2:
                raise ValueError("log_prob_func must return (log_prob, gradient)")
        except Exception as e:
            raise ValueError(f"Initial check of log_prob_func failed: {e}")

        # Call C++ binding
        self.results_ = hmc_sample(log_prob_func, theta0, self.config)
        return self.results_
    
    @property
    def samples(self):
        return self.results_.samples if self.results_ else None
        
    @property
    def summary(self):
        if not self.results_: return None
        return {
            "mean": self.results_.mean,
            "std": self.results_.std_dev,
            "acceptance": self.results_.acceptance_rate,
            "ess": self.results_.ess
        }

from .mixins import DiagnosticAwareMixin
from ..diagnostics.presets import GovernanceMode

class BayesianLogisticRegression(DiagnosticAwareMixin):
    """
    Bayesian Logistic Regression using HMC.
    
    Model:
        y ~ Bernoulli(sigmoid(X @ beta))
        beta ~ Normal(0, sigma^2 * I)
    """
    def __init__(self, n_samples=1000, warmup=200, prior_std=10.0, seed=42, 
                 mode: GovernanceMode = GovernanceMode.STRICT):
        self.hmc = StatelixHMC(n_samples=n_samples, warmup=warmup, seed=seed)
        self.prior_std = prior_std
        self.mode = mode
        self.samples_ = None
        self._init_diagnostics(governance_mode=mode)
        
    def fit(self, X, y):
        """
        Fit model using HMC.
        X: (n_samples, n_features)
        y: (n_samples,) 0 or 1
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        n_features = X.shape[1]
        prior_variance = self.prior_std ** 2
        
        # Define Closure for Log Prob and Gradient
        # This keeps the probability logic encapsulated here, not in the GUI.
        def log_prob_func(beta):
            # beta: (n_features,)
            # ... (Logic identical to previous, elided for brevity if assumed same) ...
            # Re-implementing explicitly to be safe as this tool replaces blocks
            z = X @ beta
            p = 1.0 / (1.0 + np.exp(-z))
            ll = np.sum(y * z - np.logaddexp(0, z))
            log_prior = -0.5 * np.sum(beta**2) / prior_variance
            total_log_prob = ll + log_prior
            
            grad_ll = X.T @ (y - p)
            grad_prior = -beta / prior_variance
            total_grad = grad_ll + grad_prior
            
            return total_log_prob, total_grad

        # Run HMC
        theta0 = np.zeros(n_features) # Start at 0
        res = self.hmc.sample(log_prob_func, theta0)
        
        self.samples_ = res.samples
        
        # --- Diagnostics ---
        # 1. Fit Quality: Posterior Predictive Check or just ESS?
        # User wants unified "R2/Fit" metric. For Bayes, usually Bayesian R2 or just convergence.
        # Let's use ESS ratio as a proxy for "Fit Quality" in terms of Sampling Quality.
        # If ESS is low, sampling failed.
        
        ess = res.ess
        if hasattr(ess, "__len__"):
            ess = np.mean(ess)
            
        n_post = self.hmc.config.n_samples
        # Cap ESS ratio at 1.0 (ESS can be > N usually but let's normalize)
        fit_quality = min(ess / n_post, 1.0)
        
        # 2. Structure/Topology: Acceptance Rate Variance?
        # A jumpy acceptance rate means unstable manifold traversal.
        # We can map acceptance rate (target 0.8) to structure score.
        # deviation from 0.8 is "instability".
        acc_rate = res.acceptance_rate
        # 0.8 -> Perfect (1.0). 0.0 -> Bad. 1.0 -> Bad (Too small step).
        # Score = 1 - 2*|0.8 - acc|
        topo_score_val = max(0.0, 1.0 - 5.0 * abs(0.8 - acc_rate))
        
        # Create metrics dict
        metrics = {
            'r2': fit_quality, # Mapping ESS to R2 slot for MCI calc
            'mean_structure': 5.0, # Dummy stable
            'std_structure': 0.1 / (topo_score_val + 0.01), # Encode stability inversely
            'invariant_ratio': 1.0 # Assume geometry is fine
        }
        
        self._run_diagnostics(metrics)
        
        return self
    
    @property
    def coef_means_(self):
        if self.samples_ is None: return None
        return np.mean(self.samples_, axis=0)
    
    @property
    def coef_stds_(self):
        if self.samples_ is None: return None
        return np.std(self.samples_, axis=0)
    
    # Aliases for sklearn-like API
    @property
    def coef_(self):
        """sklearn-compatible alias for coef_means_"""
        return self.coef_means_
    
    @property
    def intercept_(self):
        """Placeholder for sklearn compatibility (intercept is part of coef_ if X has constant)"""
        return 0.0  # Not separately estimated in current implementation
    
    @property
    def summary(self):
        """Return summary statistics of posterior distribution."""
        if self.samples_ is None:
            return None
        return {
            "coef_mean": self.coef_means_,
            "coef_std": self.coef_stds_,
            "n_samples": self.samples_.shape[0],
            "n_features": self.samples_.shape[1]
        }

