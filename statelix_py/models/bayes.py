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

class BayesianLogisticRegression:
    """
    Bayesian Logistic Regression using HMC.
    
    Model:
        y ~ Bernoulli(sigmoid(X @ beta))
        beta ~ Normal(0, sigma^2 * I)
    """
    def __init__(self, n_samples=1000, warmup=200, prior_std=10.0, seed=42):
        self.hmc = StatelixHMC(n_samples=n_samples, warmup=warmup, seed=seed)
        self.prior_std = prior_std
        self.samples_ = None
        
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
            
            # 1. Linear predictor: z = X @ beta
            z = X @ beta
            
            # 2. Log-Likelihood
            # ll = sum( y*z - log(1 + exp(z)) )
            # Stable computation of log(1+exp(z)) is usually softplus(z)
            # but for gradient simply:
            # p = sigmoid(z)
            # grad_ll = X.T @ (y - p)
            
            # Using numpy for vectorization
            # sigmoid(z)
            p = 1.0 / (1.0 + np.exp(-z))
            
            # LL
            # Avoid numerical issues with log(p) or log(1-p)
            # LL = y*log(p) + (1-y)*log(1-p)
            #    = y*z - log(1+exp(z))
            ll = np.sum(y * z - np.logaddexp(0, z))
            
            # Prior: beta ~ N(0, prior_std^2)
            # log_prior = -0.5 * sum(beta^2) / var
            log_prior = -0.5 * np.sum(beta**2) / prior_variance
            
            total_log_prob = ll + log_prior
            
            # 3. Gradient
            # d(LL)/dbeta = X.T @ (y - p)
            grad_ll = X.T @ (y - p)
            grad_prior = -beta / prior_variance
            
            total_grad = grad_ll + grad_prior
            
            return total_log_prob, total_grad

        # Run HMC
        theta0 = np.zeros(n_features) # Start at 0
        res = self.hmc.sample(log_prob_func, theta0)
        
        self.samples_ = res.samples
        return self
    
    @property
    def coef_means_(self):
        if self.samples_ is None: return None
        return np.mean(self.samples_, axis=0)
    
    @property
    def coef_stds_(self):
        if self.samples_ is None: return None
        return np.std(self.samples_, axis=0)
