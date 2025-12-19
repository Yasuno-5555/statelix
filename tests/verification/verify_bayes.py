
import numpy as np
import statelix.bayes
from scipy import stats

def verify_bayes_linear():
    print("\n--- Bayesian Linear Regression Verification ---")
    np.random.seed(42)
    
    # 1. Generate Data
    n_samples = 200
    n_features = 3
    X = np.random.normal(0, 1, (n_samples, n_features))
    true_beta = np.array([1.5, -2.0, 0.5])
    true_sigma = 1.2
    
    noise = np.random.normal(0, true_sigma, n_samples)
    y = X @ true_beta + noise
    
    # 2. OLS Baseline
    beta_ols, res, rank, s = np.linalg.lstsq(X, y, rcond=None)
    sigma_ols = np.sqrt(np.sum((y - X @ beta_ols)**2) / (n_samples - n_features))
    
    print(f"True Beta:  {true_beta}")
    print(f"OLS Beta:   {beta_ols}")
    print(f"True Sigma: {true_sigma:.4f}, OLS Sigma: {sigma_ols:.4f}")
    
    model = statelix.bayes.BayesianLinearRegression(X, y)
    
    # 3. MAP Estimation
    print("\n[MAP Estimation]")
    model.fit()
    map_theta = model.map_theta
    map_beta = map_theta[:n_features]
    map_log_sigma = map_theta[n_features]
    map_sigma = np.exp(map_log_sigma)
    
    print(f"MAP Beta:   {map_beta}")
    print(f"MAP Sigma:  {map_sigma:.4f}")
    
    if np.allclose(map_beta, beta_ols, atol=0.2): # Weak prior allows some drift
        print("[PASS] MAP Beta close to OLS")
    else:
        print("[WARN] MAP Beta divergence (Check prior strength?)")
        
    # 4. HMC Sampling
    print("\n[HMC Sampling]")
    hmc_res = model.sample(n_samples=1000, warmup=500)
    
    # Summary
    post_mean = hmc_res.mean
    post_beta = post_mean[:n_features]
    post_sigma = np.exp(post_mean[n_features]) # Approx (E[exp(x)] != exp(E[x]), but close for small var)
    
    print(f"HMC Beta Mean:  {post_beta}")
    print(f"HMC Sigma Approx: {post_sigma:.4f}")
    print(f"Acceptance Rate: {hmc_res.acceptance_rate:.2f}")
    print(f"Divergences: {hmc_res.n_divergences}")
    
    if hmc_res.acceptance_rate > 0.5:
        print("[PASS] HMC Acceptance Rate healthy")
    else:
        print("[WARN] HMC Acceptance Rate low")
        
    if np.allclose(post_beta, true_beta, atol=0.2):
        print("[PASS] HMC recovers True parameters")
        
    # 5. Variational Inference
    print("\n[Variational Inference]")
    vi_res = model.fit_vi(max_iter=2000)
    vi_beta = vi_res.mean[:n_features]
    vi_sigma = np.exp(vi_res.mean[n_features])
    
    print(f"VI Beta Mean:   {vi_beta}")
    print(f"VI Sigma Approx: {vi_sigma:.4f}")
    print(f"VI ELBO: {vi_res.elbo:.4f}")
    
    if np.allclose(vi_beta, true_beta, atol=0.3):
         print("[PASS] VI recovers True parameters (approx)")

if __name__ == "__main__":
    verify_bayes_linear()
