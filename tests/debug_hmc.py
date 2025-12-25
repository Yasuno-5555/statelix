import numpy as np
from statelix_py.models import StatelixHMC

def check_hmc():
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    Precision = np.linalg.inv(Sigma)
    
    def log_prob(theta):
        lp = -0.5 * theta @ Precision @ theta
        grad = -Precision @ theta
        return lp, grad

    hmc = StatelixHMC(n_samples=2000, warmup=1000, step_size=0.05, 
                      target_accept=0.8, seed=42)
    
    theta0 = np.array([0.0, 0.0])
    res = hmc.sample(log_prob, theta0)
    
    print(f"Acceptance Rate: {res.acceptance_rate}")
    print(f"ESS: {res.ess}")
    print(f"Mean: {res.mean}")
    print(f"Std Dev: {res.std_dev}")

if __name__ == "__main__":
    check_hmc()
