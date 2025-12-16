# Statelix Benchmark Results

## 1. Causal Inference: Propensity Score Matching (PSM)
**Algorithm:** Logistic Regression (PS) + Nearest Neighbor Matching (ATT).
**Comparison:** Statelix (C++ Core) vs Scikit-Learn (LogReg + NearestNeighbors).

| N | Method | Time (s) | Speedup |
|---|---|---|---|
| N | Method | Time (s) | Speedup |
|---|---|---|---|
| 1000 | Statelix | 0.0028 | 1.3x |
| 1000 | Sklearn | 0.0036 | 1.0x |
| 5000 | Statelix | 0.0030 | 1.4x |
| 10000 | Statelix | 0.0146 | 0.5x |

**Note:** Optimized `IRLS` logic (using Rank Update) improved performance by **~670x** (9.8s -> 0.015s for N=10k). Statelix is now competitive with Scikit-learn, beating it for N<10k.
**Action Item:** Completed.


## 2. Dynamic Panel: GMM (Arellano-Bond)
**Algorithm:** Difference GMM (1-step).
**Comparison:** Statelix (C++ Core) vs Python (Naive Numpy Implementation).

| N | T | Method | Time (s) | Speedup |
|---|---|---|---|---|
| 1000 | 5 | Statelix | 0.0019 | 3.5x |
| 20000 | 10 | Statelix | 0.2252 | 1.8x |

**Conclusion:** Statelix provides sub-second performance for large panel datasets (N=20k), outperforming python logic.

## 3. Bayesian Inference: Hamiltonian Monte Carlo (HMC)
**Algorithm:** HMC with No-U-Turn Sampler (NUTS-like adaptation) and Dual Averaging.
**Validation:** Gaussian Target (Recovering Mean 0).

| Dim | Samples | Time (s) | Acceptance Rate |
|---|---|---|---|
| 2 | 2000 | 0.12s | 90% |
| 10 | 2000 | 0.14s | 82% |
| 50 | 2000 | 0.23s | 84% |

**Conclusion:** HMC backend is extremely efficient, generating 2000 samples for 50-dimensional distributions in under 0.25 seconds.

