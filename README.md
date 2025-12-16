# Statelix: The Explanatory Intelligence

> **"Statelix is not a predictor. It is an explainer."**

**Statelix** is a next-generation platform designed to go beyond simple accuracy metrics. It unifies **Econometrics**, **Causal Inference**, and **Bayesian Statistics** into a single environment dedicated to answering the question: **"Why?"**

---

## ⚡ Core Philosophy

In a world obsessed with precise but opaque predictions (black-box ML), Statelix champions **Interpretability** and **Causality**.

1.  **Explanation > Prediction**: We value effect sizes ($beta$, Causal Impact) over raw accuracy ($R^2$).
2.  **Inquiry-First**: Every model is an interrogation subject. We ask it: "What if?", "Why this?", "Compared to what?".
3.  **Honest Narrative**: Our specific `Storyteller` engine translates complex coefficients into human-readable text, always attaching critical caveats (Assumptions, Standard Errors).

---

## 🔍 Key Extensions

### 1. **Inquiry Engine** (`statelix.inquiry`)
The brain of Statelix. It treats models as objects of study.
-   **Compare**: Rank models by information criteria (AIC/BIC) and likelihood.
-   **Narrative**: Auto-generate "Financial Times" style reports explaining key drivers.
-   **WhatIf**: Simulate counterfactual scenarios ("What if GDP rose by 2%?") without leaving the framework.

### 2. **Causal Inference** (`statelix.causal`)
Rigorous tools for determining cause and effect.
-   **IV2SLS**: Instrumental Variables (Two-Stage Least Squares) for endogeneity.
-   **DiffInDiff (DiD)**: Estimation of intervention effects over time.
-   **RDD**: Regression Discontinuity Design for threshold-based causal analysis.

### 3. **The Core** (`statelix.linear_model`, `statelix.bayes`)
Solid implementations of classical methods.
-   **OLS/GLM**: Robust linear models.
-   **Bayesian Regression**: Probabilistic reasoning made accessible.

---

## 🚀 Quick Start: The "Why" Workflow

```python
from statelix.causal import DiffInDiff
from statelix.inquiry import Storyteller, WhatIf

# 1. Fit a Causal Model (Difference in Differences)
did = DiffInDiff()
did.fit(Y, Group, Time)

# 2. Get the Narrative (Why did it happen?)
story = Storyteller(did, feature_names=["Effect", "Group", "Time"])
print(story.explain())
# Output:
# "Feature Effect: Has a positive causal impact (effect = 5.2). 
#  > [!WARNING] Validity depends on Parallel Trends assumption."

# 3. Ask "What If?" (Counterfactuals)
wi = WhatIf(did)
scenario = wi.simulate(base_data, {'Group': lambda x: 1}) # Force Treatment
```

## 🛠 Installation

```bash
pip install .
```

*Note: Statelix requires a C++17 compatible environment for building its optimized core extensions. However, the `inquiry` and `causal` modules are partially accessible in pure Python.*

---

## License
MIT License.
