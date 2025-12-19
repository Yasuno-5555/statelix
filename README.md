
# Statelix: The Explanatory Intelligence

> **"Statelix is not a predictor. It is an explainer."**

**Statelix** is a next-generation platform designed to go beyond simple accuracy metrics. It unifies **Econometrics**, **Causal Inference**, and **Bayesian Statistics** into a single environment dedicated to answering the question: **"Why?"**

---

## ⚡ Core Philosophy

In a world obsessed with precise but opaque predictions (black-box ML), Statelix champions **Interpretability** and **Causality**.

1.  **Explanation > Prediction**: We value effect sizes ($\beta$, Causal Impact) over raw accuracy ($R^2$).
2.  **Inquiry-First**: Every model is an interrogation subject. We ask it: "What if?", "Why this?", "Compared to what?".
3.  **Honest Narrative**: Our specific `Storyteller` engine translates complex coefficients into human-readable text, always attaching critical caveats.

---

## 🏆 Performance Benchmarks

Statelix is built on a high-performance C++ core (Eigen backend), making it significantly faster than both pure Python (sklearn/statsmodels) and R.

**Benchmark vs R (Dec 2025):**

| Method | Statelix (C++) | R (Package) | Speedup |
| :--- | :--- | :--- | :--- |
| **OLS Regression** (N=500k) | **0.64s** | 0.82s (`lm`) | **1.3x** |
| **Panel Fixed Effects** (N=50k, T=10) | **0.19s** | 1.91s (`plm`) | **10.2x** |
| **GMM / Dynamic Panel** | **<0.01s** | 3.52s (`pgmm`) | **~400x*** |

*\*Note: GMM speedup reflects C++ optimized linear algebra vs R's iterative solver overhead.*

*Tested on AMD Ryzen/Vega Environment.*

---

## 🔍 Key Features

### 1. **Complete Statistical Suite** (R/Stata Replacement)
Statelix now provides a full range of statistical tools comparable to commercial software:
-   **Hypothesis Tests**: T-test, Chi-Squared, Mann-Whitney U, Wilcoxon, Kruskal-Wallis.
-   **Discrete Choice**: Ordered Logit/Probit, Multinomial Logit.
-   **Mixed Models**: Linear Mixed Effects (LMM) with random intercepts/slopes.
-   **Survival Analysis**: Kaplan-Meier Estimator, Log-Rank Test.
-   **Structural Equation Modeling (SEM)**: Path Analysis, Mediation Analysis.

### 2. **Inquiry Engine** (`statelix.inquiry`)
The brain of Statelix. It treats models as objects of study.
-   **Auto-Narrative**: Automatically generates text like *"The analysis reveals that 80% of the effect is mediated via M"* (SEM) or *"Higher prices increase the likelihood of the 'High Quality' category"* (Ordered Logit).
-   **WhatIf**: Simulate counterfactual scenarios ("What if GDP rose by 2%?") without leaving the framework.

### 3. **Causal Inference** (`statelix.causal`)
Rigorous tools for determining cause and effect.
-   **PSM**: Propensity Score Matching (Optimization-based).
-   **DiffInDiff (DiD)**: Estimation of intervention effects over time.
-   **IV2SLS**: Instrumental Variables for endogeneity.

---

## ✅ Full Stack Verification

We maintain a rigorous benchmark suite (`benchmark/run_suite.py`) confirming the accuracy of all modules:

| Component | Status | Verified Capabilities |
|-----------|--------|------------------------|
| **Core Stats** | ✅ PASS | T-Tests, ANOVA, Non-Parametric Tests |
| **Econometrics** | ✅ PASS | OLS, WLS, Regression Diagnostics (VIF, Durbin-Watson) |
| **Discrete** | ✅ PASS | Ordered Logit, Multinomial Logit |
| **Causal** | ✅ PASS | PSM, Diff-in-Diff, Mediation Analysis |
| **Panel** | ✅ PASS | Dynamic Panel (GMM), Fixed/Random Effects |
| **Time Series** | ✅ PASS | GARCH, State Space, ARMA |
| **Inquiry** | ✅ PASS | Narrative Generation, Counterfactual Simulations |

---

## 🚀 Quick Start: The "Why" Workflow

### Example: Mediation Analysis (SEM)
```python
from statelix_py.models.sem import MediationAnalysis
from statelix_pkg.inquiry.narrative import Storyteller

# 1. Fit Mediation Model (X -> M -> Y)
med = MediationAnalysis(treatment='Education', mediator='Skill', outcome='Wage')
med.fit(data)

# 2. Ask "Why?" (Narrative Generation)
story = Storyteller(med)
print(story.explain())

# Output:
# "Analysis Narrative:
#  The analysis reveals that 65.2% of the total effect of Education on Wage is mediated (indirect).
#  - Indirect Effect (Mechanism): 0.85 (Significant). This represents the pathway through Skill.
#  - Direct Effect: 0.45. The effect remaining after accounting for Skill."
```

## 🛠 Installation

```bash
pip install .
```

---

## License
MIT License.
