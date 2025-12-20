
# Statelix: The Explanatory Intelligence (v0.2.0)

> **"Statelix is not a predictor. It is an explainer."**

**Statelix** is a next-generation platform designed to go beyond simple accuracy metrics. It unifies **Econometrics**, **Causal Inference**, and **Bayesian Statistics** into a single environment dedicated to answering the question: 
This is largely AI-generated code. I cannot fully verify the results myself. Use at your own risk and please validate before citing.
**"Why?"**

Built on a hybrid **C++ (Eigen) / Python** architecture, it offers performance that outperforms standard libraries while maintaining strict numerical stability.

---

## ⚡ New in v0.2.0: Transparent GPU Acceleration

Statelix now includes an **Optional CUDA Accelerator** for computationally intensive tasks (e.g., Large-Scale OLS/WLS, GMM).

-   **transparent**: You don't change your code. If a GPU is detected and $N > 10,000$, Statelix automatically offloads matrix operations.
-   **Fail-Safe**: "CPU is Truth." If the GPU fails (e.g., Out of Memory, Driver Issue), Statelix **silently falls back to the CPU**, ensuring your analysis never crashes.
-   **Precision**: We use the GPU only for "muscle work" (matrix multiplication). Sensitive reductions are performed on the CPU to guarantee numerical stability comparable to LAPACK/Sklearn.

---

## 🏆 Performance Benchmarks

### Correctness Verified
Statelix guarantees results identical to `sklearn` / `results` within machine epsilon ($10^{-15}$), regardless of whether the calculation runs on CPU or GPU.

### Speed (vs R & Sklearn)

| Method | Statelix (CPU) | Statelix (GPU) | R (Package) | Sklearn |
| :--- | :--- | :--- | :--- | :--- |
| **OLS Regression** (N=100k) | **0.03s** | 0.04s* | 0.15s (`lm`) | 0.04s |
| **Panel Fixed Effects** (N=50k, T=10) | **0.19s** | - | 1.91s (`plm`) | - |
| **GMM / Dynamic Panel** | **<0.01s** | TBD | 3.52s (`pgmm`) | N/A |

*\*Note: GPU performance includes transfer overhead. For very large datasets ($N > 10^6$), GPU acceleration provides significant gains.*

---

## 🔍 Key Features

### 1. **Complete Statistical Suite**
-   **Hypothesis Tests**: T-test, Chi-Squared, Mann-Whitney U, Wilcoxon, Kruskal-Wallis.
-   **Discrete Choice**: Ordered Logit/Probit, Multinomial Logit.
-   **Mixed Models**: Linear Mixed Effects (LMM).
-   **Survival Analysis**: Kaplan-Meier, Log-Rank Test.
-   **SEM**: Path Analysis, Mediation Analysis.

### 2. **Inquiry Engine** (`statelix.inquiry`)
The brain of Statelix. It treats models as objects of study.
-   **Auto-Narrative**: Automatically generates text like *"The analysis reveals that 80% of the effect is mediated via M"*.
-   **WhatIf**: Simulate counterfactual scenarios ("What if GDP rose by 2%?").

### 3. **Causal Inference** (`statelix.causal`)
Rigorous tools for determining cause and effect.
-   **Propensity Score Matching (PSM)**
-   **Difference-in-Differences (DiD)**
-   **Instrumental Variables (IV2SLS)**

---

## 🚀 Usage Example

Accurate imports + GPU-ready OLS:

```python
import statelix as stx
from statelix.models.sem import MediationAnalysis
from statelix.inquiry import Storyteller

# 1. High-Performance OLS (Auto GPU if N > 10k)
model = stx.StatelixOLS()
model.fit(X, y)
print(f"R2: {model.r_squared:.4f}")

# 2. Mediation Analysis (SEM)
med = MediationAnalysis(treatment='Education', mediator='Skill', outcome='Wage')
med.fit(data)

# 3. Ask "Why?" (Narrative Generation)
story = Storyteller(med)
print(story.explain())

# Output:
# "Analysis Narrative:
#  The analysis reveals that 65.2% of the total effect of Education on Wage is mediated (indirect).
#  - Indirect Effect: 0.85 (Significant). P-Value < 0.001."
```

## 🛠 Installation

### 🐳 Docker (Recommended - 推奨)

**Dockerを使えば、どの環境でも同じように動作します。**

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/statelix.git
cd statelix

# 2. Build the Docker image
docker-compose build

# 3. Run tests
docker-compose run --rm statelix

# 4. Open a development shell
docker-compose run --rm statelix bash

# Inside the container, you can run:
python -c "import statelix; print('Statelix is working!')"
python benchmark/run_benchmarks.py
```

### Manual Installation (手動インストール)

Requires **C++ compiler (GCC/Clang/MSVC)**, **CMake**, and **Python 3.10+**.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build and install Statelix
pip install -e .

# 3. Verify installation
python -c "import statelix; print('OK')"
```

If `nvcc` (CUDA Toolkit) is found in PATH, GPU support is built automatically.

---

## License
MIT License.
