# Getting Started with Statelix v2.3

## Environment Setup Details

### Windows
1. **Visual Studio**: Install Visual Studio 2019 or later with "Desktop development with C++".
2. **CMake**: Install from [cmake.org](https://cmake.org/download/) and add to PATH.
3. **Python**: Python 3.8+ recommended.

### Building
Using `setup.py` allows standard pip installation.

```powershell
pip install .
```

If you encounter errors about `CMake`, ensure `cmake --version` works in your terminal.
Alternatively, use the provided script to build a standalone executable:
```powershell
python packaging/build_exe.py
```

## Advanced Usage

### 1. Graph Analysis (Louvain)
Statelix expects input as an **Edge List**. 
- CSV Format: `SourceID, TargetID`
- IDs can be Strings or Integers. The GUI handles mapping automatically.

Example Data (`graph.csv`):
```csv
src,dst
UserA,UserB
UserB,UserC
UserA,UserC
```

GUI Steps:
1. Select "Graph: Louvain Communities".
2. Set "Source Node" -> `src`.
3. Set "Target Node" -> `dst`.
4. Run. The output table will show the Community ID for each User.

### 2. Causal Inference (IV)
Use this when you have an endogenous variable ($X$) correlated with the error term, and an instrument ($Z$).

Model:
1. $X = \gamma Z + \eta$ (First Stage)
2. $Y = \beta \hat{X} + \epsilon$ (Second Stage)

GUI Steps:
1. Select "Causal: IV (2SLS)".
2. Set "Outcome (Y)" -> Your dependent variable.
3. Set "Endogenous (X)" -> The variable with omitted variable bias.
4. Set "Instrument (Z)" -> The instrument variable.
5. Run. Check the "First Stage F" statistic (>10 is widely considered strong).

### 3. Propensity Score Matching (PSM)
Matches treated units to control units using similar propensity scores.
Statelix uses **HNSW** for matching, allowing it to handle millions of samples in seconds.

GUI Steps:
1. Select "Causal: PSM (Propensity Matching)".
2. Set "Outcome (Y)" -> Your dependent variable.
3. Set "Treatment (0/1)" (in Aux list) -> The binary treatment indicator.
4. Set "Covariates (X)" -> Variables used to estimate propensity.
5. Set "Caliper" -> Max allowed distance (in SD units). Default 0.2.
6. Run.

**Results:**
- **ATT**: Average Treatment Effect on the Treated.
- **Unmatched Ratio**: Portion of treated units discarded due to caliper.
- **Overlap SD**: Standard Deviation of scores (check theoretical violations).

### 4. Bayesian Logistic Regression
Uses Hamiltonian Monte Carlo (HMC) to sample from the posterior distribution.
Unlike standard Logistic Regression which gives a point estimate (MLE), this gives you a *distribution* of likely coefficients.

- **Warmup**: Samples discarded to allow the chain to converge.
- **Samples**: Number of samples to keep for inference.

Use the "Plot (Viz)" tab to check the **Trace Plot**. If the lines look like a "fuzzy caterpillar", the chain has mixed well. If it wanders slowly, increase samples or thin the chain.

### 5. Structural Equation Modeling (Mediation)
Go beyond "Does X affect Y?" to "How does X affect Y?".
Mediation analysis decomposes the Total Effect into Direct and Indirect (via Mediator) effects.

**Model Structure:**
1. $M \sim X$
2. $Y \sim X + M$

**Insight:**
If the Indirect Effect is significant, it implies $X$ works *through* $M$.

**Example:**
- Does **Education (X)** increase **Income (Y)**?
- Or does Education increase **Skills (M)**, which then increase Income?
- Statelix will tell you "50% of the effect is mediated by Skills".
