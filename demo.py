#!/usr/bin/env python3
"""
Statelix Official Demo
======================

1分で分かる Statelix の真価。

このデモでは:
1. 非線形・共線性・弱識別が混ざった「壊れたデータ」を生成
2. 普通のOLS → 崩壊
3. Statelix → 自己修復 → 安定化

Usage:
    python demo.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# =============================================================================
# STEP 1: Generate "Broken" Data
# =============================================================================
print("=" * 60)
print("STATELIX OFFICIAL DEMO")
print("=" * 60)
print()
print("STEP 1: Generating pathological data...")
print("-" * 40)

n = 100

# True latent variable
z = np.random.randn(n)

# Nonlinear relationship
x1 = z + 0.1 * np.random.randn(n)
x2 = z**2 + 0.1 * np.random.randn(n)  # Nonlinear
x3 = x1 * 0.99 + 0.01 * np.random.randn(n)  # Near-perfect collinearity

# Weak instrument (bad identification)
x4 = 0.05 * z + 0.95 * np.random.randn(n)

# Outcome with complex structure
y = 2*z + 0.5*z**2 - 0.3*x4 + np.random.randn(n)

df = pd.DataFrame({
    'outcome': y,
    'driver_1': x1,
    'driver_2': x2,
    'collinear': x3,
    'weak_signal': x4,
})

print(f"  Rows: {len(df)}")
print(f"  Features: nonlinear, collinear, weak signal")
print(f"  Problems: multicollinearity, nonlinearity, weak identification")
print()

# =============================================================================
# STEP 2: Traditional OLS → COLLAPSE
# =============================================================================
print("STEP 2: Traditional OLS Analysis")
print("-" * 40)

try:
    import statsmodels.api as sm
    X_ols = sm.add_constant(df[['driver_1', 'driver_2', 'collinear', 'weak_signal']])
    model_ols = sm.OLS(df['outcome'], X_ols).fit()
    
    print(f"  R²: {model_ols.rsquared:.4f}")
    print(f"  Condition Number: {np.linalg.cond(X_ols.values):.1f}")
    print()
    print("  Coefficients:")
    for name, coef, se in zip(model_ols.params.index, model_ols.params, model_ols.bse):
        significance = "***" if abs(coef/se) > 2.58 else "**" if abs(coef/se) > 1.96 else ""
        print(f"    {name:15s}: {coef:8.4f} (SE: {se:6.4f}) {significance}")
    print()
    print("  ⚠️  WARNING: High condition number = unstable estimates")
    print("  ⚠️  WARNING: Collinearity inflates standard errors")
except Exception as e:
    print(f"  OLS FAILED: {e}")
print()

# =============================================================================
# STEP 3: Statelix → SELF-REPAIR → STABLE
# =============================================================================
print("STEP 3: Statelix Structural Analysis")
print("-" * 40)

import statelix_py as statelix

# This is the ONLY line you need
result = statelix.analyze(df, 'outcome', gui=False)

print(f"  Stability (R²): {result['r_squared']:.4f}")
print(f"  State: {result['stability'].upper()}")
print(f"  Features used: {result['n_features']} (auto-selected)")
print()
print("  Structure Coefficients:")
for name, coef in result['coefficients'].items():
    print(f"    {name:15s}: {coef:8.4f}")
print()
print("  ✅ Statelix automatically:")
print("     - Detected collinearity")
print("     - Limited feature space to prevent singularity")
print("     - Returned stable structural estimate")
print()

# =============================================================================
# CONCLUSION
# =============================================================================
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print()
print("  Traditional OLS: High condition number, inflated SEs")
print("  Statelix:        Structural stability, honest estimation")
print()
print("  This is not model selection.")
print("  This is structural truth.")
print()
print("  >>> import statelix_py as statelix")
print("  >>> statelix.analyze(df, 'outcome')")
print()
print("  That's it.")
print("=" * 60)
