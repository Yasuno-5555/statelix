import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Loading Data...")
try:
    df = pd.read_csv('Data/SSDSE-A-2025.csv', encoding='cp932')
except:
    df = pd.read_csv('Data/SSDSE-A-2025.csv', encoding='utf-8')

print(f"Loaded {len(df)} rows.")

# SSDSE-A usually has Row 0 as English Codes (Header), Row 1 as Japanese Labels.
# The 'read_csv' took Row 0 as Header.
# So df.iloc[0] is the Japanese Labels. We should drop it.
if not str(df.iloc[0, 3]).isdigit():
    print("Dropping Japanese label row...")
    df = df.iloc[1:].copy()

# Select interesting columns (assuming standard SSDSE codes)
# A1101: Total Population
# A1301: Area (Land Area) - Wait, A1301 might not be Area in 2025 version.
# Let's just take column indices 3 and 4 assuming they are numeric metrics.
# Col 0: Code, Col 1: Prefecture, Col 2: City
target_col = df.columns[3] # A1101 usually
feature_col = df.columns[4] # A1102 usually (Male Pop?)

print(f"Target: {target_col}")
print(f"Feature: {feature_col}")

# Convert to numeric
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce')
df = df.dropna(subset=[target_col, feature_col])

X = df[[feature_col]].values.astype(np.float64)
y = df[target_col].values.astype(np.float64)

print(f"Data Shape for OLS: {X.shape}")

print("\n--- 1. OLS via Statelix C++ ---")
from statelix.linear_model import FitOLS
ols = FitOLS()
ols.fit(X, y)
print(f"R2: {ols.r_squared:.4f}")
print(f"Coef: {ols.coef_}")
print(f"Intercept: {ols.intercept_}")

print("\n--- 2. Bayesian Logistic via Statelix HMC ---")
# Create binary target: High Population (Above Median)
median_pop = np.median(y)
y_bin = (y > median_pop).astype(np.float64)

from statelix_py.models import BayesianLogisticRegression
bayes = BayesianLogisticRegression(n_samples=500, warmup=100)
bayes.fit(X, y_bin)

print("HMC Sampling Completed.")
if bayes.coef_means_ is not None:
    print(f"Post Mean Beta: {bayes.coef_means_}")
else:
    print("HMC failed or yielded no samples.")

print("\n--- 3. HNSW Search ---")
# Search for similar municipalities based on Feature
from statelix.time_series import search
# Use high-level wrapper to verify it too
from statelix_py.models import StatelixHNSW

hnsw = StatelixHNSW(n_neighbors=5, M=16)
hnsw.fit(X)
print(f"Index built.")

# Query the first item
q = X[0].reshape(1, -1)
dists, labels = hnsw.kneighbors(q)
print(f"Neighbors of first item: {labels}")

print("\nDone.")
