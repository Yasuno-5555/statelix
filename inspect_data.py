import pandas as pd
import sys

# Auto-detect encoding? Try CP932 (Shift-JIS extension)
try:
    df = pd.read_csv('Data/SSDSE-A-2025.csv', encoding='cp932')
except UnicodeDecodeError:
    df = pd.read_csv('Data/SSDSE-A-2025.csv', encoding='utf-8')

print("=== Columns ===")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print("\n=== Data Head (2 rows) ===")
print(df.head(2).to_string())
