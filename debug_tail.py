import sys
import os

log_file = 'build_log_2.txt'

try:
    with open(log_file, 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

lines = content.splitlines()
errors = []

for i, line in enumerate(lines):
    # Capture CMake Error or MSVC error Cxxxx
    if 'CMake Error' in line or 'error C' in line or 'Error:' in line or 'failed' in line.lower():
        errors.append((i, line))

print(f"Found {len(errors)} error lines")
# Print last 20
for idx, line in errors[-20:]:
    print(f"{idx}: {line.strip()}")
    # Print context
    for j in range(max(0, idx-1), min(len(lines), idx+3)):
         if j != idx:
             print(f"  {lines[j].strip()}")
    print("-" * 20)
