import sys
import os

log_file = 'build_log.txt'
if not os.path.exists(log_file):
    print("Log file not found")
    sys.exit(1)

content = ""
try:
    with open(log_file, 'r', encoding='utf-16') as f:
        content = f.read()
except:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

lines = content.splitlines()
for i, line in enumerate(lines):
    if 'CMake Error' in line or 'error:' in line.lower():
        print(f"Line {i}: {line}")
        # Print context
        for j in range(max(0, i-2), min(len(lines), i+5)):
            if j != i:
                print(f"  {lines[j]}")
        print("-" * 40)
        if i > 5000: break # Safety
