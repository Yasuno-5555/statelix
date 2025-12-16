#!/bin/sh
# Test pip install setup.py in Docker
# Simulates release build

# 1. Install deps
apt-get update >/dev/null
command -v g++ >/dev/null || apt-get install -y g++ >/dev/null
pip install pybind11 numpy pandas scikit-learn >/dev/null

# 2. Build and Install
echo "Running pip install ."
pip install . -v 2>&1 | tee pip_install_log.txt

# 3. Verify Import
echo "Running verification script..."
python tools/verify_release.py
