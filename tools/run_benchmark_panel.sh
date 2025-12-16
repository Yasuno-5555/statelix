#!/bin/sh
# Run Panel benchmark inside Docker

echo "Installing benchmark dependencies..."
pip install scikit-learn pandas numpy tabulate >/dev/null

echo "Running benchmark..."
# Run as module
export PYTHONPATH=$PYTHONPATH:.
python -m benchmarks.benchmark_panel
