#!/bin/sh
# Run PSM benchmark inside Docker

echo "Installing benchmark dependencies..."
pip install scikit-learn pandas numpy tabulate >/dev/null

echo "Running benchmark..."
# Run as module to allow relative imports
export PYTHONPATH=$PYTHONPATH:.
python -m benchmarks.benchmark_psm
