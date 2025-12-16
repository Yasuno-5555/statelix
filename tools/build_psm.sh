#!/bin/sh
# Compile minimal PSM module for benchmarking

# 1. Install Dependencies FIRST
echo "Installing dependencies..."
# Update apt
apt-get update >/dev/null

# Install g++ if missing
if ! command -v g++ > /dev/null; then
    apt-get install -y g++ >/dev/null
fi

# Install python libs
pip install pybind11 numpy >/dev/null

# 2. Get Python Includes (AFTER install)
echo "Configuring includes..."
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "PYTHON_INC: $PYTHON_INC"
echo "PYBIND_INC: $PYBIND_INC"

# 3. Compile
echo "Compiling statelix_psm..."

g++ -O3 -shared -std=c++17 -fPIC \
    -D_USE_MATH_DEFINES \
    -I/statelix/src \
    -I/statelix/vendor/eigen \
    -I$PYBIND_INC \
    -I$PYTHON_INC \
    src/bindings/python_bindings_psm.cpp \
    -o statelix_psm$SUFFIX

if [ $? -eq 0 ]; then
    echo "SUCCESS: statelix_psm built!"
    # Ensure benchmarks dir exists
    mkdir -p benchmarks
    mv statelix_psm$SUFFIX benchmarks/
    ls -lh benchmarks/statelix_psm$SUFFIX
else
    echo "FAILED: Compilation error"
    exit 1
fi
