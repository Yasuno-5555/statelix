#!/bin/sh
# Compile minimal HMC module

# 1. Install Dependencies
apt-get update >/dev/null
command -v g++ >/dev/null || apt-get install -y g++ >/dev/null
pip install pybind11 numpy >/dev/null

# 2. Includes
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 3. Compile
echo "Compiling statelix_hmc..."

g++ -O3 -shared -std=c++17 -fPIC \
    -D_USE_MATH_DEFINES \
    -I/statelix/src \
    -I/statelix/vendor/eigen \
    -I$PYBIND_INC \
    -I$PYTHON_INC \
    src/bindings/python_bindings_hmc.cpp \
    -o statelix_hmc$SUFFIX

if [ $? -eq 0 ]; then
    echo "SUCCESS: statelix_hmc built!"
    mkdir -p benchmarks
    mv statelix_hmc$SUFFIX benchmarks/
else
    echo "FAILED: Compilation error"
    exit 1
fi
