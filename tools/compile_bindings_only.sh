#!/bin/sh
apt-get update >/dev/null
apt-get install -y g++ python3-dev >/dev/null

# Install pybind11
pip3 install pybind11 >/dev/null 2>&1

# Get pybind11 include path
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "Compiling python_bindings.cpp..."
g++ -std=c++17 -fPIC -I/statelix/src -I/statelix/vendor/eigen -I/statelix/vendor/pybind11/include -I$PYBIND_INC -I$PYTHON_INC -c /statelix/src/bindings/python_bindings_clean.cpp -o bindings.o 2>&1

if [ $? -eq 0 ]; then
    echo "SUCCESS: python_bindings.cpp compiled!"
else
    echo "FAILED: Compilation errors above"
fi
