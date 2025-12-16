#!/bin/sh
# Install build dependencies
apt-get update >/dev/null && apt-get install -y g++ >/dev/null
pip install pybind11 >/dev/null 2>&1
INCLUDES=$(python3 -m pybind11 --includes)
g++ -O3 -Wall -shared -std=c++14 -fPIC $INCLUDES -I/statelix/src -I/statelix/vendor/eigen -c src/bindings/python_bindings.cpp -o bindings.o
