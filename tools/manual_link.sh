#!/bin/sh
# Manual Compile & Link Script to bypass CMake issues

# Install compiler
apt-get update >/dev/null
apt-get install -y g++ >/dev/null

# Get Python includes
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Compiling and Linking manually..."

g++ -O3 -shared -std=c++17 -fPIC \
    -I/statelix/src \
    -I/statelix/vendor/eigen \
    -I$PYBIND_INC \
    -I$PYTHON_INC \
    src/bindings/python_bindings.cpp \
    src/linear_model/ols.cpp \
    src/cluster/kmeans.cpp \
    src/stats/anova.cpp \
    src/time_series/ar_model.cpp \
    src/glm/glm_models.cpp \
    src/survival/cox.cpp \
    src/linear_model/elastic_net.cpp \
    src/time_series/dtw.cpp \
    src/search/kdtree.cpp \
    src/time_series/cpd.cpp \
    src/ml/gbdt.cpp \
    src/ml/fm.cpp \
    src/linalg/sparse_core.cpp \
    src/optimization/objective.cpp \
    -o statelix_core$SUFFIX

if [ $? -eq 0 ]; then
    echo "SUCCESS: Shared Object built!"
    ls -lh statelix_core$SUFFIX
else
    echo "FAILED: Manual link error"
    exit 1
fi
