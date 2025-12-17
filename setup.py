from setuptools import setup, Extension, find_packages
import sys
import os
import pybind11

# Helper to find headers
src_dir = os.path.abspath('src')
vendor_dir = os.path.abspath('vendor/eigen')
pybind_dir = pybind11.get_include()

# Helper for platform specific flags
# Helper for platform specific flags
# default args
cxx_args = ['-D_USE_MATH_DEFINES']

if sys.platform == 'win32':
    # MSVC specific flags
    cxx_args += ['/std:c++17', '/O2', '/bigobj', '/EHsc']
else:
    # GCC/Clang specific flags
    cxx_args += ['-std=c++17', '-O2', '-fPIC']

# Define extensions
ext_modules = [
    # 1. Causal Inference (IV, PSM, DiD, RDD)
    Extension(
        'statelix.causal',
        sources=['src/bindings/python_bindings_causal.cpp', 'src/linear_model/logistic.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 2. Panel (Econometrics)
    Extension(
        'statelix.panel',
        sources=['src/bindings/python_bindings_panel.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 3. Bayes (HMC + VI + Models)
    Extension(
        'statelix.bayes',
        sources=['src/bindings/python_bindings_bayes.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 4. Time Series
    Extension(
        'statelix.time_series',
        sources=['src/bindings/python_bindings_timeseries.cpp', 'src/time_series/cpd.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 5. Linear Models (OLS/GLM) - Disabled due to build issues
    # 5. Linear Models (OLS/GLM) - Restored for Phase 9
    Extension(
       'statelix.linear_model',
       sources=['src/bindings/python_bindings_linear.cpp', 'src/linear_model/ols.cpp'],
       include_dirs=[src_dir, vendor_dir, pybind_dir],
       extra_compile_args=cxx_args,
       language='c++'
    ),
]

setup(
    name='statelix',
    version='0.1.0',
    description='High-performance C++ Stat/Econ/ML library with Python bindings',
    # Packages
    packages=['statelix', 'statelix.inquiry', 'statelix.causal'], # Removed bayes (folder missing)
    package_dir={'statelix': 'statelix_pkg'},
    ext_modules=ext_modules,
    install_requires=['numpy>=1.21', 'pandas>=1.3', 'scikit-learn>=1.0'],
    zip_safe=False,
)
