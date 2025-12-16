from setuptools import setup, Extension, find_packages
import sys
import os
import pybind11

# Helper to find headers
src_dir = os.path.abspath('src')
vendor_dir = os.path.abspath('vendor/eigen')
pybind_dir = pybind11.get_include()

# Helper for platform specific flags
cxx_args = ['-std=c++17', '-O2', '-D_USE_MATH_DEFINES']
if sys.platform != 'win32':
    cxx_args += ['-fPIC']

# Define extensions
ext_modules = [
    # 1. PSM (Causal)
    Extension(
        'statelix.psm',
        sources=['src/bindings/python_bindings_psm.cpp'],
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
    # 3. HMC (Bayes)
    Extension(
        'statelix.hmc',
        sources=['src/bindings/python_bindings_hmc.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 4. Statelix Core (Legacy/Everything else for now)
    # We might want to keep the main bindings too, or phase it out?
    # User said "monolithic.so を諦める" (Give up on monolithic .so)
    # So we should probably NOT build the main python_bindings.cpp if we are splitting.
    # But python_bindings.cpp contains OLS, TimeSeries, etc. 
    # For now, let's include it as 'statelix.core' but it fails to compile often.
    # The user specifically said "3 modules are ready". 
    # I will stick to these 3 for the "Release" of *working* stuff.
]

setup(
    name='statelix',
    version='0.1.0',
    description='High-performance C++ Stat/Econ/ML library with Python bindings',
    # Packages
    packages=['statelix'],
    package_dir={'statelix': 'statelix_pkg'},
    ext_modules=ext_modules,
    install_requires=['numpy>=1.21', 'pandas>=1.3', 'scikit-learn>=1.0'],
    zip_safe=False,
)
