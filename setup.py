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
    # 6. Spatial Econometrics
    Extension(
        'statelix.spatial',
        sources=['src/bindings/python_bindings_spatial.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
    # 7. Graph Analysis
    Extension(
        'statelix.graph',
        sources=['src/bindings/python_bindings_graph.cpp'],
        include_dirs=[src_dir, vendor_dir, pybind_dir],
        extra_compile_args=cxx_args,
        language='c++'
    ),
]

from setuptools.command.build_ext import build_ext
import subprocess

# ... (Previous code) ...
# Define extensions
# ... (Previous extensions) ...

# ------------------------------------------------------------------------------
# CUDA Build Logic
# ------------------------------------------------------------------------------

def find_nvcc():
    try:
        # Suppress output
        subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT)
        print("NVCC DETECTED")
        return True
    except (OSError, subprocess.CalledProcessError):
        print("NVCC NOT DETECTED")
        return False

HAS_NVCC = find_nvcc()

# Custom builder to handle .cu files
class statelix_build_ext(build_ext):
    def build_extensions(self):
        # Build accelerator if present
        for ext in self.extensions:
            if ext.name == 'statelix.accelerator':
                 self.build_cuda_extension(ext)
        
        super().build_extensions()

    def build_cuda_extension(self, ext):
        cu_sources = [f for f in ext.sources if f.endswith('.cu')]
        if not cu_sources: return
        
        # Remove .cu from sources so super() doesn't fail
        ext.sources = [f for f in ext.sources if not f.endswith('.cu')]
        
        # Compile .cu files individually
        for source in cu_sources:
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            obj_name = os.path.basename(source).replace('.cu', '.obj')
            obj = os.path.join(self.build_temp, obj_name)
            
            print(f"Building CUDA source (via subprocess): {source} -> {obj}")
            
            # Create command
            cmd = ['nvcc', '-c', source, '-o', obj]
            for inc in ext.include_dirs:
                cmd.extend(['-I', inc])
            
            # Flags: /MD is critical for MSVC Python compatibility
            cmd.extend(['-O2', '-Xcompiler', '/MD']) 
            
            try:
                # Run and capture output
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                     print(f"NVCC STDOUT: {res.stdout}")
                     print(f"NVCC STDERR: {res.stderr}")
                     raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
                ext.extra_objects.append(obj)
            except Exception as e:
                print(f"Failed to compile CUDA source {source}: {e}")
                raise

        # Add CUDA runtime lib
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            lib_dir = os.path.join(cuda_path, 'lib', 'x64')
            ext.library_dirs.append(lib_dir)
            ext.libraries.append('cudart')
        else:
             print("WARNING: CUDA_PATH not set. Linker might fail to find cudart.lib.")
             ext.libraries.append('cudart')

# Register extension ONLY if NVCC is present
if HAS_NVCC:
    accelerator_ext = Extension(
        'statelix.accelerator',
        sources=['src/bindings/python_bindings_accelerator.cpp', 'src/cuda/accelerator.cu'],
        include_dirs=[src_dir, vendor_dir, pybind11.get_include()],
        extra_compile_args=cxx_args, 
        language='c++'
    )
    ext_modules.append(accelerator_ext)

setup(
    name='statelix',
    version='0.2.0',
    description='High-performance C++ Stat/Econ/ML library with Python bindings (CUDA Optional)',
    packages=['statelix', 'statelix.models', 'statelix.core', 'statelix.stats', 'statelix.gui', 'statelix.utils', 'statelix.plugins'],
    package_dir={'statelix': 'statelix_py'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': statelix_build_ext},
    install_requires=['numpy>=1.21', 'pandas>=1.3', 'scikit-learn>=1.0'],
    zip_safe=False,
)
