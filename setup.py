import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Configuration (Debug/Release)
        cfg = 'Debug' if self.debug else 'Release'
        
        # CMake Args
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        # Multi-config helpers (Windows)
        build_args = ['--config', cfg]

        # Platform specific flags
        if sys.platform.startswith("win"):
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2']

        # Ensure build dir exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='statelix',
    version='2.2.0',
    packages=['statelix_py', 'statelix_py.core', 'statelix_py.gui', 'statelix_py.models', 'statelix_py.plugins', 'statelix_py.utils'],
    package_dir={'': '.'},
    ext_modules=[CMakeExtension('statelix_py.core.statelix_core', sourcedir='.')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'pybind11>=2.10.0',
    ]
)
