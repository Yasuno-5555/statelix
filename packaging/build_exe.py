import os
import sys
import subprocess
import shutil
from pathlib import Path

def build():
    # Base Project Path
    project_dir = Path(__file__).resolve().parent.parent
    dist_dir = project_dir / "dist"
    build_dir = project_dir / "build_exe"
    
    # Entry Point
    entry_point = project_dir / "statelix_py" / "app.py"
    
    if not entry_point.exists():
        print(f"Error: Entry point not found at {entry_point}")
        sys.exit(1)

    # PyInstaller Arguments
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name=Statelix",
        "--windowed", # GUI mode (no console)
        # Paths
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        f"--specpath={project_dir / 'packaging'}",
        
        # Imports to force (Scientific libs often use dynamic loading)
        "--hidden-import=statelix_py.models",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.linear_model",
        "--hidden-import=sklearn.neighbors",
        "--hidden-import=sklearn.utils._typedefs", # Common pyinstaller fail
        "--hidden-import=scipy.sparse.csgraph",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=pandas",
        
        # Collect Data (Icons if any)
        # "--add-data=resources;resources", 
        
        # Main Script
        str(entry_point)
    ]
    
    # Check for C++ Extension
    # If the user ran 'python setup.py build_ext --inplace', the .pyd should be in statelix_py/core
    # We rely on PyInstaller's auto-analysis to find 'statelix_py.core.cpp_binding' imports.
    
    print("Staritng PyInstaller Build...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n" + "="*50)
        print(f"Build Success! Executable is at:\n{dist_dir / 'Statelix' / 'Statelix.exe'}")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"Build Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
