import os
import sys
import subprocess
import shutil
from pathlib import Path
import ctypes.util

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
        "--hidden-import=PySide6",
        
        # Aggressive collection for heavy libs
        "--collect-all=pandas",
        "--collect-all=numpy",
        "--collect-all=sklearn",
        "--collect-all=scipy",
        "--collect-all=PySide6",
        
        # Main Script
        str(entry_point)
    ]
    
    # --- FIX: Bundling Python DLL explicitly ---
    # PyInstaller sometimes fails to find pythonXX.dll if not in standard locations
    dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    
    # Check default install location
    dll_path = Path(sys.base_prefix) / dll_name
    
    if not dll_path.exists():
        # Fallback: Try searching PATH
        found = ctypes.util.find_library(dll_name[:-4])
        if found:
            dll_path = Path(found)
    
    if dll_path.exists():
        print(f"[INFO] Explicitly bundling Python DLL: {dll_path}")
        # Format: source;dest (Windows)
        cmd.append(f"--add-binary={dll_path};.")
    else:
        print(f"[WARNING] Could not locate {dll_name}. Build might fail at runtime.")

    print("Starting PyInstaller Build...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        
        # --- License Compliance ---
        print("[Compliance] Copying License Files...")
        shutil.copy(project_dir / "LICENSE", dist_dir / "Statelix" / "LICENSE")
        shutil.copy(project_dir / "NOTICE.txt", dist_dir / "Statelix" / "NOTICE.txt")

        print("\n" + "="*50)
        print(f"Build Success! Executable is at:\n{dist_dir / 'Statelix' / 'Statelix.exe'}")
        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"Build Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
