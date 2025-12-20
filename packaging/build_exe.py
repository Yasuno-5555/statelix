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

    # --- FIX: Create temp 'statelix' package alias ---
    # PyInstaller fails to resolve 'statelix.inquiry' because the directory is 'statelix_py'
    # but the code uses 'import statelix...'. We create a temp alias.
    temp_lib = project_dir / "temp_build_lib"
    temp_pkg = temp_lib / "statelix"
    
    # Clean previous temp if exists
    if temp_lib.exists():
        shutil.rmtree(temp_lib)
        
    print(f"[INFO] Creating temp alias: {temp_pkg}")
    shutil.copytree(project_dir / "statelix_py", temp_pkg)
    
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
        f"--paths={temp_lib}", # Add temp lib to search path
        
        # Imports to force (Scientific libs often use dynamic loading)
        "--hidden-import=statelix.models",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.linear_model",
        "--hidden-import=sklearn.neighbors",
        "--hidden-import=sklearn.utils._typedefs", # Common pyinstaller fail
        "--hidden-import=scipy.sparse.csgraph",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=pandas",
        "--hidden-import=PySide6",
        "--hidden-import=statelix.inquiry",
        
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
        
        # Ensure NOTICE.txt exists before copy
        notice_src = project_dir / "NOTICE.txt"
        if not notice_src.exists():
             with open(notice_src, "w") as f:
                 f.write("Statelix\nCopyright (c) 2025 MonadLab")
        shutil.copy(notice_src, dist_dir / "Statelix" / "NOTICE.txt")

        # --- Data Bundling ---
        print("[Compliance] Copying Data Files...")
        data_src = project_dir / "Data"
        data_dst = dist_dir / "Statelix" / "Data"
        if data_src.exists():
            if data_dst.exists():
                shutil.rmtree(data_dst)
            shutil.copytree(data_src, data_dst)
        
        # --- Create Inno Setup Script ---
        print("[Installer] Generating Inno Setup Script...")
        # (We will create this file in a separate step or tool call, 
        # but the logic to RUN it could be here if iscc was found)

        print("\n" + "="*50)
        print(f"Build Success! Executable is at:\n{dist_dir / 'Statelix' / 'Statelix.exe'}")
        
        # Check for Inno Setup Compiler
        iscc_path = shutil.which("iscc")
        if iscc_path:
             print(f"[Installer] Finding Inno Setup Compiler at {iscc_path}")
             # We can trigger it here if setup.iss exists
        else:
             print("[Installer] Inno Setup Compiler (ISCC) not found in PATH. Skipping installer creation.")
             print("You can manually compile the 'setup.iss' file generated in 'packaging'.")

        print("="*50)
    except subprocess.CalledProcessError as e:
        print(f"Build Failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup temp alias safely
        if temp_lib.exists():
            print(f"[INFO] Cleaning up {temp_lib}")
            shutil.rmtree(temp_lib)

if __name__ == "__main__":
    build()
