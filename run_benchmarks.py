
import subprocess
import sys
import os

def run_script(name):
    print(f"\nExample: Running {name}...")
    try:
        # Assumes scripts are in root or passed path is correct relative to CWD
        result = subprocess.run([sys.executable, name], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED: {name}")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Error running {name}: {e}")
        return False

def main():
    print("--- Statelix Self-Contained Benchmark Suite ---")
    
    # Ensure current dir is in python path
    os.environ['PYTHONPATH'] = os.getcwd()
    
    scripts = [
        "benchmark_psm.py",
        "benchmark_gmm.py"
    ]
    
    failed = []
    for s in scripts:
        if not os.path.exists(s):
            print(f"Skipping {s} (Not found)")
            continue
            
        if not run_script(s):
            failed.append(s)
            
    if failed:
        print("\nSome benchmarks FAILED:")
        for f in failed:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("\nAll benchmarks PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    main()
