
"""
Statelix Full Suite Benchmark Runner.
Executes all benchmarks and verification scripts in the benchmark directory.
"""
import sys
import os
import time
import subprocess
import glob

def run_script(path):
    print(f"\n[RUNNING] {os.path.basename(path)}...")
    start = time.time()
    
    # Run in subprocess to isolate environments/imports
    # Ensure CWD is project root to handle imports correctly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')
    
    # Use the same python interpreter
    cmd = [sys.executable, path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        duration = time.time() - start
        
        if result.returncode == 0:
            print(f"[PASS] {os.path.basename(path)} ({duration:.2f}s)")
            return True, duration, result.stdout
        else:
            print(f"[FAIL] {os.path.basename(path)} ({duration:.2f}s)")
            print("--- STDERR ---")
            print(result.stderr)
            print("--------------")
            return False, duration, result.stderr
            
    except Exception as e:
        print(f"[ERROR] Failed to execute {path}: {e}")
        return False, 0.0, str(e)

def main():
    print("==================================================")
    print("      STATELIX FULL SUITE BENCHMARK RUNNER")
    print("==================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collect all python scripts in benchmark dir except this runner
    scripts = glob.glob(os.path.join(base_dir, "*.py"))
    scripts = [s for s in scripts if os.path.basename(s) != "run_suite.py" and not os.path.basename(s).startswith("__")]
    
    # Sort for consistency
    scripts.sort()
    
    results = {}
    total_start = time.time()
    
    for s in scripts:
        name = os.path.basename(s)
        success, duration, output = run_script(s)
        results[name] = (success, duration)
        
    total_time = time.time() - total_start
    
    print("\n==================================================")
    print(f"      SUMMARY REPORT (Total: {total_time:.2f}s)")
    print("==================================================")
    print(f"{'Script':<30} | {'Status':<6} | {'Time':<8}")
    print("-" * 50)
    
    failures = 0
    for name, (success, duration) in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{name:<30} | {status:<6} | {duration:>6.2f}s")
        if not success:
            failures += 1
            
    if failures == 0:
        print("\nALL TESTS PASSED.")
        sys.exit(0)
    else:
        print(f"\n{failures} TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
