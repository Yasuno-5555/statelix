
import subprocess
import sys
import os
import time

def run_script(name):
    print(f"\n--- Running {name} ---")
    start_time = time.time()
    try:
        # Assumes scripts are in root or passed path is correct relative to CWD
        result = subprocess.run([sys.executable, name], capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"FAILED: {name} (Time: {duration:.2f}s)")
            print("ERROR OUTPUT:")
            print(result.stderr)
            print("STANDARD OUTPUT:")
            print(result.stdout)
            return False, duration
        
        print(f"PASSED: {name} (Time: {duration:.2f}s)")
        # Print last few lines of stdout to show summary
        lines = result.stdout.strip().split('\n')
        if lines:
            print("Output Summary:")
            for line in lines[-5:]:
                print(f"  {line}")
        return True, duration
    except Exception as e:
        print(f"Error executing {name}: {e}")
        return False, 0.0

def main():
    print("===========================================")
    print("   STATELIX FULL STACK BENCHMARK SUITE")
    print("===========================================")
    
    # Ensure current dir is in python path
    os.environ['PYTHONPATH'] = os.getcwd()
    
    # Core Benchmarks (Performance & Accuracy)
    benchmarks = [
        "benchmark_psm.py",       # Causal
        "benchmark_gmm.py",       # Dynamic Panel
        "benchmark_spatial.py",   # Spatial (New)
    ]
    
    # Verification Scripts (Feature Coverage)
    verifications = [
        "verify_linear_cpp.py",   # OLS/GLM Bindings
        "verify_panel.py",        # Panel Data Bindings
        "verify_bayes.py",        # HMC/Bayes Bindings
        "verify_timeseries.py",   # Time Series Bindings
        "verify_inquiry.py",      # Inquiry Engine (Narrative)
        "verify_causal_inquiry.py"# Causal Inquiry
    ]
    
    all_scripts = benchmarks + verifications
    results = {}
    
    total_start = time.time()
    
    for s in all_scripts:
        if not os.path.exists(s):
            print(f"\n[SKIP] {s} (File not found)")
            results[s] = ("SKIPPED", 0.0)
            continue
            
        success, duration = run_script(s)
        status = "PASS" if success else "FAIL"
        results[s] = (status, duration)
            
    total_time = time.time() - total_start
    
    print("\n===========================================")
    print(f"   SUMMARY REPORT (Total: {total_time:.2f}s)")
    print("===========================================")
    print(f"{'Script':<25} | {'Status':<6} | {'Time':<8}")
    print("-" * 45)
    
    failures = 0
    for name, (status, duration) in results.items():
        print(f"{name:<25} | {status:<6} | {duration:>6.2f}s")
        if status == "FAIL":
            failures += 1
            
    if failures > 0:
        print(f"\nFAILED: {failures} benchmark(s) did not pass.")
        sys.exit(1)
    else:
        print("\nSUCCESS: All benchmarks passed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
