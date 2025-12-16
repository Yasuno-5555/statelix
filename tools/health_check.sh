#!/bin/bash

# Ensure g++ is present (in case of fresh container)
apt-get update >/dev/null && apt-get install -y g++ >/dev/null

INCLUDES="-I/statelix/src -I/statelix/vendor/eigen"
FLAGS="-O0 -fsyntax-only -std=c++17" 

echo "=== Header Health Check ==="
find src -name "*.h" | while read -r file; do
    # Try to compile header (force language to c++-header)
    g++ -x c++-header $FLAGS $INCLUDES "$file" -o /dev/null 2>/tmp/err
    if [ $? -ne 0 ]; then
        echo "[FAIL] $file"
        # Print first few lines of error to help diagnostics
        head -n 3 /tmp/err
    else
        echo "[OK] $file"
    fi
done

echo ""
echo "=== Source Health Check ==="
find src -name "*.cpp" | while read -r file; do
    # Skip python_bindings.cpp as it needs pybind11 and is huge
    if [[ "$file" == *"python_bindings.cpp"* ]]; then continue; fi
    
    g++ $FLAGS $INCLUDES -c "$file" -o /dev/null 2>/tmp/err
    if [ $? -ne 0 ]; then
        echo "[FAIL] $file"
        head -n 3 /tmp/err
    else
        echo "[OK] $file"
    fi
done
