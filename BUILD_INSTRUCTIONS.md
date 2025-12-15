# How to Build Statelix

## Prerequisites
- CMake >= 3.18
- C++17 Compiler (MSVC, GCC, or Clang)
- Python >= 3.8
- Eigen3 (Required for C++ core)

## Building Python Bindings (Recommended)
This will compile the C++ core (`statelix_core`) and install the Python package (`statelix`).

```bash
pip install . -v
```

Or for development (editable mode):
```bash
pip install -e . -v
```

## Running Benchmarks
After installation, you can run the HNSW benchmark to verify performance:

```bash
python tests/bench_hnsw.py
```

## C++ Standalone Build (Optional)
If you want to build the C++ tests or standalone executable:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
