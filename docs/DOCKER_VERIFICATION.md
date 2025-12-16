# Docker Verification Guide

To verify the **Statelix** build and benchmarks (PSM, GMM) in a clean Linux environment, follow these steps.

## Prerequisites
- Docker Desktop installed and running.
- **Important**: Ensure File Sharing is enabled for the project drive (C:). 
    - Settings -> Resources -> File Sharing.

## Running Verification

We have provided a PowerShell script to automate the process:

```powershell
./tools/verify_docker.ps1
```

### What it does:
1. Builds a Docker image (`statelix-verify`) containing `cmake`, `gcc`, `OpenMP`.
2. Mounts the current directory to `/app`.
3. Runs the clean build and benchmark scripts inside the container.

### Manual Method
If the script fails, you can run manually:

```bash
docker build -t statelix-verify .
docker run --rm -it -v "%cd%":/app statelix-verify bash
# Inside container:
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cp build/*.so statelix_py/core/
python benchmark_psm.py
python benchmark_gmm.py
```
