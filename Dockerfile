FROM python:3.10-slim-bullseye

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Python Dependencies
RUN pip install --no-cache-dir numpy pandas pybind11 scipy scikit-learn

WORKDIR /statelix

# 3. Copy Source Code (Replaces git clone for local dev)
# This captures all local changes without needing a remote repo
COPY . /statelix

# 4. Build C++ Core
RUN rm -rf build && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --config Release --parallel 4

# 5. Install Extension (Manual placement for dev)
RUN find build -name "*.so" -exec cp {} statelix_py/core/ \;

# 6. Run Benchmarks (Run at container start)
CMD ["python3", "run_benchmarks.py"]
