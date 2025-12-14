# Statelix 開発用 Dockerfile
FROM ubuntu:22.04

# 必要パッケージ
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip python3-venv \
    libeigen3-dev libqt6* pkg-config curl wget \
    && rm -rf /var/lib/apt/lists/*

# Python環境
WORKDIR /statelix
RUN python3 -m venv venv
ENV PATH="/statelix/venv/bin:$PATH"

# Python依存ライブラリ
RUN pip install --upgrade pip
RUN pip install pybind11 pandas pyarrow numpy matplotlib pyqt6

# Wasmランタイム
RUN pip install wasmtime

# デフォルト作業ディレクトリ
WORKDIR /statelix
CMD ["/bin/bash"]
