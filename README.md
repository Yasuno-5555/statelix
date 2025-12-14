# Statelix v2.2

Statelix: High-Performance Statistical Analysis Software

## 概要

Statelixは、C++の高速計算コアとPythonの柔軟なインターフェースを統合した次世代の統計解析ソフトウェアです。
v2.2では、グラフ解析、因果推論、ベイズ統計、および近似最近傍探索（HNSW）機能が大幅に強化されました。

### 主な特徴

- ⚡ **高速計算コア (C++17 + Eigen)**: 大規模行列演算、スパースグラフ処理を高速化
- 📊 **多機能 GUI**: 研究者向けの直感的なパラメータ調整、インタラクティブな可視化
- 📈 **高度な統計モデル**:
    - **線形/一般化線形モデル**: OLS, Ridge, Logistic, Poisson, GLM
    - **因果推論**: 操作変数法 (IV/2SLS), 差分の差分法 (DID)
    - **グラフ解析**: Louvain Community Detection, PageRank
    - **ベイズ統計**: Hamiltonian Monte Carlo (HMC/NUTS)
    - **探索**: HNSW (Hierarchical Navigable Small World) Index
- 🐍 **完全な Python API**: Scikit-Learn 互換のインターフェース

## インストール

### 必要要件
- Windows / Linux / macOS
- Python 3.8+
- C++ コンパイラ (MSVC 2017+, GCC 9+, Clang 10+)
- CMake 3.18+

### ビルドとインストール

本バージョンより `setup.py` に CMake ビルドプロセスが統合されました。

```bash
# クローン
git clone https://github.com/statelix/statelix.git
cd statelix

# 仮想環境 (推奨)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# インストール (C++拡張モジュールのビルド含む)
pip install .
```

開発モード（編集を即座に反映）:
```bash
pip install -e .
```

## 使い方 (GUI)

```bash
# アプリケーション起動
python -m statelix_py.app
```

### v2.2 新機能の操作
1. **データロード**: CSV等をドラッグ＆ドロップ。
2. **モデル選択**:
    - **Graph**: ノード間の関係性分析。「Source Node」「Target Node」列を選択。
    - **Causal**: 因果効果の推定。「Outcome」「Treatment」「Instrument/Post」列を選択。
    - **Bayesian**: HMCを用いたロジスティック回帰。「Samples」「Warmup」を指定可能。
3. **可視化**: "プロット (Viz)" タブで HMC のトレースプロットや残差プロットを確認。

## 使い方 (Python SDK)

Scikit-Learn ライクな API で高度なモデルを利用可能です。

### 1. 近似最近傍探索 (HNSW)
```python
import numpy as np
from statelix_py.models import StatelixHNSW

# データ準備 (float64)
X = np.random.randn(10000, 128)

# インデックス構築
model = StatelixHNSW(M=16, ef_construction=200)
model.fit(X)

# 検索 (Top-5)
indices = model.transform(X[:5])
print(indices)
```

### 2. ベイズ統計 (HMC Sampler)
```python
from statelix_py.models import StatelixHMC

# 対数確率と勾配を定義 (例: 1D ガウス分布)
def log_prob(x):
    # log_p = -0.5 * x^2, grad = -x
    return -0.5 * x[0]**2, [-x[0]]

# サンプリング実行
hmc = StatelixHMC(n_samples=1000, warmup=200)
result = hmc.sample(log_prob, theta0=[0.0])

print(result.summary)
```

### 3. 線形回帰 (OLS)
```python
from statelix_py.models import StatelixOLS

model = StatelixOLS()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## プロジェクト構造

```
statelix/
├── src/                    # C++ Core
│   ├── bindings/          # Python Pybind11 Bindings
│   ├── graph/             # Louvain, PageRank
│   ├── causal/            # IV, DID
│   ├── bayes/             # HMC Sampler
│   ├── search/            # HNSW Index
│   └── ...
├── statelix_py/           # Python Package
│   ├── core/              # C++ Extension Wrappers
│   ├── models/            # Sklearn-compatible Models
│   └── gui/               # PySide6 Application
├── tests/                 # Unit Tests
└── setup.py               # Build Script
```

## 開発者向け

### テスト実行
```bash
# ユニットテスト (Python)
pytest tests/

# 手動検証スクリプト
python tests/verify_manual.py
```

## ライセンス
MIT License
