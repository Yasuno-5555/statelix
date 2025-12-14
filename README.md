# Statelix v2.2

高性能統計解析ソフトウェア - 学部生から研究者まで使える、R/Stataを超える体験

## 概要

Statelixは、GUI操作とPythonコードの両方をサポートする統計解析ソフトウェアです。C++による高速計算コアとPythonの柔軟性を組み合わせ、再現性のある研究を支援します。

### 主な特徴

- 🖥️ **直感的なGUI** - 高校生でも使える簡易モード、研究者向け詳細モード
- ⚡ **高速計算** - C++ + Eigenによる最適化された線形代数演算
- 🔌 **拡張可能** - Wasmプラグインによるカスタム分析
- 📊 **豊富なモデル** - OLS, GLM, GLMM, Survival分析
- 🔄 **完全な再現性** - ステップログ + データハッシュによる追跡
- 📈 **インタラクティブ可視化** - Matplotlib/Plotlyによるグラフ

## インストール

### 必要要件

- Python 3.8以上
- CMake 3.18以上（C++コアのビルド用）
- C++17対応コンパイラ（GCC 7+, Clang 5+, MSVC 2017+）

### pipでインストール

```bash
pip install statelix
```

### ソースからビルド

```bash
# リポジトリのクローン
git clone https://github.com/statelix/statelix.git
cd statelix

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -e ".[dev]"

# C++コアのビルド
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cmake --install .
```

## 使い方

### GUIモード

```bash
statelix
```

アプリケーションが起動し、以下の操作が可能です：

1. **データ読み込み** - CSV, Excel, Parquet, Arrowファイルをドラッグ&ドロップ
2. **モデル選択** - OLS, GLM, GLMM, Survivalから選択
3. **変数設定** - 説明変数と目的変数を選択
4. **実行** - ワンクリックで分析開始
5. **結果確認** - 表形式結果とインタラクティブグラフ
6. **エクスポート** - JSONL形式のステップログを保存

### Python SDK

```python
import pandas as pd
from statelix_py.core import DataManager
from statelix_py.models import OLSModel

# データ読み込み
dm = DataManager()
data = dm.load_csv("data.csv")

# OLSモデルの実行
model = OLSModel()
result = model.fit(data, target="y", features=["x1", "x2", "x3"])

# 結果の表示
print(result.summary())
print(f"R²: {result.r_squared:.4f}")
print(f"MSE: {result.mse:.4f}")

# 可視化
result.plot_residuals()
```

### 詳細モード（研究者向け）

詳細モードでは以下が可能です：

- パラメータの細かな調整
- Python生コードの直接実行
- Wasmプラグインの利用
- ステップごとのデータハッシュ確認

```python
from statelix_py.models import GLMMModel

# GLMMモデルの詳細設定
model = GLMMModel(
    family="binomial",
    link="logit",
    max_iter=1000,
    tol=1e-6
)
result = model.fit(
    data,
    target="outcome",
    features=["age", "gender"],
    random_effects=["subject_id"]
)
```

## プロジェクト構造

```
statelix/
├── src/                    # C++コア
│   ├── linear_model/      # 線形モデル実装
│   ├── data/              # データ処理
│   ├── utils/             # ユーティリティ
│   └── bindings/          # Pythonバインディング
├── statelix_py/           # Pythonパッケージ
│   ├── core/              # コア機能
│   ├── models/            # 統計モデル
│   ├── gui/               # GUIコンポーネント
│   ├── plugins/           # プラグインシステム
│   └── utils/             # ユーティリティ
├── tests/                 # テスト
│   ├── unit/              # ユニットテスト
│   ├── integration/       # 統合テスト
│   └── cpp/               # C++テスト
├── docs/                  # ドキュメント
├── plugins_wasm/          # Wasmプラグイン
└── CMakeLists.txt         # CMake設定
```

## 開発

### テストの実行

```bash
# Pythonテスト
pytest tests/ -v --cov=statelix_py

# C++テスト
cd build
ctest --output-on-failure
```

### コードフォーマット

```bash
# Python
black statelix_py/
flake8 statelix_py/

# C++
clang-format -i src/**/*.cpp src/**/*.h
```

### Dockerでの開発

```bash
docker build -t statelix:dev .
docker run -it -v $(pwd):/statelix statelix:dev
```

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

## 技術スタック

- **C++**: Eigen (線形代数), pybind11 (Pythonバインディング)
- **Python**: NumPy, Pandas, PyArrow
- **統計**: statsmodels, lifelines
- **GUI**: PyQt6
- **可視化**: Matplotlib, Plotly
- **プラグイン**: Wasmtime
- **テスト**: pytest, Google Test
- **CI/CD**: GitHub Actions

## 貢献

プルリクエストを歓迎します！詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## サポート

- 📖 [ドキュメント](https://statelix.readthedocs.io)
- 🐛 [Issues](https://github.com/statelix/statelix/issues)
- 💬 [Discussions](https://github.com/statelix/statelix/discussions)

## ロードマップ

- [x] v2.0: 基本機能（OLS, GUI, ステップログ）
- [x] v2.1: 高度なモデル（GLM, GLMM）
- [x] v2.2: プラグインシステム（Wasm）
- [ ] v2.3: 分散処理サポート
- [ ] v3.0: クラウド統合、プラグインストア

---

© 2025 Statelix Development Team
