# Statelix 上級ユーザーガイド

開発者・研究者向けの詳細ドキュメントです。

---

## アーキテクチャ概要

```
statelix/
├── src/                    # C++ コア（Eigen3ベース）
│   ├── linear/             # 線形モデル（OLS, WLS, GLS）
│   ├── causal/             # 因果推論（IV, DID, RDD）
│   ├── spatial/            # 空間計量経済学
│   └── accelerator/        # CUDA GPU アクセラレータ
├── statelix_py/            # Python インターフェース
│   ├── models/             # 高レベルモデルAPI
│   ├── gui/                # PySide6 GUIアプリケーション
│   └── inquiry/            # 教育モード（Storyteller）
└── packaging/              # ビルド・配布ツール
```

---

## C++ バックエンド

### コンパイル要件
- **コンパイラ**: MSVC 2019+ / GCC 9+ / Clang 10+
- **依存ライブラリ**: Eigen3, pybind11
- **オプション**: CUDA Toolkit 11+ (GPU 高速化)

### ビルド方法
```bash
pip install pybind11 numpy
python setup.py build_ext --inplace
```

### 主要なC++クラス

#### `FitOLS` (src/linear/fit_ols.hpp)
```cpp
class FitOLS {
public:
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    
    // 統計量
    Eigen::VectorXd coef_;
    double intercept_;
    Eigen::VectorXd std_errors_;
    Eigen::VectorXd t_values_;
    Eigen::VectorXd p_values_;
    double aic_, bic_;
};
```

---

## 拡張ポイント

### 新しいモデルの追加

1. **Pythonのみ**: `statelix_py/models/` に新しいモジュールを追加
2. **C++高速化**: `src/` にC++実装 → `setup.py` でバインディング追加

### カスタムアダプタ（Storyteller用）
```python
from statelix_py.inquiry.adapters import BaseAdapter

class MyModelAdapter(BaseAdapter):
    def get_coefficients(self):
        return {'param1': self.model.param1}
    
    def get_metrics(self):
        return {'r2': self.model.r_squared}
```

---

## パフォーマンス最適化

### GPU アクセラレーション
```python
from statelix import accelerator

# 利用可能性チェック
if accelerator and accelerator.is_available():
    print(f"GPU: {accelerator.device_name()}")
```

### ベンチマーク実行
```bash
python benchmarks/sklearn_comparison.py
```

### 最適化のヒント
- 10,000行以上のデータで自動的にGPU使用
- `np.ascontiguousarray()` でメモリレイアウト最適化
- バッチ処理で予測を高速化

---

## 因果推論モジュール詳細

### 利用可能な推定手法

| 手法 | クラス | 仮定 |
|------|--------|------|
| 操作変数法 (IV) | `IVEstimator` | 除外制約、関連性 |
| 差の差法 (DID) | `DIDEstimator` | 平行トレンド |
| 回帰不連続 (RDD) | `RDDEstimator` | 連続性 |
| 傾向スコア | `PropensityScore` | CIA, Overlap |
| 合成コントロール | `SyntheticControl` | バランス |

### 使用例: IV推定
```python
from statelix_py.models.causal import IVEstimator

iv = IVEstimator()
result = iv.fit(
    y=outcome,
    treatment=treatment,
    instruments=instruments,
    controls=controls
)
print(f"処置効果: {result.effect}")
print(f"第一段階 F統計量: {result.first_stage_f}")
```

---

## GUIカスタマイズ

### テーマ変更
```python
from statelix_py.gui.styles import StatelixTheme

StatelixTheme.COLOR_ACCENT = "#FF5733"
StatelixTheme.COLOR_BG_MAIN = "#1a1a2e"
```

### パネルの追加
```python
from statelix_py.gui.main_window import MainWindow

class CustomMainWindow(MainWindow):
    def init_ui(self):
        super().init_ui()
        self.tabs.addTab(MyCustomPanel(), "Custom")
```

---

## テスト

### ユニットテスト実行
```bash
pytest tests/ -v
```

### 特定モジュールのテスト
```bash
pytest tests/test_linear.py -v
pytest tests/test_causal.py -v
```

---

## コントリビューション

### コーディング規約
- Python: PEP 8
- C++: Google C++ Style Guide
- コミットメッセージ: Conventional Commits

### プルリクエスト
1. Issueを作成してディスカッション
2. フォーク → ブランチ作成
3. テスト追加・通過確認
4. プルリクエスト送信

---

## ライセンス
MIT License

## 引用
研究で使用する場合:
```bibtex
@software{statelix,
  title = {Statelix: High-Performance Statistical Analysis Platform},
  year = {2025},
  url = {https://github.com/your-repo/statelix}
}
```
