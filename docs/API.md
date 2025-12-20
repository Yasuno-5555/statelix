# Statelix API リファレンス

Statelixの主要なPython APIについて説明します。

---

## 目次
1. [StatelixOLS - 線形回帰](#statelixols)
2. [Storyteller - ナラティブ生成](#storyteller)
3. [Accelerator - GPU高速化](#accelerator)

---

## StatelixOLS

高速な最小二乗法（OLS）回帰モデル。C++/Eigen3バックエンドによる高速化。

### インポート
```python
from statelix import StatelixOLS
```

### クラス定義
```python
class StatelixOLS(fit_intercept=True)
```

**パラメータ:**
- `fit_intercept` (bool): 切片を含めるかどうか（デフォルト: True）

### メソッド

#### `fit(X, y)`
モデルを学習します。

**パラメータ:**
- `X` (array-like): 特徴量行列 (N, K)
- `y` (array-like): ターゲットベクトル (N,)

**戻り値:** self

#### `predict(X)`
予測値を返します。

**パラメータ:**
- `X` (array-like): 特徴量行列

**戻り値:** 予測値の配列

### 属性
- `coef_`: 係数ベクトル
- `intercept_`: 切片

### 使用例
```python
from statelix import StatelixOLS
import numpy as np

X = np.random.randn(1000, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(1000) * 0.1

model = StatelixOLS()
model.fit(X, y)
print(f"係数: {model.coef_}")
print(f"切片: {model.intercept_}")

predictions = model.predict(X)
```

---

## Storyteller

分析結果をナラティブ形式で説明します。教育モード向け。

### インポート
```python
from statelix_py.inquiry.narrative import Storyteller
```

### クラス定義
```python
class Storyteller(model, model_type='linear', feature_names=None)
```

**パラメータ:**
- `model`: 学習済みモデル（StatelixOLSなど）
- `model_type` (str): モデルタイプ（'linear', 'causal', 'sem', 'discrete'）
- `feature_names` (list): 特徴量の名前リスト

### メソッド

#### `explain()`
分析結果をマークダウン形式のナラティブで返します。

**戻り値:** マークダウン形式の文字列

### 出力構造
1. **Analysis Facts** - データから得られた事実
2. **Interpretation Hints** - 解釈のヒント
3. **Conclusion** - ユーザーが記入するための空欄

### 使用例
```python
from statelix import StatelixOLS
from statelix_py.inquiry.narrative import Storyteller

model = StatelixOLS()
model.fit(X, y)

story = Storyteller(model, feature_names=['X1', 'X2', 'X3'])
narrative = story.explain()
print(narrative)
```

---

## Accelerator

GPU（CUDA）を使用した行列演算の高速化モジュール。

### インポート
```python
from statelix import accelerator
```

### 関数

#### `is_available()`
GPUアクセラレーションが利用可能かどうかを確認します。

**戻り値:** bool

#### `weighted_gram_matrix(X, weights)`
重み付きグラム行列を計算します。

**パラメータ:**
- `X` (ndarray): 入力行列
- `weights` (ndarray): 重みベクトル

**戻り値:** グラム行列

### 使用例
```python
from statelix import accelerator
import numpy as np

if accelerator and accelerator.is_available():
    X = np.random.randn(10000, 100)
    weights = np.ones(10000)
    gram = accelerator.weighted_gram_matrix(X, weights)
    print("GPU accelerated!")
else:
    print("CPU fallback")
```

---

## 利用可能なモデル一覧

| モジュール | 説明 |
|-----------|------|
| `models.linear` | 線形回帰 (OLS) |
| `models.anova` | 分散分析 |
| `models.bayes` | ベイズ推定 |
| `models.causal` | 因果推論 (IV, DID, RDD) |
| `models.discrete` | 離散選択モデル (Logit, Probit) |
| `models.sem` | 構造方程式モデリング |
| `models.spatial` | 空間計量経済学 |
| `models.survival` | 生存分析 |
| `models.hypothesis_tests` | 仮説検定 |
| `models.diagnostics` | モデル診断 |

---

## バージョン情報
```python
import statelix
print(statelix.__version__)  # '0.2.0'
```
