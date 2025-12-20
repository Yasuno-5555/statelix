# Statelix v2.3 インストールガイド

## システム要件
- **OS**: Windows 10 / Windows 11 (64-bit)
- **メモリ**: 4GB以上（8GB推奨）
- **ディスク容量**: 1GB以上の空き容量

---

## インストール手順

### 1. インストーラーのダウンロード
`Statelix_Setup_v2.3.exe` を入手してください。

### 2. インストーラーの実行
1. `Statelix_Setup_v2.3.exe` をダブルクリック
2. 「このアプリがデバイスに変更を加えることを許可しますか？」と表示されたら **はい** を選択
3. インストールウィザードが起動します

### 3. インストール先の選択
- デフォルト: `C:\Program Files\Statelix`
- 変更する場合は「参照」ボタンで任意のフォルダを指定

### 4. オプション設定
- **デスクトップにショートカットを作成**: チェックを入れると、デスクトップにアイコンが作成されます

### 5. インストール完了
「完了」をクリックするとStatelixが起動します。

---

## アンインストール方法

### 方法1: 設定から
1. Windows設定 → アプリ → インストールされているアプリ
2. 「Statelix」を検索
3. 「アンインストール」をクリック

### 方法2: コントロールパネルから
1. コントロールパネル → プログラムと機能
2. 「Statelix」を選択
3. 「アンインストール」をクリック

---

## トラブルシューティング

### 起動時にエラーが出る場合
- **Visual C++ ランタイム**が必要な場合があります
- [Microsoft Visual C++ 再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe) をインストールしてください

### ウイルス対策ソフトにブロックされる場合
- PyInstallerで作成されたexeファイルは、一部のウイルス対策ソフトで誤検知される場合があります
- 信頼できる配布元からダウンロードした場合は、例外設定に追加してください

---

## サポート
問題が解決しない場合は、GitHubのIssueまたは開発者にお問い合わせください。

---

## 開発者向け: ソースからのビルド

Git cloneしてソースコードから実行・ビルドする場合の手順です。

### 前提条件
- Python 3.10以上
- Visual Studio Build Tools（C++コンパイラ）
- Inno Setup 6（インストーラー作成時のみ）

### 1. リポジトリのクローン
```bash
git clone https://github.com/Yasuno-5555/statelix
cd statelix
```

### 2. 仮想環境の作成と依存関係のインストール
```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
pip install pyinstaller
```

### 3. ソースから直接実行
```bash
python statelix_py/app.py
```

### 4. 実行ファイル（EXE）のビルド
```bash
python packaging/build_exe.py
```
成功すると `dist/Statelix/Statelix.exe` が生成されます。

### 5. インストーラーの作成（オプション）
Inno Setup 6をインストール後：
```bash
packaging\compile_installer.bat
```
成功すると `dist/Statelix_Setup_v2.3.exe` が生成されます。

### ビルド時間の目安
- PyInstaller: 約10〜15分
- Inno Setup: 約5〜7分
