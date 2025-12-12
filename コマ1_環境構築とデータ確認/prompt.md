# コマ1実装プロンプト: 環境構築とデータ確認

このプロンプトをCursor Composerにコピーして実行してください。

---

## 🤖 実装依頼

以下のタスクを順番に実装してください：

### タスク1: requirements.txt の作成

プロジェクトルート（`/home/ike/Desktop/人工知能応用データセット/`）に `requirements.txt` を作成してください。

必要なライブラリ：
- transformers>=4.30.0
- torch>=2.0.0
- datasets>=2.14.0
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- japanize-matplotlib>=1.1.3
- tqdm>=4.65.0

作成後、以下のコマンドでインストールを実行してください：
```bash
pip install -r requirements.txt
```

---

### タスク2: EDAスクリプト（01_eda.py）の作成

`01_eda.py` を作成し、以下の分析を実装してください：

#### 実装する機能：

1. **データ読み込み**
   - train.jsonl, val.jsonl, test.jsonlを読み込む
   - データ構造を確認（text, label, reasonフィールド）
   - 基本統計を表示

2. **ラベル分布の可視化**
   - 各データセット（train/val/test）のラベル分布を棒グラフで表示
   - 50:50のバランスを確認

3. **文字数分析**
   - 各データセットの文字数分布をヒストグラムで表示
   - 最小値、最大値、平均値、中央値を計算
   - Label 0とLabel 1で文字数に差があるか確認

4. **頻出語分析**
   - Label 0（明確）とLabel 1（曖昧）それぞれで頻出する単語TOP20を抽出
   - WordCloudまたは棒グラフで可視化

5. **サンプルデータ表示**
   - Label 0とLabel 1それぞれからランダムに5件ずつ表示
   - 特徴的なパターンを確認

6. **トークナイズテスト**
   - BERTトークナイザー（cl-tohoku/bert-base-japanese-v3）を使用
   - 各テキストのトークン数を計算
   - トークン数の分布をヒストグラムで表示
   - 推奨max_lengthを決定（95パーセンタイルなど）

7. **結果の保存**
   - すべての図を `figures/` フォルダに保存
   - 分析結果を `eda_report.md` にまとめる

#### スクリプトの構造：

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from collections import Counter
from transformers import BertTokenizer
import os

# figuresフォルダの作成
os.makedirs('figures', exist_ok=True)

# データ読み込み関数
def load_jsonl(filepath):
    # 実装してください
    pass

# ラベル分布の可視化
def plot_label_distribution(data_dict):
    # 実装してください
    pass

# 文字数分析
def analyze_text_length(data_dict):
    # 実装してください
    pass

# 頻出語分析
def analyze_frequent_words(data_dict):
    # 実装してください
    pass

# トークナイズ分析
def analyze_tokenization(data_dict):
    # 実装してください
    pass

# メイン実行
if __name__ == '__main__':
    # データ読み込み
    train_data = load_jsonl('train.jsonl')
    val_data = load_jsonl('val.jsonl')
    test_data = load_jsonl('test.jsonl')
    
    data_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # 各分析を実行
    plot_label_distribution(data_dict)
    analyze_text_length(data_dict)
    analyze_frequent_words(data_dict)
    analyze_tokenization(data_dict)
    
    print("EDA完了！結果はfigures/フォルダに保存されました。")
```

---

### タスク3: eda_report.mdの自動生成

`01_eda.py` の最後に、分析結果を `eda_report.md` として出力する関数を追加してください。

#### レポートに含める内容：

```markdown
# データセット分析レポート

## データ概要

| データセット | サンプル数 | Label 0 | Label 1 |
|------------|----------|---------|---------|
| Train | 1600 | 800 | 800 |
| Validation | 200 | 100 | 100 |
| Test | 200 | 100 | 100 |

## 文字数統計

| 統計量 | Train | Val | Test |
|-------|-------|-----|------|
| 最小値 | XX | XX | XX |
| 最大値 | XX | XX | XX |
| 平均値 | XX | XX | XX |
| 中央値 | XX | XX | XX |

## トークン数統計

| 統計量 | Train | Val | Test |
|-------|-------|-----|------|
| 最小値 | XX | XX | XX |
| 最大値 | XX | XX | XX |
| 平均値 | XX | XX | XX |
| 95%タイル | XX | XX | XX |

**推奨max_length**: XX

## 頻出語TOP10

### Label 0（明確な指示）
1. XXX (XX回)
2. XXX (XX回)
...

### Label 1（曖昧な指示）
1. XXX (XX回)
2. XXX (XX回)
...

## 所見

- ラベルバランス: ✅ 完全に50:50
- 文字数分布: ...
- トークン長: ...
- 特徴的なパターン: ...

## 次のコマへの推奨事項

- max_length設定: XX
- batch_size推奨: 16（メモリに応じて調整）
- 特に注意すべきデータ: ...
```

---

### タスク4: 実行とGPU確認

スクリプト作成後、以下を実行してください：

1. **GPU確認スクリプト**（`check_gpu.py`）:
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("⚠️ GPU not available. Using CPU.")
```

2. **EDA実行**:
```bash
python 01_eda.py
```

---

## ✅ 完了確認

すべて完了したら、以下をチェックしてください：

- [ ] `requirements.txt` が作成され、ライブラリがインストール済み
- [ ] `check_gpu.py` を実行し、GPU/CPU状況を確認済み
- [ ] `01_eda.py` が正常に実行できる
- [ ] `figures/` フォルダに可視化結果が保存されている
- [ ] `eda_report.md` が生成されている
- [ ] レポートの内容を確認し、データの特性を把握した

---

## 🔧 トラブルシューティング

### エラー: ModuleNotFoundError
→ `pip install -r requirements.txt` を再実行

### エラー: japanize-matplotlibが動かない
→ 代わりに `plt.rcParams['font.sans-serif'] = ['DejaVu Sans']` を使用

### GPU が認識されない
→ CPU で学習可能（時間がかかるが問題なし）

### メモリエラー
→ 次のコマでbatch_sizeを小さくすれば対応可能

---

## ⏭️ 次のコマ（コマ2）への準備

このコマが完了したら、以下を確認してください：

1. `eda_report.md` の推奨max_lengthをメモ
2. データの特徴（Label 0とLabel 1の違い）を理解
3. GPU/CPUどちらで学習するか決定

次のコマ2では、この情報を使ってBERT学習スクリプトを作成します。










