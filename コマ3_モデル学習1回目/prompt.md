# コマ3実装プロンプト: モデル学習（1回目）

このプロンプトに従って、初回学習を実行してください。

---

## 🚀 学習実行手順

### ステップ1: 事前確認

以下のスクリプトを作成して、環境を確認してください：

```python
# pre_check.py
import os
import torch
import json

print("="*60)
print("学習前チェック")
print("="*60)

# 1. ディスク容量
import shutil
total, used, free = shutil.disk_usage("/")
print(f"\nディスク容量:")
print(f"  空き容量: {free // (2**30)} GB")

# 2. GPU/CPUメモリ
if torch.cuda.is_available():
    print(f"\nGPU情報:")
    print(f"  GPU名: {torch.cuda.get_device_name(0)}")
    print(f"  総メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\nCPUモードで学習します")

# 3. データセット確認
datasets = ['train.jsonl', 'val.jsonl', 'test.jsonl']
for ds in datasets:
    if os.path.exists(ds):
        with open(ds, 'r') as f:
            count = sum(1 for _ in f)
        print(f"  {ds}: {count}件 ✅")
    else:
        print(f"  {ds}: 見つかりません ❌")

# 4. スクリプト確認
if os.path.exists('02_train_bert.py'):
    print("\n02_train_bert.py: 存在します ✅")
else:
    print("\n02_train_bert.py: 見つかりません ❌")

print("\n準備完了！学習を開始できます。")
```

実行：
```bash
python pre_check.py
```

---

### ステップ2: 学習実行

```bash
python 02_train_bert.py
```

**学習中の確認ポイント**：

1. **初期設定が表示される**
   - モデル名
   - データセットサイズ
   - デバイス（GPU/CPU）

2. **学習プログレスバー**
   - Epoch 1/3, 2/3, 3/3 と進む
   - Lossが徐々に下がることを確認

3. **Validation評価**
   - 各Epoch後にValidationが実行される
   - Accuracyが上がっていることを確認

4. **エラーがないか監視**
   - CUDA out of memory → batch_sizeを減らして再実行
   - その他のエラー → エラーメッセージを記録

---

### ステップ3: 誤分類サンプルの抽出

学習完了後、以下のスクリプトで誤分類サンプルを抽出してください：

```python
# extract_errors.py
"""
誤分類サンプルを抽出するスクリプト
"""

import json
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def predict(texts, model, tokenizer, device):
    """
    テキストのリストを予測
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    
    return predictions.cpu().numpy()

def main():
    print("誤分類サンプルを抽出中...")
    
    # モデル・トークナイザー読み込み
    model = BertForSequenceClassification.from_pretrained('./saved_model')
    tokenizer = BertJapaneseTokenizer.from_pretrained('./saved_model')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Testデータ読み込み
    test_data = load_jsonl('test.jsonl')
    texts = [item['text'] for item in test_data]
    true_labels = [item['label'] for item in test_data]
    
    # 予測
    pred_labels = predict(texts, model, tokenizer, device)
    
    # 誤分類サンプルを抽出
    error_samples = []
    for i, (text, true_label, pred_label) in enumerate(zip(texts, true_labels, pred_labels)):
        if true_label != pred_label:
            error_samples.append({
                'id': i,
                'text': text,
                'true_label': int(true_label),
                'pred_label': int(pred_label),
                'true_label_name': '明確' if true_label == 0 else '曖昧',
                'pred_label_name': '明確' if pred_label == 0 else '曖昧'
            })
    
    # 保存
    with open('error_samples.json', 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n誤分類サンプル数: {len(error_samples)}/{len(test_data)}")
    print(f"正解率: {(len(test_data) - len(error_samples)) / len(test_data) * 100:.2f}%")
    print(f"\n誤分類サンプルを error_samples.json に保存しました。")
    
    # いくつか表示
    print("\n--- 誤分類サンプル例 ---")
    for i, sample in enumerate(error_samples[:5]):
        print(f"\n{i+1}. {sample['text']}")
        print(f"   正解: {sample['true_label_name']} → 予測: {sample['pred_label_name']}")

if __name__ == '__main__':
    main()
```

実行：
```bash
python extract_errors.py
```

---

### ステップ4: 初回学習レポートの作成

`first_training_report.md` を作成してください：

```markdown
# 初回学習レポート

## 学習設定

- **モデル**: cl-tohoku/bert-base-japanese-v3
- **データセット**: train 1,600件 / val 200件 / test 200件
- **パラメータ**:
  - max_length: ___
  - batch_size: ___
  - learning_rate: ___
  - num_epochs: ___

## 学習時間

- 開始時刻: ___
- 終了時刻: ___
- 所要時間: ___ 分

## 結果

### Test評価指標

| 指標 | スコア |
|------|--------|
| Accuracy | ___ % |
| F1スコア | ___ |
| Precision | ___ |
| Recall | ___ |

### 混同行列

（figures/confusion_matrix.png を参照）

### 誤分類分析

- 誤分類サンプル数: ___ / 200
- 誤分類率: ___ %

#### 誤分類の傾向

1. **Label 0→1 の誤分類（明確を曖昧と判定）**
   - サンプル数: ___
   - 特徴: ___

2. **Label 1→0 の誤分類（曖昧を明確と判定）**
   - サンプル数: ___
   - 特徴: ___

## 考察

### うまくいった点

- ___

### 改善が必要な点

- ___

## 次のステップ（コマ4で実施）

### 改善案

1. ___
2. ___
3. ___

### 追加分析

- ___
```

---

## 📊 結果の読み方

### 目標精度の判定

- **Accuracy 80%以上**: 🎉 素晴らしい！
- **Accuracy 75-80%**: ✅ 良好！
- **Accuracy 70-75%**: 😐 改善の余地あり
- **Accuracy 70%未満**: 😢 パラメータ調整が必要

### 混同行列の読み方

```
           予測:明確  予測:曖昧
正解:明確     90        10      ← 明確なのに曖昧と誤判定（10件）
正解:曖昧     15        85      ← 曖昧なのに明確と誤判定（15件）
```

- **左上（True Positive）**: 明確を明確と正しく判定
- **右下（True Negative）**: 曖昧を曖昧と正しく判定
- **右上（False Positive）**: 明確なのに曖昧と誤判定
- **左下（False Negative）**: 曖昧なのに明確と誤判定

---

## ✅ 完了確認

- [ ] 学習が正常に完了した
- [ ] `training_results.json` が生成された
- [ ] `saved_model/` フォルダが作成された
- [ ] 混同行列が保存された
- [ ] `error_samples.json` が生成された
- [ ] `first_training_report.md` を作成した
- [ ] Test Accuracy ___ % を達成

---

## ⏭️ 次のコマ（コマ4）への準備

以下をメモしてください：

1. **達成した精度**: Accuracy ___ %
2. **誤分類の主な傾向**: ___
3. **改善すべきポイント**: ___

これらの情報を持って、コマ4でエラー分析と改善を行います。










