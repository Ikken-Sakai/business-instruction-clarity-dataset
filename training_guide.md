# BERT学習実行ガイド

## 📋 実行方法

### 通常実行（全データ）

```bash
cd /home/ike/Desktop/人工知能応用データセット
python 02_train_bert.py
```

---

## ⚙️ パラメータ調整

スクリプト内の `CONFIG` 辞書を編集してください：

### 主要パラメータ

| パラメータ | デフォルト値 | 説明 | 推奨範囲 |
|:----------|:------------|:----|:---------|
| `max_length` | **29** | トークン最大長 | 29-64 |
| `batch_size` | **16** | バッチサイズ | GPU: 16-32, CPU: 8-16 |
| `learning_rate` | **2e-5** | 学習率 | 1e-5 〜 5e-5 |
| `num_epochs` | **3** | エポック数 | 3-5 |
| `weight_decay` | **0.01** | 正則化係数 | 0.01-0.1 |

### パラメータの決定根拠

- **max_length = 29**: コマ1のEDA結果で、95%のデータが29トークン以下
- **batch_size = 16**: GPUメモリに応じた標準的な値
- **learning_rate = 2e-5**: BERTファインチューニングの標準値
- **num_epochs = 3**: 過学習を避けるための初期値

---

## ⏱️ 学習時間の目安

### GPU使用時
- **学習時間**: 約10-15分（3 epochs）
- **メモリ使用量**: 約2-4GB

### CPU使用時
- **学習時間**: 約1-2時間（3 epochs）
- **推奨**: 可能であればGPUを使用

---

## 🎯 期待される精度

### 初回学習（コマ3）
- **目標 Accuracy**: 70%以上
- **期待値**: 75-80%

### 調整後（コマ5）
- **目標 Accuracy**: 80%以上
- **期待値**: 85%前後

---

## 📂 生成されるファイル

### 学習完了後に生成されるファイル

```
人工知能応用データセット/
├── saved_model/              # 学習済みモデル
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── results/                  # 学習ログ
│   └── checkpoint-*/
├── training_results.json     # 学習結果サマリー
└── figures/
    └── confusion_matrix.png  # 混同行列
```

---

## 🔧 トラブルシューティング

### エラー: メモリ不足（CUDA out of memory）

**原因**: GPUメモリが不足しています

**解決策**:
```python
CONFIG = {
    ...
    'batch_size': 8,  # 16 → 8に減らす
    ...
}
```

さらにメモリ不足の場合:
```python
CONFIG = {
    ...
    'batch_size': 4,  # さらに減らす
    'max_length': 29,  # すでに最小化済み
    ...
}
```

---

### エラー: 学習が進まない（Loss が下がらない）

**原因**: 学習率が適切でない可能性

**解決策**:
```python
CONFIG = {
    ...
    'learning_rate': 1e-5,  # 2e-5 → 1e-5に減らす
    ...
}
```

または:
```python
CONFIG = {
    ...
    'learning_rate': 5e-5,  # 2e-5 → 5e-5に増やす
    ...
}
```

---

### 警告: 過学習が発生（Val精度 < Train精度）

**原因**: モデルが訓練データに過適合しています

**解決策1**: weight_decayを増やす
```python
CONFIG = {
    ...
    'weight_decay': 0.1,  # 0.01 → 0.1に増やす
    ...
}
```

**解決策2**: エポック数を減らす
```python
CONFIG = {
    ...
    'num_epochs': 2,  # 3 → 2に減らす
    ...
}
```

**解決策3**: Early Stoppingがデフォルトで有効
- Validationスコアが2エポック連続で改善しない場合、自動的に学習停止

---

### エラー: ファイルが見つからない（FileNotFoundError）

**原因**: データファイルが正しい場所にありません

**解決策**:
```bash
# 実行前に正しいディレクトリにいることを確認
cd /home/ike/Desktop/人工知能応用データセット
ls train.jsonl val.jsonl test.jsonl  # ファイルの存在確認
```

---

## 📊 結果の確認方法

### 1. コンソール出力を確認

学習完了後、以下のような出力が表示されます：

```
==========================================================
🎉 すべて完了しました！
==========================================================
📁 モデル保存先: ./saved_model
📊 結果保存先: training_results.json
📈 混同行列: figures/confusion_matrix.png

--- 最終スコア ---
  Test Accuracy:  78.50%
  Test F1 Score:  0.7842
  Test Precision: 0.7912
  Test Recall:    0.7773
==========================================================
```

### 2. training_results.json を確認

```bash
cat training_results.json
```

### 3. 混同行列を確認

```bash
# 画像ビューアで開く
xdg-open figures/confusion_matrix.png
```

---

## 🧪 テスト実行（推奨）

本格学習の前に、小規模データでテストすることを推奨します。

### 小規模テストデータの作成

```python
# test_small.py
import json

def create_small_dataset(input_file, output_file, n=20):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data[:n]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

create_small_dataset('train.jsonl', 'train_small.jsonl', 20)
create_small_dataset('val.jsonl', 'val_small.jsonl', 10)
create_small_dataset('test.jsonl', 'test_small.jsonl', 10)
```

### テスト実行

1. 小規模データを作成:
```bash
python test_small.py
```

2. スクリプトを編集（ファイル名を変更）:
```python
# 02_train_bert.py の create_dataset() 関数内
train_data = load_jsonl('train_small.jsonl')  # train.jsonl → train_small.jsonl
val_data = load_jsonl('val_small.jsonl')
test_data = load_jsonl('test_small.jsonl')
```

3. テスト実行:
```bash
python 02_train_bert.py
```

**所要時間**: 約3-5分（GPU使用時）

---

## 📌 次のステップ（コマ3）

テスト実行が成功したら、コマ3で本格的な学習を実行します。

### 準備事項
- ✅ スクリプトが正常に動作することを確認
- ✅ 学習時間の見積もりを把握
- ✅ GPU/CPUどちらで実行するか決定
- ✅ ファイル名を元に戻す（train.jsonl等）

### コマ3で実施すること
1. 全データで本格的に学習
2. 初期精度の確認（目標: 70%以上）
3. 誤分類サンプルの抽出
4. エラー分析の準備

---

## 💡 Tips

### GPU使用を確認する方法

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 学習中のログをファイルに保存

```bash
python 02_train_bert.py 2>&1 | tee training.log
```

### 学習をバックグラウンドで実行

```bash
nohup python 02_train_bert.py > training.log 2>&1 &
```

---

**作成日**: 2025年12月12日  
**対象コマ**: コマ2（BERT学習スクリプト作成）


