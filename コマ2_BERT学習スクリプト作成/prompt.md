# コマ2実装プロンプト: BERT学習スクリプト作成

このプロンプトをCursor Composerにコピーして実装してください。

---

## 🤖 実装依頼

`02_train_bert.py` を作成し、BERTによる二値分類モデルの学習スクリプトを実装してください。

---

## 📋 実装仕様

### 使用モデル
- **ベースモデル**: `cl-tohoku/bert-base-japanese-v3`（東北大学BERT）
- **タスク**: 二値分類（Label 0: 明確 / Label 1: 曖昧）

### 学習パラメータ（初期設定）

```python
TRAINING_CONFIG = {
    'model_name': 'cl-tohoku/bert-base-japanese-v3',
    'max_length': 128,  # コマ1のeda_report.mdから調整
    'batch_size': 16,   # GPUメモリに応じて調整
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'output_dir': './results',
    'logging_steps': 50,
    'eval_steps': 100,
    'save_steps': 100,
    'seed': 42
}
```

### データセット
- `train.jsonl` (1,600サンプル)
- `val.jsonl` (200サンプル)
- `test.jsonl` (200サンプル)

---

## 🔧 スクリプト構造

以下の構造で `02_train_bert.py` を実装してください：

```python
"""
BERT二値分類モデル学習スクリプト
外国人労働者向けビジネス指示文 曖昧性判定システム
"""

import json
import os
from datetime import datetime
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BertJapaneseTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# 設定
# ========================================

CONFIG = {
    'model_name': 'cl-tohoku/bert-base-japanese-v3',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'output_dir': './results',
    'logging_dir': './logs',
    'seed': 42
}

# ========================================
# データ読み込み
# ========================================

def load_jsonl(filepath):
    """
    JSONLファイルを読み込む
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_dataset():
    """
    HuggingFace Dataset形式に変換
    """
    # TODO: 実装してください
    # train.jsonl, val.jsonl, test.jsonl を読み込み
    # DatasetDict形式に変換
    pass

# ========================================
# トークナイズ
# ========================================

def tokenize_function(examples, tokenizer, max_length):
    """
    テキストをトークナイズ
    """
    # TODO: 実装してください
    pass

# ========================================
# 評価メトリクス
# ========================================

def compute_metrics(pred):
    """
    Accuracy, Precision, Recall, F1を計算
    """
    # TODO: 実装してください
    # pred.label_ids と pred.predictions から計算
    pass

# ========================================
# 混同行列の可視化
# ========================================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    混同行列を作成・保存
    """
    # TODO: 実装してください
    pass

# ========================================
# メイン処理
# ========================================

def main():
    print("="*60)
    print("BERT学習スクリプト開始")
    print("="*60)
    
    # シード固定
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # GPU確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット作成
    print("\n[1/6] データセット読み込み中...")
    dataset = create_dataset()
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # トークナイザー初期化
    print("\n[2/6] トークナイザー初期化中...")
    tokenizer = BertJapaneseTokenizer.from_pretrained(CONFIG['model_name'])
    
    # トークナイズ
    print("\n[3/6] トークナイズ中...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG['max_length']),
        batched=True
    )
    
    # モデル初期化
    print("\n[4/6] モデル初期化中...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2
    )
    model.to(device)
    
    # 学習設定
    print("\n[5/6] 学習設定中...")
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        weight_decay=CONFIG['weight_decay'],
        warmup_steps=CONFIG['warmup_steps'],
        logging_dir=CONFIG['logging_dir'],
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=CONFIG['seed'],
        report_to='none'  # TensorBoardなどを使わない場合
    )
    
    # Trainer初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 学習開始
    print("\n[6/6] 学習開始...")
    print("-"*60)
    train_result = trainer.train()
    
    # 学習結果の表示
    print("\n" + "="*60)
    print("学習完了！")
    print("="*60)
    print(f"学習時間: {train_result.metrics['train_runtime']:.2f}秒")
    print(f"最終Loss: {train_result.metrics['train_loss']:.4f}")
    
    # Validationデータで評価
    print("\n--- Validation結果 ---")
    val_results = trainer.evaluate()
    for key, value in val_results.items():
        print(f"{key}: {value:.4f}")
    
    # Testデータで評価
    print("\n--- Test結果 ---")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    # 混同行列の作成
    print("\n混同行列を作成中...")
    predictions = trainer.predict(tokenized_datasets['test'])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    plot_confusion_matrix(y_true, y_pred, 'figures/confusion_matrix.png')
    
    # モデル保存
    print("\nモデルを保存中...")
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    
    # 結果をファイルに保存
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'train_loss': train_result.metrics['train_loss'],
        'val_results': val_results,
        'test_results': test_results
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print("\n✅ すべて完了しました！")
    print(f"モデル保存先: ./saved_model")
    print(f"結果保存先: training_results.json")

if __name__ == '__main__':
    main()
```

---

## ✅ 実装のポイント

### 1. データ読み込み（create_dataset関数）

```python
def create_dataset():
    train_data = load_jsonl('train.jsonl')
    val_data = load_jsonl('val.jsonl')
    test_data = load_jsonl('test.jsonl')
    
    # HuggingFace Dataset形式に変換
    train_dataset = Dataset.from_dict({
        'text': [item['text'] for item in train_data],
        'label': [item['label'] for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        'text': [item['text'] for item in val_data],
        'label': [item['label'] for item in val_data]
    })
    
    test_dataset = Dataset.from_dict({
        'text': [item['text'] for item in test_data],
        'label': [item['label'] for item in test_data]
    })
    
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset
```

### 2. トークナイズ関数

```python
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None  # Datasetで使うときはNone
    )
```

### 3. 評価メトリクス

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### 4. 混同行列の可視化

```python
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['明確(0)', '曖昧(1)'],
                yticklabels=['明確(0)', '曖昧(1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"混同行列を保存: {save_path}")
```

---

## 🧪 テスト実行

スクリプト作成後、まず小規模データでテストしてください：

### テスト用小規模データの作成

```python
# test_small.py
import json

# 各データセットから10件ずつ抽出
def create_small_dataset(input_file, output_file, n=10):
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

```bash
# 小規模データでテスト（1 epoch、5分程度）
python test_small.py
python 02_train_bert.py  # train.jsonl等をtrain_small.jsonl等に変更して実行
```

---

## 📖 training_guide.md の作成

スクリプトと一緒に `training_guide.md` を作成してください：

```markdown
# BERT学習実行ガイド

## 実行方法

### 通常実行（全データ）
```bash
python 02_train_bert.py
```

### パラメータ調整

スクリプト内の `CONFIG` 辞書を編集してください：

- `max_length`: トークン最大長（推奨: 128）
- `batch_size`: バッチサイズ（GPU: 16, CPU: 8）
- `learning_rate`: 学習率（推奨: 2e-5）
- `num_epochs`: エポック数（推奨: 3-5）

## 学習時間の目安

- **GPU使用時**: 約10-15分（3 epochs）
- **CPU使用時**: 約1-2時間（3 epochs）

## 期待される精度

- **初回学習**: Accuracy 70-80%
- **調整後**: Accuracy 80-85%

## トラブルシューティング

### メモリ不足
→ batch_sizeを8または4に減らす

### 学習が進まない
→ learning_rateを1e-5または5e-5に変更

### 過学習
→ weight_decayを0.1に増やす、Early Stoppingを活用
```

---

## ✅ 完了確認

- [ ] `02_train_bert.py` が作成された
- [ ] すべての関数が実装された
- [ ] 小規模データでテスト実行が成功した
- [ ] `training_guide.md` が作成された
- [ ] 次のコマ（本格学習）の準備が整った

---

## ⏭️ 次のコマ（コマ3）への準備

完了したら、次のコマ3で本格学習を実行します。

準備事項：
- スクリプトが正常に動作することを確認
- 学習時間の見積もりを把握
- GPU/CPUどちらで実行するか決定










