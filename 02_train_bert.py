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
    Trainer
)
try:
    from transformers import EarlyStoppingCallback
except ImportError:
    EarlyStoppingCallback = None
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# 設定
# ========================================

CONFIG = {
    'model_name': 'cl-tohoku/bert-base-japanese-v3',
    'max_length': 29,  # コマ1のEDA結果から決定（95%タイル）
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
    print("データファイルを読み込み中...")
    train_data = load_jsonl('train.jsonl')
    val_data = load_jsonl('val.jsonl')
    test_data = load_jsonl('test.jsonl')
    
    print(f"  - Train: {len(train_data)}件")
    print(f"  - Val: {len(val_data)}件")
    print(f"  - Test: {len(test_data)}件")
    
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

# ========================================
# トークナイズ
# ========================================

def tokenize_function(examples, tokenizer, max_length):
    """
    テキストをトークナイズ
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None  # Datasetで使うときはNone
    )

# ========================================
# 評価メトリクス
# ========================================

def compute_metrics(pred):
    """
    Accuracy, Precision, Recall, F1を計算
    """
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

# ========================================
# 混同行列の可視化
# ========================================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    混同行列を作成・保存
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # figuresフォルダを作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['明確(0)', '曖昧(1)'],
                yticklabels=['明確(0)', '曖昧(1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  → 混同行列を保存: {save_path}")

# ========================================
# メイン処理
# ========================================

def main():
    print("="*60)
    print("BERT学習スクリプト開始")
    print("外国人労働者向けビジネス指示文 曖昧性判定システム")
    print("="*60)
    
    # シード固定
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # GPU確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用デバイス: {device}")
    if device.type == 'cuda':
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 設定の表示
    print("\n--- 学習設定 ---")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # データセット作成
    print("\n[1/6] データセット読み込み中...")
    dataset = create_dataset()
    print(f"✓ データセット読み込み完了")
    
    # トークナイザー初期化
    print("\n[2/6] トークナイザー初期化中...")
    tokenizer = BertJapaneseTokenizer.from_pretrained(CONFIG['model_name'])
    print(f"✓ トークナイザー初期化完了")
    
    # トークナイズ
    print("\n[3/6] トークナイズ中...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG['max_length']),
        batched=True,
        desc="Tokenizing"
    )
    print(f"✓ トークナイズ完了")
    
    # モデル初期化
    print("\n[4/6] モデル初期化中...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=2
    )
    model.to(device)
    
    # モデルパラメータ数の表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ モデル初期化完了")
    print(f"  - 総パラメータ数: {total_params:,}")
    print(f"  - 学習可能パラメータ数: {trainable_params:,}")
    
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
    print(f"✓ 学習設定完了")
    
    # Trainer初期化
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets['train'],
        'eval_dataset': tokenized_datasets['validation'],
        'compute_metrics': compute_metrics,
    }
    
    if EarlyStoppingCallback is not None:
        trainer_kwargs['callbacks'] = [EarlyStoppingCallback(early_stopping_patience=2)]
    
    trainer = Trainer(**trainer_kwargs)
    
    # 学習開始
    print("\n[6/6] 学習開始...")
    print("-"*60)
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()
    
    # 学習結果の表示
    print("\n" + "="*60)
    print("学習完了！")
    print("="*60)
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"学習時間: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分)")
    print(f"最終Loss: {train_result.metrics['train_loss']:.4f}")
    
    # Validationデータで評価
    print("\n--- Validation結果 ---")
    val_results = trainer.evaluate()
    for key, value in val_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Testデータで評価
    print("\n--- Test結果 ---")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 混同行列の作成
    print("\n混同行列を作成中...")
    predictions = trainer.predict(tokenized_datasets['test'])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    plot_confusion_matrix(y_true, y_pred, 'figures/confusion_matrix.png')
    
    # モデル保存
    print("\nモデルを保存中...")
    os.makedirs('./saved_model', exist_ok=True)
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
    print(f"✓ モデル保存完了: ./saved_model")
    
    # 結果をファイルに保存
    print("\n結果をファイルに保存中...")
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'training_time_seconds': elapsed_time,
        'train_loss': float(train_result.metrics['train_loss']),
        'val_results': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in val_results.items()},
        'test_results': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in test_results.items()}
    }
    
    with open('training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 結果保存完了: training_results.json")
    
    # サマリー表示
    print("\n" + "="*60)
    print("🎉 すべて完了しました！")
    print("="*60)
    print(f"📁 モデル保存先: ./saved_model")
    print(f"📊 結果保存先: training_results.json")
    print(f"📈 混同行列: figures/confusion_matrix.png")
    print("\n--- 最終スコア ---")
    print(f"  Test Accuracy:  {test_results['eval_accuracy']:.2%}")
    print(f"  Test F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"  Test Precision: {test_results['eval_precision']:.4f}")
    print(f"  Test Recall:    {test_results['eval_recall']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()


