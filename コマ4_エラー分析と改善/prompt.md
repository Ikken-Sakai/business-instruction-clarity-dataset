# コマ4実装プロンプト: エラー分析と改善

このプロンプトに従って、誤分類の詳細分析と改善策を立案してください。

---

## 🔍 エラー分析スクリプトの作成

`03_error_analysis.py` を作成してください：

```python
"""
誤分類の詳細分析スクリプト
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from collections import Counter
import numpy as np

def load_error_samples():
    """誤分類サンプルを読み込む"""
    with open('error_samples.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_error_patterns(error_samples):
    """
    誤分類パターンを分析
    """
    print("="*60)
    print("誤分類パターン分析")
    print("="*60)
    
    # Label 0→1 と Label 1→0 に分類
    label0_to_1 = [s for s in error_samples if s['true_label'] == 0]
    label1_to_0 = [s for s in error_samples if s['true_label'] == 1]
    
    print(f"\nLabel 0→1（明確を曖昧と誤判定）: {len(label0_to_1)}件")
    print(f"Label 1→0（曖昧を明確と誤判定）: {len(label1_to_0)}件")
    
    # 文字数分析
    print("\n--- 文字数分析 ---")
    len0_to_1 = [len(s['text']) for s in label0_to_1]
    len1_to_0 = [len(s['text']) for s in label1_to_0]
    
    if len0_to_1:
        print(f"Label 0→1 平均文字数: {np.mean(len0_to_1):.1f}文字")
    if len1_to_0:
        print(f"Label 1→0 平均文字数: {np.mean(len1_to_0):.1f}文字")
    
    # サンプル表示
    print("\n--- Label 0→1 の誤分類例（明確なのに曖昧と判定）---")
    for i, sample in enumerate(label0_to_1[:3]):
        print(f"{i+1}. {sample['text']}")
    
    print("\n--- Label 1→0 の誤分類例（曖昧なのに明確と判定）---")
    for i, sample in enumerate(label1_to_0[:3]):
        print(f"{i+1}. {sample['text']}")
    
    return label0_to_1, label1_to_0

def plot_error_analysis(label0_to_1, label1_to_0):
    """
    誤分類を可視化
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 誤分類の内訳
    ax1 = axes[0]
    counts = [len(label0_to_1), len(label1_to_0)]
    labels = ['明確→曖昧', '曖昧→明確']
    ax1.bar(labels, counts, color=['#ff9999', '#9999ff'])
    ax1.set_ylabel('誤分類数')
    ax1.set_title('誤分類の内訳')
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, str(v), ha='center')
    
    # 文字数分布
    ax2 = axes[1]
    if label0_to_1:
        len0 = [len(s['text']) for s in label0_to_1]
        ax2.hist(len0, bins=15, alpha=0.5, label='明確→曖昧', color='red')
    if label1_to_0:
        len1 = [len(s['text']) for s in label1_to_0]
        ax2.hist(len1, bins=15, alpha=0.5, label='曖昧→明確', color='blue')
    ax2.set_xlabel('文字数')
    ax2.set_ylabel('頻度')
    ax2.set_title('誤分類サンプルの文字数分布')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/error_patterns.png', dpi=150)
    print("\n可視化結果を figures/error_patterns.png に保存しました。")

def analyze_common_words(error_samples):
    """
    誤分類サンプルに頻出する単語を分析
    """
    print("\n--- 誤分類サンプルの頻出語 ---")
    
    # 全テキストから単語を抽出（簡易的に2文字以上で分割）
    all_text = ' '.join([s['text'] for s in error_samples])
    
    # よく出る語（簡易分析）
    common_patterns = ['早めに', 'なる早', '例の', 'いつもの', '適当に', 'ざっと', 
                      '後で', '確認', '対応', 'お願い']
    
    for pattern in common_patterns:
        count = sum(1 for s in error_samples if pattern in s['text'])
        if count > 0:
            print(f"  '{pattern}': {count}件")

def main():
    # 誤分類サンプル読み込み
    error_samples = load_error_samples()
    
    if not error_samples:
        print("誤分類サンプルがありません。精度100%！")
        return
    
    print(f"\n総誤分類数: {len(error_samples)}")
    
    # パターン分析
    label0_to_1, label1_to_0 = analyze_error_patterns(error_samples)
    
    # 可視化
    plot_error_analysis(label0_to_1, label1_to_0)
    
    # 頻出語分析
    analyze_common_words(error_samples)
    
    print("\n分析完了！")

if __name__ == '__main__':
    main()
```

---

## 📝 詳細レポートの作成

`error_analysis_report.md` を作成してください：

```markdown
# 誤分類の詳細分析レポート

## 分析対象

- モデル: 初回学習済みBERTモデル
- 誤分類数: ___ / 200件（誤分類率: ___％）

## 誤分類の内訳

| パターン | 件数 | 割合 |
|---------|-----|------|
| Label 0→1（明確を曖昧と誤判定） | ___ | ___% |
| Label 1→0（曖昧を明確と誤判定） | ___ | ___% |

## パターン別の詳細分析

### パターン1: Label 0→1（明確を曖昧と誤判定）

#### 特徴
- 平均文字数: ___ 文字
- 特徴的な傾向: ___

#### 代表例

1. テキスト: 「___」
   - 理由: ___

2. テキスト: 「___」
   - 理由: ___

3. テキスト: 「___」
   - 理由: ___

#### 誤判定の原因仮説

1. ___
2. ___
3. ___

---

### パターン2: Label 1→0（曖昧を明確と誤判定）

#### 特徴
- 平均文字数: ___ 文字
- 特徴的な傾向: ___

#### 代表例

1. テキスト: 「___」
   - 理由: ___

2. テキスト: 「___」
   - 理由: ___

3. テキスト: 「___」
   - 理由: ___

#### 誤判定の原因仮説

1. ___
2. ___
3. ___

---

## 頻出パターン

### 誤分類サンプルに頻出する表現

| 表現 | 出現回数 | 考察 |
|-----|--------|------|
| 「早めに」 | ___ | ___ |
| 「例の」 | ___ | ___ |
| 「確認」 | ___ | ___ |

---

## データの問題点

### ラベリングの妥当性

- [ ] ラベリングが明らかに間違っているサンプル: ___ 件
- [ ] グレーゾーンのサンプル: ___ 件

### データの偏り

- ___

---

## 改善策の提案

### 優先度1: すぐに実施すべき改善

#### 1. ハイパーパラメータ調整

**現在の設定:**
- learning_rate: 2e-5
- batch_size: 16
- num_epochs: 3

**調整案:**
- learning_rate: ___ （理由: ___）
- batch_size: ___ （理由: ___）
- num_epochs: ___ （理由: ___）

**期待される効果:** Accuracy +___% 向上

---

#### 2. max_lengthの調整

**現在:** 128

**調整案:** ___ （理由: ___）

**期待される効果:** ___

---

#### 3. Early Stoppingの調整

**調整案:** patience=___ （理由: ___）

---

### 優先度2: 時間があれば実施

#### 1. データ拡張

- ___

#### 2. Weight Decayの調整

- ___

#### 3. Warmup Stepsの調整

- ___

---

### 優先度3: 本番発表まで検討

#### 1. データセット拡張

- 現在2,000件 → 3,000-5,000件に拡張

#### 2. 別のBERTモデルの試行

- cl-tohoku/bert-large-japanese
- rinna/japanese-roberta-base

#### 3. アンサンブル

- 複数モデルの多数決

---

## 次のコマ（コマ5）で実施する施策

### 実施内容

1. **ハイパーパラメータ調整**
   - learning_rate: ___
   - num_epochs: ___
   - その他: ___

2. **期待される結果**
   - 目標Accuracy: ___% 以上
   - 誤分類数: ___ 件以下

---

## まとめ

### 主な発見

1. ___
2. ___
3. ___

### 改善の方向性

- ___
```

---

## 🔧 改善後の設定ファイル作成

`improved_config.json` を作成してください：

```json
{
  "model_name": "cl-tohoku/bert-base-japanese-v3",
  "max_length": 128,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 4,
  "weight_decay": 0.01,
  "warmup_steps": 100,
  "early_stopping_patience": 3,
  "notes": "初回学習の結果を踏まえて調整"
}
```

**調整のポイント:**

1. **learning_rate**
   - 精度が高かった → そのまま or 少し下げる（1e-5）
   - 精度が低かった → 上げる（3e-5, 5e-5）

2. **num_epochs**
   - 過学習していない → 4-5に増やす
   - 過学習気味 → そのままor Early Stopping

3. **batch_size**
   - メモリに余裕 → 32に増やす（学習安定化）
   - メモリ不足 → 8に減らす

---

## ✅ 完了確認

- [ ] `03_error_analysis.py` を作成・実行した
- [ ] `error_analysis_report.md` を作成した
- [ ] 誤分類パターンを3つ以上特定した
- [ ] 改善策を3つ以上立案した
- [ ] `improved_config.json` を作成した
- [ ] 次のコマで実施する施策を決定した

---

## ⏭️ 次のコマ（コマ5）への準備

以下をメモしてください：

1. **実施する改善策**: ___
2. **新しいパラメータ**: ___
3. **目標精度**: Accuracy ___% 以上










