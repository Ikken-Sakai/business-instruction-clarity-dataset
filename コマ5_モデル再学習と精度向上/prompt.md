# コマ5実装プロンプト: モデル再学習と精度向上

このプロンプトに従って、改善パラメータでの再学習を実行してください。

---

## 🔄 再学習の準備

### ステップ1: 前回モデルのバックアップ

```bash
# 前回のモデルをバックアップ
mv saved_model saved_model_v1
mv training_results.json training_results_v1.json
```

---

### ステップ2: 学習スクリプトの更新

`02_train_bert.py` の CONFIG 部分を `improved_config.json` の内容で更新してください：

```python
# 02_train_bert.pyのCONFIG部分を更新

CONFIG = {
    'model_name': 'cl-tohoku/bert-base-japanese-v3',
    'max_length': 128,  # improved_config.jsonから
    'batch_size': 16,   # improved_config.jsonから
    'learning_rate': 2e-5,  # improved_config.jsonから
    'num_epochs': 4,    # improved_config.jsonから（調整後）
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'output_dir': './results_v2',
    'logging_dir': './logs_v2',
    'seed': 42
}
```

または、新しいスクリプト `04_train_improved.py` を作成してもOKです。

---

### ステップ3: 再学習の実行

```bash
python 02_train_bert.py
```

**監視ポイント**：
- [ ] Lossが初回学習より早く下がるか
- [ ] Validation Accuracyが初回より高いか
- [ ] 過学習していないか（Train vs Val の差）

---

## 📊 結果の比較分析

### 比較スクリプトの作成

`04_compare_models.py` を作成してください：

```python
"""
初回学習と再学習の結果を比較
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import pandas as pd

def load_results(filepath):
    """結果ファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_results():
    """2つのモデルの結果を比較"""
    
    print("="*60)
    print("モデル比較分析")
    print("="*60)
    
    # 結果読み込み
    results_v1 = load_results('training_results_v1.json')
    results_v2 = load_results('training_results.json')
    
    # Test結果の比較
    print("\n--- Test結果の比較 ---\n")
    
    metrics = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']
    
    comparison_data = []
    
    for metric in metrics:
        v1_value = results_v1['test_results'].get(metric, 0)
        v2_value = results_v2['test_results'].get(metric, 0)
        improvement = v2_value - v1_value
        improvement_pct = (improvement / v1_value * 100) if v1_value > 0 else 0
        
        comparison_data.append({
            'メトリクス': metric.replace('eval_', '').upper(),
            '初回学習': f"{v1_value:.4f}",
            '再学習': f"{v2_value:.4f}",
            '改善': f"{improvement:+.4f}",
            '改善率': f"{improvement_pct:+.2f}%"
        })
        
        print(f"{metric.replace('eval_', '').upper()}")
        print(f"  初回: {v1_value:.4f}")
        print(f"  再学習: {v2_value:.4f}")
        print(f"  改善: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        print()
    
    # DataFrameで表示
    df = pd.DataFrame(comparison_data)
    print("\n--- 比較表 ---")
    print(df.to_string(index=False))
    
    # 可視化
    plot_comparison(comparison_data)
    
    # レポート作成
    create_comparison_report(comparison_data, results_v1, results_v2)
    
    return comparison_data

def plot_comparison(comparison_data):
    """
    比較グラフを作成
    """
    metrics = [d['メトリクス'] for d in comparison_data]
    v1_scores = [float(d['初回学習']) for d in comparison_data]
    v2_scores = [float(d['再学習']) for d in comparison_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(metrics))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], v1_scores, width, label='初回学習', alpha=0.8)
    ax.bar([i + width/2 for i in x], v2_scores, width, label='再学習', alpha=0.8)
    
    ax.set_ylabel('スコア')
    ax.set_title('初回学習 vs 再学習の比較')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for i, (v1, v2) in enumerate(zip(v1_scores, v2_scores)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/comparison_charts.png', dpi=150)
    print("\n比較グラフを figures/comparison_charts.png に保存しました。")

def create_comparison_report(comparison_data, results_v1, results_v2):
    """
    比較レポートを作成
    """
    with open('comparison_report.md', 'w', encoding='utf-8') as f:
        f.write("# 初回学習 vs 再学習 比較レポート\n\n")
        
        f.write("## 結果比較\n\n")
        f.write("| メトリクス | 初回学習 | 再学習 | 改善 | 改善率 |\n")
        f.write("|----------|---------|-------|------|-------|\n")
        for data in comparison_data:
            f.write(f"| {data['メトリクス']} | {data['初回学習']} | {data['再学習']} | {data['改善']} | {data['改善率']} |\n")
        
        f.write("\n## 学習設定の違い\n\n")
        f.write("### 初回学習\n")
        f.write(f"```json\n{json.dumps(results_v1['config'], indent=2, ensure_ascii=False)}\n```\n\n")
        
        f.write("### 再学習\n")
        f.write(f"```json\n{json.dumps(results_v2['config'], indent=2, ensure_ascii=False)}\n```\n\n")
        
        f.write("## 考察\n\n")
        
        # 自動判定
        acc_v1 = results_v1['test_results']['eval_accuracy']
        acc_v2 = results_v2['test_results']['eval_accuracy']
        improvement = acc_v2 - acc_v1
        
        if improvement > 0.05:
            f.write("✅ **大幅な改善が見られました！**\n\n")
        elif improvement > 0.02:
            f.write("✅ **改善が見られました。**\n\n")
        elif improvement > 0:
            f.write("😐 **わずかな改善が見られました。**\n\n")
        else:
            f.write("⚠️ **改善が見られませんでした。さらなる調整が必要です。**\n\n")
        
        f.write(f"- Accuracy改善: {improvement:+.4f} ({improvement/acc_v1*100:+.2f}%)\n")
        f.write(f"- 最終Accuracy: {acc_v2:.4f} ({acc_v2*100:.2f}%)\n\n")
        
        f.write("## 次のステップ\n\n")
        f.write("- デモアプリの開発\n")
        f.write("- プレ発表資料の作成\n")
    
    print("比較レポートを comparison_report.md に保存しました。")

if __name__ == '__main__':
    compare_results()
```

実行：
```bash
python 04_compare_models.py
```

---

## 🎮 デモ用サンプルの準備

`demo_samples.json` を作成してください：

```python
# create_demo_samples.py
"""
デモ用のサンプルテキストを準備
"""

import json

demo_samples = [
    # 明確な指示の例
    {
        "text": "今日の17時までに、A社向けの見積書をPDFで作成してSlackにアップしてください。",
        "expected_label": 0,
        "label_name": "明確",
        "reason": "期限・内容・形式・場所がすべて具体的"
    },
    {
        "text": "10月の月次売上レポートを、明日の午前中までにExcelで山田さんにメール送信してください。",
        "expected_label": 0,
        "label_name": "明確",
        "reason": "対象・期限・形式・宛先・手段が明確"
    },
    {
        "text": "顧客リストから休眠顧客30件を抽出して、今週金曜までにCSVファイルで共有してください。",
        "expected_label": 0,
        "label_name": "明確",
        "reason": "作業内容・件数・期限・形式が具体的"
    },
    
    # 曖昧な指示の例
    {
        "text": "例の件、早めに対応しといて。",
        "expected_label": 1,
        "label_name": "曖昧",
        "reason": "対象が不明（例の件）、期限が曖昧（早めに）"
    },
    {
        "text": "明日の会議資料、ざっと目を通しておいて、変なところあったら直して。",
        "expected_label": 1,
        "label_name": "曖昧",
        "reason": "程度が感覚的（ざっと、変なところ）、期限不明"
    },
    {
        "text": "いつものフォーマットで、なる早でお願い。",
        "expected_label": 1,
        "label_name": "曖昧",
        "reason": "暗黙知（いつもの）、期限が曖昧（なる早）"
    }
]

with open('demo_samples.json', 'w', encoding='utf-8') as f:
    json.dump(demo_samples, f, indent=2, ensure_ascii=False)

print("デモ用サンプルを demo_samples.json に保存しました。")
```

実行：
```bash
python create_demo_samples.py
```

---

## 📝 最終結果のまとめ

以下の情報を整理してください：

### プレ発表で使う数値

1. **データセット**
   - 総数: 2,000件
   - Train: 1,600件 / Val: 200件 / Test: 200件
   - ラベル比: 50:50

2. **モデル**
   - ベースモデル: cl-tohoku/bert-base-japanese-v3
   - タスク: 二値分類

3. **結果**
   - Test Accuracy: ___%
   - F1スコア: ___
   - Precision: ___
   - Recall: ___

4. **改善の過程**
   - 初回学習: Accuracy ___%
   - 再学習: Accuracy ___%
   - 改善幅: +___%

---

## ✅ 完了確認

- [ ] 再学習が正常に完了した
- [ ] `saved_model/` が作成された
- [ ] `04_compare_models.py` を実行した
- [ ] `comparison_report.md` が生成された
- [ ] 比較グラフが作成された
- [ ] `demo_samples.json` を準備した
- [ ] プレ発表用の数値を整理した

---

## ⏭️ 次のコマ（コマ6）への準備

以下を確認してください：

1. **最終精度**: Accuracy ___% を達成
2. **モデル保存場所**: `./saved_model/`
3. **デモ用サンプル**: `demo_samples.json`
4. **次のタスク**: デモアプリの開発

これらの情報を持って、コマ6でデモアプリを開発します。










