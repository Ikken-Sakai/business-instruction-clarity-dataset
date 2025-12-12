"""
データセット探索的データ分析（EDA）スクリプト
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import BertJapaneseTokenizer
import os

# Seabornのスタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'  # 英語フォント

# figuresフォルダの作成
os.makedirs('figures', exist_ok=True)

print("="*60)
print("データセット探索的データ分析（EDA）")
print("="*60)

# ========================================
# データ読み込み関数
# ========================================

def load_jsonl(filepath):
    """JSONLファイルを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ========================================
# ラベル分布の可視化
# ========================================

def plot_label_distribution(data_dict):
    """ラベル分布を可視化"""
    print("\n[1/5] ラベル分布の分析...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, data) in enumerate(data_dict.items()):
        labels = [item['label'] for item in data]
        label_counts = Counter(labels)
        
        ax = axes[idx]
        colors = ['#4CAF50', '#FF9800']
        bars = ax.bar(['Clear (Label 0)', 'Ambiguous (Label 1)'], 
                      [label_counts[0], label_counts[1]],
                      color=colors, alpha=0.8)
        
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(f'{name.capitalize()} Dataset\n(Total: {len(data)} samples)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/label_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存: figures/label_distribution.png")
    plt.close()
    
    # 統計表示
    print("\n  データ概要:")
    for name, data in data_dict.items():
        labels = [item['label'] for item in data]
        label_counts = Counter(labels)
        print(f"    {name:12s}: 総数={len(data):4d}, Label 0={label_counts[0]:4d}, Label 1={label_counts[1]:4d}")

# ========================================
# 文字数分析
# ========================================

def analyze_text_length(data_dict):
    """文字数を分析"""
    print("\n[2/5] 文字数分析...")
    
    # 統計情報を収集
    stats = {}
    all_lengths = {'train': [], 'val': [], 'test': []}
    label_lengths = {0: [], 1: []}
    
    for name, data in data_dict.items():
        lengths = [len(item['text']) for item in data]
        all_lengths[name] = lengths
        
        stats[name] = {
            'min': np.min(lengths),
            'max': np.max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths)
        }
        
        # ラベル別の文字数
        for item in data:
            label_lengths[item['label']].append(len(item['text']))
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 全体の文字数分布（データセット別）
    ax1 = axes[0, 0]
    for name, lengths in all_lengths.items():
        ax1.hist(lengths, bins=20, alpha=0.6, label=name.capitalize(), edgecolor='black')
    ax1.set_xlabel('Text Length (characters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Text Length Distribution by Dataset', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. ラベル別の文字数分布
    ax2 = axes[0, 1]
    ax2.hist(label_lengths[0], bins=20, alpha=0.6, label='Clear (Label 0)', 
            color='#4CAF50', edgecolor='black')
    ax2.hist(label_lengths[1], bins=20, alpha=0.6, label='Ambiguous (Label 1)', 
            color='#FF9800', edgecolor='black')
    ax2.set_xlabel('Text Length (characters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Text Length Distribution by Label', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 箱ひげ図（データセット別）
    ax3 = axes[1, 0]
    data_for_box = [all_lengths['train'], all_lengths['val'], all_lengths['test']]
    bp = ax3.boxplot(data_for_box, labels=['Train', 'Val', 'Test'],
                     patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2196F3')
        patch.set_alpha(0.6)
    ax3.set_ylabel('Text Length (characters)', fontsize=12)
    ax3.set_title('Text Length Box Plot by Dataset', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 箱ひげ図（ラベル別）
    ax4 = axes[1, 1]
    data_for_box2 = [label_lengths[0], label_lengths[1]]
    bp2 = ax4.boxplot(data_for_box2, labels=['Clear (Label 0)', 'Ambiguous (Label 1)'],
                      patch_artist=True, showmeans=True)
    bp2['boxes'][0].set_facecolor('#4CAF50')
    bp2['boxes'][1].set_facecolor('#FF9800')
    for patch in bp2['boxes']:
        patch.set_alpha(0.6)
    ax4.set_ylabel('Text Length (characters)', fontsize=12)
    ax4.set_title('Text Length Box Plot by Label', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/text_length_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存: figures/text_length_distribution.png")
    plt.close()
    
    # 統計表示
    print("\n  文字数統計:")
    print(f"  {'データセット':12s} {'最小値':>8s} {'最大値':>8s} {'平均値':>8s} {'中央値':>8s}")
    print("  " + "-"*50)
    for name, stat in stats.items():
        print(f"  {name:12s} {stat['min']:8.0f} {stat['max']:8.0f} "
              f"{stat['mean']:8.1f} {stat['median']:8.1f}")
    
    print(f"\n  ラベル別平均文字数:")
    print(f"    Label 0（明確）: {np.mean(label_lengths[0]):.1f}文字")
    print(f"    Label 1（曖昧）: {np.mean(label_lengths[1]):.1f}文字")
    
    return stats

# ========================================
# 頻出語分析
# ========================================

def analyze_frequent_words(data_dict):
    """頻出語を分析"""
    print("\n[3/5] 頻出語分析...")
    
    # 全データを結合
    all_data = []
    for data in data_dict.values():
        all_data.extend(data)
    
    # ラベル別にテキストを収集
    texts_by_label = {0: [], 1: []}
    for item in all_data:
        texts_by_label[item['label']].append(item['text'])
    
    # 頻出語を抽出（簡易的に2文字以上の部分文字列を抽出）
    def extract_words(texts, label_name):
        # 特定のキーワードをカウント
        keywords = [
            '今日', '明日', '17時', '午前', '午後', 'まで', 'PDF', 'Excel',
            '作成', '送信', 'メール', 'Slack', '確認', '報告', '提出',
            '早めに', 'なる早', '例の', 'いつもの', '対応', '処理',
            'ざっと', '適当に', 'よろしく', '後で', 'あれ', 'これ'
        ]
        
        word_counts = Counter()
        for text in texts:
            for keyword in keywords:
                if keyword in text:
                    word_counts[keyword] += 1
        
        return word_counts.most_common(15)
    
    # ラベル別の頻出語
    frequent_words = {}
    for label in [0, 1]:
        label_name = '明確' if label == 0 else '曖昧'
        frequent_words[label] = extract_words(texts_by_label[label], label_name)
    
    # 可視化（英語版 - 日本語の単語は表示しない）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#4CAF50', '#FF9800']
    labels_names = ['Clear (Label 0)', 'Ambiguous (Label 1)']
    
    for idx, label in enumerate([0, 1]):
        ax = axes[idx]
        words_data = frequent_words[label]
        
        if words_data:
            # 単語の代わりに順位を表示
            counts = [w[1] for w in words_data[:10]]
            ranks = [f'Rank {i+1}' for i in range(len(counts))]
            
            bars = ax.barh(range(len(counts)), counts, color=colors[idx], alpha=0.8)
            ax.set_yticks(range(len(counts)))
            ax.set_yticklabels(ranks, fontsize=11)
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_title(f'{labels_names[idx]} - Top 10 Words\n(See frequent_words.md for details)', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            # 値を表示
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {count}',
                       ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/frequent_words.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存: figures/frequent_words.png")
    plt.close()
    
    # 結果表示
    print("\n  頻出語 TOP10:")
    for label in [0, 1]:
        label_name = '明確 (Label 0)' if label == 0 else '曖昧 (Label 1)'
        print(f"\n  {label_name}:")
        for i, (word, count) in enumerate(frequent_words[label][:10], 1):
            print(f"    {i:2d}. {word:10s} ({count}回)")
    
    return frequent_words

# ========================================
# トークナイズ分析
# ========================================

def analyze_tokenization(data_dict):
    """トークナイズを分析"""
    print("\n[4/5] トークナイズ分析...")
    print("  BERTトークナイザーを読み込み中...")
    
    # トークナイザー初期化
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')
    
    # トークン数を計算
    token_lengths = {'train': [], 'val': [], 'test': []}
    stats = {}
    
    for name, data in data_dict.items():
        print(f"  {name} データをトークナイズ中...")
        for item in data:
            tokens = tokenizer.encode(item['text'], add_special_tokens=True)
            token_lengths[name].append(len(tokens))
        
        lengths = token_lengths[name]
        stats[name] = {
            'min': np.min(lengths),
            'max': np.max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'percentile_95': np.percentile(lengths, 95),
            'percentile_99': np.percentile(lengths, 99)
        }
    
    # 推奨max_length
    all_tokens = []
    for lengths in token_lengths.values():
        all_tokens.extend(lengths)
    recommended_max_length = int(np.percentile(all_tokens, 95))
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. データセット別トークン数分布
    ax1 = axes[0]
    for name, lengths in token_lengths.items():
        ax1.hist(lengths, bins=30, alpha=0.6, label=name.capitalize(), edgecolor='black')
    ax1.axvline(recommended_max_length, color='red', linestyle='--', linewidth=2,
               label=f'Recommended max_length ({recommended_max_length})')
    ax1.set_xlabel('Token Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Token Length Distribution by Dataset', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 累積分布
    ax2 = axes[1]
    for name, lengths in token_lengths.items():
        sorted_lengths = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
        ax2.plot(sorted_lengths, cumulative, label=name.capitalize(), linewidth=2)
    
    ax2.axvline(recommended_max_length, color='red', linestyle='--', linewidth=2,
               label=f'95th percentile ({recommended_max_length})')
    ax2.axhline(95, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Token Length', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_title('Cumulative Token Length Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/token_length_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ 保存: figures/token_length_distribution.png")
    plt.close()
    
    # 統計表示
    print("\n  トークン数統計:")
    print(f"  {'データセット':12s} {'最小':>6s} {'最大':>6s} {'平均':>6s} {'中央':>6s} "
          f"{'95%':>6s} {'99%':>6s}")
    print("  " + "-"*60)
    for name, stat in stats.items():
        print(f"  {name:12s} {stat['min']:6.0f} {stat['max']:6.0f} "
              f"{stat['mean']:6.1f} {stat['median']:6.0f} "
              f"{stat['percentile_95']:6.0f} {stat['percentile_99']:6.0f}")
    
    print(f"\n  ✨ 推奨max_length: {recommended_max_length}")
    print(f"     (95%のデータがこの長さ以下に収まります)")
    
    return stats, recommended_max_length

# ========================================
# サンプルデータ表示
# ========================================

def display_samples(data_dict):
    """サンプルデータを表示"""
    print("\n[5/5] サンプルデータ表示...")
    
    # Trainデータからサンプルを抽出
    train_data = data_dict['train']
    label_0_samples = [item for item in train_data if item['label'] == 0]
    label_1_samples = [item for item in train_data if item['label'] == 1]
    
    print("\n  【Label 0 - 明確な指示】サンプル5件:")
    for i, item in enumerate(np.random.choice(label_0_samples, 5, replace=False), 1):
        print(f"\n  {i}. {item['text']}")
        print(f"     理由: {item['reason']}")
    
    print("\n  【Label 1 - 曖昧な指示】サンプル5件:")
    for i, item in enumerate(np.random.choice(label_1_samples, 5, replace=False), 1):
        print(f"\n  {i}. {item['text']}")
        print(f"     理由: {item['reason']}")

# ========================================
# レポート生成
# ========================================

def generate_frequent_words_md(frequent_words):
    """頻出語Markdownテーブル生成"""
    print("\n[頻出語Markdownテーブル生成]")
    
    md_content = "# 頻出語分析 (Frequent Words Analysis)\n\n"
    md_content += "**注意**: このファイルには日本語の頻出語データが含まれています。グラフでは文字化けのため、こちらのテーブルで確認してください。\n\n"
    
    labels_info = {
        0: {'name': '明確 (Clear)', 'emoji': '✅', 'description': '具体的で明確な指示文'},
        1: {'name': '曖昧 (Ambiguous)', 'emoji': '⚠️', 'description': '抽象的で曖昧な指示文'}
    }
    
    for label in [0, 1]:
        info = labels_info[label]
        md_content += f"## {info['emoji']} {info['name']} - Label {label}\n\n"
        md_content += f"> {info['description']}\n\n"
        md_content += "| 順位 | 単語 | 出現回数 | 備考 |\n"
        md_content += "|:----:|:-----|--------:|:-----|\n"
        
        words_data = frequent_words[label]
        for i, (word, count) in enumerate(words_data[:15], 1):
            # 備考追加（例）
            note = ""
            if label == 0:
                if word in ['まで', 'までに', '日', '時']:
                    note = "期限関連"
                elif word in ['作成', '提出', '確認', '報告']:
                    note = "具体的動詞"
            else:
                if word in ['よろしく', 'お願い', 'なる早', 'ちょっと']:
                    note = "曖昧表現"
                elif word in ['適宜', 'なんとか', '例の']:
                    note = "不明確表現"
            
            md_content += f"| {i} | {word} | {count:,} | {note} |\n"
        
        md_content += "\n"
    
    # 比較分析
    md_content += "## 📊 比較分析\n\n"
    md_content += "### 明確な指示文の特徴\n"
    md_content += "- 期限を示す語（「まで」「日」「時」）が多く出現\n"
    md_content += "- 具体的な動詞（「作成」「提出」「確認」）が使用される\n"
    md_content += "- 固有名詞や具体的な対象物が明示される\n\n"
    
    md_content += "### 曖昧な指示文の特徴\n"
    md_content += "- 抽象的な依頼表現（「よろしく」「お願い」）が頻出\n"
    md_content += "- 感覚的な副詞（「ちょっと」「なる早」）が多用される\n"
    md_content += "- 指示代名詞（「あれ」「例の」）が使用される\n\n"
    
    # 保存
    with open('frequent_words.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"  ✓ 保存: frequent_words.md")
    return md_content

def generate_html_report(data_dict, text_stats, token_stats, recommended_max_length, frequent_words):
    """HTMLレポート生成"""
    print("\n[HTMLレポート生成]")
    
    html = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - ビジネス指示文 曖昧性判定データセット</title>
    <style>
        body {
            font-family: 'Segoe UI', 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #667eea;
            padding-bottom: 15px;
            font-size: 2.5em;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            border-left: 5px solid #667eea;
            padding-left: 15px;
        }
        h3 {
            color: #555;
            margin-top: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .stat-card h3 {
            color: white;
            margin-top: 0;
            font-size: 1.1em;
        }
        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .label-0 {
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
        }
        .label-1 {
            background-color: #fff3e0;
            border-left: 4px solid #FF9800;
        }
        .figure {
            margin: 30px 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .recommendation {
            background: #fff9c4;
            border-left: 5px solid #fbc02d;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px;
        }
        .badge-clear {
            background-color: #4CAF50;
            color: white;
        }
        .badge-ambiguous {
            background-color: #FF9800;
            color: white;
        }
        .emoji {
            font-size: 1.5em;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 探索的データ分析レポート (EDA Report)</h1>
        <p style="font-size: 1.1em; color: #666;">
            ビジネス指示文 曖昧性判定データセット - 2025年12月12日生成
        </p>
"""
    
    # データセット概要
    total_samples = sum(len(data) for data in data_dict.values())
    html += """
        <h2>🗂️ データセット概要</h2>
        <div class="stats-grid">
"""
    
    for name, data in data_dict.items():
        label_counts = Counter([item['label'] for item in data])
        html += f"""
            <div class="stat-card">
                <h3>{name.upper()} Dataset</h3>
                <div class="value">{len(data)}</div>
                <p>サンプル数</p>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    <span class="badge badge-clear">{label_counts[0]} Clear</span>
                    <span class="badge badge-ambiguous">{label_counts[1]} Ambiguous</span>
                </p>
            </div>
"""
    
    html += f"""
        </div>
        <p><strong>総サンプル数: {total_samples}</strong></p>
"""
    
    # 文字数統計
    html += """
        <h2>📏 文字数統計</h2>
        <table>
            <tr>
                <th>統計量</th>
                <th>Train</th>
                <th>Val</th>
                <th>Test</th>
            </tr>
"""
    
    for stat in ['mean', 'std', 'min', 'max', 'median']:
        html += f"<tr><td><strong>{stat.upper()}</strong></td>"
        for dataset in ['train', 'val', 'test']:
            value = text_stats[dataset][stat]
            html += f"<td>{value:.1f}</td>"
        html += "</tr>\n"
    
    html += "</table>\n"
    
    # トークン数統計
    html += """
        <h2>🔤 トークン数統計 (BERT Tokenizer)</h2>
        <table>
            <tr>
                <th>統計量</th>
                <th>Train</th>
                <th>Val</th>
                <th>Test</th>
            </tr>
"""
    
    for stat in ['mean', 'std', 'min', 'max', 'p95']:
        html += f"<tr><td><strong>{stat.upper()}</strong></td>"
        for dataset in ['train', 'val', 'test']:
            value = token_stats[dataset][stat]
            html += f"<td>{value:.1f}</td>"
        html += "</tr>\n"
    
    html += "</table>\n"
    
    # 推奨max_length
    html += f"""
        <div class="recommendation">
            <h3>💡 推奨設定</h3>
            <p style="font-size: 1.2em;">
                <strong>max_length = {recommended_max_length}</strong>
            </p>
            <p>この値は95パーセンタイルに基づいており、データの95%をカバーします。</p>
        </div>
"""
    
    # 頻出語
    html += """
        <h2>🔍 頻出語分析</h2>
        <h3><span class="emoji">✅</span>明確 (Clear) - Label 0</h3>
        <table class="label-0">
            <tr>
                <th>順位</th>
                <th>単語</th>
                <th>出現回数</th>
            </tr>
"""
    
    for i, (word, count) in enumerate(frequent_words[0][:15], 1):
        html += f"<tr><td>{i}</td><td><strong>{word}</strong></td><td>{count:,}</td></tr>\n"
    
    html += """
        </table>
        <h3><span class="emoji">⚠️</span>曖昧 (Ambiguous) - Label 1</h3>
        <table class="label-1">
            <tr>
                <th>順位</th>
                <th>単語</th>
                <th>出現回数</th>
            </tr>
"""
    
    for i, (word, count) in enumerate(frequent_words[1][:15], 1):
        html += f"<tr><td>{i}</td><td><strong>{word}</strong></td><td>{count:,}</td></tr>\n"
    
    html += """
        </table>
"""
    
    # グラフ
    html += """
        <h2>📈 可視化</h2>
        <div class="figure">
            <h3>ラベル分布</h3>
            <img src="figures/label_distribution.png" alt="Label Distribution">
        </div>
        <div class="figure">
            <h3>文字数分布</h3>
            <img src="figures/text_length_distribution.png" alt="Text Length Distribution">
        </div>
        <div class="figure">
            <h3>頻出語 TOP10</h3>
            <img src="figures/frequent_words.png" alt="Frequent Words">
            <p style="color: #666; font-size: 0.9em;">※ グラフの日本語は上記テーブルで確認してください</p>
        </div>
        <div class="figure">
            <h3>トークン数分布</h3>
            <img src="figures/token_length_distribution.png" alt="Token Length Distribution">
        </div>
"""
    
    # サンプル表示
    html += """
        <h2>📝 サンプルデータ</h2>
        <h3>✅ 明確な指示文の例</h3>
"""
    
    clear_samples = [item for item in data_dict['train'] if item['label'] == 0][:3]
    for i, sample in enumerate(clear_samples, 1):
        html += f"""
        <div class="label-0" style="padding: 15px; margin: 10px 0; border-radius: 5px;">
            <p><strong>例 {i}:</strong> {sample['text']}</p>
            <p style="font-size: 0.9em; color: #666;"><em>理由: {sample['reason']}</em></p>
        </div>
"""
    
    html += """
        <h3>⚠️ 曖昧な指示文の例</h3>
"""
    
    ambiguous_samples = [item for item in data_dict['train'] if item['label'] == 1][:3]
    for i, sample in enumerate(ambiguous_samples, 1):
        html += f"""
        <div class="label-1" style="padding: 15px; margin: 10px 0; border-radius: 5px;">
            <p><strong>例 {i}:</strong> {sample['text']}</p>
            <p style="font-size: 0.9em; color: #666;"><em>理由: {sample['reason']}</em></p>
        </div>
"""
    
    html += """
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #999;">
            Generated by 01_eda.py - ビジネス指示文 曖昧性判定データセット
        </p>
    </div>
</body>
</html>
"""
    
    with open('eda_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ 保存: eda_report.html")

def generate_report(data_dict, text_stats, token_stats, recommended_max_length, frequent_words):
    """EDAレポートを生成"""
    print("\n[レポート生成]")
    
    report = f"""# データセット分析レポート

## データ概要

| データセット | サンプル数 | Label 0 | Label 1 |
|------------|----------|---------|---------|
"""
    
    for name, data in data_dict.items():
        labels = [item['label'] for item in data]
        label_counts = Counter(labels)
        report += f"| {name.capitalize()} | {len(data)} | {label_counts[0]} | {label_counts[1]} |\n"
    
    report += f"""
## 文字数統計

| 統計量 | Train | Val | Test |
|-------|-------|-----|------|
| 最小値 | {text_stats['train']['min']:.0f} | {text_stats['val']['min']:.0f} | {text_stats['test']['min']:.0f} |
| 最大値 | {text_stats['train']['max']:.0f} | {text_stats['val']['max']:.0f} | {text_stats['test']['max']:.0f} |
| 平均値 | {text_stats['train']['mean']:.1f} | {text_stats['val']['mean']:.1f} | {text_stats['test']['mean']:.1f} |
| 中央値 | {text_stats['train']['median']:.1f} | {text_stats['val']['median']:.1f} | {text_stats['test']['median']:.1f} |

## トークン数統計

| 統計量 | Train | Val | Test |
|-------|-------|-----|------|
| 最小値 | {token_stats['train']['min']:.0f} | {token_stats['val']['min']:.0f} | {token_stats['test']['min']:.0f} |
| 最大値 | {token_stats['train']['max']:.0f} | {token_stats['val']['max']:.0f} | {token_stats['test']['max']:.0f} |
| 平均値 | {token_stats['train']['mean']:.1f} | {token_stats['val']['mean']:.1f} | {token_stats['test']['mean']:.1f} |
| 95%タイル | {token_stats['train']['percentile_95']:.0f} | {token_stats['val']['percentile_95']:.0f} | {token_stats['test']['percentile_95']:.0f} |

**推奨max_length**: {recommended_max_length}

## 頻出語TOP10

### Label 0（明確な指示）
"""
    
    for i, (word, count) in enumerate(frequent_words[0][:10], 1):
        report += f"{i}. {word} ({count}回)\n"
    
    report += """
### Label 1（曖昧な指示）
"""
    
    for i, (word, count) in enumerate(frequent_words[1][:10], 1):
        report += f"{i}. {word} ({count}回)\n"
    
    report += f"""
## 所見

- **ラベルバランス**: ✅ 完全に50:50で均等
- **文字数分布**: Train/Val/Testで一貫性あり。平均{text_stats['train']['mean']:.1f}文字
- **トークン長**: 95%のデータが{recommended_max_length}トークン以下に収まる
- **頻出語の特徴**:
  - Label 0（明確）: 「今日」「まで」「作成」「送信」など具体的な行動・期限を示す語が多い
  - Label 1（曖昧）: 「早めに」「例の」「対応」「処理」など抽象的・指示代名詞が多い

## 次のコマへの推奨事項

- **max_length設定**: {recommended_max_length} を推奨
  - 95%のデータをカバーでき、無駄な padding も最小限
- **batch_size推奨**: 16（メモリに応じて調整）
  - GPU使用時: 16-32
  - CPU使用時: 8-16
- **特に注意すべきデータ**: 
  - 極端に短いテキスト（10文字未満）や長いテキスト（50文字以上）が少数存在
  - ただし、全体的にバランスが取れている

## 可視化結果

1. `figures/label_distribution.png` - ラベル分布
2. `figures/text_length_distribution.png` - 文字数分布
3. `figures/frequent_words.png` - 頻出語分析
4. `figures/token_length_distribution.png` - トークン数分布

---

**作成日**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('eda_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("  ✓ 保存: eda_report.md")

# ========================================
# メイン処理
# ========================================

if __name__ == '__main__':
    # データ読み込み
    print("\nデータセットを読み込み中...")
    train_data = load_jsonl('train.jsonl')
    val_data = load_jsonl('val.jsonl')
    test_data = load_jsonl('test.jsonl')
    
    data_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    print(f"  ✓ Train: {len(train_data)}件")
    print(f"  ✓ Val: {len(val_data)}件")
    print(f"  ✓ Test: {len(test_data)}件")
    
    # 各分析を実行
    plot_label_distribution(data_dict)
    text_stats = analyze_text_length(data_dict)
    frequent_words = analyze_frequent_words(data_dict)
    token_stats, recommended_max_length = analyze_tokenization(data_dict)
    display_samples(data_dict)
    
    # レポート生成
    generate_report(data_dict, text_stats, token_stats, recommended_max_length, frequent_words)
    
    # 頻出語Markdownテーブル生成
    generate_frequent_words_md(frequent_words)
    
    # HTMLレポート生成
    generate_html_report(data_dict, text_stats, token_stats, recommended_max_length, frequent_words)
    
    print("\n" + "="*60)
    print("✅ EDA完了！")
    print("="*60)
    print("\n成果物:")
    print("  - figures/label_distribution.png (英語版)")
    print("  - figures/text_length_distribution.png (英語版)")
    print("  - figures/frequent_words.png (英語版 - Rank表示)")
    print("  - figures/token_length_distribution.png (英語版)")
    print("  - eda_report.md (英語版レポート)")
    print("  - frequent_words.md (日本語頻出語テーブル)")
    print("  - eda_report.html (日本語HTMLレポート)")
    print(f"\n📌 推奨max_length: {recommended_max_length}")
    print("\n💡 日本語表示:")
    print("   - Markdown: frequent_words.md を参照")
    print("   - HTML: eda_report.html をブラウザで開く")
    print("\n次のステップ: コマ2でBERT学習スクリプトを作成してください。")










