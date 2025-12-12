import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import BertJapaneseTokenizer
import os
import matplotlib.font_manager as fm

# 日本語フォントを明示的に設定
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Seabornのスタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# figuresフォルダの作成
os.makedirs('figures', exist_ok=True)

print("="*60)
print("データセット探索的データ分析（EDA）")
print("="*60)

# 現在のフォント確認
print(f"\n現在のフォント設定: {plt.rcParams['font.sans-serif']}")
print(f"フォントファミリー: {plt.rcParams['font.family']}")

# データ読み込み関数
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ラベル分布の可視化
def plot_label_distribution(data_dict):
    print("\n[ラベル分布の可視化]")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, data) in enumerate(data_dict.items()):
        labels = [item['label'] for item in data]
        label_counts = Counter(labels)
        
        ax = axes[idx]
        colors = ['#4CAF50', '#FF9800']
        bars = ax.bar(['明確 (Label 0)', '曖昧 (Label 1)'], 
                      [label_counts[0], label_counts[1]],
                      color=colors, alpha=0.8)
        
        ax.set_ylabel('サンプル数', fontsize=12)
        ax.set_title(f'{name.capitalize()} データセット\n(総数: {len(data)}件)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, [label_counts[0], label_counts[1]]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/len(data)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/label_distribution_jp.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: figures/label_distribution_jp.png")
    plt.close()

# 頻出語分析（日本語対応）
def analyze_frequent_words(data_dict):
    print("\n[頻出語分析]")
    all_data = []
    for data in data_dict.values():
        all_data.extend(data)
    
    frequent_words = {0: [], 1: []}
    
    for label in [0, 1]:
        label_texts = [item['text'] for item in all_data if item['label'] == label]
        all_text = ' '.join(label_texts)
        
        words = [word for word in all_text if len(word) > 1 and word not in ['、', '。', 'て', 'に', 'を', 'の', 'は', 'が']]
        word_counts = Counter(words)
        frequent_words[label] = word_counts.most_common(10)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#4CAF50', '#FF9800']
    labels_names = ['明確 (Label 0)', '曖昧 (Label 1)']
    
    for idx, label in enumerate([0, 1]):
        ax = axes[idx]
        words_data = frequent_words[label]
        
        if words_data:
            words = [w[0] for w in words_data[:10]]
            counts = [w[1] for w in words_data[:10]]
            
            bars = ax.barh(range(len(words)), counts, color=colors[idx], alpha=0.8)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=11)
            ax.set_xlabel('出現回数', fontsize=12)
            ax.set_title(f'{labels_names[idx]} 頻出語 TOP10', 
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/frequent_words_jp.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ 保存: figures/frequent_words_jp.png")
    plt.close()
    
    return frequent_words

# メイン処理
if __name__ == '__main__':
    # データ読み込み
    print("\n[データ読み込み]")
    data_dict = {
        'train': load_jsonl('data/train.jsonl'),
        'val': load_jsonl('data/val.jsonl'),
        'test': load_jsonl('data/test.jsonl')
    }
    
    for name, data in data_dict.items():
        print(f"  {name}: {len(data)}件")
    
    # 分析実行
    plot_label_distribution(data_dict)
    frequent_words = analyze_frequent_words(data_dict)
    
    print("\n" + "="*60)
    print("✅ 日本語対応グラフ生成完了！")
    print("="*60)
    print("\n成果物:")
    print("  - figures/label_distribution_jp.png")
    print("  - figures/frequent_words_jp.png")










