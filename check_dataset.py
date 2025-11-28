#!/usr/bin/env python3
"""
データセット品質チェックスクリプト
"""

import json
import random
from collections import Counter

def check_dataset(filepath: str):
    """データセットの統計情報を表示"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # ラベル分布
    labels = [x['label'] for x in data]
    label_0_count = labels.count(0)
    label_1_count = labels.count(1)
    
    # 文字数統計
    lengths = [len(x['text']) for x in data]
    
    print(f"\n{'='*60}")
    print(f"ファイル: {filepath}")
    print(f"{'='*60}")
    print(f"総データ数: {len(data)} 件")
    print(f"  - Label 0 (明確): {label_0_count} 件 ({label_0_count/len(data)*100:.1f}%)")
    print(f"  - Label 1 (曖昧): {label_1_count} 件 ({label_1_count/len(data)*100:.1f}%)")
    print(f"\n文字数統計:")
    print(f"  - 最小: {min(lengths)} 文字")
    print(f"  - 最大: {max(lengths)} 文字")
    print(f"  - 平均: {sum(lengths)/len(lengths):.1f} 文字")
    
    # サンプル表示
    print(f"\n【Label 0 (明確) サンプル（ランダム3件）】")
    samples_0 = random.sample([x for x in data if x['label'] == 0], min(3, label_0_count))
    for item in samples_0:
        print(f"  - {item['text']}")
        print(f"    理由: {item['reason']}")
    
    print(f"\n【Label 1 (曖昧) サンプル（ランダム3件）】")
    samples_1 = random.sample([x for x in data if x['label'] == 1], min(3, label_1_count))
    for item in samples_1:
        print(f"  - {item['text']}")
        print(f"    理由: {item['reason']}")

def main():
    print("=" * 60)
    print("データセット品質チェック")
    print("=" * 60)
    
    files = ['dataset.jsonl', 'train.jsonl', 'val.jsonl', 'test.jsonl']
    
    for filename in files:
        try:
            check_dataset(filename)
        except FileNotFoundError:
            print(f"\n[警告] {filename} が見つかりません")
    
    print("\n" + "=" * 60)
    print("チェック完了")
    print("=" * 60)

if __name__ == '__main__':
    main()


