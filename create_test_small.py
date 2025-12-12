"""
テスト用小規模データセット作成スクリプト
学習スクリプトの動作確認用
"""

import json

def create_small_dataset(input_file, output_file, n=20):
    """
    データセットから先頭n件を抽出して小規模データセットを作成
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"読み込み: {input_file} ({len(data)}件)")
    print(f"抽出: 先頭{n}件 → {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data[:n]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 作成完了\n")

if __name__ == '__main__':
    print("="*60)
    print("小規模テストデータセット作成")
    print("="*60)
    print()
    
    create_small_dataset('train.jsonl', 'train_small.jsonl', 40)
    create_small_dataset('val.jsonl', 'val_small.jsonl', 10)
    create_small_dataset('test.jsonl', 'test_small.jsonl', 10)
    
    print("="*60)
    print("✅ すべて完了！")
    print("="*60)
    print()
    print("次のステップ:")
    print("1. 02_train_bert.py のファイル名を編集")
    print("   train.jsonl → train_small.jsonl")
    print("   val.jsonl → val_small.jsonl")
    print("   test.jsonl → test_small.jsonl")
    print()
    print("2. テスト実行")
    print("   python 02_train_bert.py")


