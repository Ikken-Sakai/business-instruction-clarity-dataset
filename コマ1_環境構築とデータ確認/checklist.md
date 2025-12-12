# コマ1 完了チェックリスト

## 📝 タスク完了確認

### 環境構築
- [ ] `requirements.txt` を作成した
- [ ] `pip install -r requirements.txt` を実行し、すべてのライブラリがインストールされた
- [ ] `check_gpu.py` を実行し、GPU/CPU状況を確認した
- [ ] PyTorchが正常にインポートできることを確認した
- [ ] transformersライブラリが正常にインポートできることを確認した

### データ読み込み
- [ ] `train.jsonl` が正常に読み込める
- [ ] `val.jsonl` が正常に読み込める
- [ ] `test.jsonl` が正常に読み込める
- [ ] 各ファイルのデータ数を確認した（train:1600, val:200, test:200）
- [ ] データ構造（text, label, reason）を確認した

### EDAスクリプト
- [ ] `01_eda.py` を作成した
- [ ] スクリプトがエラーなく実行できる
- [ ] ラベル分布のグラフが生成される
- [ ] 文字数分布のグラフが生成される
- [ ] 頻出語分析のグラフが生成される
- [ ] トークン数分布のグラフが生成される
- [ ] すべてのグラフが `figures/` フォルダに保存される

### 分析結果
- [ ] `eda_report.md` が生成された
- [ ] ラベルバランスが50:50であることを確認した
- [ ] 平均文字数を把握した
- [ ] 推奨max_lengthを決定した（_____に決定）
- [ ] Label 0とLabel 1の特徴的な違いを理解した

### 可視化結果
- [ ] `figures/label_distribution.png` が存在する
- [ ] `figures/text_length_distribution.png` が存在する
- [ ] `figures/token_length_distribution.png` が存在する
- [ ] `figures/frequent_words_label0.png` が存在する
- [ ] `figures/frequent_words_label1.png` が存在する

## 🎯 理解度確認

以下の質問に答えられますか？

1. **データセットの特徴**
   - [ ] 各データセットのサンプル数を言える
   - [ ] ラベル0とラベル1の定義を説明できる
   - [ ] 平均文字数を言える

2. **分析結果の解釈**
   - [ ] Label 0（明確）とLabel 1（曖昧）で頻出する語の違いを説明できる
   - [ ] トークン数の分布を把握している
   - [ ] max_lengthを128にする理由を説明できる

3. **次のステップの準備**
   - [ ] BERT学習で使用するトークナイザーを知っている（cl-tohoku/bert-base-japanese-v3）
   - [ ] batch_sizeをいくつから試すか決めている
   - [ ] GPU/CPUどちらで学習するか決めている

## 📊 成果物確認

以下のファイルが存在しますか？

```
人工知能応用データセット/
├── requirements.txt          ✅
├── check_gpu.py              ✅
├── 01_eda.py                 ✅
├── eda_report.md             ✅
└── figures/
    ├── label_distribution.png           ✅
    ├── text_length_distribution.png     ✅
    ├── token_length_distribution.png    ✅
    ├── frequent_words_label0.png        ✅
    └── frequent_words_label1.png        ✅
```

## ⏱️ 時間確認

- 開始時刻: _____
- 終了時刻: _____
- 所要時間: _____ 分

**目標: 90分以内に完了**

## 🔄 GitHubへの更新

### コマ1完了時のコミット

すべてのタスクが完了したら、GitHubに変更をプッシュしましょう。

```bash
# 作業ディレクトリに移動
cd "/home/ike/Desktop/人工知能応用データセット"

# 変更ファイルを確認
git status

# 新規作成したファイルを追加
git add requirements.txt
git add check_gpu.py
git add 01_eda.py
git add eda_report.md
git add figures/

# コマ1のチェックリストも更新
git add "コマ1_環境構築とデータ確認/checklist.md"

# コミット
git commit -m "Complete コマ1: 環境構築とデータ確認

- requirements.txtを作成
- EDAスクリプトを実装
- データセット分析レポートを作成
- 可視化結果を生成
- 推奨max_length: ___ を決定"

# プッシュ
git push origin main
```

### 完了確認
- [ ] git status でコミットするファイルを確認した
- [ ] git add でファイルを追加した
- [ ] git commit でコミットした
- [ ] git push でプッシュした
- [ ] GitHubのリポジトリで反映を確認した

### トラブルシューティング

**エラー: "fatal: not a git repository"**
```bash
# Gitリポジトリを初期化
git init
git remote add origin <your-repository-url>
```

**エラー: "rejected - non-fast-forward"**
```bash
# リモートの変更を取得してマージ
git pull origin main --rebase
git push origin main
```

---

## ⏭️ 次のコマへ

すべてのチェックボックスが✅になり、GitHubへのプッシュも完了したら、**コマ2_BERT学習スクリプト作成** のフォルダに進んでください。

### 引き継ぎ事項メモ

- 推奨max_length: _____
- 平均トークン数: _____
- GPU使用: はい / いいえ
- 特記事項: _____
