# コマ2 完了チェックリスト

## 📝 タスク完了確認

### スクリプト実装
- [ ] `02_train_bert.py` を作成した
- [ ] `load_jsonl()` 関数を実装した
- [ ] `create_dataset()` 関数を実装した
- [ ] `tokenize_function()` 関数を実装した
- [ ] `compute_metrics()` 関数を実装した
- [ ] `plot_confusion_matrix()` 関数を実装した
- [ ] `main()` 関数を実装した

### テスト実行
- [ ] 小規模データ（train_small.jsonl等）を作成した
- [ ] テスト実行が成功した
- [ ] エラーが発生した場合は修正した
- [ ] データフローを確認した（読み込み→トークナイズ→学習）

### パラメータ設定
- [ ] max_lengthをコマ1の結果から設定した（推奨: _____）
- [ ] batch_sizeを環境に応じて設定した（GPU: 16, CPU: 8）
- [ ] learning_rateを設定した（推奨: 2e-5）
- [ ] num_epochsを設定した（推奨: 3）

### 出力確認
- [ ] スクリプト実行時にGPU/CPU情報が表示される
- [ ] データセットのサンプル数が表示される
- [ ] 学習プログレスバーが表示される
- [ ] 評価メトリクス（Accuracy, F1等）が計算される
- [ ] 混同行列が保存される

### ドキュメント
- [ ] `training_guide.md` を作成した
- [ ] 実行方法を記載した
- [ ] パラメータ説明を記載した
- [ ] トラブルシューティングを記載した

## 🎯 動作確認

以下をテストしましたか？

### 基本動作
- [ ] スクリプトがエラーなく起動する
- [ ] データセットが正常に読み込まれる
- [ ] トークナイザーが正常に動作する
- [ ] モデルが初期化される
- [ ] 学習が開始される

### 評価機能
- [ ] Validationでの評価が実行される
- [ ] Testでの評価が実行される
- [ ] Accuracy, F1, Precision, Recallが計算される
- [ ] 混同行列が生成される

### 保存機能
- [ ] モデルが `./saved_model` に保存される
- [ ] トークナイザーが保存される
- [ ] 結果が `training_results.json` に保存される
- [ ] 混同行列が `figures/confusion_matrix.png` に保存される

## 📊 成果物確認

以下のファイルが存在しますか？

```
人工知能応用データセット/
├── 02_train_bert.py          ✅
├── training_guide.md          ✅
├── test_small.py              ✅（オプション）
├── train_small.jsonl          ✅（テスト用）
├── val_small.jsonl            ✅（テスト用）
└── test_small.jsonl           ✅（テスト用）
```

## 🧪 テスト結果

### 小規模データでのテスト実行結果

- 実行時間: _____ 秒
- Train Loss: _____
- Val Accuracy: _____
- エラーの有無: はい / いいえ
- エラー内容: _____

### 学習時間の見積もり

- GPU使用時: 約 _____ 分（3 epochs）
- CPU使用時: 約 _____ 分（3 epochs）

## 💡 理解度確認

以下を説明できますか？

1. **BERTの仕組み**
   - [ ] BERTが何をするモデルか説明できる
   - [ ] トークナイザーの役割を説明できる
   - [ ] max_lengthの意味を説明できる

2. **学習パラメータ**
   - [ ] batch_sizeの役割を説明できる
   - [ ] learning_rateの役割を説明できる
   - [ ] num_epochsの意味を説明できる

3. **評価メトリクス**
   - [ ] Accuracyの計算方法を説明できる
   - [ ] F1スコアが何を表すか説明できる
   - [ ] 混同行列の読み方を説明できる

## ⚠️ 注意事項

### 次のコマ（コマ3）で本格学習を実行する前に

- [ ] スクリプトが正常に動作することを確認した
- [ ] パラメータが適切に設定されている
- [ ] 学習時間の見積もりを把握した
- [ ] GPU/CPUどちらで実行するか決めた

### よくあるエラーと対処法

#### エラー: CUDA out of memory
→ batch_sizeを8または4に減らす

#### エラー: Tokenizer not found
→ インターネット接続を確認、またはローカルキャッシュを確認

#### 警告: Some weights were not initialized
→ 正常な警告（分類ヘッドは新規初期化されるため）

## ⏱️ 時間確認

- 開始時刻: _____
- 終了時刻: _____
- 所要時間: _____ 分

**目標: 90分以内に完了**

## 🔄 GitHubへの更新

### コマ2完了時のコミット

```bash
cd "/home/ike/Desktop/人工知能応用データセット"

# 変更ファイルを確認
git status

# 新規作成したファイルを追加
git add 02_train_bert.py
git add training_guide.md
git add test_small.py
git add train_small.jsonl val_small.jsonl test_small.jsonl

# チェックリストも更新
git add "コマ2_BERT学習スクリプト作成/checklist.md"

# コミット
git commit -m "Complete コマ2: BERT学習スクリプト作成

- BERT学習スクリプトを実装
- データ前処理パイプラインを構築
- 評価メトリクスを実装
- 小規模データでテスト実行完了
- 本格学習の準備完了"

# プッシュ
git push origin main
```

### 完了確認
- [ ] git status でコミットするファイルを確認した
- [ ] git add でファイルを追加した
- [ ] git commit でコミットした
- [ ] git push でプッシュした
- [ ] GitHubで反映を確認した

---

## ⏭️ 次のコマへ

すべてのチェックボックスが✅になり、GitHubへのプッシュも完了したら、**コマ3_モデル学習1回目** のフォルダに進んでください。

### 引き継ぎ事項メモ

- max_length設定: _____
- batch_size設定: _____
- 学習時間見積もり: _____ 分
- GPU使用: はい / いいえ
- 特記事項: _____










