# コマ3 完了チェックリスト

## 📝 学習実行確認

### 事前確認
- [ ] `pre_check.py` を実行した
- [ ] ディスク容量が十分にある（5GB以上推奨）
- [ ] データセットが存在する（train/val/test）
- [ ] `02_train_bert.py` が存在する

### 学習実行
- [ ] `python 02_train_bert.py` を実行した
- [ ] 学習が正常に開始された
- [ ] Epoch 1/3 が完了した
- [ ] Epoch 2/3 が完了した
- [ ] Epoch 3/3 が完了した
- [ ] エラーなく完了した

### 学習過程の監視
- [ ] 学習中のLossが表示された
- [ ] Lossが徐々に下がることを確認した
- [ ] Validation評価が各Epoch後に実行された
- [ ] メモリエラーが発生しなかった

## 📊 結果確認

### 生成ファイル
- [ ] `saved_model/` フォルダが作成された
- [ ] `training_results.json` が生成された
- [ ] `figures/confusion_matrix.png` が生成された
- [ ] `logs/` フォルダにログが保存された

### 評価指標
- [ ] Test Accuracy: _____ %
- [ ] F1スコア: _____
- [ ] Precision: _____
- [ ] Recall: _____

### 精度判定
- [ ] Accuracy 80%以上 → 🎉 素晴らしい！
- [ ] Accuracy 75-80% → ✅ 良好
- [ ] Accuracy 70-75% → 😐 改善の余地あり
- [ ] Accuracy 70%未満 → 😢 要改善

## 🔍 誤分類分析

### スクリプト実行
- [ ] `extract_errors.py` を作成した
- [ ] スクリプトを実行した
- [ ] `error_samples.json` が生成された
- [ ] 誤分類サンプル数を確認した

### 誤分類の内訳
- [ ] Label 0→1の誤分類数: _____
- [ ] Label 1→0の誤分類数: _____
- [ ] 合計誤分類数: _____

### 誤分類の傾向分析
- [ ] 誤分類サンプルを5件以上確認した
- [ ] どのようなパターンで誤分類するか分析した
- [ ] 改善ポイントをリストアップした

## 📝 レポート作成

### first_training_report.md
- [ ] レポートを作成した
- [ ] 学習設定を記載した
- [ ] 評価指標を記載した
- [ ] 混同行列を分析した
- [ ] 誤分類の傾向を記載した
- [ ] 考察を記載した
- [ ] 次のステップを記載した

## 💡 理解度確認

以下を説明できますか？

### 学習結果の理解
- [ ] Accuracyが何を意味するか説明できる
- [ ] F1スコアの重要性を説明できる
- [ ] 混同行列の読み方を説明できる

### 誤分類の理解
- [ ] どのようなサンプルが誤分類されたか説明できる
- [ ] Label 0→1とLabel 1→0の違いを説明できる
- [ ] 誤分類の原因を仮説として説明できる

### 改善の方向性
- [ ] どうすれば精度が上がるか案を3つ以上挙げられる
- [ ] パラメータ調整の方向性を説明できる
- [ ] データの問題点（あれば）を指摘できる

## 📂 成果物確認

以下のファイルが存在しますか？

```
人工知能応用データセット/
├── saved_model/
│   ├── config.json                    ✅
│   ├── pytorch_model.bin              ✅
│   └── tokenizer files                ✅
├── training_results.json              ✅
├── figures/
│   └── confusion_matrix.png           ✅
├── error_samples.json                 ✅
├── first_training_report.md           ✅
├── pre_check.py                       ✅
└── extract_errors.py                  ✅
```

## ⏱️ 時間確認

- 開始時刻: _____
- 終了時刻: _____
- 所要時間: _____ 分（学習時間含む）

**目標: 90分以内に完了**

※学習に時間がかかる場合は、待ち時間に次のコマの準備をしてもOK

## 🎯 目標達成度

### 必須目標
- [ ] 学習が正常に完了した
- [ ] Test Accuracy 70%以上を達成した
- [ ] 誤分類サンプルを抽出した
- [ ] レポートを作成した

### 追加目標
- [ ] Accuracy 75%以上を達成した
- [ ] 誤分類の傾向を詳細に分析した
- [ ] 具体的な改善案を3つ以上考えた

## ⚠️ トラブルシューティング

### 発生したエラーと対処

| エラー内容 | 対処方法 | 解決: はい/いいえ |
|----------|---------|-----------------|
| _____ | _____ | _____ |

### 学習が途中で止まった場合
- [ ] エラーメッセージを記録した
- [ ] パラメータを調整した（batch_size等）
- [ ] 再実行して成功した

## 🔄 GitHubへの更新

### コマ3完了時のコミット

```bash
cd "/home/ike/Desktop/人工知能応用データセット"

# 変更ファイルを確認
git status

# 学習結果を追加（モデルファイルは大きいので.gitignoreで除外推奨）
git add training_results.json
git add error_samples.json
git add first_training_report.md
git add figures/confusion_matrix.png
git add pre_check.py
git add extract_errors.py

# チェックリストも更新
git add "コマ3_モデル学習1回目/checklist.md"

# 注意: saved_model/は大きいので、.gitignoreに追加することを推奨
# echo "saved_model/" >> .gitignore
# echo "*.bin" >> .gitignore
# git add .gitignore

# コミット
git commit -m "Complete コマ3: モデル学習（1回目）

- BERT学習を完了（Test Accuracy: __%）
- 学習結果を記録
- 誤分類サンプルを抽出
- 混同行列を生成
- 初回学習レポートを作成"

# プッシュ
git push origin main
```

### 完了確認
- [ ] git status でコミットするファイルを確認した
- [ ] saved_model/を.gitignoreに追加した（推奨）
- [ ] git add でファイルを追加した
- [ ] git commit でコミットした
- [ ] git push でプッシュした
- [ ] GitHubで反映を確認した

### 注意: 大きなファイルの扱い

モデルファイル（saved_model/）は数百MB〜数GBになるため、GitHubにプッシュすると問題が発生します。

**推奨対応**:
```bash
# .gitignoreに追加
echo "saved_model/" >> .gitignore
echo "results/" >> .gitignore
echo "logs/" >> .gitignore
echo "*.bin" >> .gitignore
echo "*.pth" >> .gitignore

git add .gitignore
git commit -m "Add .gitignore for large model files"
```

---

## ⏭️ 次のコマへ

すべてのチェックボックスが✅になり、GitHubへのプッシュも完了したら、**コマ4_エラー分析と改善** のフォルダに進んでください。

### 引き継ぎ事項メモ

- Test Accuracy: _____ %
- 誤分類傾向: _____
- 改善ポイント1: _____
- 改善ポイント2: _____
- 改善ポイント3: _____










