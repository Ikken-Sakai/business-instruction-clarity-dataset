# 📚 GitHub更新ガイド

## 🔧 初期設定（初回のみ）

### 1. Gitリポジトリの確認

```bash
cd "/home/ike/Desktop/人工知能応用データセット"

# Gitリポジトリかどうか確認
git status
```

**リポジトリでない場合**:
```bash
# 初期化
git init

# リモートリポジトリを追加
git remote add origin <your-repository-url>

# 例: git remote add origin https://github.com/username/repo-name.git
```

---

### 2. .gitignore の設定

大きなモデルファイルをGitHubにプッシュしないように設定します。

```bash
# 推奨の.gitignoreをコピー
cp .gitignore_recommended .gitignore

# または手動で作成
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc

# モデルファイル（重要！）
saved_model/
*.bin
*.pth
results/
logs/

# その他
.DS_Store
.vscode/
EOF

# .gitignoreをコミット
git add .gitignore
git commit -m "Add .gitignore to exclude large model files"
git push origin main
```

---

## 📝 各コマ完了時の更新手順

各コマのチェックリストに詳細な手順がありますが、基本的な流れは以下の通りです：

### 基本的なGitワークフロー

```bash
# 1. 現在の状態を確認
git status

# 2. 変更されたファイルを確認
git diff

# 3. ファイルを追加（ステージング）
git add <file1> <file2> ...

# または、すべてのファイルを追加（注意！）
# git add .

# 4. コミット（変更を記録）
git commit -m "コミットメッセージ"

# 5. リモートリポジトリにプッシュ
git push origin main
```

---

## 🎯 各コマでコミットするファイル一覧

### コマ1: 環境構築とデータ確認

```bash
git add requirements.txt
git add check_gpu.py
git add 01_eda.py
git add eda_report.md
git add figures/
git add "コマ1_環境構築とデータ確認/checklist.md"

git commit -m "Complete コマ1: 環境構築とデータ確認"
git push origin main
```

---

### コマ2: BERT学習スクリプト作成

```bash
git add 02_train_bert.py
git add training_guide.md
git add "コマ2_BERT学習スクリプト作成/checklist.md"

git commit -m "Complete コマ2: BERT学習スクリプト作成"
git push origin main
```

---

### コマ3: モデル学習（1回目）

```bash
# 注意: saved_model/は.gitignoreで除外する
git add training_results.json
git add error_samples.json
git add first_training_report.md
git add figures/confusion_matrix.png
git add "コマ3_モデル学習1回目/checklist.md"

git commit -m "Complete コマ3: モデル学習（1回目） - Accuracy: __%"
git push origin main
```

---

### コマ4: エラー分析と改善

```bash
git add 03_error_analysis.py
git add error_analysis_report.md
git add improved_config.json
git add figures/error_patterns.png
git add "コマ4_エラー分析と改善/checklist.md"

git commit -m "Complete コマ4: エラー分析と改善"
git push origin main
```

---

### コマ5: モデル再学習と精度向上

```bash
git add training_results.json
git add 04_compare_models.py
git add comparison_report.md
git add demo_samples.json
git add figures/comparison_charts.png
git add "コマ5_モデル再学習と精度向上/checklist.md"

git commit -m "Complete コマ5: モデル再学習 - Accuracy: __%（改善: +__%）"
git push origin main
```

---

### コマ6: デモアプリ開発（基本機能）

```bash
git add demo_app.py
git add demo_guide.md
git add screenshots/
git add "コマ6_デモアプリ開発1/checklist.md"

git commit -m "Complete コマ6: デモアプリ開発（基本機能）"
git push origin main
```

---

### コマ7: デモアプリ改善とプレ発表準備

```bash
git add "プレ発表スライド.pptx"
git add "発表ノート.md"
git add "想定質問と回答.md"
git add "コマ7_デモアプリ開発2とプレ発表準備/checklist.md"

git commit -m "Complete コマ7: プレ発表準備"
git push origin main
```

---

### コマ8: プレ発表資料完成

```bash
git add "プレ発表スライド_最終版.pptx"
git add "プレ発表スライド_最終版.pdf"
git add "リハーサル記録.md"
git add "発表チェックリスト.md"
git add "コマ8_プレ発表資料完成/checklist.md"

git commit -m "Complete コマ8: プレ発表資料完成 🎉"
git push origin main
```

---

## ⚠️ よくあるエラーと対処法

### エラー1: "fatal: not a git repository"

**原因**: Gitリポジトリとして初期化されていない

**対処**:
```bash
git init
git remote add origin <your-repository-url>
```

---

### エラー2: "rejected - non-fast-forward"

**原因**: リモートリポジトリに新しい変更がある

**対処**:
```bash
# リモートの変更を取得
git pull origin main --rebase

# 再度プッシュ
git push origin main
```

---

### エラー3: "this exceeds GitHub's file size limit of 100 MB"

**原因**: 100MBを超えるファイルをコミットしようとしている（モデルファイル等）

**対処**:
```bash
# .gitignoreにsaved_model/を追加
echo "saved_model/" >> .gitignore
echo "*.bin" >> .gitignore

# キャッシュをクリア
git rm -r --cached saved_model/

# 再度コミット
git add .gitignore
git commit -m "Exclude large model files from repository"
git push origin main
```

---

### エラー4: "fatal: remote origin already exists"

**原因**: リモートリポジトリが既に設定されている

**対処**:
```bash
# 既存のリモートを確認
git remote -v

# 必要に応じて変更
git remote set-url origin <new-repository-url>
```

---

## 💡 便利なGitコマンド

### 変更内容の確認

```bash
# ステージングされていない変更を表示
git diff

# ステージングされた変更を表示
git diff --cached

# 最近のコミット履歴を表示
git log --oneline -10
```

---

### コミットの修正

```bash
# 直前のコミットメッセージを修正
git commit --amend -m "新しいコミットメッセージ"

# 直前のコミットにファイルを追加
git add <forgotten-file>
git commit --amend --no-edit
```

---

### ブランチの管理

```bash
# 新しいブランチを作成
git branch feature-branch

# ブランチに切り替え
git checkout feature-branch

# ブランチを作成して切り替え（一括）
git checkout -b feature-branch

# ブランチ一覧を表示
git branch -a
```

---

## 📊 推奨されるコミットメッセージの書き方

### 良いコミットメッセージの例

```
Complete コマ3: モデル学習（1回目）

- BERT学習を完了（Test Accuracy: 78%）
- 学習結果を記録
- 誤分類サンプルを抽出
- 混同行列を生成
```

### 悪いコミットメッセージの例

```
update  # 何を更新したか不明
fix bug  # どのバグを修正したか不明
aaa  # 意味不明
```

---

## 🎯 コミットのタイミング

### 推奨されるコミットのタイミング

✅ **各コマ完了時**
- 明確な区切りがある
- 成果物が揃っている

✅ **大きな機能を実装した時**
- スクリプトが完成した
- 重要な分析が終わった

✅ **問題を修正した時**
- バグを修正した
- エラーを解決した

### 避けるべきコミット

❌ 未完成のコード
❌ エラーが残っている状態
❌ 一度に大量のファイルをまとめてコミット

---

## 📚 参考情報

### GitHub Desktop（GUI）を使う方法

コマンドラインが苦手な場合は、GitHub Desktopを使うこともできます：

1. [GitHub Desktop](https://desktop.github.com/)をダウンロード
2. リポジトリを開く
3. 変更されたファイルを確認
4. コミットメッセージを入力
5. 「Commit to main」ボタンをクリック
6. 「Push origin」ボタンをクリック

---

## ✅ チェックリスト

各コマ完了時に以下を確認：

- [ ] `git status` で変更を確認した
- [ ] `.gitignore` で大きなファイルを除外した
- [ ] `git add` でファイルを追加した
- [ ] 適切なコミットメッセージを書いた
- [ ] `git commit` でコミットした
- [ ] `git push` でプッシュした
- [ ] GitHubのリポジトリで反映を確認した

---

**GitHub更新を習慣化して、作業の履歴をしっかり記録しましょう！** 📝✨










