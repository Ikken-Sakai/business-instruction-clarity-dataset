# GitHub リポジトリへのプッシュ手順

## 1. GitHubでリモートリポジトリのURLを確認

GitHubで作成したリポジトリのページから、リポジトリのURLをコピーしてください。

例：
- HTTPS: `https://github.com/your-username/repository-name.git`
- SSH: `git@github.com:your-username/repository-name.git`

## 2. リモートリポジトリを追加

```bash
cd /home/ike/Desktop/人工知能応用データセット

# HTTPSの場合
git remote add origin https://github.com/your-username/repository-name.git

# または、SSHの場合
git remote add origin git@github.com:your-username/repository-name.git
```

## 3. リモートリポジトリを確認

```bash
git remote -v
```

以下のように表示されればOK：
```
origin  https://github.com/your-username/repository-name.git (fetch)
origin  https://github.com/your-username/repository-name.git (push)
```

## 4. GitHubにプッシュ

```bash
git push -u origin main
```

初回プッシュ時は、GitHubの認証情報を求められる場合があります。

### HTTPSの場合
- ユーザー名: GitHubのユーザー名
- パスワード: **Personal Access Token**（GitHubパスワードではありません）

Personal Access Tokenの作成方法：
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → "Generate new token (classic)"
3. スコープで "repo" にチェック
4. "Generate token" をクリック
5. 表示されたトークンをコピー（後で表示できないので注意）

### SSHの場合
SSH鍵が設定されていれば、パスワード入力なしでプッシュできます。

SSH鍵の設定方法：
```bash
# SSH鍵を生成（まだ持っていない場合）
ssh-keygen -t ed25519 -C "your_email@example.com"

# SSH鍵をクリップボードにコピー
cat ~/.ssh/id_ed25519.pub

# GitHubに登録
# GitHub → Settings → SSH and GPG keys → New SSH key
# コピーした公開鍵を貼り付け
```

## 5. プッシュの確認

GitHubのリポジトリページをブラウザで開いて、ファイルがアップロードされていることを確認してください。

## トラブルシューティング

### エラー: `fatal: remote origin already exists`

```bash
# 既存のリモートを削除
git remote remove origin

# 改めてリモートを追加
git remote add origin https://github.com/your-username/repository-name.git
```

### エラー: `error: failed to push some refs`

GitHubリポジトリに既にコンテンツ（README.mdなど）がある場合：

```bash
# リモートの変更を取得してマージ
git pull origin main --allow-unrelated-histories

# 再度プッシュ
git push -u origin main
```

### ブランチ名が `master` の場合

```bash
# ブランチ名を main に変更
git branch -M main

# プッシュ
git push -u origin main
```

## 完了後の確認事項

✅ GitHubリポジトリにファイルがアップロードされている
✅ README.mdが正しく表示されている
✅ `.gitignore`が機能している（`__pycache__/`などが除外されている）
✅ LICENSEファイルが認識されている

## 今後の開発フロー

```bash
# 変更を加えた後
git add .
git commit -m "変更内容を記述"
git push

# 最新の変更を取得
git pull
```





