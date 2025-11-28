# 🗣️ ビジネス指示文 曖昧性判定データセット

[![License](https://img.shields.io/badge/License-Research%20Only-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Japanese-red.svg)]()
[![Dataset Size](https://img.shields.io/badge/Dataset-2000%20samples-green.svg)]()

> 外国人労働者のための日本語ビジネス指示文の明確性判定 BERT 学習用データセット

---

## 📋 目次

- [プロジェクト概要](#プロジェクト概要)
- [研究背景](#研究背景)
- [データセット構成](#データセット構成)
- [ラベル定義](#ラベル定義)
- [詳細判定ルール](#詳細判定ルールアノテーション基準)
- [データ形式](#データ形式)
- [使用方法](#使用方法)
- [データ生成方法](#データ生成方法)
- [データ品質チェック](#データ品質チェック)
- [プロジェクト構造](#プロジェクト構造)
- [今後の展望](#今後の展望)
- [貢献方法](#貢献方法)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス利用規約)

---

## プロジェクト概要

外国人労働者が日本の職場で直面する「**指示内容の曖昧さ**」や「**暗黙の了解**」によるミスコミュニケーションを解決するため、上司の指示文が「**明確（Actionable）**」か「**曖昧（Ambiguous）**」かを判定する **BERT 二値分類モデル**の構築を目指します。

本リポジトリには、そのモデル学習に使用するための **2,000 件の日本語ビジネス指示文データセット**が含まれています。

### 🎯 プロジェクトの目的

1. **外国人労働者の職場適応支援**: 曖昧な指示を検出し、明確化を促す
2. **上司のコミュニケーション改善**: 指示の曖昧さをリアルタイムでフィードバック
3. **日本語教育への応用**: ビジネス日本語教材として活用

---

## 研究背景

### 課題認識

日本の職場では、以下のような「曖昧な指示」が頻繁に使用されます：

- 「**例の件、早めに対応しといて**」
- 「**明日の会議資料、ざっと目を通しておいて**」
- 「**いつものフォーマットで、なる早でお願い**」

これらの指示は、日本人であれば文脈や暗黙知から意図を理解できますが、**非母語話者にとっては「何を」「いつまでに」すべきか不明確**で、ミスコミュニケーションの原因となります。

### 解決アプローチ

BERT を用いた二値分類モデルにより、指示文の「明確性」を自動判定し、曖昧な指示に対してはリアルタイムで警告・改善提案を行うシステムの構築を目指します。

---

## データセット構成

### 📊 データ量

| 項目 | 件数 | 割合 |
|:-----|:-----|:-----|
| **総データ数** | 2,000 件 | 100% |
| 学習データ (train) | 1,600 件 | 80% |
| 検証データ (validation) | 200 件 | 10% |
| テストデータ (test) | 200 件 | 10% |

### 📁 ファイル一覧

| ファイル名 | 件数 | 用途 | サイズ |
|:----------|:-----|:-----|:-------|
| `dataset.jsonl` | 2,000件 | 全データ（統合版） | 338KB |
| `train.jsonl` | 1,600件 | 学習用（80%） | 271KB |
| `val.jsonl` | 200件 | 検証用（10%） | 34KB |
| `test.jsonl` | 200件 | テスト用（10%） | 34KB |

### 🎯 ラベル分布

全てのファイルで **Label 0（明確）と Label 1（曖昧）が 50:50** の均等な分布。

- **Label 0（明確）**: 1,000 件（50%）
- **Label 1（曖昧）**: 1,000 件（50%）

### 📈 バリエーション

- **業種**: 営業、事務、IT・エンジニア、製造、接客・サービス、物流、経理、人事、マーケティング、カスタマーサポート
- **シチュエーション**: 会議前、締切対応、日常業務、トラブル対応、顧客対応など
- **口調**: 丁寧語、タメ口、命令形
- **文字数**: 10〜80 文字程度（平均 24.2 文字）

---

## ラベル定義

### Label 0: 明確（Clear / Actionable）

**定義**: 外国人労働者が追加質問なしに、即座に具体的な行動に着手・完了できる指示。

#### 必須条件（両方を満たす）

1. **具体的な「期限（When）」が明示されている**
   - ✅ 良い例：「今日17時までに」「明日の午前中に」「金曜日までに」
   - ❌ 悪い例：「早めに」「なる早で」「後で」

2. **具体的な「行動内容（What）」が明示されている**
   - ✅ 良い例：「メール送信する」「印刷する」「Excelで集計する」
   - ❌ 悪い例：「対応する」「処理する」「よろしく」

#### 特徴

- **動詞**: 操作的・具体的（メール送信、印刷、修正、保存、発注、電話など）
- **対象**: 固有名詞・具体的名称（A社の見積書、10月の月次レポート、顧客リストなど）
- **期限**: 絶対的・数値的（15時までに、今日中に、30分以内になど）
- **成果物**: 定量的（PDF化して、3案作成して、5名に送るなど）

#### サンプル

```json
{"text": "今日の17時までに、A社向けの見積書をPDFで作成してSlackにアップしてください。", "label": 0, "reason": "期限（今日17時）、対象（A社向け見積書）、形式（PDF）、場所（Slack）が全て具体的"}
```

```json
{"text": "10月の月次売上レポートを、明日の午前中までにExcelで山田さんにメール送信してください。", "label": 0, "reason": "対象（10月月次売上レポート）、期限（明日午前中）、形式（Excel）、宛先（山田さん）、手段（メール）が明確"}
```

---

### Label 1: 曖昧（Ambiguous）

**定義**: What または When が欠如しており、作業者が「どうすればいいですか？」と確認しないと着手できない、あるいは誤解するリスクがある指示。

#### 該当条件（いずれか1つでも該当）

1. **期限・優先度（When）が不明確または不在**
   - ❌ 悪い例：「早めに」「なる早で」「後で」「そのうち」「手が空いたら」

2. **行動内容・対象（What）が不明確または抽象的**
   - ❌ 悪い例：「対応する」「処理する」「例の件」「あれ」「いつものやつ」

3. **程度・質（How）が感覚的で判断できない**
   - ❌ 悪い例：「ざっと」「適当に」「いい感じに」「ちゃんと」「しっかりと」

#### 特徴

- **動詞**: 包括的・抽象的（対応する、処理する、確認する、よしなにやる、進める、やっておくなど）
- **対象**: 指示代名詞・暗黙知（例の件、あの資料、いつものやつ、あれ、この前のなど）
- **期限**: 相対的・主観的（早めに、手が空いたら、急ぎで、そのうちなど）
- **程度**: 感覚的な副詞（ざっと、適当に、軽く、入念になど）

#### サンプル

```json
{"text": "例の件、早めに対応しといて。", "label": 1, "reason": "「例の件」が指示代名詞で対象不明、「早めに」が主観的で期限不明、「対応」が抽象的"}
```

```json
{"text": "明日の会議資料、ざっと目を通しておいて、変なところあったら直して。", "label": 1, "reason": "「ざっと」「変なところ」が感覚的で具体性に欠け、いつまでに完了すべきか不明（会議前？何時？）"}
```

---

## 詳細判定ルール（アノテーション基準）

### 1. 内容・動作 (What) の判定

| 判定項目 | 曖昧 (Label 1) の特徴 | 明確 (Label 0) の特徴 |
|:---------|:---------------------|:---------------------|
| **動詞** | 包括的・抽象的<br>（例：対応する、処理する、確認する、よしなにやる、進める、やっておく） | 操作的・具体的<br>（例：メールを送る、修正する、保存する、発注する、電話する、印刷する） |
| **対象** | 指示代名詞・暗黙知<br>（例：例の件、あの資料、いつものやつ、あれ、この前の） | 固有名詞・具体的名称<br>（例：A社の請求書、議事録ファイル、10月の月次レポート、顧客リスト） |
| **ゴール** | 状態が不明<br>（例：いい感じに、形にしておいて、まとめておいて、なんとかして） | 成果物が定義されている<br>（例：PDF化して、印刷して、リストにして、要約して、3案作成して） |

### 2. 期限・優先度 (When) の判定

| 判定項目 | 曖昧 (Label 1) の特徴 | 明確 (Label 0) の特徴 |
|:---------|:---------------------|:---------------------|
| **期限** | 相対的・主観的<br>（例：なる早で、手が空いたら、早めに、そのうち、後で、急ぎで） | 絶対的・数値的<br>（例：15時までに、今日中に、金曜の朝一で、30分以内に、明日の午前中までに） |
| **優先度** | 矛盾・遠慮<br>（例：急ぎじゃないけど早めに、できれば今日中に、時間あるときで） | 確定的<br>（例：最優先で、他の作業を止めて、期限厳守で） |

### 3. 程度・質 (How) の判定

| 判定項目 | 曖昧 (Label 1) の特徴 | 明確 (Label 0) の特徴 |
|:---------|:---------------------|:---------------------|
| **程度** | 感覚的な副詞<br>（例：ざっと、適当に、軽く、しっかりと、入念に、ちゃんと） | 定量的な指示<br>（例：誤字がないか、全ページ、3案出す、5名に送る） |

### 4. 特例ルール（グレーゾーン判定）

以下のケースは、明確にルール化しています：

| ケース | 判定 | 理由 |
|:-------|:-----|:-----|
| **「今日中」** | 明確 (0) | デッドラインが日付変更前と確定するため |
| **「明日」「来週」** | 明確 (0) | 期限が特定できるため（時刻不明でも日付が確定） |
| **「Aさんに聞いて」** | 明確 (0) | 「聞く」という次のアクションが明確なため |
| **専門用語・社内用語**<br>（例：ポンチ絵、ガラガラポン、アジェンダ） | 曖昧 (1) | 非母語話者には理解困難なため |
| **「いつものフォーマットで」** | 曖昧 (1) | 暗黙知に依存しており、外国人労働者には不明 |
| **「念のため」「一応」** | 曖昧 (1) | 程度や必要性が不明確 |

---

## データ形式

### JSONL（JSON Lines）形式

1行に1つのJSONオブジェクト。各行は独立したJSONとしてパース可能です。

```json
{"text": "指示文", "label": 0 or 1, "reason": "判定根拠"}
```

### フィールド説明

| フィールド | 型 | 説明 |
|:----------|:---|:-----|
| `text` | string | ビジネス指示文（日本語） |
| `label` | int | `0`: 明確な指示 / `1`: 曖昧な指示 |
| `reason` | string | ラベル判定の理由・根拠 |

---

## 使用方法

### 1. データ読み込み（Python）

```python
import json

# 1ファイル読み込み
with open('train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print(f"総データ数: {len(data)}")
print(f"サンプル: {data[0]}")

# Label分布確認
labels = [item['label'] for item in data]
print(f"Label 0: {labels.count(0)} 件")
print(f"Label 1: {labels.count(1)} 件")
```

### 2. HuggingFace Datasets で読み込み

```python
from datasets import load_dataset

# 全データセットを読み込み
dataset = load_dataset('json', data_files={
    'train': 'train.jsonl',
    'validation': 'val.jsonl',
    'test': 'test.jsonl'
})

print(dataset)
print(f"Train samples: {len(dataset['train'])}")
print(f"Val samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}")
```

### 3. BERT で学習する例

```python
from datasets import load_dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# データ読み込み
dataset = load_dataset('json', data_files={
    'train': 'train.jsonl',
    'validation': 'val.jsonl',
    'test': 'test.jsonl'
})

# 東北大学 BERT（日本語）を使用
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3')

# トークナイズ
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length',
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# モデル初期化
model = BertForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese-v3',
    num_labels=2
)

# 学習設定
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# 評価メトリクス
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer 初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

# 学習開始
trainer.train()

# テストデータで評価
test_results = trainer.evaluate(tokenized_datasets['test'])
print(f"Test Results: {test_results}")
```

---

## データ生成方法

### 生成手法

本データセットは、以下の方法で生成されました：

1. **ルールベース + テンプレート方式**
   - 詳細なアノテーションガイドラインに基づいた判定ルール
   - 多様な業種・シチュエーション・口調のテンプレート

2. **バリエーション戦略**
   - 8つの異なるパターンで生成し、自然な表現を実現
   - 同じパターンの繰り返しを避け、実際の日本企業で使われる表現を重視

3. **品質管理**
   - 生成後のサンプリングチェック
   - ラベルバランスの厳密な管理（50:50）
   - 文字数・表現の多様性確認

### 再生成方法

データセットを再生成したい場合は、`generate_dataset.py` を実行してください：

```bash
python generate_dataset.py
```

詳細な生成プロンプトは `dataset_generation_prompt.md` を参照してください。

---

## データ品質チェック

### チェックスクリプト実行

```bash
python check_dataset.py
```

このスクリプトは以下を表示します：

- ✅ 各ファイルのデータ数とラベル分布
- ✅ 文字数統計（最小/最大/平均）
- ✅ ランダムサンプル（各ラベル3件ずつ）

### 品質基準チェックリスト

#### 1. ラベルバランス
- [ ] 全体で 曖昧:明確 = 50:50 になっているか
- [ ] train / val / test それぞれでラベル比率が保たれているか

#### 2. データ量
- [ ] dataset.jsonl: 2,000 件
- [ ] train.jsonl: 1,600 件
- [ ] val.jsonl: 200 件
- [ ] test.jsonl: 200 件

#### 3. ルール遵守

**明確な指示 (Label 0) について**:
- [ ] What（行動内容）が具体的か
- [ ] When（期限）が明示されているか
- [ ] 外国人労働者が即座に行動に移せる内容か

**曖昧な指示 (Label 1) について**:
- [ ] What または When のいずれか（または両方）が欠如しているか
- [ ] 指示代名詞や感覚的な表現が含まれているか
- [ ] 追加質問なしでは行動に移しづらい内容か

#### 4. 多様性
- [ ] 同じパターンの繰り返しが少ないか
- [ ] 様々な業種・シチュエーションがカバーされているか
- [ ] 口調（丁寧語/タメ口/命令形）にバリエーションがあるか
- [ ] 文字数にバラつきがあるか（10〜80文字程度）

#### 5. 自然さ
- [ ] 実際の日本企業で使われそうなリアルな表現か
- [ ] 不自然な日本語や文法ミスがないか

---

## プロジェクト構造

```
.
├── README.md                          # このファイル
├── dataset.jsonl                      # 全データ（2,000件）
├── train.jsonl                        # 学習用データ（1,600件）
├── val.jsonl                          # 検証用データ（200件）
├── test.jsonl                         # テストデータ（200件）
├── generate_dataset.py                # データ生成スクリプト
├── check_dataset.py                   # データ品質チェックスクリプト
├── dataset_generation_prompt.md       # データ生成用詳細ガイド（Cursor Composer用）
└── データセット概要.txt                 # データセット概要（日本語）
```

---

## 今後の展望

### 短期目標

- [ ] **BERT モデルの学習・評価**
  - 東北大学 BERT（cl-tohoku/bert-base-japanese-v3）での学習
  - F1スコア 85% 以上を目標

- [ ] **Web デモアプリの開発**
  - Gradio / Streamlit を使用した簡易デモ
  - 指示文を入力すると曖昧性判定 + 改善提案を表示

### 中期目標

- [ ] **データセット拡張**
  - 3,000件 → 5,000件への拡張
  - メール文、チャット文への対応

- [ ] **多クラス分類への拡張**
  - 曖昧性の種類別分類（What欠如 / When欠如 / How欠如）

### 長期目標

- [ ] **実際の職場での実証実験**
  - 外国人労働者を雇用する企業との連携
  - ユーザーフィードバックによるモデル改善

- [ ] **多言語対応**
  - 英語・中国語・ベトナム語への翻訳・対応

---

## 貢献方法

本プロジェクトへの貢献を歓迎します！

### 貢献の種類

1. **データ品質の改善**
   - 不適切なラベルの報告・修正
   - 新しいサンプルの追加提案

2. **ドキュメントの改善**
   - README や判定ルールの明確化
   - 使用例・チュートリアルの追加

3. **コードの改善**
   - データ生成スクリプトの最適化
   - 新しいチェックスクリプトの追加

### 貢献手順

1. このリポジトリを Fork
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチに Push (`git push origin feature/amazing-feature`)
5. Pull Request を作成

### Issue の作成

以下のような内容で Issue を作成してください：

- **バグ報告**: データの不具合、ラベルミスなど
- **機能提案**: 新しいチェック機能、改善提案など
- **質問**: データセットの使用方法、判定ルールの解釈など

---

## 参考文献

- 安部 (2018). 「外国人労働者とのコミュニケーション課題」
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *arXiv preprint arXiv:1810.04805*.
- 東中竜一郎 et al. (2021). 「日本語対話データセットを用いたBERT事前学習モデルの評価」

---

## ライセンス・利用規約

### 利用条件

- ✅ 本データセットは**研究目的でのみ使用**してください
- ❌ **商用利用は禁止**です
- ✅ データの再配布には**出典を明記**してください

### 推奨される引用方法

```
ビジネス指示文 曖昧性判定データセット (2024)
GitHub: https://github.com/[your-username]/[repository-name]
```

### 倫理的配慮

- 実在の企業名・個人名が含まれる場合は匿名化してください
- 差別的・攻撃的な表現が含まれないよう注意してください
- データセット生成における倫理的配慮を遵守してください

---

## トラブルシューティング

### 文字化けする場合

```python
# UTF-8 エンコーディングを明示
with open('train.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
```

### メモリ不足の場合

```python
# ストリーミング読み込み（1行ずつ処理）
def read_in_batches(filepath, batch_size=100):
    batch = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

# 使用例
for batch in read_in_batches('train.jsonl'):
    # バッチごとに処理
    process(batch)
```

### JSON パースエラーの場合

```python
import json

# エラー行を特定
with open('train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error at line {i}: {e}")
            print(f"Content: {line[:100]}")
```

---

## お問い合わせ

データセットに関する質問や改善提案があれば、Issue を作成してください。

---

## 更新履歴

- **2024-11-28**: 初版リリース（v1.0.0）
  - 2,000件のデータセット作成
  - 学習/検証/テスト分割（80/10/10）
  - 判定ルール・アノテーション基準の確立

---

**Made with ❤️ for foreign workers in Japan**
