# コマ6実装プロンプト: デモアプリ開発（基本機能）

このプロンプトに従って、Gradioを使ったデモアプリを作成してください。

---

## 📦 Gradioのインストール

```bash
pip install gradio
```

---

## 🎨 デモアプリの実装

`demo_app.py` を作成してください：

```python
"""
外国人労働者向けビジネス指示文 曖昧性判定デモアプリ
Gradio版
"""

import gradio as gr
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import json

# ========================================
# モデル読み込み
# ========================================

print("モデルを読み込んでいます...")
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertJapaneseTokenizer.from_pretrained('./saved_model')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"モデル読み込み完了（デバイス: {device}）")

# ========================================
# 推論関数
# ========================================

def predict(text):
    """
    テキストを入力して曖昧性を判定
    
    Args:
        text (str): 判定したい指示文
    
    Returns:
        tuple: (判定結果, 信頼度, 詳細説明)
    """
    
    if not text or text.strip() == "":
        return "⚠️ テキストを入力してください", "", ""
    
    # トークナイズ
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
    
    # 結果のフォーマット
    if pred_label == 0:
        result = "✅ 明確な指示"
        color = "green"
        explanation = "この指示は外国人労働者にとって理解しやすく、具体的な内容です。"
    else:
        result = "⚠️ 曖昧な指示"
        color = "orange"
        explanation = "この指示には曖昧な要素が含まれています。より具体的に書き直すことを推奨します。"
    
    # 信頼度表示
    confidence_text = f"信頼度: {confidence*100:.1f}%"
    
    # 詳細説明
    prob_clear = probs[0][0].item() * 100
    prob_ambiguous = probs[0][1].item() * 100
    
    detail = f"""
### 判定の詳細

- **明確な指示の確率**: {prob_clear:.1f}%
- **曖昧な指示の確率**: {prob_ambiguous:.1f}%

{explanation}
    """
    
    return result, confidence_text, detail

# ========================================
# サンプルテキスト
# ========================================

# demo_samples.jsonから読み込み
try:
    with open('demo_samples.json', 'r', encoding='utf-8') as f:
        demo_samples = json.load(f)
    
    examples = [
        [sample['text']] for sample in demo_samples
    ]
except:
    # ファイルがない場合はハードコード
    examples = [
        ["今日の17時までに、A社向けの見積書をPDFで作成してSlackにアップしてください。"],
        ["10月の月次売上レポートを、明日の午前中までにExcelで山田さんにメール送信してください。"],
        ["例の件、早めに対応しといて。"],
        ["明日の会議資料、ざっと目を通しておいて、変なところあったら直して。"],
        ["いつものフォーマットで、なる早でお願い。"]
    ]

# ========================================
# Gradio UI
# ========================================

with gr.Blocks(title="ビジネス指示文 曖昧性判定システム") as demo:
    
    gr.Markdown("""
    # 🗣️ ビジネス指示文 曖昧性判定システム
    
    外国人労働者が理解しやすい「明確な指示」か、理解しづらい「曖昧な指示」かを自動判定します。
    
    **使い方**: 下のテキストボックスに日本語の指示文を入力して、「判定する」ボタンを押してください。
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="指示文を入力してください",
                placeholder="例: 今日の17時までに、報告書を作成してください。",
                lines=3
            )
            
            submit_btn = gr.Button("判定する", variant="primary")
            
            gr.Markdown("### サンプルテキスト（クリックして試してください）")
            gr.Examples(
                examples=examples,
                inputs=input_text,
                label="サンプル"
            )
        
        with gr.Column():
            result_text = gr.Textbox(
                label="判定結果",
                lines=1
            )
            confidence_text = gr.Textbox(
                label="信頼度",
                lines=1
            )
            detail_text = gr.Markdown("")
    
    submit_btn.click(
        fn=predict,
        inputs=input_text,
        outputs=[result_text, confidence_text, detail_text]
    )
    
    gr.Markdown("""
    ---
    ### 判定基準
    
    **✅ 明確な指示の特徴**:
    - 期限（When）が具体的: 「今日17時まで」「明日午前中まで」
    - 行動内容（What）が具体的: 「PDFで作成」「メール送信」
    - 対象が明確: 「A社向けの見積書」「10月の月次レポート」
    
    **⚠️ 曖昧な指示の特徴**:
    - 期限が不明確: 「早めに」「なる早で」「後で」
    - 行動内容が抽象的: 「対応する」「処理する」「よろしく」
    - 対象が不明確: 「例の件」「いつものやつ」「あれ」
    """)

# ========================================
# アプリ起動
# ========================================

if __name__ == "__main__":
    demo.launch(
        share=False,  # 公開リンクを作成しない
        server_name="127.0.0.1",  # ローカルホストのみ
        server_port=7860  # ポート番号
    )
```

---

## 🚀 アプリの起動

```bash
python demo_app.py
```

ブラウザで `http://127.0.0.1:7860` にアクセスしてください。

---

## 🧪 テスト手順

### テスト1: 明確な指示

以下のテキストを入力して、「✅ 明確な指示」と判定されることを確認：

1. 「今日の17時までに、A社向けの見積書をPDFで作成してSlackにアップしてください。」
2. 「10月の月次売上レポートを、明日の午前中までにExcelで山田さんにメール送信してください。」
3. 「顧客リストから休眠顧客30件を抽出して、今週金曜までにCSVファイルで共有してください。」

**期待される結果**: すべて「明確」と判定される

---

### テスト2: 曖昧な指示

以下のテキストを入力して、「⚠️ 曖昧な指示」と判定されることを確認：

1. 「例の件、早めに対応しといて。」
2. 「明日の会議資料、ざっと目を通しておいて、変なところあったら直して。」
3. 「いつものフォーマットで、なる早でお願い。」

**期待される結果**: すべて「曖昧」と判定される

---

### テスト3: エッジケース

以下も試してみてください：

1. **空白テキスト**: エラーメッセージが表示される
2. **非常に短いテキスト**: 「確認」→ 曖昧と判定される可能性
3. **非常に長いテキスト**: 正常に判定される（128トークンで切り捨て）

---

## 📸 スクリーンショットの保存

デモアプリが動作したら、スクリーンショットを保存してください：

```bash
mkdir -p screenshots
```

1. デモアプリのUI全体
2. 明確な指示の判定結果
3. 曖昧な指示の判定結果

保存先: `screenshots/demo_ui_*.png`

---

## 📖 demo_guide.md の作成

```markdown
# デモアプリ使用ガイド

## 起動方法

```bash
python demo_app.py
```

ブラウザで `http://127.0.0.1:7860` にアクセス

## 使い方

1. テキストボックスに日本語の指示文を入力
2. 「判定する」ボタンをクリック
3. 判定結果・信頼度・詳細が表示されます

## サンプルテキスト

画面下部の「サンプル」からクリックして試すこともできます。

## 判定結果の見方

- **✅ 明確な指示**: 外国人労働者が理解しやすい指示
- **⚠️ 曖昧な指示**: 曖昧な要素が含まれており、書き直しを推奨

## 技術仕様

- モデル: cl-tohoku/bert-base-japanese-v3
- フレームワーク: Gradio
- 精度: Test Accuracy ___% （最終精度を記載）

## トラブルシューティング

### アプリが起動しない
→ `pip install gradio` を実行

### モデルが見つからない
→ `saved_model/` フォルダが存在するか確認

### ポート7860が使用中
→ `demo_app.py` の `server_port` を別の番号に変更
```

---

## ✅ 完了確認

- [ ] Gradioがインストールされた
- [ ] `demo_app.py` が作成された
- [ ] アプリが正常に起動する
- [ ] 明確な指示が正しく判定される
- [ ] 曖昧な指示が正しく判定される
- [ ] UIが分かりやすい
- [ ] スクリーンショットを保存した
- [ ] `demo_guide.md` を作成した

---

## 🐛 よくあるエラーと対処法

### エラー: "No module named 'gradio'"
```bash
pip install gradio
```

### エラー: "saved_model not found"
→ `demo_app.py` と同じディレクトリに `saved_model/` フォルダがあるか確認

### エラー: "Address already in use"
→ 別のアプリがポート7860を使用中。`server_port=7861` に変更

---

## ⏭️ 次のコマ（コマ7）への準備

以下を確認してください：

1. **デモアプリが動作する**: 正常に起動し、判定できる
2. **スクリーンショット**: プレ発表で使える画像を保存
3. **改善点**: 次のコマで追加したい機能をメモ

次のコマ7では、デモアプリに機能を追加し、プレ発表の準備を進めます。










