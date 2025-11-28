#!/usr/bin/env python3
"""
ビジネス指示文 曖昧性判定データセット生成スクリプト
ルールベース + テンプレート方式で2,000件のデータを生成
"""

import json
import random
from typing import List, Dict

# 業種リスト
INDUSTRIES = [
    "営業", "事務", "IT", "製造", "接客", "物流", "経理", "人事", 
    "マーケティング", "カスタマーサポート"
]

# 曖昧な指示のパーツ
AMBIGUOUS_SUBJECTS = [
    "例の件", "あの資料", "この前のやつ", "さっきの話", "あれ",
    "これ", "いつものやつ", "その案件", "あの仕事", "前回のこと"
]

AMBIGUOUS_VERBS = [
    "対応しといて", "処理しておいて", "確認しといて", "よろしく",
    "進めといて", "やっといて", "なんとかして", "お願い"
]

AMBIGUOUS_TIME = [
    "早めに", "なる早で", "手が空いたら", "そのうち", "後で",
    "急ぎで", "できれば今日中に", "時間あるときで"
]

AMBIGUOUS_HOW = [
    "ざっと", "適当に", "軽く", "いい感じに", "ちゃんと",
    "しっかり", "ちょっと", "適宜"
]

# 明確な指示のパーツ
CLEAR_SUBJECTS = [
    "A社の見積書", "10月の月次レポート", "顧客リスト", "議事録",
    "B社向けの提案書", "売上データ", "在庫管理表", "請求書",
    "契約書", "プレゼン資料", "納品書", "発注書"
]

CLEAR_VERBS = [
    "メール送信してください", "印刷してください", "修正してください",
    "作成してください", "確認してください", "共有してください",
    "保存してください", "アップロードしてください", "送付してください",
    "提出してください"
]

CLEAR_TIME = [
    "今日の17時までに", "明日の午前中までに", "今日中に",
    "15時までに", "今すぐ", "30分以内に", "金曜日までに",
    "来週月曜の朝一で", "明後日の正午までに", "本日中に"
]

CLEAR_DETAILS = [
    "PDFで", "Excelで", "3案作成して", "全ページチェックして",
    "5名に送って", "誤字がないか確認して", "フォーマットを統一して",
    "Slackにアップして", "印刷して配布して", "メールで佐藤さんに送って"
]

# 業務内容（業種別）
TASKS = {
    "営業": [
        "顧客訪問の準備", "見積書作成", "提案書作成", "商談資料準備",
        "受注報告", "営業報告書作成", "クライアント対応"
    ],
    "事務": [
        "データ入力", "書類整理", "電話対応記録", "郵便物発送",
        "備品発注", "会議室予約", "ファイル整理"
    ],
    "IT": [
        "サーバー監視", "バグ修正", "デプロイ作業", "コードレビュー",
        "ドキュメント更新", "テスト実行", "システム設定変更"
    ],
    "製造": [
        "在庫確認", "生産計画作成", "品質検査", "設備点検",
        "作業日報作成", "材料発注", "工程管理"
    ],
    "接客": [
        "接客対応", "クレーム対応", "レジ締め", "商品陳列",
        "在庫補充", "店内清掃", "予約管理"
    ],
    "物流": [
        "配送手配", "在庫管理", "入荷検品", "出荷準備",
        "配送ルート確認", "荷物追跡", "倉庫整理"
    ],
    "経理": [
        "請求書処理", "経費精算", "帳簿記入", "振込手続き",
        "月次決算資料作成", "領収書整理", "予算管理"
    ],
    "人事": [
        "採用面接準備", "勤怠管理", "給与計算", "研修資料作成",
        "人事評価資料作成", "労務管理", "社員対応"
    ],
    "マーケティング": [
        "広告資料作成", "SNS投稿", "アンケート分析", "キャンペーン企画",
        "市場調査", "プレスリリース作成", "メルマガ配信"
    ],
    "カスタマーサポート": [
        "問い合わせ対応", "マニュアル更新", "FAQ作成", "クレーム記録",
        "対応履歴入力", "顧客フォロー", "サポートチケット処理"
    ]
}

def generate_ambiguous_instructions(count: int) -> List[Dict]:
    """曖昧な指示を生成"""
    instructions = []
    
    patterns = [
        # パターン1: 指示代名詞 + 曖昧な時間 + 曖昧な動詞
        lambda: f"{random.choice(AMBIGUOUS_SUBJECTS)}、{random.choice(AMBIGUOUS_TIME)}{random.choice(AMBIGUOUS_VERBS)}。",
        
        # パターン2: 具体的な対象 + 曖昧な副詞 + 曖昧な動詞
        lambda: f"{random.choice(CLEAR_SUBJECTS)}、{random.choice(AMBIGUOUS_HOW)}{random.choice(AMBIGUOUS_VERBS)}。",
        
        # パターン3: 曖昧な対象 + よろしく
        lambda: f"{random.choice(AMBIGUOUS_SUBJECTS)}、よろしく頼むわ。",
        
        # パターン4: 具体的な対象 + 曖昧な時間のみ
        lambda: f"{random.choice(CLEAR_SUBJECTS)}の{random.choice(['作成', '確認', '修正', '整理'])}、{random.choice(AMBIGUOUS_TIME)}お願い。",
        
        # パターン5: 曖昧な対象 + 具体的な動詞（時間なし）
        lambda: f"{random.choice(AMBIGUOUS_SUBJECTS)}、{random.choice(['印刷', 'メール送信', '修正', '確認'])}しておいて。",
        
        # パターン6: 業務タスク + 曖昧な指示
        lambda: f"{random.choice([t for tasks in TASKS.values() for t in tasks])}、{random.choice(AMBIGUOUS_HOW)}{random.choice(['やっといて', 'お願い', '進めといて'])}。",
        
        # パターン7: 会議関連の曖昧な指示
        lambda: f"{'明日' if random.random() > 0.5 else '来週'}の会議資料、{random.choice(AMBIGUOUS_HOW)}{'目を通して' if random.random() > 0.5 else '確認して'}おいて。",
        
        # パターン8: 短い曖昧な指示
        lambda: f"{random.choice(['これ', 'あれ', 'その件'])}、{random.choice(['ちゃんと', 'しっかり', '適当に'])}やっといて。",
    ]
    
    reasons_map = {
        0: "指示代名詞で対象不明、期限が主観的で不明確、動詞が抽象的",
        1: "副詞が感覚的で程度不明、期限の記載なし",
        2: "「よろしく」が抽象的で何をすべきか不明、期限の記載なし",
        3: "期限が主観的で不明確",
        4: "対象が指示代名詞で不明、期限の記載なし",
        5: "副詞が感覚的で具体性に欠ける",
        6: "副詞が感覚的、期限の記載なし",
        7: "指示代名詞、感覚的副詞で何をいつまでにすべきか不明"
    }
    
    for i in range(count):
        pattern_idx = random.randint(0, len(patterns) - 1)
        text = patterns[pattern_idx]()
        
        instructions.append({
            "text": text,
            "label": 1,
            "reason": reasons_map.get(pattern_idx, "What または When が欠如しており、具体的な行動に移せない")
        })
    
    return instructions

def generate_clear_instructions(count: int) -> List[Dict]:
    """明確な指示を生成"""
    instructions = []
    
    patterns = [
        # パターン1: 対象 + 時間 + 動詞
        lambda: f"{random.choice(CLEAR_SUBJECTS)}を、{random.choice(CLEAR_TIME)}{random.choice(CLEAR_VERBS)}。",
        
        # パターン2: 対象 + 詳細 + 時間 + 動詞
        lambda: f"{random.choice(CLEAR_SUBJECTS)}を{random.choice(CLEAR_DETAILS)}、{random.choice(CLEAR_TIME)}{random.choice(CLEAR_VERBS)}。",
        
        # パターン3: 時間 + 対象 + 詳細 + 動詞
        lambda: f"{random.choice(CLEAR_TIME)}、{random.choice(CLEAR_SUBJECTS)}を{random.choice(CLEAR_DETAILS)}{random.choice(CLEAR_VERBS)}。",
        
        # パターン4: 会議関連の明確な指示
        lambda: f"{random.choice(['営業チーム全員', '関係者全員', 'プロジェクトメンバー'])}に、{random.choice(CLEAR_TIME)}{'Outlookで' if random.random() > 0.5 else 'Teamsで'}ミーティング招待を送ってください。",
        
        # パターン5: 顧客対応の明確な指示
        lambda: f"{random.choice(['A社', 'B社', 'C社'])}の{random.choice(['佐藤さん', '田中さん', '山田さん'])}に、{random.choice(CLEAR_TIME)}電話して{random.choice(['納期の確認', '進捗の報告', '見積もりの説明'])}をしてください。",
        
        # パターン6: データ処理の明確な指示
        lambda: f"{random.choice(['先月', '今月', '10月', '第3四半期'])}の{random.choice(['売上データ', '在庫データ', '顧客データ', '経費データ'])}を、{random.choice(CLEAR_TIME)}Excelで集計して{random.choice(['山田さん', '佐藤さん', '課長'])}にメール送信してください。",
        
        # パターン7: 会議室・設備予約
        lambda: f"会議室{random.choice(['A', 'B', 'C', '第1', '第2'])}を、{random.choice(['15時から16時', '10時から12時', '13時から14時半', '明日の午前中'])}で{random.choice(['今すぐ', '本日中に'])}予約してください。",
        
        # パターン8: 印刷・配布系
        lambda: f"{random.choice(CLEAR_SUBJECTS)}を{random.choice(['10部', '20部', '5部', '全員分'])}、{random.choice(CLEAR_TIME)}印刷して{random.choice(['会議室に配置', '参加者に配布', '各部署に配布'])}してください。",
    ]
    
    reasons_map = {
        0: "対象、期限、行動内容が全て具体的に明示されている",
        1: "対象、形式、期限、行動が明確で即座に実行可能",
        2: "期限、対象、詳細な指示が全て明確",
        3: "対象、手段、期限が全て明確で即座に行動に移せる",
        4: "対象（会社名・担当者名）、期限、手段（電話）、内容が具体的",
        5: "期間、対象、形式（Excel）、宛先、手段（メール）が全て明確",
        6: "対象（会議室）、時間、期限が明確で即座に予約可能",
        7: "対象、数量、期限、配布先が全て具体的"
    }
    
    for i in range(count):
        pattern_idx = random.randint(0, len(patterns) - 1)
        text = patterns[pattern_idx]()
        
        instructions.append({
            "text": text,
            "label": 0,
            "reason": reasons_map.get(pattern_idx, "What と When が両方明示されており、即座に行動に移せる")
        })
    
    return instructions

def split_dataset(data: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """データセットを train/val/test に分割（stratified）"""
    random.seed(seed)
    
    # ラベル別に分離
    label_0 = [item for item in data if item['label'] == 0]
    label_1 = [item for item in data if item['label'] == 1]
    
    # シャッフル
    random.shuffle(label_0)
    random.shuffle(label_1)
    
    # 各ラベルを分割
    def split_list(lst, ratios):
        n = len(lst)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    train_0, val_0, test_0 = split_list(label_0, (train_ratio, val_ratio, test_ratio))
    train_1, val_1, test_1 = split_list(label_1, (train_ratio, val_ratio, test_ratio))
    
    # 結合してシャッフル
    train_data = train_0 + train_1
    val_data = val_0 + val_1
    test_data = test_0 + test_1
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def main():
    print("=" * 60)
    print("ビジネス指示文データセット生成開始")
    print("=" * 60)
    
    # データ生成
    print("\n曖昧な指示（Label 1）を生成中...")
    ambiguous_data = generate_ambiguous_instructions(1000)
    print(f"✓ 1,000件生成完了")
    
    print("\n明確な指示（Label 0）を生成中...")
    clear_data = generate_clear_instructions(1000)
    print(f"✓ 1,000件生成完了")
    
    # 統合とシャッフル
    all_data = ambiguous_data + clear_data
    random.shuffle(all_data)
    
    # dataset.jsonl に保存
    print("\n全データを dataset.jsonl に保存中...")
    with open('dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ dataset.jsonl 保存完了（{len(all_data)}件）")
    
    # train/val/test に分割
    print("\nデータを train/val/test に分割中...")
    train_data, val_data, test_data = split_dataset(all_data)
    
    # 各ファイルに保存
    datasets = [
        ('train.jsonl', train_data),
        ('val.jsonl', val_data),
        ('test.jsonl', test_data)
    ]
    
    for filename, dataset in datasets:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        label_0_count = sum(1 for x in dataset if x['label'] == 0)
        label_1_count = sum(1 for x in dataset if x['label'] == 1)
        print(f"✓ {filename}: {len(dataset)}件（明確: {label_0_count}, 曖昧: {label_1_count}）")
    
    # 統計情報表示
    print("\n" + "=" * 60)
    print("データ生成完了！")
    print("=" * 60)
    print(f"総データ数: {len(all_data)} 件")
    print(f"  - Label 0 (明確): {sum(1 for x in all_data if x['label'] == 0)} 件")
    print(f"  - Label 1 (曖昧): {sum(1 for x in all_data if x['label'] == 1)} 件")
    print(f"\n分割:")
    print(f"  - 学習データ (train.jsonl): {len(train_data)} 件")
    print(f"  - 検証データ (val.jsonl): {len(val_data)} 件")
    print(f"  - テストデータ (test.jsonl): {len(test_data)} 件")
    
    # サンプル表示
    print("\n" + "=" * 60)
    print("サンプル表示")
    print("=" * 60)
    print("\n【明確な指示 (Label 0) サンプル】")
    for item in [x for x in all_data if x['label'] == 0][:3]:
        print(f"  - {item['text']}")
    
    print("\n【曖昧な指示 (Label 1) サンプル】")
    for item in [x for x in all_data if x['label'] == 1][:3]:
        print(f"  - {item['text']}")

if __name__ == '__main__':
    main()


