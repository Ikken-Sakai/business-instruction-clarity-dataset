import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 利用可能な日本語フォントを検索
print("=== 利用可能な日本語フォント ===")
japanese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Japan' in font.name:
        japanese_fonts.append((font.name, font.fname))
        print(f"{font.name}: {font.fname}")

if not japanese_fonts:
    print("❌ 日本語フォントが見つかりませんでした")
else:
    # 最初の日本語フォントでテスト
    font_name = japanese_fonts[0][0]
    print(f"\n=== {font_name} でテスト ===")
    
    # フォント設定
    plt.rcParams['font.family'] = font_name
    
    # テスト描画
    fig, ax = plt.subplots(figsize=(8, 4))
    test_text = ['明確', '曖昧', '会議', '明日', 'プロジェクト']
    ax.barh(range(len(test_text)), [5, 4, 3, 2, 1])
    ax.set_yticks(range(len(test_text)))
    ax.set_yticklabels(test_text)
    ax.set_title('日本語フォントテスト')
    ax.set_xlabel('テスト値')
    
    plt.tight_layout()
    plt.savefig('/home/ike/Desktop/人工知能応用データセット/figures/font_test.png', dpi=150)
    print(f"✅ テスト画像を保存: figures/font_test.png")
    plt.close()










