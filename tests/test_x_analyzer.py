\
import pandas as pd

from src.x_analyzer import add_auto_tags, add_metrics, auto_tag, normalize_columns, summarize_by_tag


def test_auto_tag_body_type():
    assert auto_tag("1番どの体型が好き？細め？ぽっちゃり？") == "体型比較"


def test_auto_tag_price():
    assert auto_tag("今このファッションが流行ってるみたいだよ！高くない？？ ¥159,500") == "価格ツッコミ"


def test_metrics_rates():
    df = pd.DataFrame({
        "インプレッション数": [1000],
        "エンゲージメント": [100],
        "いいね": [10],
        "返信": [5],
        "リポスト": [2],
        "プロフィールへのアクセス数": [20],
        "詳細のクリック数": [30],
        "ポスト本文": ["黒ストのデニール比較"],
    })
    norm = normalize_columns(df)
    out = add_auto_tags(add_metrics(norm))
    assert round(float(out.loc[0, "engagement_rate"]), 3) == 0.1
    assert out.loc[0, "auto_tag"] == "ストッキング図鑑"


def test_summarize_by_tag():
    df = pd.DataFrame({
        "text": ["A", "B"],
        "manual_tag": ["体型比較", "体型比較"],
        "impressions": [100, 300],
        "engagements": [10, 30],
        "likes": [1, 3],
        "replies": [1, 3],
        "reposts": [0, 1],
        "bookmarks": [0, 2],
        "profile_visits": [2, 6],
        "detail_clicks": [5, 15],
        "new_follows": [0, 1],
        "conversation_score": [0.1, 0.2],
        "monetize_hint_score": [0.1, 0.2],
    })
    summary = summarize_by_tag(df)
    assert int(summary.loc[0, "impressions"]) == 400
