\
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.x_analyzer import (
    add_auto_tags,
    add_metrics,
    extract_winning_clues,
    format_percent_columns,
    load_csv_files,
    propose_next_posts,
    summarize_by_tag,
)

st.set_page_config(
    page_title="水無月絵空 X運用OS",
    page_icon="📈",
    layout="wide",
)

st.title("水無月絵空 X運用OS")
st.caption("XアナリティクスCSVを読み込み、投稿タイプ別に勝ち筋を見える化するダッシュボード。")

with st.sidebar:
    st.header("CSVアップロード")
    files = st.file_uploader(
        "XアナリティクスCSVを複数選択",
        type=["csv"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    st.subheader("見るべき指標")
    st.write("1. プロフィールアクセス率")
    st.write("2. 返信率")
    st.write("3. 詳細クリック率")
    st.write("4. リポスト率")
    st.write("5. ブックマーク率")

if not files:
    st.info("まずCSVをアップロードしてください。`account_analytics_content_*.csv` が最もおすすめです。")
    st.stop()

raw = load_csv_files(files)
df = add_auto_tags(add_metrics(raw))

content_df = df[df.get("source_kind", "") == "content"].copy()
if content_df.empty:
    content_df = df.copy()

st.subheader("全体サマリー")
total_impressions = int(content_df["impressions"].sum())
total_engagements = int(content_df["engagements"].sum())
total_profile = int(content_df["profile_visits"].sum())
total_detail = int(content_df["detail_clicks"].sum())
total_replies = int(content_df["replies"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("インプレッション", f"{total_impressions:,}")
c2.metric("エンゲージメント", f"{total_engagements:,}")
c3.metric("プロフィールアクセス", f"{total_profile:,}")
c4.metric("詳細クリック", f"{total_detail:,}")
c5.metric("返信", f"{total_replies:,}")

st.markdown("---")

st.subheader("投稿タイプを確認・修正")
edit_cols = [
    "date",
    "text",
    "auto_tag",
    "manual_tag",
    "impressions",
    "engagements",
    "likes",
    "replies",
    "reposts",
    "bookmarks",
    "profile_visits",
    "detail_clicks",
]
view = content_df[[c for c in edit_cols if c in content_df.columns]].sort_values("impressions", ascending=False).head(200)

edited = st.data_editor(
    view,
    use_container_width=True,
    height=420,
    column_config={
        "manual_tag": st.column_config.SelectboxColumn(
            "manual_tag",
            options=[
                "体型比較",
                "価格ツッコミ",
                "女性誌レビュー風",
                "ストッキング図鑑",
                "国別比較",
                "スナップ写真",
                "SNS考察",
                "未分類",
            ],
        )
    },
    disabled=[c for c in view.columns if c != "manual_tag"],
)

# 画面上では上位200件の手動タグを反映
content_df = content_df.copy()
for idx, row in edited.iterrows():
    if idx in content_df.index:
        content_df.at[idx, "manual_tag"] = row["manual_tag"]

st.markdown("---")

st.subheader("タグ別比較")
tag_summary = summarize_by_tag(content_df)
if not tag_summary.empty:
    tag_view = format_percent_columns(tag_summary)
    st.dataframe(tag_view, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            tag_summary,
            x="manual_tag",
            y="impressions",
            title="タグ別インプレッション",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            tag_summary,
            x="manual_tag",
            y="profile_visit_rate",
            title="タグ別プロフィールアクセス率",
            text_auto=".3%",
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("勝ち投稿TOP20")
top_posts = content_df.sort_values("impressions", ascending=False).head(20)
top_cols = [
    "manual_tag",
    "impressions",
    "engagements",
    "profile_visits",
    "detail_clicks",
    "replies",
    "reposts",
    "bookmarks",
    "engagement_rate",
    "profile_visit_rate",
    "reply_rate",
    "detail_click_rate",
    "text",
    "link",
]
st.dataframe(format_percent_columns(top_posts[[c for c in top_cols if c in top_posts.columns]]), use_container_width=True)

st.markdown("---")

st.subheader("勝ち筋メモ")
for clue in extract_winning_clues(content_df):
    st.write("・" + clue)

st.subheader("次に作る投稿案")
for i, idea in enumerate(propose_next_posts(content_df), start=1):
    with st.expander(f"{i}. {idea['theme']}", expanded=i == 1):
        st.write("**狙い**:", idea["why"])
        st.write("**本文フック**:", idea["hook"])
        st.write("**画像設計**:", idea["image"])

st.markdown("---")

st.subheader("投稿案メモ")
memo = st.text_area(
    "次に試したい投稿テーマ・画像プロンプト・リプ誘導を書き留める",
    height=180,
    placeholder="例：O型派の理由を5分類する女性誌風図鑑。CTAは『O型派、理由を一言で教えて』",
)
st.download_button(
    "メモをMarkdownで保存",
    data=memo,
    file_name="next_post_memo.md",
    mime="text/markdown",
)
