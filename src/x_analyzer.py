\
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


CONTENT_COLUMNS = {
    "post_id": ["ポストID", "post_id", "Post ID"],
    "date": ["日付", "Date", "date"],
    "text": ["ポスト本文", "本文", "text", "Post text"],
    "link": ["ポストのリンク", "link", "URL"],
    "impressions": ["インプレッション数", "impressions", "Impressions"],
    "likes": ["いいね", "likes", "Likes"],
    "engagements": ["エンゲージメント", "engagements", "Engagements"],
    "bookmarks": ["ブックマーク", "bookmarks", "Bookmarks"],
    "shares": ["共有された回数\\", "共有された回数", "shares", "Shares"],
    "new_follows": ["新しいフォロー", "new_follows", "New follows"],
    "replies": ["返信", "replies", "Replies"],
    "reposts": ["リポスト", "reposts", "Reposts"],
    "profile_visits": ["プロフィールへのアクセス数", "profile_visits", "Profile visits"],
    "detail_clicks": ["詳細のクリック数", "detail_clicks", "Detail clicks"],
    "url_clicks": ["URLのクリック数", "url_clicks", "URL clicks"],
    "hashtag_clicks": ["ハッシュタグのクリック数", "hashtag_clicks", "Hashtag clicks"],
    "permalink_clicks": ["パーマリンクのクリック数", "permalink_clicks", "Permalink clicks"],
}

OVERVIEW_COLUMNS = {
    "date": ["Date", "日付", "date"],
    "impressions": ["インプレッション数", "impressions"],
    "likes": ["いいね", "likes"],
    "engagements": ["エンゲージメント", "engagements"],
    "bookmarks": ["ブックマーク", "bookmarks"],
    "shares": ["共有された回数\\", "共有された回数", "shares"],
    "new_follows": ["新しいフォロー", "new_follows"],
    "unfollows": ["フォロー解除", "unfollows"],
    "replies": ["返信", "replies"],
    "reposts": ["リポスト", "reposts"],
    "profile_visits": ["プロフィールへのアクセス数", "profile_visits"],
    "posts_created": ["ポストを作成", "posts_created"],
    "video_views": ["動画再生数", "video_views"],
    "media_views": ["メディアの再生数", "media_views"],
}

NUMERIC_COLUMNS = [
    "impressions",
    "likes",
    "engagements",
    "bookmarks",
    "shares",
    "new_follows",
    "unfollows",
    "replies",
    "reposts",
    "profile_visits",
    "detail_clicks",
    "url_clicks",
    "hashtag_clicks",
    "permalink_clicks",
    "posts_created",
    "video_views",
    "media_views",
]

TAG_RULES: list[tuple[str, list[str]]] = [
    ("体型比較", ["体型", "ぽっちゃり", "細め", "スレンダー", "O型", "A型", "X型", "I型", "骨格"]),
    ("価格ツッコミ", ["高くない", "価格", "値段", "円", "¥", "いくら", "高すぎ", "安い"]),
    ("女性誌レビュー風", ["女性誌", "レビュー", "トレンド", "流行", "カラー", "コーデ", "ファッション"]),
    ("ストッキング図鑑", ["ストッキング", "黒スト", "デニール", "パンスト", "タイツ", "光沢"]),
    ("国別比較", ["日本", "韓国", "中国", "台湾", "タイ", "ロシア", "フランス", "アメリカ", "インド", "イタリア"]),
    ("スナップ写真", ["スナップ", "自撮り", "写真", "街", "カフェ", "オフィス", "ライブ配信"]),
    ("SNS考察", ["アルゴ", "X", "Premium", "インプ", "収益", "Grok", "Phoenix"]),
]


def _strip_bom_and_spaces(value: str) -> str:
    return value.replace("\ufeff", "").strip()


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {_strip_bom_and_spaces(str(c)): c for c in df.columns}
    for c in candidates:
        if c in normalized:
            return normalized[c]
    return None


def normalize_columns(df: pd.DataFrame, kind: str = "content") -> pd.DataFrame:
    mapping = CONTENT_COLUMNS if kind == "content" else OVERVIEW_COLUMNS
    rename_map: dict[str, str] = {}

    for canonical, candidates in mapping.items():
        actual = _find_col(df, candidates)
        if actual is not None:
            rename_map[actual] = canonical

    out = df.rename(columns=rename_map).copy()

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("—", "0", regex=False)
                .str.replace("-", "0", regex=False)
            )
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float)

    if "date" in out.columns:
        out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce", utc=True)

    if "text" not in out.columns:
        out["text"] = ""

    return out


def load_csv_files(files: list) -> pd.DataFrame:
    frames = []
    for file in files:
        df = pd.read_csv(file)
        # content CSV has post body; overview does not.
        kind = "content" if "ポスト本文" in df.columns or "Post text" in df.columns else "overview"
        norm = normalize_columns(df, kind=kind)
        norm["source_kind"] = kind
        frames.append(norm)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    if "post_id" in out.columns:
        out = out.sort_values("impressions", ascending=False).drop_duplicates(subset=["post_id"], keep="first")
    return out


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["impressions", "engagements", "likes", "replies", "reposts", "bookmarks",
                "profile_visits", "detail_clicks", "new_follows"]:
        if col not in out.columns:
            out[col] = 0.0

    out["engagement_rate"] = safe_div(out["engagements"], out["impressions"])
    out["like_rate"] = safe_div(out["likes"], out["impressions"])
    out["reply_rate"] = safe_div(out["replies"], out["impressions"])
    out["repost_rate"] = safe_div(out["reposts"], out["impressions"])
    out["bookmark_rate"] = safe_div(out["bookmarks"], out["impressions"])
    out["profile_visit_rate"] = safe_div(out["profile_visits"], out["impressions"])
    out["detail_click_rate"] = safe_div(out["detail_clicks"], out["impressions"])
    out["follow_rate"] = safe_div(out["new_follows"], out["impressions"])

    # 水無月絵空向けの仮スコア。会話・深掘り・プロフィール導線を重めにする。
    out["conversation_score"] = (
        out["reply_rate"] * 35
        + out["repost_rate"] * 25
        + out["bookmark_rate"] * 20
        + out["profile_visit_rate"] * 15
        + out["detail_click_rate"] * 5
    )
    out["monetize_hint_score"] = (
        out["profile_visit_rate"] * 40
        + out["follow_rate"] * 30
        + out["detail_click_rate"] * 20
        + out["reply_rate"] * 10
    )
    return out


def auto_tag(text: str) -> str:
    if not isinstance(text, str):
        return "未分類"
    scores: dict[str, int] = {}
    for tag, keywords in TAG_RULES:
        scores[tag] = sum(1 for kw in keywords if kw.lower() in text.lower())
    best_tag, best_score = max(scores.items(), key=lambda item: item[1])
    return best_tag if best_score > 0 else "未分類"


def add_auto_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["auto_tag"] = out["text"].fillna("").map(auto_tag)
    if "manual_tag" not in out.columns:
        out["manual_tag"] = out["auto_tag"]
    return out


def summarize_by_tag(df: pd.DataFrame, tag_col: str = "manual_tag") -> pd.DataFrame:
    if df.empty or tag_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(tag_col, dropna=False).agg(
        posts=("text", "count"),
        impressions=("impressions", "sum"),
        engagements=("engagements", "sum"),
        likes=("likes", "sum"),
        replies=("replies", "sum"),
        reposts=("reposts", "sum"),
        bookmarks=("bookmarks", "sum"),
        profile_visits=("profile_visits", "sum"),
        detail_clicks=("detail_clicks", "sum"),
        new_follows=("new_follows", "sum"),
        avg_conversation_score=("conversation_score", "mean"),
        avg_monetize_hint_score=("monetize_hint_score", "mean"),
    ).reset_index()

    grouped["engagement_rate"] = grouped["engagements"] / grouped["impressions"].replace(0, np.nan)
    grouped["profile_visit_rate"] = grouped["profile_visits"] / grouped["impressions"].replace(0, np.nan)
    grouped["reply_rate"] = grouped["replies"] / grouped["impressions"].replace(0, np.nan)
    grouped = grouped.fillna(0).sort_values("impressions", ascending=False)
    return grouped


def extract_winning_clues(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["CSVを読み込むと、勝ちパターンをここに表示します。"]

    top = df.sort_values("impressions", ascending=False).head(20)
    clues: list[str] = []

    top_tags = top["manual_tag"].value_counts().head(3).to_dict()
    if top_tags:
        clues.append("上位投稿で多い型: " + " / ".join([f"{k}({v}本)" for k, v in top_tags.items()]))

    if top["reply_rate"].mean() > df["reply_rate"].mean():
        clues.append("上位投稿は返信率が平均より高め。選択型・質問型・賛否型を継続。")
    if top["profile_visit_rate"].mean() > df["profile_visit_rate"].mean():
        clues.append("上位投稿はプロフィールアクセス率が高め。価格違和感・商品UI・続きが気になる設計が有効。")
    if top["detail_click_rate"].mean() > df["detail_click_rate"].mean():
        clues.append("上位投稿は詳細クリック率が高め。画像内情報量や“確認したくなる違和感”が効いている可能性。")
    if top["bookmark_rate"].mean() > df["bookmark_rate"].mean():
        clues.append("上位投稿はブックマーク率が高め。図鑑・比較・保存用UIを強化。")

    if not clues:
        clues.append("現時点では明確な差が薄い。手動タグを付けてから再分析してください。")
    return clues


def propose_next_posts(df: pd.DataFrame) -> list[dict[str, str]]:
    ideas = [
        {
            "theme": "O型/ぽっちゃり深掘り図鑑",
            "why": "体型比較の返信を続編化しやすい。会話型バズ向き。",
            "hook": "O型派が多すぎる。結局どこが刺さってるの？",
            "image": "女性誌診断風・黒スト脚/腰回り/服映えの5理由カード",
        },
        {
            "theme": "価格帯4比較",
            "why": "価格ツッコミは詳細クリック・プロフィールアクセスを誘発しやすい。",
            "hook": "上下いくらまでなら出せる？ 980円/3,980円/15,900円/159,500円",
            "image": "架空EC商品ページ風・4価格帯比較UI",
        },
        {
            "theme": "ストッキング10デニール/30デニール比較",
            "why": "既存フォロワー文脈と保存率が噛み合う。",
            "hook": "同じ黒ストでも印象違いすぎる。何デニールが一番好き？",
            "image": "3〜5段の脚元比較・女性誌レビュー風",
        },
        {
            "theme": "国別メイク/髪型バストアップ比較",
            "why": "顔差分が明確で、保存・引用・好みコメントが取りやすい。",
            "hook": "国別メイク、どれが一番刺さる？",
            "image": "10カ国比較・バストアップ・髪型とメイクが主役",
        },
        {
            "theme": "コメント欄分析スクショ風",
            "why": "新アルゴ向けに会話継続を作りやすい。",
            "hook": "コメント欄を読んだら、好みが3派に分かれてた。",
            "image": "Xコメント分析風・3派閥マップ",
        },
    ]

    if df.empty:
        return ideas

    tag_summary = summarize_by_tag(df)
    if not tag_summary.empty:
        top_tag = str(tag_summary.iloc[0]["manual_tag"])
        ideas.insert(0, {
            "theme": f"{top_tag}の続編",
            "why": f"直近データでは「{top_tag}」が最もインプレッションを集めています。",
            "hook": f"{top_tag}、コメント欄を見ると意見が割れてる。",
            "image": f"{top_tag}の深掘り図鑑・比較・コメント欄分析UI",
        })
    return ideas[:6]


def format_percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [c for c in out.columns if c.endswith("_rate")]:
        out[col] = (out[col] * 100).round(3)
    return out
