#!/usr/bin/env python3
"""
5_clean_and_merge.py - Clean, deduplicate, and merge all scraped articles
from every source into final men_articles.csv and women_articles.csv.

Usage:
    python 5_clean_and_merge.py
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from config import RAW_MEN, RAW_WOMEN, PROCESSED_DIR, logger


def strip_html(text: str) -> str:
    """Remove any residual HTML tags from text."""
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        # Fallback: regex strip
        return re.sub(r"<[^>]+>", " ", str(text))


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fix_encoding(text: str) -> str:
    """Fix common encoding artifacts."""
    if not text:
        return ""
    # Normalise unicode
    text = unicodedata.normalize("NFC", text)
    # Common mojibake fixes
    replacements = {
        "\u00e2\u0080\u0099": "'",
        "\u00e2\u0080\u0098": "'",
        "\u00e2\u0080\u009c": '"',
        "\u00e2\u0080\u009d": '"',
        "\u00e2\u0080\u0093": "\u2013",  # en-dash
        "\u00e2\u0080\u0094": "\u2014",  # em-dash
        "\u00e2\u0080\u00a6": "\u2026",  # ellipsis
        "\u00c2": "",
        "\u00a0": " ",  # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_text(text) -> str:
    """Full cleaning pipeline for article body text."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = fix_encoding(text)
    text = normalise_whitespace(text)
    return text


def load_all_articles(raw_dir: Path) -> list[dict]:
    """Load all JSON files from all subdirectories of a raw data dir."""
    articles = []
    for json_file in raw_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            articles.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    return articles


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate articles by:
    1. Exact URL match
    2. Exact title match (case-insensitive)
    3. Near-duplicate body text (first 200 chars match)
    """
    before = len(df)

    # 1. URL dedup
    df = df.drop_duplicates(subset="url", keep="first")

    # 2. Title dedup (case-insensitive)
    df["_title_lower"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(subset="_title_lower", keep="first")
    df = df.drop(columns=["_title_lower"])

    # 3. Body prefix dedup (first 200 chars)
    df["_body_prefix"] = df["body_text"].str[:200].str.lower().str.strip()
    df = df.drop_duplicates(subset="_body_prefix", keep="first")
    df = df.drop(columns=["_body_prefix"])

    after = len(df)
    logger.info(f"  Deduplication: {before} → {after} articles ({before - after} removed)")
    return df


def process_gender(gender: str, raw_dir: Path):
    """Load, clean, dedup, and export articles for one gender."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {gender.upper()} articles from: {raw_dir}")
    logger.info(f"{'='*60}")

    articles = load_all_articles(raw_dir)
    if not articles:
        logger.warning(f"No articles found in {raw_dir}")
        return None

    logger.info(f"Loaded {len(articles)} raw articles")

    # Convert to DataFrame
    df = pd.DataFrame(articles)

    # Source breakdown
    if "source_pipeline" in df.columns:
        logger.info(f"Source breakdown:\n{df['source_pipeline'].value_counts().to_string()}")

    # Clean text
    logger.info("Cleaning text...")
    df["body_text"] = df["body_text"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)

    # Remove articles with very short body text
    min_words = 50
    df["word_count"] = df["body_text"].apply(lambda x: len(x.split()))
    before_filter = len(df)
    df = df[df["word_count"] >= min_words]
    logger.info(f"  Filtered short articles (<{min_words} words): {before_filter} → {len(df)}")

    # Deduplicate
    df = deduplicate(df)

    # Standardise columns
    standard_cols = [
        "article_id", "url", "title", "date", "domain",
        "source_country", "language", "gender_category",
        "source_pipeline", "body_text", "word_count", "scraped_at",
    ]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[standard_cols]

    # Sort by date
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # Export
    out_path = PROCESSED_DIR / f"{gender}_articles.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Exported {len(df)} articles to {out_path}")

    # Also save as pickle for NER pipeline
    pkl_path = PROCESSED_DIR / f"{gender}_articles.pkl"
    df.to_pickle(pkl_path)
    logger.info(f"Exported pickle to {pkl_path}")

    # Summary stats
    print(f"\n--- {gender.upper()} Summary ---")
    print(f"Total articles: {len(df)}")
    print(f"Unique sources: {df['source_pipeline'].nunique()}")
    print(f"Source breakdown: {df['source_pipeline'].value_counts().to_dict()}")
    print(f"Unique domains: {df['domain'].nunique()}")
    print(f"Avg word count: {df['word_count'].mean():.0f}")
    print(f"Median word count: {df['word_count'].median():.0f}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print()

    return df


def main():
    men_df = process_gender("men", RAW_MEN)
    women_df = process_gender("women", RAW_WOMEN)

    # Combined summary
    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    men_count = len(men_df) if men_df is not None else 0
    women_count = len(women_df) if women_df is not None else 0
    print(f"Men's articles:   {men_count}")
    print(f"Women's articles: {women_count}")
    print(f"Total articles:   {men_count + women_count}")

    if men_count > 0 and women_count > 0:
        ratio = men_count / women_count if women_count > 0 else float("inf")
        print(f"Men/Women ratio:  {ratio:.1f}:1")


if __name__ == "__main__":
    main()
