#!/usr/bin/env python3
"""
prepare_final_inputs.py — Combine women's data from kaggle_processed and
final_processed into a single file, copy men's data, and create the
final_outputs input directory.
"""

import pandas as pd
from pathlib import Path

BASE = Path("data/kaggle_processed")
FINAL_PROCESSED = Path("data/final_processed/women_articles.csv")
OUT_DIR = BASE / "final_outputs"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Men's / Other articles ─────────────────────────────────
    other = pd.read_csv(BASE / "other" / "articles.csv")
    print(f"Other (men's) articles loaded: {len(other)}")
    other.to_csv(OUT_DIR / "other_articles.csv", index=False)
    print(f"  → Saved to {OUT_DIR / 'other_articles.csv'}")

    # ── 2. Women's articles: combine two sources ──────────────────
    # Source A: kaggle_processed/women/articles.csv
    women_kaggle = pd.read_csv(BASE / "women" / "articles.csv")
    print(f"\nWomen (kaggle_processed): {len(women_kaggle)}")

    # Source B: final_processed/women_articles.csv (different schema)
    women_final = pd.read_csv(FINAL_PROCESSED)
    print(f"Women (final_processed):  {len(women_final)}")

    # Normalise final_processed schema to match kaggle_processed
    women_final_norm = pd.DataFrame({
        "article_id": women_final["article_id"],
        "title": women_final["title"],
        "body_text": women_final["body_text"],
        "link": women_final["url"],
        "author": "",  # not available
        "publish_time": women_final.get("date", ""),
        "source": women_final["domain"].fillna("unknown"),
    })

    # Filter: body_text must be non-empty and > 50 chars
    women_final_norm = women_final_norm[
        women_final_norm["body_text"].fillna("").str.len() > 50
    ]
    print(f"Women (final_processed) after filtering short articles: {len(women_final_norm)}")

    # Combine
    combined = pd.concat([women_kaggle, women_final_norm], ignore_index=True)
    print(f"Combined women's articles (before dedup): {len(combined)}")

    # Deduplicate on title (normalised lowercase stripped)
    combined["_dedup_key"] = combined["title"].fillna("").str.lower().str.strip()
    combined = combined.drop_duplicates(subset="_dedup_key", keep="first")
    combined = combined.drop(columns=["_dedup_key"])

    # Re-assign article_ids to avoid collisions
    combined = combined.reset_index(drop=True)
    combined["article_id"] = range(1, len(combined) + 1)

    print(f"Combined women's articles (after dedup): {len(combined)}")
    combined.to_csv(OUT_DIR / "women_articles.csv", index=False)
    print(f"  → Saved to {OUT_DIR / 'women_articles.csv'}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL INPUT DIRECTORY: {OUT_DIR}")
    print(f"  other_articles.csv: {len(other)} articles")
    print(f"  women_articles.csv: {len(combined)} articles")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
