#!/usr/bin/env python3
"""
classify_kaggle.py – Combine all Kaggle football CSVs into one unified dataset,
classify into 'women' vs 'other' (predominantly men's), and create a test subset.

Outputs:
  data/kaggle_processed/women/articles.csv
  data/kaggle_processed/other/articles.csv
  data/kaggle_processed/test/women_articles.csv   (up to 1000)
  data/kaggle_processed/test/other_articles.csv    (1000)
  data/kaggle_processed/stats.txt

Usage:
    python final_codes/classify_kaggle.py
    python final_codes/classify_kaggle.py --kaggle-dir data/kaggle_data_football
    python final_codes/classify_kaggle.py --test-size 500
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# ─── Women's football keyword patterns ────────────────────────────────────────
# Compiled regex for speed; case-insensitive
WOMEN_PATTERNS = re.compile(
    r"""
    women'?s\s+(?:football|soccer|world\s+cup|champions\s+league|super\s+league|euros?|
                   european\s+championship|game|match|team|national\s+team|final|
                   semi.?final|quarter.?final|cup)|
    \bwsl\b|
    \bnwsl\b|
    \buwcl\b|
    \blionesses?\b|
    \buswnt\b|
    \bcanwnt\b|
    \bmatildas\b|
    \bfemeni\b|
    \bfemenino\b|
    \bfeminines?\b|
    \bfrauen\b|
    \bshe\s*believes?\s*cup\b|
    \barnold\s+clark\s+cup\b|
    \bliga\s+f\b|
    \bw[\-\s]?league\b|
    \bwomen'?s\s+premier\b|
    \bbarclays\s+wsl\b|
    women'?s\s+euro|
    \bfifa\s+women\b|
    \bwomen'?s\s+fa\s+cup\b|
    \bsuper\s+falcons\b|
    \bbanyana\s+banyana\b|
    \bfootball\s+ferns\b|
    \breggae\s+girlz\b|
    \bnadeshiko\b|
    \bgotham\s+fc\b|
    \bportland\s+thorns\b|
    \bkansas\s+city\s+current\b|
    \bsan\s+diego\s+wave\b|
    \bwashington\s+spirit\b|
    \bchicago\s+red\s+stars\b|
    \bhouston\s+dash\b|
    \bOL\s+reign\b|
    \bangel\s+city\b|
    \bnorth\s+carolina\s+courage\b|
    \borlando\s+pride\b|
    (?:chelsea|arsenal|manchester\s+city|man\s+city|liverpool|barcelona|
       real\s+madrid|bayern|lyon|psg|wolfsburg|tottenham)\s+wom[ae]n
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ─── CSV Schema Mapping ──────────────────────────────────────────────────────
# Each CSV has a slightly different schema; we normalise to a common one.

# Target columns: article_id, title, source, author, publish_time, body_text, link

SCHEMA_MAP = {
    # filename -> {original_col: target_col}
    "final-articles.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
        "publish-time": "publish_time",
        "source": "source",
    },
    "goal-news.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
        "publish-time": "publish_time",
        "source": "source",
    },
    "df_analyst.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
        "publish_time": "publish_time",
    },
    "skysports.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
    },
    "tribuna_articles.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
        "source": "source",
        "publish_time": "publish_time",
    },
    "allfootball.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "publish_time": "publish_time",
        "author": "author",
    },
    "live_mint.csv": {
        "title": "title",
        "content": "body_text",
        "link": "link",
        "author": "author",
        "publish_time": "publish_time",
    },
}


def load_and_normalise(kaggle_dir: Path) -> pd.DataFrame:
    """Load all CSVs, normalise column names, concatenate."""
    all_dfs = []

    for filename, col_map in SCHEMA_MAP.items():
        fpath = kaggle_dir / filename
        if not fpath.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        try:
            df = pd.read_csv(fpath, on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"  [ERROR] Failed to load {filename}: {e}")
            continue

        # Drop unnamed index columns
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        # Rename columns
        rename_dict = {}
        for orig_col, target_col in col_map.items():
            # Handle case-insensitive matching
            for c in df.columns:
                if c.lower().strip() == orig_col.lower().strip():
                    rename_dict[c] = target_col
                    break
        df = df.rename(columns=rename_dict)

        # Ensure all target columns exist
        for target_col in ["title", "body_text", "link", "author", "publish_time", "source"]:
            if target_col not in df.columns:
                df[target_col] = ""

        # Add source tag if missing
        if df["source"].isna().all() or (df["source"] == "").all():
            df["source"] = filename.replace(".csv", "")

        df = df[["title", "body_text", "link", "author", "publish_time", "source"]]
        print(f"  [OK] {filename}: {len(df)} rows")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No CSVs loaded!")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop duplicates on title + link
    before = len(combined)
    combined = combined.drop_duplicates(subset=["title", "link"], keep="first")
    after = len(combined)
    print(f"\n  Deduplication: {before} -> {after} ({before - after} duplicates removed)")

    # Add article_id
    combined.insert(0, "article_id", range(1, len(combined) + 1))

    # Drop rows with no body text
    combined = combined.dropna(subset=["body_text"])
    combined = combined[combined["body_text"].str.strip().str.len() > 50]
    print(f"  After filtering empty/short articles: {len(combined)} rows")

    return combined.reset_index(drop=True)


def classify_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'category' column: 'women' or 'other'."""

    def _is_women(row):
        text = f"{row.get('title', '')} {row.get('body_text', '')}"
        if not isinstance(text, str):
            return False
        return bool(WOMEN_PATTERNS.search(text))

    print("\n  Classifying articles (women vs other)...")
    df["category"] = df.apply(lambda r: "women" if _is_women(r) else "other", axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Combine & classify Kaggle football CSVs")
    parser.add_argument("--kaggle-dir", type=str,
                        default="data/kaggle_data_football",
                        help="Path to folder with raw Kaggle CSVs")
    parser.add_argument("--output-dir", type=str,
                        default="data/kaggle_processed",
                        help="Output directory")
    parser.add_argument("--test-size", type=int, default=1000,
                        help="Number of articles per category for test subset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    output_dir = Path(args.output_dir)

    print(f"{'='*60}")
    print("Step 1: Loading & normalising CSVs")
    print(f"{'='*60}")
    combined = load_and_normalise(kaggle_dir)

    print(f"\n{'='*60}")
    print("Step 2: Classifying articles")
    print(f"{'='*60}")
    combined = classify_gender(combined)

    women_df = combined[combined["category"] == "women"].copy()
    other_df = combined[combined["category"] == "other"].copy()

    print(f"\n  Results:")
    print(f"    Women:  {len(women_df)} articles")
    print(f"    Other:  {len(other_df)} articles")
    print(f"    Total:  {len(combined)} articles")

    # Source breakdown — ensure source is string
    combined["source"] = combined["source"].fillna("unknown").astype(str)
    print(f"\n  Per-source breakdown:")
    for src in combined["source"].unique():
        src_df = combined[combined["source"] == src]
        w = len(src_df[src_df["category"] == "women"])
        o = len(src_df[src_df["category"] == "other"])
        print(f"    {src:30s}  women={w:5d}  other={o:5d}  total={w+o:5d}")

    # ─── Save full datasets ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3: Saving outputs")
    print(f"{'='*60}")

    women_dir = output_dir / "women"
    other_dir = output_dir / "other"
    test_dir = output_dir / "test"

    for d in [women_dir, other_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save full splits (rename category columns for compatibility with 7_run_ner.py)
    women_out = women_df.drop(columns=["category"])
    other_out = other_df.drop(columns=["category"])

    women_out.to_csv(women_dir / "articles.csv", index=False)
    other_out.to_csv(other_dir / "articles.csv", index=False)
    print(f"  Saved {len(women_out)} women articles -> {women_dir / 'articles.csv'}")
    print(f"  Saved {len(other_out)} other articles -> {other_dir / 'articles.csv'}")

    # ─── Create test subset ──────────────────────────────────────────────────
    n_women_test = min(args.test_size, len(women_df))
    n_other_test = min(args.test_size, len(other_df))

    women_test = women_df.sample(n=n_women_test, random_state=args.seed).drop(columns=["category"])
    other_test = other_df.sample(n=n_other_test, random_state=args.seed).drop(columns=["category"])

    # Save with names expected by 7_run_ner.py
    women_test.to_csv(test_dir / "women_articles.csv", index=False)
    other_test.to_csv(test_dir / "other_articles.csv", index=False)
    print(f"  Saved {len(women_test)} women test articles -> {test_dir / 'women_articles.csv'}")
    print(f"  Saved {len(other_test)} other test articles -> {test_dir / 'other_articles.csv'}")

    # ─── Stats file ──────────────────────────────────────────────────────────
    stats_path = output_dir / "stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Total articles: {len(combined)}\n")
        f.write(f"Women articles: {len(women_df)}\n")
        f.write(f"Other articles: {len(other_df)}\n")
        f.write(f"Test women: {len(women_test)}\n")
        f.write(f"Test other: {len(other_test)}\n")
        f.write(f"\nPer-source breakdown:\n")
        for src in combined["source"].unique():
            src_df = combined[combined["source"] == src]
            w = len(src_df[src_df["category"] == "women"])
            o = len(src_df[src_df["category"] == "other"])
            f.write(f"  {src:30s}  women={w:5d}  other={o:5d}\n")
    print(f"  Stats saved to {stats_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
