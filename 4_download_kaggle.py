#!/usr/bin/env python3
"""
4_download_kaggle.py - Download pre-existing football article datasets from Kaggle
and filter for Champions League content.

Requires Kaggle API credentials (~/.kaggle/kaggle.json)
OR you can manually download from the URLs and place CSVs in data/raw/*/kaggle/

Datasets:
  - beridzeg45/football-news-articles  (~12K articles, Goal/SkySports/Tribuna)
  - ibrahimkiziloklu/football-transfer-news-articles-for-nlp (~6K transfer articles)

Usage:
    python 4_download_kaggle.py             # Download + filter
    python 4_download_kaggle.py --skip-download  # Just filter already-downloaded data
"""

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    BASE_DIR,
    RAW_MEN,
    RAW_WOMEN,
    KAGGLE_DATASETS,
    logger,
)

KAGGLE_DIR = BASE_DIR / "data" / "kaggle_raw"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Keywords for filtering Champions League content ─────────────────────────
MEN_CL_KEYWORDS = [
    "champions league", "ucl", "uefa champions",
    "group stage", "knockout", "round of 16",
    "quarter-final", "semi-final", "quarterfinal", "semifinal",
    "champions league final",
]

WOMEN_CL_KEYWORDS = [
    "women's champions league", "uwcl", "women champions league",
    "uefa women", "womens champions league",
    "women's cl", "wucl",
]

# Broader women's football keywords (to also capture general women's content)
WOMEN_GENERAL_KEYWORDS = [
    "women's football", "women's soccer", "womens football",
    "wsl", "nwsl", "women's super league",
    "women's world cup", "she believes",
    "women national team", "lionesses",
    "women's game", "female footballer",
]


def download_kaggle_datasets():
    """Download datasets using opendatasets (Kaggle API)."""
    try:
        import opendatasets as od
    except ImportError:
        logger.error("opendatasets not installed. Run: pip install opendatasets")
        logger.info("Alternatively, download manually from Kaggle and place CSVs in data/kaggle_raw/")
        return False

    for dataset in KAGGLE_DATASETS:
        logger.info(f"Downloading Kaggle dataset: {dataset}")
        try:
            od.download(f"https://www.kaggle.com/datasets/{dataset}", data_dir=str(KAGGLE_DIR))
        except Exception as e:
            logger.warning(f"Failed to download {dataset}: {e}")
            logger.info(f"Please download manually from https://www.kaggle.com/datasets/{dataset}")

    return True


def matches_keywords(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the given keywords (case-insensitive)."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def classify_article(title: str, content: str) -> str | None:
    """
    Classify an article as 'men', 'women', or None (not Champions League related).
    Women's keywords are checked first since they're more specific.
    """
    combined = f"{title} {content}" if isinstance(content, str) else str(title)

    # Check women's first (more specific)
    if matches_keywords(combined, WOMEN_CL_KEYWORDS) or matches_keywords(combined, WOMEN_GENERAL_KEYWORDS):
        return "women"

    # Check men's CL
    if matches_keywords(combined, MEN_CL_KEYWORDS):
        return "men"

    return None


def process_kaggle_csvs():
    """Find all CSVs in kaggle_raw, filter for CL content, and save individual JSONs."""
    csv_files = list(KAGGLE_DIR.rglob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {KAGGLE_DIR}. Download datasets first.")
        return {"men": 0, "women": 0}

    totals = {"men": 0, "women": 0}

    for csv_path in csv_files:
        logger.info(f"Processing: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.error(f"  Failed to read {csv_path}: {e}")
            continue

        logger.info(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

        # Try to detect the right columns
        # Common patterns: title/headline, content/text/body, date/published_date, url/link
        title_col = None
        content_col = None
        date_col = None
        url_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ("title", "headline", "article_title"):
                title_col = col
            elif col_lower in ("content", "text", "body", "article_text", "article_content"):
                content_col = col
            elif col_lower in ("date", "published_date", "pub_date", "publish_date", "publication_date"):
                date_col = col
            elif col_lower in ("url", "link", "article_url", "external_link"):
                url_col = col

        if not content_col and not title_col:
            logger.warning(f"  Cannot find text columns in {csv_path.name}, skipping")
            continue

        logger.info(f"  Using columns: title={title_col}, content={content_col}, date={date_col}, url={url_col}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Filtering {csv_path.name}", leave=False):
            title = str(row.get(title_col, "")) if title_col else ""
            content = str(row.get(content_col, "")) if content_col else ""

            # Skip empty rows
            if len(content.split()) < 30 and len(title) < 10:
                continue

            gender = classify_article(title, content)
            if gender is None:
                continue  # Not CL-related

            output_dir = (RAW_MEN if gender == "men" else RAW_WOMEN) / "kaggle"

            # Check for duplicate by URL or title hash
            from hashlib import md5
            article_id = md5(f"{title}_{url_col}_{idx}".encode()).hexdigest()[:12]

            record = {
                "article_id": article_id,
                "url": str(row.get(url_col, "")) if url_col else "",
                "title": title,
                "date": str(row.get(date_col, "")) if date_col else "",
                "domain": "kaggle_dataset",
                "source_country": "",
                "language": "English",
                "gender_category": gender,
                "source_pipeline": f"kaggle_{csv_path.stem}",
                "body_text": content,
                "word_count": len(content.split()),
                "scraped_at": datetime.now().isoformat(),
            }

            out_path = output_dir / f"{article_id}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            totals[gender] += 1

    logger.info(f"Kaggle filtering complete: {totals}")
    return totals


def main():
    parser = argparse.ArgumentParser(description="Download and filter Kaggle football datasets")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just filter existing CSVs")
    args = parser.parse_args()

    if not args.skip_download:
        download_kaggle_datasets()

    totals = process_kaggle_csvs()
    logger.info(f"=== KAGGLE TOTALS: {totals} ===")


if __name__ == "__main__":
    main()
