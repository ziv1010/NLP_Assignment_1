#!/usr/bin/env python3
"""
2_scrape_bbc_rss.py - Scrape football articles from BBC Sport RSS feeds,
then extract full text using Trafilatura.

Usage:
    python 2_scrape_bbc_rss.py              # Full scrape
    python 2_scrape_bbc_rss.py --test       # Test (5 articles per feed)
"""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import feedparser
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config

from config import (
    BBC_RSS_FEEDS,
    BBC_DELAY_SECONDS,
    TRAFILATURA_DELAY_SECONDS,
    RAW_MEN,
    RAW_WOMEN,
    logger,
)

# Trafilatura config
TRAF_CONFIG = use_config()
TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "200")
TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")


def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def load_existing_urls(output_dir: Path) -> set[str]:
    """Load URLs from existing JSON files to skip duplicates."""
    seen = set()
    for f in output_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            seen.add(data.get("url", ""))
        except Exception:
            pass
    return seen


def extract_full_text(url: str) -> str | None:
    """Download and extract main article text using Trafilatura."""
    try:
        downloaded = fetch_url(url)
        if downloaded is None:
            return None
        text = extract(
            downloaded,
            config=TRAF_CONFIG,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        return text
    except Exception as e:
        logger.debug(f"Trafilatura failed for {url}: {e}")
        return None


def scrape_rss_feed(feed_url: str, gender: str, output_dir: Path, max_articles: int | None = None):
    """Parse an RSS feed, extract full text for each entry, and save."""
    logger.info(f"[{gender.upper()}] Parsing RSS feed: {feed_url}")

    seen_urls = load_existing_urls(output_dir)
    feed = feedparser.parse(feed_url)

    if not feed.entries:
        logger.warning(f"  No entries found in feed: {feed_url}")
        return 0

    entries = feed.entries[:max_articles] if max_articles else feed.entries
    logger.info(f"  Found {len(feed.entries)} entries, processing {len(entries)}")

    saved = 0
    for entry in entries:
        url = entry.get("link", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Extract full text
        body_text = extract_full_text(url)
        time.sleep(TRAFILATURA_DELAY_SECONDS)

        if not body_text or len(body_text.split()) < 50:
            logger.debug(f"  Skipped (too short): {url}")
            continue

        # Parse date
        pub_date = ""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_date = datetime(*entry.published_parsed[:6]).isoformat()
        elif hasattr(entry, "published"):
            pub_date = entry.published

        article_id = url_to_id(url)
        record = {
            "article_id": article_id,
            "url": url,
            "title": entry.get("title", ""),
            "date": pub_date,
            "domain": "bbc.co.uk",
            "source_country": "United Kingdom",
            "language": "English",
            "gender_category": gender,
            "source_pipeline": "bbc_rss",
            "body_text": body_text,
            "word_count": len(body_text.split()),
            "scraped_at": datetime.now().isoformat(),
        }

        out_path = output_dir / f"{article_id}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        saved += 1

    logger.info(f"[{gender.upper()}] BBC RSS: saved {saved} new articles from {feed_url}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="BBC Sport RSS football scraper")
    parser.add_argument("--test", action="store_true", help="Test mode: 5 articles per feed")
    args = parser.parse_args()

    max_articles = 5 if args.test else None
    totals = {"men": 0, "women": 0}

    for gender, feeds in BBC_RSS_FEEDS.items():
        output_dir = (RAW_MEN if gender == "men" else RAW_WOMEN) / "bbc"
        for feed_url in feeds:
            n = scrape_rss_feed(feed_url, gender, output_dir, max_articles=max_articles)
            totals[gender] += n
            time.sleep(BBC_DELAY_SECONDS)

    logger.info(f"=== BBC RSS TOTALS: {totals} ===")


if __name__ == "__main__":
    main()
