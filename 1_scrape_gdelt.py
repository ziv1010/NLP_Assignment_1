#!/usr/bin/env python3
"""
1_scrape_gdelt.py - Scrape football article URLs via GDELT DOC 2.0 API,
then extract full text via Trafilatura with concurrent downloading.

Focused on Champions League (men's + women's).

Usage:
    python 1_scrape_gdelt.py                    # Full scrape
    python 1_scrape_gdelt.py --test             # Test mode (2 queries, 1 window)
    python 1_scrape_gdelt.py --gender women     # Only women's queries
    python 1_scrape_gdelt.py --gender men       # Only men's queries
    python 1_scrape_gdelt.py --urls-only        # Phase 1: just collect URLs (fast)
    python 1_scrape_gdelt.py --extract-only     # Phase 2: extract text from saved URLs
    python 1_scrape_gdelt.py --workers 8        # Number of concurrent text extractors
"""

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config

from config import (
    GDELT_BASE_URL,
    GDELT_MAX_RECORDS,
    GDELT_MEN_QUERIES,
    GDELT_WOMEN_QUERIES,
    GDELT_DELAY_SECONDS,
    RAW_MEN,
    RAW_WOMEN,
    DATE_FROM,
    DATE_TO,
    logger,
)

# ─── Trafilatura config ──────────────────────────────────────────────────────
TRAF_CONFIG = use_config()
TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")

# Domains that frequently block/404 — skip these to save time
BLOCKED_DOMAINS = {
    "kildare-nationalist.ie", "carlow-nationalist.ie", "roscommonherald.ie",
    "laois-nationalist.ie", "waterford-news.ie", "offalyindependent.ie",
    "limericklive.ie", "galwaydaily.com",
}


def url_to_id(url: str) -> str:
    """Generate a short deterministic ID from a URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def generate_date_windows(start: str, end: str, window_days: int = 60):
    """Yield (start_dt, end_dt) tuples for sliding date windows."""
    fmt = "%Y-%m-%d"
    current = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    while current < end_dt:
        window_end = min(current + timedelta(days=window_days), end_dt)
        yield current, window_end
        current = window_end + timedelta(days=1)


def query_gdelt(query: str, start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Query GDELT DOC 2.0 API and return list of article metadata dicts."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": GDELT_MAX_RECORDS,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }

    try:
        resp = requests.get(GDELT_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except requests.exceptions.JSONDecodeError:
        logger.warning(f"Non-JSON response for: {query[:40]}... ({start_dt.date()}→{end_dt.date()})")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"GDELT request failed: {e}")
        return []


def extract_full_text(url: str) -> tuple[str | None, str | None]:
    """Download and extract article text using Trafilatura. Returns (text, title)."""
    try:
        downloaded = fetch_url(url)
        if downloaded is None:
            return None, None
        text = extract(
            downloaded,
            config=TRAF_CONFIG,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        # Try to extract title
        title = None
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(downloaded, "lxml")
            og = soup.find("meta", property="og:title")
            if og:
                title = og.get("content", "")
            elif soup.title:
                title = soup.title.get_text(strip=True)
        except Exception:
            pass
        return text, title
    except Exception as e:
        logger.debug(f"Trafilatura failed for {url}: {e}")
        return None, None


def is_blocked_domain(url: str) -> bool:
    """Check if URL belongs to a known-blocked domain."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        return domain in BLOCKED_DOMAINS
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Collect URLs from GDELT (fast — no text extraction)
# ═══════════════════════════════════════════════════════════════════════════════


def collect_urls(
    gender: str,
    queries: list[str],
    output_dir: Path,
    max_queries: int | None = None,
    max_windows: int | None = None,
):
    """Collect article URLs from GDELT and save as URL-only JSONs."""
    seen_urls: set[str] = set()

    # Load already-scraped URLs
    for f in output_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            seen_urls.add(data.get("url", ""))
        except Exception:
            pass
    logger.info(f"[{gender.upper()}] {len(seen_urls)} existing articles, will skip duplicates.")

    queries_to_run = queries[:max_queries] if max_queries else queries
    total_new_urls = 0

    for qi, query in enumerate(queries_to_run, 1):
        logger.info(f"[{gender.upper()}] Query {qi}/{len(queries_to_run)}: {query[:60]}...")
        window_count = 0

        for win_start, win_end in generate_date_windows(DATE_FROM, DATE_TO, window_days=60):
            if max_windows and window_count >= max_windows:
                break
            window_count += 1

            time.sleep(GDELT_DELAY_SECONDS)
            articles = query_gdelt(query, win_start, win_end)
            if not articles:
                continue

            new_in_window = 0
            for article in articles:
                url = article.get("url", "")
                if not url or url in seen_urls or is_blocked_domain(url):
                    continue
                seen_urls.add(url)

                # Filter: prefer English
                lang = article.get("language", "")
                if lang and lang.lower() not in ("english", ""):
                    continue

                article_id = url_to_id(url)
                record = {
                    "article_id": article_id,
                    "url": url,
                    "title": article.get("title", ""),
                    "date": article.get("seendate", ""),
                    "domain": article.get("domain", ""),
                    "source_country": article.get("sourcecountry", ""),
                    "language": article.get("language", ""),
                    "gender_category": gender,
                    "source_pipeline": "gdelt",
                    "body_text": "",  # Will be filled in extraction phase
                    "word_count": 0,
                    "scraped_at": datetime.now().isoformat(),
                    "_text_extracted": False,  # Flag for extraction phase
                }

                out_path = output_dir / f"{article_id}.json"
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                new_in_window += 1
                total_new_urls += 1

            logger.info(
                f"  {win_start.date()}→{win_end.date()}: {len(articles)} URLs, {new_in_window} new"
            )

    logger.info(f"[{gender.upper()}] URL collection: {total_new_urls} new URLs saved")
    return total_new_urls


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Extract full text concurrently
# ═══════════════════════════════════════════════════════════════════════════════


def extract_single_article(json_path: Path) -> tuple[Path, bool]:
    """Extract text for a single article JSON file. Returns (path, success)."""
    try:
        data = json.loads(json_path.read_text())

        # Skip if already extracted
        if data.get("_text_extracted") and data.get("body_text"):
            return json_path, True

        url = data.get("url", "")
        if not url or is_blocked_domain(url):
            return json_path, False

        text, title = extract_full_text(url)
        if text and len(text.split()) >= 30:
            data["body_text"] = text
            data["word_count"] = len(text.split())
            data["_text_extracted"] = True
            if title and not data.get("title"):
                data["title"] = title
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            return json_path, True
        else:
            return json_path, False
    except Exception as e:
        logger.debug(f"Error extracting {json_path.name}: {e}")
        return json_path, False


def batch_extract_text(output_dir: Path, gender: str, workers: int = 6):
    """Extract text for all URL-only JSONs in directory using thread pool."""
    # Find files needing extraction
    json_files = list(output_dir.glob("*.json"))
    needs_extraction = []
    already_done = 0

    for f in json_files:
        try:
            data = json.loads(f.read_text())
            if data.get("_text_extracted") and data.get("body_text"):
                already_done += 1
            else:
                needs_extraction.append(f)
        except Exception:
            needs_extraction.append(f)

    logger.info(
        f"[{gender.upper()}] Extraction: {len(needs_extraction)} pending, "
        f"{already_done} already done, {workers} workers"
    )

    if not needs_extraction:
        return already_done

    success_count = already_done
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_single_article, f): f
            for f in needs_extraction
        }
        with tqdm(total=len(needs_extraction), desc=f"  [{gender.upper()}] Extracting text") as pbar:
            for future in as_completed(futures):
                path, success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    # Delete files with no text to keep directory clean
                    try:
                        data = json.loads(path.read_text())
                        if not data.get("body_text"):
                            path.unlink()
                    except Exception:
                        pass
                pbar.update(1)

    logger.info(
        f"[{gender.upper()}] Extraction complete: {success_count} with text, "
        f"{fail_count} failed/removed"
    )
    return success_count


# ═══════════════════════════════════════════════════════════════════════════════
# Combined pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def full_pipeline(
    gender: str,
    queries: list[str],
    output_dir: Path,
    max_queries: int | None = None,
    max_windows: int | None = None,
    workers: int = 6,
    urls_only: bool = False,
    extract_only: bool = False,
):
    """Run full pipeline: collect URLs then extract text."""
    if not extract_only:
        collect_urls(gender, queries, output_dir, max_queries, max_windows)

    if not urls_only:
        return batch_extract_text(output_dir, gender, workers)

    return 0


def main():
    parser = argparse.ArgumentParser(description="GDELT + Trafilatura football article scraper")
    parser.add_argument("--test", action="store_true", help="Test mode: 2 queries, 1 window each")
    parser.add_argument("--gender", choices=["men", "women", "both"], default="both")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None, help="Max date windows per query")
    parser.add_argument("--urls-only", action="store_true", help="Phase 1 only: collect URLs")
    parser.add_argument("--extract-only", action="store_true", help="Phase 2 only: extract text")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent text extractors")
    args = parser.parse_args()

    if args.test:
        args.max_queries = 2
        args.max_windows = 1

    totals = {}
    if args.gender in ("men", "both"):
        totals["men"] = full_pipeline(
            "men", GDELT_MEN_QUERIES, RAW_MEN / "gdelt",
            max_queries=args.max_queries,
            max_windows=args.max_windows,
            workers=args.workers,
            urls_only=args.urls_only,
            extract_only=args.extract_only,
        )

    if args.gender in ("women", "both"):
        totals["women"] = full_pipeline(
            "women", GDELT_WOMEN_QUERIES, RAW_WOMEN / "gdelt",
            max_queries=args.max_queries,
            max_windows=args.max_windows,
            workers=args.workers,
            urls_only=args.urls_only,
            extract_only=args.extract_only,
        )

    logger.info(f"=== FINAL TOTALS: {totals} ===")


if __name__ == "__main__":
    main()
