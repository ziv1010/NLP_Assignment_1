#!/usr/bin/env python3
"""
3_scrape_web_direct.py - Directly scrape football news sites for
Champions League articles (men's + women's).

Targets: UEFA.com, ESPN, SkySports, Goal.com, The Athletic (free), etc.
Uses Trafilatura's built-in sitemapping and crawling capabilities.

Usage:
    python 3_scrape_web_direct.py               # Full crawl
    python 3_scrape_web_direct.py --test         # Test: 10 articles per site
    python 3_scrape_web_direct.py --gender women # Only women's sites
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
from tqdm import tqdm

from config import (
    RAW_MEN,
    RAW_WOMEN,
    WEB_SCRAPE_DELAY_SECONDS,
    TRAFILATURA_DELAY_SECONDS,
    logger,
)

# Trafilatura config
TRAF_CONFIG = use_config()
TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "200")
TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")

# ─── Scraping targets with URL patterns ──────────────────────────────────────
# Each target has: seed URLs to discover article links, and URL patterns to match
SCRAPE_TARGETS = {
    "men": [
        {
            "name": "UEFA UCL News",
            "seed_urls": [
                "https://www.uefa.com/uefachampionsleague/news/",
            ],
            "link_patterns": [r"/uefachampionsleague/news/"],
            "domain": "uefa.com",
            "max_pages": 5,  # number of listing pages to crawl for links
        },
        {
            "name": "SkySports UCL",
            "seed_urls": [
                "https://www.skysports.com/champions-league-news",
            ],
            "link_patterns": [r"/football/news/", r"/champions-league"],
            "domain": "skysports.com",
            "max_pages": 3,
        },
        {
            "name": "Goal.com UCL",
            "seed_urls": [
                "https://www.goal.com/en/champions-league/news",
            ],
            "link_patterns": [r"/en/news/", r"/en/lists/"],
            "domain": "goal.com",
            "max_pages": 3,
        },
    ],
    "women": [
        {
            "name": "UEFA UWCL News",
            "seed_urls": [
                "https://www.uefa.com/womenschampionsleague/news/",
            ],
            "link_patterns": [r"/womenschampionsleague/news/"],
            "domain": "uefa.com",
            "max_pages": 5,
        },
        {
            "name": "SkySports Women's Football",
            "seed_urls": [
                "https://www.skysports.com/womens-football-news",
            ],
            "link_patterns": [r"/football/news/", r"/womens-football"],
            "domain": "skysports.com",
            "max_pages": 3,
        },
        {
            "name": "Goal.com Women's Football",
            "seed_urls": [
                "https://www.goal.com/en/womens-football/news",
            ],
            "link_patterns": [r"/en/news/", r"/en/lists/"],
            "domain": "goal.com",
            "max_pages": 3,
        },
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def load_existing_urls(output_dir: Path) -> set[str]:
    seen = set()
    for f in output_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            seen.add(data.get("url", ""))
        except Exception:
            pass
    return seen


def discover_article_links(seed_url: str, link_patterns: list[str]) -> list[str]:
    """Fetch a listing page and extract URLs matching the given patterns."""
    try:
        resp = requests.get(seed_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  Failed to fetch seed page {seed_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Make absolute
        if href.startswith("/"):
            parsed = urlparse(seed_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"

        # Check if matches any pattern
        for pattern in link_patterns:
            if re.search(pattern, href):
                links.add(href)
                break

    return list(links)


def extract_full_text(url: str) -> tuple[str | None, str | None]:
    """Download and extract article text + title using Trafilatura."""
    try:
        downloaded = fetch_url(url)
        if downloaded is None:
            return None, None

        # Extract text
        text = extract(
            downloaded,
            config=TRAF_CONFIG,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )

        # Try to get title from HTML
        title = None
        soup = BeautifulSoup(downloaded, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Also look for og:title
        og_title = soup.find("meta", property="og:title")
        if og_title:
            title = og_title.get("content", title)

        return text, title
    except Exception as e:
        logger.debug(f"  Extraction failed for {url}: {e}")
        return None, None


def scrape_target(target: dict, gender: str, output_dir: Path, max_articles: int | None = None):
    """Scrape a single target site for articles."""
    name = target["name"]
    logger.info(f"[{gender.upper()}] Scraping: {name}")

    seen_urls = load_existing_urls(output_dir)
    all_links = []

    # Discover article links from seed URLs
    for seed_url in target["seed_urls"]:
        links = discover_article_links(seed_url, target["link_patterns"])
        all_links.extend(links)
        time.sleep(WEB_SCRAPE_DELAY_SECONDS)

    # Deduplicate
    all_links = [url for url in set(all_links) if url not in seen_urls]
    if max_articles:
        all_links = all_links[:max_articles]

    logger.info(f"  Found {len(all_links)} new article URLs")

    saved = 0
    for url in tqdm(all_links, desc=f"  {name}", leave=False):
        body_text, title = extract_full_text(url)
        time.sleep(TRAFILATURA_DELAY_SECONDS)

        if not body_text or len(body_text.split()) < 50:
            continue

        article_id = url_to_id(url)
        record = {
            "article_id": article_id,
            "url": url,
            "title": title or "",
            "date": "",  # Often hard to extract from arbitrary sites
            "domain": target["domain"],
            "source_country": "",
            "language": "English",
            "gender_category": gender,
            "source_pipeline": "web_scraped",
            "body_text": body_text,
            "word_count": len(body_text.split()),
            "scraped_at": datetime.now().isoformat(),
        }

        out_path = output_dir / f"{article_id}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        saved += 1

    logger.info(f"[{gender.upper()}] {name}: saved {saved} articles")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Direct web scraper for CL football articles")
    parser.add_argument("--test", action="store_true", help="Test mode: 10 articles per site")
    parser.add_argument("--gender", choices=["men", "women", "both"], default="both")
    args = parser.parse_args()

    max_articles = 10 if args.test else None
    totals = {"men": 0, "women": 0}

    for gender in ["men", "women"]:
        if args.gender != "both" and args.gender != gender:
            continue
        output_dir = (RAW_MEN if gender == "men" else RAW_WOMEN) / "web_scraped"
        for target in SCRAPE_TARGETS.get(gender, []):
            n = scrape_target(target, gender, output_dir, max_articles=max_articles)
            totals[gender] += n

    logger.info(f"=== WEB SCRAPE TOTALS: {totals} ===")


if __name__ == "__main__":
    main()
