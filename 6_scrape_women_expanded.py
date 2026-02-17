#!/usr/bin/env python3
"""
6_scrape_women_expanded.py - Expanded women's football scraper targeting
sites known to have good women's football coverage.

Sources:
  - BBC Sport Women's Football (sitemap / archive)
  - SkySports Women's Football (paginated news listing)
  - ESPN Women's Football
  - The Athletic (free articles)
  - She Kicks
  - Goal.com Women's Football
  - UEFA UWCL

Also re-queries GDELT with much simpler, unquoted search terms.

Usage:
    python 6_scrape_women_expanded.py              # Full scrape
    python 6_scrape_women_expanded.py --test        # Test: 10 per source
"""

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from trafilatura import fetch_url, extract, sitemaps
from trafilatura.settings import use_config

from config import RAW_WOMEN, GDELT_BASE_URL, DATE_FROM, DATE_TO, logger

# Trafilatura config
TRAF_CONFIG = use_config()
TRAF_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
TRAF_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")

OUTPUT_DIR = RAW_WOMEN / "gdelt"  # reuse same dir

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def load_existing_urls() -> set[str]:
    """Load all existing URLs across all women's source dirs."""
    seen = set()
    for subdir in RAW_WOMEN.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    seen.add(data.get("url", ""))
                except Exception:
                    pass
    return seen


def extract_and_save(url: str, seen_urls: set, source_name: str, output_dir: Path) -> bool:
    """Extract text from URL and save as JSON. Returns True if saved."""
    if url in seen_urls:
        return False
    seen_urls.add(url)

    try:
        downloaded = fetch_url(url)
        if not downloaded:
            return False

        text = extract(
            downloaded,
            config=TRAF_CONFIG,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )

        if not text or len(text.split()) < 50:
            return False

        # Get title
        title = ""
        try:
            soup = BeautifulSoup(downloaded, "lxml")
            og = soup.find("meta", property="og:title")
            if og:
                title = og.get("content", "")
            elif soup.title:
                title = soup.title.get_text(strip=True)
        except Exception:
            pass

        # Get date from meta tags
        pub_date = ""
        try:
            soup = BeautifulSoup(downloaded, "lxml")
            for meta_name in ["article:published_time", "og:article:published_time",
                              "datePublished", "date"]:
                tag = soup.find("meta", property=meta_name) or soup.find("meta", attrs={"name": meta_name})
                if tag and tag.get("content"):
                    pub_date = tag["content"]
                    break
            if not pub_date:
                time_tag = soup.find("time")
                if time_tag and time_tag.get("datetime"):
                    pub_date = time_tag["datetime"]
        except Exception:
            pass

        article_id = url_to_id(url)
        record = {
            "article_id": article_id,
            "url": url,
            "title": title,
            "date": pub_date,
            "domain": urlparse(url).netloc.replace("www.", ""),
            "source_country": "",
            "language": "English",
            "gender_category": "women",
            "source_pipeline": f"women_expanded_{source_name}",
            "body_text": text,
            "word_count": len(text.split()),
            "scraped_at": datetime.now().isoformat(),
        }

        out_path = output_dir / f"{article_id}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return True
    except Exception as e:
        logger.debug(f"Failed {url}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: BBC Women's Football archive via sitemap / section pages
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_bbc_womens(seen_urls: set, output_dir: Path, max_articles: int | None = None) -> int:
    """Scrape BBC Sport women's football section."""
    logger.info("[WOMEN] Scraping BBC Women's Football...")

    # BBC Sport women's section URLs — multiple entry points
    seed_pages = [
        "https://www.bbc.co.uk/sport/football/womens",
        "https://www.bbc.co.uk/sport/football/womens-super-league",
        "https://www.bbc.co.uk/sport/football/womens-champions-league",
        "https://www.bbc.co.uk/sport/football/womens-european-championship",
        "https://www.bbc.co.uk/sport/football/womens-world-cup",
    ]

    all_links = set()
    for seed in seed_pages:
        try:
            resp = requests.get(seed, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/sport/football/" in href and href not in all_links:
                    if href.startswith("/"):
                        href = f"https://www.bbc.co.uk{href}"
                    # Only article-like URLs (contain numbers = article IDs)
                    if re.search(r"/\d{5,}", href) or re.search(r"articles/", href):
                        all_links.add(href)
        except Exception as e:
            logger.warning(f"  BBC seed page failed: {seed} — {e}")
        time.sleep(1)

    links = [u for u in all_links if u not in seen_urls]
    if max_articles:
        links = links[:max_articles]
    logger.info(f"  Found {len(links)} new BBC article URLs")

    saved = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_and_save, url, seen_urls, "bbc", output_dir): url for url in links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  BBC Women's", leave=False):
            if future.result():
                saved += 1

    logger.info(f"  BBC Women's: saved {saved} articles")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: SkySports paginated news
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_skysports_womens(seen_urls: set, output_dir: Path, max_articles: int | None = None) -> int:
    """Scrape SkySports women's football news pages."""
    logger.info("[WOMEN] Scraping SkySports Women's Football...")

    all_links = set()
    # SkySports paginates with ?page=N
    for page in range(1, 21):  # Up to 20 pages
        url = f"https://www.skysports.com/womens-football-news?page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            links_found = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/football/news/" in href or "/womens-football" in href:
                    if href.startswith("/"):
                        href = f"https://www.skysports.com{href}"
                    if re.search(r"/\d{5,}", href) and href not in all_links:
                        all_links.add(href)
                        links_found += 1
            if links_found == 0:
                break  # No more articles
        except Exception as e:
            logger.warning(f"  SkySports page {page} failed: {e}")
            break
        time.sleep(1)

    links = [u for u in all_links if u not in seen_urls]
    if max_articles:
        links = links[:max_articles]
    logger.info(f"  Found {len(links)} new SkySports article URLs")

    saved = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_and_save, url, seen_urls, "skysports", output_dir): url for url in links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  SkySports Women's", leave=False):
            if future.result():
                saved += 1

    logger.info(f"  SkySports Women's: saved {saved} articles")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: GDELT with simpler, unquoted search terms
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_gdelt_simple(seen_urls: set, output_dir: Path, max_articles: int | None = None) -> int:
    """Re-query GDELT with simpler terms and directly extract text."""
    logger.info("[WOMEN] Scraping GDELT with simple queries...")

    # Simpler queries — no quotes, use sourcecountrylanguage filters
    simple_queries = [
        "women football match report",
        "women soccer league result",
        "WSL football Chelsea Arsenal",
        "NWSL soccer match report",
        "women world cup football",
        "women super league football goal",
        "women champions league football",
        "Barcelona women football",
        "Chelsea women football match",
        "Arsenal women football result",
        "Lyon women football champion",
        "Lionesses England women football",
        "women football transfer",
        "women football injury manager",
        "UWCL football match",
        "women football preview",
        "women football analysis tactical",
        "women football press conference",
        "women football highlights recap",
        "Liga F women Spain football",
    ]

    from datetime import timedelta

    all_urls = []
    for qi, query in enumerate(simple_queries, 1):
        logger.info(f"  GDELT simple query {qi}/{len(simple_queries)}: {query[:50]}...")

        # Slide through date windows
        fmt = "%Y-%m-%d"
        current = datetime.strptime(DATE_FROM, fmt)
        end_dt = datetime.strptime(DATE_TO, fmt)

        while current < end_dt:
            win_end = min(current + timedelta(days=90), end_dt)
            params = {
                "query": query + " sourcelang:english",
                "mode": "ArtList",
                "format": "json",
                "maxrecords": 250,
                "startdatetime": current.strftime("%Y%m%d%H%M%S"),
                "enddatetime": win_end.strftime("%Y%m%d%H%M%S"),
                "sort": "DateDesc",
            }
            try:
                resp = requests.get(GDELT_BASE_URL, params=params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    articles = data.get("articles", [])
                    for a in articles:
                        url = a.get("url", "")
                        if url and url not in seen_urls:
                            all_urls.append({
                                "url": url,
                                "title": a.get("title", ""),
                                "date": a.get("seendate", ""),
                                "domain": a.get("domain", ""),
                            })
                            seen_urls.add(url)
            except Exception:
                pass
            current = win_end + timedelta(days=1)
            time.sleep(2)

    logger.info(f"  GDELT simple: found {len(all_urls)} new URLs")
    if max_articles:
        all_urls = all_urls[:max_articles]

    # Concurrent extraction
    saved = 0

    def _extract(item):
        url = item["url"]
        try:
            downloaded = fetch_url(url)
            if not downloaded:
                return False
            text = extract(downloaded, config=TRAF_CONFIG, include_comments=False,
                          include_tables=False, favor_recall=True)
            if not text or len(text.split()) < 50:
                return False

            article_id = url_to_id(url)
            record = {
                "article_id": article_id,
                "url": url,
                "title": item.get("title", ""),
                "date": item.get("date", ""),
                "domain": item.get("domain", ""),
                "source_country": "",
                "language": "English",
                "gender_category": "women",
                "source_pipeline": "women_expanded_gdelt_simple",
                "body_text": text,
                "word_count": len(text.split()),
                "scraped_at": datetime.now().isoformat(),
            }
            out_path = output_dir / f"{article_id}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_extract, item) for item in all_urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="  GDELT simple extract", leave=False):
            if future.result():
                saved += 1

    logger.info(f"  GDELT simple: saved {saved} articles")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Source 4: Trafilatura sitemap crawling for women's football sites
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_via_sitemaps(seen_urls: set, output_dir: Path, max_articles: int | None = None) -> int:
    """Use Trafilatura's sitemap discovery to find women's football articles."""
    logger.info("[WOMEN] Crawling sitemaps for women's football sites...")

    sites = [
        ("https://www.espn.com/soccer/", ["women", "nwsl", "wsl", "uwcl", "womens"]),
        ("https://shekicks.net/", None),  # Entire site is women's football
        ("https://theathletic.com/football/womens-football/", None),
    ]

    all_urls = []
    for site_url, keywords in sites:
        try:
            logger.info(f"  Discovering sitemap for {site_url}...")
            discovered = sitemaps.sitemap_search(site_url)
            if discovered:
                for url in discovered:
                    if url not in seen_urls:
                        if keywords is None:
                            all_urls.append(url)
                        else:
                            url_lower = url.lower()
                            if any(kw in url_lower for kw in keywords):
                                all_urls.append(url)
                logger.info(f"    Found {len(all_urls)} candidate URLs")
        except Exception as e:
            logger.warning(f"  Sitemap failed for {site_url}: {e}")

    if max_articles:
        all_urls = all_urls[:max_articles]
    logger.info(f"  Total sitemap URLs to extract: {len(all_urls)}")

    saved = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_and_save, url, seen_urls, "sitemap", output_dir): url for url in all_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Sitemap extract", leave=False):
            if future.result():
                saved += 1

    logger.info(f"  Sitemap crawl: saved {saved} articles")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Expanded women's football scraper")
    parser.add_argument("--test", action="store_true", help="Test mode: 10 per source")
    args = parser.parse_args()

    max_articles = 10 if args.test else None
    output_dir = RAW_WOMEN / "gdelt"  # Put all in same dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_urls = load_existing_urls()
    logger.info(f"[WOMEN] Starting expanded scrape. {len(seen_urls)} existing URLs to skip.")

    totals = {}
    totals["bbc"] = scrape_bbc_womens(seen_urls, output_dir, max_articles)
    totals["skysports"] = scrape_skysports_womens(seen_urls, output_dir, max_articles)
    totals["gdelt_simple"] = scrape_gdelt_simple(seen_urls, output_dir, max_articles)
    totals["sitemaps"] = scrape_via_sitemaps(seen_urls, output_dir, max_articles)

    total = sum(totals.values())
    logger.info(f"\n=== EXPANDED WOMEN'S SCRAPE COMPLETE ===")
    logger.info(f"Source breakdown: {totals}")
    logger.info(f"Total new women's articles: {total}")
    logger.info(f"Total women's files: {len(list(output_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
