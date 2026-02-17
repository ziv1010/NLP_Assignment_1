#!/usr/bin/env python3
"""
run_all_scrapers.py - Master script to run all scraping pipelines.

Usage:
    python run_all_scrapers.py          # Full scrape from all sources
    python run_all_scrapers.py --test   # Test mode (small samples from each)
"""

import argparse
import subprocess
import sys
from datetime import datetime

from config import logger


def run_script(name: str, args: list[str] = []):
    """Run a Python script and return success status."""
    cmd = [sys.executable, name] + args
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning(f"  {name} exited with code {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  {name} timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"  {name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all scraping pipelines")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--skip", nargs="+", default=[], help="Scripts to skip (e.g. gdelt bbc)")
    args = parser.parse_args()

    start = datetime.now()
    test_flag = ["--test"] if args.test else []
    results = {}

    # 1. GDELT + Trafilatura
    if "gdelt" not in args.skip:
        results["gdelt"] = run_script("1_scrape_gdelt.py", test_flag)
    else:
        logger.info("Skipping GDELT scraper")

    # 2. BBC RSS
    if "bbc" not in args.skip:
        results["bbc"] = run_script("2_scrape_bbc_rss.py", test_flag)
    else:
        logger.info("Skipping BBC RSS scraper")

    # 3. Direct web scraping
    if "web" not in args.skip:
        results["web"] = run_script("3_scrape_web_direct.py", test_flag)
    else:
        logger.info("Skipping web scraper")

    # 4. Kaggle (skip in test mode - requires setup)
    if "kaggle" not in args.skip and not args.test:
        results["kaggle"] = run_script("4_download_kaggle.py")
    else:
        logger.info("Skipping Kaggle downloader")

    # 5. Clean and merge
    if "clean" not in args.skip:
        results["clean"] = run_script("5_clean_and_merge.py")
    else:
        logger.info("Skipping cleaning pipeline")

    elapsed = datetime.now() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {elapsed}")
    logger.info(f"Results: {results}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
