"""
config.py - Shared configuration for football article scraping pipeline.
Focuses on Champions League (men's + women's) content.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_MEN = DATA_DIR / "raw" / "men"
RAW_WOMEN = DATA_DIR / "raw" / "women"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

# Create all data directories
for d in [
    RAW_MEN / "gdelt",
    RAW_MEN / "bbc",
    RAW_MEN / "web_scraped",
    RAW_MEN / "kaggle",
    RAW_WOMEN / "gdelt",
    RAW_WOMEN / "bbc",
    RAW_WOMEN / "web_scraped",
    RAW_WOMEN / "kaggle",
    PROCESSED_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Date Range ───────────────────────────────────────────────────────────────
DATE_FROM = "2023-01-01"
DATE_TO = "2026-02-13"

# ─── GDELT DOC API ────────────────────────────────────────────────────────────
# Max 250 results per query; we use sliding date windows
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250

# Search queries for men's Champions League
GDELT_MEN_QUERIES = [
    '"Champions League" match report football',
    '"Champions League" football player goal',
    '"Champions League" football manager tactics',
    '"Champions League" football transfer',
    '"Champions League" football injury squad',
    '"UEFA Champions League" football analysis',
    '"Champions League" football semi final',
    '"Champions League" football group stage',
    '"Champions League" final football',
    '"Champions League" football knockout round',
    '"Champions League" football draw fixtures',
    '"Champions League" football review recap',
    '"Champions League" football press conference',
    '"Champions League" football preview prediction',
    '"Champions League" football highlights result',
]

# Search queries for women's football (broad coverage)
GDELT_WOMEN_QUERIES = [
    # ── UWCL / Women's Champions League ──
    "\"Women's Champions League\" football",
    "\"UEFA Women's Champions League\" football",
    "\"UWCL\" football",
    "\"UWCL\" match report",
    "\"Women's Champions League\" final",
    "\"Women's Champions League\" group stage knockout",
    # ── WSL (Barclays Women's Super League) ──
    "\"Women's Super League\" football",
    "\"WSL\" football match report",
    "\"Barclays Women's Super League\" football",
    "\"WSL\" goal player result",
    "\"WSL\" manager tactics football",
    "\"WSL\" transfer signing football",
    "\"WSL\" injury squad football",
    # ── Liga F (Spain) ──
    "\"Liga F\" football",
    "\"Liga F\" match report women",
    "\"Liga F\" women football result",
    # ── NWSL (USA) ──
    "\"NWSL\" football match",
    "\"NWSL\" soccer report",
    "\"National Women's Soccer League\" match",
    # ── Women's World Cup ──
    "\"Women's World Cup\" football",
    "\"FIFA Women's World Cup\" match report",
    "\"Women's World Cup\" goal player",
    # ── General women's football ──
    "women's football match report",
    "women's football player goal analysis",
    "women's football transfer signing",
    "women's football manager tactics",
    "women's football injury squad",
    "women's football preview prediction",
    "women's football highlights result recap",
    "women's football press conference",
    "women's soccer match report",
    # ── Top women's club teams ──
    "\"Barcelona Femeni\" football",
    "\"Chelsea Women\" football",
    "\"Lyon Feminines\" football",
    "\"Arsenal Women\" football",
    "\"Manchester City Women\" football",
    "\"Bayern Munich Women\" football",
    "\"Real Madrid Femenino\" football",
    # ── National teams ──
    "\"Lionesses\" football England women",
    "\"USWNT\" soccer football",
    "\"La Roja\" women football Spain",
]

# ─── BBC RSS Feeds ────────────────────────────────────────────────────────────
BBC_RSS_FEEDS = {
    "men": [
        "https://feeds.bbci.co.uk/sport/football/champions-league/rss.xml",
        "https://feeds.bbci.co.uk/sport/football/european-championship/rss.xml",
    ],
    "women": [
        # BBC doesn't have a specific UWCL RSS, but women's football feed covers it
        "https://feeds.bbci.co.uk/sport/football/womens/rss.xml",
    ],
}

# ─── Web Scraping Targets ────────────────────────────────────────────────────
# Sites to scrape directly for Champions League articles
WEB_SCRAPE_TARGETS = {
    "men": {
        "uefa_ucl": "https://www.uefa.com/uefachampionsleague/news/",
        "espn_ucl": "https://www.espn.com/soccer/league/_/name/uefa.champions/uefa-champions-league",
    },
    "women": {
        "uefa_uwcl": "https://www.uefa.com/womenschampionsleague/news/",
        "espn_uwcl": "https://www.espn.com/soccer/league/_/name/uefa.wchampions/uefa-women-champions-league",
    },
}

# ─── Kaggle Datasets ─────────────────────────────────────────────────────────
KAGGLE_DATASETS = [
    "beridzeg45/football-news-articles",           # ~12K articles from Goal, SkySports, Tribuna
    "ibrahimkiziloklu/football-transfer-news-articles-for-nlp",  # ~6K transfer articles
]

# ─── Rate Limiting ────────────────────────────────────────────────────────────
GDELT_DELAY_SECONDS = 2        # Delay between GDELT queries
TRAFILATURA_DELAY_SECONDS = 1  # Delay between full-text fetches
BBC_DELAY_SECONDS = 1
WEB_SCRAPE_DELAY_SECONDS = 2

# ─── Logging ──────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scrape.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("football_scraper")
