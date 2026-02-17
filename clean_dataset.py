#!/usr/bin/env python3
"""
clean_dataset.py - Build final football-only datasets.

This script filters article records and keeps only true football/soccer content.
It reads from `data/processed` by default, falls back to `data/raw` if needed,
and writes final outputs into `data/final_processed`.

Usage:
    python clean_dataset.py
    python clean_dataset.py --input-mode raw
    python clean_dataset.py --output-dir data/final_processed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, PROCESSED_DIR, RAW_MEN, RAW_WOMEN, logger


# Soccer-specific anchor terms (high precision)
SOCCER_ANCHOR_TERMS = {
    "soccer",
    "association football",
    "uefa",
    "fifa",
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "champions league",
    "europa league",
    "conference league",
    "fa cup",
    "carabao cup",
    "copa del rey",
    "club world cup",
    "mls",
    "nwsl",
    "uwcl",
    "wsl",
    "women's super league",
    "liga f",
    "women's football",
    "offside",
    "goalkeeper",
    "free kick",
    "penalty shootout",
    "yellow card",
    "red card",
    "clean sheet",
    "hat-trick",
    "transfer window",
    "ballon d'or",
}

# Broader football context terms (used only as supporting evidence)
SOCCER_CONTEXT_TERMS = {
    "football",
    "goal",
    "assist",
    "match",
    "fixture",
    "striker",
    "midfielder",
    "defender",
    "winger",
    "manager",
    "coach",
    "club",
    "squad",
    "lineup",
    "relegation",
    "promotion",
    "knockout",
    "group stage",
    "quarter-final",
    "semi-final",
    "injury time",
    "stoppage time",
    "transfer",
    "loan move",
    "signing",
    "penalty",
    "kick-off",
    "full time",
    "half time",
}

# Team/national-side hints
SOCCER_TEAM_TERMS = {
    "arsenal",
    "chelsea",
    "liverpool",
    "man utd",
    "manchester united",
    "man city",
    "manchester city",
    "tottenham",
    "spurs",
    "newcastle",
    "aston villa",
    "west ham",
    "everton",
    "real madrid",
    "barcelona",
    "atletico madrid",
    "bayern",
    "dortmund",
    "juventus",
    "inter milan",
    "ac milan",
    "psg",
    "orlando pride",
    "washington spirit",
    "portland thorns",
    "lionesses",
    "uswnt",
    "chelsea women",
    "arsenal women",
    "barcelona femeni",
    "real madrid femenino",
    "celtic",
    "rangers",
    "galatasaray",
}

# Terms that strongly suggest non-soccer content
NON_SOCCER_TERMS = {
    "touchdown",
    "quarterback",
    "running back",
    "wide receiver",
    "super bowl",
    "nfl",
    "nba",
    "wnba",
    "mlb",
    "nhl",
    "innings",
    "home run",
    "slam dunk",
    "three-pointer",
    "baseball",
    "basketball",
    "ice hockey",
    "cricket",
    "tennis",
    "golf",
    "formula 1",
    "ufc",
    "boxing",
    "ncaa football",
    "college football",
    "rugby",
    "volleyball",
}

SOCCER_URL_HINTS = (
    "/sport/football",
    "/football/",
    "/soccer/",
    "premier-league",
    "champions-league",
    "womens-football",
    "women-football",
    "uefa",
    "fifa",
    "nwsl",
    "mls",
    "copa-del-rey",
    "la-liga",
    "serie-a",
    "bundesliga",
    "ligue-1",
)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _count_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)


def is_football_article(title: str, body: str, url: str, min_chars: int = 80) -> bool:
    """
    Rule-based football classifier tuned for precision.

    Strategy:
    - Require clear football signal in title or URL
    - Confirm football context in body text
    - Reject obvious non-football sports noise
    """
    title_t = _clean_text(title)
    body_t = _clean_text(body)
    url_t = _clean_text(url)

    if len(body_t) < min_chars and len(title_t) < 20:
        return False

    text = f"{title_t} {body_t}"

    title_anchor_hits = _count_hits(title_t, SOCCER_ANCHOR_TERMS)
    title_team_hits = _count_hits(title_t, SOCCER_TEAM_TERMS)
    title_context_hits = _count_hits(title_t, SOCCER_CONTEXT_TERMS)
    title_nonsoccer_hits = _count_hits(title_t, NON_SOCCER_TERMS)

    body_anchor_hits = _count_hits(body_t, SOCCER_ANCHOR_TERMS)
    body_team_hits = _count_hits(body_t, SOCCER_TEAM_TERMS)
    body_context_hits = _count_hits(body_t, SOCCER_CONTEXT_TERMS)
    body_nonsoccer_hits = _count_hits(body_t, NON_SOCCER_TERMS)

    has_soccer_url = any(hint in url_t for hint in SOCCER_URL_HINTS)
    title_or_url_signal = (title_anchor_hits + title_team_hits) >= 1 or has_soccer_url

    # Require explicit football signal in title or URL to reduce body-only noise.
    if not title_or_url_signal:
        return False

    # Early rejects for obvious non-football content.
    if title_nonsoccer_hits >= 1 and title_anchor_hits == 0 and title_team_hits == 0 and not has_soccer_url:
        return False

    if body_nonsoccer_hits >= 2 and title_anchor_hits == 0 and title_team_hits == 0 and not has_soccer_url:
        return False

    # American-football disambiguation.
    if (
        "american football" in text
        or "college football" in text
        or "nfl" in text
        or "touchdown" in text
        or "quarterback" in text
    ) and (title_anchor_hits + title_team_hits + body_anchor_hits + body_team_hits) < 3:
        return False

    # Positive decisions.
    if title_anchor_hits >= 1 and (body_anchor_hits >= 1 or body_team_hits >= 1):
        return True

    if title_team_hits >= 1 and (body_anchor_hits >= 1 or body_team_hits >= 1 or body_context_hits >= 2):
        return True

    if has_soccer_url and (body_anchor_hits >= 1 or body_team_hits >= 1 or body_context_hits >= 3):
        return True

    total_soccer_hits = title_anchor_hits + title_team_hits + body_anchor_hits + body_team_hits
    total_context_hits = title_context_hits + body_context_hits
    if total_soccer_hits >= 4 and total_context_hits >= 3 and body_nonsoccer_hits == 0:
        return True

    return False


def load_articles_from_raw(raw_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for json_file in raw_dir.rglob("*.json"):
        try:
            records.append(json.loads(json_file.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"Failed to load {json_file}: {exc}")
    return pd.DataFrame(records)


def load_articles(gender: str, input_mode: str) -> pd.DataFrame:
    processed_path = PROCESSED_DIR / f"{gender}_articles.csv"
    raw_path = RAW_MEN if gender == "men" else RAW_WOMEN

    if input_mode in {"processed", "auto"} and processed_path.exists():
        logger.info(f"[{gender}] Loading processed data from {processed_path}")
        return pd.read_csv(processed_path)

    if input_mode in {"raw", "auto"}:
        logger.info(f"[{gender}] Loading raw JSON data from {raw_path}")
        return load_articles_from_raw(raw_path)

    raise FileNotFoundError(f"No input data found for {gender}. Checked: {processed_path}, {raw_path}")


def filter_and_save(df: pd.DataFrame, gender: str, output_dir: Path) -> pd.DataFrame:
    logger.info(f"[{gender}] Filtering {len(df)} articles for true football content...")

    keep_indices: list[int] = []
    dropped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Filter {gender}"):
        title = row.get("title", "")
        body = row.get("body_text", "") or row.get("body", "") or row.get("text", "")
        url = row.get("url", "")

        if is_football_article(title=title, body=body, url=url):
            keep_indices.append(idx)
        else:
            dropped += 1

    clean_df = df.loc[keep_indices].copy()

    # Standardize expected columns for downstream steps
    standard_cols = [
        "article_id",
        "url",
        "title",
        "date",
        "domain",
        "source_country",
        "language",
        "gender_category",
        "source_pipeline",
        "body_text",
        "word_count",
        "scraped_at",
    ]
    for col in standard_cols:
        if col not in clean_df.columns:
            clean_df[col] = ""

    clean_df = clean_df[standard_cols]

    # Recompute word_count safely
    clean_df["body_text"] = clean_df["body_text"].fillna("").astype(str)
    clean_df["word_count"] = clean_df["body_text"].str.split().str.len()

    out_csv = output_dir / f"{gender}_articles.csv"
    out_pkl = output_dir / f"{gender}_articles.pkl"

    clean_df.to_csv(out_csv, index=False)
    clean_df.to_pickle(out_pkl)

    logger.info(
        f"[{gender}] Saved {len(clean_df)} football articles to {out_csv} (Dropped {dropped})"
    )
    return clean_df


def write_summary(men_df: pd.DataFrame, women_df: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "men_articles": int(len(men_df)),
        "women_articles": int(len(women_df)),
        "total_articles": int(len(men_df) + len(women_df)),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote summary to {summary_path}: {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter final football-only articles")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "processed", "raw"],
        default="auto",
        help="Where to read input articles from (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "final_processed",
        help="Directory for final football-only outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    men_df = load_articles("men", args.input_mode)
    women_df = load_articles("women", args.input_mode)

    men_clean = filter_and_save(men_df, "men", output_dir)
    women_clean = filter_and_save(women_df, "women", output_dir)

    write_summary(men_clean, women_clean, output_dir)


if __name__ == "__main__":
    main()
