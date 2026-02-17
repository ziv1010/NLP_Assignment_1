#!/usr/bin/env python3
"""
7_run_ner.py – Run Named Entity Recognition on all scraped football articles
using multiple spaCy models for comparative analysis.

Enhanced with SocCor Players.csv gazetteer for improved entity classification.

Outputs:
  - {output_dir}/{model_name}/other_entities.pkl
  - {output_dir}/{model_name}/women_entities.pkl
  - {output_dir}/{model_name}/other_entity_freq.pkl
  - {output_dir}/{model_name}/women_entity_freq.pkl

Usage:
    python final_codes/7_run_ner.py                         # All models, full corpus
    python final_codes/7_run_ner.py --models lg             # Specific model
    python final_codes/7_run_ner.py --test                  # 50 articles only
    python final_codes/7_run_ner.py --subset 1000 --output-dir data/test_outputs
    python final_codes/7_run_ner.py --input-dir data/kaggle_processed/test --models lg
"""

import argparse
import csv
import logging
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

# ─── Logging (standalone, no config.py dependency) ──────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("football_ner")

# ─── Defaults ────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path("data/kaggle_processed/test")
SOCCOR_METADATA = Path("data/soccor/SocCor/Metadata")

# ─── Model registry ──────────────────────────────────────────────────
MODELS = {
    "sm": "en_core_web_sm",
    "lg": "en_core_web_lg",
    "trf": "en_core_web_trf",
}

# ═══════════════════════════════════════════════════════════════════════
# Text normaliser — used for gazetteer matching and frequency counting
# ═══════════════════════════════════════════════════════════════════════

def norm(s: str) -> str:
    """Normalise entity text for consistent gazetteer lookup."""
    s = s.lower().strip()
    s = re.sub(r"^[\"'\u201c\u201d\u2018\u2019]+|[\"'\u201c\u201d\u2018\u2019]+$", "", s)
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s&/'-]", "", s)
    return s.strip()


# ═══════════════════════════════════════════════════════════════════════
# Load SocCor Players.csv as a FOOTBALL_PLAYERS gazetteer
# ═══════════════════════════════════════════════════════════════════════

def load_soccor_players(metadata_dir: Path) -> set:
    """
    Load Players.csv from SocCor metadata and build a normalised set of player
    names (full name + surname) for gazetteer matching.
    Separate from FOOTBALL_CLUBS — player names and club names are never mixed.
    """
    players_csv = metadata_dir / "Players.csv"
    player_names = set()

    if not players_csv.exists():
        logger.warning(f"Players.csv not found at {players_csv}, continuing without player gazetteer")
        return player_names

    with open(players_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue
            # Add full name (normalised)
            player_names.add(norm(name))
            # Add surname only (last word)
            parts = name.split()
            if len(parts) > 1:
                player_names.add(norm(parts[-1]))

    logger.info(f"Loaded {len(player_names)} player name entries from SocCor Players.csv")
    return player_names


def load_soccor_clubs(metadata_dir: Path) -> set:
    """
    Load club names from Players.csv 'club' column to augment the FOOTBALL_CLUBS gazetteer.
    """
    players_csv = metadata_dir / "Players.csv"
    clubs = set()

    if not players_csv.exists():
        return clubs

    with open(players_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            club = row.get("club  ", "").strip()  # Note: column has trailing spaces
            if not club:
                club = row.get("club", "").strip()
            if club:
                clubs.add(norm(club))

    return clubs


# ═══════════════════════════════════════════════════════════════════════
# Static Gazetteers (all entries must be pre-normalised)
# ═══════════════════════════════════════════════════════════════════════

FOOTBALL_COMPETITIONS = {
    "champions league", "premier league", "la liga", "serie a", "bundesliga",
    "ligue 1", "wsl", "women's super league", "liga f", "nwsl",
    "uwcl", "women's champions league", "uefa champions league",
    "fa cup", "europa league", "conference league", "world cup",
    "women's world cup", "euros", "european championship", "copa del rey",
    "carabao cup", "efl cup", "community shield", "super cup",
    "club world cup", "ballon d'or", "fifa best", "euro 2024",
    "euro 2020", "nations league", "copa america", "league cup",
    "fa women's super league", "barclays wsl", "championship",
    "she believes cup", "arnold clark cup", "finalissima",
}

FOOTBALL_CLUBS = {
    # Premier League / WSL
    "chelsea", "arsenal", "liverpool", "tottenham", "spurs",
    "manchester united", "man utd", "manchester city", "man city",
    "newcastle", "west ham", "aston villa", "everton", "brighton",
    "wolves", "wolverhampton", "brentford", "fulham", "bournemouth",
    "crystal palace", "burnley", "luton", "sheffield united",
    "nottingham forest", "leicester",
    # La Liga / Liga F
    "barcelona", "barca", "real madrid", "atletico madrid", "atletico",
    "sevilla", "real betis", "villarreal", "real sociedad", "athletic bilbao",
    "valencia", "celta vigo", "getafe", "girona", "osasuna", "mallorca",
    # Bundesliga
    "bayern munich", "bayern", "borussia dortmund", "dortmund", "bayer leverkusen",
    "leverkusen", "rb leipzig", "leipzig", "wolfsburg", "frankfurt",
    "eintracht frankfurt", "stuttgart", "freiburg", "gladbach",
    "borussia monchengladbach", "hoffenheim", "mainz", "augsburg",
    # Serie A
    "juventus", "inter milan", "inter", "ac milan", "milan", "napoli", "roma",
    "lazio", "fiorentina", "atalanta", "torino", "sampdoria", "bologna",
    "udinese", "sassuolo", "genoa", "cagliari", "verona", "empoli",
    # Ligue 1
    "psg", "paris saint-germain", "marseille", "lyon", "monaco", "lille",
    "nice", "rennes", "lens", "montpellier", "toulouse", "strasbourg",
    # Other notable clubs
    "ajax", "psv", "feyenoord", "benfica", "porto", "sporting",
    "celtic", "rangers", "galatasaray", "fenerbahce", "besiktas",
    # Women's specific
    "gotham fc", "racing louisville", "portland thorns", "ol reign",
    "angel city", "san diego wave", "washington spirit", "north carolina courage",
    "orlando pride", "houston dash", "kansas city current", "chicago red stars",
    # Nicknames that spaCy may tag as PERSON
    "blues", "reds", "gunners", "magpies", "hammers", "foxes",
    "toffees", "villans", "saints", "hornets", "cherries",
    "matildas", "lionesses", "uswnt", "canwnt", "super falcons",
    "banyana banyana", "football ferns",
}

FOOTBALL_MEDIA = {
    "tnt sports", "sky sports", "bbc sport", "hbo max", "hbo",
    "instagram", "youtube", "espn", "itv", "channel 4", "channel 5",
    "prime video", "nbc sports", "peacock", "bbc", "sky",
    "bt sport", "dazn", "paramount+", "cbs sports", "fox sports",
    "talksport", "the athletic", "optus sport", "telegraph",
    "guardian", "daily mail", "mirror", "sun", "times", "independent",
    "nytimes", "washington post", "usa today", "ap", "reuters",
    "cnn", "goal", "onefootball", "fotmob", "sofascore",
}

NON_FOOTBALL_NOISE = {
    "trump", "donald trump", "biden", "joe biden", "nfl", "nba", "mlb",
    "ncaa", "super bowl", "congress", "senate", "white house",
    "republican", "democrat", "putin", "ukraine", "gaza", "israel",
    "hamas", "nasa", "spacex", "tesla", "apple", "google", "amazon",
    "facebook", "meta", "microsoft", "twitter", "netflix", "disney",
    "marvel", "star wars", "taylor swift", "beyonce", "grammy", "oscar",
    "bafta", "emmy", "goldman sachs", "jpmorgan", "stock market",
    "wall street", "fed", "inflation", "recession", "gdp",
}

FOOTBALL_ORGANISATIONS = {
    "fifa", "uefa", "the fa", "fa", "conmebol", "caf", "afc",
    "concacaf", "ofc", "cas", "dfb", "rfef", "figc", "fff", "lfp",
    "efl", "pgmol", "var", "nwsl", "wsl", "premier league", "mls",
    "pfa", "fpro", "eca", "ifab",
}


# ═══════════════════════════════════════════════════════════════════════
# Article Filter
# ═══════════════════════════════════════════════════════════════════════

KEYWORDS_FOOTBALL = {
    "football", "soccer", "league", "cup", "match", "game", "player",
    "team", "club", "manager", "coach", "goal", "assist", "score",
    "win", "loss", "draw", "stadium", "pitch", "referee", "var",
    "offside", "penalty", "corner", "free kick", "throw in",
    "goalkeeper", "defender", "midfielder", "striker", "forward",
    "winger", "captain", "squad", "roster", "lineup", "transfer",
    "loan", "signing", "contract", "injury", "suspension", "table",
    "standings", "points", "relegation", "promotion", "playoff",
    "final", "semi-final", "quarter-final", "champion", "title",
    "trophy", "medal", "tournament", "qualifier", "group stage",
    "knockout", "round of 16", "fixture", "result", "highlight",
    "replay", "extra time", "shootout", "aggregate", "away goal",
    "clean sheet", "hat-trick", "brace", "own goal", "red card",
    "yellow card", "booking", "foul", "handball", "diving",
    "simulation", "time wasting", "added time", "injury time",
    "stoppage time", "half time", "full time", "kick off",
    "first half", "second half", "season", "pre-season",
    "friendly", "international break", "world cup", "euro",
    "copa america", "asian cup", "afcon", "gold cup",
    "champions league", "europa league", "conference league",
    "libertadores", "sudamericana", "recopa", "super cup",
    "club world cup", "olympics", "ballon d'or", "uefa", "fifa",
}

def is_football_article(text: str, threshold: int = 3) -> bool:
    """Check if an article is likely about football."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    if any(w in text_lower for w in ["touchdown", "quarterback", "inning", "homerun", "slam dunk"]):
        return False
    count = 0
    for kw in KEYWORDS_FOOTBALL:
        if kw in text_lower:
            count += 1
            if count >= threshold:
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Entity classifier
# ═══════════════════════════════════════════════════════════════════════

# This will be populated at runtime from Players.csv
FOOTBALL_PLAYERS: set = set()


def classify_football_entity(text: str, spacy_label: str) -> str:
    """
    Map a spaCy NER label to a football-domain category.
    Uses exact-match gazetteers (after normalisation) to correct
    common spaCy misclassifications.

    Returns one of:
        PLAYER, CLUB, COMPETITION, LOCATION, NATIONALITY,
        MEDIA, GOVERNING_BODY, ORG_OTHER, NUMERIC, NOISE, MISC
    """
    t = norm(text)

    # ── 1. Noise filter ──────────────────────────────────────────────
    if t in NON_FOOTBALL_NOISE:
        return "NOISE"

    # ── 2. Exact-match gazetteers (highest priority) ─────────────────
    if t in FOOTBALL_COMPETITIONS:
        return "COMPETITION"

    if t in FOOTBALL_MEDIA:
        return "MEDIA"

    if t in FOOTBALL_CLUBS:
        return "CLUB"

    if t in FOOTBALL_ORGANISATIONS:
        return "GOVERNING_BODY"

    # ── 3. SocCor player gazetteer (new!) ────────────────────────────
    # If the entity matches a known player name from Players.csv,
    # classify as PLAYER regardless of spaCy label.
    if t in FOOTBALL_PLAYERS:
        return "PLAYER"

    # ── 4. spaCy-label-based fallbacks ───────────────────────────────
    if spacy_label == "PERSON":
        return "PLAYER"
    elif spacy_label in ("GPE", "LOC", "FAC"):
        return "LOCATION"
    elif spacy_label == "NORP":
        return "NATIONALITY"
    elif spacy_label == "ORG":
        return "ORG_OTHER"
    elif spacy_label == "EVENT":
        return "EVENT_OTHER"
    elif spacy_label == "DATE":
        return "DATE"
    elif spacy_label == "TIME":
        return "TIME"
    elif spacy_label in ("CARDINAL", "ORDINAL"):
        return "NUMERIC"
    elif spacy_label == "MONEY":
        return "MONEY"
    else:
        return "MISC"


# ═══════════════════════════════════════════════════════════════════════
# NER pipeline
# ═══════════════════════════════════════════════════════════════════════

def process_articles(
    df: pd.DataFrame,
    nlp,
    model_name: str,
    gender: str,
    output_dir: Path,
    batch_size: int = 100,
):
    """
    Run NER on all articles and save entity extractions.

    Schema:
      article_id, entity_text, entity_text_norm, entity_label, spacy_label,
      start_char, end_char, sentence_context, article_title
    """
    all_entities = []
    entity_freq = Counter()

    texts = df["body_text"].fillna("").tolist()
    article_ids = df["article_id"].tolist()
    titles = df["title"].fillna("").tolist()

    logger.info(f"[{gender.upper()}] Running NER with {model_name} on {len(texts)} articles...")
    start_time = time.time()
    skipped_articles = 0
    processed_count = 0

    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {gender} ({model_name})"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = article_ids[i : i + batch_size]
        batch_titles = titles[i : i + batch_size]

        # Filter articles - keep only football-relevant ones
        valid_indices = []
        for idx, text in enumerate(batch_texts):
            if is_football_article(text):
                valid_indices.append(idx)
            else:
                skipped_articles += 1

        if not valid_indices:
            continue

        batch_texts = [batch_texts[j] for j in valid_indices]
        batch_ids = [batch_ids[j] for j in valid_indices]
        batch_titles = [batch_titles[j] for j in valid_indices]
        processed_count += len(valid_indices)

        # Truncate very long texts to avoid memory issues (keep first 10K chars)
        batch_texts = [t[:10000] if len(t) > 10000 else t for t in batch_texts]

        docs = list(nlp.pipe(batch_texts, disable=["tagger", "parser", "lemmatizer"]))

        for doc, art_id, art_title in zip(docs, batch_ids, batch_titles):
            for ent in doc.ents:
                entity_text_clean = ent.text.strip()
                if not entity_text_clean or len(entity_text_clean) < 2:
                    continue

                football_label = classify_football_entity(entity_text_clean, ent.label_)

                if football_label == "NOISE":
                    continue

                entity_norm = norm(entity_text_clean)

                # Get sentence context (±50 chars around entity)
                ctx_start = max(0, ent.start_char - 50)
                ctx_end = min(len(doc.text), ent.end_char + 50)
                context = doc.text[ctx_start:ctx_end].replace("\n", " ")

                all_entities.append({
                    "article_id": art_id,
                    "entity_text": entity_text_clean,
                    "entity_text_norm": entity_norm,
                    "entity_label": football_label,
                    "spacy_label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "sentence_context": context,
                    "article_title": art_title,
                    "gender_category": gender,
                    "model": model_name,
                })

                key = (entity_norm, football_label)
                entity_freq[key] += 1

    elapsed = time.time() - start_time
    logger.info(
        f"  {gender} ({model_name}): {len(all_entities)} entities extracted from {processed_count} articles "
        f"({skipped_articles} skipped as non-football) "
        f"in {elapsed:.1f}s ({processed_count/max(elapsed,0.01):.1f} articles/sec)"
    )

    # Convert to DataFrames
    entities_df = pd.DataFrame(all_entities)
    freq_df = pd.DataFrame(
        [(text, label, count) for (text, label), count in entity_freq.most_common()],
        columns=["entity_text", "entity_label", "frequency"],
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    entities_df.to_pickle(output_dir / f"{gender}_entities.pkl")
    freq_df.to_pickle(output_dir / f"{gender}_entity_freq.pkl")
    entities_df.to_csv(output_dir / f"{gender}_entities.csv", index=False)
    freq_df.to_csv(output_dir / f"{gender}_entity_freq.csv", index=False)

    # Summary
    logger.info(f"  Saved to {output_dir}/")
    logger.info(f"  Entity label distribution:")
    if len(entities_df) > 0:
        label_counts = entities_df["entity_label"].value_counts()
        for label, cnt in label_counts.items():
            logger.info(f"    {label}: {cnt}")
    logger.info(f"  Top 15 entities:")
    for _, row in freq_df.head(15).iterrows():
        logger.info(f"    {row['entity_text']} ({row['entity_label']}): {row['frequency']}")

    return entities_df, freq_df


def main():
    global FOOTBALL_PLAYERS, FOOTBALL_CLUBS

    parser = argparse.ArgumentParser(description="Run NER on football articles")
    parser.add_argument("--models", nargs="+", default=["sm", "lg"],
                        help="Models to run: sm, lg, trf")
    parser.add_argument("--test", action="store_true", help="Run on 50 articles only")
    parser.add_argument("--subset", type=int, default=0,
                        help="Number of articles per gender (e.g. --subset 1000)")
    parser.add_argument("--output-dir", type=str, default="data/ner_outputs",
                        help="Output directory (e.g. data/test_outputs)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory containing women_articles.csv / other_articles.csv")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--soccor-metadata", type=str, default=str(SOCCOR_METADATA),
                        help="Path to SocCor Metadata dir for player gazetteer")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    input_base = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR

    # ── Load SocCor gazetteers ───────────────────────────────────────
    metadata_dir = Path(args.soccor_metadata)
    FOOTBALL_PLAYERS = load_soccor_players(metadata_dir)
    soccor_clubs = load_soccor_clubs(metadata_dir)
    FOOTBALL_CLUBS.update(soccor_clubs)
    logger.info(f"Added {len(soccor_clubs)} clubs from SocCor to FOOTBALL_CLUBS gazetteer")

    # ── Load data ────────────────────────────────────────────────────
    logger.info(f"Loading data from: {input_base}")

    # Support both naming conventions:
    #   old: men_articles.csv / women_articles.csv
    #   new: other_articles.csv / women_articles.csv
    other_path = input_base / "other_articles.csv"
    if not other_path.exists():
        other_path = input_base / "men_articles.csv"
    women_path = input_base / "women_articles.csv"

    if not other_path.exists():
        logger.error(f"Cannot find other_articles.csv or men_articles.csv in {input_base}")
        return
    if not women_path.exists():
        logger.error(f"Cannot find women_articles.csv in {input_base}")
        return

    other_df = pd.read_csv(other_path)
    women_df = pd.read_csv(women_path)

    if args.test:
        other_df = other_df.head(50)
        women_df = women_df.head(50)
    elif args.subset > 0:
        other_df = other_df.head(args.subset)
        women_df = women_df.head(args.subset)

    logger.info(f"Loaded {len(other_df)} other + {len(women_df)} women's articles")
    logger.info(f"Output dir: {output_base}")

    for model_key in args.models:
        if model_key not in MODELS:
            logger.warning(f"Unknown model: {model_key}, skipping")
            continue

        model_name = MODELS[model_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"{'='*60}")

        try:
            nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"Model {model_name} not installed. Run: python -m spacy download {model_name}")
            continue

        model_output_dir = output_base / model_key

        other_ents, other_freq = process_articles(
            other_df, nlp, model_name, "other", model_output_dir, args.batch_size
        )
        women_ents, women_freq = process_articles(
            women_df, nlp, model_name, "women", model_output_dir, args.batch_size
        )

        # Combined summary
        logger.info(f"\n--- {model_name} Summary ---")
        logger.info(f"  Other:  {len(other_ents)} entities, {len(other_freq)} unique")
        logger.info(f"  Women: {len(women_ents)} entities, {len(women_freq)} unique")

        # Free memory
        del nlp

    logger.info("\n=== NER EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
