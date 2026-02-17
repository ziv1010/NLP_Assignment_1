#!/usr/bin/env python3
"""
7_run_ner.py – Run Named Entity Recognition on all scraped football articles
using multiple spaCy models for comparative analysis.

Outputs:
  - {output_dir}/{model_name}/men_entities.pkl
  - {output_dir}/{model_name}/women_entities.pkl
  - {output_dir}/{model_name}/men_entity_freq.pkl
  - {output_dir}/{model_name}/women_entity_freq.pkl

Usage:
    python 7_run_ner.py                         # All models, full corpus
    python 7_run_ner.py --models lg             # Specific model
    python 7_run_ner.py --test                  # 50 articles only
    python 7_run_ner.py --subset 1000 --output-dir data/test_outputs
"""

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

from config import PROCESSED_DIR, logger

# ─── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "sm": "en_core_web_sm",
    "lg": "en_core_web_lg",
    "trf": "en_core_web_trf",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Text normaliser — used for gazetteer matching and frequency counting
# ═══════════════════════════════════════════════════════════════════════════════

def norm(s: str) -> str:
    """Normalise entity text for consistent gazetteer lookup."""
    s = s.lower().strip()
    s = re.sub(r"^[\"'""'']+|[\"'""'']+$", "", s)   # trim quotes
    s = re.sub(r"^the\s+", "", s)                     # drop leading "the"
    s = re.sub(r"\s+", " ", s)                        # collapse spaces
    s = re.sub(r"[^\w\s&/'-]", "", s)                 # drop most punctuation (keep &, /, ', -)
    return s.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Gazetteers (all entries must be pre-normalised)
# ═══════════════════════════════════════════════════════════════════════════════

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
    "reading", "birmingham", "bristol city", "west ham united",
    # Nicknames that spaCy may tag as PERSON
    "blues", "reds", "gunners", "magpies", "hammers", "foxes",
    "toffees", "villans", "saints", "hornets", "cherries",
    "matildas", "lionesses",
}

FOOTBALL_MEDIA = {
    "tnt sports", "sky sports", "bbc sport", "hbo max", "hbo",
    "instagram", "youtube", "espn", "itv", "channel 4", "channel 5",
    "prime video", "nbc sports", "peacock", "bbc", "sky",
    "bt sport", "dazn", "paramount+", "cbs sports", "fox sports",
    "talksport", "the athletic", "optus sport",
}

NON_FOOTBALL_NOISE = {
    "trump", "donald trump", "biden", "joe biden", "nfl", "nba", "mlb",
    "ncaa", "super bowl", "congress", "senate", "white house",
    "republican", "democrat", "putin", "ukraine", "gaza", "israel",
    "hamas", "nasa", "spacex", "tesla", "apple", "google", "amazon",
    "facebook", "meta", "microsoft", "twitter",
}

FOOTBALL_ORGANISATIONS = {
    "fifa", "uefa", "the fa", "fa", "conmebol", "caf", "afc",
    "concacaf", "ofc", "cas", "dfb", "rfef", "figc", "fff", "lfp",
    "efl", "pgmol", "var",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Entity classifier
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ── 1. Noise filter ──────────────────────────────────────────────────
    if t in NON_FOOTBALL_NOISE:
        return "NOISE"

    # ── 2. Exact-match gazetteers (highest priority) ─────────────────────
    if t in FOOTBALL_COMPETITIONS:
        return "COMPETITION"

    if t in FOOTBALL_MEDIA:
        return "MEDIA"

    if t in FOOTBALL_CLUBS:
        return "CLUB"

    if t in FOOTBALL_ORGANISATIONS:
        return "GOVERNING_BODY"

    # ── 3. spaCy-label-based fallbacks ───────────────────────────────────
    if spacy_label == "PERSON":
        return "PLAYER"
    elif spacy_label in ("GPE", "LOC", "FAC"):
        return "LOCATION"
    elif spacy_label == "NORP":
        return "NATIONALITY"
    elif spacy_label == "ORG":
        # NOT forcing ORG→CLUB; unrecognised ORGs stay as ORG_OTHER
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


# ═══════════════════════════════════════════════════════════════════════════════
# NER pipeline
# ═══════════════════════════════════════════════════════════════════════════════

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

    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {gender} ({model_name})"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = article_ids[i : i + batch_size]
        batch_titles = titles[i : i + batch_size]

        # Truncate very long texts to avoid memory issues (keep first 10K chars)
        batch_texts = [t[:10000] if len(t) > 10000 else t for t in batch_texts]

        docs = list(nlp.pipe(batch_texts, disable=["tagger", "parser", "lemmatizer"]))

        for doc, art_id, art_title in zip(docs, batch_ids, batch_titles):
            for ent in doc.ents:
                entity_text_clean = ent.text.strip()
                if not entity_text_clean or len(entity_text_clean) < 2:
                    continue

                football_label = classify_football_entity(entity_text_clean, ent.label_)

                # Skip noise
                if football_label == "NOISE":
                    continue

                # Normalised version for deduplication
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

                # Frequency counter uses normalised text
                key = (entity_norm, football_label)
                entity_freq[key] += 1

    elapsed = time.time() - start_time
    logger.info(
        f"  {gender} ({model_name}): {len(all_entities)} entities extracted "
        f"in {elapsed:.1f}s ({len(texts)/elapsed:.1f} articles/sec)"
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
    parser = argparse.ArgumentParser(description="Run NER on football articles")
    parser.add_argument("--models", nargs="+", default=["sm", "lg"],
                        help="Models to run: sm, lg, trf")
    parser.add_argument("--test", action="store_true", help="Run on 50 articles only")
    parser.add_argument("--subset", type=int, default=0,
                        help="Number of articles per gender (e.g. --subset 1000)")
    parser.add_argument("--output-dir", type=str, default="data/ner_outputs",
                        help="Output directory (e.g. data/test_outputs)")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    output_base = Path(args.output_dir)

    # Load data
    men_df = pd.read_csv(PROCESSED_DIR / "men_articles.csv")
    women_df = pd.read_csv(PROCESSED_DIR / "women_articles.csv")

    if args.test:
        men_df = men_df.head(50)
        women_df = women_df.head(50)
    elif args.subset > 0:
        men_df = men_df.head(args.subset)
        women_df = women_df.head(args.subset)

    logger.info(f"Loaded {len(men_df)} men's + {len(women_df)} women's articles")
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

        men_ents, men_freq = process_articles(
            men_df, nlp, model_name, "men", model_output_dir, args.batch_size
        )
        women_ents, women_freq = process_articles(
            women_df, nlp, model_name, "women", model_output_dir, args.batch_size
        )

        # Combined summary
        logger.info(f"\n--- {model_name} Summary ---")
        logger.info(f"  Men:   {len(men_ents)} entities, {len(men_freq)} unique")
        logger.info(f"  Women: {len(women_ents)} entities, {len(women_freq)} unique")

        # Free memory
        del nlp

    logger.info("\n=== NER EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
