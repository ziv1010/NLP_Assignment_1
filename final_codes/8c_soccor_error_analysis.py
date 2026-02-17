#!/usr/bin/env python3
"""
8c_soccor_error_analysis.py – Detailed error analysis for SocCor NER evaluation.

Re-runs the SocCor evaluation and records WHICH specific players each model
missed (FN) and which spurious detections (FP) each model made.

Outputs:
  - soccor_error_detail.csv       – every FN/FP with player name, doc, model
  - soccor_error_summary.txt      – human-readable error report
  - soccor_errors_all_models.csv  – players missed by ALL models (shared blind spots)

Usage:
    python final_codes/8c_soccor_error_analysis.py
    python final_codes/8c_soccor_error_analysis.py --models sm lg trf
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy

# ─── Paths ────────────────────────────────────────────────────────────────────
SOCCOR_DIR = Path("data/soccor/SocCor/Textdata/cleaned_data")
METADATA_DIR = Path("data/soccor/SocCor/Metadata")
NER_OUTPUT = Path("data/ner_outputs")

MODELS = {
    "sm": "en_core_web_sm",
    "lg": "en_core_web_lg",
    "trf": "en_core_web_trf",
}

TAG_PATTERN = re.compile(r"<([A-Z\-']+)_([A-Z\-]+)_([A-Z]{3})>")


# ─── Data loading (reused from 8b) ───────────────────────────────────────────

def load_player_metadata() -> dict:
    players_csv = METADATA_DIR / "Players.csv"
    token_to_name = {}
    if not players_csv.exists():
        print(f"  [WARNING] Players.csv not found at {players_csv}")
        return token_to_name
    with open(players_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = row.get("name", "").strip()
            token = row.get("token", "").strip()
            if name and token:
                token_to_name[token] = name
    print(f"  Loaded {len(token_to_name)} player name mappings from Players.csv")
    return token_to_name


def tag_to_fallback_name(tag_match: re.Match) -> str:
    raw = tag_match.group(1).replace("-", " ").title()
    raw = re.sub(r"(\w)'(\w)", lambda m: m.group(1) + "'" + m.group(2), raw)
    return raw


def load_soccor_english(token_to_name: dict):
    documents = []
    for subdir in ["Reports", "Highlights", "Games", "Livetickers"]:
        bbc_dir = SOCCOR_DIR / subdir / "BBC"
        if not bbc_dir.exists():
            continue
        for json_file in sorted(bbc_dir.glob("*.json")):
            try:
                raw = json_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, list):
                    annotated_text = " ".join(
                        item.get("text", "") for item in data if isinstance(item, dict)
                    )
                elif isinstance(data, dict):
                    annotated_text = data.get("text", "")
                else:
                    continue

                if not annotated_text or len(annotated_text) < 20:
                    continue

                gold_entities = []
                for match in TAG_PATTERN.finditer(annotated_text):
                    tag_text = match.group(0)
                    position = match.group(2).replace("-", " ").title()
                    country = match.group(3)

                    if tag_text in token_to_name:
                        player_name = token_to_name[tag_text]
                    else:
                        player_name = tag_to_fallback_name(match)

                    gold_entities.append({
                        "player_name": player_name,
                        "tag_text": tag_text,
                        "position": position,
                        "country": country,
                        "start": match.start(),
                        "end": match.end(),
                    })

                def replace_tag(m):
                    t = m.group(0)
                    if t in token_to_name:
                        return token_to_name[t]
                    return tag_to_fallback_name(m)

                clean_text = TAG_PATTERN.sub(replace_tag, annotated_text)

                recalculated_gold = []
                for gold in gold_entities:
                    name = gold["player_name"]
                    idx = clean_text.find(name)
                    if idx >= 0:
                        recalculated_gold.append({
                            **gold,
                            "start_clean": idx,
                            "end_clean": idx + len(name),
                        })
                    else:
                        recalculated_gold.append({
                            **gold,
                            "start_clean": -1,
                            "end_clean": -1,
                        })

                documents.append({
                    "file": str(json_file.relative_to(SOCCOR_DIR)),
                    "annotated_text": annotated_text,
                    "clean_text": clean_text,
                    "gold_entities": recalculated_gold,
                    "source": subdir,
                })
            except Exception as e:
                print(f"  [WARNING] Failed to load {json_file}: {e}")

    print(f"\n  Loaded {len(documents)} English SocCor documents")
    total_entities = sum(len(d["gold_entities"]) for d in documents)
    print(f"  Total gold player mentions: {total_entities}")
    return documents


# ─── Error-level evaluation ──────────────────────────────────────────────────

def evaluate_with_errors(model_key: str, model_name: str, documents: list):
    """
    Like evaluate_model_on_soccor but returns detailed FN/FP records
    with the actual player names and context.
    """
    print(f"\n{'='*60}")
    print(f"Error Analysis: {model_name}")
    print(f"{'='*60}")

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"  [ERROR] Model {model_name} not installed.")
        return None

    error_rows = []  # list of dicts: {model, file, source, error_type, player_name, ...}
    tp_total = 0
    fn_total = 0
    fp_total = 0

    for doc_info in documents:
        clean_text = doc_info["clean_text"]
        gold_players = doc_info["gold_entities"]
        doc_file = doc_info["file"]
        doc_source = doc_info["source"]

        doc = nlp(clean_text)
        spacy_persons = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                spacy_persons.append({
                    "text": ent.text.strip(),
                    "text_lower": ent.text.strip().lower(),
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

        matched_spacy = set()

        for gold in gold_players:
            gold_name_lower = gold["player_name"].lower()
            gold_surname = gold_name_lower.split()[-1] if gold_name_lower else ""
            gold_parts = gold_name_lower.split()

            found = False
            matched_spacy_text = ""
            for i, sp in enumerate(spacy_persons):
                if i in matched_spacy:
                    continue
                sp_lower = sp["text_lower"]

                if (gold_name_lower == sp_lower or
                    gold_name_lower in sp_lower or
                    sp_lower in gold_name_lower or
                    gold_surname == sp_lower or
                    gold_surname in sp_lower or
                    (len(gold_parts) > 1 and any(part == sp_lower for part in gold_parts))):
                    found = True
                    matched_spacy.add(i)
                    matched_spacy_text = sp["text"]
                    break

            if found:
                tp_total += 1
            else:
                fn_total += 1
                # Extract a snippet of context around the gold entity position
                start = max(0, gold.get("start_clean", 0) - 40)
                end = min(len(clean_text), gold.get("end_clean", 0) + 40)
                context_snippet = clean_text[start:end].replace("\n", " ").strip()

                error_rows.append({
                    "model": model_name,
                    "model_key": model_key,
                    "file": doc_file,
                    "source": doc_source,
                    "error_type": "FN",
                    "player_name": gold["player_name"],
                    "tag_text": gold["tag_text"],
                    "position": gold["position"],
                    "country": gold["country"],
                    "spacy_text": "",
                    "context": context_snippet,
                })

        # FP: spaCy PERSON entities not matched to any gold
        for i, sp in enumerate(spacy_persons):
            if i not in matched_spacy:
                fp_total += 1
                start = max(0, sp["start"] - 40)
                end = min(len(clean_text), sp["end"] + 40)
                context_snippet = clean_text[start:end].replace("\n", " ").strip()

                error_rows.append({
                    "model": model_name,
                    "model_key": model_key,
                    "file": doc_file,
                    "source": doc_source,
                    "error_type": "FP",
                    "player_name": "",
                    "tag_text": "",
                    "position": "",
                    "country": "",
                    "spacy_text": sp["text"],
                    "context": context_snippet,
                })

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  TP={tp_total}, FN={fn_total}, FP={fp_total}")
    print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  Error records collected: {len(error_rows)}")

    del nlp
    return error_rows


def main():
    parser = argparse.ArgumentParser(
        description="Detailed SocCor NER error analysis"
    )
    parser.add_argument("--models", nargs="+", default=["sm", "lg"],
                        help="Models to evaluate: sm, lg, trf")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("Loading SocCor data with Players.csv metadata")
    print(f"{'='*60}")

    token_to_name = load_player_metadata()
    documents = load_soccor_english(token_to_name)

    if not documents:
        print("[ERROR] No SocCor documents found!")
        return

    all_errors = []
    model_keys_evaluated = []
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"[WARNING] Unknown model: {model_key}")
            continue
        errors = evaluate_with_errors(model_key, MODELS[model_key], documents)
        if errors is not None:
            all_errors.extend(errors)
            model_keys_evaluated.append(model_key)

    NER_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed error CSV
    error_df = pd.DataFrame(all_errors)
    error_df.to_csv(NER_OUTPUT / "soccor_error_detail.csv", index=False)
    print(f"\n  Saved: {NER_OUTPUT / 'soccor_error_detail.csv'} ({len(error_df)} rows)")

    # ─── Analysis ────────────────────────────────────────────────────────────
    fn_df = error_df[error_df["error_type"] == "FN"]
    fp_df = error_df[error_df["error_type"] == "FP"]

    with open(NER_OUTPUT / "soccor_error_summary.txt", "w", encoding="utf-8") as f:
        f.write("SocCor NER Error Analysis\n")
        f.write("=" * 72 + "\n\n")

        # ── Per-model FN analysis ────────────────────────────────────────────
        for model_key in model_keys_evaluated:
            model_name = MODELS[model_key]
            model_fn = fn_df[fn_df["model"] == model_name]
            model_fp = fp_df[fp_df["model"] == model_name]

            f.write(f"\n{'='*72}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*72}\n")
            f.write(f"  False Negatives (missed players): {len(model_fn)}\n")
            f.write(f"  False Positives (spurious detections): {len(model_fp)}\n\n")

            # Most frequently missed players
            f.write("  TOP-20 MOST FREQUENTLY MISSED PLAYERS:\n")
            missed_counts = model_fn["player_name"].value_counts().head(20)
            for player, count in missed_counts.items():
                countries = model_fn[model_fn["player_name"] == player]["country"].unique()
                positions = model_fn[model_fn["player_name"] == player]["position"].unique()
                f.write(f"    {player} ({', '.join(countries)}, {', '.join(positions)}): missed {count} times\n")

            # Missed by position
            f.write("\n  MISSES BY POSITION:\n")
            pos_counts = model_fn["position"].value_counts()
            for pos, count in pos_counts.items():
                f.write(f"    {pos}: {count}\n")

            # Missed by country
            f.write("\n  MISSES BY COUNTRY (top-15):\n")
            country_counts = model_fn["country"].value_counts().head(15)
            for country, count in country_counts.items():
                f.write(f"    {country}: {count}\n")

            # Missed by document type
            f.write("\n  MISSES BY DOCUMENT TYPE:\n")
            source_counts = model_fn["source"].value_counts()
            for source, count in source_counts.items():
                f.write(f"    {source}: {count}\n")

            # Most common FP texts
            f.write("\n  TOP-20 MOST COMMON FALSE POSITIVES (spurious PERSON entities):\n")
            fp_counts = model_fp["spacy_text"].value_counts().head(20)
            for text, count in fp_counts.items():
                f.write(f"    \"{text}\": {count} times\n")

            # Sample context for FNs
            f.write("\n  SAMPLE MISSED CONTEXTS (first 10):\n")
            for _, row in model_fn.head(10).iterrows():
                f.write(f"    Player: {row['player_name']} | Tag: {row['tag_text']}\n")
                f.write(f"    Context: \"...{row['context']}...\"\n\n")

        # ── Cross-model analysis: shared blind spots ─────────────────────────
        if len(model_keys_evaluated) > 1:
            f.write(f"\n{'='*72}\n")
            f.write("CROSS-MODEL ANALYSIS: SHARED BLIND SPOTS\n")
            f.write(f"{'='*72}\n\n")

            # For each (file, player_name), check if ALL models missed it
            fn_pivot = fn_df.groupby(["file", "player_name", "country", "position"])["model"].apply(set).reset_index()
            all_model_names = {MODELS[k] for k in model_keys_evaluated}
            shared_misses = fn_pivot[fn_pivot["model"] == all_model_names]

            f.write(f"  Players missed by ALL {len(model_keys_evaluated)} models: "
                    f"{len(shared_misses)} instances\n\n")

            if len(shared_misses) > 0:
                # Save to CSV
                shared_misses_export = shared_misses.copy()
                shared_misses_export["models_missed"] = shared_misses_export["model"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                shared_misses_export = shared_misses_export.drop(columns=["model"])
                shared_misses_export.to_csv(
                    NER_OUTPUT / "soccor_errors_all_models.csv", index=False
                )
                print(f"  Saved: {NER_OUTPUT / 'soccor_errors_all_models.csv'}")

                # Most commonly shared missed players
                shared_player_counts = shared_misses["player_name"].value_counts()
                f.write("  TOP-20 PLAYERS MISSED BY ALL MODELS:\n")
                for player, count in shared_player_counts.head(20).items():
                    countries = shared_misses[shared_misses["player_name"] == player]["country"].unique()
                    f.write(f"    {player} ({', '.join(countries)}): {count} instances\n")

                # By position
                f.write("\n  SHARED MISSES BY POSITION:\n")
                for pos, count in shared_misses["position"].value_counts().items():
                    f.write(f"    {pos}: {count}\n")

                # By country
                f.write("\n  SHARED MISSES BY COUNTRY:\n")
                for country, count in shared_misses["country"].value_counts().head(15).items():
                    f.write(f"    {country}: {count}\n")

                # Contextual patterns: check for surname-only, hyphenated, etc.
                f.write("\n  ERROR PATTERN ANALYSIS:\n")

                # Check how many missed names are single-word
                single_word = shared_misses[shared_misses["player_name"].str.split().str.len() == 1]
                f.write(f"    Single-word names missed: {len(single_word)} / {len(shared_misses)} "
                        f"({len(single_word)/max(len(shared_misses),1)*100:.1f}%)\n")

                # Hyphenated names
                hyphenated = shared_misses[shared_misses["player_name"].str.contains("-", na=False)]
                f.write(f"    Hyphenated names missed: {len(hyphenated)} / {len(shared_misses)} "
                        f"({len(hyphenated)/max(len(shared_misses),1)*100:.1f}%)\n")

                # Names with apostrophes
                apostrophe = shared_misses[shared_misses["player_name"].str.contains("'", na=False)]
                f.write(f"    Names with apostrophes missed: {len(apostrophe)} / {len(shared_misses)} "
                        f"({len(apostrophe)/max(len(shared_misses),1)*100:.1f}%)\n")

                # Non-English origin names (rough proxy: non-Western European countries)
                eastern_countries = {"TUR", "GEO", "ROU", "UKR", "SVN", "SRB", "SVK", "CZE",
                                     "HUN", "POL", "CRO", "ALB"}
                eastern_misses = shared_misses[shared_misses["country"].isin(eastern_countries)]
                f.write(f"    Eastern/Central European players missed: {len(eastern_misses)} / {len(shared_misses)} "
                        f"({len(eastern_misses)/max(len(shared_misses),1)*100:.1f}%)\n")

            # ── FP analysis across models ────────────────────────────────────
            f.write(f"\n{'='*72}\n")
            f.write("CROSS-MODEL ANALYSIS: COMMON FALSE POSITIVES\n")
            f.write(f"{'='*72}\n\n")

            fp_pivot = fp_df.groupby(["file", "spacy_text"])["model"].apply(set).reset_index()
            shared_fp = fp_pivot[fp_pivot["model"] == all_model_names]
            f.write(f"  Spurious entities detected by ALL models: {len(shared_fp)}\n")

            if len(shared_fp) > 0:
                fp_text_counts = shared_fp["spacy_text"].value_counts().head(20)
                f.write("\n  TOP-20 FALSE POSITIVES SHARED BY ALL MODELS:\n")
                for text, count in fp_text_counts.items():
                    f.write(f"    \"{text}\": {count} times\n")

    print(f"\n  Saved: {NER_OUTPUT / 'soccor_error_summary.txt'}")
    print(f"\n{'='*60}")
    print("Error analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
