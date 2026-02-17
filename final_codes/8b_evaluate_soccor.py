#!/usr/bin/env python3
"""
8b_evaluate_soccor.py – Evaluate NER models on the SocCor UEFA EURO 2024
football corpus (domain-specific player mention benchmark).

SocCor uses inline tags like <BELLINGHAM_ATTACKING-MIDFIELD_ENG> as ground-truth
player annotations. We use Players.csv to map tokens to real full names,
strip the tags, run spaCy NER, and compare.

Outputs:
  - data/ner_outputs/soccor_evaluation.csv
  - data/ner_outputs/soccor_evaluation.pkl
  - data/ner_outputs/soccor_per_doc.csv

Usage:
    python final_codes/8b_evaluate_soccor.py
    python final_codes/8b_evaluate_soccor.py --models sm lg trf
"""

import argparse
import csv
import json
import re
from collections import defaultdict
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

# SocCor tag pattern: <NAME_POSITION_COUNTRY>
TAG_PATTERN = re.compile(r"<([A-Z\-']+)_([A-Z\-]+)_([A-Z]{3})>")


def load_player_metadata() -> dict:
    """
    Load Players.csv and build a mapping from token -> full name.
    e.g. '<WIRTZ_ATTACKING-MIDFIELD_GER>' -> 'Florian Wirtz'
    """
    players_csv = METADATA_DIR / "Players.csv"
    token_to_name = {}

    if not players_csv.exists():
        print(f"  [WARNING] Players.csv not found at {players_csv}, falling back to tag-derived names")
        return token_to_name

    # CSV uses semicolons as delimiter
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
    """Derive a name from the tag itself as a fallback (e.g. BELLINGHAM -> Bellingham)."""
    raw = tag_match.group(1).replace("-", " ").title()
    # Fix apostrophes like O'Brien
    raw = re.sub(r"(\w)'(\w)", lambda m: m.group(1) + "'" + m.group(2), raw)
    return raw


def load_soccor_english(token_to_name: dict):
    """
    Load all English (BBC) SocCor texts and extract gold player annotations.
    Uses Players.csv to map tokens to real full names when available.
    """
    documents = []

    for subdir in ["Reports", "Highlights", "Games", "Livetickers"]:
        bbc_dir = SOCCOR_DIR / subdir / "BBC"
        if not bbc_dir.exists():
            continue
        for json_file in sorted(bbc_dir.glob("*.json")):
            try:
                raw = json_file.read_text(encoding="utf-8")

                # Handle both single-object and array-of-objects JSON
                data = json.loads(raw)
                if isinstance(data, list):
                    # Concatenate all text segments
                    annotated_text = " ".join(item.get("text", "") for item in data if isinstance(item, dict))
                elif isinstance(data, dict):
                    annotated_text = data.get("text", "")
                else:
                    continue

                if not annotated_text or len(annotated_text) < 20:
                    continue

                # ── Extract gold entities ────────────────────────────────
                gold_entities = []
                for match in TAG_PATTERN.finditer(annotated_text):
                    tag_text = match.group(0)
                    position = match.group(2).replace("-", " ").title()
                    country = match.group(3)

                    # Try Players.csv first, fallback to tag-derived name
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

                # ── Strip tags to get clean text for NER ─────────────────
                def replace_tag(m):
                    t = m.group(0)
                    if t in token_to_name:
                        return token_to_name[t]
                    return tag_to_fallback_name(m)

                clean_text = TAG_PATTERN.sub(replace_tag, annotated_text)

                # Recalculate gold entity positions in the clean text
                recalculated_gold = []
                for gold in gold_entities:
                    name = gold["player_name"]
                    # Find position of this name in clean text
                    # (search starting near where we'd expect it)
                    idx = clean_text.find(name)
                    if idx >= 0:
                        recalculated_gold.append({
                            **gold,
                            "start_clean": idx,
                            "end_clean": idx + len(name),
                        })
                    else:
                        # Name might not be found verbatim (accent issues etc.)
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


def evaluate_model_on_soccor(model_key: str, model_name: str, documents: list):
    """
    Evaluate a spaCy model's ability to detect player mentions in SocCor.

    For each gold player mention, check if spaCy found a PERSON entity
    overlapping / matching that player name in the text.
    """
    print(f"\n{'='*60}")
    print(f"SocCor Evaluation: {model_name}")
    print(f"{'='*60}")

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"  [ERROR] Model {model_name} not installed. Run: python -m spacy download {model_name}")
        return None

    tp = 0  # True positives: gold player found by spaCy as PERSON
    fn = 0  # False negatives: gold player missed by spaCy
    fp = 0  # False positives: spaCy found PERSON not in gold set

    per_doc_results = []

    for doc_info in documents:
        clean_text = doc_info["clean_text"]
        gold_players = doc_info["gold_entities"]

        # Run spaCy
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

        # Match gold → spaCy predictions
        matched_spacy = set()
        doc_tp = 0
        doc_fn = 0

        for gold in gold_players:
            gold_name_lower = gold["player_name"].lower()
            # Surname = last word of full name
            gold_surname = gold_name_lower.split()[-1] if gold_name_lower else ""
            # First name
            gold_parts = gold_name_lower.split()

            found = False
            for i, sp in enumerate(spacy_persons):
                if i in matched_spacy:
                    continue
                sp_lower = sp["text_lower"]

                # Match conditions (from strictest to loosest):
                # 1. Exact full name match
                # 2. Gold name contains spaCy text or vice versa
                # 3. Surname match
                if (gold_name_lower == sp_lower or
                    gold_name_lower in sp_lower or
                    sp_lower in gold_name_lower or
                    gold_surname == sp_lower or
                    gold_surname in sp_lower or
                    (len(gold_parts) > 1 and any(part == sp_lower for part in gold_parts))):
                    found = True
                    matched_spacy.add(i)
                    break

            if found:
                tp += 1
                doc_tp += 1
            else:
                fn += 1
                doc_fn += 1

        # FP: spaCy PERSON entities not matched to any gold
        doc_fp = len(spacy_persons) - len(matched_spacy)
        fp += doc_fp

        per_doc_results.append({
            "file": doc_info["file"],
            "source": doc_info["source"],
            "gold_count": len(gold_players),
            "spacy_person_count": len(spacy_persons),
            "tp": doc_tp,
            "fn": doc_fn,
            "fp": doc_fp,
        })

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"SocCor Results: {model_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Value':>10}")
    print("-" * 27)
    print(f"{'True Pos':<15} {tp:>10}")
    print(f"{'False Neg':<15} {fn:>10}")
    print(f"{'False Pos':<15} {fp:>10}")
    print(f"{'Precision':<15} {precision:>10.3f}")
    print(f"{'Recall':<15} {recall:>10.3f}")
    print(f"{'F1-Score':<15} {f1:>10.3f}")
    print()

    del nlp

    return {
        "model": model_name,
        "model_key": model_key,
        "tp": tp, "fn": fn, "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_doc": per_doc_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NER models on SocCor football corpus"
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

    all_results = {}
    for model_key in args.models:
        if model_key not in MODELS:
            print(f"[WARNING] Unknown model: {model_key}")
            continue
        result = evaluate_model_on_soccor(model_key, MODELS[model_key], documents)
        if result:
            all_results[model_key] = result

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON (SocCor — Player Detection)")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 57)
        for key, result in all_results.items():
            print(f"{result['model']:<25} {result['precision']:>10.3f} "
                  f"{result['recall']:>10.3f} {result['f1']:>10.3f}")

    # Save results
    NER_OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_key, result in all_results.items():
        rows.append({
            "model": result["model"],
            "model_key": model_key,
            "benchmark": "SocCor",
            "entity_type": "PLAYER",
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "true_positives": result["tp"],
            "false_negatives": result["fn"],
            "false_positives": result["fp"],
        })

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(NER_OUTPUT / "soccor_evaluation.csv", index=False)
    eval_df.to_pickle(NER_OUTPUT / "soccor_evaluation.pkl")

    # Per-doc details
    all_doc_results = []
    for model_key, result in all_results.items():
        for doc in result["per_doc"]:
            doc["model"] = result["model"]
            all_doc_results.append(doc)
    pd.DataFrame(all_doc_results).to_csv(NER_OUTPUT / "soccor_per_doc.csv", index=False)

    print(f"\nSocCor evaluation saved to {NER_OUTPUT}/")
    print("  - soccor_evaluation.csv / .pkl")
    print("  - soccor_per_doc.csv")


if __name__ == "__main__":
    main()
