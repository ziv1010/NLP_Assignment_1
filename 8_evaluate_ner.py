#!/usr/bin/env python3
"""
8_evaluate_ner.py – Evaluate NER models on CoNLL-2003 benchmark and
compare performance across spaCy model sizes.

Outputs:
  - data/ner_outputs/evaluation_results.csv
  - data/ner_outputs/evaluation_results.pkl
  - Prints comparison table

Usage:
    python 8_evaluate_ner.py
    python 8_evaluate_ner.py --models sm lg trf
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

from config import logger

NER_OUTPUT = Path("data/ner_outputs")

MODELS = {
    "sm": "en_core_web_sm",
    "lg": "en_core_web_lg",
    "trf": "en_core_web_trf",
}

# CoNLL-2003 label mapping: dataset int → string
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# spaCy label → CoNLL label mapping
SPACY_TO_CONLL = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "PRODUCT": "MISC",
}


def align_spacy_to_conll(doc, tokens):
    """
    Align spaCy entity predictions to CoNLL token-level BIO tags.
    Returns list of predicted BIO tags matching the input tokens.
    """
    pred_tags = ["O"] * len(tokens)

    # Build character offset → token index mapping
    char_to_token = {}
    offset = 0
    for i, token in enumerate(tokens):
        for c in range(offset, offset + len(token)):
            char_to_token[c] = i
        offset += len(token) + 1  # +1 for space

    # Map spaCy entities to token indices
    text = " ".join(tokens)
    spacy_doc = doc

    for ent in spacy_doc.ents:
        conll_label = SPACY_TO_CONLL.get(ent.label_)
        if not conll_label:
            continue

        # Find token indices for this entity
        start_token = char_to_token.get(ent.start_char)
        end_token = char_to_token.get(ent.end_char - 1)

        if start_token is not None and end_token is not None:
            pred_tags[start_token] = f"B-{conll_label}"
            for t in range(start_token + 1, end_token + 1):
                if t < len(pred_tags):
                    pred_tags[t] = f"I-{conll_label}"

    return pred_tags


def compute_metrics(true_tags_all, pred_tags_all):
    """Compute per-entity-type and overall precision, recall, F1."""
    # Count true positives, false positives, false negatives at entity level
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    def extract_entities(tags):
        """Extract entity spans from BIO tags."""
        entities = []
        current = None
        start = -1
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if current:
                    entities.append((current, start, i))
                current = tag[2:]
                start = i
            elif tag.startswith("I-"):
                if current != tag[2:]:
                    if current:
                        entities.append((current, start, i))
                    current = tag[2:]
                    start = i
            else:
                if current:
                    entities.append((current, start, i))
                current = None
        if current:
            entities.append((current, start, len(tags)))
        return set((etype, s, e) for etype, s, e in entities)

    for true_tags, pred_tags in zip(true_tags_all, pred_tags_all):
        true_ents = extract_entities(true_tags)
        pred_ents = extract_entities(pred_tags)

        for ent in true_ents & pred_ents:
            tp[ent[0]] += 1
        for ent in pred_ents - true_ents:
            fp[ent[0]] += 1
        for ent in true_ents - pred_ents:
            fn[ent[0]] += 1

    results = {}
    all_types = set(list(tp.keys()) + list(fp.keys()) + list(fn.keys()))
    total_tp = total_fp = total_fn = 0

    for etype in sorted(all_types):
        p = tp[etype] / (tp[etype] + fp[etype]) if (tp[etype] + fp[etype]) > 0 else 0
        r = tp[etype] / (tp[etype] + fn[etype]) if (tp[etype] + fn[etype]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[etype] = {"precision": p, "recall": r, "f1": f1, "support": tp[etype] + fn[etype]}
        total_tp += tp[etype]
        total_fp += fp[etype]
        total_fn += fn[etype]

    # Overall (micro-average)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    results["OVERALL"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1,
                          "support": total_tp + total_fn}

    return results


def evaluate_on_conll(model_key: str, model_name: str):
    """Evaluate a spaCy model on CoNLL-2003 test set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name} on CoNLL-2003 test set")
    logger.info(f"{'='*60}")

    # Load model
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.error(f"Model {model_name} not installed")
        return None

    # Load CoNLL-2003 dataset (use parquet revision for compatibility)
    logger.info("Loading CoNLL-2003 from HuggingFace...")
    try:
        dataset = load_dataset("conll2003", revision="refs/convert/parquet")
    except Exception as e:
        logger.error(f"Failed to load CoNLL-2003: {e}")
        return None

    test_data = dataset["test"]
    logger.info(f"Test set: {len(test_data)} sentences")

    true_tags_all = []
    pred_tags_all = []

    for example in tqdm(test_data, desc=f"  CoNLL eval ({model_key})"):
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        # Convert int tags to string BIO tags
        true_tags = [CONLL_LABELS[t] for t in ner_tags]
        true_tags_all.append(true_tags)

        # Run spaCy on joined text
        text = " ".join(tokens)
        doc = nlp(text)
        pred_tags = align_spacy_to_conll(doc, tokens)
        pred_tags_all.append(pred_tags)

    # Compute metrics
    results = compute_metrics(true_tags_all, pred_tags_all)

    # Print results table
    print(f"\n{'='*60}")
    print(f"CoNLL-2003 Results: {model_name}")
    print(f"{'='*60}")
    print(f"{'Entity Type':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for etype in ["PER", "ORG", "LOC", "MISC", "OVERALL"]:
        if etype in results:
            r = results[etype]
            print(f"{etype:<12} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f} {r['support']:>10}")
    print()

    del nlp
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER models on CoNLL-2003")
    parser.add_argument("--models", nargs="+", default=["sm", "lg", "trf"])
    args = parser.parse_args()

    all_results = {}
    for model_key in args.models:
        if model_key not in MODELS:
            continue
        results = evaluate_on_conll(model_key, MODELS[model_key])
        if results:
            all_results[model_key] = results

    # Build comparison table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON (CoNLL-2003 Overall)")
        print(f"{'='*60}")
        print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 57)
        for key, results in all_results.items():
            r = results["OVERALL"]
            print(f"{MODELS[key]:<25} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1']:>10.3f}")

    # Save results
    NER_OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_key, results in all_results.items():
        for etype, metrics in results.items():
            rows.append({
                "model": MODELS[model_key],
                "model_key": model_key,
                "entity_type": etype,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
            })

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(NER_OUTPUT / "evaluation_results.csv", index=False)
    eval_df.to_pickle(NER_OUTPUT / "evaluation_results.pkl")
    logger.info(f"Evaluation results saved to {NER_OUTPUT}/evaluation_results.csv")


if __name__ == "__main__":
    main()
