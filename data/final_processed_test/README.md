# Final Processed Test Data

This folder contains a **clean subset** for testing and analysis, randomly sampled from `data/final_processed/`.

## Contents
- **Men**: 1,500 articles
- **Women**: 1,500 articles

## NER Outputs
Run `python 7_run_ner.py --input-dir data/final_processed_test` to populate `ner_outputs/`.
Results include:
- `men_entities.csv` / `women_entities.csv` (Full entity list)
- `men_entity_freq.csv` / `women_entity_freq.csv` (Aggregated counts)
