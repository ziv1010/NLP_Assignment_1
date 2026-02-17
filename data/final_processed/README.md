# Final Processed Data

This folder contains the **full cleaned dataset** after filtering the original GDELT scrape for football relevance.

## Cleaning Logic
- **Filter**: `clean_dataset.py` with `is_football_article` function.
- **Criteria**: >= 3 unique football keywords (e.g., `goal`, `match`, `league`).
- **Exclusions**: Articles with US Sports terms (`touchdown`, `quarterback`, etc.).

## Stats
- **Men**: 5,927 articles (from 9,743)
- **Women**: 5,322 articles (from 9,540)
