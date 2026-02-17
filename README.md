# NLP Assignment 1 — NER-Based Analysis of Football Journalism

> **Repository:** [github.com/ziv1010/NLP_Assignment_1](https://github.com/ziv1010/NLP_Assignment_1)

## Overview

This project uses **Named Entity Recognition (NER)** to analyse gendered framing differences in football journalism. We extract and classify entities (players, clubs, competitions, etc.) from ~14,000 articles covering men's and women's football, then test **15 hypotheses** about structural and linguistic framing patterns.

### Key Findings
- **H10 — Name Formality:** Women's players are named formally (full name 62.6%); men's players use surname-only (47.6%)
- **H14 — Meta-Discourse:** Women's coverage uses ~1.9× more growth/visibility language near institutional entities
- **D1 — Entity Diversity:** Women's articles spread mentions more evenly across entity types (medium effect, *d* = −0.44)
- **H12 — Attribute Mix:** Women described with mentality/effort terms; men with physicality/tactical language

---

## Project Structure

```
NLP_Assignment_1/
├── final_codes/                    # Final analysis scripts (use these)
│   ├── 7_run_ner.py                # NER extraction with gazetteer enhancement
│   ├── 8b_evaluate_soccor.py       # SocCor ground-truth evaluation
│   ├── 8c_soccor_error_analysis.py # Detailed NER error analysis
│   ├── 10_hypothesis_analysis.py   # All 15 hypotheses + diagnostic D1
│   ├── classify_kaggle.py          # Gender classification of Kaggle articles
│   ├── prepare_final_inputs.py     # Final corpus assembly
│   └── config.py                   # Shared configuration and paths
│
├── 1_scrape_gdelt.py               # GDELT news scraper
├── 2_scrape_bbc_rss.py             # BBC RSS feed scraper
├── 3_scrape_web_direct.py          # Direct web scraping (UEFA, Goal.com, etc.)
├── 4_download_kaggle.py            # Kaggle dataset downloader
├── 5_clean_and_merge.py            # Data cleaning and deduplication
├── 6_scrape_women_expanded.py      # Expanded women's football scraping
├── run_all_scrapers.py             # Orchestrator for all scrapers
│
├── data/
│   ├── raw/                        # Raw scraped and downloaded data
│   ├── kaggle_processed/           # Processed Kaggle data
│   │   └── final_outputs/
│   │       ├── analysis/           # Hypothesis test plots and results
│   │       └── ner_outputs/        # NER extraction outputs
│   ├── soccor/                     # SocCor evaluation corpus
│   └── final_processed/            # Legacy processed data
│
├── latex/
│   ├── report.tex                  # Full LaTeX report (30 pages)
│   ├── report.pdf                  # Compiled PDF
│   └── figures/                    # All hypothesis plots
│
├── requirements.txt
└── .gitignore
```

---

## Data Pipeline

The pipeline runs in numbered stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1–3, 6 | `1_scrape_gdelt.py` → `6_scrape_women_expanded.py` | Scrape articles from GDELT, BBC RSS, UEFA, Goal.com, etc. |
| 4 | `4_download_kaggle.py` | Download 7 Kaggle football article datasets |
| 5 | `5_clean_and_merge.py` | Clean, deduplicate, and merge scraped data |
| — | `classify_kaggle.py` | Classify Kaggle articles as men's/women's via keywords |
| — | `prepare_final_inputs.py` | Assemble final corpus (11,386 men's + ~2,879 women's) |
| 5a | `8b_evaluate_soccor.py` | Evaluate spaCy models on SocCor ground truth |
| 5b | `8c_soccor_error_analysis.py` | Detailed error analysis of NER failures |
| 6 | `7_run_ner.py` | Run NER with 5 gazetteers + context capture (±50 chars) |
| 7 | `10_hypothesis_analysis.py` | Test 15 hypotheses with statistical rigour |

---

## Hypotheses Tested

### A. Core Football (H1–H5)
| ID | Hypothesis | Verdict |
|----|-----------|---------|
| H1 | Player prominence follows Zipfian distribution | ✅ Supported |
| H2 | Club co-occurrence clusters by competition | ✅ Supported |
| H3 | Manager mentions co-occur with negative terms | ⚠️ Partial |
| H4 | Transfer articles spike MONEY + player-club links | ✅ Supported |
| H5 | Women's coverage shows different injury patterns | ⚠️ Partial |

### B. Men vs. Women Portrayal (H9–H16)
| ID | Hypothesis | Verdict |
|----|-----------|---------|
| H9 | Men's articles name more distinct players | ⚠️ Partial |
| H10 | Women's coverage uses more formal naming | ✅ **Strong** |
| H11 | Youth vs. experience framing differs | ⚠️ Partial |
| H12 | Women: mentality terms; Men: physicality terms | ✅ Supported |
| H13 | Women: relational framing; Men: star framing | ✅ Supported |
| H14 | Women's coverage has more growth/visibility terms | ✅ **Strong** |
| H15 | Credit assignment differs in win contexts | ⚠️ Partial |
| H16 | Framing varies by news source | ✅ Supported (Men's) |

### C. Diagnostic
| ID | Hypothesis | Verdict |
|----|-----------|---------|
| D1 | Entity diversity entropy differs by gender | ✅ Significant |

All tests use **Benjamini-Hochberg FDR correction** across 34 tests, with effect sizes (Cohen's *d*, Cramér's *V*, Cohen's *h*) and bootstrap 95% CIs.

---

## NER Pipeline

- **Model:** `en_core_web_lg` (F1 = 0.812, Recall = 0.880 on SocCor)
- **5 Gazetteers:** Competition (40+), Media (35+), Club (349), Governing Body (20+), Player (1,279 from SocCor)
- **Context capture:** ±50 character window around each entity mention
- **Output:** 37,089 men's entities + 24,464 women's entities

---

## Setup & Usage

### Requirements
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Run Hypothesis Analysis
```bash
python final_codes/10_hypothesis_analysis.py \
    --ner-dir data/kaggle_processed/final_outputs/ner_outputs/lg \
    --art-dir data/kaggle_processed/final_outputs \
    --output-dir data/kaggle_processed/final_outputs/analysis
```

### Compile Report
```bash
cd latex && pdflatex report.tex && pdflatex report.tex
```

---

## Report

The full **30-page LaTeX report** is available at [`latex/report.pdf`](latex/report.pdf). It covers:
- Data collection and processing pipeline
- SocCor NER evaluation with detailed error analysis
- Gazetteer-enhanced entity classification
- All 15 hypothesis results with statistical tests and plots
- Discussion, limitations, and future work

---

## License

This project was developed as part of an NLP course assignment.