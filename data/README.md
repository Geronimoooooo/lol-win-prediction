# Data Directory

This directory is intentionally tracked as a placeholder for datasets.
Actual data files (CSV, JSON, etc.) are NOT committed to this repository
for licensing and size reasons. See `.gitignore` for excluded patterns.

## Structure

- `raw/` — Original, unmodified datasets (ignored by git)
- `processed/` — Cleaned and transformed data (ignored by git)

## How to Reproduce

1. Download the dataset from Kaggle:
   [LoL Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
2. Place `high_diamond_ranked_10min.csv` in `data/raw/`