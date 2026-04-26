# LoL Win Prediction: Early-Game Analysis

**Status:** Completed (April 2026).

This project builds a tool that can predict the outcome of a League of Legends match from the first 10 minutes from start.
The tool can be useful for e-sports coaches, game analysts or competitive players who want to understand how early-game decisions affect the final match outcome.
Model outputs the probability of the blue team winning.

## Business Goal

Build a **production-ready** ML system that predicts LoL match outcomes from 
10-minute features, achieving robust ROC-AUC > 0.80 with full reproducibility 
and CLI-based inference.

### Stakeholders
- **E-sports coaches** — understand impact of early-game decisions on match outcome
- **Game analysts at developers** (Riot, etc.) — quantify importance of in-game features
- **Competitive players** — recognize patterns that lead to comebacks vs lost games

## ML Task
- Type: Binary classification
- Target: `blueWins` (1 / 0)
- Input: Aggregated statistics from first 10 minutes (both teams)
- Prediction moment: Minute 10 of the match
- Prediction window: Until match end (~20-40 minutes ahead)

## Dataset
- **Source:** [Kaggle — LoL Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)
- **Size:** ~10,000 matches
- **Rank:** Diamond solo queue
- **Region:** EUW (European West)
- **Features:** Pre-aggregated statistics from the first 10 minutes for both teams (gold, kills, objectives, wards, CS, experience, etc.)

## Metrics
- Primary: ROC-AUC (classes are near-balanced ~50/50)
- Secondary: F1-score
- Additional: Accuracy, confusion matrix for interpretation

## Baselines
- Random guess (~50%)
- Simple Logistic Regression on raw features

## Model Progression

Models are evaluated in order of increasing complexity to identify the simplest model 
that achieves competitive performance:

1. **Logistic Regression** — baseline, requires careful preprocessing (scaling, 
   multicollinearity removal)
2. **Random Forest** — non-linear ensemble, demonstrates overfitting diagnosis 
   and regularization tuning
3. **LightGBM** — gradient boosting with early stopping
4. **Final model selection** — based on test ROC-AUC and train-test gap

## Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, LightGBM
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Notebooks:** Jupyter
- **Code organization:** modular `src/` layout, CLI scripts via `argparse`
- **Reproducibility:** `requirements.txt`, fixed `random_state=42`, separate train/val/test

## Project Structure
```
lol-win-prediction/
├── data/
│   ├── raw/                   # Original datasets (not tracked in git)
│   └── processed/             # Cleaned/transformed data (not tracked in git)
├── notebooks/                 # Jupyter notebooks for exploration and modeling
│   ├── 01_eda.ipynb           # portfolio EDA
│   ├── 02_baseline.ipynb      # Logistic Regression
│   ├── 03_random_forest.ipynb # RF (default + tuned)
│   └── 04_lightgbm.ipynb      # LightGBM (default + tuned) + final comparison
├── src/                       # Source code modules (reusable functions)
│   ├── __init__.py
│   ├── config.py              # paths, hyperparameters
│   ├── preprocessing.py       # methods for preparing the data
│   ├── model.py               # LoLPredictor class (train, predict, save/load)
│   └── evaluation.py          # metrics and visualization
├── scripts/
│   ├── train.py               # train launch and save
│   └── predict.py             # inference for new data
├── reports/
│   └── figures/               # Saved plots and visualizations
├── README.md
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Results & Findings
### Model Comparison

| Model | Test Accuracy | Test ROC-AUC | Train-Test Gap | CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.7201 | **0.8068** | ~0.01 | 0.8107 ± 0.008 |
| Random Forest (default) | 0.7206 | 0.8006 | 0.20 | 0.7973 ± 0.009 |
| Random Forest (tuned) | 0.7191 | 0.8044 | 0.06 | 0.8066 ± 0.009 |
| LightGBM (default) | 0.7146 | 0.7986 | 0.05 | 0.7786 ± 0.0099 |
| LightGBM (tuned) | 0.7212 | 0.8020 | 0.02 | 0.8024 ± 0.0082 |

**1. Logistic regression wins this dataset.** Despite trying three fundamentally 
different algorithms (linear, bagging, boosting), all models converge to ~0.80-0.81 
ROC-AUC. The simplest model performs best.

**2. The dataset exhibits an information ceiling around 0.81 ROC-AUC.** This is 
consistent across models and splits, suggesting we've extracted the available signal 
from the current features.

**3. Feature importance confirms single-feature dominance.** `blueGoldDiff` 
outweighs all other features by an order of magnitude across all models. This 
explains why:
- Non-linear methods (RF, LGBM) don't find additional patterns
- LightGBM's default early stopping triggered at iteration 8 — the model 
  captured the main signal quickly and found nothing useful beyond it

**4. Breaking the 0.81 ceiling requires either:**
- **Feature engineering** — derived features like `wards_in_enemy_jungle`, 
  `kills_ratio`, `gold_diff_per_minute` to capture patterns not expressed linearly
- **More data** — more matches across patches and ranks
- **Richer features** — early-game composition data, player skill levels, 
  item timings, not just aggregated stats

## How to Use

### Setup

```bash
# Clone the repository
git clone https://github.com/Geronimoooooo/lol-win-prediction.git
cd lol-win-prediction

# Create virtual environment (Python 3.12+)
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download from Kaggle: [LoL Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

Place `high_diamond_ranked_10min.csv` in `data/raw/`.

### Train a Model

```bash
python -m scripts.train
```

This will:
- Load the raw dataset
- Apply preprocessing (drop redundant features, train/test split, scaling)
- Train Logistic Regression
- Evaluate on test set
- Save the trained model to `models/logreg_baseline.pkl`

### Make Predictions on New Data

```bash
python -m scripts.predict --input data/raw/high_diamond_ranked_10min.csv --output predictions.csv
```

Predictions are saved as a CSV with two columns:
- `prediction` — predicted class (0 = blue loses, 1 = blue wins)
- `probability_blue_wins` — model confidence for blue victory (0.0 - 1.0)

## Author

**Artyom Papkov**

- [LinkedIn](https://www.linkedin.com/in/artyom-papkov/)
- [GitHub](https://github.com/Geronimoooooo)
