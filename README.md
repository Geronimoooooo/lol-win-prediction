# LoL Win Prediction: Early-Game Analysis

**Status:** Currently in development.

This project builds a tool that can predict the outcome of a League of Legends match from the first 10 minutes from start.
The tool can be useful for e-sports coaches, game analysts or competitive players who want to understand how early-game decisions affect the final match outcome.
Model outputs the probability of the blue team winning.

## Business Goal
Build a model that predicts the outcome of a League of Legends match 
based on the first 10 minutes of gameplay. 

Practical value:
- For e-sports coaches: understand how critical early-game mistakes are
- For game analysts: identify which early factors determine match outcome

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
1. Logistic Regression (baseline + teaches preprocessing)
2. Random Forest (out-of-the-box tree ensemble)
3. LightGBM (main model with cross-validation and early stopping)
4. [Optional] Hyperparameter tuning via Optuna

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
│   └── evaluation.py          # metric's method
├── scripts/
│   ├── train.py               # train launch and save
│   └── predict.py             # inference for new data
├── reports/
│   └── figures/               # Saved plots and visualizations
├── README.md
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Results
TBD (will be described after development, first results after 1 and 2 points of Model Progression)

## How to run
TBD (will be described after development)

## Author

**Artyom Papkov**

- [LinkedIn](https://www.linkedin.com/in/artyom-papkov/)
- [GitHub](https://github.com/Geronimoooooo)

