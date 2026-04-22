# LoL Win Prediction: Early-Game Analysis
Create a tool which can predict an outcome of a match after 10 minutes from start.
Basically the tool can be useful for e-sports trainers, game analytics or players who want to understand their mistakes in early game phase.
Model gives only prediction of wining as a variety of blue team win chance.
Now in development.

## Business Goal
Build a model that predicts the outcome of a League of Legends match 
based on the first 10 minutes of gameplay. 

Practical value:
- For e-sports coaches: understand how critical early-game mistakes are
- For game analysts: identify which early factors determine match outcome

## ML Task
- Type: Binary classification
- Target: `blue_team_wins` (1 / 0)
- Input: Aggregated statistics from first 10 minutes (both teams)
- Prediction moment: Minute 10 of the match
- Prediction window: Until match end (~20-40 minutes ahead)

## Dataset
Kaggle: "League of Legends Diamond Ranked Games (10 min)"
- ~10K matches, Diamond ranked solo queue
- Pre-aggregated first-10-min features (gold, kills, objectives, wards, etc.)

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
TBD (will be defined in first commit)

## Results
TBD (will be described after development, first results after 1 and 2 points of Model Progression)

## How to run
TBD (will be described after development)

## Author
Papkov Artyom:
[Linkedin](https://www.linkedin.com/in/artyom-papkov/)
[GitHub](https://github.com/Geronimoooooo)
[Telegram](@geronimoooooooo)

