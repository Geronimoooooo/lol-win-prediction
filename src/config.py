"""Configuration constants for the LoL Win Prediction project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw' / 'high_diamond_ranked_10min.csv'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'logreg_baseline.pkl'

# Target columns
TARGET = 'blueWins'

# Features to drop
COLUMNS_TO_DROP = [
    # Identifier
    'gameId',
    # Definitional duplicates
    'redKills',
    'redDeaths',
    # Deterministic complement
    'redFirstBlood',
    # Linear transforms
    'blueGoldPerMin',
    'redGoldPerMin',
    'blueCSPerMin',
    'redCSPerMin',
    # Mirror diffs
    'redGoldDiff',
    'redExperienceDiff',
    # Aggregates (keep detail)
    'blueEliteMonsters',
    'redEliteMonsters',
]

# Train test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters
LOGREG_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
}

# Cross-validation
CV_FOLDS = 5