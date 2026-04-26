"""Data loading and preprocessing functions for LoL Win Prediction."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLUMNS_TO_DROP,
    DATA_RAW,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

def load_data(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load the raw data from a CSV
    Args:
        path: Path to the CSV file. Defaults to DATA_RAW from config.
    
    Returns:
        DataFrame with the loaded dat
    """
    df = pd.read_csv(path)
    return df

def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns identified as redundant during EDA.
    
    Removes 12 columns to reduce multicollinearity:
    - identifier (gameId)
    - definitional duplicates (redKills, redDeaths)
    - mirror diffs, linear transforms, aggregates
    
    Args:
        df: Raw DataFrame with all 40 columns.
    
    Returns:
        DataFrame with only retained features and target.
    """
    df = df.drop(columns=COLUMNS_TO_DROP)
    return df

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features matrix X and target vector y.
    
    Args:
        df: DataFrame containing both features and target column.
    
    Returns:
        (X, y) — features DataFrame and target Series.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def split_train_test(X: pd.DataFrame,
                     y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.
    
    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Fraction of data to use as test set.
        random_state: Seed for reproducibility.
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size,
                                                        stratify = y,
                                                        random_state = random_state)
    return X_train, X_test, y_train, y_test