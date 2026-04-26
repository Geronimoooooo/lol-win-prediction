"""LoLPredictor class — wrapper around the trained model with scaler."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_MODEL_PATH, LOGREG_PARAMS

class LoLPredictor:
    """Logistic regression predictor for LoL match outcomes.
    
    Encapsulates the StandardScaler and LogisticRegression model to ensure 
    consistent preprocessing during inference.
    """
    def __init__(self, params: dict = None) -> None:
        """Initialize the predictor with model hyperparameters.
        
        Args:
            params: Dict of LogisticRegression hyperparameters. 
                Defaults to LOGREG_PARAMS from config if None.
        """
        params = LOGREG_PARAMS if params is None else params
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**params)
        pass

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the scaler on X, then train the model.
        
        Args:
            X: Training features.
            y: Training target.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary class labels (0 or 1).
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of predicted classes (0 = blue loses, 1 = blue wins).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of the positive class (blue wins).
        
        Args:
            X: Features to predict on.
        
        Returns:
            Array of probabilities for class 1.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        """Save the entire predictor (scaler + model) to a pickle file.
        
        Args:
            path: Destination path. Defaults to DEFAULT_MODEL_PATH.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open (path, 'wb') as f:
            pickle.dump(self, f)
        pass

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> 'LoLPredictor':
        """Load a previously saved predictor from disk.
        
        Args:
            path: Path to the pickle file.
        
        Returns:
            Loaded LoLPredictor instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)