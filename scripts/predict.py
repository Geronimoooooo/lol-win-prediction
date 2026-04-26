"""Run inference using a trained LoLPredictor model.

Usage:
    python -m scripts.predict --input <path_to_csv> --output <path_to_predictions>

Example:
    python -m scripts.predict --input data/raw/high_diamond_ranked_10min.csv --output predictions.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_MODEL_PATH
from src.model import LoLPredictor
from src.preprocessing import drop_redundant_features

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict LoL match outcomes from 10-minute features.')

    parser.add_argument('--input',
                        type=Path,
                        required=True,
                        help='Path to the input CSV with match features.',)
    
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='Path to save the predictions CSV.')
    
    parser.add_argument('--model',
                        type=Path,
                        default=DEFAULT_MODEL_PATH,
                        help=f'Path to the saved model (default: {DEFAULT_MODEL_PATH}).')
    
    return parser.parse_args()

def main() -> None:
    """End-to-end inference pipeline."""
    args = parse_args()

    print('=' * 60)
    print('LoL Win Prediction — Inference Pipeline')
    print('=' * 60)

    print('\n[01/05] Loading data')
    df = pd.read_csv(args.input)
    print(f'Data shape: {df.shape}')

    print('\n[02/05] Preprocessing data')
    df = drop_redundant_features(df)
    X = df.drop(columns=['blueWins'], errors='ignore')
    print(f'Features: {X.shape}')

    print('\n[03/05] Loading trained model')
    predictor = LoLPredictor.load(args.model)
    print('Model loaded successfully.')

    print('\n[04/05] Generating predictions')
    predictions = predictor.predict(X)
    probabilities = predictor.predict_proba(X)

    print('\n[05/05] Saving predictions')
    results = pd.DataFrame({
        'prediction': predictions,
        'probability_blue_wins': probabilities,
    })
    results.to_csv(args.output, index=False)
    print(f'Predictions saved to {args.output}')

    print('\n' + '=' * 60)
    print(f'Done. Total matches scored: {len(predictions)}')
    print('=' * 60)

if __name__ == '__main__':
    main()