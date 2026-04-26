"""Train a LoLPredictor model and save it to disk.

Usage:
    python -m scripts.train
"""

from src.config import DEFAULT_MODEL_PATH
from src.evaluation import print_metrics_report
from src.model import LoLPredictor
from src.preprocessing import (
    drop_redundant_features,
    load_data,
    split_features_target,
    split_train_test,
)

def main() -> None:
    """End-to-end training pipeline."""
    print('=' * 60)
    print('LoL Win Prediction — Training Pipeline')
    print('=' * 60)

    print('\n[01/05] Loading data')
    df = load_data()
    print(f'Data shape: {df.shape}')

    print('\n[02/05] Preprocessing data')
    df = drop_redundant_features(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    print('\n[03/05] Training model')
    predictor = LoLPredictor()
    predictor.train(X_train, y_train)
    print('Model training complete.')

    print('\n[04/05] Evaluating on test set')
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)
    print_metrics_report(y_test, y_pred, y_pred_proba)

    print(f'\n[05/05] Saving model to {DEFAULT_MODEL_PATH}')
    predictor.save()
    print('Model saved successfully.')

    print('\n' + '=' * 60)
    print('Training complete.')
    print('=' * 60)

if __name__ == '__main__':
    main()