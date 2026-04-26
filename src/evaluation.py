"""Evaluation metrics and visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """Compute key classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1.
    
    Returns:
        Dict with keys: 'accuracy', 'roc_auc', 'f1'.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred)
    }

def print_metrics_report(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
    """Print a formatted text report of all metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1.
    """
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['Blue Loses', 'Blue Wins']))
    pass

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, ax=None) -> None:
    """Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        ax: Optional matplotlib Axes; if None, creates a new figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Blue Loses', 'BlueWins'],
                yticklabels=['Blue Loses', 'BlueWins'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pass

def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, ax=None, label: str = 'Model') -> None:
    """Plot ROC curve with AUC annotation.
    
    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for class 1.
        ax: Optional matplotlib Axes; if None, creates a new figure.
        label: Label for the curve in legend.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUC=0,5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    pass