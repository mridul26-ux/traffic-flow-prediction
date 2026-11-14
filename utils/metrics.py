"""
metrics.py
Evaluation Metrics for Traffic Flow Regression and Congestion Classification.
Computes MAE, RMSE, MAPE, R2, Accuracy, Precision, Recall, F1, and Confusion Matrices.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def calculate_regression_metrics(y_true, y_pred):
    """
    Computes standard time-series regression metrics:
      - Mean Absolute Error (MAE)
      - Root Mean Squared Error (RMSE)
      - Mean Absolute Percentage Error (MAPE)
      - Coefficient of Determination (R^2)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Safe MAPE handling zero or near-zero flow
    nonzero_mask = np.abs(y_true) > 1e-4
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
    else:
        mape = 0.0
        
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 2),
        "R2 Score": round(float(r2), 4)
    }


def calculate_classification_metrics(y_true, y_pred, class_names=None):
    """
    Computes classification metrics for congestion level / vehicle detection:
      - Accuracy
      - Macro & Weighted Precision, Recall, F1-Score
      - Confusion Matrix
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    else:
        report = classification_report(y_true, y_pred, zero_division=0)
        
    return {
        "Accuracy": round(float(acc), 4),
        "Macro Precision": round(float(prec_macro), 4),
        "Macro Recall": round(float(rec_macro), 4),
        "Macro F1": round(float(f1_macro), 4),
        "Weighted Precision": round(float(prec_weighted), 4),
        "Weighted Recall": round(float(rec_weighted), 4),
        "Weighted F1": round(float(f1_weighted), 4),
        "Confusion Matrix": cm,
        "Classification Report": report
    }
