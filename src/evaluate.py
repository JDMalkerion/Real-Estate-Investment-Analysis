"""
Evaluation and visualization utilities for MLflow tracking.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate standard classification performance metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate standard regression performance metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """Generate and save confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Recommended (0)", "Good Investment (1)"],
        yticklabels=["Not Recommended (0)", "Good Investment (1)"]
    )
    plt.title(title, fontsize=12, pad=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str,
    output_path: Path,
    top_n: int = 15
) -> None:
    """Generate and save horizontal feature importance bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx][::-1]
    top_scores = importances[idx][::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_names)), top_scores, color="#1f77b4", align="center")
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel("Importance Score")
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """Generate and save residual scatter and error distribution plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: Predicted vs Residuals
    sample_size = min(3000, len(y_true))
    idx = np.random.choice(len(y_true), sample_size, replace=False)
    axes[0].scatter(y_pred[idx], residuals[idx], alpha=0.3, color="#1f77b4", edgecolors="none")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Values (₹ Lakhs)")
    axes[0].set_ylabel("Residuals (True - Pred)")
    axes[0].set_title("Residuals vs Fitted")

    # Histogram: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c", bins=40)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual Error (₹ Lakhs)")
    axes[1].set_title("Residual Error Distribution")

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
