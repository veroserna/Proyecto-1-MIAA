"""Funciones para evaluar modelos de clasificación de audio (protocolo Hito 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Calcula métricas estándar y macro ROC-AUC / cmAP para clasificación multiclase o multietiqueta."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if np.ndim(y_true) == 1
        else None,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            if np.ndim(y_true) == 1:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            else:
                aucs, aps = [], []
                for j in range(y_true.shape[1]):
                    if 0 < y_true[:, j].sum() < len(y_true):
                        aucs.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
                        aps.append(average_precision_score(y_true[:, j], y_prob[:, j]))
                metrics["roc_auc_macro"] = float(np.mean(aucs)) if aucs else float("nan")
                metrics["cmap"] = float(np.mean(aps)) if aps else float("nan")
        except ValueError:
            metrics["roc_auc_macro"] = float("nan")
    if class_names is not None and np.ndim(y_true) == 1:
        per_class_f1 = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
        metrics["per_class_f1"] = dict(zip(class_names, map(float, per_class_f1)))
    return metrics
