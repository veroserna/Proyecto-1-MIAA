"""Funciones para evaluar modelos de clasificación de audio."""

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred):
    """Calcula métricas estándar para clasificación multiclase."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
