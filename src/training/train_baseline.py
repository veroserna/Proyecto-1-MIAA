"""Entrenamiento del modelo base para clasificación de audio."""


def train_baseline(model, X_train, y_train):
    """Entrena un modelo base y devuelve el resultado."""
    model.fit(X_train, y_train)
    return {"status": "trained", "model": model.__class__.__name__}
