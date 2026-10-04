"""Entrenamiento de un clasificador SVM sobre features acústicas."""


def train_svm(model, X_train, y_train):
    """Entrena un modelo SVM y devuelve el resultado."""
    model.fit(X_train, y_train)
    return {"status": "trained", "model": model.__class__.__name__}
