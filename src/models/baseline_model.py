"""Modelo base para clasificación de especies."""


class BaselineBirdModel:
    """Representa un modelo de referencia para identificación de aves."""

    def __init__(self, name: str = "baseline"):
        self.name = name

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        return [self.y_[0] for _ in range(len(X))]
