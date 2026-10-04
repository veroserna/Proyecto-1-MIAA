"""Modelo de referencia tipo ensemble para audio."""


class EnsembleBirdModel:
    """Modelo placeholder para ensemble o voting."""

    def __init__(self, name: str = "ensemble"):
        self.name = name

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        return [self.y_[0] for _ in range(len(X))]
