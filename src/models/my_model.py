"""Definición del modelo base del proyecto."""


class MyModel:
    """Clase placeholder para la arquitectura del proyecto."""

    def __init__(self, name: str = "modelo_base"):
        self.name = name

    def predict(self, data):
        """Ejemplo de predicción."""
        return {"model": self.name, "input_size": len(data) if hasattr(data, "__len__") else 0}
