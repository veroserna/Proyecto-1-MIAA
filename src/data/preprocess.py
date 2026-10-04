"""Funciones para cargar y preparar datos."""

from pathlib import Path


def load_data(file_path: str):
    """Carga un archivo de datos desde la ruta indicada."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    return path


def clean_data(data):
    """Placeholder para limpieza de datos."""
    return data
