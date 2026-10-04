"""Carga local de metadatos y rutas de audio del dataset BirdCLEF+ 2025."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "raw"
AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".flac"}


def load_metadata(split: str = "train", data_dir: str | Path | None = None) -> pd.DataFrame:
    """Lee los metadatos CSV de un split del dataset."""
    dataset_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    metadata_path = dataset_dir / f"{split}.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_birdclef_data(
    split: str = "train",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Carga metadatos y agrega la ruta local correspondiente a cada grabación."""
    dataset_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    metadata = load_metadata(split=split, data_dir=dataset_dir)

    if "filename" not in metadata.columns:
        raise ValueError(f"Los metadatos de '{split}' no contienen la columna 'filename'.")

    audio_dir = dataset_dir / f"{split}_audio"
    metadata["audio_path"] = metadata["filename"].map(
        lambda filename: str(audio_dir / str(filename))
    )
    return metadata


def list_audio_files(directory: str | Path) -> list[Path]:
    """Lista recursivamente archivos de audio compatibles en un directorio."""
    audio_dir = Path(directory)
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"No se encontró el directorio de audio: {audio_dir}")

    return sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
