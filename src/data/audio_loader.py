"""Carga local de metadatos y rutas de audio del dataset BirdCLEF+ 2025."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.helpers import batch_ogg_header_info, ogg_header_info

DATA_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".flac"}


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resuelve el directorio de datos crudos (por defecto `src/data/raw`, con fallback a raíz)."""
    if data_dir is not None:
        return Path(data_dir)
    if (DATA_DIR / "train.csv").is_file():
        return DATA_DIR
    root_dir = DATA_DIR.parents[2]
    if (root_dir / "train.csv").is_file():
        return root_dir
    return DATA_DIR


def load_metadata(split: str = "train", data_dir: str | Path | None = None) -> pd.DataFrame:
    """Lee los metadatos CSV de un split del dataset."""
    dataset_dir = resolve_data_dir(data_dir)
    metadata_path = dataset_dir / f"{split}.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {metadata_path}")
    return pd.read_csv(metadata_path)


def load_taxonomy(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Lee la tabla taxonómica `taxonomy.csv`."""
    dataset_dir = resolve_data_dir(data_dir)
    tax_path = dataset_dir / "taxonomy.csv"
    if not tax_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de taxonomía: {tax_path}")
    return pd.read_csv(tax_path)


def load_birdclef_data(
    split: str = "train",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Carga metadatos y agrega la ruta local correspondiente a cada grabación."""
    dataset_dir = resolve_data_dir(data_dir)
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


def extract_ogg_durations(
    data_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
    max_workers: int = 16,
) -> pd.DataFrame:
    """Extrae o carga desde caché la duración e integridad de todos los audios de `train.csv`."""
    out_path = Path(cache_path) if cache_path is not None else (PROCESSED_DIR / "train_durations.csv")
    if out_path.is_file():
        return pd.read_csv(out_path)

    dataset_dir = resolve_data_dir(data_dir)
    meta = load_metadata("train", data_dir=dataset_dir)
    audio_dir = dataset_dir / "train_audio"
    paths = [audio_dir / f for f in meta["filename"]]
    df_dur = batch_ogg_header_info(paths, max_workers=max_workers)
    df_dur.insert(0, "filename", meta["filename"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dur.to_csv(out_path, index=False)
    return df_dur
