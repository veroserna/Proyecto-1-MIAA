"""Punto de entrada para consultar el dataset BirdCLEF descargado localmente."""
#Recuerda descargar el dataset BirdCLEF+ 2025 desde https://www.kaggle.com/competitions/birdclef-2025/data y colocarlo en la carpeta src/data/raw.

import argparse

from src.data.audio_loader import load_birdclef_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carga metadatos y rutas de audio locales de BirdCLEF+ 2025."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split a cargar (por defecto: train; busca <split>.csv y <split>_audio/).",
    )
    parser.add_argument(
        "--data-dir",
        help="Carpeta raíz del dataset. Por defecto: src/data/raw.",
    )
    args = parser.parse_args()

    dataset = load_birdclef_data(split=args.split, data_dir=args.data_dir)
    print(f"Registros cargados: {len(dataset):,}")
    print(f"Especies: {dataset['primary_label'].nunique():,}")
    print(f"Audio de ejemplo: {dataset['audio_path'].iloc[0]}")


if __name__ == "__main__":
    main()
