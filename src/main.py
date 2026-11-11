"""Punto de entrada principal del proyecto MIAA (BirdCLEF+ 2025)."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocess import (
    SUBSET_DEFINITION,
    build_species_summary,
    build_subset_datasets,
    create_grouped_splits,
    export_hito1_artifacts,
    load_enriched_metadata,
)
from src.training.hito4_clustering import run_hito4_clustering
from src.training.hito5_baseline_svm import run_hito5_baseline_svm
from src.training.preprocessing.hito2_audio_conditioning import (
    run_hito2_conditioning_pipeline,
)
from src.utils.helpers import print_header


def run_hito1() -> None:
    print_header("Proyecto 1 MIAA — BirdCLEF+ 2025 (Hito 1: EDA y Subconjunto Magdalena Medio)")
    df, tax = load_enriched_metadata()
    sp_summary = build_species_summary(df, tax)
    df["split"] = create_grouped_splits(df, {s[0] for s in SUBSET_DEFINITION})
    sub_rec, sp_sel, exclusions = build_subset_datasets(df, sp_summary)
    export_hito1_artifacts(df, sp_summary, sub_rec, sp_sel, exclusions)
    print(f"Total grabaciones cargadas : {len(df):,} ({df['primary_label'].nunique()} especies)")
    print(f"Especies elegibles (E1-E5) : {int(sp_summary['eligible'].sum())}")
    print(f"Subconjunto seleccionado   : {len(sp_sel)} especies | {len(sub_rec):,} grabaciones | {sub_rec['duration_s'].sum()/3600:.1f} h")
    print(f"Particiones (split)        : {sub_rec['split'].value_counts().sort_index().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline principal BirdCLEF+ 2025 (MIAA Universidad Icesi)")
    parser.add_argument(
        "--stage",
        choices=["hito1", "hito2", "hito3", "hito4", "hito5", "all"],
        default="all",
        help="Etapa del pipeline a ejecutar (por defecto: all).",
    )
    parser.add_argument(
        "--no-wav",
        action="store_true",
        help="En Hito 2, calcula metadatos y métricas de clips sin reescribir archivos .wav.",
    )
    # Argumentos opcionales para Hito 3. No alteran el comportamiento de Hitos 1/2.
    parser.add_argument("--verify-only", action="store_true", help="Hito 3: auditar catálogo y WAV sin extraer características")
    parser.add_argument("--limit", type=int, default=None, help="Hito 3: muestra de prueba (mínimo 9; recomendado 60)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Hito 3: formato de features")
    parser.add_argument("--overwrite", action="store_true", help="Hito 3: reemplazar resultados anteriores del Hito 3")

    args = parser.parse_args()

    if args.stage in ("hito1", "all"):
        run_hito1()
    if args.stage in ("hito2", "all"):
        run_hito2_conditioning_pipeline(save_wav=not args.no_wav)

    if args.stage == "hito3":
        # Importación diferida: Hitos 1 y 2 no requieren instalar librosa/pyarrow.
        from src.training.preprocessing.hito3_feature_pipeline import run_hito3
        run_hito3(
            limit=args.limit,
            fmt=args.format,
            verify_only=args.verify_only,
            overwrite=args.overwrite,
        )
    if args.stage == "hito4":
        run_hito4_clustering()
    if args.stage == "hito5":
        run_hito5_baseline_svm()


if __name__ == "__main__":
    main()
