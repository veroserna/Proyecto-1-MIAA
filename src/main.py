"""Punto de entrada principal del proyecto MIAA (BirdCLEF+ 2025)."""

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
from src.utils.helpers import print_header


def main():
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


if __name__ == "__main__":
    main()
