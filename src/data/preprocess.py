"""Preprocesamiento de metadatos, selección del subconjunto del Magdalena Medio y particiones (Hito 1)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.audio_loader import (
    PROCESSED_DIR,
    extract_ogg_durations,
    load_birdclef_data,
    load_taxonomy,
)
from src.utils.helpers import (
    COLOMBIA_LAT_BOUNDS,
    COLOMBIA_LON_BOUNDS,
    EL_SILENCIO_LAT,
    EL_SILENCIO_LON,
    REGION_RADIUS_KM,
    gini_coefficient,
    haversine_km,
    parse_list_field,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_HITO1_DIR = PROJECT_ROOT / "experiments" / "results" / "hito1"

SEED = 42
EL_SILENCIO_COORDS = (EL_SILENCIO_LAT, EL_SILENCIO_LON)
COLOMBIA_BBOX = {"lat": COLOMBIA_LAT_BOUNDS, "lon": COLOMBIA_LON_BOUNDS}

ELIGIBILITY_CRITERIA = {
    "class_name": "Aves",
    "min_recordings": 120,
    "min_rating4": 60,
    "min_authors": 40,
    "min_regional": 10,
}

MIN_DURATION_S = 1.0
MIN_XC_RATING = 2.0

SUBSET_DEFINITION = [
    ("colcha1", "Cracidae", "Galliformes", "diurna", "borde de bosque / bosque secundario",
     "Endémica de Colombia (valles del Magdalena y Cauca); especie emblemática regional con la mayor proporción de grabaciones locales (128/145 a <=300 km). Coros ruidosos al amanecer."),
    ("littin1", "Tinamidae", "Tinamiformes", "crepuscular", "suelo de bosque",
     "Silbidos tonales ascendentes del sotobosque; activa al alba/ocaso, representa el estrato terrestre del bosque."),
    ("wbwwre1", "Troglodytidae", "Passeriformes", "diurna", "sotobosque de bosque húmedo",
     "Canto frecuente en bosque húmedo de tierras bajas; 2.ª especie con más grabaciones regionales (74). Indicador de bosque."),
    ("chbant1", "Thamnophilidae", "Passeriformes", "diurna", "sotobosque de bosque húmedo",
     "Hormiguero de interior de bosque, sensible a fragmentación; aporta un gremio insectívoro de sotobosque."),
    ("linwoo1", "Picidae", "Piciformes", "diurna", "borde de bosque",
     "Incluye tamborileo (sonido no vocal, 'drumming') además de llamadas: diversidad de tipo de señal."),
    ("orcpar", "Psittacidae", "Psittaciformes", "diurna", "dosel / áreas abiertas",
     "Llamadas de bandada de banda ancha y alta tasa de repetición; gremio frugívoro/granívoro de dosel."),
    ("bugtan", "Thraupidae", "Passeriformes", "diurna", "dosel / borde",
     "Tangara común de borde. Par confundible con Palm Tanager (cantos chirriantes muy similares)."),
    ("paltan1", "Thraupidae", "Passeriformes", "diurna", "dosel / palmas / borde",
     "Par confundible con Blue-gray Tanager; permite medir discriminación fina intra-familia."),
    ("grekis", "Tyrannidae", "Passeriformes", "diurna", "áreas abiertas / borde / riberas",
     "Especie más grabada del dataset (990); muy común en el Magdalena Medio. Par confundible con Rusty-margined Flycatcher."),
    ("rumfly1", "Tyrannidae", "Passeriformes", "diurna", "borde cerca de agua",
     "Plumaje y vocalizaciones semejantes a Great Kiskadee ('kiskadee-like'); alta evidencia regional (48)."),
    ("roahaw", "Accipitridae", "Accipitriformes", "diurna", "borde / áreas abiertas",
     "Rapaz más común; mayor número de grabaciones regionales (104). Llamadas largas descendentes."),
    ("laufal1", "Falconidae", "Falconiformes", "crepuscular/diurna", "borde de bosque",
     "Rapaz con duetos largos y estereotipados, audibles a gran distancia; 7 h de audio."),
    ("soulap1", "Charadriidae", "Charadriiformes", "diurna/nocturna", "potreros / orillas de ciénaga",
     "Representa el paisaje ganadero y humedales (ciénagas) del Magdalena Medio; llama también de noche."),
    ("gycwor1", "Rallidae", "Gruiformes", "crepuscular", "humedales / bosque ribereño",
     "Coros fuertes al alba/ocaso en humedales; par confundible con los coros de Colombian Chachalaca."),
    ("trsowl", "Strigidae", "Strigiformes", "nocturna", "borde / áreas abiertas",
     "Gremio nocturno (los paisajes sonoros cubren 24 h). Trino corto; par confundible con Common Pauraque."),
    ("compau", "Caprimulgidae", "Caprimulgiformes", "nocturna", "áreas abiertas / borde",
     "Chotacabras nocturno muy común; silbido característico. Completa el gremio nocturno."),
]


def load_data(file_path: str):
    """Carga un archivo de datos desde la ruta indicada (compatibilidad base del repo)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    return path


def clean_data(data):
    """Placeholder para limpieza de datos (compatibilidad base del repo)."""
    return data


def load_enriched_metadata(
    data_dir: str | Path | None = None,
    durations_cache: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga `train.csv` + `taxonomy.csv` + duraciones OGG y calcula variables derivadas."""
    df = load_birdclef_data("train", data_dir=data_dir)
    tax = load_taxonomy(data_dir=data_dir)
    dur = extract_ogg_durations(data_dir=data_dir, cache_path=durations_cache)

    df = df.merge(
        dur[["filename", "duration_s", "sample_rate", "channels", "size_bytes"]],
        on="filename",
        validate="1:1",
    )
    df = df.merge(tax[["primary_label", "class_name"]], on="primary_label", validate="m:1")
    df["secondary"] = df["secondary_labels"].apply(parse_list_field)
    df["types"] = df["type"].apply(lambda s: [t.lower() for t in parse_list_field(s)])
    df["n_secondary"] = df["secondary"].str.len()
    df["dist_km"] = haversine_km(df["latitude"], df["longitude"])
    df["regional"] = df["dist_km"] <= REGION_RADIUS_KM
    df["colombia"] = df["latitude"].between(*COLOMBIA_BBOX["lat"]) & df["longitude"].between(*COLOMBIA_BBOX["lon"])
    df["rated"] = df["rating"] > 0
    return df, tax


def build_species_summary(df: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla resumen de las 206 especies y evalúa los criterios E1–E5."""
    g = df.groupby("primary_label").agg(
        n_recordings=("filename", "size"),
        hours=("duration_s", lambda s: s.sum() / 3600.0),
        median_dur_s=("duration_s", "median"),
        n_xc=("collection", lambda s: (s == "XC").sum()),
        n_inat=("collection", lambda s: (s == "iNat").sum()),
        n_csa=("collection", lambda s: (s == "CSA").sum()),
        n_rating4=("rating", lambda s: (s >= 4).sum()),
        mean_rating_rated=("rating", lambda s: s[s > 0].mean()),
        n_authors=("author", "nunique"),
        max_author_share=("author", lambda s: s.value_counts(normalize=True).iloc[0]),
        n_regional=("regional", "sum"),
        n_colombia=("colombia", "sum"),
        median_dist_km=("dist_km", "median"),
        pct_with_secondary=("n_secondary", lambda s: (s > 0).mean() * 100.0),
    )
    g = tax.set_index("primary_label")[["scientific_name", "common_name", "class_name"]].join(g)
    g["eligible"] = (
        (g["class_name"] == ELIGIBILITY_CRITERIA["class_name"])
        & (g["n_recordings"] >= ELIGIBILITY_CRITERIA["min_recordings"])
        & (g["n_rating4"] >= ELIGIBILITY_CRITERIA["min_rating4"])
        & (g["n_authors"] >= ELIGIBILITY_CRITERIA["min_authors"])
        & (g["n_regional"] >= ELIGIBILITY_CRITERIA["min_regional"])
    )
    selected_codes = [s[0] for s in SUBSET_DEFINITION]
    g["selected"] = g.index.isin(selected_codes)
    return g.sort_values("n_recordings", ascending=False)


def create_grouped_splits(df: pd.DataFrame, subset_codes: set[str], seed: int = SEED) -> pd.Series:
    """Particiona todas las grabaciones en `test` (~14.3 %) y `fold0`..`fold4` agrupando por autor."""
    y = df["primary_label"].where(df["primary_label"].isin(subset_codes), "__other__").values
    groups = df["author"].fillna("unknown").values
    split = pd.Series("", index=df.index, dtype=object)

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(df, y, groups))
    split.iloc[test_idx] = "test"

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (_, val_pos) in enumerate(inner.split(dev_idx, y[dev_idx], groups[dev_idx])):
        split.iloc[dev_idx[val_pos]] = f"fold{k}"
    return split


def build_subset_datasets(
    df: pd.DataFrame,
    sp_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Filtra grabaciones del subconjunto (I1–I5), fusiona duplicados y genera tablas finales."""
    codes = [s[0] for s in SUBSET_DEFINITION]
    code_set = set(codes)

    if "split" not in df.columns:
        df = df.copy()
        df["split"] = create_grouped_splits(df, code_set)

    sub = df[df["primary_label"].isin(code_set)].copy()
    n_initial = len(sub)
    n_lt1s = int((sub["duration_s"] < MIN_DURATION_S).sum())
    n_low_xc = int(((sub["collection"] == "XC") & sub["rating"].between(0.1, MIN_XC_RATING - 0.01)).sum())

    keep = (sub["duration_s"] >= MIN_DURATION_S) & ~(
        (sub["collection"] == "XC") & sub["rating"].between(0.1, MIN_XC_RATING - 0.01)
    )
    sub = sub[keep].copy()

    sub["labels"] = sub.apply(
        lambda r: sorted({r["primary_label"]} | {s for s in r["secondary"] if s in code_set}),
        axis=1,
    )
    url_labels = sub.groupby("url")["primary_label"].apply(set)
    dup_in_subset = url_labels[url_labels.str.len() > 1]
    for url, labs in dup_in_subset.items():
        idx = sub.index[sub["url"] == url]
        merged = sorted(set().union(*[set(sub.at[i, "labels"]) for i in idx]) | labs)
        for i in idx:
            sub.at[i, "labels"] = merged

    drop_dup = sub["url"].duplicated(keep="first")
    n_merged = int(drop_dup.sum())
    sub = sub[~drop_dup].copy()
    sub["n_labels"] = sub["labels"].str.len()

    meta_def = pd.DataFrame(
        SUBSET_DEFINITION,
        columns=["primary_label", "family", "order", "activity", "habitat", "rationale"],
    ).set_index("primary_label")

    inc = sub.groupby("primary_label").agg(
        n_incl=("filename", "size"),
        hours_incl=("duration_s", lambda s: s.sum() / 3600.0),
        n_regional_incl=("regional", "sum"),
        n_authors_incl=("author", "nunique"),
    )
    split_ct = pd.crosstab(sub["primary_label"], sub["split"])
    sp_sel = sp_summary.loc[codes].join(meta_def).join(inc).join(split_ct)
    sp_sel["weight_inv_freq"] = (sp_sel["n_incl"].sum() / (len(sp_sel) * sp_sel["n_incl"])).round(3)

    exclusions = {
        "initial": n_initial,
        "duration_lt_1s": n_lt1s,
        "xc_rating_lt_2": n_low_xc,
        "duplicate_audio_merged": n_merged,
        "kept": len(sub),
    }
    return sub, sp_sel, exclusions


def export_hito1_artifacts(
    df: pd.DataFrame,
    sp_summary: pd.DataFrame,
    sub: pd.DataFrame,
    sp_sel: pd.DataFrame,
    exclusions: dict,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_HITO1_DIR,
) -> None:
    """Guarda los CSVs procesados en `src/data/processed/` y tablas/resúmenes en `experiments/results/hito1/`."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df[["filename", "primary_label", "author", "split"]].to_csv(
        processed_dir / "split_all_recordings.csv", index=False
    )

    out_cols = [
        "filename", "primary_label", "labels", "secondary", "collection", "rating",
        "duration_s", "author", "latitude", "longitude", "dist_km", "regional",
        "split", "url", "license",
    ]
    sub_out = sub[out_cols].copy()
    sub_out["labels"] = sub_out["labels"].apply(" ".join)
    sub_out["secondary"] = sub_out["secondary"].apply(" ".join)
    sub_out.to_csv(processed_dir / "subset_recordings.csv", index=False)
    sp_sel.to_csv(processed_dir / "subset_species.csv", encoding="utf-8")

    sp_summary.round(3).to_csv(tables_dir / "species_summary_all.csv", encoding="utf-8")
    sp_summary[sp_summary["eligible"]].round(3).to_csv(tables_dir / "eligible_pool.csv", encoding="utf-8")

    funnel = pd.Series({
        "Total especies": len(sp_summary),
        "E1 Aves": int((sp_summary["class_name"] == "Aves").sum()),
        f"E2 + >= {ELIGIBILITY_CRITERIA['min_recordings']} grabaciones": int(
            ((sp_summary["class_name"] == "Aves") & (sp_summary["n_recordings"] >= ELIGIBILITY_CRITERIA["min_recordings"])).sum()
        ),
        f"E3 + >= {ELIGIBILITY_CRITERIA['min_rating4']} con rating>=4": int(
            ((sp_summary["class_name"] == "Aves")
             & (sp_summary["n_recordings"] >= ELIGIBILITY_CRITERIA["min_recordings"])
             & (sp_summary["n_rating4"] >= ELIGIBILITY_CRITERIA["min_rating4"])).sum()
        ),
        f"E4 + >= {ELIGIBILITY_CRITERIA['min_authors']} autores": int(
            ((sp_summary["class_name"] == "Aves")
             & (sp_summary["n_recordings"] >= ELIGIBILITY_CRITERIA["min_recordings"])
             & (sp_summary["n_rating4"] >= ELIGIBILITY_CRITERIA["min_rating4"])
             & (sp_summary["n_authors"] >= ELIGIBILITY_CRITERIA["min_authors"])).sum()
        ),
        f"E5 + >= {ELIGIBILITY_CRITERIA['min_regional']} regionales": int(sp_summary["eligible"].sum()),
        "Seleccionadas (R1-R5)": len(sp_sel),
    })
    funnel.to_csv(tables_dir / "selection_funnel.csv", header=["n_species"])
    pd.Series([t for ts in df["types"] for t in ts]).value_counts().head(30).to_csv(
        tables_dir / "top_types.csv", header=["n"]
    )
    df.groupby("collection").agg(
        n=("filename", "size"),
        hours=("duration_s", lambda s: s.sum() / 3600.0),
        median_dur=("duration_s", "median"),
        pct_rated=("rated", "mean"),
        pct_secondary=("n_secondary", lambda s: (s > 0).mean()),
        missing_coords=("latitude", lambda s: s.isna().sum()),
    ).round(3).to_csv(tables_dir / "collection_summary.csv")
