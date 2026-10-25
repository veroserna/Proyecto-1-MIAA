"""Hito 1 – EDA del metadata de BirdCLEF 2025 y selección de especies del Magdalena Medio.

Genera (todo reproducible, semilla fija):
  reports/hito1/figures/*.png         figuras del informe
  reports/hito1/tables/*.csv          tablas de soporte
  reports/hito1/stats.json            cifras globales citadas en el informe
  data_derived/subset_species.csv     especies definitivas + justificación
  data_derived/subset_recordings.csv  grabaciones incluidas + partición (test / fold 0-4)
  data_derived/split_all_recordings.csv  partición de TODO train.csv (para evaluación open-set)

Requisitos: pandas, numpy, matplotlib, scikit-learn.
Prerrequisito: python eda/ogg_durations.py  (data_derived/train_durations.csv)

Uso:
    python eda/hito1_eda.py
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW = (ROOT / "src" / "data" / "raw") if (ROOT / "src" / "data" / "raw" / "train.csv").is_file() else ROOT
OUT = ROOT / "experiments" / "results" / "hito1"
FIG = OUT / "figures"
TAB = OUT / "tables"
DER = ROOT / "src" / "data" / "processed"
for d in (FIG, TAB, DER):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
EL_SILENCIO = (6.76, -74.21)  # recording_location.txt (Yondó, Antioquia)
REGION_KM = 300  # radio "regional": Magdalena Medio + vertientes andinas adyacentes
COLOMBIA_BBOX = dict(lat=(-4.3, 12.5), lon=(-79.1, -66.8))

# ----------------------------------------------------------------------------
# Criterios de elegibilidad (cuantitativos, por especie)
# ----------------------------------------------------------------------------
ELIG = dict(
    class_name="Aves",   # E1: no-aves tienen mediana de 5-6 grabaciones -> no particionables
    min_recordings=120,  # E2: con 15 % de test quedan >= ~18 grabaciones por especie
    min_rating4=60,      # E3: >= 60 grabaciones XC con rating >= 4 (calidad alta)
    min_authors=40,      # E4: permite partición agrupada por autor sin fuga
    min_regional=10,     # E5: >= 10 grabaciones a <= 300 km de El Silencio
)

# ----------------------------------------------------------------------------
# Subconjunto definitivo (selección curada sobre el pool elegible)
# Reglas: R1 <= 2 especies por familia; R2 cubrir gremios ecoacústicos;
# R3 >= 1 especie emblemática/endémica regional; R4 >= 3 pares confundibles;
# R5 a igualdad de gremio, preferir mayor evidencia regional (n_regional).
# ----------------------------------------------------------------------------
SUBSET = [
    # code, family, order, activity, habitat/guild, rationale
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
CONFUSABLE_PAIRS = [("grekis", "rumfly1"), ("bugtan", "paltan1"), ("colcha1", "gycwor1"), ("trsowl", "compau")]

# Criterios de inclusión a nivel de grabación
MIN_DURATION_S = 1.0       # descartar clips < 1 s (no completan una ventana útil)
MIN_XC_RATING = 2.0        # XC con rating 0.5-1.5 = muy ruidoso; 0 = sin calificar (se conserva)

CLASS_COLORS = {"Aves": "#1f77b4", "Amphibia": "#2ca02c", "Insecta": "#ff7f0e", "Mammalia": "#9467bd"}


# ----------------------------------------------------------------------------
def parse_list(s: str) -> list[str]:
    return [x.strip() for x in ast.literal_eval(s) if x and x.strip()]


def haversine_km(lat, lon, lat0=EL_SILENCIO[0], lon0=EL_SILENCIO[1]):
    la, lo, la0, lo0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la) * np.cos(la0) * np.sin((lo - lo0) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW / "train.csv")
    dur = pd.read_csv(DER / "train_durations.csv")
    tax = pd.read_csv(RAW / "taxonomy.csv")
    df = df.merge(dur[["filename", "duration_s"]], on="filename", validate="1:1")
    df = df.merge(tax[["primary_label", "class_name"]], on="primary_label", validate="m:1")
    df["secondary"] = df.secondary_labels.apply(parse_list)
    df["types"] = df["type"].apply(lambda s: [t.lower() for t in parse_list(s)])
    df["n_secondary"] = df.secondary.str.len()
    df["dist_km"] = haversine_km(df.latitude, df.longitude)
    df["regional"] = df.dist_km <= REGION_KM
    df["colombia"] = df.latitude.between(*COLOMBIA_BBOX["lat"]) & df.longitude.between(*COLOMBIA_BBOX["lon"])
    df["rated"] = df.rating > 0
    return df, tax


def species_table(df: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("primary_label").agg(
        n_recordings=("filename", "size"),
        hours=("duration_s", lambda s: s.sum() / 3600),
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
        pct_with_secondary=("n_secondary", lambda s: (s > 0).mean() * 100),
    )
    g = tax.set_index("primary_label")[["scientific_name", "common_name", "class_name"]].join(g)
    g["eligible"] = (
        (g.class_name == ELIG["class_name"])
        & (g.n_recordings >= ELIG["min_recordings"])
        & (g.n_rating4 >= ELIG["min_rating4"])
        & (g.n_authors >= ELIG["min_authors"])
        & (g.n_regional >= ELIG["min_regional"])
    )
    g["selected"] = g.index.isin([s[0] for s in SUBSET])
    return g.sort_values("n_recordings", ascending=False)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def fig_species_distribution(sp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    colors = sp.class_name.map(CLASS_COLORS)
    ax.bar(range(len(sp)), sp.n_recordings, color=colors, width=1.0)
    sel = np.where(sp.selected.values)[0]
    ax.scatter(sel, sp.n_recordings.values[sel] * 1.25, marker="v", color="crimson", s=18, zorder=3, label="seleccionada")
    ax.set_yscale("log")
    ax.set_xlabel("Especies (ordenadas por n.º de grabaciones)")
    ax.set_ylabel("Grabaciones (log)")
    ax.set_title("Distribución de grabaciones por especie (206 clases) – cola larga")
    ax.axhline(ELIG["min_recordings"], ls="--", c="gray", lw=1)
    ax.text(len(sp) - 1, ELIG["min_recordings"] * 1.1, f"umbral E2 = {ELIG['min_recordings']}", ha="right", fontsize=8, color="gray")
    handles = [Patch(color=col, label=c) for c, col in CLASS_COLORS.items()]
    handles.append(Line2D([], [], marker="v", color="crimson", ls="", label="seleccionada"))
    ax.legend(handles=handles, ncol=5, fontsize=8)
    ax.set_xlim(-1, len(sp))
    savefig("01_recordings_per_species.png")


def fig_class_summary(df: pd.DataFrame) -> None:
    c = df.groupby("class_name").agg(species=("primary_label", "nunique"), recs=("filename", "size"),
                                      hours=("duration_s", lambda s: s.sum() / 3600))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col, title in zip(axes, ["species", "recs", "hours"], ["Especies", "Grabaciones", "Horas de audio"]):
        ax.bar(c.index, c[col], color=[CLASS_COLORS[i] for i in c.index])
        ax.set_title(title)
        for i, v in enumerate(c[col]):
            ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    savefig("02_class_summary.png")


def fig_durations(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.logspace(np.log10(0.5), np.log10(df.duration_s.max()), 60)
    for coll, sub in df.groupby("collection"):
        axes[0].hist(sub.duration_s, bins=bins, alpha=0.6, label=f"{coll} (n={len(sub):,})")
    axes[0].set_xscale("log")
    for x, lab in [(5, "5 s"), (60, "60 s")]:
        axes[0].axvline(x, ls="--", c="k", lw=0.8)
        axes[0].text(x, axes[0].get_ylim()[1] * 0.9, lab, fontsize=8)
    axes[0].set_xlabel("Duración (s, log)")
    axes[0].set_ylabel("Grabaciones")
    axes[0].set_title("Duración de grabaciones por colección")
    axes[0].legend(fontsize=8)
    order = df.groupby("class_name").duration_s.median().sort_values().index
    axes[1].boxplot([df.loc[df.class_name == c, "duration_s"] for c in order], showfliers=False)
    axes[1].set_xticks(range(1, len(order) + 1), order)
    axes[1].set_ylabel("Duración (s)")
    axes[1].set_title("Duración por clase taxonómica (sin outliers)")
    savefig("03_durations.png")


def fig_rating(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    ct = pd.crosstab(df.rating, df.collection)
    ct.plot.bar(stacked=True, ax=axes[0], color={"XC": "#1f77b4", "iNat": "#ff7f0e", "CSA": "#2ca02c"})
    axes[0].set_title("Rating por colección (0 = sin calificar)")
    axes[0].set_xlabel("rating")
    types = pd.Series([t for ts in df.types for t in ts]).value_counts().head(15)[::-1]
    axes[1].barh(types.index, types.values, color="#555")
    axes[1].set_title("Top-15 valores de 'type' (vocalización)")
    savefig("04_rating_and_types.png")


def fig_map(df: pd.DataFrame, sub: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    d = df.dropna(subset=["latitude"])
    axes[0].scatter(d.longitude, d.latitude, s=1, c="lightgray")
    s = sub.dropna(subset=["latitude"])
    axes[0].scatter(s.longitude, s.latitude, s=2, c="tab:blue")
    axes[0].scatter(*EL_SILENCIO[::-1], marker="*", s=200, c="red")
    axes[0].set_title("Ubicación de grabaciones (global)")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="", color="lightgray", label="todas (206 spp.)"),
        Line2D([], [], marker="o", ls="", color="tab:blue", label="subconjunto (16 spp.)"),
        Line2D([], [], marker="*", ls="", color="red", markersize=12, label="El Silencio"),
    ], fontsize=8, loc="lower left")
    ax = axes[1]
    ax.scatter(d.longitude, d.latitude, s=3, c="lightgray")
    ax.scatter(s.longitude, s.latitude, s=6, c="tab:blue")
    ax.scatter(*EL_SILENCIO[::-1], marker="*", s=250, c="red")
    t = np.linspace(0, 2 * np.pi, 200)
    dlat = REGION_KM / 111.0
    dlon = REGION_KM / (111.0 * np.cos(np.radians(EL_SILENCIO[0])))
    ax.plot(EL_SILENCIO[1] + dlon * np.cos(t), EL_SILENCIO[0] + dlat * np.sin(t), "r--", lw=1, label=f"{REGION_KM} km")
    ax.set_xlim(-80, -66)
    ax.set_ylim(-5, 13)
    ax.set_title("Zoom Colombia – radio regional")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    savefig("05_map.png")


def fig_distance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    d = df.dist_km.dropna()
    ax.hist(d, bins=np.logspace(0, np.log10(d.max()), 60), color="#777")
    ax.set_xscale("log")
    ax.axvline(REGION_KM, c="r", ls="--")
    ax.text(REGION_KM * 1.1, ax.get_ylim()[1] * 0.9, f"{REGION_KM} km: {(d <= REGION_KM).mean():.1%}", color="r", fontsize=8)
    ax.set_xlabel("Distancia a El Silencio (km, log)")
    ax.set_ylabel("Grabaciones")
    ax.set_title("Desplazamiento de dominio geográfico")
    savefig("06_distance_to_site.png")


def fig_subset(sp_sel: pd.DataFrame, rec: pd.DataFrame) -> None:
    order = sp_sel.sort_values("n_incl").index
    labels = [f"{c} – {sp_sel.loc[c, 'common_name']}" for c in order]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    axes[0].barh(labels, sp_sel.loc[order, "n_incl"], color="tab:blue")
    axes[0].set_title("Grabaciones incluidas")
    for i, v in enumerate(sp_sel.loc[order, "n_incl"]):
        axes[0].text(v, i, f" {v}", va="center", fontsize=7)
    axes[1].barh(labels, sp_sel.loc[order, "hours_incl"], color="tab:green")
    axes[1].set_title("Horas de audio")
    xc = sp_sel.loc[order, ["n_regional_incl", "n_incl"]]
    axes[2].barh(labels, xc.n_regional_incl / xc.n_incl * 100, color="tab:red")
    axes[2].set_title(f"% grabaciones a <= {REGION_KM} km")
    savefig("07_subset_distribution.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ct = pd.crosstab(rec.primary_label, rec.split).loc[order[::-1]]
    ct = ct[["train"] + [c for c in ct.columns if c != "train"]] if "train" in ct else ct
    ct.plot.bar(stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Partición por especie (folds 0-4 = desarrollo/CV, test = hold-out)")
    ax.set_ylabel("Grabaciones")
    savefig("08_split_per_species.png")


def fig_cooccurrence(rec: pd.DataFrame, codes: list[str]) -> None:
    m = pd.DataFrame(0, index=codes, columns=codes)
    for p, secs in zip(rec.primary_label, rec.secondary):
        for s in secs:
            if s in codes:
                m.loc[p, s] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(len(codes)), codes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(codes)), codes, fontsize=8)
    for i in range(len(codes)):
        for j in range(len(codes)):
            if m.values[i, j]:
                ax.text(j, i, m.values[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("etiqueta secundaria")
    ax.set_ylabel("etiqueta primaria")
    ax.set_title("Co-ocurrencia primaria→secundaria dentro del subconjunto")
    plt.colorbar(im, ax=ax, shrink=0.7)
    savefig("09_subset_cooccurrence.png")
    m.to_csv(TAB / "subset_cooccurrence.csv")


# ----------------------------------------------------------------------------
# Partición
# ----------------------------------------------------------------------------
def make_split(df: pd.DataFrame, codes: set[str]) -> pd.Series:
    """Partición agrupada por autor sobre TODO train.csv.

    - Grupo = autor (evita que el mismo recordista/equipo/sitio y los audios iNat
      duplicados con distinta etiqueta caigan a ambos lados).
    - Estratificación = etiqueta primaria si pertenece al subconjunto, si no '__other__'.
    - 1/7 (~14 %) hold-out de test; el resto en 5 folds de CV.
    """
    y = df.primary_label.where(df.primary_label.isin(codes), "__other__").values
    groups = df.author.fillna("unknown").values
    split = pd.Series("", index=df.index, dtype=object)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(df, y, groups))
    split.iloc[test_idx] = "test"
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(inner.split(dev_idx, y[dev_idx], groups[dev_idx])):
        split.iloc[dev_idx[va]] = f"fold{k}"
    return split


# ----------------------------------------------------------------------------
def main() -> None:
    df, tax = load()
    sp = species_table(df, tax)
    codes = [s[0] for s in SUBSET]
    assert set(codes) <= set(sp.index[sp.eligible]), "Hay especies seleccionadas no elegibles"
    assert 10 <= len(codes) <= 20

    # ---------------- estadísticas globales ----------------
    dup_url = df[df.url.duplicated(keep=False)]
    stats = {
        "n_recordings": len(df),
        "n_species": int(df.primary_label.nunique()),
        "species_per_class": df.groupby("class_name").primary_label.nunique().to_dict(),
        "recordings_per_class": df.class_name.value_counts().to_dict(),
        "hours_total": round(df.duration_s.sum() / 3600, 1),
        "duration_s": df.duration_s.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict(),
        "n_shorter_5s": int((df.duration_s < 5).sum()),
        "n_shorter_1s": int((df.duration_s < 1).sum()),
        "n_longer_300s": int((df.duration_s > 300).sum()),
        "collections": df.collection.value_counts().to_dict(),
        "rating_unrated_pct": round((df.rating == 0).mean() * 100, 1),
        "rating_xc_ge4_pct": round((df.loc[df.collection == "XC", "rating"] >= 4).mean() * 100, 1),
        "missing_coords": int(df.latitude.isna().sum()),
        "pct_with_secondary": round((df.n_secondary > 0).mean() * 100, 1),
        "pct_with_type": round((df.types.str.len() > 0).mean() * 100, 1),
        "n_authors": int(df.author.nunique()),
        "n_duplicate_url_rows": len(dup_url),
        "pct_within_region": round(df.regional.mean() * 100, 2),
        "pct_in_colombia": round(df.colombia.mean() * 100, 2),
        "recs_per_species": sp.n_recordings.describe().round(1).to_dict(),
        "imbalance_ratio_all": round(sp.n_recordings.max() / sp.n_recordings.min(), 1),
        "gini_recs_all": None,
        "species_lt_20_recs": int((sp.n_recordings < 20).sum()),
        "species_lt_10_recs": int((sp.n_recordings < 10).sum()),
        "n_eligible": int(sp.eligible.sum()),
        "licenses": df.license.value_counts().to_dict(),
    }

    def gini(x):
        x = np.sort(np.asarray(x, float))
        n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

    stats["gini_recs_all"] = round(gini(sp.n_recordings), 3)
    secs = pd.Series([s for l in df.secondary for s in l])
    stats["secondary_in_taxonomy_pct"] = round(secs.isin(tax.primary_label).mean() * 100, 1)
    stats["n_secondary_labels_total"] = int(len(secs))

    # ---------------- subconjunto: grabaciones ----------------
    df["split"] = make_split(df, set(codes))
    df[["filename", "primary_label", "author", "split"]].to_csv(DER / "split_all_recordings.csv", index=False)

    sub = df[df.primary_label.isin(codes)].copy()
    n0 = len(sub)
    excl = {
        "duration_lt_1s": int((sub.duration_s < MIN_DURATION_S).sum()),
        "xc_rating_lt_2": int(((sub.collection == "XC") & sub.rating.between(0.1, MIN_XC_RATING - 0.01)).sum()),
    }
    keep = (sub.duration_s >= MIN_DURATION_S) & ~((sub.collection == "XC") & sub.rating.between(0.1, MIN_XC_RATING - 0.01))
    sub = sub[keep]
    # Fusión de audios duplicados (misma URL) -> multietiqueta
    sub["labels"] = sub.apply(lambda r: sorted({r.primary_label} | {s for s in r.secondary if s in codes}), axis=1)
    url_labels = sub.groupby("url").primary_label.apply(lambda s: set(s))
    dup_in_subset = url_labels[url_labels.str.len() > 1]
    for url, labs in dup_in_subset.items():
        idx = sub.index[sub.url == url]
        merged = sorted(set().union(*[set(sub.at[i, "labels"]) for i in idx]) | labs)
        for i in idx:
            sub.at[i, "labels"] = merged
    drop_dup = sub.url.duplicated(keep="first")
    excl["duplicate_audio_merged"] = int(drop_dup.sum())
    sub = sub[~drop_dup]
    excl["kept"] = len(sub)
    excl["initial"] = n0
    sub["n_labels"] = sub.labels.str.len()

    out_cols = ["filename", "primary_label", "labels", "secondary", "collection", "rating", "duration_s",
                "author", "latitude", "longitude", "dist_km", "regional", "split", "url", "license"]
    sub_out = sub[out_cols].copy()
    sub_out["labels"] = sub_out.labels.apply(" ".join)
    sub_out["secondary"] = sub_out.secondary.apply(" ".join)
    sub_out.to_csv(DER / "subset_recordings.csv", index=False)

    # ---------------- subconjunto: especies ----------------
    meta = pd.DataFrame(SUBSET, columns=["primary_label", "family", "order", "activity", "habitat", "rationale"]).set_index("primary_label")
    inc = sub.groupby("primary_label").agg(n_incl=("filename", "size"), hours_incl=("duration_s", lambda s: s.sum() / 3600),
                                           n_regional_incl=("regional", "sum"), n_authors_incl=("author", "nunique"))
    sp_sel = sp.loc[codes].join(meta).join(inc)
    split_ct = pd.crosstab(sub.primary_label, sub.split)
    sp_sel = sp_sel.join(split_ct)
    sp_sel["weight_inv_freq"] = (sp_sel.n_incl.sum() / (len(sp_sel) * sp_sel.n_incl)).round(3)
    sp_sel.to_csv(DER / "subset_species.csv", encoding="utf-8")

    stats["subset"] = {
        "n_species": len(codes),
        "families": int(meta.family.nunique()),
        "orders": int(meta.order.nunique()),
        "recordings_kept": int(len(sub)),
        "hours": round(sub.duration_s.sum() / 3600, 1),
        "exclusions": excl,
        "duplicate_urls_merged": {u: sorted(v) for u, v in dup_in_subset.items()},
        "imbalance_ratio": round(sp_sel.n_incl.max() / sp_sel.n_incl.min(), 2),
        "gini": round(gini(sp_sel.n_incl), 3),
        "min_test_per_species": int(split_ct["test"].min()),
        "split_counts": sub.split.value_counts().sort_index().to_dict(),
        "split_pct": (sub.split.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
        "pct_multilabel": round((sub.n_labels > 1).mean() * 100, 2),
        "pct_regional": round(sub.regional.mean() * 100, 1),
        "author_overlap_test_dev": int(len(set(sub.author[sub.split == "test"]) & set(sub.author[sub.split != "test"]))),
        "open_set_test_negatives": int(((df.split == "test") & ~df.primary_label.isin(codes)
                                        & ~df.secondary.apply(lambda l: any(s in codes for s in l))).sum()),
    }
    # verificación de no-fuga entre folds
    for k in range(5):
        a = set(df.author[df.split == f"fold{k}"])
        b = set(df.author[(df.split != f"fold{k}")])
        assert not (a & b), f"fuga de autor en fold{k}"

    # ---------------- tablas ----------------
    sp.round(3).to_csv(TAB / "species_summary_all.csv", encoding="utf-8")
    sp[sp.eligible].round(3).to_csv(TAB / "eligible_pool.csv", encoding="utf-8")
    funnel = pd.Series({
        "Total especies": len(sp),
        "E1 Aves": int((sp.class_name == "Aves").sum()),
        f"E2 + >= {ELIG['min_recordings']} grabaciones": int(((sp.class_name == "Aves") & (sp.n_recordings >= ELIG["min_recordings"])).sum()),
        f"E3 + >= {ELIG['min_rating4']} con rating>=4": int(((sp.class_name == "Aves") & (sp.n_recordings >= ELIG["min_recordings"]) & (sp.n_rating4 >= ELIG["min_rating4"])).sum()),
        f"E4 + >= {ELIG['min_authors']} autores": int(((sp.class_name == "Aves") & (sp.n_recordings >= ELIG["min_recordings"]) & (sp.n_rating4 >= ELIG["min_rating4"]) & (sp.n_authors >= ELIG["min_authors"])).sum()),
        f"E5 + >= {ELIG['min_regional']} regionales": int(sp.eligible.sum()),
        "Seleccionadas (R1-R5)": len(codes),
    })
    funnel.to_csv(TAB / "selection_funnel.csv", header=["n_species"])
    stats["funnel"] = funnel.to_dict()
    pd.Series([t for ts in df.types for t in ts]).value_counts().head(30).to_csv(TAB / "top_types.csv", header=["n"])
    df.groupby("collection").agg(n=("filename", "size"), hours=("duration_s", lambda s: s.sum() / 3600),
                                 median_dur=("duration_s", "median"), pct_rated=("rated", "mean"),
                                 pct_secondary=("n_secondary", lambda s: (s > 0).mean()),
                                 missing_coords=("latitude", lambda s: s.isna().sum())).round(3).to_csv(TAB / "collection_summary.csv")

    # ---------------- figuras ----------------
    fig_species_distribution(sp)
    fig_class_summary(df)
    fig_durations(df)
    fig_rating(df)
    fig_map(df, sub)
    fig_distance(df)
    fig_subset(sp_sel, sub)
    fig_cooccurrence(sub, codes)

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False, default=float))
    cols = ["common_name", "family", "n_recordings", "n_incl", "hours_incl", "n_rating4", "n_authors", "n_regional",
            "max_author_share", "test", "fold0", "fold1", "fold2", "fold3", "fold4", "weight_inv_freq"]
    print(sp_sel[cols].round(2).to_string())


if __name__ == "__main__":
    main()
