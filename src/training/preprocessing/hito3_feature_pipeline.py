"""Pipeline ejecutable e independiente del Hito 3 de Proyecto-1-MIAA.

Comandos (desde la RAÍZ del repositorio):
    python -m src.training.preprocessing.hito3_feature_pipeline --verify-only --limit 60
    python -m src.training.preprocessing.hito3_feature_pipeline --limit 60
    python -m src.training.preprocessing.hito3_feature_pipeline

El orquestador principal lo ejecuta con --stage hito3 sin reejecutar Hitos 1/2. Salida oficial: src/data/features/features.csv.
Las salidas de muestra se guardan SOLO en experiments/results/hito3/preview/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_features import (  # noqa: E402
    DEFAULT_CONFIG, FeatureConfig, extract_features_from_wav, feature_column_names
)

PROCESSED_DIR = PROJECT_ROOT / "src" / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "src" / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "hito3"
EXPECTED_PARTITION = {
    "fold0": "train", "fold1": "train", "fold2": "train",
    "fold3": "train", "fold4": "val", "test": "test"
}
REQUIRED = (
    "clip_id", "clip_path", "recording_id", "filename", "primary_label",
    "labels", "author", "split", "partition", "sr", "n_samples"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def read_and_audit_manifest(path: Path) -> tuple[pd.DataFrame, dict]:
    """Audita IDs, autores, grabaciones, folds y esquema sin cambiar los datos."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el catálogo Hito 2: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing_cols = sorted(set(REQUIRED) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Faltan columnas en clips_metadata.csv: {missing_cols}")
    if df.empty:
        raise ValueError("El catálogo de clips está vacío")
    for col in ("clip_id", "clip_path", "recording_id", "filename", "primary_label",
                "labels", "author", "split", "partition"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Hay valores faltantes en columna obligatoria: {col}")
    if df["clip_id"].duplicated().any():
        raise ValueError("clip_id duplicado en el catálogo")
    if df["clip_path"].duplicated().any():
        raise ValueError("clip_path duplicado en el catálogo")
    if not df["split"].isin(EXPECTED_PARTITION).all():
        raise ValueError("Existen valores desconocidos en 'split'")
    expected_partitions = df["split"].map(EXPECTED_PARTITION)
    if not df["partition"].eq(expected_partitions).all():
        raise ValueError("split y partition son inconsistentes")
    if set(df["partition"].unique()) != {"train", "val", "test"}:
        raise ValueError("Deben existir las particiones train, val y test")
    if (df["sr"] != DEFAULT_CONFIG.sr).any() or (df["n_samples"] != DEFAULT_CONFIG.sr * DEFAULT_CONFIG.duration_s).any():
        raise ValueError("Los metadatos no indican 32 kHz y 160.000 muestras en todos los clips")

    # No basta con separar por clip: impedir grabación o autor en distintos conjuntos.
    for col in ("recording_id", "filename", "author"):
        grouped = df.groupby(col, dropna=False)["partition"].nunique()
        bad_count = int((grouped > 1).sum())
        if bad_count:
            raise ValueError(f"Fuga entre particiones: {bad_count} valores de {col} repetidos")
    return df, {
        "n_manifest": int(len(df)),
        "n_species": int(df["primary_label"].nunique()),
        "n_authors": int(df["author"].nunique()),
        "counts_by_partition": {k: int(v) for k, v in df["partition"].value_counts().to_dict().items()},
        "leakage_author_recording_filename": 0,
        "manifest_sha256": sha256_file(path),
    }


def sample_across_partitions(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Muestra determinista con train/val/test sin alterar el catálogo completo."""
    if limit < 9:
        raise ValueError("Para la prueba inicial usar --limit 9 o más (recomendado: 60)")
    if limit >= len(df):
        return df.copy().reset_index(drop=True)
    requested = {
        "train": max(4, round(limit * 0.70)),
        "val": max(1, round(limit * 0.17)),
    }
    requested["test"] = max(1, limit - requested["train"] - requested["val"])
    while sum(requested.values()) > limit:
        requested["train"] -= 1
    selected_indices: list[int] = []
    for ix, partition in enumerate(("train", "val", "test")):
        part = df[df["partition"] == partition]
        n = min(len(part), requested[partition])
        selected_indices.extend(part.sample(n=n, random_state=seed + ix).index.tolist())
    if len(selected_indices) < limit:
        remaining = df.loc[~df.index.isin(selected_indices)]
        selected_indices.extend(remaining.sample(n=limit-len(selected_indices), random_state=seed+10).index.tolist())
    return df.loc[selected_indices].sort_values(["partition", "clip_id"]).reset_index(drop=True)


def resolve_clip_path(processed_dir: Path, rel_path: str) -> Path:
    """El Hito 2 usa clips_5s/particion/especie/clip.wav (ruta relativa a processed)."""
    path_text = str(rel_path).replace("\\", "/")
    pure = PurePosixPath(path_text)
    if pure.is_absolute() or PureWindowsPath(path_text).drive or ".." in pure.parts:
        raise ValueError(f"Ruta no permitida en clip_path: {rel_path}")
    if pure.suffix.lower() != ".wav" or len(pure.parts) < 4 or pure.parts[0] != "clips_5s":
        raise ValueError(f"Ruta inesperada del Hito 2: {rel_path}")
    root = processed_dir.resolve()
    resolved = (root / Path(*pure.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Ruta WAV fuera de processed/: {rel_path}")
    return resolved


def verify_clip_files(df: pd.DataFrame, processed_dir: Path) -> None:
    missing: list[str] = []
    for rel in df["clip_path"]:
        path = resolve_clip_path(processed_dir, str(rel))
        if not path.is_file():
            if len(missing) < 8:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Hay WAV referenciados que NO están presentes. Ejemplos:\n  " +
            "\n  ".join(missing) +
            "\nObtén del equipo la carpeta clips_5s/ correspondiente a este catálogo; no reproceses Hito 2 sin acordarlo."
        )


def _pca_figures(pca_df: pd.DataFrame, ratios: np.ndarray, figs_dir: Path) -> None:
    """Grafica solamente TRAIN: test permanece reservado para la evaluación final."""
    train = pca_df[pca_df["partition"] == "train"]
    species = sorted(train["primary_label"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(species)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, specie in enumerate(species):
        points = train[train["primary_label"] == specie]
        ax.scatter(points["PC1"], points["PC2"], s=9, alpha=.52,
                   color=cmap(i), label=specie, rasterized=True)
    ax.set_title("Hito 3 · PCA exploratorio (solo entrenamiento)")
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}% varianza)")
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="Especie principal")
    fig.tight_layout()
    fig.savefig(figs_dir / "pca_2d_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    for i, specie in enumerate(species):
        points = train[train["primary_label"] == specie]
        ax.scatter(points["PC1"], points["PC2"], points["PC3"],
                   s=8, alpha=.4, color=cmap(i), label=specie, rasterized=True)
    ax.set_title("Hito 3 · PCA 3D (solo entrenamiento)")
    for j, axis in enumerate(("PC1", "PC2", "PC3")):
        getattr(ax, f"set_{'xyz'[j]}label")(f"{axis} ({ratios[j]*100:.1f}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1), fontsize=8)
    fig.savefig(figs_dir / "pca_3d_train.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_df(df: pd.DataFrame, target: Path, fmt: str) -> None:
    if fmt == "csv":
        df.to_csv(target, index=False, float_format="%.9g", encoding="utf-8")
    elif fmt == "parquet":
        try:
            df.to_parquet(target, index=False, engine="pyarrow")
        except ImportError as exc:
            raise RuntimeError("Para guardar Parquet instala: python -m pip install pyarrow") from exc
    else:
        raise ValueError(f"Formato desconocido: {fmt}")


def _package_versions() -> dict[str, str]:
    ans = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "librosa", "scikit-learn", "matplotlib", "joblib"):
        try:
            ans[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            ans[package] = "not installed"
    return ans


def run_hito3(
    *, manifest_path: Path = PROCESSED_DIR / "clips_metadata.csv",
    limit: int | None = None,
    seed: int = 42,
    fmt: str = "csv",
    verify_only: bool = False,
    overwrite: bool = False,
    config: FeatureConfig = DEFAULT_CONFIG,
    feature_dir: Path = FEATURES_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Valida, extrae, escala con train, proyecta PCA train y guarda artefactos.

    No escribe resultados oficiales si se usa --limit, y falla sin crear tablas
    oficiales cuando detecta clips corruptos o inválidos.
    """
    manifest_path = Path(manifest_path)
    all_meta, initial_audit = read_and_audit_manifest(manifest_path)
    preview = limit is not None  # --limit N SIEMPRE genera preview, aunque N >= total
    selected = sample_across_partitions(all_meta, limit, seed) if preview else all_meta.copy()
    counts = selected["partition"].value_counts().to_dict()
    if counts.get("train", 0) < 4:
        raise ValueError("Se necesitan al menos 4 clips de train para el PCA de 3 componentes")
    processed_dir = manifest_path.parent
    verify_clip_files(selected, processed_dir)
    print(f"Catálogo verificado: {len(all_meta):,} filas; clips a usar: {len(selected):,}")
    print(f"Particiones: {counts}; fugas de autor/grabación: ninguna en el catálogo")
    if verify_only:
        return {**initial_audit, "n_to_process": int(len(selected)), "verified_only": True}
    if fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("Para --format parquet instala antes: python -m pip install pyarrow")

    out_features = Path(results_dir) / "preview" / "data" if preview else Path(feature_dir)
    out_results = Path(results_dir) / "preview" if preview else Path(results_dir)
    files = [out_features / f"features_raw.{fmt}", out_features / f"features.{fmt}",
             out_results / "artifacts" / "scaler_hito3.joblib",
             out_results / "artifacts" / "pca_hito3.joblib"]
    if not overwrite:
        found = [str(p) for p in files if p.exists()]
        if found:
            raise FileExistsError("Ya existen resultados. Para reemplazarlos explícitamente usa --overwrite:\n" + "\n".join(found))

    rows: list[dict[str, float]] = []
    qa_rows: list[dict] = []
    errors: list[dict] = []
    for idx, (_, meta) in enumerate(selected.iterrows(), start=1):
        clip_id = str(meta["clip_id"])
        try:
            wav_path = resolve_clip_path(processed_dir, str(meta["clip_path"]))
            values, quality = extract_features_from_wav(wav_path, config=config)
            rows.append(values)
            qa_rows.append({"clip_id": clip_id, **quality})
        except Exception as exc:
            errors.append({"clip_id": clip_id, "clip_path": str(meta["clip_path"]),
                           "error_type": type(exc).__name__, "error": str(exc)})
        if idx % 250 == 0 or idx == len(selected):
            print(f"Procesados: {idx:,}/{len(selected):,} | errores: {len(errors)}", flush=True)

    if errors:
        out_results.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors).to_csv(out_results / "clip_errors.csv", index=False, encoding="utf-8")
        raise RuntimeError(
            f"{len(errors)} clips con errores. Revisa {out_results / 'clip_errors.csv'}. "
            "No se generó el entregable oficial (evitamos una tabla incompleta)."
        )

    columns = feature_column_names(config)
    X = pd.DataFrame(rows, columns=columns, dtype=np.float32)
    if X.shape != (len(selected), len(columns)):
        raise AssertionError(f"Dimensiones inesperadas: {X.shape}")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Se detectaron NaN o infinitos en las características")
    mask_train = selected["partition"].eq("train").to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.loc[mask_train, columns])
    # Recuperar el mismo orden original de las filas, sin volver a ajustar.
    all_scaled = scaler.transform(X[columns])
    if int(scaler.n_samples_seen_) != int(mask_train.sum()):
        raise AssertionError("StandardScaler no se ajustó con el tamaño esperado de entrenamiento")
    if not np.allclose(np.asarray(X_scaled), np.asarray(all_scaled[mask_train]), rtol=1e-6, atol=1e-6):
        raise AssertionError("Inconsistencia de transformaciones del scaler")
    scaled_df = pd.DataFrame(all_scaled, columns=columns).astype(np.float32)
    if not np.isfinite(scaled_df.to_numpy()).all():
        raise ValueError("StandardScaler produjo valores inválidos")

    pca = PCA(n_components=3, svd_solver="full")
    _ = pca.fit_transform(all_scaled[mask_train])
    if int(pca.n_samples_) != int(mask_train.sum()):
        raise AssertionError("PCA no se ajustó exclusivamente con entrenamiento")
    # Para la exploración: train y val; NO se visualiza test reservado.
    mask_exp = selected["partition"].isin(["train", "val"]).to_numpy()
    pca_vals = pca.transform(all_scaled[mask_exp])
    pca_df = selected.loc[mask_exp, ["clip_id", "recording_id", "primary_label", "author", "split", "partition"]].reset_index(drop=True)
    pca_df = pd.concat([pca_df, pd.DataFrame(pca_vals, columns=["PC1", "PC2", "PC3"])], axis=1)

    raw_df = pd.concat([selected.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    final_df = pd.concat([selected.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)
    if not raw_df["clip_id"].equals(final_df["clip_id"]):
        raise AssertionError("Se alteró el orden de los clips")
    if not np.allclose(np.mean(all_scaled[mask_train], axis=0), 0, atol=2e-4):
        raise AssertionError("Las features escaladas de train no tienen media próxima a cero")

    out_features.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_results / "artifacts"
    figures_dir = out_results / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_df(raw_df, out_features / f"features_raw.{fmt}", fmt)
    _save_df(final_df, out_features / f"features.{fmt}", fmt)
    pca_df.to_csv(out_results / "pca_scores_train_val.csv", index=False, encoding="utf-8", float_format="%.9g")
    pd.DataFrame(qa_rows).to_csv(out_results / "clip_quality_hito3.csv", index=False, encoding="utf-8")
    joblib.dump(scaler, artifacts_dir / "scaler_hito3.joblib")
    joblib.dump(pca, artifacts_dir / "pca_hito3.joblib")
    _pca_figures(pca_df, pca.explained_variance_ratio_, figures_dir)

    report = {
        "status": "PREVIEW_NOT_OFFICIAL" if preview else "FULL_EXECUTION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_manifest": initial_audit["n_manifest"],
        "n_processed": int(len(selected)),
        "n_species_processed": int(selected["primary_label"].nunique()),
        "counts_by_partition": {k: int(v) for k, v in counts.items()},
        "n_features": int(len(columns)),
        "feature_columns": columns,
        "acoustic_configuration": config.to_dict(),
        "seed_for_preview": seed,
        "n_train_used_to_fit_scaler_and_pca": int(mask_train.sum()),
        "n_samples_seen_scaler": int(scaler.n_samples_seen_),
        "n_samples_seen_pca": int(pca.n_samples_),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pca_scoring_partitions": ["train", "val"],
        "test_used_to_fit_transformations": False,
        "invalid_feature_values": 0,
        "missing_or_invalid_clips": 0,
        "shared_authors_or_recordings_between_partitions": 0,
        "manifest_sha256": initial_audit["manifest_sha256"],
        "python_and_library_versions": _package_versions(),
        "feature_file": str(out_features / f"features.{fmt}"),
        "feature_raw_file": str(out_features / f"features_raw.{fmt}"),
        "notes": [
            "La extracción acústica no utiliza etiquetas ni autores como features.",
            "Metadatos y etiquetas se conservan solo para trazabilidad/control.",
            "El PCA es exploratorio. Para validación cruzada futura reajustar scaler/PCA dentro de cada fold usando features_raw.",
            "La selección original puede implicar etiquetado débil (grabaciones con especies anotadas no necesariamente audibles en todos los clips).",
            "Los momentos espectrales de skewness y kurtosis se calculan sobre frecuencias ponderadas por magnitud; kurtosis de Fisher.",
        ],
    }
    (out_results / "hito3_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_results / "feature_columns.json").write_text(
        json.dumps(columns, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nHITO 3 FINALIZADO" if not preview else "\nVISTA PREVIA DEL HITO 3 FINALIZADA")
    print(f"Features: {out_features / f'features.{fmt}'}")
    print(f"PCA: {figures_dir / 'pca_2d_train.png'}")
    print(f"Clips: {len(selected):,} | columnas acústicas: {len(columns)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hito 3 MIAA: características WAV → CSV/Parquet, StandardScaler y PCA")
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "clips_metadata.csv",
                        help="Ruta al clips_metadata.csv del Hito 2")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ejecutar una muestra (mínimo 9); guarda resultados solo en preview/")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para seleccionar la muestra")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--verify-only", action="store_true", help="Verificar catálogo y existencia de WAV sin extraer ni guardar")
    parser.add_argument("--overwrite", action="store_true", help="Permitir sobrescribir resultados previos del Hito 3")
    args = parser.parse_args()
    run_hito3(manifest_path=args.manifest, limit=args.limit, seed=args.seed, fmt=args.format,
              verify_only=args.verify_only, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
