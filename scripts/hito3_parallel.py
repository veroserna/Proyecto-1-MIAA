"""Ejecución PARALELA del Hito 3 (solo acelera la extracción; resultados idénticos).

Reutiliza las funciones del pipeline oficial (src/training/preprocessing/hito3_feature_pipeline.py)
y reproduce exactamente su post-procesamiento: scaler/PCA ajustados solo con train,
mismos artefactos y mismo formato de salida. La única diferencia es que la extracción
de features por clip se hace con ProcessPoolExecutor en vez de en serie.

Uso (desde la raíz del repo):
    uv run python scripts/hito3_parallel.py --workers 16
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_features import (  # noqa: E402
    DEFAULT_CONFIG, extract_features_from_wav, feature_column_names,
)
from src.training.preprocessing.hito3_feature_pipeline import (  # noqa: E402
    PROCESSED_DIR, FEATURES_DIR, RESULTS_DIR,
    read_and_audit_manifest, resolve_clip_path, verify_clip_files,
    _pca_figures, _package_versions, _save_df,
)


def _extract_one(args: tuple) -> tuple:
    """Worker: extrae features de un clip. Debe ser picklable (top-level)."""
    clip_id, wav_path = args
    try:
        values, quality = extract_features_from_wav(wav_path, config=DEFAULT_CONFIG)
        return (clip_id, values, quality, None)
    except Exception as exc:  # noqa: BLE001 - se reporta igual que el pipeline oficial
        return (clip_id, None, None, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hito 3 paralelo (mismos artefactos que el pipeline oficial)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    t_start = time.time()
    manifest_path = PROCESSED_DIR / "clips_metadata.csv"

    # 1) Auditoría oficial del catálogo (idéntica al pipeline en serie)
    selected, initial_audit = read_and_audit_manifest(manifest_path)
    verify_clip_files(selected, manifest_path.parent)
    counts = selected["partition"].value_counts().to_dict()
    print(f"Catálogo verificado: {len(selected):,} filas | Particiones: {counts}")

    # 2) Resolver rutas en el proceso padre (rápido) y lanzar extracción paralela
    clip_ids = selected["clip_id"].astype(str).tolist()
    wav_paths = [
        str(resolve_clip_path(manifest_path.parent, str(rel)))
        for rel in selected["clip_path"]
    ]
    rows: dict[str, dict] = {}
    qa_rows: dict[str, dict] = {}
    errors: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for clip_id, values, quality, error in pool.map(
            _extract_one, zip(clip_ids, wav_paths), chunksize=32
        ):
            done += 1
            if error is not None:
                errors.append({"clip_id": clip_id, "error": error})
            else:
                rows[clip_id] = values
                qa_rows[clip_id] = {"clip_id": clip_id, **quality}
            if done % 1000 == 0 or done == len(clip_ids):
                elapsed = time.time() - t_start
                print(f"Procesados: {done:,}/{len(clip_ids):,} | errores: {len(errors)} | {elapsed:.0f}s", flush=True)

    out_results = RESULTS_DIR
    if errors:
        out_results.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors).to_csv(out_results / "clip_errors.csv", index=False, encoding="utf-8")
        raise RuntimeError(f"{len(errors)} clips con errores. Revisa {out_results / 'clip_errors.csv'}.")

    # 3) Post-procesamiento IDÉNTICO al pipeline oficial (mismo orden de filas)
    columns = feature_column_names(DEFAULT_CONFIG)
    X = pd.DataFrame([rows[cid] for cid in clip_ids], columns=columns, dtype=np.float32)
    if X.shape != (len(selected), len(columns)):
        raise AssertionError(f"Dimensiones inesperadas: {X.shape}")
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Se detectaron NaN o infinitos en las características")

    mask_train = selected["partition"].eq("train").to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.loc[mask_train, columns])
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

    # 4) Guardado de artefactos (mismas rutas y formatos que el oficial)
    out_features = FEATURES_DIR
    out_features.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_results / "artifacts"
    figures_dir = out_results / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fmt = "csv"
    _save_df(raw_df, out_features / f"features_raw.{fmt}", fmt)
    _save_df(final_df, out_features / f"features.{fmt}", fmt)
    pca_df.to_csv(out_results / "pca_scores_train_val.csv", index=False, encoding="utf-8", float_format="%.9g")
    pd.DataFrame([qa_rows[cid] for cid in clip_ids]).to_csv(
        out_results / "clip_quality_hito3.csv", index=False, encoding="utf-8")
    joblib.dump(scaler, artifacts_dir / "scaler_hito3.joblib")
    joblib.dump(pca, artifacts_dir / "pca_hito3.joblib")
    _pca_figures(pca_df, pca.explained_variance_ratio_, figures_dir)

    report = {
        "status": "FULL_EXECUTION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_manifest": initial_audit["n_manifest"],
        "n_processed": int(len(selected)),
        "n_species_processed": int(selected["primary_label"].nunique()),
        "counts_by_partition": {k: int(v) for k, v in counts.items()},
        "n_features": int(len(columns)),
        "feature_columns": columns,
        "acoustic_configuration": DEFAULT_CONFIG.to_dict(),
        "seed_for_preview": 42,
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
        "execution_note": "Extracción paralela (ProcessPoolExecutor); post-procesamiento idéntico al pipeline oficial.",
    }
    (out_results / "hito3_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_results / "feature_columns.json").write_text(
        json.dumps(columns, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nHITO 3 FINALIZADO (paralelo, {args.workers} workers) en {time.time()-t_start:.0f}s")
    print(f"Features: {out_features / f'features.{fmt}'}")
    print(f"Varianza PCA: {report['pca_explained_variance_ratio']}")


if __name__ == "__main__":
    main()
