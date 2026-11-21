"""Evaluación final (23–27 nov) — una sola vez sobre el conjunto de prueba.

Protocolo (según `docs/plan_trabajo.md` y la retroalimentación del evaluador):
- Modelo seleccionado: XGBoost (mejor Macro-F1 en validación del Hito 6).
- Hiperparámetros: configuración modal entre los 5 folds del Hito 6
  (`experiments/results/hito6/config.json` → `best_params_by_fold`); no se vuelve
  a buscar sobre test.
- Ajuste de umbrales: un umbral por especie se elige maximizando F1 sobre el fold de
  validación (fold4 = partition `val`), protocolo de la primera entrega.
- Entrenamiento final: train (fold0..fold3) + val (fold4) con los hiperparámetros
  fijos; el scaler se ajusta sobre train+val.
- Evaluación UNA SOLA VEZ sobre `test`, con agregación por grabación (máximo por
  especie), matrices de confusión por especie y métricas por clase.

Salidas en `experiments/results/evaluacion_final/`:
- `test_metrics.json`, `per_class_test.csv`, `thresholds.json`, `config.json`,
- `figures/confusion_matrix_<especie>.png` y `figures/confusion_matrices_grid.png`,
- `artifacts/final_model.joblib`, `artifacts/scaler.joblib`.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.utils.helpers import print_header  # noqa: E402
from src.training.hito5_baseline_svm import (  # noqa: E402
    build_multilabel_matrix, aggregate_by_recording, load_feature_columns,
)
from src.training.hito6_ensembles import _cuda_available, FEATURES_CSV  # noqa: E402

HITO6_CONFIG = PROJECT_ROOT / "experiments" / "results" / "hito6" / "config.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "evaluacion_final"
SEED = 42
TRAIN_FOLDS = ["fold0", "fold1", "fold2", "fold3"]
VAL_FOLD = "fold4"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def _modal_best_params(config: dict, model: str = "xgboost") -> dict:
    """Configuración modal (voto mayoritario) de los mejores parámetros por fold."""
    by_fold = config["best_params_by_fold"][model]
    keys = next(iter(by_fold.values())).keys()
    modal = {}
    for key in keys:
        values = [params[key] for params in by_fold.values()]
        modal[key] = Counter(map(repr, values)).most_common(1)[0][0]
        # Recuperar el valor original (no su repr)
        for params in by_fold.values():
            if repr(params[key]) == modal[key]:
                modal[key] = params[key]
                break
    return modal


def _make_xgb(params: dict, seed: int, use_cuda: bool) -> OneVsRestClassifier:
    import xgboost as xgb
    inner = {k.replace("estimator__", ""): v for k, v in params.items()}
    base = xgb.XGBClassifier(
        tree_method="hist",
        device="cuda" if use_cuda else "cpu",
        eval_metric="logloss",
        random_state=seed,
        **inner,
    )
    return OneVsRestClassifier(base, n_jobs=1)


def _tune_thresholds(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Umbral por especie que maximiza F1 sobre el fold de validación."""
    thresholds = np.full(scores.shape[1], 0.5)
    for j in range(scores.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in THRESHOLD_GRID:
            f1 = f1_score(y_true[:, j], (scores[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[j] = best_t
    return thresholds


def _plot_confusion_matrices(cm_rows: list[dict], species: list[str], out_png: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for ax, row, sp in zip(axes.ravel(), cm_rows, species):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_title(sp)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    fig.suptitle("Matrices de confusión por especie — evaluación final en test")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_evaluacion_final(
    seed: int = SEED,
    features_csv: Path = FEATURES_CSV,
    hito6_config: Path = HITO6_CONFIG,
    out_dir: Path = OUT_DIR,
) -> dict:
    print_header("Evaluación final — XGBoost, umbrales ajustados en val, una sola pasada por test")
    out_dir = Path(out_dir)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    config6 = json.loads(Path(hito6_config).read_text(encoding="utf-8"))
    best_params = _modal_best_params(config6, "xgboost")
    use_cuda = _cuda_available()
    print(f"Dispositivo XGBoost: {'cuda (GPU)' if use_cuda else 'cpu'}")
    print(f"Hiperparámetros (configuración modal de los 5 folds del Hito 6): {best_params}")

    feature_cols = load_feature_columns()
    df = pd.read_csv(features_csv)
    species = sorted(df["primary_label"].unique().tolist())

    train_mask = df["split"].isin(TRAIN_FOLDS).to_numpy()
    val_mask = (df["split"] == VAL_FOLD).to_numpy()
    test_mask = (df["partition"] == "test").to_numpy()
    print(f"Clips — train: {train_mask.sum():,} | val: {val_mask.sum():,} | test: {test_mask.sum():,}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    Y = build_multilabel_matrix(df, species)
    rec = df["recording_id"].to_numpy()

    # 1) Entrenar en train, ajustar umbrales en val
    t0 = time.time()
    scaler_tv = StandardScaler().fit(X[train_mask])
    model_tv = _make_xgb(best_params, seed, use_cuda)
    model_tv.fit(scaler_tv.transform(X[train_mask]), Y[train_mask])
    scores_val = model_tv.predict_proba(scaler_tv.transform(X[val_mask]))
    thresholds = _tune_thresholds(scores_val, Y[val_mask])
    agg_val_s, agg_val_y = aggregate_by_recording(rec[val_mask], scores_val, Y[val_mask])
    val_pred = (agg_val_s >= thresholds).astype(int)
    val_metrics = compute_metrics(agg_val_y, val_pred, agg_val_s)
    print(f"Val (fold4) con umbrales ajustados: Macro-F1 = {val_metrics['f1_macro']:.3f} "
          f"({time.time() - t0:.0f} s)")

    # 2) Reentrenar en train+val y evaluar UNA SOLA VEZ en test
    t0 = time.time()
    tv_mask = train_mask | val_mask
    scaler = StandardScaler().fit(X[tv_mask])
    model = _make_xgb(best_params, seed, use_cuda)
    model.fit(scaler.transform(X[tv_mask]), Y[tv_mask])
    scores_test = model.predict_proba(scaler.transform(X[test_mask]))
    agg_test_s, agg_test_y = aggregate_by_recording(rec[test_mask], scores_test, Y[test_mask])
    test_pred = (agg_test_s >= thresholds).astype(int)
    test_metrics = compute_metrics(agg_test_y, test_pred, agg_test_s)
    print(f"TEST (una sola evaluación): Macro-F1 = {test_metrics['f1_macro']:.3f} | "
          f"ROC-AUC = {test_metrics['roc_auc_macro']:.3f} ({time.time() - t0:.0f} s)")

    # 3) Métricas y matrices de confusión por especie
    per_class_rows, cm_rows = [], []
    for j, sp in enumerate(species):
        y_true_j, y_pred_j = agg_test_y[:, j], test_pred[:, j]
        tn, fp, fn, tp = confusion_matrix(y_true_j, y_pred_j, labels=[0, 1]).ravel()
        cm_rows.append({"species": sp, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        support = int(y_true_j.sum())
        per_class_rows.append({
            "species": sp,
            "support_recordings": support,
            "f1": f1_score(y_true_j, y_pred_j, zero_division=0),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "threshold": float(thresholds[j]),
        })
        cm_fig, cm_ax = plt.subplots(figsize=(4, 4))
        cm = np.array([[tn, fp], [fn, tp]])
        cm_ax.imshow(cm, cmap="Blues")
        for (i, j2), v in np.ndenumerate(cm):
            cm_ax.text(j2, i, str(v), ha="center", va="center",
                       color="white" if v > cm.max() / 2 else "black")
        cm_ax.set_title(f"{sp} — test")
        cm_ax.set_xticks([0, 1]); cm_ax.set_yticks([0, 1])
        cm_ax.set_xlabel("Predicho"); cm_ax.set_ylabel("Real")
        cm_fig.tight_layout()
        cm_fig.savefig(out_dir / "figures" / f"confusion_matrix_{sp}.png", dpi=150)
        plt.close(cm_fig)
    _plot_confusion_matrices(cm_rows, species, out_dir / "figures" / "confusion_matrices_grid.png")

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(out_dir / "per_class_test.csv", index=False)
    (out_dir / "thresholds.json").write_text(
        json.dumps({sp: float(t) for sp, t in zip(species, thresholds)}, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps({
        "test_metrics": test_metrics,
        "val_metrics_with_tuned_thresholds": val_metrics,
        "n_test_recordings": int(len(agg_test_y)),
        "n_test_clips": int(test_mask.sum()),
    }, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps({
        "seed": seed,
        "model": "xgboost",
        "best_params_modal_hito6": best_params,
        "train_folds": TRAIN_FOLDS,
        "val_fold_for_thresholds": VAL_FOLD,
        "threshold_tuning": "F1 por especie sobre fold4 (val), rejilla 0.05-0.95",
        "aggregation": "máximo por recording_id",
        "device": "cuda" if use_cuda else "cpu",
        "test_used_once": True,
    }, indent=2), encoding="utf-8")
    joblib.dump(model, out_dir / "artifacts" / "final_model.joblib")
    joblib.dump(scaler, out_dir / "artifacts" / "scaler.joblib")

    print(f"Artefactos en: {out_dir}")
    return {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "per_class": per_class_df,
        "thresholds": dict(zip(species, thresholds.tolist())),
        "config": {"best_params": best_params, "seed": seed},
    }
