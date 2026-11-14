"""Hito 5 — Comparacion preliminar baseline vs. SVM (lineal, RBF, polinomial).

Validacion cruzada agrupada por autor reutilizando los 5 folds ya fijados en Hito 1/2
(columna `split`: fold0..fold4). La tarea se trata como multietiqueta (16 especies);
las predicciones por clip se agregan por `recording_id` con el maximo por especie antes
de calcular Macro-F1 / Macro ROC-AUC (protocolo de docs/entrega_01_proyecto_innovacion.md).

No usa la particion `test` en ningun momento.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.utils.helpers import print_header  # noqa: E402

FEATURES_CSV = PROJECT_ROOT / "src" / "data" / "features" / "features.csv"
HITO3_SUMMARY = PROJECT_ROOT / "experiments" / "results" / "hito3" / "hito3_summary.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "supervised"
SEED = 42
FOLDS = [f"fold{i}" for i in range(5)]
# Tope inicial ampliado SOLO para SVM (no para el baseline). Usa None para entrenar
# con todos los clips del fold. El valor 5000 es un punto de partida, no un óptimo validado.
SVM_MAX_TRAIN_SAMPLES: int | None = 5000
MODELS = ["baseline_logreg", "svm_linear", "svm_rbf", "svm_poly"]


def load_feature_columns() -> list[str]:
    with open(HITO3_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)
    return list(summary["feature_columns"])


def build_multilabel_matrix(df: pd.DataFrame, species: list[str]) -> np.ndarray:
    """Vector binario por especie: 1 si aparece como primary_label o en secondary (dentro del subconjunto)."""
    sp_index = {s: i for i, s in enumerate(species)}
    Y = np.zeros((len(df), len(species)), dtype=np.int8)
    primary = df["primary_label"].to_numpy()
    secondary = df["secondary"].fillna("").to_numpy()
    for row_i, (p, sec) in enumerate(zip(primary, secondary)):
        if p in sp_index:
            Y[row_i, sp_index[p]] = 1
        if sec:
            for code in str(sec).split():
                if code in sp_index:
                    Y[row_i, sp_index[code]] = 1
    return Y


def aggregate_by_recording(
    recording_ids: np.ndarray, scores: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Agrega scores y etiquetas por recording_id usando el maximo por especie."""
    df_agg = pd.DataFrame(scores)
    df_agg["recording_id"] = recording_ids
    df_true = pd.DataFrame(y_true)
    df_true["recording_id"] = recording_ids
    agg_scores = df_agg.groupby("recording_id").max().to_numpy()
    agg_true = df_true.groupby("recording_id").max().to_numpy()
    return agg_scores, agg_true


def subsample_train(X, Y, groups, max_samples: int | None, seed: int):
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples debe ser un entero positivo o None.")
    if max_samples is None or len(X) <= max_samples:
        return X, Y, groups
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], Y[idx], groups[idx]


def run_hito5_baseline_svm(
    seed: int = SEED,
    svm_max_train_samples: int | None = SVM_MAX_TRAIN_SAMPLES,
) -> dict:
    print_header("Hito 5 — Baseline y SVM (lineal, RBF, polinomial) con CV agrupada por autor")
    if svm_max_train_samples is not None and svm_max_train_samples < 1:
        raise ValueError("svm_max_train_samples debe ser un entero positivo o None.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_columns()
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["partition"] != "test"].reset_index(drop=True)
    species = sorted(df["primary_label"].unique().tolist())
    print(f"Clips usados (train+val, sin test): {len(df):,} | especies: {len(species)}")

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    Y_all = build_multilabel_matrix(df, species)
    split_all = df["split"].to_numpy()
    rec_all = df["recording_id"].to_numpy()

    fold_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for fold in FOLDS:
        t0 = time.time()
        val_mask = split_all == fold
        train_mask = (~val_mask) & np.isin(split_all, FOLDS)

        X_train, Y_train = X_all[train_mask], Y_all[train_mask]
        X_val, Y_val, rec_val = X_all[val_mask], Y_all[val_mask], rec_all[val_mask]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)

        for model_name in MODELS:
            if model_name == "baseline_logreg":
                clf = OneVsRestClassifier(
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
                )
                clf.fit(X_train_s, Y_train)
                scores_val = clf.predict_proba(X_val_s)
                threshold = 0.5
            else:
                kernel = model_name.split("_")[1]
                X_tr_sub, Y_tr_sub, _ = subsample_train(
                    X_train_s, Y_train, np.arange(len(X_train_s)), svm_max_train_samples, seed
                )
                clf = OneVsRestClassifier(
                    SVC(
                        kernel=kernel,
                        degree=3 if kernel == "poly" else 3,
                        class_weight="balanced",
                        random_state=seed,
                    )
                )
                clf.fit(X_tr_sub, Y_tr_sub)
                scores_val = clf.decision_function(X_val_s)
                threshold = 0.0

            y_pred_val = (scores_val >= threshold).astype(int)
            agg_scores, agg_true = aggregate_by_recording(rec_val, scores_val, Y_val)
            agg_pred, _ = aggregate_by_recording(rec_val, y_pred_val, Y_val)
            agg_pred = (agg_pred >= 1).astype(int)

            m = compute_metrics(agg_true, agg_pred, y_prob=agg_scores, class_names=None)
            per_class_f1_arr = f1_score(agg_true, agg_pred, average=None, zero_division=0)
            per_class_f1 = dict(enumerate(map(float, per_class_f1_arr)))

            fold_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "n_train_clips": int(train_mask.sum()),
                    "n_model_train_clips": int(
                        len(X_train_s) if model_name == "baseline_logreg" else len(X_tr_sub)
                    ),
                    "n_val_clips": int(val_mask.sum()),
                    "n_val_recordings": int(agg_true.shape[0]),
                    "accuracy": m["accuracy"],
                    "f1_macro": m["f1_macro"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro": m["recall_macro"],
                    "roc_auc_macro": m.get("roc_auc_macro"),
                    "cmap": m.get("cmap"),
                    "seconds": round(time.time() - t0, 1),
                }
            )
            for sp_idx, sp_code in enumerate(species):
                per_class_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "species": sp_code,
                        "f1": per_class_f1.get(sp_idx),
                    }
                )
            print(
                f"  [{fold}] {model_name:16s} | F1-macro={m['f1_macro']:.3f} "
                f"| ROC-AUC-macro={m.get('roc_auc_macro', float('nan')):.3f} "
                f"| ({time.time() - t0:.1f}s)"
            )

    scores_df = pd.DataFrame(fold_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    scores_df.to_csv(OUT_DIR / "cv_scores_by_fold.csv", index=False)
    per_class_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    summary = (
        scores_df.groupby("model")[["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_macro"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_dict = {
        model: {
            metric: {"mean": float(summary.loc[model, (metric, "mean")]), "std": float(summary.loc[model, (metric, "std")])}
            for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_macro"]
        }
        for model in summary.index
    }
    config = {
        "seed": seed,
        "folds": FOLDS,
        "models": MODELS,
        "class_weight": "balanced",
        "svm_max_train_samples_per_fold": svm_max_train_samples,
        "svm_uses_full_fold": svm_max_train_samples is None,
        "threshold_baseline": 0.5,
        "threshold_svm_decision_function": 0.0,
        "n_species": len(species),
        "species": species,
    }
    with open(OUT_DIR / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "summary_by_model": summary_dict}, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nResumen (promedio +/- std entre 5 folds):")
    print(summary.to_string())
    print(f"\nResultados guardados en: {OUT_DIR}")
    return {"cv_scores": scores_df, "summary": summary_dict, "config": config}


if __name__ == "__main__":
    run_hito5_baseline_svm()
