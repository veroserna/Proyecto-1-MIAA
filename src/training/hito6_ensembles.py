"""Hito 6 — Ensambles optimizados: Random Forest y XGBoost con ajuste de hiperparámetros.

Protocolo (consistente con Hito 5 y la primera entrega):
- Tarea multietiqueta sobre las 16 especies (primary_label + secondary del subconjunto).
- Validación cruzada externa de 5 folds agrupados por autor (columna `split`, fold0..fold4);
  la partición `test` NO se usa en ningún momento.
- El StandardScaler se reajusta dentro de cada fold de entrenamiento (los árboles no lo
  requieren, pero se conserva la regla del protocolo para comparabilidad).
- Búsqueda de hiperparámetros (RandomizedSearchCV) SOLO dentro del fold de entrenamiento,
  con validación interna GroupKFold(3) agrupada por autor: nunca ve validación externa.
- Predicciones por clip agregadas por recording_id (máximo por especie) antes de evaluar.
- Umbral fijo 0.5 (idéntico al Hito 5) para no introducir grados de libertad extra.
- Se reportan IC del 95 % (t de Student sobre los 5 folds) y prueba pareada de Wilcoxon
  contra la línea base del Hito 5 (SVM RBF), respondiendo a la retroalimentación del
  evaluador sobre intervalos/pruebas pareadas.
- SMOTE queda fuera de esta corrida: las métricas del Hito 5 no mostraron colapso de
  recall por clase (recall macro 0.60-0.88), y un remuestreo multietiqueta requeriría
  validación adicional; se documenta como trabajo condicional del informe final.
"""
from __future__ import annotations

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
from scipy.stats import t as t_dist, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.utils.helpers import print_header  # noqa: E402
from src.training.hito5_baseline_svm import (  # noqa: E402
    build_multilabel_matrix, aggregate_by_recording, load_feature_columns,
)

FEATURES_CSV = PROJECT_ROOT / "src" / "data" / "features" / "features.csv"
HITO5_SCORES = PROJECT_ROOT / "experiments" / "results" / "hito4_5" / "supervised" / "cv_scores_by_fold.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "hito6"
SEED = 42
FOLDS = [f"fold{i}" for i in range(5)]
MODELS = ["random_forest", "xgboost"]
N_ITER_SEARCH = 12          # candidatos por modelo por fold (Random Search)
INNER_SPLITS = 3            # GroupKFold interno (por autor) dentro de cada fold de entrenamiento
THRESHOLD = 0.5

RF_PARAM_DIST = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2", 0.3],
    "class_weight": ["balanced", "balanced_subsample"],
}
XGB_PARAM_DIST = {
    "estimator__n_estimators": [300, 600, 900],
    "estimator__max_depth": [4, 6, 8, 10],
    "estimator__learning_rate": [0.03, 0.05, 0.1],
    "estimator__subsample": [0.7, 0.8, 1.0],
    "estimator__colsample_bytree": [0.5, 0.7, 1.0],
    "estimator__min_child_weight": [1, 5, 10],
    "estimator__scale_pos_weight": [1, 3, 10],
}


def _cuda_available() -> bool:
    try:
        import xgboost as xgb
        xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=2).fit(
            np.random.rand(20, 4), (np.random.rand(20) > 0.5).astype(int))
        return True
    except Exception:
        return False


def _resolve_device(device: str = "auto") -> bool:
    """True si XGBoost debe usar CUDA. `device`: 'auto' (detecta), 'cuda' o 'cpu'."""
    if device == "cuda":
        return True
    if device == "cpu":
        return False
    return _cuda_available()


def _make_model(name: str, seed: int, use_cuda: bool):
    """Construye (estimador, distribución) para la búsqueda aleatoria."""
    if name == "random_forest":
        # RandomForest soporta multietiqueta de forma nativa (multioutput), sin OvR.
        return RandomForestClassifier(random_state=seed, n_jobs=-1), RF_PARAM_DIST
    if name == "xgboost":
        import xgboost as xgb
        base = xgb.XGBClassifier(
            tree_method="hist",
            device="cuda" if use_cuda else "cpu",
            eval_metric="logloss",
            random_state=seed,
        )
        return OneVsRestClassifier(base, n_jobs=1), XGB_PARAM_DIST
    raise ValueError(f"Modelo desconocido: {name}")


def _scores_proba(model_name: str, fitted, X: np.ndarray) -> np.ndarray:
    """Score continuo por especie (probabilidad de la clase positiva)."""
    if model_name == "random_forest":
        proba_list = fitted.predict_proba(X)  # multietiqueta nativo: lista por salida
        return np.column_stack([
            p[:, 1] if p.shape[1] == 2 else np.zeros(len(X)) for p in proba_list
        ])
    return fitted.predict_proba(X)


def _feature_importance(model_name: str, fitted, feature_cols: list[str]) -> np.ndarray:
    if model_name == "random_forest":
        return np.asarray(fitted.feature_importances_, dtype=float)
    # OvR de XGBoost: promedio del gain normalizado entre las 16 salidas
    imps = [np.asarray(est.feature_importances_, dtype=float) for est in fitted.estimators_]
    return np.mean(imps, axis=0)


def _t_ci95(values: list[float]) -> tuple[float, float]:
    """IC 95 % t de Student sobre 5 folds (n pequeño: interpretar como orientativo)."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    half = t_dist.ppf(0.975, df=len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean()), float(half)


def run_hito6_ensembles(
    seed: int = SEED,
    n_iter: int = N_ITER_SEARCH,
    features_csv: Path = FEATURES_CSV,
    out_dir: Path = OUT_DIR,
    device: str = "auto",
) -> dict:
    print_header("Hito 6 — Random Forest y XGBoost con búsqueda dentro de CV agrupada por autor")
    out_dir = Path(out_dir)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    use_cuda = _resolve_device(device)
    print(f"XGBoost device: {'cuda (GPU)' if use_cuda else 'cpu'}")

    feature_cols = load_feature_columns()
    df = pd.read_csv(features_csv)
    df = df[df["partition"] != "test"].reset_index(drop=True)
    species = sorted(df["primary_label"].unique().tolist())
    print(f"Clips usados (train+val, sin test): {len(df):,} | especies: {len(species)}")

    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    Y_all = build_multilabel_matrix(df, species)
    split_all = df["split"].to_numpy()
    rec_all = df["recording_id"].to_numpy()
    author_all = df["author"].to_numpy()

    fold_rows: list[dict] = []
    per_class_rows: list[dict] = []
    importance_rows: list[dict] = []
    best_params_log: dict[str, dict] = {m: {} for m in MODELS}

    for fold in FOLDS:
        t0 = time.time()
        val_mask = split_all == fold
        train_mask = (~val_mask) & np.isin(split_all, FOLDS)

        X_train, Y_train = X_all[train_mask], Y_all[train_mask]
        X_val, Y_val, rec_val = X_all[val_mask], Y_all[val_mask], rec_all[val_mask]
        groups_train = author_all[train_mask]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)

        inner_cv = GroupKFold(n_splits=INNER_SPLITS)

        for model_name in MODELS:
            estimator, param_dist = _make_model(model_name, seed, use_cuda)
            search = RandomizedSearchCV(
                estimator,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="f1_macro",
                cv=inner_cv,
                random_state=seed,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train_s, Y_train, groups=groups_train)
            best = search.best_estimator_
            best_params_log[model_name][fold] = {
                k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                for k, v in search.best_params_.items()
            }

            scores_val = _scores_proba(model_name, best, X_val_s)
            y_pred_val = (scores_val >= THRESHOLD).astype(int)
            agg_scores, agg_true = aggregate_by_recording(rec_val, scores_val, Y_val)
            agg_pred, _ = aggregate_by_recording(rec_val, y_pred_val, Y_val)
            agg_pred = (agg_pred >= 1).astype(int)

            m = compute_metrics(agg_true, agg_pred, y_prob=agg_scores, class_names=None)
            per_class_f1_arr = f1_score(agg_true, agg_pred, average=None, zero_division=0)

            fold_rows.append({
                "fold": fold,
                "model": model_name,
                "n_train_clips": int(train_mask.sum()),
                "n_val_clips": int(val_mask.sum()),
                "n_val_recordings": int(agg_true.shape[0]),
                "cv_f1_macro_inner_best": float(search.best_score_),
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "roc_auc_macro": m.get("roc_auc_macro"),
                "cmap": m.get("cmap"),
                "seconds": round(time.time() - t0, 1),
            })
            for sp_idx, sp_code in enumerate(species):
                per_class_rows.append({
                    "fold": fold, "model": model_name,
                    "species": sp_code, "f1": float(per_class_f1_arr[sp_idx]),
                })
            for f_idx, fname in enumerate(feature_cols):
                importance_rows.append({
                    "fold": fold, "model": model_name,
                    "feature": fname,
                    "importance": float(_feature_importance(model_name, best, feature_cols)[f_idx]),
                })
            joblib.dump(best, out_dir / "artifacts" / f"{model_name}_{fold}.joblib")
            print(f"  [{fold}] {model_name:14s} | F1-macro={m['f1_macro']:.3f} "
                  f"| ROC-AUC={m.get('roc_auc_macro', float('nan')):.3f} "
                  f"| (acum {time.time() - t0:.0f}s)", flush=True)

    scores_df = pd.DataFrame(fold_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    importance_df = pd.DataFrame(importance_rows)
    scores_df.to_csv(out_dir / "cv_scores_by_fold.csv", index=False)
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # --- Resumen con IC del 95 % (t de Student, 5 folds) ---
    summary_dict = {}
    for model in MODELS:
        sub = scores_df[scores_df["model"] == model]
        entry = {}
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_macro"]:
            mean_v, half = _t_ci95(sub[metric].tolist())
            entry[metric] = {
                "mean": round(mean_v, 4),
                "std": round(float(sub[metric].std(ddof=1)), 4),
                "ci95_low": round(mean_v - half, 4),
                "ci95_high": round(mean_v + half, 4),
            }
        summary_dict[model] = entry

    # --- Prueba pareada de Wilcoxon contra la línea base del Hito 5 (SVM RBF) ---
    paired = {}
    if Path(HITO5_SCORES).is_file():
        h5 = pd.read_csv(HITO5_SCORES)
        base = h5[h5["model"] == "svm_rbf"].set_index("fold")["f1_macro"]
        for model in MODELS:
            cur = scores_df[scores_df["model"] == model].set_index("fold")["f1_macro"]
            common = base.index.intersection(cur.index)
            diff = cur[common] - base[common]
            try:
                stat = wilcoxon(diff.tolist())
                p_val = float(stat.pvalue)
            except ValueError:
                p_val = float("nan")
            paired[model] = {
                "baseline": "hito5_svm_rbf",
                "folds": list(common),
                "diff_f1_macro_per_fold": [round(float(d), 4) for d in diff],
                "mean_diff": round(float(diff.mean()), 4),
                "wilcoxon_p": round(p_val, 4),
                "note": "n=5 folds: potencia baja; interpretar como evidencia orientativa, no concluyente.",
            }
    with open(out_dir / "paired_tests_vs_hito5.json", "w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2, ensure_ascii=False)

    config = {
        "seed": seed,
        "folds": FOLDS,
        "models": MODELS,
        "n_iter_search": n_iter,
        "inner_cv": f"GroupKFold({INNER_SPLITS}) por autor, solo dentro del fold de entrenamiento",
        "threshold": THRESHOLD,
        "xgboost_device": "cuda" if use_cuda else "cpu",
        "smote": "no aplicado: recall macro Hito 5 = 0.60-0.88 (sin colapso); remuestreo multietiqueta queda como trabajo condicional",
        "n_species": len(species),
        "species": species,
        "best_params_by_fold": best_params_log,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": {k: v for k, v in config.items() if k != "best_params_by_fold"},
                   "summary_by_model": summary_dict}, f, indent=2, ensure_ascii=False)

    # --- Figura de importancia de características (top 20, promedio entre folds) ---
    for model in MODELS:
        imp = (importance_df[importance_df["model"] == model]
               .groupby("feature")["importance"].agg(["mean", "std"])
               .sort_values("mean", ascending=False).head(20).iloc[::-1])
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(imp.index, imp["mean"], xerr=imp["std"], color="#2c7fb8")
        ax.set_title(f"Hito 6 · Importancia de características (top 20) — {model}")
        ax.set_xlabel("Importancia media ± std entre folds")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / f"feature_importance_{model}.png", dpi=150)
        plt.close(fig)

    print("\nResumen (media ± std, IC95% entre 5 folds):")
    for model in MODELS:
        s = summary_dict[model]
        print(f"  {model:14s} F1={s['f1_macro']['mean']:.3f} "
              f"[{s['f1_macro']['ci95_low']:.3f}, {s['f1_macro']['ci95_high']:.3f}] "
              f"| AUC={s['roc_auc_macro']['mean']:.3f} "
              f"[{s['roc_auc_macro']['ci95_low']:.3f}, {s['roc_auc_macro']['ci95_high']:.3f}]")
    if paired:
        for model, p in paired.items():
            print(f"  Wilcoxon {model} vs SVM-RBF (Hito 5): ΔF1 medio={p['mean_diff']:+.4f}, p={p['wilcoxon_p']}")
    print(f"\nResultados guardados en: {out_dir}")
    return {"cv_scores": scores_df, "summary": summary_dict, "config": config}


if __name__ == "__main__":
    run_hito6_ensembles()
