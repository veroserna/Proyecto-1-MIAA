"""Análisis de sensibilidad de umbral para Hito 6 (post-corrida).

Protocolo (según definición de la primera entrega): los umbrales por especie se eligen
con el fold de VALIDACIÓN de cada vuelta (maximizando F1 por especie sobre ese fold),
nunca con test. Se reporta como análisis complementario a la corrida oficial con
umbral fijo 0.5, reutilizando los modelos ya guardados en artifacts/.

Salida: experiments/results/hito6/threshold_sensitivity.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hito5_baseline_svm import (  # noqa: E402
    build_multilabel_matrix, aggregate_by_recording, load_feature_columns,
)
from src.training.hito6_ensembles import FEATURES_CSV, OUT_DIR, FOLDS, MODELS, _scores_proba  # noqa: E402

GRID = np.round(np.arange(0.02, 0.91, 0.02), 2)


def main() -> None:
    feature_cols = load_feature_columns()
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["partition"] != "test"].reset_index(drop=True)
    species = sorted(df["primary_label"].unique().tolist())
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    Y_all = build_multilabel_matrix(df, species)
    split_all = df["split"].to_numpy()
    rec_all = df["recording_id"].to_numpy()

    results: dict[str, dict] = {}
    for model_name in MODELS:
        per_fold = []
        for fold in FOLDS:
            val_mask = split_all == fold
            train_mask = (~val_mask) & np.isin(split_all, FOLDS)
            scaler = StandardScaler().fit(X_all[train_mask])
            X_val_s = scaler.transform(X_all[val_mask])
            Y_val, rec_val = Y_all[val_mask], rec_all[val_mask]

            model = joblib.load(OUT_DIR / "artifacts" / f"{model_name}_{fold}.joblib")
            scores_val = _scores_proba(model_name, model, X_val_s)
            agg_scores, agg_true = aggregate_by_recording(rec_val, scores_val, Y_val)

            # Umbral por especie elegido en el fold de validación (máximo F1 por especie).
            thresholds = np.zeros(len(species))
            for j in range(len(species)):
                y_j = agg_true[:, j]
                if y_j.sum() == 0:
                    thresholds[j] = 0.5
                    continue
                f1s = [f1_score(y_j, (agg_scores[:, j] >= t).astype(int), zero_division=0) for t in GRID]
                thresholds[j] = float(GRID[int(np.argmax(f1s))])

            agg_pred = (agg_scores >= thresholds[None, :]).astype(int)
            f1_macro = float(f1_score(agg_true, agg_pred, average="macro", zero_division=0))
            per_fold.append({"fold": fold, "f1_macro_tuned": round(f1_macro, 4),
                             "threshold_mean": round(float(thresholds.mean()), 3)})

        vals = [p["f1_macro_tuned"] for p in per_fold]
        results[model_name] = {
            "per_fold": per_fold,
            "f1_macro_mean": round(float(np.mean(vals)), 4),
            "f1_macro_std": round(float(np.std(vals, ddof=1)), 4),
            "note": "Umbral por especie elegido maximizando F1 sobre el propio fold de validación (protocolo de la primera entrega). No usa test.",
        }
        print(f"{model_name}: F1 con umbral ajustado = {np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}")

    out = OUT_DIR / "threshold_sensitivity.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
