"""Ajuste de umbrales out-of-fold (OOF) para el modelo final — NO toca test.

Reconstruye las predicciones de validación de los 5 folds del Hito 6 (cada clip de
train+val fue validado exactamente una vez), reajusta el StandardScaler del fold de
forma determinista (StandardScaler solo depende de los datos de entrenamiento del fold),
carga el modelo XGBoost guardado por fold y obtiene scores OOF para los 17.893 clips.

Con esos scores agregados por grabación (máximo por especie) se ajusta un umbral por
especie maximizando F1 — mismo criterio de la primera entrega, pero con ~5x más datos
de validación que el ajuste solo sobre fold4, lo que reduce el sobreajuste del umbral.

Salida: `experiments/results/hito6/oof_thresholds.json` con umbrales OOF, umbrales
fold4 (los usados en la evaluación final) y comparación de Macro-F1 en validación.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.hito5_baseline_svm import (  # noqa: E402
    build_multilabel_matrix, aggregate_by_recording, load_feature_columns,
)
from src.training.hito6_ensembles import FEATURES_CSV, OUT_DIR, FOLDS  # noqa: E402
from src.training.hito7_evaluacion_final import (  # noqa: E402
    _tune_thresholds, VAL_FOLD,
)

FINAL_THRESHOLDS = PROJECT_ROOT / "experiments" / "results" / "evaluacion_final" / "thresholds.json"


def main() -> None:
    feature_cols = load_feature_columns()
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["partition"] != "test"].reset_index(drop=True)  # NUNCA test
    species = sorted(df["primary_label"].unique().tolist())

    X = df[feature_cols].to_numpy(dtype=np.float64)
    Y = build_multilabel_matrix(df, species)
    split = df["split"].to_numpy()
    rec = df["recording_id"].to_numpy()

    oof_scores = np.full((len(df), len(species)), np.nan)
    for fold in FOLDS:
        val_mask = split == fold
        train_mask = (~val_mask) & np.isin(split, FOLDS)
        scaler = StandardScaler().fit(X[train_mask])  # determinista: replica el del Hito 6
        model = joblib.load(OUT_DIR / "artifacts" / f"xgboost_{fold}.joblib")
        oof_scores[val_mask] = model.predict_proba(scaler.transform(X[val_mask]))
        print(f"{fold}: scores OOF para {val_mask.sum():,} clips")

    assert not np.isnan(oof_scores).any()
    agg_s, agg_y = aggregate_by_recording(rec, oof_scores, Y)
    print(f"Grabaciones OOF: {len(agg_y):,}")

    # Umbrales OOF vs umbrales fold4 (los de la evaluación final)
    thr_oof = _tune_thresholds(agg_s, agg_y)
    thr_fold4 = json.loads(Path(FINAL_THRESHOLDS).read_text(encoding="utf-8"))

    pred_oof = (agg_s >= thr_oof).astype(int)
    pred_fold4 = (agg_s >= np.array([thr_fold4[sp] for sp in species])).astype(int)
    m_oof = compute_metrics(agg_y, pred_oof, agg_s)
    m_fold4 = compute_metrics(agg_y, pred_fold4, agg_s)

    print(f"\nMacro-F1 en validación (OOF, {len(agg_y):,} grabaciones):")
    print(f"  umbrales fold4 (evaluación final): {m_fold4['f1_macro']:.3f}")
    print(f"  umbrales OOF:                      {m_oof['f1_macro']:.3f}")
    print("\nEspecie        fold4   OOF")
    for sp, t4, to in zip(species, [thr_fold4[s] for s in species], thr_oof):
        flag = "  <-- difiere" if abs(t4 - to) > 0.10 else ""
        print(f"{sp:<12} {t4:>5.2f} {to:>5.2f}{flag}")

    out = {
        "nota": "Umbrales ajustados con predicciones out-of-fold de los 5 folds (17.893 clips; sin test).",
        "macro_f1_val_umbrales_fold4": m_fold4["f1_macro"],
        "macro_f1_val_umbrales_oof": m_oof["f1_macro"],
        "thresholds_oof": {sp: float(t) for sp, t in zip(species, thr_oof)},
        "thresholds_fold4": thr_fold4,
    }
    (OUT_DIR / "oof_thresholds.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGuardado en {OUT_DIR / 'oof_thresholds.json'}")


if __name__ == "__main__":
    main()
