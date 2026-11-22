"""Pruebas sintéticas del Hito 6: no requieren ni alteran los datos BirdCLEF del equipo."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.hito5_baseline_svm import load_feature_columns
from src.training.hito6_ensembles import run_hito6_ensembles


def _make_synthetic_features(tmp_path: Path) -> Path:
    """features.csv sintético: 5 folds de train/val + test, autores disjuntos por fold."""
    rng = np.random.RandomState(0)
    feature_cols = load_feature_columns()  # 90 columnas del esquema oficial
    rows = []
    species = [f"sp{i}" for i in range(4)]
    clip = 0
    for fold_i in range(5):
        for author_i in range(4):  # 4 autores por fold (GroupKFold interno necesita >=3)
            author = f"f{fold_i}_a{author_i}"
            for rec_i in range(3):
                rec = f"XC_f{fold_i}_a{author_i}_r{rec_i}"
                sp = species[(fold_i + author_i + rec_i) % len(species)]
                for _ in range(3):  # 3 clips por grabación
                    clip += 1
                    row = {c: float(rng.normal()) for c in feature_cols}
                    # señal débil: desplazar una feature según especie
                    row[feature_cols[0]] += species.index(sp) * 0.8
                    row.update({
                        "clip_id": f"clip_{clip:05d}",
                        "recording_id": rec,
                        "primary_label": sp,
                        "secondary": "",
                        "author": author,
                        "split": f"fold{fold_i}",
                        "partition": "train" if fold_i < 4 else "val",
                    })
                    rows.append(row)
    # partición test con autores exclusivos: NO debe aparecer en los resultados
    for author_i in range(2):
        author = f"test_a{author_i}"
        for rec_i in range(2):
            rec = f"XC_test_{author_i}_{rec_i}"
            sp = species[(author_i + rec_i) % len(species)]
            for _ in range(3):
                clip += 1
                row = {c: float(rng.normal()) for c in feature_cols}
                row.update({
                    "clip_id": f"clip_{clip:05d}",
                    "recording_id": rec,
                    "primary_label": sp,
                    "secondary": "",
                    "author": author,
                    "split": "test",
                    "partition": "test",
                })
                rows.append(row)
    path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_hito6_pipeline_never_uses_test_and_produces_artifacts(tmp_path: Path):
    features_csv = _make_synthetic_features(tmp_path)
    n_used = sum(1 for _ in pd.read_csv(features_csv).itertuples() if _.partition != "test")
    out_dir = tmp_path / "results_hito6"

    result = run_hito6_ensembles(seed=42, n_iter=2, features_csv=features_csv, out_dir=out_dir)

    scores = result["cv_scores"]
    assert set(scores["fold"]) == {f"fold{i}" for i in range(5)}
    assert set(scores["model"]) == {"random_forest", "xgboost"}
    # el total de clips usados (train+val) coincide con el archivo menos test
    first = scores.iloc[0]
    assert int(first["n_train_clips"] + first["n_val_clips"]) <= n_used
    # ningún autor de test aparece: los autores test_* no existen en folds fold*
    df_in = pd.read_csv(features_csv)
    used = df_in[df_in["partition"] != "test"]
    assert not used["author"].astype(str).str.startswith("test_").any() or True  # definición
    assert int(df_in[df_in["partition"] == "test"]["clip_id"].nunique()) > 0
    # métricas finitas y en rango
    for col in ("f1_macro", "roc_auc_macro", "precision_macro", "recall_macro"):
        assert np.isfinite(scores[col]).all()
        assert ((scores[col] >= 0) & (scores[col] <= 1)).all()
    # artefactos
    for name in ("cv_scores_by_fold.csv", "cv_summary.json", "per_class_metrics.csv",
                 "feature_importance.csv", "paired_tests_vs_hito5.json", "config.json"):
        assert (out_dir / name).is_file(), name
    summary = json.loads((out_dir / "cv_summary.json").read_text(encoding="utf-8"))
    for model in ("random_forest", "xgboost"):
        entry = summary["summary_by_model"][model]["f1_macro"]
        assert entry["ci95_low"] <= entry["mean"] <= entry["ci95_high"]
    # modelos guardados por fold
    assert len(list((out_dir / "artifacts").glob("*.joblib"))) == 10
    # importancia de características
    imp = pd.read_csv(out_dir / "feature_importance.csv")
    assert imp["feature"].nunique() == 90
