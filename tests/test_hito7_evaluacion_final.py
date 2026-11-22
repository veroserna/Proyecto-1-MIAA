"""Prueba sintética de la evaluación final: datos artificiales, una pasada por test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.hito7_evaluacion_final import (
    _modal_best_params, _tune_thresholds, run_evaluacion_final,
)


def _synthetic_features_csv(tmp_path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    feature_cols = ["mfcc_00_base_mean", "mfcc_00_base_std", "zcr_mean", "centroid_mean"]
    species = ["spA", "spB", "spC"]
    rows = []
    clip = 0
    for split, partition, n in [
        ("fold0", "train", 60), ("fold1", "train", 60), ("fold2", "train", 60),
        ("fold3", "train", 60), ("fold4", "val", 60), ("test", "test", 60),
    ]:
        for i in range(n):
            sp = species[i % len(species)]
            signal = rng.normal(0, 1, len(feature_cols))
            signal[species.index(sp) % len(feature_cols)] += 3.0  # señal separable
            row = dict(zip(feature_cols, signal))
            row.update({
                "primary_label": sp,
                "secondary": "",
                "partition": partition,
                "split": split,
                "recording_id": f"rec_{split}_{i // 5}",
                "author": f"auth_{split}_{i % 7}",
            })
            rows.append(row)
            clip += 1
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _synthetic_hito6_config(tmp_path: Path) -> Path:
    cfg = {
        "best_params_by_fold": {
            "xgboost": {f"fold{i}": {
                "estimator__n_estimators": 50,
                "estimator__max_depth": 3,
                "estimator__learning_rate": 0.1,
                "estimator__subsample": 1.0,
                "estimator__colsample_bytree": 1.0,
                "estimator__min_child_weight": 1,
                "estimator__scale_pos_weight": 1,
            } for i in range(5)}
        }
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_modal_best_params():
    cfg = {"best_params_by_fold": {"xgboost": {
        "fold0": {"a": 1, "b": 2},
        "fold1": {"a": 1, "b": 3},
        "fold2": {"a": 2, "b": 2},
    }}}
    modal = _modal_best_params(cfg)
    assert modal == {"a": 1, "b": 2}


def test_tune_thresholds_per_species():
    rng = np.random.default_rng(1)
    y = (rng.random((200, 3)) > 0.5).astype(int)
    scores = y * 0.6 + rng.random((200, 3)) * 0.3
    t = _tune_thresholds(scores, y)
    assert t.shape == (3,)
    assert np.all((t >= 0.05) & (t <= 0.95))


def test_run_evaluacion_final_synthetic(tmp_path, monkeypatch):
    import src.training.hito7_evaluacion_final as mod
    csv_path = _synthetic_features_csv(tmp_path)
    cfg_path = _synthetic_hito6_config(tmp_path)
    out_dir = tmp_path / "results"
    monkeypatch.setattr(
        "src.training.hito7_evaluacion_final.load_feature_columns",
        lambda: ["mfcc_00_base_mean", "mfcc_00_base_std", "zcr_mean", "centroid_mean"],
    )
    result = run_evaluacion_final(
        features_csv=csv_path, hito6_config=cfg_path, out_dir=out_dir,
        threshold_source="fold4",
    )
    assert 0.0 <= result["test_metrics"]["f1_macro"] <= 1.0
    assert (out_dir / "test_metrics.json").exists()
    assert (out_dir / "thresholds.json").exists()
    assert (out_dir / "per_class_test.csv").exists()
    assert (out_dir / "figures" / "confusion_matrices_grid.png").exists()
    # test se evaluó una sola vez y quedó registrado
    saved = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["test_used_once"] is True
    assert len(result["thresholds"]) == 3
