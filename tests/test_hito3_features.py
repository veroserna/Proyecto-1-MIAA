"""Pruebas sintéticas: no requieren ni alteran los datos BirdCLEF del equipo."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from src.data.audio_features import (
    DEFAULT_CONFIG, extract_features, extract_features_from_wav, feature_column_names,
)
from src.training.preprocessing.hito3_feature_pipeline import (
    read_and_audit_manifest, run_hito3, resolve_clip_path,
)


def _wave(freq: float, phase: float = 0.0) -> np.ndarray:
    n = DEFAULT_CONFIG.sr * DEFAULT_CONFIG.duration_s
    t = np.arange(n, dtype=np.float32) / DEFAULT_CONFIG.sr
    y = 0.4 * np.sin(2 * np.pi * freq * t + phase)
    y += 0.15 * np.sin(2 * np.pi * (2 * freq) * t)
    return y.astype(np.float32)


def test_90_features_finite_and_reproducible():
    y = _wave(540)
    values1, qa1 = extract_features(y, DEFAULT_CONFIG.sr)
    values2, qa2 = extract_features(y, DEFAULT_CONFIG.sr)
    assert len(values1) == 90
    assert list(values1) == feature_column_names()
    assert qa1 == qa2
    assert qa1["n_spectral_frames"] == 497
    assert np.isfinite(list(values1.values())).all()
    assert np.allclose(list(values1.values()), list(values2.values()), rtol=1e-10, atol=1e-10)


def test_silence_is_not_converted_to_fabricated_features():
    with pytest.raises(ValueError, match="silencioso"):
        extract_features(np.zeros(160000, dtype=np.float32), 32000)


def test_invalid_audio_format_is_rejected(tmp_path: Path):
    wavpath = tmp_path / "bad.wav"
    wavfile.write(wavpath, 16000, np.zeros(80000, dtype=np.int16))
    with pytest.raises(ValueError, match="Frecuencia"):
        extract_features_from_wav(wavpath)


def _make_fixture(tmp_path: Path) -> Path:
    processed = tmp_path / "src" / "data" / "processed"
    processed.mkdir(parents=True)
    rows = []
    partitions = ["train"] * 5 + ["val"] * 2 + ["test"] * 2
    for i, partition in enumerate(partitions):
        species = "ave1" if i % 2 == 0 else "ave2"
        name = f"{species}_XC{i:06d}_c00"
        rel = f"clips_5s/{partition}/{species}/{name}.wav"
        path = processed / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        y = _wave(450 + i * 70, phase=i/10)
        wavfile.write(path, 32000, np.round(y * 32767).astype(np.int16))
        rows.append({
            "clip_id": name, "clip_path": rel, "recording_id": f"XC{i:06d}",
            "filename": f"{species}/XC{i:06d}.ogg", "author": f"autor{i}",
            "primary_label": species, "labels": species,
            "split": "fold0" if partition == "train" else "fold4" if partition == "val" else "test",
            "partition": partition, "sr": 32000, "n_samples": 160000,
            "quality_tier": "high", "source_win_idx": 0,
        })
    manifest = processed / "clips_metadata.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_complete_pipeline_never_fits_on_val_test(tmp_path: Path):
    manifest = _make_fixture(tmp_path)
    out_features = tmp_path / "src" / "data" / "features"
    out_results = tmp_path / "experiments" / "results" / "hito3"
    summary = run_hito3(manifest_path=manifest, feature_dir=out_features,
                        results_dir=out_results)
    assert summary["n_processed"] == 9
    assert summary["n_features"] == 90
    assert summary["n_samples_seen_scaler"] == 5
    assert summary["n_samples_seen_pca"] == 5
    assert summary["pca_scoring_partitions"] == ["train", "val"]
    assert summary["test_used_to_fit_transformations"] is False
    raw = pd.read_csv(out_features / "features_raw.csv")
    scaled = pd.read_csv(out_features / "features.csv")
    assert raw.shape == scaled.shape
    assert raw["clip_id"].tolist() == scaled["clip_id"].tolist()
    assert scaled["clip_id"].nunique() == 9
    train = scaled[scaled["partition"] == "train"]
    cols = feature_column_names()
    assert np.allclose(train[cols].mean().to_numpy(), 0, atol=2e-4)
    assert np.isfinite(scaled[cols].to_numpy()).all()
    pca = pd.read_csv(out_results / "pca_scores_train_val.csv")
    assert len(pca) == 7
    assert set(pca["partition"]) == {"train", "val"}
    assert (out_results / "figures" / "pca_2d_train.png").is_file()
    assert (out_results / "figures" / "pca_3d_train.png").is_file()


def test_audit_detects_author_leakage(tmp_path: Path):
    manifest = _make_fixture(tmp_path)
    df = pd.read_csv(manifest)
    df.loc[5, "author"] = df.loc[0, "author"]
    df.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Fuga"):
        read_and_audit_manifest(manifest)


def test_dangerous_paths_are_not_allowed(tmp_path: Path):
    with pytest.raises(ValueError, match="Ruta"):
        resolve_clip_path(tmp_path, "../archivo.wav")
