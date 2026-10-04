import pandas as pd
import pytest

from src.data.audio_loader import list_audio_files, load_birdclef_data, load_metadata


def test_load_birdclef_data_adds_audio_path(tmp_path):
    audio_dir = tmp_path / "train_audio" / "species_a"
    audio_dir.mkdir(parents=True)
    (audio_dir / "bird.ogg").touch()
    pd.DataFrame(
        {"filename": ["species_a/bird.ogg"], "primary_label": ["species_a"]}
    ).to_csv(tmp_path / "train.csv", index=False)

    result = load_birdclef_data(data_dir=tmp_path)

    assert result.loc[0, "primary_label"] == "species_a"
    assert result.loc[0, "audio_path"] == str(audio_dir / "bird.ogg")


def test_load_metadata_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_metadata(data_dir=tmp_path)


def test_list_audio_files_finds_ogg_recursively(tmp_path):
    audio_dir = tmp_path / "species_a"
    audio_dir.mkdir()
    expected = audio_dir / "bird.ogg"
    expected.touch()
    (audio_dir / "notes.txt").touch()

    assert list_audio_files(tmp_path) == [expected]
