from pathlib import Path

import pytest
from piblin.data import ExperimentSet, MeasurementSet

from hermes_rheo import RheoCompassReader, TriosRheoReader, read_file

DATA = Path(__file__).parent / "test_data"


@pytest.mark.parametrize(
    ("filename", "reader"),
    [
        ("test_creep_apac.txt", TriosRheoReader),
        ("test_antonpaar_702e_tts.csv", RheoCompassReader),
    ],
)
def test_read_file_matches_reader(filename, reader):
    path = DATA / filename
    direct = reader().data_from_filepath(str(path), create_composite_datasets=True)

    explicit = read_file(path, file_reader=reader, create_composite_datasets=True)
    inferred = read_file(path, create_composite_datasets=True)

    for collection in (explicit, inferred):
        assert isinstance(collection, MeasurementSet)
        assert len(collection.measurements) == len(direct.measurements)


def test_read_file_accepts_reader_instance():
    path = DATA / "test_creep_apac.txt"
    result = read_file(path, file_reader=TriosRheoReader(), create_composite_datasets=True)
    assert len(result.measurements) > 0


def test_read_file_returns_experiment_set_for_consistent_data():
    path = DATA / "test_antonpaar_702e_tts.csv"
    result = read_file(path, file_reader=RheoCompassReader, create_composite_datasets=True)
    assert isinstance(result, ExperimentSet)


def test_read_file_unknown_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("")
    with pytest.raises(ValueError, match="pass file_reader explicitly"):
        read_file(path)
