from pathlib import Path

import pytest
from piblin.data import ExperimentSet, MeasurementSet

from hermes_rheo import RheoCompassReader, TriosRheoReader, read_file

DATA = Path(__file__).parent / "test_data"


@pytest.mark.parametrize(
    ("filename", "reader"),
    [
        ("test_aresg2_trios_creep.txt", TriosRheoReader),
        ("test_mcr702e_rheocompass_frequency_sweep_tts.csv", RheoCompassReader),
    ],
)
def test_read_file_infers_reader(filename, reader):
    """Infer the reader from supported file formats."""
    path = DATA / filename
    direct = reader().data_from_filepath(str(path), create_composite_datasets=True)
    inferred = read_file(path, create_composite_datasets=True)

    assert isinstance(inferred, MeasurementSet)
    assert len(inferred.measurements) == len(direct.measurements)


def test_read_file_accepts_reader_instance():
    """Accept an explicitly configured reader instance."""
    result = read_file(
        DATA / "test_aresg2_trios_creep.txt",
        file_reader=TriosRheoReader(),
        create_composite_datasets=True,
    )

    assert len(result.measurements) == 4


def test_read_file_returns_experiment_set_for_consistent_data():
    """Return an ExperimentSet when the measurements are compatible."""
    result = read_file(
        DATA / "test_mcr702e_rheocompass_frequency_sweep_tts.csv",
        file_reader=RheoCompassReader,
        create_composite_datasets=True,
    )

    assert isinstance(result, ExperimentSet)
    dataset = result.measurements[0].datasets[0]
    assert dataset.x_label == "angular frequency(rad/s)"  # units stored without brackets


def test_read_file_rejects_unknown_extension(tmp_path):
    """Require an explicit reader for unsupported extensions."""
    path = tmp_path / "data.xyz"
    path.write_text("")

    with pytest.raises(ValueError, match="pass file_reader explicitly"):
        read_file(path)
