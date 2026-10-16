import pytest

from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader


@pytest.mark.parametrize(
    ("filename", "expected_measurements"),
    [
        ("test_mcr702e_rheocompass_frequency_sweep_tts.csv", 33),
        ("test_mcr702e_rheocompass_stress_relaxation.csv", 1),
    ],
)
def test_rheocompass_csv_loads(datadir, filename, expected_measurements):
    """Load RheoCompass CSV measurements."""
    data = RheoCompassReader().data_from_filepath(
        str(datadir.join(filename)), create_composite_datasets=False
    )

    assert len(data.measurements) == expected_measurements
