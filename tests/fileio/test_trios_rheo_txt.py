import pytest

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader


@pytest.mark.parametrize(
    ("filename", "expected_measurements"),
    [
        ("test_aresg2_trios_lims_single_step.txt", 3),
        ("test_aresg2_trios_lims_multiple_step.txt", 8),
    ],
)
def test_trios_lims_export_loads(datadir, filename, expected_measurements):
    """Load TRIOS LIMS exports with the expected steps."""
    data = TriosRheoReader().data_from_filepath(
        str(datadir.join(filename)), create_composite_datasets=False
    )

    assert len(data.measurements) == expected_measurements
