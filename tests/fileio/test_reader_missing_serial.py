import pytest

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def _patched_serial_file(datadir, source_name, serial):
    """Return a fixture copy with a replacement serial number."""
    source = datadir.join(source_name)
    destination = datadir.join(f"patched_{source_name}")
    contents = "\n".join(
        f"Instrument serial number\t{serial}"
        if line.startswith("Instrument serial number")
        else line
        for line in source.read().splitlines()
    )
    destination.write(contents)
    return str(destination)


@pytest.mark.parametrize("serial", ["", "9999-0000"])
def test_reader_accepts_unrecognized_serial(datadir, serial):
    """Parse Creep data without a recognized serial number."""
    filepath = _patched_serial_file(datadir, "test_aresg2_trios_creep.txt", serial)

    measurements = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)

    assert len(measurements.measurements) == 4


def test_owchirp_requires_wave_metadata(datadir):
    """Reject Arbitrary Wave data without wave metadata."""
    filepath = _patched_serial_file(datadir, "test_dhr30_trios_owchirp.txt", "9999-0000")
    measurements = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)

    with pytest.raises(ValueError, match="[Ww]ave metadata"):
        OWChirpAnalysis()(measurements)
