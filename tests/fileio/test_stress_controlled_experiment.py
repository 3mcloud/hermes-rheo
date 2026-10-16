from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_stress_controlled_experiment_loads(datadir):
    """Retain the DHR serial number after processing."""
    data = TriosRheoReader().data_from_filepath(
        str(datadir.join("test_dhr30_trios_owchirp.txt")), create_composite_datasets=True
    )

    processed = OWChirpAnalysis()(data)

    assert processed.details["instrument_serial_number"].startswith("5343")
