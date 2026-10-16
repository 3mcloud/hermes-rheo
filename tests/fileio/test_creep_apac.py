from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_creep_apac(datadir):
    """Process the Stress Control and Creep steps."""
    filepath = datadir.join("test_aresg2_trios_creep.txt")
    measurements = TriosRheoReader().data_from_filepath(
        filepath=filepath, create_composite_datasets=True
    )

    processed = OWChirpAnalysis()(measurements)

    assert [measurement.conditions["method"] for measurement in processed.measurements] == [
        "Stress Control - 1",
        "Creep - 2",
        "Creep - 3",
        "Creep - 4",
    ]
