from piblin.data import ExperimentSet

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_amplitude_sweep_loads(datadir):
    """Load and process an amplitude sweep."""
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join("test_aresg2_trios_amplitude_sweep.txt"),
        create_composite_datasets=True,
    )

    processed = OWChirpAnalysis()(ExperimentSet(measurements.measurements))

    assert len(processed.measurements) == 1
    assert processed.measurements[0].conditions["method"] == "Amplitude - 1"
