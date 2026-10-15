from piblin.data import ExperimentSet
from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_amplitude_apac(datadir):
    filepath = datadir.join("test_amplitude_apac.txt")
    experiment = ExperimentSet(
        TriosRheoReader()
        .data_from_filepath(filepath=filepath, create_composite_datasets=True)
        .measurements
    )

    processed = OWChirpAnalysis()(experiment)

    assert len(processed.measurements) == 1

    assert processed.measurements[0].conditions["method"] == "Amplitude - 1"
