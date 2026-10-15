import numpy as np
from piblin.data import ExperimentSet

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis


def test_trios_lims_export_file_single_step(datadir):
    filepath = datadir.join("TRIOS_LIMS_export_single_step.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 3


def test_trios_lims_export_file_multi_step(datadir):
    filepath = datadir.join("TRIOS_LIMS_export_multiple_step.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=False)
    assert len(data.measurements) == 8


def test_temperature_ramp(datadir):
    filepath = datadir.join("test_temp_ramp.txt")
    experiment = ExperimentSet(
        TriosRheoReader()
        .data_from_filepath(filepath=filepath, create_composite_datasets=True)
        .measurements
    )
    pipeline = OWChirpAnalysis()
    processed_experiment = pipeline(experiment)
    assert len(processed_experiment.measurements) == 1
    assert isinstance(processed_experiment.measurements[0].datasets[0].x_values, np.ndarray)
