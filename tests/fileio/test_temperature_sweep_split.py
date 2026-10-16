import numpy as np
import pytest
from piblin.data import ExperimentSet

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep


def _load_experiment(datadir, filename):
    measurements = TriosRheoReader().data_from_filepath(
        filepath=datadir.join(filename), create_composite_datasets=True
    )
    return ExperimentSet(measurements.measurements)


def test_temperature_sweep_split(datadir):
    """Split a single sweep into one measurement per temperature."""
    experiment = _load_experiment(datadir, "test_rsag2_trios_temp_sweep.txt")

    processed = (OWChirpAnalysis() + SplitTemperatureSweep())(experiment)

    assert len(processed.measurements) == 19
    assert len(processed.measurements[0].datasets) == 1


def test_temperature_sweep_split_rejects_wrong_step_count(datadir):
    """Reject an explicit step count that disagrees with the input."""
    experiment = _load_experiment(datadir, "test_rsag2_trios_temp_sweep.txt")

    with pytest.raises(ValueError, match="Expected 3 measurements, got 1"):
        (OWChirpAnalysis() + SplitTemperatureSweep(n_steps=3))(experiment)


def test_multi_step_temperature_sweep_preserves_each_temperature(datadir):
    """Keep adjacent 2.5 °C and 5 °C sweeps separate."""
    experiment = _load_experiment(datadir, "test_rsag2_trios_multi_step_temp_sweep.txt")

    processed = (OWChirpAnalysis() + SplitTemperatureSweep(n_steps=3))(experiment)
    states = np.sort(np.asarray(AutomatedMasterCurve()(processed).states, dtype=float))
    expected_states = np.array(
        [
            *range(-40, 25, 5),
            *np.arange(22.5, 52.5, 2.5),
            *range(55, 85, 5),
        ]
    )

    assert len(processed.measurements) == 31
    assert all(
        len(measurement.datasets[0].data_arrays[0]) == 6 for measurement in processed.measurements
    )
    np.testing.assert_allclose(states, expected_states, atol=1.0)
