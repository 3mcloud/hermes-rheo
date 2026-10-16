import numpy as np
import pytest
from piblin.data import Measurement, OneDimensionalCompositeDataset

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve
from hermes_rheo.transforms.owchirp_analysis import OWChirpAnalysis
from hermes_rheo.transforms.split_temperature_sweep import SplitTemperatureSweep


@pytest.mark.parametrize(
    ("filename", "expected_shifts"),
    [
        ("test_aresg2_trios_frequency_sweep_tts.txt", 20),
        ("test_aresg2_trios_owchirp_tts.txt", 10),
    ],
)
def test_tts_builds_master_curve(datadir, filename, expected_shifts):
    """Build a storage-modulus master curve."""
    data = TriosRheoReader().data_from_filepath(
        str(datadir.join(filename)), create_composite_datasets=True
    )

    master_curve = (
        OWChirpAnalysis()
        + AutomatedMasterCurve(
            state="temperature",
            y="storage modulus",
            vertical_shift=False,
            reverse_data=False,
        )
    )(data)

    assert len(master_curve.hparams[0]) == expected_shifts


def test_tts_sorts_non_monotonic_temperatures(datadir):
    """Sort temperature-sweep states before building a master curve."""
    data = TriosRheoReader().data_from_filepath(
        str(datadir.join("test_dhr30_trios_temp_sweep_tts.txt")),
        create_composite_datasets=True,
    )

    master_curve = (
        OWChirpAnalysis()
        + SplitTemperatureSweep(n_steps=5)
        + AutomatedMasterCurve(
            state="Temperature",
            state_mode="average",
            y="complex modulus",
            vertical_shift=False,
            reverse_data=False,
            sort_states=True,
        )
    )(data)

    assert len(master_curve.hparams[0]) == 26


def test_tts_returns_measurement_with_shift_data(datadir):
    """Expose shifted curves and shift factors as datasets."""
    data = TriosRheoReader().data_from_filepath(
        str(datadir.join("test_aresg2_trios_frequency_sweep_tts.txt")),
        create_composite_datasets=True,
    )
    master_curve = (
        OWChirpAnalysis()
        + AutomatedMasterCurve(state="temperature", y="storage modulus", vertical_shift=False)
    )(data)
    shifted, shifts = master_curve.datasets

    assert isinstance(master_curve, Measurement)
    assert all(
        isinstance(dataset, OneDimensionalCompositeDataset) for dataset in master_curve.datasets
    )
    assert len(shifted.data_arrays[0]) == len(shifted.data_arrays[1])
    assert len(shifts.data_arrays[0]) == 20
    assert master_curve.master_curve is not None
    master_curve.change_ref(master_curve.states[0])
    assert np.isclose(np.asarray(master_curve.datasets[1].data_arrays[1])[0], 1.0)
    assert master_curve.details["reference_state"] == master_curve.states[0]
