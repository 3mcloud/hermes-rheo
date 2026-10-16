"""Tests for recovery-rheology outputs."""

from pathlib import Path

import numpy as np
import pytest
from piblin.data.data_collections.measurement_set import MeasurementSet

from hermes_rheo import read_file
from hermes_rheo.file_readers.rheo_compass_csv import RheoCompassReader
from hermes_rheo.transforms.recovery_rheology import (
    RecoveryRheology,
    RecoveryRheologyMeasurementSet,
)

DATA = Path(__file__).parents[2] / "tutorial_notebooks" / "tutorial_data"


def _load(name: str):
    path = DATA / name
    if not path.exists():
        pytest.skip("tutorial data missing")
    return read_file(str(path), file_reader=RheoCompassReader, create_composite_dataset=True)


def test_oscillatory_returns_piblin_measurements():
    out = RecoveryRheology(method="oscillatory", omega=0.316)(
        _load("mcr702e_rheocompass_recovery_oscillatory.csv")
    )
    assert isinstance(out, RecoveryRheologyMeasurementSet)
    assert isinstance(out, MeasurementSet)
    assert len(out.measurements) == 2
    for m in out.measurements:
        ds = m.datasets[0]
        assert ds.dependent_variable_names
        assert m.details["method"] == "oscillatory"
        assert np.isclose(m.details["angular_frequency"], 0.316)
    paired = next(
        m for m in out.measurements if m.conditions["component"] == "paired oscillation data"
    )
    assert paired.datasets[0].dependent_variable_names[-1] == "stress"
    assert np.allclose(paired.datasets[0].dependent_variable_data, out.result.recov_anal_data[:, 4])


@pytest.mark.parametrize("protocol", ["step_strain", "creep", "startup_shear"])
def test_step_returns_piblin_measurements(protocol):
    out = RecoveryRheology(method="step")(_load(f"mcr702e_rheocompass_{protocol}_recovery.csv"))
    assert isinstance(out, MeasurementSet)
    assert len(out.measurements) > 1
    assert all(m.details["method"] == "step" for m in out.measurements)
    # 7 step durations, each run in both directions; the start-up shear file
    # has one point flagged invalid, which must not drop its step.
    assert out.result.recov_anal_data.shape[0] == 7
    assert np.all(np.isfinite(out.result.recov_anal_data))


def test_step_blocks_are_paired_by_duration():
    from hermes_rheo.transforms.recovery_rheology import _average_by_duration

    # [sign, time, strain, stress, recovery time, unrecoverable strain]
    pos = np.array([[1, 2.0, 1.0, 10, 100, 0.2], [1, 5.0, 2.0, 10, 100, 0.4]])
    neg = np.array([[-1, 2.0, -3.0, -10, 100, -0.4], [-1, 8.0, -4.0, -10, 100, -1.0]])
    out = _average_by_duration(pos, neg)
    assert np.allclose(out[:, 0], [2.0, 5.0, 8.0])
    assert np.allclose(out[:, 1], [2.0, 2.0, 4.0])
    assert np.allclose(out[:, 4], [0.3, 0.4, 1.0])
    assert np.allclose(out[:, 5], out[:, 1] - out[:, 4])
