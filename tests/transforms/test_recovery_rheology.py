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
    assert out.result.recov_anal_data.shape == (7, 9)
    assert np.all(np.isfinite(out.result.recov_anal_data))
    step = next(m for m in out.measurements if m.conditions["component"] == "step results")
    assert "unrecoverable strain rate" in step.datasets[0].data_array_names


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


def test_step_strain_rates_are_time_derivatives():
    from hermes_rheo.transforms.recovery_rheology import _step_strain_rates

    # [time, total strain, stress, recovery time, unrecoverable strain, recoverable strain]
    t = np.array([1.0, 2.0, 4.0, 8.0])
    data = np.column_stack([t, 3 * t, np.ones(4), np.ones(4), 0.5 * t, 2.5 * t])
    rates = _step_strain_rates(data)
    assert np.allclose(rates, [3.0, 0.5, 2.5])
    assert np.all(np.isnan(_step_strain_rates(data[:1])))


def test_transient_recovered_strain_is_measured_from_end_of_deformation():
    import matplotlib

    matplotlib.use("Agg")
    out = RecoveryRheology(method="step")(_load("mcr702e_rheocompass_startup_shear_recovery.csv"))
    curve = out.result.metadata["transient_curves"][0]
    fig, ax = out.result.plot_transient(logx=False)
    t_r, gamma_rec = ax.lines[0].get_data()
    matplotlib.pyplot.close(fig)
    # The instantaneous recoil before the first recovery sample is included.
    assert t_r[0] == 0.0 and gamma_rec[0] == 0.0
    assert np.isclose(t_r[1], curve["rec_time"][0] - curve["creep_time"][-1])
    assert np.isclose(gamma_rec[1], abs(curve["creep_strain"][-1] - curve["rec_strain"][0]))
    assert gamma_rec[1] > 1.0
