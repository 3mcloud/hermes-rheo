"""Tests for the mutation number of correlated OWCh windows and its plot."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from piblin.data import Measurement, MeasurementSet, OneDimensionalCompositeDataset  # noqa: E402

from hermes_rheo.transforms.mutation_number import MutationNumber  # noqa: E402

OMEGA = np.array([1.0, 10.0, 100.0])  # rad/s
START_TIMES = np.array([0.0, 20.0, 40.0, 140.0, 160.0])  # s; 100 s dwell after 40 s
DECAY_RATE = 0.01  # 1/s, so Mu = -T * DECAY_RATE = -0.1 for T = 10 s


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _dashed(ax) -> list:
    """Dashed guide lines (errorbar caps are lines too)."""
    return [line for line in ax.lines if line.get_linestyle() == "--"]


def _window(start: float) -> Measurement:
    """A correlated window whose |G*| decays as exp(-DECAY_RATE * t)."""
    time = start + np.linspace(0.0, 10.0, OMEGA.size)
    dataset = OneDimensionalCompositeDataset(
        data_arrays=[
            OMEGA,
            time,
            np.full_like(OMEGA, 25.0 + start / 10.0),
            1e5 * OMEGA**0.5 * np.exp(-DECAY_RATE * start),
        ],
        data_array_names=["angular frequency", "time", "temperature", "complex modulus"],
        data_array_units=["rad/s", "s", "°C", "Pa"],
        default_independent_name="angular frequency",
        default_dependent_name="complex modulus",
        source="synthetic TORCh",
    )
    return Measurement(datasets=[dataset], details={"Step name": f"Correlated - {start:g}"})


@pytest.fixture
def ramp() -> MeasurementSet:
    return MeasurementSet([_window(t) for t in START_TIMES], merge_redundant=False)


def test_correlated_mutation_number(ramp):
    """One series per frequency with time, Mu, temperature and frequency columns."""
    result = MutationNumber(owch_length=10.0)(ramp)

    assert [m.conditions["Angular Frequency"] for m in result.measurements] == list(OMEGA)
    dataset = result.measurements[0].datasets[0]
    assert dataset.data_array_names == [
        "time",
        "mutation number",
        "temperature",
        "angular frequency",
    ]
    assert dataset.data_array_units == ["s", "", "°C", "rad/s"]
    assert np.allclose(dataset.data_arrays[0], START_TIMES[1:])
    assert np.allclose(dataset.data_arrays[1], -0.1)
    assert np.allclose(dataset.data_arrays[2], 25.0 + START_TIMES[1:] / 10.0)


def test_absolute_and_dt_threshold(ramp):
    """``absolute`` gives |Mu|; ``dt_threshold`` drops the pair across the dwell."""
    result = MutationNumber(owch_length=10.0, absolute=True, dt_threshold=50.0)(ramp)

    dataset = result.measurements[0].datasets[0]
    assert np.allclose(dataset.data_arrays[0], [20.0, 40.0, 160.0])
    assert np.allclose(dataset.data_arrays[1], 0.1)
    details = result.measurements[0].details
    assert details["absolute"] is True
    assert details["dt_threshold"] == 50.0
    assert details["n_points"] == 3


def test_visualize_labels_threshold_and_colorbar(ramp):
    """Both panels carry Mu labels and the threshold; colouring adds a temperature bar."""
    result = MutationNumber(owch_length=10.0, absolute=True)(ramp)
    fig, (ax_all, ax_mean) = result.visualize(
        show_all_frequencies=True, color_by="temperature", cmap="coolwarm"
    )

    assert ax_all.get_ylabel() == r"Mutation Number $|\mathrm{Mu}|$"
    assert "Frequency-averaged" in ax_mean.get_ylabel()
    assert r"\overline" in ax_mean.get_ylabel()
    assert ax_mean.get_xlabel() == "time (s)"
    for ax in (ax_all, ax_mean):
        assert [line.get_ydata()[0] for line in _dashed(ax)] == [0.15]
    assert fig.axes[-1].get_ylabel() == "temperature (°C)"


def test_visualize_without_threshold(ramp):
    """``threshold=None`` omits the guide line; the default colours by time."""
    fig, ax = MutationNumber(owch_length=10.0)(ramp).visualize(threshold=None)

    assert not _dashed(ax)
    assert len(fig.axes) == 1
    assert ax.get_ylabel() == r"Frequency-averaged Mutation Number $\overline{\mathrm{Mu}}$"
