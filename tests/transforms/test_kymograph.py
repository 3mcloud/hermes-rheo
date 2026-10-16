"""Tests for kymograph generation."""

from __future__ import annotations

import numpy as np
from piblin.data import (
    Measurement,
    MeasurementSet,
    OneDimensionalCompositeDataset,
)

from hermes_rheo.transforms.kymograph import Kymograph, KymographResult


def _make_torch_like_measurement(temperature: float) -> Measurement:
    """Build a measurement with all axes stored in a dataset."""
    omega = np.logspace(0, 2, 20)  # angular frequency, rad/s
    g_prime = 1000.0 + 10.0 * omega * (1.0 + 0.01 * temperature)
    temp_arr = np.full_like(omega, float(temperature))
    ds = OneDimensionalCompositeDataset(
        data_arrays=[omega, g_prime, temp_arr],
        data_array_names=["angular frequency", "storage modulus", "temperature"],
        data_array_units=["rad/s", "Pa", "°C"],
        default_independent_name="angular frequency",
        default_dependent_name="storage modulus",
        source="synthetic torch",
    )
    return Measurement(datasets=[ds])


def _make_mutation_like_measurement(angular_frequency: float) -> Measurement:
    """Build a measurement with frequency stored as a condition."""
    t = np.linspace(0.0, 500.0, 30)
    mu = 0.1 * np.sin(2 * np.pi * t / 500.0) + 0.01 * angular_frequency
    ds = OneDimensionalCompositeDataset(
        data_arrays=[t, mu],
        data_array_names=["time", "mutation number"],
        data_array_units=["s", ""],
        default_independent_name="time",
        default_dependent_name="mutation number",
        source="synthetic mutation",
    )
    return Measurement(
        datasets=[ds],
        conditions={"Angular Frequency": float(angular_frequency)},
    )


def test_kymograph_all_axes_from_dataset_columns():
    """Grid datasets that contain every requested axis."""
    experiment = MeasurementSet(
        measurements=[_make_torch_like_measurement(T) for T in (25, 50, 75, 100, 125)]
    )
    result = Kymograph(
        quantity="storage modulus",
        x="temperature",
        y="angular frequency",
        n_x=40,
        n_y=20,
    )(experiment)

    assert isinstance(result, KymographResult)
    assert result.z_values.shape == (40, 20)
    assert result.quantity == "storage modulus"
    assert result.z_unit == "Pa"
    assert result.x_unit == "°C"
    assert result.y_unit == "rad/s"


def test_kymograph_axis_falls_back_to_condition():
    """Grid an axis supplied by each measurement condition."""
    experiment = MeasurementSet(
        measurements=[_make_mutation_like_measurement(w) for w in (0.1, 1.0, 10.0, 100.0)]
    )
    result = Kymograph(
        quantity="mutation number",
        x="time",
        y="angular frequency",
        n_x=30,
        n_y=15,
        log_y=True,
    )(experiment)

    assert isinstance(result, KymographResult)
    assert result.z_values.shape == (30, 15)
    assert result.quantity == "mutation number"
    # Four distinct frequencies from the four measurement conditions.
    assert np.unique(result.raw_y).size == 4
    assert np.allclose(sorted(np.unique(result.raw_y)), [0.1, 1.0, 10.0, 100.0])


def test_kymograph_supports_plain_one_dimensional_dataset():
    """Plain :class:`OneDimensionalDataset` inputs (no `data_array_names`) are gridded."""
    from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import (
        OneDimensionalDataset,
    )

    def _make_1d_measurement(angular_frequency: float) -> Measurement:
        t = np.linspace(0.0, 300.0, 20)
        mu = 0.05 + 0.01 * angular_frequency + 0.001 * t
        ds = OneDimensionalDataset(
            dependent_variable_data=mu,
            dependent_variable_names=["mutation number"],
            dependent_variable_units=[""],
            independent_variable_data=[t],
            independent_variable_names=["time"],
            independent_variable_units=["s"],
            source="synthetic 1D",
        )
        return Measurement(
            datasets=[ds],
            conditions={"Angular Frequency": float(angular_frequency)},
        )

    experiment = MeasurementSet(measurements=[_make_1d_measurement(w) for w in (0.5, 5.0, 50.0)])
    result = Kymograph(
        quantity="mutation number",
        x="time",
        y="angular frequency",
        n_x=25,
        n_y=10,
        log_y=True,
    )(experiment)

    assert isinstance(result, KymographResult)
    assert result.z_values.shape == (25, 10)
    assert np.unique(result.raw_y).size == 3


def test_kymograph_carries_source_conditions_and_details():
    """Shared source metadata and gridding parameters propagate to the result."""
    ms = []
    for t in (20.0, 40.0, 60.0):
        m = _make_torch_like_measurement(t)
        m = Measurement(
            datasets=m.datasets,
            conditions={"sample": "A", "temperature setpoint": t},
            details={"operator": "x"},
        )
        ms.append(m)
    result = Kymograph(quantity="storage modulus", x="temperature", y="angular frequency")(
        MeasurementSet(ms)
    )
    assert result.conditions == {"sample": "A"}
    assert result.details["operator"] == "x"
    assert result.details["quantity"] == "storage modulus"
    assert result.details["n_source_measurements"] == 3
    assert "hermes-rheo" in result.details["software"]
    meas = result.to_measurement()
    assert isinstance(meas, Measurement)
    assert meas.datasets[0] is result
    assert meas.conditions["sample"] == "A"


def test_kymograph_swapped_axes_grid_in_log_x():
    """Frequency on x with ``log_x`` gives a log-spaced x grid and a log x-axis."""
    experiment = MeasurementSet(
        measurements=[_make_torch_like_measurement(T) for T in (25, 50, 75, 100, 125)]
    )
    result = Kymograph(
        quantity="storage modulus",
        x="angular frequency",
        y="temperature",
        log_x=True,
        log_y=False,
        n_x=20,
        n_y=10,
    )(experiment)

    assert (result.x_unit, result.y_unit) == ("rad/s", "°C")
    assert np.allclose(np.diff(np.log10(result.x_values)), np.log10(100) / 19)
    assert np.allclose(np.diff(result.y_values), 100 / 9)
    _, ax = result.plot()
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "linear")


def test_kymograph_plot_colour_limits():
    """``vmin``/``vmax`` set the colour scale, linear or logarithmic."""
    experiment = MeasurementSet(
        measurements=[_make_torch_like_measurement(T) for T in (25, 50, 75, 100, 125)]
    )
    result = Kymograph(quantity="storage modulus", n_x=10, n_y=10)(experiment)

    _, ax = result.plot(vmin=1100.0, vmax=1500.0)
    assert ax.collections[0].get_clim() == (1100.0, 1500.0)
    _, ax = result.plot(log_z=True, vmax=1500.0)
    assert ax.collections[0].norm.vmin == result.z_values.min()
    assert ax.collections[0].norm.vmax == 1500.0
