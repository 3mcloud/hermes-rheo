"""Tests for :class:`~hermes_rheo.transforms.kymograph.Kymograph`.

Covers the primary use case (all three axes come from dataset columns)
and the condition-fallback path used to build a Mutation Number
kymograph (frequency lives in ``measurement.conditions`` rather than in
the dataset).
"""
from __future__ import annotations

import numpy as np

from piblin.data import (
    Measurement,
    MeasurementSet,
    OneDimensionalCompositeDataset,
)

from hermes_rheo.transforms.kymograph import Kymograph, KymographResult


def _make_torch_like_measurement(temperature: float) -> Measurement:
    """One OWChirp-like measurement at a fixed temperature — carries all
    three axes as dataset columns."""
    omega = np.logspace(0, 2, 20)                    # angular frequency, rad/s
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
    """One MutationNumber-like measurement — mutation values vs time at a
    single frequency; the frequency is stored in ``conditions`` rather
    than as a dataset column."""
    t = np.linspace(0.0, 500.0, 30)
    mu = 0.1 * np.sin(2 * np.pi * t / 500.0) + 0.01 * angular_frequency
    ds = OneDimensionalCompositeDataset(
        data_arrays=[t, mu],
        data_array_names=["time", "mutation number"],
        data_array_units=["s", "a.u."],
        default_independent_name="time",
        default_dependent_name="mutation number",
        source="synthetic mutation",
    )
    return Measurement(
        datasets=[ds],
        conditions={"Angular Frequency": float(angular_frequency)},
    )


def test_kymograph_all_axes_from_dataset_columns():
    """Standard TORCh path — every axis is a dataset column."""
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
    """Mutation Number path — ``angular frequency`` is not a dataset
    column but is present in every measurement's ``conditions`` dict.
    The transform must broadcast that scalar across the time axis."""
    experiment = MeasurementSet(
        measurements=[
            _make_mutation_like_measurement(w) for w in (0.1, 1.0, 10.0, 100.0)
        ]
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
    """`MutationNumber` produces plain :class:`OneDimensionalDataset`
    measurements (no `data_array_names` attribute).  `Kymograph` must
    handle these via the independent / dependent fallback."""
    from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import (
        OneDimensionalDataset,
    )

    def _make_1d_measurement(angular_frequency: float) -> Measurement:
        t = np.linspace(0.0, 300.0, 20)
        mu = 0.05 + 0.01 * angular_frequency + 0.001 * t
        ds = OneDimensionalDataset(
            dependent_variable_data=mu,
            dependent_variable_names=["mutation number"],
            dependent_variable_units=["a.u."],
            independent_variable_data=[t],
            independent_variable_names=["time"],
            independent_variable_units=["s"],
            source="synthetic 1D",
        )
        return Measurement(
            datasets=[ds],
            conditions={"Angular Frequency": float(angular_frequency)},
        )

    experiment = MeasurementSet(
        measurements=[_make_1d_measurement(w) for w in (0.5, 5.0, 50.0)]
    )
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
