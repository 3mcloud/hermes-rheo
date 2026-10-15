"""Regression tests for StatesToConditions and MutationNumber.

Locks in the following fixes:

* :class:`~hermes_rheo.transforms.states_to_conditions.StatesToConditions`
  no longer hardcodes ``switch_coordinates(independent_name='temperature')``
  — it preserves whatever independent variable is currently active on the
  dataset.
* :class:`~hermes_rheo.transforms.mutation_number.MutationNumber` has the
  same hardcoding removed from its ``_state_to_condition`` helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from piblin.data import (
    Measurement,
    MeasurementSet,
    OneDimensionalCompositeDataset,
)

from hermes_rheo.transforms.states_to_conditions import StatesToConditions
from hermes_rheo.transforms.mutation_number import MutationNumber


# ---------------------------------------------------------------------------
# StatesToConditions — fixed hardcoded independent-name bug
# ---------------------------------------------------------------------------

def _make_measurement_without_temperature(state_name: str, state_values):
    """Build a Measurement whose dataset does NOT contain a 'temperature' array."""
    x = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    ds = OneDimensionalCompositeDataset(
        data_arrays=[x, np.asarray(state_values, dtype=float)],
        data_array_names=["time", state_name],
        data_array_units=["s", "-"],
        default_independent_name="time",
        default_dependent_name=state_name,
        source="synthetic",
    )
    return Measurement(datasets=[ds])


@pytest.mark.parametrize(
    "method,expected",
    [("average", 2.5), ("first point", 1.0), ("last point", 4.0)],
)
def test_states_to_conditions_no_temperature_column(method, expected):
    """StatesToConditions works even when the dataset has no 'temperature' column."""
    meas = _make_measurement_without_temperature("strain", [1.0, 2.0, 3.0, 4.0])
    transform = StatesToConditions(state="strain", method=method)
    out = transform._apply(meas)
    assert out.conditions["strain"] == pytest.approx(expected)


def test_states_to_conditions_missing_state_raises():
    """StatesToConditions raises when the requested state is not in any dataset."""
    meas = _make_measurement_without_temperature("strain", [1.0, 2.0, 3.0, 4.0])
    transform = StatesToConditions(state="stress", method="average")
    with pytest.raises(ValueError, match="not found"):
        transform._apply(meas)


def test_states_to_conditions_invalid_method_raises():
    """StatesToConditions raises when an unknown aggregation method is passed."""
    meas = _make_measurement_without_temperature("strain", [1.0, 2.0, 3.0, 4.0])
    transform = StatesToConditions(state="strain", method="median")  # not supported
    with pytest.raises(ValueError, match="Invalid method"):
        transform._apply(meas)


# ---------------------------------------------------------------------------
# MutationNumber._state_to_condition — hardcoded-temperature bug removed
# ---------------------------------------------------------------------------

def test_mutation_number_state_to_condition_no_temperature_column():
    """MutationNumber._state_to_condition works on datasets without 'temperature'."""
    x = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    ds = OneDimensionalCompositeDataset(
        data_arrays=[x, np.asarray([10.0, 20.0, 30.0, 40.0])],
        data_array_names=["time", "custom_state"],
        data_array_units=["s", "-"],
        default_independent_name="time",
        default_dependent_name="custom_state",
        source="synthetic",
    )
    m = Measurement(datasets=[ds])
    mset = MeasurementSet([m], merge_redundant=False)

    tf = MutationNumber(state="custom_state", state_sampling="first point")
    tf._state_to_condition(mset)
    assert mset.measurements[0].conditions["custom_state"] == pytest.approx(10.0)
