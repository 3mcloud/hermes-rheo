"""Tests for state-to-condition transforms."""

from __future__ import annotations

import numpy as np
import pytest
from piblin.data import Measurement, MeasurementSet, OneDimensionalCompositeDataset

from hermes_rheo.transforms.mutation_number import MutationNumber
from hermes_rheo.transforms.states_to_conditions import StatesToConditions


def _make_measurement(state_name: str, state_values) -> Measurement:
    """Build a measurement with a non-temperature state array."""
    time = np.asarray([0.0, 1.0, 2.0, 3.0])
    dataset = OneDimensionalCompositeDataset(
        data_arrays=[time, np.asarray(state_values)],
        data_array_names=["time", state_name],
        data_array_units=["s", "-"],
        default_independent_name="time",
        default_dependent_name=state_name,
        source="synthetic",
    )
    return Measurement(datasets=[dataset])


@pytest.mark.parametrize(
    ("method", "expected"),
    [("average", 2.5), ("first point", 1.0), ("last point", 4.0)],
)
def test_states_to_conditions_uses_requested_aggregation(method, expected):
    """StatesToConditions preserves non-temperature coordinates."""
    measurement = _make_measurement("strain", [1.0, 2.0, 3.0, 4.0])

    result = StatesToConditions(state="strain", method=method)._apply(measurement)

    assert result.conditions["strain"] == pytest.approx(expected)


def test_states_to_conditions_missing_state_raises():
    """StatesToConditions rejects states absent from every dataset."""
    measurement = _make_measurement("strain", [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="not found"):
        StatesToConditions(state="stress", method="average")._apply(measurement)


def test_states_to_conditions_invalid_method_raises():
    """StatesToConditions rejects unsupported aggregation methods."""
    measurement = _make_measurement("strain", [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Invalid method"):
        StatesToConditions(state="strain", method="median")._apply(measurement)


def test_mutation_number_state_to_condition_preserves_time_coordinate():
    """MutationNumber samples states without a temperature array."""
    measurement_set = MeasurementSet(
        [_make_measurement("custom state", [10.0, 20.0, 30.0, 40.0])],
        merge_redundant=False,
    )

    MutationNumber(state="custom state", state_sampling="first point")._state_to_condition(
        measurement_set
    )

    assert measurement_set.measurements[0].conditions["custom state"] == pytest.approx(10.0)
