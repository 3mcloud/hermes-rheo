"""Tests for :mod:`hermes_rheo.visualization.units`."""

from __future__ import annotations

import numpy as np
import pytest
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement_set import MeasurementSet
from piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset import (
    OneDimensionalCompositeDataset,
)

from hermes_rheo.visualization import get_unit, unit_suffix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dataset():
    """A minimal composite dataset with three arrays of assorted units."""
    return OneDimensionalCompositeDataset(
        data_arrays=[
            np.array([1.0, 2.0, 3.0]),  # angular frequency
            np.array([10.0, 20.0, 30.0]),  # storage modulus
            np.array([0.1, 0.2, 0.3]),  # tan(delta) — dimensionless
        ],
        data_array_names=[
            "angular frequency",
            "storage modulus",
            "tan(delta)",
        ],
        data_array_units=["rad/s", "MPa", ""],
        default_independent_name="angular frequency",
        default_dependent_name="storage modulus",
    )


@pytest.fixture()
def measurement(dataset):
    return Measurement(datasets=[dataset])


@pytest.fixture()
def measurement_set(measurement):
    return MeasurementSet(measurements=[measurement])


# ---------------------------------------------------------------------------
# get_unit
# ---------------------------------------------------------------------------


def test_get_unit_from_dataset(dataset):
    assert get_unit(dataset, "storage modulus") == "MPa"
    assert get_unit(dataset, "angular frequency") == "rad/s"


def test_get_unit_from_measurement(measurement):
    assert get_unit(measurement, "storage modulus") == "MPa"


def test_get_unit_from_measurement_set(measurement_set):
    assert get_unit(measurement_set, "storage modulus") == "MPa"


def test_get_unit_empty_string_when_unit_is_blank(dataset):
    """Dimensionless arrays store '' — the helper preserves that."""
    assert get_unit(dataset, "tan(delta)") == ""


def test_get_unit_unknown_name_returns_empty(dataset):
    assert get_unit(dataset, "not-a-real-array") == ""


def test_get_unit_handles_empty_container():
    """An empty MeasurementSet does not raise."""
    empty = MeasurementSet(measurements=[])
    assert get_unit(empty, "storage modulus") == ""


# ---------------------------------------------------------------------------
# unit_suffix
# ---------------------------------------------------------------------------


def test_unit_suffix_paren_default():
    assert unit_suffix("MPa") == " (MPa)"


def test_unit_suffix_bracket_style():
    assert unit_suffix("MPa", style="bracket") == " [MPa]"


def test_unit_suffix_empty_returns_empty():
    assert unit_suffix("") == ""
    assert unit_suffix("", style="bracket") == ""


def test_unit_suffix_custom_separator():
    assert unit_suffix("MPa", sep="") == "(MPa)"


def test_unit_suffix_invalid_style_raises():
    with pytest.raises(ValueError):
        unit_suffix("MPa", style="curly")
