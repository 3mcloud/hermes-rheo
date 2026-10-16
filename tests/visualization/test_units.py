"""Tests for visualization unit helpers."""

from __future__ import annotations

import numpy as np
import pytest
from piblin.data.data_collections.measurement import Measurement
from piblin.data.data_collections.measurement_set import MeasurementSet
from piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset import (
    OneDimensionalCompositeDataset,
)

from hermes_rheo.visualization import axis_label, get_unit, unit_suffix


@pytest.fixture()
def dataset():
    return OneDimensionalCompositeDataset(
        data_arrays=[
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([0.1, 0.2, 0.3]),
        ],
        data_array_names=["angular frequency", "storage modulus", "tan(delta)"],
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


@pytest.mark.parametrize("source", ["dataset", "measurement", "measurement_set"])
def test_get_unit_from_piblin_containers(source, dataset, measurement, measurement_set):
    container = {
        "dataset": dataset,
        "measurement": measurement,
        "measurement_set": measurement_set,
    }[source]
    assert get_unit(container, "storage modulus") == "MPa"


def test_get_unit_handles_dimensionless_missing_and_empty_data(dataset):
    assert get_unit(dataset, "tan(delta)") == ""
    assert get_unit(dataset, "not-a-real-array") == ""
    assert get_unit(MeasurementSet(measurements=[]), "storage modulus") == ""


@pytest.mark.parametrize(
    ("unit", "style", "separator", "expected"),
    [
        ("MPa", "paren", " ", " (MPa)"),
        ("MPa", "bracket", " ", " [MPa]"),
        ("", "paren", " ", ""),
        ("MPa", "paren", "", "(MPa)"),
    ],
)
def test_unit_suffix_formats_units(unit, style, separator, expected):
    assert unit_suffix(unit, style=style, sep=separator) == expected


def test_unit_suffix_rejects_unknown_style():
    with pytest.raises(ValueError):
        unit_suffix("MPa", style="curly")


def test_axis_label_uses_and_overrides_dataset_units(dataset):
    assert axis_label(dataset, "storage modulus") == "storage modulus (MPa)"
    assert axis_label(dataset, "tan(delta)") == "tan(delta)"
    assert axis_label(dataset, "storage modulus", text="G'", unit="Pa") == "G' (Pa)"
    assert axis_label(dataset, "storage modulus", style="bracket") == "storage modulus [MPa]"
