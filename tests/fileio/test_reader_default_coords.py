"""Tests for per-method default coordinates assigned by the readers.

These tests lock in the behaviour that a file read by
:class:`~hermes_rheo.file_readers.trios_rheo_txt.TriosRheoReader` or
:class:`~hermes_rheo.file_readers.rheo_compass_csv.RheoCompassReader` produces
:class:`~piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset.OneDimensionalCompositeDataset`
objects whose default (independent, dependent) coordinates match the natural
scientific view of the measurement (e.g. ``strain=f(time)`` for Creep,
``storage modulus=f(angular frequency)`` for a frequency sweep).  This
replaces the historical ``Temperature=f(Temperature)`` fallback.

Regression tests are also included for:

* the ``default_coords="uniform"`` opt-in that reproduces the old
  consistency-preserving behaviour, and
* the ``_lookup_method_defaults`` / ``_canonicalise_method`` helper units.
"""

from __future__ import annotations

import pytest

from hermes_rheo.file_readers.rheo_compass_csv import (
    RheoCompassReader,
    _lookup_rheocompass_defaults,
)
from hermes_rheo.file_readers.trios_rheo_txt import (
    TriosRheoReader,
    _canonicalise_method,
    _lookup_method_defaults,
)

# ---------------------------------------------------------------------------
# TriosRheoReader — per-method natural defaults from real fixtures
# ---------------------------------------------------------------------------

_TRIOS_EXPECTED_DEFAULTS = [
    # (fixture_filename,               expected_indep,        expected_dep)
    ("test_temp_ramp.txt", "temperature", "storage modulus"),
    ("test_amplitude_apac.txt", "oscillation strain", "storage modulus"),
    ("test_frequency_sweep_tts_ARES.txt", "angular frequency", "storage modulus"),
    ("test_owchirp_DHR.txt", "step time", "strain"),
    ("test_temp_sweep_tts.txt", "temperature", "storage modulus"),
    ("test_temperature_sweep_RSAG2.txt", "temperature", "storage modulus"),
    ("test_owchirp_tts_ARES.txt", "step time", "strain"),
    # test_fitting.txt starts with a Temperature Sweep step
    ("test_fitting.txt", "temperature", "storage modulus"),
]


@pytest.mark.parametrize("fname,expected_x,expected_y", _TRIOS_EXPECTED_DEFAULTS)
def test_trios_natural_defaults(datadir, fname, expected_x, expected_y):
    """Each TRIOS fixture's first measurement carries its natural default coords."""
    filepath = datadir.join(fname)
    mset = TriosRheoReader().data_from_filepath(
        filepath=str(filepath), create_composite_datasets=True
    )
    ds = mset.measurements[0].datasets[0]
    assert ds.independent_variable_name == expected_x
    assert ds.dependent_variable_name == expected_y


def test_trios_no_measurement_has_identical_x_and_y(datadir):
    """The historical ``Temperature=f(Temperature)`` never occurs anymore."""
    for fname, _, _ in _TRIOS_EXPECTED_DEFAULTS:
        filepath = datadir.join(fname)
        mset = TriosRheoReader().data_from_filepath(
            filepath=str(filepath), create_composite_datasets=True
        )
        for i, m in enumerate(mset.measurements):
            for j, ds in enumerate(m.datasets):
                assert ds.independent_variable_name != ds.dependent_variable_name, (
                    f"{fname} m[{i}].datasets[{j}] has X==Y=={ds.independent_variable_name!r}"
                )


def test_trios_creep_one_line_description(datadir):
    """Locks the display string produced by ``print(experiment)`` for creep."""
    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(
        filepath=str(filepath), create_composite_datasets=True
    )
    # The file has 4 measurements: 1x Stress Control + 3x Creep.  The 2nd
    # measurement is the first Creep step.
    ds = mset.measurements[1].datasets[0]
    assert ds.independent_variable_name == "time"
    assert ds.dependent_variable_name == "strain"
    # Sanity-check the one_line_description formatting.
    desc = ds.one_line_description
    assert "strain" in desc.lower()
    assert "time" in desc.lower()
    # Assert the historical bad label is definitely gone.
    assert "temperature=f(temperature" not in desc.lower()


# ---------------------------------------------------------------------------
# TriosRheoReader — uniform mode
# ---------------------------------------------------------------------------


def test_trios_uniform_mode_reproduces_old_behaviour(datadir):
    """``default_coords="uniform"`` restores ``Temperature=f(Temperature)``."""
    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(
        filepath=str(filepath),
        create_composite_datasets=True,
        default_coords="uniform",
    )
    for m in mset.measurements:
        ds = m.datasets[0]
        assert ds.independent_variable_name == "Temperature"
        assert ds.dependent_variable_name == "Temperature"


def test_trios_uniform_permits_experimentset_wrap(datadir):
    """Uniform mode keeps mixed-method files consistent enough for ExperimentSet."""
    from piblin.data import ExperimentSet

    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(
        filepath=str(filepath),
        create_composite_datasets=True,
        default_coords="uniform",
    )
    # Would raise InconsistentMeasurementsError under natural defaults.
    exp = ExperimentSet(mset.measurements)
    assert exp.num_experiments == 4


def test_trios_legacy_temperature_alias_still_accepted(datadir):
    """The 1.x spelling ``"legacy-temperature"`` behaves like ``"uniform"``."""
    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(
        filepath=str(filepath),
        create_composite_datasets=True,
        default_coords="legacy-temperature",
    )
    ds = mset.measurements[0].datasets[0]
    assert ds.independent_variable_name == ds.dependent_variable_name == "Temperature"


# ---------------------------------------------------------------------------
# _canonicalise_method — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Step name\tTemperature Ramp - 1", "temperature ramp"),
        ("Frequency sweep Repeat - 3", "frequency sweep"),
        ("Creep - 2", "creep"),
        ("Arbitrary Wave - 1", "arbitrary wave"),
        ("  Step name\tAmplitude Sweep  ", "amplitude sweep"),
        ("Stress Control - 1", "stress control"),
    ],
)
def test_canonicalise_method(raw, expected):
    assert _canonicalise_method(raw) == expected


# ---------------------------------------------------------------------------
# _lookup_method_defaults — unit tests (no fixture required)
# ---------------------------------------------------------------------------

_TRIOS_LOOKUP_CASES = [
    # (method_line,              variable_names_include,                    expected)
    ("Step name\tCreep - 2", ["time", "strain", "temperature"], ("time", "strain")),
    (
        "Step name\tFrequency sweep",
        ["angular frequency", "storage modulus"],
        ("angular frequency", "storage modulus"),
    ),
    (
        "Temperature Ramp - 1",
        ["temperature", "storage modulus"],
        ("temperature", "storage modulus"),
    ),
    ("Arbitrary Wave - 5", ["step time", "strain", "stress"], ("step time", "strain")),
    ("Frequency Chirp - 1", ["step time", "strain", "stress"], ("step time", "strain")),
    ("Flow Sweep - 1", ["shear rate", "viscosity"], ("shear rate", "viscosity")),
    (
        "Multiwave - 1",
        ["angular frequency", "storage modulus"],
        ("angular frequency", "storage modulus"),
    ),
]


@pytest.mark.parametrize("method,names,expected", _TRIOS_LOOKUP_CASES)
def test_lookup_method_defaults_happy_path(method, names, expected):
    result = _lookup_method_defaults(method, names, fallback_independent="Temperature")
    assert result == expected


def test_lookup_method_defaults_missing_columns_falls_back_to_temperature():
    """Missing default columns → fallback to the injected Temperature column."""
    result = _lookup_method_defaults(
        "Frequency sweep - 1",
        variable_names=["temperature", "Temperature", "something else"],
        fallback_independent="Temperature",
    )
    # Neither 'angular frequency' nor 'storage modulus' present → fallback.
    assert result[0] == "Temperature"
    assert result[1] != result[0]


def test_lookup_method_defaults_unknown_method_falls_back():
    """A method that matches no rule uses the safe (fallback, first-dep) pair."""
    result = _lookup_method_defaults(
        "Some Weird Experimental Step - 1",
        variable_names=["a", "b", "c", "Temperature"],
        fallback_independent="Temperature",
    )
    assert result[0] == "Temperature"
    assert result[1] in {"a", "b", "c"}


# ---------------------------------------------------------------------------
# RheoCompassReader — natural defaults from real fixtures
# ---------------------------------------------------------------------------

_RHEOCOMPASS_EXPECTED_DEFAULTS = [
    # (fixture_filename,                     expected_indep,       expected_dep)
    ("test_antonpaar_702e_tts.csv", "angular frequency", "storage modulus"),
    ("test_antonpaar_htr_stress_relax.csv", "time", "shear stress"),
]


@pytest.mark.parametrize("fname,expected_x,expected_y", _RHEOCOMPASS_EXPECTED_DEFAULTS)
def test_rheocompass_natural_defaults(datadir, fname, expected_x, expected_y):
    filepath = datadir.join(fname)
    mset = RheoCompassReader().data_from_filepath(str(filepath))
    ds = mset.measurements[0].datasets[0]
    assert ds.independent_variable_name == expected_x
    assert ds.dependent_variable_name == expected_y


@pytest.mark.parametrize(
    "names,expected",
    [
        # Frequency sweep — angular freq + G' present
        (
            ["time", "angular frequency", "storage modulus"],
            ("angular frequency", "storage modulus"),
        ),
        # Flow sweep
        (["time", "shear rate", "viscosity"], ("shear rate", "viscosity")),
        # Stress relaxation / creep fallback
        (["time", "shear strain", "shear stress"], ("time", "shear stress")),
    ],
)
def test_rheocompass_lookup_rules(names, expected):
    assert _lookup_rheocompass_defaults(names) == expected
