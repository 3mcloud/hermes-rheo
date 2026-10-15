"""Tests for :class:`hermes_rheo.transforms.owchirp_analysis.OWChirpAnalysis`.

These tests lock in that:

1. ``OWChirpAnalysis`` produces the same frequency-domain moduli as the
   deprecated ``RheoAnalysis`` alias on the same input fixture (numerical
   equivalence).
2. Non-chirp measurements pass through unchanged (no-op behaviour).
3. ``RheoAnalysis`` still works and emits a ``DeprecationWarning`` exactly
   once per construction.
4. ``OWChirpAnalysis`` still uses the pipeline operator (``+``) form and
   integrates with the existing master-curve and temperature-ramp
   transforms.
"""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytest

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.automated_mastercurve import AutomatedMasterCurve
from hermes_rheo.transforms.owchirp_analysis import (
    OWChirpAnalysis,
    _canonicalise_method,
)
from hermes_rheo.transforms.rheo_analysis import RheoAnalysis


# Reuse the fixture-copy pattern from tests/fileio/conftest.py — this test
# module lives in tests/transforms/ so points at ../fileio/test_data.
@pytest.fixture()
def datadir(tmpdir, request):
    module_dir = os.path.dirname(request.module.__file__)
    test_dir = os.path.abspath(os.path.join(module_dir, "..", "fileio", "test_data"))
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir


# ---------------------------------------------------------------------------
# 1) Numerical equivalence with the deprecated RheoAnalysis alias
# ---------------------------------------------------------------------------


def test_owchirp_analysis_equivalent_to_rheo_analysis_on_owchirp_file(datadir):
    """OWChirpAnalysis and RheoAnalysis produce identical moduli on an OWChirp fixture."""
    filepath = datadir.join("test_owchirp_DHR.txt")

    mset_new = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    mset_old = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)

    processed_new = OWChirpAnalysis()(mset_new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        processed_old = RheoAnalysis()(mset_old)

    new_ds = processed_new.measurements[0].datasets[0]
    old_ds = processed_old.measurements[0].datasets[0]

    # Frequency-space datasets must be numerically identical
    for name in (
        "angular frequency",
        "storage modulus",
        "loss modulus",
        "complex modulus",
        "tan(delta)",
    ):
        idx_new = new_ds.data_array_names.index(name)
        idx_old = old_ds.data_array_names.index(name)
        np.testing.assert_allclose(
            new_ds.data_arrays[idx_new],
            old_ds.data_arrays[idx_old],
            rtol=1e-12,
            err_msg=f"{name} differs between OWChirpAnalysis and RheoAnalysis",
        )


# ---------------------------------------------------------------------------
# 2) Non-chirp measurements are pass-through
# ---------------------------------------------------------------------------


def test_owchirp_analysis_noop_on_temperature_ramp(datadir):
    """A temperature-ramp measurement is unchanged by OWChirpAnalysis."""
    filepath = datadir.join("test_temp_ramp.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    ds_before = mset.measurements[0].datasets[0]
    n_arrays_before = len(ds_before.data_arrays)
    x_name_before = ds_before.independent_variable_name
    y_name_before = ds_before.dependent_variable_name

    processed = OWChirpAnalysis()(mset)

    ds_after = processed.measurements[0].datasets[0]
    assert len(ds_after.data_arrays) == n_arrays_before
    assert ds_after.independent_variable_name == x_name_before
    assert ds_after.dependent_variable_name == y_name_before
    # And the reader-assigned default is preserved.
    assert x_name_before == "temperature"
    assert y_name_before == "storage modulus"


def test_owchirp_analysis_noop_on_creep(datadir):
    """A creep measurement is unchanged by OWChirpAnalysis."""
    filepath = datadir.join("test_creep_apac.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    ds_before = mset.measurements[1].datasets[0]  # first Creep step
    x_before = ds_before.independent_variable_name
    y_before = ds_before.dependent_variable_name

    processed = OWChirpAnalysis()(mset)

    ds_after = processed.measurements[1].datasets[0]
    assert ds_after.independent_variable_name == x_before == "time"
    assert ds_after.dependent_variable_name == y_before == "strain"


# ---------------------------------------------------------------------------
# 3) DeprecationWarning on RheoAnalysis construction
# ---------------------------------------------------------------------------


def test_rheo_analysis_emits_deprecation_warning():
    """Constructing RheoAnalysis emits a single DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RheoAnalysis()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "OWChirpAnalysis" in str(deprecations[0].message)


def test_rheo_analysis_still_works_end_to_end(datadir):
    """RheoAnalysis(...) forwards to OWChirpAnalysis and processes an OWChirp fixture."""
    filepath = datadir.join("test_owchirp_DHR.txt")
    mset = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        processed = RheoAnalysis()(mset)
    assert processed.details["instrument_serial_number"].startswith("5343")


# ---------------------------------------------------------------------------
# 4) Pipeline integration
# ---------------------------------------------------------------------------


def test_owchirp_analysis_composes_with_master_curve(datadir):
    """OWChirpAnalysis + AutomatedMasterCurve pipeline works on OWChirp TTS data."""
    filepath = datadir.join("test_owchirp_tts_ARES.txt")
    data = TriosRheoReader().data_from_filepath(str(filepath), create_composite_datasets=True)
    pipeline = OWChirpAnalysis() + AutomatedMasterCurve(
        state="temperature",
        y="storage modulus",
        vertical_shift=False,
        reverse_data=False,
        measurements=None,
    )
    master_curve = pipeline(data)
    assert len(master_curve.hparams[0]) == 10


# ---------------------------------------------------------------------------
# 5) _canonicalise_method / helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Step name\tArbitrary Wave - 3", "arbitrary wave"),
        ("Frequency Chirp Repeat - 1", "frequency chirp"),
        ("Creep - 2", "creep"),
        ("  Temperature Ramp - 1  ", "temperature ramp"),
    ],
)
def test_canonicalise_method(raw, expected):
    assert _canonicalise_method(raw) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.5 mrad", 0.0025),
        ("1e3 N.m", 1000.0),
        ("0.5 µN.m.s²", 0.5e-6),
        ("2.02 mrad/N.m", 2.02e-3),
    ],
)
def test_parse_value_with_prefix(text, expected):
    result = OWChirpAnalysis.parse_value_with_prefix(text)
    assert result == pytest.approx(expected, rel=1e-9)


def test_parse_value_with_prefix_none_returns_nan():
    result = OWChirpAnalysis.parse_value_with_prefix(None)
    assert np.isnan(result)
