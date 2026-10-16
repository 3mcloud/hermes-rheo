"""Tests for :mod:`hermes_rheo.transforms.fit_model`.

These tests lock in that:

1. :class:`MultiModeMaxwellModel` in ``nnls`` mode recovers a synthetic
   Prony series (with a rubbery-plateau modulus) essentially exactly.
2. :class:`FitModel` fits a **single dataset** selected from an
   experiment (isothermal / one-temperature frequency sweep) as well as
   a master curve with the built-in Maxwell model.
3. A missing pyRheo installation raises an actionable ``ImportError``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from piblin.data.data_collections.measurement import Measurement
from piblin.data.datasets.abc.split_datasets.one_dimensional_composite_dataset import (
    OneDimensionalCompositeDataset,
)
from piblin.transform.pipeline import Pipeline

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.fit_model import FitModel, MultiModeMaxwellModel


def test_multimode_maxwell_nnls_recovers_prony_series():
    omega = np.logspace(-2, 3, 80)
    taus = np.array([1e-2, 1.0, 1e2])
    g_modes = np.array([1e5, 5e4, 2e4])
    g_e = 1e3

    gp, gpp = MultiModeMaxwellModel._moduli(omega, g_modes, taus, Ge=g_e)

    model = MultiModeMaxwellModel(mode="nnls", equilibrium_modulus=True).fit(omega, gp, gpp)
    gp_fit, gpp_fit = model.predict(omega)

    np.testing.assert_allclose(gp_fit, gp, rtol=1e-3)
    np.testing.assert_allclose(gpp_fit, gpp, rtol=1e-3)
    assert model.equilibrium_modulus_value == pytest.approx(g_e, rel=0.05)


def test_multimode_maxwell_without_plateau_has_no_equilibrium():
    omega = np.logspace(-1, 2, 50)
    taus = np.array([0.1, 10.0])
    g_modes = np.array([1e4, 1e4])
    gp, gpp = MultiModeMaxwellModel._moduli(omega, g_modes, taus)

    model = MultiModeMaxwellModel(mode="nnls", equilibrium_modulus=False).fit(omega, gp, gpp)

    assert model.equilibrium_modulus_value == 0.0
    np.testing.assert_allclose(model.predict(omega)[1], gpp, rtol=1e-2)


def test_fitmodel_fits_single_dataset_with_maxwell(datadir):
    filepath = datadir / "test_aresg2_trios_frequency_sweep_tts.txt"
    experiment = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)

    fit = FitModel(
        model="custom",
        model_name="MultiModeMaxwell",
        measurement_index=0,
    )
    result = fit(experiment)
    assert isinstance(result, Measurement)

    assert fit.fitted_model is not None
    assert fit.omega_common is not None
    # a single temperature's sweep -> far fewer points than 20 measurements
    assert len(fit.omega_common) < 60
    gp_pred, gpp_pred = fit.fitted_model.predict(fit.omega_common)
    assert np.all(np.isfinite(gp_pred))
    assert np.all(np.isfinite(gpp_pred))


def test_fitmodel_measurement_index_selects_different_dataset(datadir):
    filepath = datadir / "test_aresg2_trios_frequency_sweep_tts.txt"
    experiment = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)

    fit0 = FitModel(model="custom", model_name="MultiModeMaxwell", measurement_index=0)
    fit1 = FitModel(model="custom", model_name="MultiModeMaxwell", measurement_index=5)
    fit0(experiment)
    fit1(experiment)

    # Different temperatures -> different moduli magnitudes
    assert not np.allclose(fit0.G_prime.mean(), fit1.G_prime.mean())


def test_fitmodel_returns_piblin_measurement_with_metadata(datadir):
    filepath = datadir / "test_aresg2_trios_frequency_sweep_tts.txt"
    experiment = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)

    result = FitModel(model="custom", model_name="MultiModeMaxwell")(experiment)

    assert isinstance(result, Measurement)
    dataset = result.datasets[0]
    assert isinstance(dataset, OneDimensionalCompositeDataset)
    assert list(dataset.data_array_names) == [
        "angular frequency",
        "storage modulus",
        "loss modulus",
        "fitted storage modulus",
        "fitted loss modulus",
    ]
    assert list(dataset.data_array_units) == ["rad/s", "Pa", "Pa", "Pa", "Pa"]
    assert all(np.all(np.isfinite(a)) for a in dataset.data_arrays)

    details = result.details
    assert details["model"] == "GeneralizedMaxwell"
    assert len(details["parameters"]["G_modes"]) == len(details["parameters"]["tau_modes"])
    assert details["goodness_of_fit"]["rss"] >= 0
    assert 0.5 < details["goodness_of_fit"]["r_squared"] <= 1.0
    assert details["domain"] == "oscillation"
    assert any("d0sm00354a" in ref for ref in details["references"])
    assert details["software"].startswith("hermes-rheo ")
    assert result.conditions["fitted"] is True


def test_fitmodel_works_in_pipeline_and_relaxation_domain(datadir):
    filepath = datadir / "test_aresg2_trios_frequency_sweep_tts.txt"
    experiment = TriosRheoReader().data_from_filepath(filepath, create_composite_datasets=True)
    fit = FitModel(model="custom", model_name="MultiModeMaxwell")
    assert isinstance(fit + fit, Pipeline)
    assert isinstance(Pipeline([fit])(experiment), Measurement)

    with pytest.raises(ValueError, match="only valid for domain='oscillation'"):
        FitModel(model="custom", model_name="MultiModeMaxwell", domain="relaxation")(experiment)


@pytest.mark.parametrize("domain", ["oscillation", "relaxation"])
def test_missing_pyrheo_raises_install_hint(monkeypatch, domain):
    from hermes_rheo.transforms.fit_model import _load_pyrheo_model

    for name in ("pyRheo", "pyRheo.oscillation_model", "pyRheo.relaxation_model"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pip install git\+https://github.com/mirandi1/pyRheo"):
        _load_pyrheo_model(domain)
