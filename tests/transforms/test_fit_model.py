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

import os
import sys
import shutil

import numpy as np
import pytest

from hermes_rheo.file_readers.trios_rheo_txt import TriosRheoReader
from hermes_rheo.transforms.fit_model import FitModel, MultiModeMaxwellModel


@pytest.fixture()
def datadir(tmpdir, request):
    module_dir = os.path.dirname(request.module.__file__)
    test_dir = os.path.abspath(os.path.join(module_dir, "..", "fileio", "test_data"))
    if os.path.isdir(test_dir):
        shutil.copytree(test_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir


# ---------------------------------------------------------------------------
# 1) MultiModeMaxwell NNLS synthetic round-trip
# ---------------------------------------------------------------------------

def test_multimode_maxwell_nnls_recovers_prony_series():
    omega = np.logspace(-2, 3, 80)
    taus = np.array([1e-2, 1.0, 1e2])
    g_modes = np.array([1e5, 5e4, 2e4])
    g_e = 1e3

    gp, gpp = MultiModeMaxwellModel._moduli(omega, g_modes, taus, Ge=g_e)

    model = MultiModeMaxwellModel(mode="nnls", equilibrium_modulus=True).fit(
        omega, gp, gpp
    )
    gp_fit, gpp_fit = model.predict(omega)

    np.testing.assert_allclose(gp_fit, gp, rtol=1e-3)
    np.testing.assert_allclose(gpp_fit, gpp, rtol=1e-3)
    assert model.equilibrium_modulus_value == pytest.approx(g_e, rel=0.05)


def test_multimode_maxwell_without_plateau_has_no_equilibrium():
    omega = np.logspace(-1, 2, 50)
    taus = np.array([0.1, 10.0])
    g_modes = np.array([1e4, 1e4])
    gp, gpp = MultiModeMaxwellModel._moduli(omega, g_modes, taus)

    model = MultiModeMaxwellModel(mode="nnls", equilibrium_modulus=False).fit(
        omega, gp, gpp
    )

    assert model.equilibrium_modulus_value == 0.0
    np.testing.assert_allclose(model.predict(omega)[1], gpp, rtol=1e-2)


# ---------------------------------------------------------------------------
# 2) FitModel single-dataset fitting
# ---------------------------------------------------------------------------

def test_fitmodel_fits_single_dataset_with_maxwell(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    experiment = TriosRheoReader().data_from_filepath(
        str(filepath), create_composite_datasets=True
    )

    fit = FitModel(
        model="custom",
        model_name="MultiModeMaxwell",
        measurement_index=0,
    )(experiment)

    assert fit.fitted_model is not None
    assert fit.omega_common is not None
    # a single temperature's sweep -> far fewer points than 20 measurements
    assert len(fit.omega_common) < 60
    gp_pred, gpp_pred = fit.fitted_model.predict(fit.omega_common)
    assert np.all(np.isfinite(gp_pred))
    assert np.all(np.isfinite(gpp_pred))


def test_fitmodel_measurement_index_selects_different_dataset(datadir):
    filepath = datadir.join("test_frequency_sweep_tts_ARES.txt")
    experiment = TriosRheoReader().data_from_filepath(
        str(filepath), create_composite_datasets=True
    )

    fit0 = FitModel(model="custom", model_name="MultiModeMaxwell",
                    measurement_index=0)(experiment)
    fit1 = FitModel(model="custom", model_name="MultiModeMaxwell",
                    measurement_index=5)(experiment)

    # Different temperatures -> different moduli magnitudes
    assert not np.allclose(fit0.G_prime.mean(), fit1.G_prime.mean())


@pytest.mark.parametrize("domain", ["oscillation", "relaxation"])
def test_missing_pyrheo_raises_install_hint(monkeypatch, domain):
    from hermes_rheo.transforms.fit_model import _load_pyrheo_model

    for name in ("pyRheo", "pyRheo.oscillation_model", "pyRheo.relaxation_model"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"pip install git\+https://github.com/mirandi1/pyRheo"):
        _load_pyrheo_model(domain)
