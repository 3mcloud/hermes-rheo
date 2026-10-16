"""Tests for shift-factor models."""

from __future__ import annotations

import numpy as np
import pytest

from hermes_rheo.transforms.shift_factor_models import (
    ArrheniusModel,
    ShiftFactorModel,
    WLFModel,
)

_R = 8.314462618
_LN10 = np.log(10.0)


class _MockMasterCurve:
    """Minimal stand-in exposing the attributes ShiftFactorModel reads."""

    def __init__(self, states, shift_factors):
        self.states = list(states)
        self.hparams = [list(shift_factors)]


def test_wlf_recovers_constants():
    temperatures = np.linspace(-40.0, 150.0, 25)
    c1, c2, t_ref = 15.0, 120.0, -40.0
    log_aT = -c1 * (temperatures - t_ref) / (c2 + (temperatures - t_ref))

    model = WLFModel().fit(temperatures, log_aT, reference_temperature=t_ref)

    assert model.C1 == pytest.approx(c1, rel=1e-3)
    assert model.C2 == pytest.approx(c2, rel=1e-3)
    assert model.rss < 1e-6
    np.testing.assert_allclose(model.predict(temperatures), log_aT, atol=1e-3)


def test_arrhenius_recovers_activation_energy():
    temperatures = np.linspace(-40.0, 150.0, 25)
    t_ref = 25.0
    ea = 1.2e5  # J/mol
    t_kelvin = temperatures + 273.15
    t_ref_kelvin = t_ref + 273.15
    log_aT = (ea / (_R * _LN10)) * (1.0 / t_kelvin - 1.0 / t_ref_kelvin)

    model = ArrheniusModel().fit(temperatures, log_aT, reference_temperature=t_ref)

    assert model.activation_energy == pytest.approx(ea, rel=1e-4)
    assert model.rss < 1e-10
    np.testing.assert_allclose(model.predict(temperatures), log_aT, atol=1e-6)


def test_arrhenius_unit_conversion_matches_kelvin_input():
    temperatures_c = np.linspace(0.0, 120.0, 15)
    temperatures_k = temperatures_c + 273.15
    ea, t_ref_c = 8.0e4, 40.0
    log_aT = (ea / (_R * _LN10)) * (1.0 / (temperatures_c + 273.15) - 1.0 / (t_ref_c + 273.15))

    model_c = ArrheniusModel(temperature_unit="C").fit(
        temperatures_c, log_aT, reference_temperature=t_ref_c
    )
    model_k = ArrheniusModel(temperature_unit="K").fit(
        temperatures_k, log_aT, reference_temperature=t_ref_c + 273.15
    )

    assert model_c.activation_energy == pytest.approx(model_k.activation_energy, rel=1e-6)


def test_shift_factor_model_reads_mastercurve_object():
    temperatures = np.linspace(-40.0, 150.0, 20)
    c1, c2, t_ref = 15.0, 60.0, -40.0
    log_aT = -c1 * (temperatures - t_ref) / (c2 + (temperatures - t_ref))
    shift_factors = 10.0**log_aT

    mc = _MockMasterCurve(temperatures, shift_factors)
    fit = ShiftFactorModel(model="WLF")(mc)

    assert fit.fitted_model.C1 == pytest.approx(c1, rel=1e-3)
    assert fit.fitted_model.C2 == pytest.approx(c2, rel=1e-3)
    assert fit.parameters["reference_temperature"] == pytest.approx(t_ref, abs=1e-6)


def test_shift_factor_model_reads_tuple_input():
    temperatures = np.linspace(0.0, 100.0, 12)
    ea, t_ref = 9.0e4, 50.0
    log_aT = (ea / (_R * _LN10)) * (1.0 / (temperatures + 273.15) - 1.0 / (t_ref + 273.15))
    shift_factors = 10.0**log_aT

    fit = ShiftFactorModel(model="Arrhenius", reference_temperature=t_ref)(
        (temperatures, shift_factors)
    )

    assert fit.fitted_model.activation_energy == pytest.approx(ea, rel=1e-3)
    predicted = fit.predict(temperatures)
    np.testing.assert_allclose(predicted, log_aT, atol=1e-6)


def test_shift_factor_model_rejects_unknown_model():
    with pytest.raises(ValueError):
        ShiftFactorModel(model="not-a-model")
